"""Depth Expert — auxiliary prediction head matching QDepth-VLA Table 1.

Architecture (QDepth-VLA arXiv:2510.14836 §3.3 + Table 1):

    SigLIP image features (B, N_img, D_img)
        |
        + lightweight MLP project           ->  (B, N_img, hidden=1024)
        + Transformer (18 layers, 8 heads,
                       hidden=1024, intermediate=4096)
        + linear out -> embedding_dim (160)  ->  (B, N_img, 160)
        + reshape (B, 160, H_lat, W_lat)
        + shallow CNN decoder                ->  (B, 160, H_lat, W_lat)
        + per-position L2 similarity to VQ codebook (K=256 codes, dim 160)
        + cross-entropy with ground-truth indices from a frozen VQ-VAE

QDepth-VLA emphasises this expert mirrors the action expert architecturally to
keep training-time behaviour aligned. The output is **never** consumed by the
action expert directly; its purpose is to drive geometric grounding into the
SigLIP encoder via co-training.

This module is geometry-agnostic: instantiate with `embedding_dim` matching
the depth/normal/plane VQ-VAE (160 / 160 / 160 by default; codebook size K
varies but the expert architecture does not).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DepthExpertConfig:
    """QDepth-VLA Table 1 defaults."""

    siglip_dim: int = 1152      # SigLIP-So400m feature dim per token (PaliGemma-3B default)
    n_img_tokens: int = 256     # 16x16 patches at 224x224 / 14
    latent_h: int = 16          # matches VQ-VAE 16x16 grid
    latent_w: int = 16
    embedding_dim: int = 160    # = VQVAEConfig.embedding_dim
    num_embeddings: int = 256   # = VQVAEConfig.num_embeddings
    hidden: int = 1024          # transformer hidden width
    intermediate: int = 4096    # FFN inner width
    n_layers: int = 18
    n_heads: int = 8
    dropout: float = 0.0
    similarity_temperature: float = 1.0  # tau in QDepth-VLA eq. (3); 1.0 is the default unless ablated


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block (no cross-attention; sequence is self-contained)."""

    def __init__(self, hidden: int, n_heads: int, intermediate: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class DepthExpert(nn.Module):
    """Auxiliary expert producing per-spatial codebook logits."""

    def __init__(self, cfg: DepthExpertConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DepthExpertConfig()
        c = self.cfg

        # 1. lightweight MLP project SigLIP -> expert hidden
        self.input_proj = nn.Sequential(
            nn.LayerNorm(c.siglip_dim),
            nn.Linear(c.siglip_dim, c.hidden),
            nn.GELU(),
            nn.Linear(c.hidden, c.hidden),
        )

        # learnable position bias indexed by latent_h * latent_w == n_img_tokens
        self.pos_emb = nn.Parameter(torch.zeros(1, c.n_img_tokens, c.hidden))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # 2. transformer body (mirrors action expert size)
        self.blocks = nn.ModuleList([
            _TransformerBlock(c.hidden, c.n_heads, c.intermediate, c.dropout)
            for _ in range(c.n_layers)
        ])
        self.final_norm = nn.LayerNorm(c.hidden)

        # 3. project hidden -> embedding_dim
        self.embed_proj = nn.Linear(c.hidden, c.embedding_dim)

        # 4. shallow CNN decoder operating on (B, embedding_dim, H_lat, W_lat)
        #    Two 3x3 conv blocks for spatial smoothing — kept shallow per QDepth-VLA §3.3.
        self.cnn_decoder = nn.Sequential(
            nn.Conv2d(c.embedding_dim, c.embedding_dim, 3, padding=1),
            nn.GroupNorm(8, c.embedding_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(c.embedding_dim, c.embedding_dim, 3, padding=1),
        )

        # 5. similarity-to-codebook is computed against an externally-supplied
        #    codebook tensor at forward time; the depth expert is codebook-agnostic.
        #    This decouples it from the VQVAE module and lets us share the same
        #    expert architecture across depth/normal/plane heads later.

    def forward(
        self,
        siglip_features: torch.Tensor,
        codebook: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args
            siglip_features : (B, N_img, D_img) SigLIP visual tokens, pre-language fusion.
            codebook        : (K, embedding_dim) frozen VQ-VAE codebook (use `vqvae.quantizer.codebook.weight`).

        Returns
            logits          : (B, N_lat, K) per-spatial cross-entropy logits over codebook.
            predicted_embed : (B, embedding_dim, H_lat, W_lat) raw projected vectors before similarity.
        """
        c = self.cfg
        b, n_img, d = siglip_features.shape
        assert n_img == c.n_img_tokens, f"expected {c.n_img_tokens} tokens, got {n_img}"
        assert d == c.siglip_dim, f"expected siglip dim {c.siglip_dim}, got {d}"
        assert codebook.shape == (c.num_embeddings, c.embedding_dim), (
            f"codebook shape {tuple(codebook.shape)} must equal "
            f"({c.num_embeddings}, {c.embedding_dim})"
        )

        x = self.input_proj(siglip_features) + self.pos_emb
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)
        x = self.embed_proj(x)  # (B, N_img, embedding_dim)

        # reshape to spatial 2D for the CNN decoder
        x = x.transpose(1, 2).reshape(b, c.embedding_dim, c.latent_h, c.latent_w)
        x = self.cnn_decoder(x)
        # back to per-spatial vectors for similarity
        flat = x.permute(0, 2, 3, 1).reshape(b, c.latent_h * c.latent_w, c.embedding_dim)

        # ℓ_{i,k} = -1/τ * ||x_i - c_k||² (QDepth-VLA eq. 3)
        x_sq = (flat ** 2).sum(dim=-1, keepdim=True)              # (B, N_lat, 1)
        c_sq = (codebook ** 2).sum(dim=-1)                        # (K,)
        xc = flat @ codebook.t()                                  # (B, N_lat, K)
        dists = x_sq + c_sq.view(1, 1, -1) - 2.0 * xc             # (B, N_lat, K)
        logits = -dists / max(c.similarity_temperature, 1e-6)

        return logits, x


__all__ = ["DepthExpert", "DepthExpertConfig"]
