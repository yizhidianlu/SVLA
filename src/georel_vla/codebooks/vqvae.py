"""Vector-Quantised Variational Autoencoder for per-head depth tokenisation.

Implements the standard VQ-VAE (van den Oord et al. 2017, arXiv:1711.00937)
with the configuration QDepth-VLA (arXiv:2510.14836) §3.2 reports as
"already accurate enough":

    K = 256 codebook entries, dim = 160, latent grid 16 x 16,
    AdamW lr = 1e-5, commitment weight beta = 0.25.

The same module also serves as the architectural template for the K=128 normal
codebook (Phase 2) and K=64 plane codebook (Phase 2). Vary `num_embeddings`
and the decoder output channels accordingly.

Forward pass returns:
    z_q     -- quantised latent (straight-through to z_e for gradient flow)
    indices -- (B, H, W) long tensor of nearest-neighbour codebook indices
    losses  -- dict with `recon`, `codebook`, `commitment`, `total`
    reconstruction -- decoder output, same shape as `x`
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQVAEConfig:
    """QDepth-VLA-default sizing.

    For depth (1-channel) on 256x256 input, downsample factor 16 -> 16x16
    latent grid -> 256 tokens, matching QDepth-VLA's "256 latent positions per
    frame" used by the depth expert at training time.
    """

    in_channels: int = 1            # 1 for depth, 3 for normal (xyz), 1 for plane id
    out_channels: int = 1           # mirrors in_channels
    embedding_dim: int = 160        # QDepth-VLA d=160
    num_embeddings: int = 256       # QDepth-VLA K=256
    commitment_weight: float = 0.25 # van den Oord 2017 default; QDepth-VLA inherits
    hidden: int = 128               # encoder/decoder channel width
    downsample: int = 16            # 256 input -> 16 latent
    use_ema: bool = False           # straight VQ; EMA optional, kept off for parity


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _build_encoder(in_channels: int, hidden: int, embedding_dim: int, downsample: int) -> nn.Module:
    """Stride-2 conv stack repeated log2(downsample) times + 2 resblocks + 1x1 to embed_dim."""
    assert downsample in (4, 8, 16, 32), f"downsample must be a power of 2 in [4, 32], got {downsample}"
    n_down = int.bit_length(downsample) - 1  # 16 -> 4 down stages
    layers: list[nn.Module] = [nn.Conv2d(in_channels, hidden, 3, padding=1)]
    for _ in range(n_down):
        layers.extend([
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1),
        ])
    layers.extend([_ResBlock(hidden), _ResBlock(hidden)])
    layers.append(nn.Conv2d(hidden, embedding_dim, 1))
    return nn.Sequential(*layers)


def _build_decoder(out_channels: int, hidden: int, embedding_dim: int, downsample: int) -> nn.Module:
    """Mirror of the encoder via stride-2 transposed conv."""
    n_up = int.bit_length(downsample) - 1
    layers: list[nn.Module] = [
        nn.Conv2d(embedding_dim, hidden, 1),
        _ResBlock(hidden),
        _ResBlock(hidden),
    ]
    for _ in range(n_up):
        layers.extend([
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1),
        ])
    layers.append(nn.Conv2d(hidden, out_channels, 3, padding=1))
    return nn.Sequential(*layers)


class VectorQuantizer(nn.Module):
    """Standard VQ layer with straight-through gradient (van den Oord 2017 eq. 1-3)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_weight: float = 0.25) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        # Codebook initialisation: small uniform — matches the reference impl.
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args
            z_e: (B, D, H, W) encoder output.

        Returns
            z_q:     (B, D, H, W) quantised latent with straight-through gradient.
            indices: (B, H, W) long codebook indices.
            losses:  dict with `codebook` (codebook update term) and
                     `commitment` (encoder commitment term). Reconstruction
                     term is computed externally against the decoder output.
        """
        b, d, h, w = z_e.shape
        assert d == self.embedding_dim, f"channel mismatch: got {d}, want {self.embedding_dim}"

        # Flatten spatial dims to (B*H*W, D) for nearest-neighbour search.
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, d)
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c
        x_sq = (flat ** 2).sum(dim=1, keepdim=True)
        c_sq = (self.codebook.weight ** 2).sum(dim=1)
        xc = flat @ self.codebook.weight.t()
        dists = x_sq + c_sq - 2.0 * xc
        indices = torch.argmin(dists, dim=1)  # (B*H*W,)
        z_q_flat = self.codebook(indices)     # (B*H*W, D)
        z_q = z_q_flat.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        # Codebook + commitment losses (van den Oord 2017 eq. 3).
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss = F.mse_loss(z_e, z_q.detach())

        # Straight-through estimator: gradient of z_q wrt z_e == identity.
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, indices.view(b, h, w), {
            "codebook": codebook_loss,
            "commitment": self.commitment_weight * commit_loss,
        }

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode a (..., ) index tensor back to (..., D) embedding tensor."""
        return self.codebook(indices)


class VQVAE(nn.Module):
    """Encoder + VectorQuantizer + Decoder. Matches QDepth-VLA §3.2 sizing by default."""

    def __init__(self, cfg: VQVAEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VQVAEConfig()
        self.encoder = _build_encoder(
            self.cfg.in_channels, self.cfg.hidden, self.cfg.embedding_dim, self.cfg.downsample,
        )
        self.quantizer = VectorQuantizer(
            self.cfg.num_embeddings, self.cfg.embedding_dim, self.cfg.commitment_weight,
        )
        self.decoder = _build_decoder(
            self.cfg.out_channels, self.cfg.hidden, self.cfg.embedding_dim, self.cfg.downsample,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, H, W) discrete code indices for `x`. Used as supervision target."""
        z_e = self.encode(x)
        with torch.no_grad():
            _, indices, _ = self.quantizer(z_e)
        return indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode (B, H, W) indices back to reconstructed `x` of shape (B, C, H*ds, W*ds)."""
        z_q = self.quantizer.lookup(indices).permute(0, 3, 1, 2).contiguous()
        return self.decode(z_q)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Full training-time pass.

        Returns
            recon:   reconstructed input
            indices: (B, H, W) discrete codes (for downstream depth-expert target)
            losses:  dict with `recon`, `codebook`, `commitment`, `total`
        """
        z_e = self.encode(x)
        z_q, indices, vq_losses = self.quantizer(z_e)
        recon = self.decode(z_q)
        recon_loss = F.mse_loss(recon, x)
        losses = {
            "recon": recon_loss,
            "codebook": vq_losses["codebook"],
            "commitment": vq_losses["commitment"],
        }
        losses["total"] = losses["recon"] + losses["codebook"] + losses["commitment"]
        return recon, indices, losses


__all__ = ["VQVAE", "VQVAEConfig", "VectorQuantizer"]
