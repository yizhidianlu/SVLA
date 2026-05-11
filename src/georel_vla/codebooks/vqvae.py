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

    EMA path (use_ema=True, default since Phase-1.7b v2): updates the codebook
    via van den Oord 2017 §A.1 EMA rule (DeepMind sonnet implementation),
    which is more stable than the gradient-based codebook loss and crucial for
    avoiding the "few codes dominate" collapse we hit with the L2 path on
    LIBERO depth (only 6/256 codes alive after 1500 steps).

    K-means-style init from the first batch + periodic dead-code reset address
    the same problem (Razavi et al. 2019). Both default ON.
    """

    in_channels: int = 1            # 1 for depth, 3 for normal (xyz), 1 for plane id
    out_channels: int = 1           # mirrors in_channels
    embedding_dim: int = 160        # QDepth-VLA d=160
    num_embeddings: int = 256       # QDepth-VLA K=256
    commitment_weight: float = 0.25 # van den Oord 2017 default; QDepth-VLA inherits
    hidden: int = 128               # encoder/decoder channel width
    downsample: int = 16            # 256 input -> 16 latent
    use_ema: bool = True            # default ON since Phase-1.7b v2 (codebook collapse fix)
    ema_decay: float = 0.99         # codebook EMA momentum
    dead_code_threshold: float = 1.0   # cluster_size_ema below this counts as dead
    reset_dead_codes_every: int = 100  # 0 disables; otherwise reset every N EMA updates
    kmeans_init: bool = True        # init codebook from first batch's encoder samples


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
    """VQ layer with optional EMA codebook updates + dead-code reset.

    Two modes:
      * `use_ema=False` — vanilla van den Oord 2017 with L2 codebook loss
        (gradient updates the codebook).
      * `use_ema=True`  — EMA updates the codebook (no codebook loss); the
        commitment loss still pushes the encoder; periodic dead-code reset
        replaces unused codes with samples from the current batch's encoder
        outputs. Optional k-means-style init reseeds the codebook on the
        first training batch.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_weight: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1.0,
        reset_dead_codes_every: int = 100,
        kmeans_init: bool = True,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        self.reset_dead_codes_every = reset_dead_codes_every
        self.kmeans_init = kmeans_init

        # Codebook initialisation: small uniform — matches the reference impl.
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        if use_ema:
            # In EMA mode the codebook is updated by hand (not by the autograd loss),
            # so freeze its grad to keep optimisers from touching it.
            self.codebook.weight.requires_grad = False
            self.register_buffer("cluster_size_ema", torch.zeros(num_embeddings))
            self.register_buffer("dw_ema", self.codebook.weight.data.clone())
            self.register_buffer("ema_step", torch.zeros(1, dtype=torch.long))
            self._kmeans_done: bool = not kmeans_init   # True = skip k-means init

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args
            z_e: (B, D, H, W) encoder output.

        Returns
            z_q:     (B, D, H, W) quantised latent with straight-through gradient.
            indices: (B, H, W) long codebook indices.
            losses:  dict with `codebook` (codebook update term, 0 in EMA mode) and
                     `commitment` (encoder commitment term).
        """
        b, d, h, w = z_e.shape
        assert d == self.embedding_dim, f"channel mismatch: got {d}, want {self.embedding_dim}"

        flat = z_e.permute(0, 2, 3, 1).reshape(-1, d)
        x_sq = (flat ** 2).sum(dim=1, keepdim=True)
        c_sq = (self.codebook.weight ** 2).sum(dim=1)
        xc = flat @ self.codebook.weight.t()
        dists = x_sq + c_sq - 2.0 * xc
        indices = torch.argmin(dists, dim=1)
        z_q_flat = self.codebook(indices)
        z_q = z_q_flat.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        if self.use_ema:
            codebook_loss = torch.zeros((), device=z_e.device, dtype=z_e.dtype)
            commit_loss = F.mse_loss(z_e, z_q.detach())
            if self.training:
                self._ema_update(flat.detach(), indices.detach())
        else:
            codebook_loss = F.mse_loss(z_q, z_e.detach())
            commit_loss = F.mse_loss(z_e, z_q.detach())

        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, indices.view(b, h, w), {
            "codebook": codebook_loss,
            "commitment": self.commitment_weight * commit_loss,
        }

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        """One EMA step + (every N steps) a dead-code reset using `flat` samples."""
        K = self.num_embeddings
        # First-batch k-means-style init seeds the codebook from real encoder outputs.
        if not self._kmeans_done:
            n = flat.size(0)
            if n >= K:
                idx = torch.randperm(n, device=flat.device)[:K]
                seed = flat[idx]
                self.codebook.weight.data.copy_(seed)
                self.dw_ema.copy_(seed)
                self.cluster_size_ema.fill_(1.0)
                self._kmeans_done = True
            return  # skip the EMA update on this seed step

        # Re-quantise with the new (post-init) codebook to get aligned indices.
        # Caller passed pre-init indices; cheap to redo and avoids stale assignments
        # when we just reseeded above.
        x_sq = (flat ** 2).sum(dim=1, keepdim=True)
        c_sq = (self.codebook.weight ** 2).sum(dim=1)
        xc = flat @ self.codebook.weight.t()
        dists = x_sq + c_sq - 2.0 * xc
        indices = torch.argmin(dists, dim=1)

        one_hot = F.one_hot(indices, K).to(flat.dtype)        # (N, K)
        cluster_size = one_hot.sum(dim=0)                      # (K,)
        dw = one_hot.t() @ flat                                # (K, D)

        # EMA accumulators
        self.cluster_size_ema.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.dw_ema.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

        # Laplace smoothing keeps division stable when a code is empty
        n_total = self.cluster_size_ema.sum()
        smooth = (self.cluster_size_ema + 1e-5) / (n_total + K * 1e-5) * n_total
        self.codebook.weight.data.copy_(self.dw_ema / smooth.unsqueeze(1))

        self.ema_step.add_(1)
        if (self.reset_dead_codes_every > 0
            and int(self.ema_step.item()) % self.reset_dead_codes_every == 0):
            self._reset_dead_codes(flat)

    @torch.no_grad()
    def _reset_dead_codes(self, flat: torch.Tensor) -> None:
        """Replace codes whose EMA cluster size is below threshold with sample vectors."""
        dead = self.cluster_size_ema < self.dead_code_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return
        n = flat.size(0)
        take = min(n_dead, n)
        idx = torch.randperm(n, device=flat.device)[:take]
        repl = flat[idx]
        # Pad with first sample if we somehow have fewer encoder samples than dead codes.
        if take < n_dead:
            pad = repl[0:1].expand(n_dead - take, repl.size(1))
            repl = torch.cat([repl, pad], dim=0)
        target_idx = dead.nonzero(as_tuple=True)[0]
        self.codebook.weight.data[target_idx] = repl
        self.dw_ema[target_idx] = repl
        self.cluster_size_ema[target_idx] = 1.0

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
            num_embeddings=self.cfg.num_embeddings,
            embedding_dim=self.cfg.embedding_dim,
            commitment_weight=self.cfg.commitment_weight,
            use_ema=self.cfg.use_ema,
            ema_decay=self.cfg.ema_decay,
            dead_code_threshold=self.cfg.dead_code_threshold,
            reset_dead_codes_every=self.cfg.reset_dead_codes_every,
            kmeans_init=self.cfg.kmeans_init,
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
