"""GeoRelVLA — the GeoRel-VLA top-level model.

Composes a VLA `backbone` (open-π₀ in production via `Pi0Backbone`, a tiny
CNN in tests via `StubBackbone`) with our `DepthExpert` and a **frozen**
VQ-VAE codebook for depth tokens. Loss = `L_action` (provided by the
backbone in Phase 1.7) + `lambda_t * L_depth` (cross-entropy of expert
logits against pre-computed VQ-VAE indices).

Per QDepth-VLA arXiv:2510.14836:

* §3.2 — VQ-VAE is pretrained independently and **frozen** during VLA
  training. We hold the codebook as a non-trainable buffer.
* §3.3 — depth expert reads SigLIP features **before language fusion**
  to avoid semantic interference. The backbone's
  `encode_image_to_siglip()` returns exactly that slice.
* §3.4.3 — total loss is `L_action + lambda_t * L_depth`,
  `lambda_t = lambda_0 * gamma^t` with `lambda_0 = 0.01`,
  `gamma ≈ 0.9999` per training step.

Phase split
* Phase 1.6 (this file): depth-side wiring + co-training loss helper +
  `compute_losses()`. Action side is `None` until the backbone supports
  `forward_action()`.
* Phase 1.7 (next): switch the default backbone to `Pi0Backbone`, fill
  `Pi0Backbone.load()` + `forward_action()`, plug the CFM action loss
  into `compute_losses()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbones.base import VLABackbone
from .codebooks.vqvae import VQVAE, VQVAEConfig
from .experts.depth_expert import DepthExpert, DepthExpertConfig
from .losses.qdepth import (
    TokenLayout,
    build_hybrid_attention_mask,
    depth_ce_loss,
)


@dataclass
class GeoRelVLAConfig:
    """Top-level config. Phase 1 keeps only depth + action heads enabled.

    Phase 2 toggles `use_normal` and `use_plane`; Phase 3 toggles `use_support`
    + `use_derivability`. They live here so the same config dataclass survives
    end-to-end without a v2.
    """

    backbone: str = "open-pi-0"

    # depth head config (mirror VQ-VAE / DepthExpert defaults)
    depth_codebook_size: int = 256
    depth_embedding_dim: int = 160
    depth_latent_h: int = 16
    depth_latent_w: int = 16

    # head toggles — Phase 1 only depth.
    use_depth: bool = True
    use_normal: bool = False
    use_plane: bool = False
    use_support: bool = False
    use_cross_consistency: bool = False
    use_derivability: bool = False

    # loss weights (initial values; QDepth-VLA defaults — schedule applies in train)
    lambda_depth: float = 0.01
    lambda_normal: float = 0.01
    lambda_plane: float = 0.01
    lambda_support: float = 0.005
    lambda_cross_consistency: float = 0.002
    lambda_derivability: float = 0.003

    # Phase-2 normal head config.
    normal_codebook_size: int = 128
    normal_embedding_dim: int = 160
    normal_latent_h: int = 16
    normal_latent_w: int = 16

    # action chunk size for the (Phase-1.7) action-loss compute.
    action_chunk_size: int = 4

    # proprio token count for hybrid attention layout (open-π₀ default = cond_steps=1).
    n_proprio_tokens: int = 1


def _exp_decay_lambda(lambda_0: float, gamma: float, step: int) -> float:
    """`lambda_t = lambda_0 * gamma^t` per QDepth-VLA §3.4.3."""
    return float(lambda_0 * (gamma ** max(0, int(step))))


class GeoRelVLA(nn.Module):
    """Backbone + DepthExpert + frozen VQ-VAE codebook.

    Construction is explicit (pass instantiated components) so callers can
    swap backbones / share VQ-VAEs across runs. Use `GeoRelVLA.from_config`
    for the convenience path that builds a tiny stub model end-to-end.
    """

    def __init__(
        self,
        cfg: GeoRelVLAConfig,
        backbone: VLABackbone,
        depth_expert: DepthExpert,
        depth_codebook: torch.Tensor,
        normal_expert: DepthExpert | None = None,
        normal_codebook: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, VLABackbone):
            raise TypeError(
                f"backbone must subclass VLABackbone; got {type(backbone).__name__}"
            )
        if depth_codebook.shape != (cfg.depth_codebook_size, cfg.depth_embedding_dim):
            raise ValueError(
                f"depth_codebook shape {tuple(depth_codebook.shape)} != "
                f"({cfg.depth_codebook_size}, {cfg.depth_embedding_dim})"
            )
        if depth_expert.cfg.num_embeddings != cfg.depth_codebook_size:
            raise ValueError(
                "depth_expert.cfg.num_embeddings must match cfg.depth_codebook_size"
            )
        if depth_expert.cfg.siglip_dim != backbone.siglip_dim:
            raise ValueError(
                f"depth_expert.cfg.siglip_dim ({depth_expert.cfg.siglip_dim}) "
                f"!= backbone.siglip_dim ({backbone.siglip_dim})"
            )

        self.cfg = cfg
        self.backbone = backbone
        self.depth_expert = depth_expert

        # Codebook is frozen during VLA training (QDepth-VLA §3.2).
        # Held as a buffer so it round-trips with state_dict() / .to(device).
        self.register_buffer("depth_codebook", depth_codebook.detach().clone())

        # Phase-2 optional normal head. When provided + cfg.use_normal, the
        # forward pass also produces normal logits and compute_losses adds the
        # auxiliary L_normal term with its own lambda + gamma schedule.
        self.normal_expert = None
        if normal_expert is not None and cfg.use_normal:
            self.normal_expert = normal_expert
            if normal_codebook is None:
                raise ValueError("normal_codebook required when normal_expert is provided")
            self.register_buffer("normal_codebook", normal_codebook.detach().clone())

    # -- factory ---------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: GeoRelVLAConfig | None = None,
        backbone: VLABackbone | None = None,
    ) -> GeoRelVLA:
        """Build a self-contained model with sensible defaults.

        If `backbone` is None, instantiates a `StubBackbone` so callers can
        smoke-test the whole pipeline without open-pi-zero / PaliGemma. Real
        training in Phase 1.7 always passes a `Pi0Backbone` explicitly.
        """
        from .backbones.stub import StubBackbone, StubBackboneConfig

        cfg = cfg or GeoRelVLAConfig()
        backbone = backbone or StubBackbone(StubBackboneConfig(
            siglip_dim=1152, n_image_tokens=256,
        ))
        depth_expert = DepthExpert(DepthExpertConfig(
            siglip_dim=backbone.siglip_dim,
            n_img_tokens=backbone.n_image_tokens,
            latent_h=cfg.depth_latent_h,
            latent_w=cfg.depth_latent_w,
            embedding_dim=cfg.depth_embedding_dim,
            num_embeddings=cfg.depth_codebook_size,
        ))
        # build a fresh VQ-VAE just to get a properly-shaped codebook;
        # in production Phase 1.7 passes a *trained* codebook here.
        vq = VQVAE(VQVAEConfig(
            num_embeddings=cfg.depth_codebook_size,
            embedding_dim=cfg.depth_embedding_dim,
        ))
        codebook = vq.quantizer.codebook.weight.detach().clone()
        return cls(cfg, backbone, depth_expert, codebook)

    # -- forward ---------------------------------------------------------

    def encode_siglip(self, rgb: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) RGB -> (B, N_img, siglip_dim) image tokens (pre language fusion)."""
        return self.backbone.encode_image_to_siglip(rgb)

    def forward_depth(self, siglip_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the depth expert; return logits + raw embed for downstream losses."""
        depth_logits, depth_embed = self.depth_expert(siglip_features, self.depth_codebook)
        return {"depth_logits": depth_logits, "depth_embed": depth_embed}

    def forward_normal(self, siglip_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the normal expert (Phase-2); empty if disabled."""
        if self.normal_expert is None:
            return {}
        n_logits, n_embed = self.normal_expert(siglip_features, self.normal_codebook)
        return {"normal_logits": n_logits, "normal_embed": n_embed}

    def forward(
        self,
        rgb: torch.Tensor,
        depth_target_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Phase-1.6 forward path — depth side end-to-end; action side is None.

        Args
            rgb                   : (B, 3, H, W).
            depth_target_indices  : (B, latent_h * latent_w) long tensor from a
                                    pretrained VQ-VAE. If None, no depth-CE loss
                                    is returned (use this path at inference).

        Returns dict with keys:
            siglip                : (B, N_img, siglip_dim)
            depth_logits          : (B, N_lat, K)
            depth_embed           : (B, embed_dim, latent_h, latent_w)
            losses                : dict of named losses (see below). Empty if
                                    depth_target_indices is None.
            action                : None (Phase-1.7 plugs in the action chunk).
        """
        siglip = self.encode_siglip(rgb)
        depth_out = self.forward_depth(siglip)

        out: dict[str, torch.Tensor | None | dict] = {
            "siglip": siglip,
            **depth_out,
            "action": None,
            "losses": {},
        }

        if depth_target_indices is not None:
            losses = self.compute_losses(
                depth_logits=depth_out["depth_logits"],
                depth_target_indices=depth_target_indices,
                step=0,                              # caller can override schedule
                gamma=1.0,                           # no decay if step=0
            )
            out["losses"] = losses

        return out

    # -- losses ----------------------------------------------------------

    def compute_losses(
        self,
        depth_logits: torch.Tensor,
        depth_target_indices: torch.Tensor,
        action_pred: torch.Tensor | None = None,
        action_target: torch.Tensor | None = None,
        normal_logits: torch.Tensor | None = None,
        normal_target_indices: torch.Tensor | None = None,
        step: int = 0,
        gamma: float = 0.9999,
    ) -> dict[str, torch.Tensor]:
        """Combine the per-head losses with the QDepth-VLA exponential decay."""
        losses: dict[str, torch.Tensor] = {}
        device = depth_logits.device

        # depth CE
        l_depth = depth_ce_loss(depth_logits, depth_target_indices)
        losses["depth"] = l_depth
        lambda_d = _exp_decay_lambda(self.cfg.lambda_depth, gamma, step)
        weighted_total = lambda_d * l_depth
        losses["lambda_depth_t"] = torch.tensor(lambda_d, device=device)

        # Phase-2 normal CE (only if both logits and target supplied)
        if normal_logits is not None and normal_target_indices is not None:
            l_normal = depth_ce_loss(normal_logits, normal_target_indices)
            losses["normal"] = l_normal
            lambda_n = _exp_decay_lambda(self.cfg.lambda_normal, gamma, step)
            weighted_total = weighted_total + lambda_n * l_normal
            losses["lambda_normal_t"] = torch.tensor(lambda_n, device=device)

        if action_pred is not None and action_target is not None:
            l_action = nn.functional.mse_loss(action_pred, action_target)
            losses["action"] = l_action
            weighted_total = weighted_total + l_action

        losses["total"] = weighted_total
        return losses

    # -- attention layout ------------------------------------------------

    def attention_layout(
        self,
        n_text: int,
        include_depth: bool = True,
        include_action: bool = True,
    ) -> TokenLayout:
        """Compose the QDepth-VLA hybrid TokenLayout for this model.

        Order matches `losses.qdepth.build_hybrid_attention_mask` —
        text | image | depth | proprio | action.
        """
        return TokenLayout(
            n_text=n_text,
            n_image=self.backbone.n_image_tokens,
            n_depth=self.cfg.depth_latent_h * self.cfg.depth_latent_w if include_depth else 0,
            n_proprio=self.cfg.n_proprio_tokens,
            n_action=self.cfg.action_chunk_size if include_action else 0,
        )

    def build_attention_mask(
        self,
        n_text: int,
        include_depth: bool = True,
        include_action: bool = True,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        return build_hybrid_attention_mask(
            self.attention_layout(n_text, include_depth, include_action),
            device=device,
        )

    # -- plumbing --------------------------------------------------------

    def freeze_codebook(self) -> None:
        """No-op marker: the codebook is already a buffer (non-trainable).

        Provided so callers can be explicit about intent and so future
        refactors that promote the codebook to an nn.Parameter break loudly.
        """
        # Buffers are never in `parameters()`; verify on demand:
        for n, p in self.named_parameters():
            if "depth_codebook" in n:
                raise RuntimeError(
                    f"depth_codebook leaked into parameters at {n}; promote back to a buffer"
                )

    def __repr__(self) -> str:  # pragma: no cover
        heads = [
            n for n, on in [
                ("depth", self.cfg.use_depth),
                ("normal", self.cfg.use_normal),
                ("plane", self.cfg.use_plane),
                ("support", self.cfg.use_support),
            ] if on
        ]
        bk = type(self.backbone).__name__
        return f"GeoRelVLA(backbone={bk}, heads={heads})"


__all__ = ["GeoRelVLA", "GeoRelVLAConfig"]
