"""GeoRel-VLA top-level model wiring (skeleton).

Composition: open-π₀ action expert + four auxiliary experts (depth, normal, plane,
support). Loss = L_action + λ_d·L_depth + λ_n·L_normal + λ_p·L_plane + λ_s·L_support
       + λ_xc·L_xc + λ_dr·L_dr.

See docs/experiment_blueprint.md §2 for the full architectural spec and
docs/ideation_summary.md §4 for the thesis. Bodies are implemented in CODING
stage (Phase 1 reproduces QDepth-VLA = depth head only; Phase 2 adds normal +
plane; Phase 3 adds support).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GeoRelVLAConfig:
    """Top-level config. Filled in CODING stage to mirror configs/*.yaml."""

    backbone: str = "open-pi-0"
    vlm: str = "google/paligemma-3b-mix-224"
    image_size: int = 224

    # head toggles — controlled per-phase via configs/*.yaml
    use_depth: bool = True
    use_normal: bool = False
    use_plane: bool = False
    use_support: bool = False
    use_cross_consistency: bool = False
    use_derivability: bool = False

    # codebook sizes
    depth_codebook_size: int = 256
    normal_codebook_size: int = 128
    plane_codebook_size: int = 64

    # loss weights (initial values; QDepth-VLA defaults — schedule applies in train)
    lambda_depth: float = 0.01
    lambda_normal: float = 0.01
    lambda_plane: float = 0.01
    lambda_support: float = 0.005
    lambda_cross_consistency: float = 0.002
    lambda_derivability: float = 0.003


class GeoRelVLA:
    """Skeleton. CODING stage replaces this with a torch.nn.Module subclass."""

    def __init__(self, cfg: GeoRelVLAConfig) -> None:
        self.cfg = cfg
        # backbone + experts wired in CODING stage

    def __repr__(self) -> str:  # pragma: no cover
        heads = [
            n
            for n, on in [
                ("depth", self.cfg.use_depth),
                ("normal", self.cfg.use_normal),
                ("plane", self.cfg.use_plane),
                ("support", self.cfg.use_support),
            ]
            if on
        ]
        return f"GeoRelVLA(backbone={self.cfg.backbone}, heads={heads})"
