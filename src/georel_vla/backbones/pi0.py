"""Thin wrapper around open-pi-zero's PiZero (`third_party/open-pi-zero`).

Status (Phase 1.3, 2026-05-11)
    * `third_party/open-pi-zero` is pinned via git submodule.
    * This module declares the *interface* that GeoRel-VLA's depth expert and
      model.py expect from the backbone (load + forward + SigLIP-feature
      extraction hook), with a working `is_available()` probe.
    * Methods that require the actual upstream `PiZero` instance raise
      NotImplementedError — they are filled in during Phase 1.6 (model.py
      wiring), once we have a Hydra config that selects a concrete checkpoint.

Why this split
    open-pi-zero instantiates PiZero from a heavy Hydra config (PaliGemma 3B
    VLM + ~0.3B action expert + flow-matching action head). Standing up that
    config the same time as we land the VQ-VAE / depth expert / loss module
    would make the diff unreviewable. By cleanly separating "interface" and
    "implementation" here, the depth expert + loss work can be unit-tested
    against a small stub today, and the real wiring is a self-contained
    change in Phase 1.6.

Integration plan (Phase 1.6)
    1. Pick a base config from `third_party/open-pi-zero/config/` (likely
       `train/pg_oxe.yaml` then specialise for LIBERO via overrides).
    2. Resolve via `hydra.utils.instantiate(cfg)` and load the published
       checkpoint from `allenzren/open-pi-zero` on HuggingFace.
    3. Surface the SigLIP image features by hooking into PiZero's
       `joint_model.vlm` — we need the 256 visual tokens *before* language
       fusion so they can feed the depth expert (per QDepth-VLA §3.3
       "depth expert takes the visual embeddings from the SigLIP encoder ...
       before language fusion to avoid semantic interference").
    4. Replace the NotImplementedError stubs below with the real glue.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch import nn


THIRD_PARTY_OPEN_PI_ZERO = Path(__file__).resolve().parents[3] / "third_party" / "open-pi-zero"


def is_available() -> bool:
    """Return True if open-pi-zero submodule + its python deps can be imported."""
    if not (THIRD_PARTY_OPEN_PI_ZERO / "src" / "model" / "vla" / "pizero.py").is_file():
        return False
    # Ensure the submodule's src/ is on sys.path so `from src.model.vla.pizero import PiZero` works.
    add_third_party_to_syspath()
    try:
        spec = importlib.util.find_spec("src.model.vla.pizero")
    except (ImportError, ValueError, ModuleNotFoundError):
        return False
    return spec is not None


def add_third_party_to_syspath() -> None:
    """Idempotently insert `third_party/open-pi-zero/` at sys.path[0].

    open-pi-zero uses `from src.model... import ...` absolute imports, so its
    repo root (not the `src/` dir) needs to be on sys.path.
    """
    p = str(THIRD_PARTY_OPEN_PI_ZERO)
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass
class Pi0BackboneConfig:
    """Light config for the wrapper; the real PiZero Hydra config arrives in Phase 1.6."""

    config_path: str | None = None          # e.g., "config/train/pg_oxe.yaml" relative to submodule
    config_name: str | None = None          # for Hydra overrides
    checkpoint: str | None = None           # "allenzren/open-pi-zero" or local path
    device: str = "cuda"
    dtype: str = "bf16"
    image_size: int = 224                   # SigLIP input
    expose_siglip_features: bool = True     # whether forward() also returns pre-fusion image tokens


class Pi0Backbone:
    """Adapter exposing the slice of open-pi-zero our model.py + depth expert need.

    Phase 1.6 will replace the stubs with calls into `third_party/open-pi-zero`.
    """

    def __init__(self, cfg: Pi0BackboneConfig | None = None) -> None:
        self.cfg = cfg or Pi0BackboneConfig()
        self._pizero: nn.Module | None = None  # populated in load()

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Instantiate the upstream PiZero and load the checkpoint.

        Filled in Phase 1.6 — needs the Hydra config wiring.
        """
        raise NotImplementedError("Phase 1.6 — see docstring at the top of this file.")

    # -- forward hooks the depth expert + model.py call -------------------

    def encode_image_to_siglip(self, rgb: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) RGB -> (B, n_tokens, siglip_dim) pre-fusion visual tokens.

        This is the slice that gets fed into the depth expert per
        QDepth-VLA §3.3 "before language fusion to avoid semantic
        interference". Phase 1.6 wires this through PiZero's SigLIP module.
        """
        raise NotImplementedError("Phase 1.6 — hooks into joint_model.vlm SigLIP.")

    def forward_action(
        self,
        rgb: torch.Tensor,
        language_tokens: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Standard PiZero forward returning (action_chunk, aux_dict).

        Phase 1.6 wires through PiZero.forward / .generate_actions.
        `aux_dict` will surface intermediates (SigLIP features, hybrid-attn
        layout) that the depth expert and the cross-head consistency loss
        in Phase 2 will consume.
        """
        raise NotImplementedError("Phase 1.6 — calls PiZero.forward.")

    # -- introspection ----------------------------------------------------

    @property
    def siglip_dim(self) -> int:
        return 1152

    @property
    def n_image_tokens(self) -> int:
        return 256

    def describe(self) -> dict[str, Any]:
        return {
            "submodule_path": str(THIRD_PARTY_OPEN_PI_ZERO),
            "submodule_available": is_available(),
            "config": self.cfg.__dict__,
            "loaded": self._pizero is not None,
        }


__all__ = ["Pi0Backbone", "Pi0BackboneConfig", "is_available", "add_third_party_to_syspath"]
