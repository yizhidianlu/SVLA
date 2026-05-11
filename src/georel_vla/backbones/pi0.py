"""Real `Pi0Backbone` — wraps open-pi-zero's PiZero (`third_party/open-pi-zero`).

Phase 1.7a: `load()` and `encode_image_to_siglip()` are now wired. The
checkpoint loader is best-effort (skipped if PaliGemma weights aren't on
disk yet — the depth expert can train on top of random-init SigLIP, which
matches QDepth-VLA's recipe of "VLM is fine-tuned during VLA training").

Architecture surface exposed to the rest of GeoRel-VLA:
    * `siglip_dim = 1152`, `n_image_tokens = 256` (PaliGemma-3B / SigLIP-So400m)
    * `encode_image_to_siglip(rgb) -> (B, 256, 1152)`           — what depth expert eats
    * `forward_action(...)` (Phase 1.7c) — CFM action head call

Submodule layout assumption: `third_party/open-pi-zero/src/model/vla/pizero.py`
exists; the wrapper inserts `third_party/open-pi-zero/` on sys.path because
the upstream uses `from src.model... import ...` absolute imports.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .base import VLABackbone

log = logging.getLogger(__name__)

THIRD_PARTY_OPEN_PI_ZERO = Path(__file__).resolve().parents[3] / "third_party" / "open-pi-zero"

#: Default LIBERO model config that ships with this repo (configs/pi0_libero.yaml).
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "pi0_libero.yaml"
)

#: Where open-pi-zero's README expects PaliGemma weights:
#:     `${TRANSFORMERS_CACHE}/paligemma-3b-pt-224`
#: With our `HF_HOME=/root/autodl-tmp/hf` convention this becomes
#:     `/root/autodl-tmp/hf/paligemma-3b-pt-224`.
DEFAULT_PALIGEMMA_DIR_NAME = "paligemma-3b-pt-224"


def is_available() -> bool:
    """Return True if open-pi-zero submodule + its python deps can be imported."""
    if not (THIRD_PARTY_OPEN_PI_ZERO / "src" / "model" / "vla" / "pizero.py").is_file():
        return False
    add_third_party_to_syspath()
    try:
        spec = importlib.util.find_spec("src.model.vla.pizero")
    except (ImportError, ValueError, ModuleNotFoundError):
        return False
    return spec is not None


def add_third_party_to_syspath() -> None:
    """Idempotently insert `third_party/open-pi-zero/` at sys.path[0]."""
    p = str(THIRD_PARTY_OPEN_PI_ZERO)
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass
class Pi0BackboneConfig:
    """Wrapper-level config — distinct from the deep PiZero Hydra config."""

    config_path: Path | None = None         # defaults to DEFAULT_CONFIG_PATH
    paligemma_dir: Path | None = None       # default = $TRANSFORMERS_CACHE/paligemma-3b-pt-224
    load_paligemma: bool = True             # if False, leave SigLIP / Gemma random-init
    device: str = "cuda"
    dtype: str = "bf16"                     # `bf16` | `fp16` | `fp32`
    image_size: int = 224
    expose_siglip_features: bool = True


class Pi0Backbone(VLABackbone):
    """Wraps open-pi-zero PiZero; exposes the slice GeoRelVLA + DepthExpert need."""

    siglip_dim: int = 1152
    n_image_tokens: int = 256

    def __init__(self, cfg: Pi0BackboneConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or Pi0BackboneConfig()
        self._pizero: nn.Module | None = None
        self._pizero_cfg: Any = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Instantiate the upstream PiZero and (best-effort) load PaliGemma."""
        if self._pizero is not None:
            return  # idempotent

        if not is_available():
            raise RuntimeError(
                f"open-pi-zero submodule not initialised at {THIRD_PARTY_OPEN_PI_ZERO}. "
                f"Run `git submodule update --init --recursive` and `pip install --no-deps -e "
                f"{THIRD_PARTY_OPEN_PI_ZERO}` plus `pip install hydra-core omegaconf einops`."
            )

        from omegaconf import OmegaConf

        cfg_path = self.cfg.config_path or DEFAULT_CONFIG_PATH
        if not cfg_path.is_file():
            raise FileNotFoundError(f"PiZero config not found: {cfg_path}")
        pi_cfg = OmegaConf.load(cfg_path)
        OmegaConf.resolve(pi_cfg)

        from src.model.vla.pizero import PiZero

        log.info("Pi0Backbone: instantiating PiZero from %s", cfg_path)
        pizero = PiZero(pi_cfg)
        self._pizero = pizero
        self._pizero_cfg = pi_cfg

        if self.cfg.load_paligemma:
            self._best_effort_load_paligemma()

        # Move to requested device + dtype.
        target_dtype = self._resolve_dtype(self.cfg.dtype)
        self._pizero = self._pizero.to(device=self.cfg.device, dtype=target_dtype)

    def _resolve_dtype(self, name: str) -> torch.dtype:
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(name, torch.float32)

    def _best_effort_load_paligemma(self) -> None:
        """Load PaliGemma vision + LM weights into vision_tower / joint_model.vlm.

        Skipped (with a warning) if the PaliGemma snapshot directory is not
        on disk — random-init SigLIP / Gemma is acceptable for the depth
        expert smoke + early Phase-1.7 development; Phase 1.7c will require
        real weights for the action-loss to converge.
        """
        cache = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or ""
        pal_dir = self.cfg.paligemma_dir or (Path(cache) / DEFAULT_PALIGEMMA_DIR_NAME)
        if not pal_dir.is_dir():
            log.warning(
                "PaliGemma directory not found at %s — leaving SigLIP / Gemma "
                "random-init. Run `cd %s && git clone https://huggingface.co/google/paligemma-3b-pt-224` "
                "(or set Pi0BackboneConfig.paligemma_dir) before Phase-1.7c training.",
                pal_dir, cache or "<TRANSFORMERS_CACHE>",
            )
            return

        log.info("Pi0Backbone: loading PaliGemma weights from %s", pal_dir)
        # PiZero.load_pretrained_weights() takes no positional args — it reads
        # self.cfg.pretrained_model_path. Inject our resolved path into the
        # OmegaConf cfg before calling.
        from omegaconf import OmegaConf
        OmegaConf.set_struct(self._pizero.cfg, False)
        self._pizero.cfg.pretrained_model_path = str(pal_dir)
        try:
            self._pizero.load_pretrained_weights()
        except Exception as exc:
            log.warning(
                "PaliGemma load failed (%s); leaving SigLIP / Gemma random-init. "
                "Phase 1.7c training will need real weights.", exc,
            )

    # -- forward hooks ----------------------------------------------------

    def encode_image_to_siglip(self, rgb: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) RGB -> (B, 256, 1152) pre-fusion SigLIP features."""
        if self._pizero is None:
            raise RuntimeError("Pi0Backbone.load() must be called before forward")
        if rgb.ndim != 4 or rgb.size(1) != 3:
            raise ValueError(f"expected (B, 3, H, W); got {tuple(rgb.shape)}")
        # Match the dtype the backbone is in (bf16 by default).
        target_param = next(self._pizero.vision_tower.parameters())
        x = rgb.to(device=target_param.device, dtype=target_param.dtype)
        out = self._pizero.vision_tower(x)
        # SiglipVisionModel.forward returns (B, num_image_tokens, hidden_size).
        if out.ndim != 3 or out.size(1) != self.n_image_tokens or out.size(2) != self.siglip_dim:
            raise RuntimeError(
                f"SigLIP output shape {tuple(out.shape)} does not match "
                f"(B, {self.n_image_tokens}, {self.siglip_dim})"
            )
        return out

    def forward_action(
        self,
        rgb: torch.Tensor,
        language_tokens: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Standard PiZero action head. Filled in Phase 1.7c."""
        raise NotImplementedError("Phase 1.7c — calls PiZero.forward / .infer_action.")

    # -- introspection ----------------------------------------------------

    @property
    def pizero(self) -> nn.Module | None:
        """Expose the wrapped PiZero instance for downstream modules in 1.7c."""
        return self._pizero

    def describe(self) -> dict[str, Any]:
        n_params = (
            sum(p.numel() for p in self._pizero.parameters()) if self._pizero is not None else 0
        )
        return {
            "submodule_path": str(THIRD_PARTY_OPEN_PI_ZERO),
            "submodule_available": is_available(),
            "config": self.cfg.__dict__,
            "loaded": self._pizero is not None,
            "n_params": n_params,
        }


__all__ = [
    "Pi0Backbone", "Pi0BackboneConfig",
    "is_available", "add_third_party_to_syspath",
    "DEFAULT_CONFIG_PATH", "DEFAULT_PALIGEMMA_DIR_NAME",
]
