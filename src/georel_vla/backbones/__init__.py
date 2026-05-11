"""Action / VLM backbones consumed by GeoRel-VLA.

Phase 1 uses open-pi-zero (allenzren/open-pi-zero, pinned via git submodule
in `third_party/open-pi-zero`). The wrapper in `pi0.py` is intentionally a
*skeleton* — full forward integration (loading PaliGemma + SigLIP + action
expert weights, exposing SigLIP features to the depth expert, and the hybrid
attention plumbing) lands in Phase 1.7 (training scripts), at which point
`Pi0Backbone.load()` stops raising NotImplementedError.

For unit tests and dry-run smoke we use `StubBackbone`, a tiny CNN that
returns tensors of the same shape (256 image tokens of 1152 dim each) so
the rest of the model can be exercised without paying the PaliGemma load.

`VLABackbone` is the ABC both subclasses extend. Add new backbones (e.g.,
OpenVLA, raw HF PaliGemma) by inheriting from it and implementing
`encode_image_to_siglip`.
"""

from .base import VLABackbone
from .pi0 import Pi0Backbone, Pi0BackboneConfig
from .stub import StubBackbone, StubBackboneConfig

__all__ = [
    "VLABackbone",
    "Pi0Backbone",
    "Pi0BackboneConfig",
    "StubBackbone",
    "StubBackboneConfig",
]
