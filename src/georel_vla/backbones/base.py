"""Abstract VLA backbone interface that GeoRelVLA composes against.

GeoRelVLA reads `siglip_dim` and `n_image_tokens` off the backbone, calls
`encode_image_to_siglip(rgb)` to get pre-fusion image tokens for the depth
expert, and (Phase 1.7) will call `forward_action(...)` to produce the action
chunk + the CFM action-loss.

Concrete subclasses:
* `Pi0Backbone` (backbones/pi0.py) — wraps open-pi-zero; production path.
  `load()` and `encode_image_to_siglip()` are still NotImplementedError stubs
  until Phase 1.7.
* `StubBackbone` (backbones/stub.py) — tiny CNN producing tensors of the
  same shape; used for unit tests and dry-run smoke runs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VLABackbone(nn.Module):
    """Interface contract — concrete backbones override `encode_image_to_siglip`.

    Subclasses set `siglip_dim` and `n_image_tokens` as class attrs so the
    depth expert can size itself off the backbone without a forward pass.
    """

    #: SigLIP feature dim per token (PaliGemma-3B default is 1152).
    siglip_dim: int = 1152
    #: Number of visual tokens per image (16x16 patches at 224x224 / 14 = 256).
    n_image_tokens: int = 256

    def encode_image_to_siglip(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args
            rgb : (B, 3, H, W) float tensor; backbone may resize internally.

        Returns
            (B, n_image_tokens, siglip_dim) pre-language-fusion visual tokens.
        """
        raise NotImplementedError("override in VLABackbone subclass")

    # forward_action stays as a Phase-1.7 hook — see Pi0Backbone for details.


__all__ = ["VLABackbone"]
