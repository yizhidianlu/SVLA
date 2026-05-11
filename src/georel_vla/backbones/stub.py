"""Tiny CNN backbone that returns same-shape tensors as a real SigLIP encoder.

For unit tests and CI smoke runs without open-pi-zero / PaliGemma weights.
Real training in Phase 1.7 uses `Pi0Backbone` instead.

The architecture is deliberately small (~0.5 M params) so tests stay fast;
shapes match PaliGemma-3B SigLIP defaults: 256 image tokens of 1152 dim
each at 224x224 input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import VLABackbone


@dataclass
class StubBackboneConfig:
    siglip_dim: int = 1152
    n_image_tokens: int = 256
    image_size: int = 224           # ViT-So400m default; assumes 14x14 -> 16x16 patches
    hidden: int = 64
    n_conv_stages: int = 4          # 4 stride-2 stages: 224 -> 14
    out_channels_eq_siglip: bool = True


class StubBackbone(VLABackbone):
    """Random-init CNN producing tensors shaped like SigLIP outputs.

    Use only for tests / smoke-style verifications. Encoder is a small
    stride-2 conv stack down-sampling 224 -> 14 (4 stages), reshaped to
    (B, 196, hidden) and projected up to (B, 256, siglip_dim) via a learnable
    [256, 196] linear mix + per-token MLP.

    Why 256 not 196?  PaliGemma-3B's SigLIP uses 224/14 = 16 -> 256 tokens
    after applying a small 1x1 pre-projection; we mimic that 256-token shape
    exactly via a learned token-count adapter rather than getting bogged
    down in matching SigLIP's exact patchification.
    """

    def __init__(self, cfg: StubBackboneConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or StubBackboneConfig()
        c = self.cfg

        # set class-level descriptors for the depth expert to read off.
        # We set on the instance so multiple stubs with different sizes
        # don't stomp on each other in tests.
        self.__dict__["siglip_dim"] = c.siglip_dim
        self.__dict__["n_image_tokens"] = c.n_image_tokens

        # stride-2 conv stack: 224 -> 14 in four stages.
        layers: list[nn.Module] = [nn.Conv2d(3, c.hidden, 3, padding=1)]
        for _ in range(c.n_conv_stages):
            layers.extend([
                nn.SiLU(inplace=True),
                nn.Conv2d(c.hidden, c.hidden, 4, stride=2, padding=1),
            ])
        layers.append(nn.SiLU(inplace=True))
        self.encoder = nn.Sequential(*layers)

        # token-count adapter: 14*14 = 196 -> n_image_tokens via learnable mix.
        # In practice depth_expert.cfg.n_img_tokens is the target.
        spatial_after = (c.image_size // (2 ** c.n_conv_stages)) ** 2
        self.token_mix = nn.Linear(spatial_after, c.n_image_tokens)
        self.proj = nn.Linear(c.hidden, c.siglip_dim)

    def encode_image_to_siglip(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.size(1) != 3:
            raise ValueError(f"expected (B, 3, H, W); got {tuple(rgb.shape)}")
        x = self.encoder(rgb)                 # (B, hidden, S, S) where S = H/2^stages
        b, h, s, _ = x.shape
        x = x.reshape(b, h, s * s)            # (B, hidden, S*S)
        x = self.token_mix(x)                 # (B, hidden, n_tokens)
        x = x.transpose(1, 2)                 # (B, n_tokens, hidden)
        x = self.proj(x)                      # (B, n_tokens, siglip_dim)
        return x


__all__ = ["StubBackbone", "StubBackboneConfig"]
