"""QDepth-VLA loss term + hybrid-attention mask construction.

References
    QDepth-VLA, arXiv:2510.14836:
    * §3.4.1 — eq. (4): cross-entropy of depth-expert logits against VQ-VAE
      ground-truth code indices.
    * §3.3   — hybrid attention mask:
        - text attends within text only
        - image attends within image only
        - depth attends image + text
        - action attends ALL preceding (text + image + depth + proprio)
      Designed so depth supervision shapes the visual encoder without leaking
      noisy depth signal into the pretrained VLM semantic alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def depth_ce_loss(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """QDepth-VLA eq. (4): cross-entropy over codebook.

    Args
        logits         : (B, N_lat, K) depth-expert logits.
        target_indices : (B, N_lat) long tensor of VQ-VAE indices.
        reduction      : 'mean' | 'sum' | 'none'.
        ignore_index   : positions to skip (e.g., padded frames).

    Returns
        Scalar (or (B, N_lat) if reduction='none') loss.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be (B, N_lat, K); got shape {tuple(logits.shape)}")
    b, n_lat, k = logits.shape
    if target_indices.shape != (b, n_lat):
        raise ValueError(
            f"target shape {tuple(target_indices.shape)} must equal {(b, n_lat)}"
        )
    return F.cross_entropy(
        logits.reshape(b * n_lat, k),
        target_indices.reshape(b * n_lat),
        reduction=reduction,
        ignore_index=ignore_index,
    )


@dataclass
class TokenLayout:
    """How many tokens of each modality, in concatenation order.

    Concatenation order assumed by `build_hybrid_attention_mask`:
        [ text | image | depth | proprio | action ]

    Use 0 for any modality not present in a particular forward pass (e.g., set
    `n_depth = 0` at inference, or `n_action = 0` during VLM-only encoding).
    """

    n_text: int
    n_image: int
    n_depth: int = 0
    n_proprio: int = 0
    n_action: int = 0

    @property
    def total(self) -> int:
        return self.n_text + self.n_image + self.n_depth + self.n_proprio + self.n_action

    def boundaries(self) -> dict[str, slice]:
        s_text = slice(0, self.n_text)
        s_image = slice(self.n_text, self.n_text + self.n_image)
        s_depth = slice(s_image.stop, s_image.stop + self.n_depth)
        s_proprio = slice(s_depth.stop, s_depth.stop + self.n_proprio)
        s_action = slice(s_proprio.stop, s_proprio.stop + self.n_action)
        return {
            "text": s_text, "image": s_image, "depth": s_depth,
            "proprio": s_proprio, "action": s_action,
        }


def build_hybrid_attention_mask(
    layout: TokenLayout,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Construct the QDepth-VLA hybrid attention mask (§3.3).

    Returns a square boolean mask of shape (T, T) where T == layout.total,
    using the convention `True` means "query may attend to key" (matching
    PyTorch `MultiheadAttention(attn_mask=...)` when `dtype=torch.bool`
    semantics are inverted via `~mask`; we return the *attend* form so call
    sites can choose).

    Visual layout (rows = query modality, cols = key modality, T = True):
                    text  image  depth  proprio  action
        text         T     F      F      F        F
        image        F     T      F      F        F
        depth        T     T      T      F        F
        proprio      F     F      F      T        F
        action       T     T      T      T        T (causal within action span)

    Notes
        * Within-modality blocks are always self-attendable (the diagonal is True).
        * Action span uses CAUSAL attention (lower-triangular within its block)
          so action tokens are produced left-to-right; the rest is full block.
        * Proprio attends only itself; action sees proprio.
        * Depth never attends action / proprio (cannot leak future state).
    """
    t = layout.total
    if t == 0:
        return torch.empty(0, 0, dtype=dtype, device=device)

    mask = torch.zeros(t, t, dtype=torch.bool, device=device)
    spans = layout.boundaries()

    # Within-modality self-attention
    for name in ("text", "image", "depth", "proprio", "action"):
        s = spans[name]
        if s.stop > s.start:
            mask[s, s] = True

    # depth attends text + image
    if layout.n_depth > 0:
        mask[spans["depth"], spans["text"]] = True
        mask[spans["depth"], spans["image"]] = True

    # action attends text + image + depth + proprio
    if layout.n_action > 0:
        for k in ("text", "image", "depth", "proprio"):
            mask[spans["action"], spans[k]] = True

        # Causal mask within the action span (lower-triangular, including diagonal)
        a = spans["action"]
        action_block = mask[a, a].clone()
        causal = torch.tril(torch.ones_like(action_block))
        mask[a, a] = action_block & causal.bool()

    return mask.to(dtype)


def attn_mask_for_pytorch_mha(attend_mask: torch.Tensor) -> torch.Tensor:
    """Convert (T, T) bool *attend* mask to PyTorch MHA additive float mask.

    PyTorch nn.MultiheadAttention(attn_mask=...) expects:
      - bool: True means "do NOT attend" (opposite convention)
      - float: 0.0 means attend, -inf means do not attend

    We always emit float -inf form for safety across PyTorch versions.
    """
    out = torch.zeros_like(attend_mask, dtype=torch.float32)
    out.masked_fill_(~attend_mask.bool(), float("-inf"))
    return out


__all__ = [
    "TokenLayout",
    "depth_ce_loss",
    "build_hybrid_attention_mask",
    "attn_mask_for_pytorch_mha",
]
