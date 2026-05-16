"""Compute surface normals from depth maps using camera intrinsics.

Surface normal is the cross product of the local 3D-surface tangent vectors,
computed from depth gradients in image space + perspective unprojection.

Used by Phase 2 multi-head training: a frozen normal VQ-VAE codebook
provides quantised normal indices as auxiliary cross-entropy targets,
without requiring separate ground-truth normal extraction (we just derive
it analytically from the depth GT already on disk).
"""

from __future__ import annotations

import numpy as np
import torch


# LIBERO Robosuite agentview default intrinsics (256x256 render, ~90 FOV).
# These are approximate; surface normal quantisation only needs consistent
# units across pretrain / train, not perfect calibration.
_DEFAULT_FX = 128.0
_DEFAULT_FY = 128.0
_DEFAULT_CX = 128.0
_DEFAULT_CY = 128.0


def depth_to_normal_np(
    depth: np.ndarray,
    fx: float = _DEFAULT_FX,
    fy: float = _DEFAULT_FY,
    cx: float = _DEFAULT_CX,
    cy: float = _DEFAULT_CY,
) -> np.ndarray:
    """Analytic surface normal from (H, W) depth.

    Returns (H, W, 3) unit-norm vectors. Pixels with depth <= 0 get (0, 0, 1).
    """
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    v = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
    z = depth.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # gradients with central differences
    dxdu = np.gradient(x, axis=1)
    dydv = np.gradient(y, axis=0)
    dzdu = np.gradient(z, axis=1)
    dzdv = np.gradient(z, axis=0)

    # tangent vectors in 3D
    t_u = np.stack([dxdu, np.zeros_like(dxdu), dzdu], axis=-1)
    t_v = np.stack([np.zeros_like(dydv), dydv, dzdv], axis=-1)
    n = np.cross(t_u, t_v)
    norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    n = n / norm
    # invalid depth -> upward normal
    n[depth <= 0.0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return n.astype(np.float32)


def depth_to_normal_torch(depth: torch.Tensor) -> torch.Tensor:
    """Batched depth -> normal on GPU. depth (B, 1, H, W) -> normal (B, 3, H, W)."""
    if depth.ndim != 4 or depth.size(1) != 1:
        raise ValueError(f"depth must be (B,1,H,W); got {tuple(depth.shape)}")
    b, _, h, w = depth.shape
    device = depth.device
    u = torch.arange(w, device=device, dtype=depth.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    v = torch.arange(h, device=device, dtype=depth.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    fx = fy = float(h) / 2.0  # approximate FOV ~90deg
    cx = cy = float(h) / 2.0
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    # discrete differences via 1-D conv-style slicing
    dx_u = torch.zeros_like(x); dx_u[..., 1:-1] = (x[..., 2:] - x[..., :-2]) * 0.5
    dy_v = torch.zeros_like(y); dy_v[:, :, 1:-1, :] = (y[:, :, 2:, :] - y[:, :, :-2, :]) * 0.5
    dz_u = torch.zeros_like(z); dz_u[..., 1:-1] = (z[..., 2:] - z[..., :-2]) * 0.5
    dz_v = torch.zeros_like(z); dz_v[:, :, 1:-1, :] = (z[:, :, 2:, :] - z[:, :, :-2, :]) * 0.5
    # cross product
    nx = -dy_v * dz_u
    ny = -dx_u * dz_v
    nz = dx_u * dy_v
    n = torch.cat([nx, ny, nz], dim=1)  # (B, 3, H, W)
    norm = n.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    return n / norm


__all__ = ["depth_to_normal_np", "depth_to_normal_torch"]
