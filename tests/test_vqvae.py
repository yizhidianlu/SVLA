"""Unit tests for src/georel_vla/codebooks/vqvae.py.

Skipped on machines without torch (e.g., the GitHub-Actions Ubuntu runner
without torch in the dev deps). The full suite runs on the AutoDL pssa-vla
env where torch 2.4.1+cu124 is available.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from georel_vla.codebooks.vqvae import VQVAE, VectorQuantizer, VQVAEConfig  # noqa: E402


def test_qdepth_default_sizing() -> None:
    """QDepth-VLA §3.2: K=256, dim=160, 16x16 grid."""
    cfg = VQVAEConfig()
    assert cfg.num_embeddings == 256
    assert cfg.embedding_dim == 160
    assert cfg.commitment_weight == 0.25
    assert cfg.downsample == 16


def test_vqvae_forward_shapes_default() -> None:
    """1-channel depth in @ 256x256 -> 16x16 indices, perfect-shape reconstruction."""
    torch.manual_seed(0)
    vqvae = VQVAE(VQVAEConfig())
    x = torch.rand(2, 1, 256, 256)
    recon, indices, losses = vqvae(x)
    assert recon.shape == x.shape, recon.shape
    assert indices.shape == (2, 16, 16), indices.shape
    assert indices.dtype == torch.long
    assert (indices >= 0).all() and (indices < 256).all()
    for k in ("recon", "codebook", "commitment", "total"):
        assert k in losses
        assert losses[k].ndim == 0  # scalar


def test_vqvae_normal_3channel() -> None:
    """Normal map is 3-channel xyz; same VQ-VAE module with in/out_channels=3."""
    cfg = VQVAEConfig(in_channels=3, out_channels=3, num_embeddings=128)
    vqvae = VQVAE(cfg)
    x = torch.rand(1, 3, 256, 256) * 2 - 1  # normals in [-1, 1]
    recon, indices, losses = vqvae(x)
    assert recon.shape == x.shape
    assert indices.shape == (1, 16, 16)
    assert (indices < 128).all()


def test_vqvae_plane_64_codes() -> None:
    """Plane mask is 1-channel categorical; QDepth-style K=64 ablation."""
    cfg = VQVAEConfig(in_channels=1, out_channels=1, num_embeddings=64)
    vqvae = VQVAE(cfg)
    x = torch.rand(1, 1, 256, 256)
    _, indices, _ = vqvae(x)
    assert (indices < 64).all()


def test_vqvae_loss_decreases_with_overfit() -> None:
    """Sanity: VQ-VAE can overfit a fixed batch in a few SGD steps."""
    torch.manual_seed(0)
    vqvae = VQVAE(VQVAEConfig(downsample=8, hidden=32, embedding_dim=32, num_embeddings=64))
    x = torch.rand(2, 1, 64, 64)  # smaller for speed
    opt = torch.optim.AdamW(vqvae.parameters(), lr=1e-3)
    losses = []
    for _ in range(15):
        _, _, ld = vqvae(x)
        opt.zero_grad()
        ld["total"].backward()
        opt.step()
        losses.append(ld["total"].item())
    assert losses[-1] < losses[0] * 0.7, f"loss did not decrease: first={losses[0]:.4f} last={losses[-1]:.4f}"


def test_indices_round_trip() -> None:
    """encode_indices -> decode_indices should be deterministic."""
    torch.manual_seed(0)
    vqvae = VQVAE(VQVAEConfig(downsample=4, hidden=16, embedding_dim=16, num_embeddings=32))
    x = torch.rand(1, 1, 32, 32)
    idx_a = vqvae.encode_indices(x)
    idx_b = vqvae.encode_indices(x)
    assert torch.equal(idx_a, idx_b), "encode_indices must be deterministic for fixed weights"
    recon = vqvae.decode_indices(idx_a)
    assert recon.shape == x.shape


def test_quantizer_straight_through_gradient() -> None:
    """Straight-through estimator: gradient of quantiser output wrt encoder
    output should be identity (i.e., grad just flows back through z_e + sg(c-z_e))."""
    torch.manual_seed(0)
    q = VectorQuantizer(num_embeddings=8, embedding_dim=4, commitment_weight=0.25)
    z_e = torch.randn(1, 4, 2, 2, requires_grad=True)
    z_q, _, _ = q(z_e)
    # Sum should backprop directly through z_e thanks to the STE
    z_q.sum().backward()
    assert z_e.grad is not None
    expected = torch.ones_like(z_e)
    # All-ones because grad of sum wrt input is ones for an identity STE pass
    assert torch.allclose(z_e.grad, expected, atol=1e-6), z_e.grad
