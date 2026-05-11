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
    q = VectorQuantizer(num_embeddings=8, embedding_dim=4, commitment_weight=0.25,
                        use_ema=False)  # gradient-style for this test
    z_e = torch.randn(1, 4, 2, 2, requires_grad=True)
    z_q, _, _ = q(z_e)
    # Sum should backprop directly through z_e thanks to the STE
    z_q.sum().backward()
    assert z_e.grad is not None
    expected = torch.ones_like(z_e)
    # All-ones because grad of sum wrt input is ones for an identity STE pass
    assert torch.allclose(z_e.grad, expected, atol=1e-6), z_e.grad


def test_ema_default_freezes_codebook_param() -> None:
    """EMA path: codebook.weight is updated by hand; gradient must be off."""
    q = VectorQuantizer(num_embeddings=16, embedding_dim=4, use_ema=True)
    assert q.codebook.weight.requires_grad is False
    assert hasattr(q, "cluster_size_ema")
    assert hasattr(q, "dw_ema")
    assert hasattr(q, "ema_step")


def test_ema_kmeans_init_seeds_from_first_batch() -> None:
    """k-means init replaces the uniform-random codebook with samples from the first batch."""
    torch.manual_seed(0)
    q = VectorQuantizer(num_embeddings=8, embedding_dim=4, use_ema=True, kmeans_init=True)
    initial = q.codebook.weight.data.clone()
    z_e = torch.randn(2, 4, 4, 4) * 100.0   # large values so re-init shows up clearly
    q.train()
    q(z_e)
    # After the first training forward, codebook should have moved to the new init.
    assert not torch.allclose(initial, q.codebook.weight.data, atol=1e-2)
    assert q._kmeans_done is True


def test_ema_updates_codebook_each_step() -> None:
    """EMA: cluster_size_ema accumulates, codebook drifts toward batch statistics."""
    torch.manual_seed(0)
    q = VectorQuantizer(num_embeddings=8, embedding_dim=4, use_ema=True,
                        ema_decay=0.5, kmeans_init=False)  # no init seed; pure EMA
    z_e = torch.randn(2, 4, 4, 4)
    q.train()
    cb_before = q.codebook.weight.data.clone()
    for _ in range(3):
        q(z_e)
    assert q.cluster_size_ema.sum().item() > 0
    assert not torch.allclose(cb_before, q.codebook.weight.data, atol=1e-4)


def test_ema_dead_code_reset_revives_codes() -> None:
    """Dead-code reset replaces low-utilisation codes with encoder samples."""
    torch.manual_seed(0)
    q = VectorQuantizer(
        num_embeddings=8, embedding_dim=4,
        use_ema=True, ema_decay=0.5,
        kmeans_init=False, reset_dead_codes_every=1, dead_code_threshold=0.1,
    )
    # Drive only one code: encoder outputs near zero -> nearest code is whichever is closest to 0.
    z_e_concentrated = torch.zeros(2, 4, 4, 4) + 1e-3 * torch.randn(2, 4, 4, 4)
    q.train()
    for _ in range(5):
        q(z_e_concentrated)
    # Some codes will be dead under this skewed distribution; the reset path must
    # have fired (we at least ran 5 EMA steps and reset_every=1 with dead<0.1).
    cb_after_concentrated = q.codebook.weight.data.clone()
    # Now run on broader data — codes that were "revived" via reset should now
    # be different from their pre-reset values too.
    z_e_broad = torch.randn(2, 4, 4, 4) * 5.0
    q(z_e_broad)
    assert not torch.allclose(cb_after_concentrated, q.codebook.weight.data, atol=1e-4)


def test_ema_codes_used_grows_over_steps() -> None:
    """Sanity: with EMA + reset, codebook utilisation should be > 25% after a few epochs."""
    torch.manual_seed(0)
    cfg = VQVAEConfig(
        in_channels=1, out_channels=1,
        downsample=8, hidden=32, embedding_dim=32, num_embeddings=64,
        use_ema=True, ema_decay=0.95, reset_dead_codes_every=10, kmeans_init=True,
    )
    vqvae = VQVAE(cfg)
    x = torch.rand(8, 1, 64, 64) * 5.0
    opt = torch.optim.AdamW(
        [p for p in vqvae.parameters() if p.requires_grad], lr=1e-3,
    )
    used_history = []
    for _ in range(40):
        _, indices, ld = vqvae(x)
        opt.zero_grad()
        ld["total"].backward()
        opt.step()
        used_history.append(int(indices.unique().numel()))
    # With EMA + reset on 8x16 samples per step, we should engage many codes —
    # the L2 path on this same toy gets stuck at 1-3.
    assert used_history[-1] >= 16, f"codes used only {used_history[-1]}/64; EMA failing to spread"
