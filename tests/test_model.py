"""Unit tests for src/georel_vla/model.py — the GeoRelVLA integration.

All tests use StubBackbone (tiny CNN) so they run without open-pi-zero
weights. Real PaliGemma/Pi0 forward is exercised in Phase 1.7.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from georel_vla.backbones.base import VLABackbone  # noqa: E402
from georel_vla.backbones.stub import StubBackbone, StubBackboneConfig  # noqa: E402
from georel_vla.codebooks.vqvae import VQVAE, VQVAEConfig  # noqa: E402
from georel_vla.experts.depth_expert import DepthExpert, DepthExpertConfig  # noqa: E402
from georel_vla.model import GeoRelVLA, GeoRelVLAConfig  # noqa: E402

# ----- factory -----


def test_from_config_builds_with_stub() -> None:
    """Default `.from_config()` should give a self-contained model."""
    m = GeoRelVLA.from_config()
    assert isinstance(m.backbone, VLABackbone)
    assert m.depth_codebook.shape == (256, 160)
    # codebook is a buffer, NOT a parameter
    param_names = {n for n, _ in m.named_parameters()}
    assert all("depth_codebook" not in n for n in param_names)


def test_from_config_custom_cfg_propagates() -> None:
    cfg = GeoRelVLAConfig(depth_codebook_size=128, depth_embedding_dim=64)
    m = GeoRelVLA.from_config(cfg)
    assert m.depth_codebook.shape == (128, 64)
    assert m.depth_expert.cfg.num_embeddings == 128
    assert m.depth_expert.cfg.embedding_dim == 64


# ----- construction guards -----


def _small_components() -> tuple[StubBackbone, DepthExpert, torch.Tensor, GeoRelVLAConfig]:
    """Tiny set used by most unit tests (faster than 256-token defaults)."""
    cfg = GeoRelVLAConfig(
        depth_codebook_size=32, depth_embedding_dim=8,
        depth_latent_h=4, depth_latent_w=4,
    )
    backbone = StubBackbone(StubBackboneConfig(
        siglip_dim=64, n_image_tokens=16, image_size=32,
        hidden=16, n_conv_stages=2,
    ))
    expert = DepthExpert(DepthExpertConfig(
        siglip_dim=64, n_img_tokens=16, latent_h=4, latent_w=4,
        embedding_dim=8, num_embeddings=32, hidden=32, intermediate=64,
        n_layers=2, n_heads=4,
    ))
    codebook = torch.randn(32, 8)
    return backbone, expert, codebook, cfg


def test_constructor_validates_codebook_shape() -> None:
    backbone, expert, _, cfg = _small_components()
    bad_codebook = torch.randn(33, 8)  # wrong K
    with pytest.raises(ValueError, match="depth_codebook shape"):
        GeoRelVLA(cfg, backbone, expert, bad_codebook)


def test_constructor_validates_expert_codebook_match() -> None:
    backbone = StubBackbone(StubBackboneConfig(
        siglip_dim=64, n_image_tokens=16, image_size=32, hidden=16, n_conv_stages=2,
    ))
    cfg = GeoRelVLAConfig(depth_codebook_size=32, depth_embedding_dim=8,
                          depth_latent_h=4, depth_latent_w=4)
    expert = DepthExpert(DepthExpertConfig(
        siglip_dim=64, n_img_tokens=16, latent_h=4, latent_w=4,
        embedding_dim=8, num_embeddings=64,  # mismatch with cfg.depth_codebook_size=32
        hidden=32, intermediate=64, n_layers=2, n_heads=4,
    ))
    codebook = torch.randn(32, 8)
    with pytest.raises(ValueError, match="num_embeddings"):
        GeoRelVLA(cfg, backbone, expert, codebook)


def test_constructor_rejects_non_backbone() -> None:
    backbone, expert, codebook, cfg = _small_components()
    with pytest.raises(TypeError):
        GeoRelVLA(cfg, object(), expert, codebook)  # type: ignore[arg-type]


def test_constructor_validates_siglip_dim_match() -> None:
    cfg = GeoRelVLAConfig(depth_codebook_size=32, depth_embedding_dim=8,
                          depth_latent_h=4, depth_latent_w=4)
    backbone = StubBackbone(StubBackboneConfig(
        siglip_dim=64, n_image_tokens=16, image_size=32, hidden=16, n_conv_stages=2,
    ))
    expert = DepthExpert(DepthExpertConfig(
        siglip_dim=128,        # MISMATCH against backbone.siglip_dim=64
        n_img_tokens=16, latent_h=4, latent_w=4,
        embedding_dim=8, num_embeddings=32,
        hidden=32, intermediate=64, n_layers=2, n_heads=4,
    ))
    codebook = torch.randn(32, 8)
    with pytest.raises(ValueError, match="siglip_dim"):
        GeoRelVLA(cfg, backbone, expert, codebook)


# ----- forward -----


def test_forward_shapes() -> None:
    backbone, expert, codebook, cfg = _small_components()
    m = GeoRelVLA(cfg, backbone, expert, codebook)
    rgb = torch.randn(2, 3, 32, 32)
    out = m(rgb)
    assert out["siglip"].shape == (2, 16, 64)
    assert out["depth_logits"].shape == (2, 16, 32)            # (B, latent_h*latent_w, K)
    assert out["depth_embed"].shape == (2, 8, 4, 4)
    assert out["action"] is None
    assert out["losses"] == {}


def test_forward_with_target_returns_loss() -> None:
    backbone, expert, codebook, cfg = _small_components()
    m = GeoRelVLA(cfg, backbone, expert, codebook)
    rgb = torch.randn(2, 3, 32, 32)
    target = torch.randint(0, 32, (2, 16))
    out = m(rgb, depth_target_indices=target)
    losses = out["losses"]
    assert "depth" in losses and losses["depth"].ndim == 0
    assert "total" in losses
    # total = lambda_d * depth (action absent -> total == lambda_d * depth)
    assert torch.isclose(losses["total"], losses["depth"] * cfg.lambda_depth, atol=1e-5)


# ----- loss schedule -----


def test_compute_losses_lambda_decay() -> None:
    backbone, expert, codebook, cfg = _small_components()
    m = GeoRelVLA(cfg, backbone, expert, codebook)
    logits = torch.randn(2, 16, 32)
    target = torch.randint(0, 32, (2, 16))
    l0 = m.compute_losses(logits, target, step=0, gamma=0.999)
    l100 = m.compute_losses(logits, target, step=100, gamma=0.999)
    # lambda decays with step -> total at step 100 < total at step 0 for the same loss value
    assert l100["total"].item() < l0["total"].item()
    # depth (unweighted) is unchanged
    assert torch.isclose(l0["depth"], l100["depth"])


def test_compute_losses_with_action() -> None:
    backbone, expert, codebook, cfg = _small_components()
    m = GeoRelVLA(cfg, backbone, expert, codebook)
    logits = torch.randn(2, 16, 32)
    target = torch.randint(0, 32, (2, 16))
    a_pred = torch.zeros(2, 4, 7)
    a_target = torch.ones(2, 4, 7)
    losses = m.compute_losses(logits, target, action_pred=a_pred, action_target=a_target)
    assert "action" in losses
    assert losses["action"].item() > 0  # MSE between zeros and ones = 1.0
    # total includes both
    assert losses["total"].item() > losses["depth"].item() * cfg.lambda_depth


# ----- training-loop invariants -----


def test_loss_decreases_during_overfit() -> None:
    """Sanity end-to-end: forward + backward + step on a fixed batch shrinks L_total."""
    torch.manual_seed(0)
    backbone, expert, codebook, cfg = _small_components()
    m = GeoRelVLA(cfg, backbone, expert, codebook)
    rgb = torch.randn(2, 3, 32, 32)
    target = torch.randint(0, 32, (2, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(20):
        out = m(rgb, depth_target_indices=target)
        loss = out["losses"]["total"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, f"loss did not drop: {losses[0]:.4e} -> {losses[-1]:.4e}"


def test_codebook_does_not_get_updated_during_training() -> None:
    """The codebook is a buffer — confirm gradients never touch it."""
    backbone, expert, codebook, cfg = _small_components()
    m = GeoRelVLA(cfg, backbone, expert, codebook)
    snapshot = m.depth_codebook.clone()
    rgb = torch.randn(1, 3, 32, 32)
    target = torch.randint(0, 32, (1, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(5):
        out = m(rgb, depth_target_indices=target)
        opt.zero_grad()
        out["losses"]["total"].backward()
        opt.step()
    assert torch.equal(m.depth_codebook, snapshot), "codebook should be frozen but moved"


def test_freeze_codebook_assertion_passes() -> None:
    """Helper should pass on a fresh model (codebook is a buffer)."""
    m = GeoRelVLA.from_config()
    m.freeze_codebook()


# ----- attention layout -----


def test_attention_layout_default_full_pipeline() -> None:
    backbone, expert, codebook, cfg = _small_components()
    m = GeoRelVLA(cfg, backbone, expert, codebook)
    layout = m.attention_layout(n_text=20)
    assert layout.n_image == 16
    assert layout.n_depth == 16          # 4*4
    assert layout.n_proprio == cfg.n_proprio_tokens
    assert layout.n_action == cfg.action_chunk_size


def test_attention_layout_inference_paths() -> None:
    """At inference we may drop depth and action."""
    backbone, expert, codebook, cfg = _small_components()
    m = GeoRelVLA(cfg, backbone, expert, codebook)
    inf = m.attention_layout(n_text=20, include_depth=False, include_action=False)
    assert inf.n_depth == 0
    assert inf.n_action == 0


def test_build_attention_mask_shape() -> None:
    backbone, expert, codebook, cfg = _small_components()
    m = GeoRelVLA(cfg, backbone, expert, codebook)
    mask = m.build_attention_mask(n_text=10)
    layout = m.attention_layout(n_text=10)
    assert mask.shape == (layout.total, layout.total)


# ----- VQ-VAE / DepthExpert / GeoRelVLA pipeline ----


def test_vqvae_indices_feed_geo_rel_vla_loss() -> None:
    """End-to-end: encode synthetic depth -> VQ-VAE indices -> GeoRelVLA depth-CE loss."""
    torch.manual_seed(0)
    vq = VQVAE(VQVAEConfig(
        in_channels=1, out_channels=1, num_embeddings=32, embedding_dim=8,
        downsample=8, hidden=16,
    ))
    cfg = GeoRelVLAConfig(depth_codebook_size=32, depth_embedding_dim=8,
                          depth_latent_h=4, depth_latent_w=4)
    backbone = StubBackbone(StubBackboneConfig(
        siglip_dim=64, n_image_tokens=16, image_size=32, hidden=16, n_conv_stages=2,
    ))
    expert = DepthExpert(DepthExpertConfig(
        siglip_dim=64, n_img_tokens=16, latent_h=4, latent_w=4,
        embedding_dim=8, num_embeddings=32, hidden=32, intermediate=64,
        n_layers=2, n_heads=4,
    ))
    codebook = vq.quantizer.codebook.weight.detach().clone()
    m = GeoRelVLA(cfg, backbone, expert, codebook)

    rgb = torch.randn(2, 3, 32, 32)
    depth = torch.rand(2, 1, 32, 32)
    indices = vq.encode_indices(depth)             # (B, 4, 4)
    indices_flat = indices.reshape(2, -1)          # (B, 16)
    out = m(rgb, depth_target_indices=indices_flat)
    assert out["losses"]["depth"].item() > 0
    assert out["losses"]["depth"].requires_grad
