"""Unit tests for src/georel_vla/experts/depth_expert.py."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from georel_vla.experts.depth_expert import DepthExpert, DepthExpertConfig  # noqa: E402


def _small_cfg() -> DepthExpertConfig:
    """Smaller config used by all tests for speed (real Phase-1 uses defaults)."""
    return DepthExpertConfig(
        siglip_dim=64, n_img_tokens=16, latent_h=4, latent_w=4,
        embedding_dim=8, num_embeddings=32, hidden=64, intermediate=128,
        n_layers=2, n_heads=4,
    )


def test_qdepth_default_sizing() -> None:
    """QDepth-VLA Table 1: 18 layers, 8 heads, hidden 1024, intermediate 4096."""
    cfg = DepthExpertConfig()
    assert cfg.n_layers == 18
    assert cfg.n_heads == 8
    assert cfg.hidden == 1024
    assert cfg.intermediate == 4096
    assert cfg.embedding_dim == 160
    assert cfg.num_embeddings == 256
    assert cfg.n_img_tokens == 256
    assert cfg.latent_h == 16 and cfg.latent_w == 16


def test_forward_shapes() -> None:
    cfg = _small_cfg()
    expert = DepthExpert(cfg)
    siglip = torch.randn(2, cfg.n_img_tokens, cfg.siglip_dim)
    codebook = torch.randn(cfg.num_embeddings, cfg.embedding_dim)
    logits, embed = expert(siglip, codebook)
    assert logits.shape == (2, cfg.latent_h * cfg.latent_w, cfg.num_embeddings)
    assert embed.shape == (2, cfg.embedding_dim, cfg.latent_h, cfg.latent_w)


def test_argmax_aligns_with_nearest_codebook_entry() -> None:
    """If the projected vector is exactly one of the codebook entries, the CE
    target argmax should pick that index out — verifying the negative-distance
    similarity scoring."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    expert = DepthExpert(cfg)
    expert.eval()
    siglip = torch.randn(1, cfg.n_img_tokens, cfg.siglip_dim)
    codebook = torch.randn(cfg.num_embeddings, cfg.embedding_dim)

    with torch.no_grad():
        _, embed = expert(siglip, codebook)
        # For each spatial position, manually find nearest code:
        flat = embed.permute(0, 2, 3, 1).reshape(1, -1, cfg.embedding_dim)
        d = ((flat.unsqueeze(2) - codebook.unsqueeze(0).unsqueeze(0)) ** 2).sum(-1)
        nearest = d.argmin(-1)
        # And the expert's own argmax (= argmin distance):
        logits, _ = expert(siglip, codebook)
        argmax_logits = logits.argmax(-1)
    assert torch.equal(nearest, argmax_logits)


def test_codebook_size_mismatch_raises() -> None:
    cfg = _small_cfg()
    expert = DepthExpert(cfg)
    siglip = torch.randn(1, cfg.n_img_tokens, cfg.siglip_dim)
    bad_codebook = torch.randn(cfg.num_embeddings + 1, cfg.embedding_dim)
    with pytest.raises(AssertionError):
        expert(siglip, bad_codebook)


def test_input_token_count_mismatch_raises() -> None:
    cfg = _small_cfg()
    expert = DepthExpert(cfg)
    bad_siglip = torch.randn(1, cfg.n_img_tokens + 1, cfg.siglip_dim)
    codebook = torch.randn(cfg.num_embeddings, cfg.embedding_dim)
    with pytest.raises(AssertionError):
        expert(bad_siglip, codebook)


def test_loss_drops_during_overfit() -> None:
    """Sanity: when training to predict a fixed target, the depth-CE loss falls."""
    from georel_vla.losses.qdepth import depth_ce_loss

    torch.manual_seed(0)
    cfg = _small_cfg()
    expert = DepthExpert(cfg)
    siglip = torch.randn(2, cfg.n_img_tokens, cfg.siglip_dim)
    codebook = torch.randn(cfg.num_embeddings, cfg.embedding_dim)
    target = torch.randint(0, cfg.num_embeddings, (2, cfg.latent_h * cfg.latent_w))
    opt = torch.optim.AdamW(expert.parameters(), lr=1e-3)
    losses = []
    for _ in range(20):
        logits, _ = expert(siglip, codebook)
        loss = depth_ce_loss(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, f"loss did not drop: {losses[0]:.4f} -> {losses[-1]:.4f}"
