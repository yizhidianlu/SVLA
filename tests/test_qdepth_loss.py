"""Unit tests for src/georel_vla/losses/qdepth.py."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from georel_vla.losses.qdepth import (  # noqa: E402
    TokenLayout,
    attn_mask_for_pytorch_mha,
    build_hybrid_attention_mask,
    depth_ce_loss,
)


def test_depth_ce_loss_perfect_prediction_is_low() -> None:
    target = torch.randint(0, 256, (4, 100))
    # one-hot at target -> minimal loss after softmax
    logits = torch.full((4, 100, 256), -10.0)
    logits.scatter_(2, target.unsqueeze(-1), 10.0)
    loss = depth_ce_loss(logits, target)
    assert loss.item() < 1e-3


def test_depth_ce_loss_uniform_gives_log_k() -> None:
    """Uniform logits across K classes should give CE close to log K."""
    k = 256
    target = torch.zeros(2, 50, dtype=torch.long)
    logits = torch.zeros(2, 50, k)  # uniform
    loss = depth_ce_loss(logits, target)
    expected = torch.log(torch.tensor(float(k))).item()
    assert abs(loss.item() - expected) < 1e-4, f"got {loss.item():.4f} want {expected:.4f}"


def test_depth_ce_loss_shape_validation() -> None:
    with pytest.raises(ValueError):
        depth_ce_loss(torch.zeros(4, 256), torch.zeros(4, 50, dtype=torch.long))
    with pytest.raises(ValueError):
        depth_ce_loss(torch.zeros(4, 50, 256), torch.zeros(4, 49, dtype=torch.long))


def test_layout_total_and_boundaries() -> None:
    layout = TokenLayout(n_text=20, n_image=256, n_depth=256, n_proprio=8, n_action=16)
    assert layout.total == 20 + 256 + 256 + 8 + 16
    sp = layout.boundaries()
    assert sp["text"] == slice(0, 20)
    assert sp["image"] == slice(20, 276)
    assert sp["depth"] == slice(276, 532)
    assert sp["proprio"] == slice(532, 540)
    assert sp["action"] == slice(540, 556)


def test_hybrid_mask_within_modality_blocks() -> None:
    """text|image|depth|proprio|action self-blocks all True; cross-block off-diagonal as spec."""
    layout = TokenLayout(n_text=3, n_image=4, n_depth=2, n_proprio=2, n_action=3)
    m = build_hybrid_attention_mask(layout)
    sp = layout.boundaries()

    # within-modality (excluding action which is causal)
    for name in ("text", "image", "depth", "proprio"):
        s = sp[name]
        assert m[s, s].all(), f"{name} self-attention block not all True"

    # text <-> image: no
    assert not m[sp["text"], sp["image"]].any()
    assert not m[sp["image"], sp["text"]].any()


def test_hybrid_mask_depth_attends_text_and_image() -> None:
    layout = TokenLayout(n_text=3, n_image=4, n_depth=2, n_proprio=0, n_action=0)
    m = build_hybrid_attention_mask(layout)
    sp = layout.boundaries()
    assert m[sp["depth"], sp["text"]].all()
    assert m[sp["depth"], sp["image"]].all()
    # but depth does NOT attend back to nothing else (no proprio/action here)


def test_hybrid_mask_action_attends_all_preceding_and_is_causal_within() -> None:
    layout = TokenLayout(n_text=2, n_image=3, n_depth=2, n_proprio=2, n_action=4)
    m = build_hybrid_attention_mask(layout)
    sp = layout.boundaries()

    # Action attends all preceding
    for k in ("text", "image", "depth", "proprio"):
        assert m[sp["action"], sp[k]].all(), f"action does not attend {k}"

    # Causal within action span
    a = sp["action"]
    action_block = m[a, a]
    expected = torch.tril(torch.ones_like(action_block, dtype=torch.bool))
    assert torch.equal(action_block, expected), action_block


def test_hybrid_mask_depth_does_not_leak_action_or_proprio() -> None:
    layout = TokenLayout(n_text=2, n_image=3, n_depth=2, n_proprio=2, n_action=2)
    m = build_hybrid_attention_mask(layout)
    sp = layout.boundaries()
    assert not m[sp["depth"], sp["proprio"]].any(), "depth must not leak into proprio"
    assert not m[sp["depth"], sp["action"]].any(), "depth must not leak into action (future)"


def test_attn_mask_for_pytorch_mha_inverts_correctly() -> None:
    layout = TokenLayout(n_text=2, n_image=2, n_depth=1, n_proprio=0, n_action=1)
    attend = build_hybrid_attention_mask(layout)
    additive = attn_mask_for_pytorch_mha(attend)
    # 0 where attend, -inf where not
    assert (additive[attend.bool()] == 0).all()
    assert torch.isinf(additive[~attend.bool()]).all()


def test_no_modality_yields_empty_mask() -> None:
    m = build_hybrid_attention_mask(TokenLayout(0, 0))
    assert m.shape == (0, 0)
