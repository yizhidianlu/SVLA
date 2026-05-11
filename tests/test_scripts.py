"""Lightweight tests for scripts/{pretrain_vqvae,train,eval_libero}.py.

We test:
* argparse setup (--help works without heavy deps via the lazy-import pattern)
* dataset streaming (a tiny synthetic .npz shard end-to-end)

Real model + LIBERO env exercise lives in remote runs, not unit tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _import_script(name: str):
    """Import scripts/<name>.py as a module (without invoking main())."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        return __import__(name)
    finally:
        # leave on sys.path; tests are short-lived
        pass


# ----- pretrain_vqvae ----------------------------------------------------


def test_pretrain_vqvae_help_does_not_import_torch(monkeypatch) -> None:
    """--help must work even if torch isn't installed (lazy import)."""
    monkeypatch.setattr(sys, "argv", ["pretrain_vqvae.py", "--help"])
    mod = _import_script("pretrain_vqvae")
    with pytest.raises(SystemExit) as ei:
        mod.main()
    assert ei.value.code == 0


def test_pretrain_vqvae_dataset_streams_synthetic_npz(tmp_path) -> None:
    """_DepthShardDataset reads .npz shards and yields per-frame depth arrays."""
    import numpy as np
    rng = np.random.default_rng(0)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for i in range(2):
        depth = rng.uniform(0, 5, size=(4, 16, 16)).astype(np.float16)
        rgb = rng.integers(0, 256, size=(4, 16, 16, 3), dtype=np.uint8)
        np.savez_compressed(shard_dir / f"task{i:02d}_demo.npz",
                            rgb=rgb, depth=depth, action=np.zeros((4, 7), np.float32),
                            meta="{}")

    mod = _import_script("pretrain_vqvae")
    ds = mod._DepthShardDataset(shard_dir, depth_clip_m=5.0, shuffle_shards=False)
    assert ds.shard_count() == 2
    frames = list(ds.stream_frames())
    assert len(frames) == 8                    # 2 shards × 4 frames
    assert all(f.shape == (1, 16, 16) for f in frames)
    batches = list(ds.yield_batches(batch_size=3))
    assert sum(b.shape[0] for b in batches) == 8


def test_pretrain_vqvae_missing_dir_raises(tmp_path) -> None:
    mod = _import_script("pretrain_vqvae")
    with pytest.raises(FileNotFoundError):
        mod._DepthShardDataset(tmp_path / "does-not-exist")


# ----- train --------------------------------------------------------------


def test_train_help(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--help"])
    mod = _import_script("train")
    with pytest.raises(SystemExit) as ei:
        mod.main()
    assert ei.value.code == 0


def test_train_libero_shard_dataset(tmp_path) -> None:
    """_LiberoShardDataset yields (rgb, depth) frame pairs."""
    import numpy as np
    rng = np.random.default_rng(0)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for i in range(2):
        depth = rng.uniform(0, 5, size=(3, 16, 16)).astype(np.float16)
        rgb = rng.integers(0, 256, size=(3, 16, 16, 3), dtype=np.uint8)
        np.savez_compressed(shard_dir / f"task{i:02d}_demo.npz",
                            rgb=rgb, depth=depth, action=np.zeros((3, 7), np.float32),
                            meta="{}")
    mod = _import_script("train")
    ds = mod._LiberoShardDataset(shard_dir, shuffle_shards=False)
    pairs = list(ds.stream())
    assert len(pairs) == 6                     # 2 × 3
    assert pairs[0][0].dtype == np.uint8        # rgb
    assert pairs[0][1].dtype == np.float32      # depth (cast in stream)
    batches = list(ds.batches(batch_size=4))
    # 6 frames in batches of 4 -> [4, 2]
    assert [b[0].shape[0] for b in batches] == [4, 2]


# ----- eval_libero -------------------------------------------------------


def test_eval_libero_help(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["eval_libero.py", "--help"])
    mod = _import_script("eval_libero")
    with pytest.raises(SystemExit) as ei:
        mod.main()
    assert ei.value.code == 0


def test_eval_libero_image_fix_round_trip() -> None:
    """180° rotate twice == identity."""
    import numpy as np
    mod = _import_script("eval_libero")
    rgb = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    rgb2 = mod._libero_image_fix(mod._libero_image_fix(rgb))
    assert np.array_equal(rgb, rgb2)


def test_eval_libero_action_fix_gripper() -> None:
    """Gripper action [0,1] -> sign({-1, +1}) flipped."""
    import numpy as np
    mod = _import_script("eval_libero")
    open_ = np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32)   # open
    close = np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)   # close
    assert mod._libero_action_fix(open_)[-1] == 1.0    # OpenVLA close (1) -> LIBERO open (1) flipped
    assert mod._libero_action_fix(close)[-1] == -1.0   # OpenVLA open (0) -> LIBERO close (-1) flipped


def test_eval_libero_unknown_policy_raises() -> None:
    mod = _import_script("eval_libero")
    with pytest.raises(ValueError, match="unknown --policy"):
        mod._build_policy("does-not-exist", "model", "key")


def test_eval_libero_georel_policy_requires_torch_and_paligemma() -> None:
    """After Phase 1.7c.2, georel policy is real — but it lazily imports torch +
    georel_vla.backbones.pi0 + open-pi-zero, all of which are absent in the CI
    Ubuntu runner. The build call should fail at the lazy import with a
    ModuleNotFoundError pointing at one of those, not silently succeed."""
    mod = _import_script("eval_libero")
    with pytest.raises((ModuleNotFoundError, ImportError, FileNotFoundError, RuntimeError)):
        mod._build_policy("georel", "model", "key")
