"""Smoke tests for backbones.pi0 and data.libero_geom.

Real functional behaviour is verified on the AutoDL pssa-vla env:
* `Pi0Backbone.load()` requires the open-pi-zero submodule + its uv-managed
  deps, which we install in Phase 1.7.
* `LiberoDepthExtractor.extract_task()` requires libero / robosuite / mujoco
  / EGL and a GPU; we exercise it via the smoke run launched from
  `scripts/extract_libero_depth_gt.py` after this commit lands.

The backbone tests need torch (Phase-1.6 made backbones.* nn.Module
subclasses), so the whole module is gated on torch availability. The
libero_geom-only tests still run via test_libero_geom_no_torch.py if/when
we add it; for now CI without torch will skip everything in this file and
remote pssa-vla covers full coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")


def test_backbone_module_importable() -> None:
    from georel_vla.backbones import pi0
    assert hasattr(pi0, "Pi0Backbone")
    assert hasattr(pi0, "Pi0BackboneConfig")
    assert hasattr(pi0, "is_available")
    assert hasattr(pi0, "add_third_party_to_syspath")


def test_backbone_config_defaults() -> None:
    from georel_vla.backbones.pi0 import Pi0BackboneConfig
    cfg = Pi0BackboneConfig()
    assert cfg.image_size == 224
    assert cfg.dtype == "bf16"
    assert cfg.expose_siglip_features is True


def test_backbone_describe_works_without_load() -> None:
    from georel_vla.backbones.pi0 import Pi0Backbone
    info = Pi0Backbone().describe()
    assert isinstance(info, dict)
    assert "submodule_path" in info
    assert "submodule_available" in info
    assert info["loaded"] is False


def test_backbone_load_signals_missing_submodule_when_absent() -> None:
    """If the submodule isn't initialised (CI checkout w/o --recursive),
    `load()` should raise a clear RuntimeError rather than crashing
    half-instantiated. Phase 1.7a wired the real instantiation path."""
    import pytest as _pytest

    from georel_vla.backbones.pi0 import Pi0Backbone, is_available
    if is_available():
        _pytest.skip("submodule + deps available — covered by test_pi0_load_real")
    with _pytest.raises(RuntimeError, match="submodule not initialised"):
        Pi0Backbone().load()


def test_pi0_load_real_instantiates() -> None:
    """When open-pi-zero is fully installed (remote `geo-rel-vla` env), the
    real `load()` path stands up the model. Skipped elsewhere."""
    import pytest as _pytest

    from georel_vla.backbones.pi0 import Pi0Backbone, Pi0BackboneConfig, is_available
    if not is_available():
        _pytest.skip("open-pi-zero submodule + deps not available")
    try:
        import omegaconf  # noqa: F401
    except ImportError:
        _pytest.skip("omegaconf not installed (open-pi-zero install incomplete)")

    bk = Pi0Backbone(Pi0BackboneConfig(
        device="cpu", dtype="fp32", load_paligemma=False,
    ))
    bk.load()
    info = bk.describe()
    assert info["loaded"] is True
    assert info["n_params"] > 3_000_000_000, f"PiZero too small: {info['n_params']}"


def test_pi0_encode_image_to_siglip_shape() -> None:
    """Real forward through SigLIP — verifies (B, 256, 1152) contract."""
    import pytest as _pytest

    from georel_vla.backbones.pi0 import Pi0Backbone, Pi0BackboneConfig, is_available
    if not is_available():
        _pytest.skip("open-pi-zero submodule + deps not available")
    try:
        import omegaconf  # noqa: F401
    except ImportError:
        _pytest.skip("omegaconf not installed (open-pi-zero install incomplete)")

    import torch
    bk = Pi0Backbone(Pi0BackboneConfig(
        device="cpu", dtype="fp32", load_paligemma=False,
    ))
    bk.load()
    out = bk.encode_image_to_siglip(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 256, 1152), out.shape


def test_submodule_pinned_on_disk() -> None:
    """If the submodule was checked out (clone with --recursive or
    `git submodule update --init`), the upstream pizero.py must be on disk.

    `git submodule add` registers an empty placeholder directory even before
    `update --init` is run, so `dir.exists()` is not enough to detect "not
    initialised yet" — probe the inner file directly.
    """
    import pytest as _pytest

    from georel_vla.backbones.pi0 import THIRD_PARTY_OPEN_PI_ZERO
    pizero = THIRD_PARTY_OPEN_PI_ZERO / "src" / "model" / "vla" / "pizero.py"
    if not pizero.is_file():
        _pytest.skip("open-pi-zero submodule not initialised in this checkout")
    assert pizero.is_file()


def test_libero_geom_importable_without_mujoco() -> None:
    """The module must import cleanly even on machines without libero / robosuite.

    Phase 1.4 design: all heavy deps are imported inside method bodies, so
    the module surface is safe to import for help / linting / config-only use.
    """
    from georel_vla.data import libero_geom
    assert hasattr(libero_geom, "LiberoDepthExtractor")
    assert hasattr(libero_geom, "LiberoExtractorConfig")
    assert hasattr(libero_geom, "LiberoFrame")


def test_libero_extractor_config_defaults() -> None:
    from georel_vla.data.libero_geom import (
        DEFAULT_DEMOS_ROOT,
        DEFAULT_LIBERO_ROOT,
        DEFAULT_OUT_ROOT,
        LiberoExtractorConfig,
    )
    cfg = LiberoExtractorConfig()
    assert cfg.libero_root == DEFAULT_LIBERO_ROOT
    assert cfg.demos_root == DEFAULT_DEMOS_ROOT
    assert cfg.out_root == DEFAULT_OUT_ROOT
    assert cfg.camera == "agentview"
    assert cfg.resolution == 256
    assert cfg.stride == 1
    assert cfg.compress is True


def test_libero_extractor_can_be_constructed_without_libero() -> None:
    """Constructing the extractor only touches env vars; heavy imports come at first call."""
    from georel_vla.data.libero_geom import LiberoDepthExtractor, LiberoExtractorConfig
    ex = LiberoDepthExtractor(LiberoExtractorConfig())
    assert ex.cfg.camera == "agentview"


def test_cli_task_id_parser() -> None:
    """parse_task_ids must accept range and comma specs and clamp to [0, n_tasks)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from extract_libero_depth_gt import parse_task_ids
    assert parse_task_ids("0-9", n_tasks=10) == list(range(10))
    assert parse_task_ids("0,3,5", n_tasks=10) == [0, 3, 5]
    assert parse_task_ids("0-2,7", n_tasks=10) == [0, 1, 2, 7]
    # Out-of-range ids are silently dropped:
    assert parse_task_ids("0-15", n_tasks=10) == list(range(10))
    # None == all
    assert parse_task_ids(None, n_tasks=4) == [0, 1, 2, 3]
