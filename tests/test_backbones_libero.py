"""Smoke tests for backbones.pi0 and data.libero_geom.

Real functional behaviour is verified on the AutoDL pssa-vla env:
* `Pi0Backbone.load()` requires the open-pi-zero submodule + its uv-managed
  deps, which we install in Phase 1.6.
* `LiberoDepthExtractor.extract_task()` requires libero / robosuite / mujoco
  / EGL and a GPU; we exercise it via the smoke run launched from
  `scripts/extract_libero_depth_gt.py` after this commit lands.

These tests cover only the lightweight surface: import the modules, build
the config dataclasses, parse CLI helpers, and probe submodule presence.
"""

from __future__ import annotations

from pathlib import Path


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


def test_backbone_load_raises_until_phase1_6() -> None:
    """Phase 1.6 will replace this contract; until then the wrapper signals
    'not yet wired' via NotImplementedError."""
    import pytest as _pytest

    from georel_vla.backbones.pi0 import Pi0Backbone
    with _pytest.raises(NotImplementedError):
        Pi0Backbone().load()


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
