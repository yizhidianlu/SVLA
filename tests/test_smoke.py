"""Smoke tests — import + config load. Real model tests land with CODING stage."""

from pathlib import Path

import yaml


def test_package_imports() -> None:
    import georel_vla
    assert georel_vla.__version__ == "0.0.1"


def test_subpackages_import() -> None:
    from georel_vla import codebooks, data, experts, losses, model  # noqa: F401


def test_model_config_constructs() -> None:
    """Phase 1.6: GeoRelVLA needs explicit backbone+expert+codebook.
    `.from_config()` builds a stub-backbone version for CI/local smoke."""
    import pytest
    pytest.importorskip("torch")
    from georel_vla.model import GeoRelVLA, GeoRelVLAConfig
    cfg = GeoRelVLAConfig()
    obj = GeoRelVLA.from_config(cfg)
    assert obj.cfg.backbone == "open-pi-0"
    assert "depth" in repr(obj)


def test_phase_configs_parse() -> None:
    cfg_dir = Path(__file__).resolve().parent.parent / "configs"
    for name in [
        "default.yaml",
        "phase1_qdepth_repro.yaml",
        "phase2_geo3head.yaml",
        "phase3_geo_sup.yaml",
    ]:
        with open(cfg_dir / name, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict), f"{name} did not parse to a dict"
        if name == "default.yaml":
            assert cfg["backbone"]["name"] == "open-pi-0"
        else:
            # phase configs should declare inherits
            assert cfg.get("inherits") == "default.yaml", f"{name} missing inherits"


def test_decisions_locked_match_blueprint() -> None:
    """Catch drift between docs/experiment_blueprint.json and configs/."""
    bp = Path(__file__).resolve().parent.parent / "docs" / "experiment_blueprint.json"
    import json
    with open(bp, encoding="utf-8") as f:
        data = json.load(f)
    assert data["candidate_name"] == "GeoRel-VLA"
    assert data["head_count"] == 4
    expected_heads = {"depth", "normal", "plane", "support"}
    decision = next(d for d in data["decisions_locked"] if d["item"] == "head_cardinality")
    assert set(decision["heads"]) == expected_heads
