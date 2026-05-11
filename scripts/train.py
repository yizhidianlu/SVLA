#!/usr/bin/env python
"""Main training entry point — skeleton.

Usage:
    python scripts/train.py --config configs/phase1_qdepth_repro.yaml

Body filled in CODING stage. Phase 1 = reproduce QDepth-VLA (depth head only).
Phase 2 = add normal + plane + L_xc. Phase 3 = add support + L_dr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--variant", type=str, default=None,
                   help="Ablation variant id from configs/*.yaml `ablations.variants`")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("inherits", None)
    if base is not None:
        with open(path.parent / base, encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        merged = _deep_merge(base_cfg, cfg)
        return merged
    return cfg


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    print(f"[train] loaded config: {args.config.name}")
    print(f"[train] backbone={cfg['backbone']['name']}")
    print(f"[train] heads enabled: "
          f"{[k for k, v in cfg['heads'].items() if v.get('enabled')]}")
    print("[train] body not yet implemented — see CODING stage (Phase 1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
