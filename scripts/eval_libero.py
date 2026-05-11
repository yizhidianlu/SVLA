#!/usr/bin/env python
"""LIBERO evaluation — skeleton.

Usage:
    python scripts/eval_libero.py --config configs/phase1_qdepth_repro.yaml --suite spatial
    python scripts/eval_libero.py --config configs/phase3_geo_sup.yaml      --suite long

Implements PSSA workspace's two documented fixes (--libero-action-fix gripper sign+flip,
--libero-image-fix 180-deg rotate) to match OpenVLA-7B-finetuned-libero training distribution.
Body filled in CODING stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--suite", choices=["spatial", "object", "goal", "long"], required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--rollouts-per-task", type=int, default=50)
    p.add_argument("--libero-action-fix", action="store_true", default=True,
                   help="Apply gripper [0,1]->sign({-1,+1}) + flip sign (PSSA-documented).")
    p.add_argument("--libero-image-fix", action="store_true", default=True,
                   help="Apply 180-deg rotate to RGB to match OpenVLA-finetune training (PSSA-documented).")
    args = p.parse_args()
    print(f"[eval_libero] cfg={args.config.name} suite={args.suite} "
          f"rollouts/task={args.rollouts_per_task}")
    print("[eval_libero] body not yet implemented — see CODING stage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
