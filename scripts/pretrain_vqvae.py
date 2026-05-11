#!/usr/bin/env python
"""Per-head VQ-VAE codebook pretraining — skeleton.

Usage:
    python scripts/pretrain_vqvae.py --head depth  --config configs/phase1_qdepth_repro.yaml
    python scripts/pretrain_vqvae.py --head normal --config configs/phase2_geo3head.yaml
    python scripts/pretrain_vqvae.py --head plane  --config configs/phase2_geo3head.yaml

Trained on Robosuite-extracted GT from LIBERO-90 plus pseudo-labels for OXE.
Body filled in CODING stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head", choices=["depth", "normal", "plane"], required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    print(f"[pretrain_vqvae] head={args.head} cfg={args.config.name}")
    print("[pretrain_vqvae] body not yet implemented — see CODING stage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
