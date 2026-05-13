#!/usr/bin/env python
"""Offline LIBERO -> (rgb, depth, action) extractor for VQ-VAE pretraining.

Typical Phase-1 invocation (run on AutoDL A800 in `pssa-vla` env):

    python scripts/extract_libero_depth_gt.py \\
        --suite libero_spatial --task-ids 0-9 \\
        --out-dir /autodl-fs/data/svla/data/libero_depth_gt \\
        --stride 2 --max-steps 400

Then for the smaller LIBERO-90 superset (Phase 1.7 VLA pretrain):

    for s in libero_spatial libero_object libero_goal libero_10; do
        python scripts/extract_libero_depth_gt.py --suite $s
    done
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_task_ids(spec: str | None, n_tasks: int) -> list[int]:
    """Accept "0-9", "0,3,5", "0-2,7", or None (== all tasks)."""
    if spec is None:
        return list(range(n_tasks))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(t for t in out if 0 <= t < n_tasks)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="libero_spatial",
                   choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    p.add_argument("--task-ids", default=None,
                   help='Comma- or range-spec, e.g., "0-9" or "0,3,5"; default = all tasks in suite')
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override out root (default = LiberoExtractorConfig.DEFAULT_OUT_ROOT)")
    p.add_argument("--demos-root", type=Path, default=None,
                   help="Override demos root (default = /root/autodl-tmp/datasets)")
    p.add_argument("--libero-root", type=Path, default=None,
                   help="Override LIBERO source tree root (default = /root/autodl-tmp/LIBERO)")
    p.add_argument("--camera", default="agentview")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--max-demos", type=int, default=None)
    p.add_argument("--stride", type=int, default=1,
                   help="Subsample every N-th frame (Phase-1 default 1; bump to 2-4 to fit on disk)")
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _patch_robosuite_deepcopy() -> None:
    """Robosuite's `SingleArm.__init__` deepcopy(controller_config) is the
    bottleneck of LIBERO-90 extraction — by ~demo 11 it slows from 50 ms to
    several minutes per call, with stack samples confirming live execution
    inside copy.deepcopy on a controller_config dict that grows by reference
    each time. Replace the `__init__` to use shallow copy: we never reuse
    the configs across envs, so mutations are scoped to the env's lifetime.
    """
    import copy as _copy
    import robosuite.robots.single_arm as _sa
    _orig = _sa.SingleArm.__init__

    def _fast_init(self, *args, **kwargs):
        _dc = _copy.deepcopy
        _copy.deepcopy = _copy.copy
        try:
            _orig(self, *args, **kwargs)
        finally:
            _copy.deepcopy = _dc

    _sa.SingleArm.__init__ = _fast_init


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Patch robosuite BEFORE any env construction.
    _patch_robosuite_deepcopy()

    # Import the heavy stuff only after argparse so `--help` works without MuJoCo.
    from georel_vla.data.libero_geom import (
        LiberoDepthExtractor,
        LiberoExtractorConfig,
    )

    cfg_kwargs: dict = dict(
        camera=args.camera,
        resolution=args.resolution,
        max_steps_per_demo=args.max_steps,
        max_demos_per_task=args.max_demos,
        stride=args.stride,
        compress=not args.no_compress,
        skip_existing=not args.overwrite,
        seed=args.seed,
    )
    if args.out_dir is not None:
        cfg_kwargs["out_root"] = args.out_dir
    if args.demos_root is not None:
        cfg_kwargs["demos_root"] = args.demos_root
    if args.libero_root is not None:
        cfg_kwargs["libero_root"] = args.libero_root
    cfg = LiberoExtractorConfig(**cfg_kwargs)

    print(f"[extract] suite={args.suite} task_ids={args.task_ids or 'ALL'}")
    print(f"[extract] out  = {cfg.out_root}")
    print(f"[extract] demos= {cfg.demos_root}")
    print(f"[extract] cam={cfg.camera} res={cfg.resolution} stride={cfg.stride} maxsteps={cfg.max_steps_per_demo}")

    # Resolve task ids once we know the benchmark's n_tasks.
    extractor = LiberoDepthExtractor(cfg)
    bench = extractor._load_benchmark(args.suite)  # noqa: SLF001 — script-only convenience
    task_ids = parse_task_ids(args.task_ids, bench.n_tasks)
    print(f"[extract] running {len(task_ids)} task(s): {task_ids}")

    n_written = 0
    for path in extractor.extract_suite(args.suite, task_ids=task_ids):
        n_written += 1
    print(f"[extract] DONE — {n_written} npz file(s) under {cfg.out_root / args.suite}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
