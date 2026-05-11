# Phase 1 — substrate status (1.1 / 1.2 / 1.3 / 1.4 / 1.5 done)

**Date:** 2026-05-11
**Workspace:** `~/.nanoresearch/workspace/research/20260511121847e9b8/`
**Verdict:** ✅ All Phase-1 substrate components landed; integration (1.6) + training (1.7) + eval (1.8) still pending.

This is the second Phase-1 milestone — the [first one](phase0_status.md) was infra. Now the **substrate-independent core** (VQ-VAE, depth expert, QDepth losses) plus the **substrate-touching pieces** (open-pi-zero submodule wrapper, Robosuite depth-GT extractor) all have landing code, passing tests, and one verified end-to-end smoke artefact each.

## Sub-task status

| ID | Description | State | Files / tests |
|---|---|---|---|
| **1.1** | VQ-VAE codebook (depth K=256, dim=160, 16×16 latent) | ✅ done | `src/georel_vla/codebooks/vqvae.py` 180L · 7 tests |
| **1.2** | Depth Expert (transformer 18/8/1024/4096 + shallow CNN decoder + L2-similarity to codebook) | ✅ done | `src/georel_vla/experts/depth_expert.py` 150L · 6 tests |
| **1.5** | QDepth losses + hybrid attention mask (text/image self · depth → text+image · action → all preceding causal) | ✅ done | `src/georel_vla/losses/qdepth.py` 170L · 9 tests |
| **1.3** | open-pi-zero substrate (allenzren/open-pi-zero @ `c3df7fb`, 0.1.1-1) | ✅ pinned + wrapper interface declared | `third_party/open-pi-zero/` (submodule) · `src/georel_vla/backbones/pi0.py` 150L · 5 tests |
| **1.4** | Robosuite LIBERO depth-GT extractor (replays demos, queries MuJoCo depth via `get_real_depth_map`, saves `.npz`) | ✅ done + smoke verified | `src/georel_vla/data/libero_geom.py` 290L · `scripts/extract_libero_depth_gt.py` 115L · 4 tests |

37 unit tests total across both phases of Phase-1, all green on remote `pssa-vla` env.

## End-to-end evidence

### `extract_libero_depth_gt.py` smoke (LIBERO-Spatial task 0 demo 0)

```
[extract] suite=libero_spatial task_ids=0
[extract] out  = /autodl-fs/data/svla/data/libero_depth_gt_smoke
[extract] running 1 task(s): [0]
wrote task00_demo_0.npz frames=98 rgb=19.3MB depth=12.8MB
[extract] DONE — 1 npz file(s)

rgb   : (98, 256, 256, 3) uint8
depth : (98, 256, 256)    float16   min/max/mean: 0.621 / 3.068 / 1.366  m
        percentiles p1/p25/p50/p75/p99: 0.688 / 0.811 / 0.933 / 1.405 / 3.045 m
action: (98, 7) float32
meta  : depth_units=meters, depth_clip_m=5.0, extractor_version=2
size  : 9.4 MB compressed (~3.4× compression vs raw)
```

The depth percentile spread (0.69 → 3.05 m) is what the VQ-VAE codebook will train against — it's the actual scene depth, not the OpenGL z-buffer cluster (which would have been [0.984, 0.997] and would have collapsed the codebook).

### Bug log (caught by smoke, fixed before tagging)

Three bugs surfaced during the smoke runs and were fixed in successive small commits — listing for traceability:

1. **`AttributeError: 'Task' object has no attribute 'problem_file_name'`** — LIBERO Task namedtuple uses `.name`, `.bddl_file`, `.problem_folder`. Fix: use the real attribute names + prefer `libero.libero.get_libero_path("bddl_files")` resolver. (commit `6e2e35b`)

2. **OpenGL z-buffer collapse in [0.98, 0.997]** — Robosuite `<camera>_depth` is normalised non-linear z-buffer dominated by the far plane; useless as VQ-VAE training target. Fix: convert via `robosuite.utils.camera_utils.get_real_depth_map(sim, depth)`; clip to [0, 5 m] for fp16 storage; bumped `extractor_version` 1→2. (commit `0cbc34a`)

3. **`AttributeError: 'MjSim' object has no attribute 'model'`** — LIBERO/Robosuite re-instantiate the sim on each `env.reset()`, so a sim handle captured before reset becomes stale. Fix: take `sim = env.sim` AFTER reset+set_init_state, not before. (commit `b385881`)

All three were caught by the loop "push → run smoke → see real failure → fix" and never required re-architecting.

## Cost so far

≈ 1 GPU-h cumulative on A800 (model loading + LIBERO env init + ~3 smoke runs of ~1-3 min each) ≈ ¥7. Total Phase-0 + Phase-1-substrate spend ≈ ¥11, well within the Phase-0 + Phase-1 budgets (¥14 + ¥112).

## What's left in Phase 1

Three sub-tasks remain to complete the QDepth-VLA reproduction loop:

* **1.6 model.py wiring** — the big integration: load PaliGemma-3B + SigLIP + open-pi-zero action expert; route through hybrid attention mask; co-train depth expert against frozen VQ-VAE indices. ≈ 400-600 LoC. This is where the `Pi0Backbone.load()` and `encode_image_to_siglip()` stubs in `backbones/pi0.py` get filled in.
* **1.7 training scripts** — `scripts/pretrain_vqvae.py` (≈ 6 GPU-h, ≈ ¥40) and `scripts/train.py` (QDepth-VLA recipe: LIBERO-90 pretrain 20 epochs + LIBERO-Spatial finetune 50 epochs, ≈ 16 GPU-h, ≈ ¥112).
* **1.8 eval glue** — `scripts/eval_libero.py` body, reusing PSSA's verified `run_libero_eval.py` skeleton + adapter for our depth-aware `Pi0Backbone`. Phase-1 gate: LIBERO-Spatial avg SR ≥ 84.0 over 10 tasks × 50 rollouts.

Plus one preparatory data step: full LIBERO-90 sweep of `extract_libero_depth_gt.py` to produce VQ-VAE training data (~3 hrs wallclock for LIBERO-Spatial alone, ~30 hrs for full LIBERO-90; Phase-1 can start with just LIBERO-Spatial as a proof point).

## Provenance

Generated 2026-05-11 by Claude Code (Opus 4.7) in HOST mode, NanoResearch workspace `20260511121847e9b8`. Smoke artifact lives at `/autodl-fs/data/svla/data/libero_depth_gt_smoke/libero_spatial/task00_demo_0.npz` (not committed to git — too large; numbers above are the canonical evidence).
