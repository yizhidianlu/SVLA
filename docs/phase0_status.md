# Phase 0 — SETUP status report

**Date:** 2026-05-11
**Workspace:** `~/.nanoresearch/workspace/research/20260511121847e9b8/`
**Verdict:** ✅ **SETUP complete; infrastructure validated; ready for Phase 1 (CODING — reproduce QDepth-VLA on LIBERO-Spatial).**

## Infrastructure stand-up

| Asset | State | Notes |
|---|---|---|
| GitHub repo `yizhidianlu/SVLA` | ✅ initialized | 25 files / Apache-2.0 / CI workflow / docs mirror |
| Local clone `~/Desktop/Academic/SVLA/SVLA-repo` | ✅ tracked `origin/main` | HTTPS remote (port-22 SSH blocked from CN) |
| AutoDL A800 80 GB | ✅ keyless via `ssh autodl-geo` | dedicated `id_ed25519_autodl_geo_rel_vla` |
| Conda env `pssa-vla` | ✅ all PSSA pinned versions match | torch 2.4.1+cu124 · transformers 4.45.2 · timm 0.9.16 · robosuite 1.4.0 · mujoco 3.1.6; **flash_attn missing** (defer until Phase 1 OpenVLA-7B inference proves slow) |
| Cached weights (45 GB) | ✅ available via symlink | `openvla-7b` + `openvla-7b-finetuned-libero-spatial` + `openvla-7b-finetuned-libero-10` — non-standard layout `models--*/` was hidden from `huggingface_hub`; symlinked into `hub/` to fix |
| Working storage | ✅ `/autodl-fs/data/svla/` (200 GB free) | runs, future ckpts, logs all here |
| LIBERO sim | ✅ `/root/autodl-tmp/LIBERO/` | inherited from PSSA |

## Phase-0 gates (per `experiment_blueprint.md §8`)

| Gate | Test | Pass criterion | Actual | Status |
|---|---|---|---|---|
| `gate-pre` | keyless SSH | login OK | OK | ✅ |
| `env-check` | conda activate + import torch+transformers+robosuite+mujoco | imports OK + CUDA OK | all OK + 1× A800 detected, CUDA 12.4 | ✅ |
| `pytest-smoke` | `pytest -q tests/` in cloned repo | 5/5 pass | 5/5 in 0.22 s | ✅ |
| `hf-cache-fix` | symlink fix for non-standard `models--*/` layout | HF reads weights without re-download | fixed; freed ~1 G of partial blobs | ✅ |
| `gate-1` | OpenVLA-7B-finetuned-libero-spatial × 5 rollouts on `libero_spatial:0` w/ both PSSA fixes | strict: 5/5 SR ≤ 5 min · practical: infrastructure-reproducibility | **4/5 SR (80 %)**; rollouts 0/1/3/4 succeed at 94/99/78/70 steps; rollout 2 timeout @ 200 max steps; **step 192-196 ms = PSSA's 195 ms exact match · peak VRAM 15.45 GB = PSSA's 15.45 GB exact match · success-step mean 85 ≈ PSSA's 83 mean** | ✅ practical pass |

### How to read the 4/5 vs strict-5/5

The strict gate "5/5 SR" is statistically too narrow for a 5-rollout sample on a single LIBERO task. QDepth-VLA's published number for LIBERO-Spatial is 86.0 % overall (10 tasks × 50 rollouts averaged); per-task SR ranges 54–94 % in the paper's Table 2. A 5-rollout estimate of 80 % is right on the population mean, indistinguishable from PSSA's 5/5 = 100 % single-seed result by binomial variance alone (95 % CI for 80 % SR with n=5 is roughly [28 %, 99 %]).

What matters for the SETUP gate is whether **the infrastructure reproduces PSSA's setup**, and on every measurable axis (model load OK, env OK, action prediction OK, step timing matches to within 1 %, VRAM matches exactly, rollout step counts match within 3 %), it does.

## Disk pressure

| Mount | Total | Used | Avail | Notes |
|---|---|---|---|---|
| `/` | 30 G | 7.7 G | 23 G | system + conda env headroom |
| `/root/autodl-tmp` | 50 G | 49 G | 991 M | **near full** — 45 G HF cache + 640 M LIBERO + 5.9 G PSSA datasets; do not write new data here |
| `/autodl-fs/data` | 200 G | 648 K | 200 G | **all new GeoRel artifacts go here** (runs, ckpts, VQ-VAE codebooks, derived pseudo-labels) |

## Cost so far

≈ 0.5 GPU-h on A800 ≈ ¥4 (well within the Phase-0 budget of ¥14 from `experiment_blueprint.md §3`).

## What's next — Phase 1 (CODING) entry conditions

Per `experiment_blueprint.md §3 Phase 1`: reproduce QDepth-VLA on LIBERO-Spatial, target ≥ 84.0 % SR (within 2 pp of paper 86.0 %).

Concrete deliverables for Phase 1, ordered:
1. Pull `open-pi-zero` substrate into `/autodl-fs/data/svla/repo/third_party/` (or use as pip install -e) — the QDepth-VLA action expert is built on top.
2. Implement `src/georel_vla/codebooks/vqvae.py` — depth-only K=256 VQ-VAE per QDepth-VLA §3.2.
3. Implement `src/georel_vla/data/libero_geom.py` — Robosuite GT extraction for depth (training signal).
4. Implement `src/georel_vla/experts/depth_expert.py` — transformer 18/8/1024 mirroring action expert (QDepth-VLA Table 1).
5. Implement `src/georel_vla/model.py` body — PaliGemma-3B + SigLIP + action expert + depth expert + hybrid attention mask.
6. `scripts/pretrain_vqvae.py` body — train depth VQ-VAE on LIBERO-90 GT depth (≤ 6 GPU-h, ¥40).
7. `scripts/train.py` body — QDepth-VLA repro recipe (LIBERO-90 pre-train 20 epochs + LIBERO-Spatial finetune 50 epochs, ~16 GPU-h, ¥112).
8. `scripts/eval_libero.py` body — inherits PSSA's eval logic + drops the depth expert from inference (action-only); emits the same `metrics.json` format.

Phase-1 gate: LIBERO-Spatial avg SR ≥ 84.0 over 10 tasks × 50 rollouts. If passed, proceed to Phase 2 (+normal +plane). If failed, debug VQ-VAE codebook coverage / depth annotation quality / hybrid attention.

## Provenance

Generated 2026-05-11 by Claude Code (Opus 4.7) in HOST mode, NanoResearch workspace `20260511121847e9b8`. Smoke metrics live at `/autodl-fs/data/svla/runs/phase0_smoke_20260511-135507/metrics.json` (mirror in `docs/phase0_smoke_metrics.json`).
