# SVLA — GeoRel-VLA

**Geometry-and-Relation-aware Vision-Language-Action model.**

GeoRel-VLA replaces 2D-region visual grounding in Vision-Language-Action models with three decoupled geometric prediction heads (depth, surface normal, plane segmentation) plus a single derivability-constrained support head, producing a structured spatial schema for occlusion-robust, long-horizon, fine-grained robotic manipulation.

> Status: scaffold (Phase 0 / SETUP). Module bodies are stubs; first end-to-end training enters with Phase 1 (QDepth-VLA reproduction). See [`docs/experiment_blueprint.md`](docs/experiment_blueprint.md) for the full execution plan.

## Why this exists

Current VLA models (OpenVLA, π₀, RT-2 …) ground language in 2D image-region attention; this collapses object position, orientation, and inter-object relations into the same embedding subspace. Recent work has begun to inject 3D awareness — QDepth-VLA (depth-only auxiliary tokens), GST-VLA (Gaussian primitives + DA-CoT), GeoVLA (point-cloud input), MolmoAct (depth as reasoning tokens), Spatial Forcing (implicit VGGT alignment). None of them yet supervise **depth + normal + plane jointly as decoupled heads** *and* emit a **geometry-grounded symbolic support output**. GeoRel-VLA is the combination of the two.

See [`docs/literature_seed.md`](docs/literature_seed.md) for the full competitive landscape and gap analysis.

## Architecture (locked at 4 heads)

```
                  ┌─── Depth Expert      ─→ K_d=256 quantized depth tokens (VQ-VAE)
   SigLIP feats ──┼─── Normal Expert     ─→ K_n=128 quantized normal tokens
                  ├─── Plane Expert      ─→ K_p=64 quantized plane-mask tokens
                  └─── Support Expert    ─→ NxN symbolic support graph
                                          + intermediate contact-patch attention
                                            (shaped by L_dr derivability — no
                                             separate contact supervision)

   open-π₀ Action Expert (CFM head) ←── attends to all four expert outputs
                                        + image + text + proprio
```

Cross-head consistency loss `L_xc` ties depth ↔ normal ↔ plane (depth-normal orthogonality + plane-normal variance + plane-depth RANSAC residual). Derivability loss `L_dr` requires every predicted "A supports B" edge to admit a contact patch on A that lies in a single plane region with B's bottom face directly above it under gravity.

Contact relations therefore **emerge** as the geometric prerequisite the constraint enforces, not as a separately supervised target.

## Layout

```
SVLA/
├── docs/                       # plan & literature mirror from NanoResearch workspace
│   ├── ideation_summary.md     # thesis, gap, contributions, locked decisions
│   ├── experiment_blueprint.md # phased execution plan with go/no-go gates
│   ├── experiment_blueprint.json
│   ├── literature_seed.md      # 30-paper annotated bibliography
│   └── workspace_manifest.json # NanoResearch manifest snapshot
├── src/georel_vla/             # Python package (skeleton — bodies in CODING stage)
│   ├── experts/                # depth, normal, plane, support
│   ├── losses/                 # L_xc, L_dr
│   ├── data/                   # Robosuite GT extraction + OXE pseudo-labels
│   ├── codebooks/              # per-head VQ-VAE
│   └── model.py                # full GeoRel-VLA = open-π₀ + 4 experts
├── configs/                    # phased YAML configs (Phase 1, 2, 3)
├── scripts/                    # train / pretrain VQ-VAE / eval LIBERO / eval Simpler
├── tests/                      # smoke import + 1-step forward
└── .github/workflows/          # CI (lint + import check)
```

## Quickstart (after Phase 0 setup)

```bash
# Inherit the pssa-vla conda env (transformers 4.45.2 / torch 2.4.1+cu124 / robosuite 1.4.0 / mujoco 3.1.6)
conda activate pssa-vla
pip install -e .

# Phase 1 — reproduce QDepth-VLA on LIBERO-Spatial
python scripts/pretrain_vqvae.py --head depth --config configs/phase1_qdepth_repro.yaml
python scripts/train.py            --config configs/phase1_qdepth_repro.yaml
python scripts/eval_libero.py      --config configs/phase1_qdepth_repro.yaml --suite spatial

# Phase 2 — add normal + plane heads (RQ1, RQ2)
python scripts/pretrain_vqvae.py --head normal --config configs/phase2_geo3head.yaml
python scripts/pretrain_vqvae.py --head plane  --config configs/phase2_geo3head.yaml
python scripts/train.py            --config configs/phase2_geo3head.yaml

# Phase 3 — add support head with derivability (RQ3, RQ4)
python scripts/train.py            --config configs/phase3_geo_sup.yaml
```

## Targets (provisional, LIBERO single-view)

| Suite | QDepth-VLA | GeoRel-VLA target | Δ |
|---|---|---|---|
| Spatial | 86.0 | ≥ 87.5 | +1.5 |
| Object | 88.8 | ≥ 89.0 | +0.2 |
| Goal | 94.0 | ≥ 94.0 | 0 |
| **Long** | 72.6 | **≥ 76.0** | **+3.4** |
| **Stack-Block** (Simpler-WidowX) | 39.6 | **≥ 48.0** | **+8.0** |
| **stack-on-yellow** (real Piper) | 10.0 | **≥ 30.0** | **+20.0** |

## Compute envelope

≈ 188 GPU-h / ≈ ¥1,276 / ≈ 19 wallclock days at 2× A800 80 GB on AutoDL. See [`docs/experiment_blueprint.md §3`](docs/experiment_blueprint.md) for the 5-phase budget breakdown and gating.

## Provenance

- Project initiated 2026-05-11 in NanoResearch workspace `20260511121847e9b8` (Claude Code HOST mode for IDEATION + PLANNING).
- Companion workspace `20260509000456c3a8` (PSSA-VLA) explores a different angle on the same VLA-grounding bottleneck (temporal scene-entity tracking). Literature is **not** shared between the two workspaces.

## License

Apache 2.0 — see [LICENSE](LICENSE).
