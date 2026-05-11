# Experiment Blueprint — GeoRel-VLA

**Workspace:** `20260511121847e9b8`
**Stage:** PLANNING
**Date:** 2026-05-11
**Mode:** Claude Code HOST (Opus 4.7)

> *This blueprint is the executable plan. SETUP / CODING / EXECUTION stages must follow it without inventing scope. Any divergence requires updating this file (and incrementing manifest checkpoint).*

---

## §1 · Research questions

The paper-level claim from `plans/ideation_summary.md` is:

> Decoupled multi-task geometric supervision (depth + normal + plane) + a single derivability-constrained support head (contact emerges as a byproduct) inside a VLA improves manipulation in the regimes where 2D-region grounding most visibly fails.

Decomposed into four falsifiable research questions:

| RQ | Question | How we answer it | Where it appears in the paper |
|---|---|---|---|
| **RQ1** | Does adding **depth + normal + plane** as three decoupled auxiliary heads outperform a single-head depth-only auxiliary supervision (i.e., QDepth-VLA)? | Phase-2 main result on LIBERO single-view; expect the +normal +plane variant to beat the depth-only variant on LIBERO-Spatial and LIBERO-Long by ≥ 1.5 pp each. | §5.1 main table |
| **RQ2** | Does the **cross-head consistency loss** (depth ↔ normal ↔ plane) improve any of the three heads' downstream task contribution, vs three independently trained heads? | Phase-2 ablation: with vs without the consistency term. Expect ≥ 0.5 pp avg gain. | §5.3 ablation |
| **RQ3** | Does an **explicit support head** (Phase 3) provide a measurable additional gain on stacking-/placement-heavy tasks (Simpler Stack-Block, real Piper stack-on-yellow), beyond the geometric heads alone? | Phase-3 main result: +support vs Phase-2 best. Expect ≥ 5 pp gain on Simpler Stack-Block, ≥ 15 pp gain on real Piper stack-on-yellow. | §5.2 specialized eval |
| **RQ4** | Does the **derivability constraint** on the support head improve interpretability and induce a usable implicit contact representation, without sacrificing downstream SR vs an unconstrained support head? | Phase-3 ablation: derivability on / off. Measure (a) support F1 on Robosuite GT, (b) implicit contact-patch quality (extracted from support head's intermediate attention vs Robosuite contact GT, target F1 ≥ 0.85), (c) downstream SR (within ±0.5 pp). | §5.4 ablation + §6 case study |

If RQ1 + RQ3 fail simultaneously the project should be re-scoped to a "geometric supervision is harder than it looks" negative-result note (still publishable as a workshop paper, not as the main story).

---

## §2 · Method overview

### 2.1 Architecture

We follow QDepth-VLA's choice of **open-π₀** as the action backbone (PaliGemma-3B VLM + SigLIP encoder + transformer action expert + Conditional-Flow-Matching action loss). Companion workspace `20260509000456c3a8` already has open-π₀ checkpoints reproduced on LIBERO; we **inherit its environment + version pins** (transformers 4.45.2, timm 0.9.16, robosuite 1.4.0, mujoco 3.1.6) but not its literature.

We extend the QDepth-VLA "depth expert" pattern to **four experts** (cardinality finalized at user sign-off — contact head dropped to avoid module bloat; contact emerges as the geometric prerequisite enforced by the support head's derivability constraint):

```
                  ┌─── Depth Expert      ─→ K_d=256 quantized depth tokens
   SigLIP feats ──┼─── Normal Expert     ─→ K_n=128 quantized normal tokens
                  ├─── Plane Expert      ─→ K_p=64 quantized plane-mask tokens
                  └─── Support Expert    ─→ symbolic support graph (N×N edges)
                                            + intermediate contact-patch attention
                                              (no separate supervision; constrained
                                               via L_dr — see §2.4)

   Action Expert (open-π₀ CFM head) ←── attends to all four expert outputs
                                        + image tokens + text tokens + proprio
```

Each **geometric expert** mirrors QDepth-VLA's depth expert (transformer, 18 layers, 8 heads, 1024 hidden, 4096 intermediate). Each takes the SigLIP visual embeddings **before** language fusion to avoid semantic interference (per QDepth-VLA §3.3). Each predicts quantized tokens from a per-head VQ-VAE codebook pretrained on simulator GT.

The **support expert** is a smaller (6 layers, 8 heads) graph-attention network operating on the union of (a) image-region tokens augmented with the three geometric-head outputs and (b) proprioceptive state. It emits `N×N` adjacency logits over object slots (N ≤ 8 to match LIBERO scene complexity) for the support relation, plus a per-edge intermediate "contact-patch attention" map over the depth-back-projected pixel grid. The contact-patch map is **not** independently supervised — it is the activation that the L_dr derivability constraint reads (§2.4) to verify the predicted support edge is geometrically grounded.

### 2.2 Losses

```
L_total = L_action                      # CFM action loss (π₀ identical)
        + λ_d · L_depth                 # CE over quantized depth codes
        + λ_n · L_normal                # CE over quantized normal codes (cosine-similarity-aware)
        + λ_p · L_plane                 # CE over quantized plane-mask codes
        + λ_s · L_support               # BCE over support adjacency
        + λ_xc · L_cross_consistency    # See 2.3
        + λ_dr · L_derivability         # See 2.4 — also indirectly shapes the support head's contact-patch attention
```

with default weights: `λ_d = λ_n = λ_p = 0.01` (matching QDepth-VLA's depth weight), `λ_s = 0.005`, `λ_xc = 0.002`, `λ_dr = 0.003` (slightly heavier than the original 5-head plan because L_dr now carries the entire contact signal — single source of geometric grounding for the relational head). All decay exponentially at the QDepth-VLA rate (γ = 0.9999 / step).

### 2.3 Cross-head consistency loss `L_xc`

Forces the three geometric heads to be locally coherent:

1. **Depth ↔ normal**: at each pixel, the predicted normal must be approximately orthogonal to the local depth gradient. Implemented as `1 − cos(predicted_normal, depth_gradient_normal)`, averaged over pixels with valid GT.
2. **Plane ↔ normal**: pixels assigned to the same plane mask should have similar predicted normals. Implemented as variance-of-normals within each plane mask.
3. **Plane ↔ depth**: pixels assigned to the same plane mask should fit a plane equation in 3D (RANSAC residual). Implemented as mean RANSAC residual within each mask.

Decoded jointly through the VQ-VAE decoders, then differentiable via straight-through estimator on the codebook lookup.

### 2.4 Derivability loss `L_dr`

Forces the support head to be geometrically grounded in the depth + normal + plane outputs:

1. **Contact-patch existence**: each predicted "A supports B" edge must have non-empty support-head contact-patch attention overlapping (a) A's segmentation in the depth field and (b) B's segmentation in the depth field, with min depth-back-projected distance ≤ threshold (1 cm in sim coords). Loss = ReLU(min_distance − 1 cm) for predicted positive edges.
2. **Plane-aligned support patch**: the contact-patch attention on A's surface must lie predominantly within a single plane-mask region (variance of plane-id within the patch ≤ threshold). Loss = patch-plane-id variance for predicted positive edges.
3. **Gravity alignment**: B's bottom face (lowest 25 % of B's depth-back-projected points) should sit directly above the contact patch in the predicted gravity direction (camera ↓ in LIBERO / Simpler / Piper). Loss = ReLU(misalignment_angle − 15°) + ReLU(vertical_offset − 2 cm).

These three terms together (i) guarantee a contact representation emerges inside the support head's attention without separate supervision, (ii) tie the support output back to all three geometric heads, (iii) make the support output interpretable post-hoc — at inference we can extract the contact patch from the support head's attention map and visualize "what part of A is holding B." All are soft constraints, gradient flows back into the geometric experts.

### 2.5 Training recipe

- Co-training with **per-head weight schedule**: depth weight starts at 0.01 and decays; normal / plane weights start at 0 and ramp up over the first 2 epochs each; support weight starts at 0 and ramps up over epochs 4-8 (curriculum: stable depth → stable multi-geometry → grounded support; the L_dr term naturally needs the geometric heads to already be informative).
- Codebook pretraining: independent VQ-VAE per geometric head, on Robosuite ground-truth maps from LIBERO-90 plus pseudo-labels for OXE (Video-Depth-Anything for depth, Omnidata-V2 for normals, region-growing planar segmentation for planes). Support has no codebook — adjacency logits + attention map are direct outputs.
- Optimization: AdamW, lr = 5e-5 (matches QDepth-VLA), cosine schedule, 200 warmup steps, 2-4 × A800 80 GB FSDP, per-GPU batch 32, gradient accumulation to effective global batch 1024 (matches QDepth-VLA effective batch despite fewer GPUs).
- Action chunk size: 4 (sim) and 16 (real-Piper), matching QDepth-VLA.

### 2.6 Inference

**RGB-only at inference** — the geometric and relational heads are kept active during training to shape the visual encoder, but at inference we either (a) drop them (pure RGB → action) or (b) keep them and expose the relational outputs to the action expert (default — matches our paper claim).

---

## §3 · Phased plan with go/no-go gates

This is the **most important** section. Each phase has a budget and a gate; failing a gate triggers re-planning.

### Phase 0 — Setup (target: 1 day)

- Spin a fresh AutoDL A800 80 GB instance (single GPU sufficient for Phase 0).
- Clone the PSSA workspace's environment recipe (`experiment/env/conda.yml` + `pip-requirements.txt`).
- Set `HF_ENDPOINT=https://hf-mirror.com` and `git config insteadOf https://gh-proxy.com/https://github.com/` per PSSA-VLA's documented network workarounds (huggingface.co + GitHub are throttled / blocked from CN).
- `apt install libegl1 libgl1` on the AutoDL container.
- Reproduce LIBERO smoke test: load OpenVLA-7B-finetuned-libero-spatial, run 5 rollouts on `libero_spatial:0`, expect 5/5 success rate (PSSA workspace verified this on 5/10).
- **Smoke gate**: 5/5 SR within 5 min wall-clock. Fail ⇒ re-check `--libero-action-fix` (gripper sign + flip) and `--libero-image-fix` (180° rotate) per PSSA workspace's documented fixes.

### Phase 1 — Reproduce QDepth-VLA baseline (target: 2 days, ≈ ¥120)

Goal: prove our infra can hit QDepth-VLA's published numbers within ±2 pp on LIBERO-Spatial and LIBERO-Long single-view. **No new method yet — pure reproduction.**

- Pull QDepth-VLA training code (or implement it from arXiv 2510.14836 §3 if not released — should be ~600 LoC on top of open-π₀).
- VQ-VAE pretrain on LIBERO-90 depth maps from Robosuite (K=256, latent 16×16, 1e-5 lr, AdamW, ≤ 6 GPU-h on 1× A800).
- Co-train QDepth-VLA on LIBERO-90 for 20 epochs + finetune on LIBERO-Spatial for 50 epochs (per QDepth-VLA recipe). Compute estimate: 16 GPU-h × ¥7/GPU-h ≈ ¥112.
- Evaluate on LIBERO-Spatial (10 tasks × 50 rollouts).
- **Phase-1 gate**: LIBERO-Spatial ≥ 84.0 (within 2 pp of QDepth-VLA's 86.0). Fail ⇒ debug VQ-VAE codebook coverage, depth annotation quality (ViDA on OXE / Robosuite GT on LIBERO), or hybrid attention mask.

### Phase 2 — Add normal + plane heads (target: 5 days, ≈ ¥400)

Goal: **answer RQ1 and RQ2.**

Sub-phases:

- **Phase 2a (≈ ¥80)**: pretrain normal-VQ-VAE (K=128) on Robosuite-extracted normals + Omnidata-V2 pseudo-labels on OXE. Pretrain plane-VQ-VAE (K=64) on Robosuite-extracted plane masks + region-growing pseudo-labels on OXE.
- **Phase 2b (≈ ¥150)**: train the +normal +plane variant on LIBERO-90 + finetune on LIBERO-Spatial / Object / Goal / Long (4 suites). Compare to Phase-1 QDepth-VLA reproduction on the same 4 suites.
- **Phase 2c (≈ ¥120)**: ablation runs — turn off `L_xc` (cross-head consistency), turn off normal head, turn off plane head, turn off both. 4 ablation runs × LIBERO-Spatial × 50 rollouts.
- **Phase 2d (≈ ¥50)**: evaluate Phase-2 best variant on Simpler-WidowX (Stack-Block) to anchor RQ3.

- **Phase-2 gate (RQ1)**: +normal +plane variant beats Phase-1 reproduction on **at least 2 of 4 LIBERO suites** by ≥ 1.5 pp. Fail ⇒ pivot — drop relational claim and rewrite as multi-task geometric supervision paper (still a contribution but narrower).
- **Phase-2 gate (RQ2)**: removing `L_xc` hurts avg LIBERO SR by ≥ 0.5 pp. Fail ⇒ accept consistency loss as not-load-bearing in §5.3 and discuss why.

### Phase 3 — Add support head with derivability constraint (target: 6 days, ≈ ¥500)

Goal: **answer RQ3 and RQ4.**

- **Phase 3a (≈ ¥60)**: extract support ground truth from Robosuite. Compute pairwise contact via `mujoco.mj_collision()` over 50-ms windows; derive support = topological sort over the contact graph using gravity vector — A supports B iff (a) A contacts B, (b) B's centroid lies above A's contact patch along −z, (c) B has no other contact below it. Validate against hand-labeled gold set on 50 random LIBERO-Spatial frames; expect support F1 ≥ 0.9, contact-patch IoU ≥ 0.85 (used later as the L_dr ground-truth target for interpretability eval, not for direct supervision).
- **Phase 3b (≈ ¥180)**: train the +support variant on LIBERO-90 + finetune on 4 LIBERO suites + Simpler-WidowX-Stack-Block. Compare to Phase-2 best.
- **Phase 3c (≈ ¥120)**: ablations — (i) turn off derivability `L_dr` (does the support head still help downstream tasks? does its attention still concentrate on contact patches?), (ii) turn off support head entirely (recover Phase-2 best for Phase-3-suite numbers).
- **Phase 3d (≈ ¥140)**: real-world validation — finetune on 4 Piper-arm tasks (50 trajectories each, mirroring QDepth-VLA setup) and evaluate 10 trials per task. Compare to QDepth-VLA's published Piper numbers, especially on `task4 stack green-on-yellow` where QDepth-VLA scored only 10.0 %.

- **Phase-3 gate (RQ3)**: +support variant beats Phase-2 best on Simpler Stack-Block by ≥ 5 pp **and** on real-Piper stack-on-yellow by ≥ 15 pp. Fail ⇒ keep support head but reframe as auxiliary structural supervision rather than as a separately-emitted interpretable output.
- **Phase-3 gate (RQ4)**: support-prediction F1 (with derivability) within ±2 pp of the no-L_dr variant on downstream SR; **and** the support head's contact-patch attention vs Robosuite contact-patch GT shows F1 ≥ 0.85 (with L_dr) vs ≤ 0.50 (without L_dr) — this is the experimental claim that "contact emerges from derivability." Fail ⇒ honestly report that contact does not emerge cleanly without explicit supervision; reframe paper as multi-geometry + supervised support, drop the "contact emerges" claim.

### Phase 4 — Robustness & writeup (target: 5 days, ≈ ¥250)

- LIBERO-Plus visual-perturbation evaluation on Phase-3 best (Long-VLA / ReconVLA established this; ~¥80 for sweeps).
- Custom occluded-LIBERO: programmatically add box occlusions to LIBERO-Spatial frames at 25 / 50 / 75% area; evaluate degradation curves vs QDepth-VLA reproduction (~¥50).
- Train fresh figure-quality model on best config + larger seed (~¥80 for figure-ready checkpoints).
- ~¥40 reserved for re-runs / failed sweeps.
- Write paper draft (no compute).

**Total compute envelope**: ≈ **¥1,400** + **30-40 wall-clock days** at 1-2 A800 utilization. Within PSSA workspace's documented cost profile (¥330 for Phase 1 sweep).

### Total budget summary (revised after 5→4 head decision)

| Phase | Goal | Wallclock | GPU-h | ¥ (AutoDL) |
|---|---|---|---|---|
| 0 | Setup + smoke | 1 d | 2 | 14 |
| 1 | Reproduce QDepth-VLA | 2 d | 16 | 112 |
| 2 | +normal +plane (RQ1, RQ2) | 5 d | 60 | 400 |
| 3 | +support w/ derivability (RQ3, RQ4) | 6 d | 75 | 500 |
| 4 | Robustness + figures | 5 d | 35 | 250 |
| **Total** | | **19 d** | **188** | **≈ ¥1,276** |

19 wall-clock days assumes 2× A800 with task isolation as PSSA-VLA documented; can compress to ~11 days with 4× A800. Saved ~¥100 + 1 day vs the original 5-head plan by removing the contact expert.

---

## §4 · Datasets and benchmarks

### 4.1 Training data

| Dataset | Use | Pseudo-label source for normals / planes |
|---|---|---|
| **LIBERO-90** (Liu et al. 2023, arXiv:2306.03310) | VLA pretraining (3 epochs Phase-1, 6 Phase-2/3) | Robosuite GT — depth, normals, plane masks all queryable from MuJoCo renderer; contact / support from `mj_collision` + topo sort |
| **LIBERO-Spatial / Object / Goal / Long** | Per-suite finetune (50 epochs each suite per QDepth recipe) | Same as above |
| **OXE — Fractal subset** | Phase 1 pretrain (9 epochs per QDepth recipe) | Video-Depth-Anything (depth), Omnidata-V2 (normals), region-growing on depth (planes); contact / support **N/A** (no GT available — heads see no loss on OXE) |
| **OXE — Bridge subset** | Simpler-WidowX pretrain (13 epochs) | Same as Fractal |

### 4.2 Evaluation benchmarks

| Benchmark | Tasks × rollouts | Why it tests our claim | Phase introduced |
|---|---|---|---|
| **LIBERO-Spatial** | 10 × 50 | RQ1 — does multi-geometry beat depth-only? | Phase 1, 2, 3 |
| **LIBERO-Object** | 10 × 50 | Sanity baseline (object identity, not geometric) | Phase 2, 3 |
| **LIBERO-Goal** | 10 × 50 | Sanity baseline (goal generalization) | Phase 2, 3 |
| **LIBERO-Long** | 10 × 50 | RQ3 (long-horizon = relations matter) | Phase 2, 3 |
| **Simpler — Google Robot** | Pick Coke / Move Near / Drawer / Put-Apple, 10 × ≥ 24 each | Cross-platform generalization sanity | Phase 3 |
| **Simpler — WidowX250** | Carrot / Eggplant / Spoon / **Stack-Block**, 10 × ≥ 24 each | RQ3 — Stack-Block is the canonical support-relation task | Phase 2d, 3 |
| **Real-world Piper** | 4 tasks × 10 trials each (replicate QDepth-VLA's protocol) | RQ3, real-world deployment claim | Phase 3d |
| **LIBERO-Plus** | Visual perturbations on 4 suites, 10 × 50 | Robustness | Phase 4 |
| **Custom occluded-LIBERO** | LIBERO-Spatial with 25/50/75% box occlusion overlay, 10 × 30 | Occlusion claim from user motivation | Phase 4 |

CALVIN ABC-D is **not in scope** for the first paper — too costly to add a non-LIBERO simulator. Mentioned in `papers/literature_seed.md` only as future-work bridge to PSSA workspace.

---

## §5 · Baselines

### 5.1 Primary baselines (must be in main table)

These are the works our headline numbers must beat or match.

| Baseline | Source | LIBERO single-view avg (pub) | LIBERO multi-view avg (pub) | Reproducibility |
|---|---|---|---|---|
| **QDepth-VLA** | arXiv:2510.14836 | 85.4 | 94.9 | Reproduced in Phase 1 (own number stored) |
| **OpenVLA-finetuned** | arXiv:2406.09246 | 76.5 | — | PSSA workspace already has OpenVLA-7B-finetuned-libero-spatial running at 80.2 single-seed (within paper variance) |
| **open-π₀** | Allen Z. Ren reimpl. | 77.7 | 94.2 | Need to reproduce as part of Phase 1 |
| **CoT-VLA-7B** | arXiv:2503.22020 | 81.1 | — | Cite published number, no reproduce (compute-heavy) |
| **GeoVLA** | arXiv:2508.09071 | — | 97.7 | Cite published number; would need point cloud at inference, not our regime |
| **3D-CAVLA** | arXiv:2505.05800 | 82.6 | 98.1 | Cite published number; multi-view depth at inference, upper-bound reference |
| **GST-VLA** | arXiv:2603.09079 | TBD (read full text in Phase 0) | TBD | **Must reproduce or cite** — this is our closest competitor |
| **Spatial Forcing** | arXiv:2510.12276 | TBD | TBD | **Must include as additive baseline** — try GeoRel-VLA + SF alignment loss in Phase 4 |

### 5.2 Secondary baselines (cited only)

| Baseline | LIBERO multi-view avg | Notes |
|---|---|---|
| Diffusion Policy | 72.4 | RGB diffusion baseline |
| Octo-finetuned | 75.1 | Open generalist |
| π₀-FAST finetuned | 85.5 | Faster sampling variant |
| π₀ finetuned | 94.2 | Strong general-VLA |
| UniVLA | 95.2 | Latent-action axis |
| 4D-VLA | 88.6 | Spatiotemporal axis |
| DreamVLA | 92.6 | Future-depth dream axis |
| SpatialVLA | 78.1 single-view | Position-encoding axis |

### 5.3 Ablation baselines (vs ourselves)

- **GeoRel-VLA−normal**: Phase-2 minus normal head
- **GeoRel-VLA−plane**: Phase-2 minus plane head
- **GeoRel-VLA−normal,plane**: equals Phase-1 QDepth-VLA reproduction
- **GeoRel-VLA−L_xc**: Phase-2 minus cross-head consistency
- **GeoRel-VLA−contact**: Phase-3 minus contact head
- **GeoRel-VLA−support**: Phase-3 minus support head
- **GeoRel-VLA−L_dr**: Phase-3 minus derivability constraint
- **GeoRel-VLA + Spatial-Forcing alignment**: Phase-3 + SF loss (Phase 4)

---

## §6 · Metrics

### 6.1 Primary

- **Suite-level success rate** (SR): mean across 10 tasks × 50 rollouts. Bootstrap 95% CI over rollouts.
- **Per-task SR**: report all 40 per-task numbers in appendix for transparency.
- **Real-world SR**: 4 tasks × 10 trials, plus qualitative video.

### 6.2 Auxiliary (for ablation and analysis)

- **Depth reconstruction quality**: RMSE / abs-rel on validation depth maps (sanity check Phase 1).
- **Normal angular error**: median angular error on validation normals (Phase 2).
- **Plane mask IoU**: mean per-mask IoU vs Robosuite GT (Phase 2).
- **Contact F1, Support F1**: vs Robosuite-extracted GT on validation (Phase 3).
- **Derivability conformance rate**: fraction of predicted relations consistent with the geometric heads under the §2.4 thresholds.

### 6.3 Reporting policy

- All numbers come from a **single reported seed** in main tables, **3 seeds for headline ablations** (mean ± std), per common VLA practice.
- Real-world numbers are 10 trials per task; we **do not** average across tasks (each task SR reported separately).
- **No cherry-picking**: every checkpoint that ran to completion is logged in `logs/`; we report the latest, not the best-of-N.

---

## §7 · Ablation matrix (compact)

| Variant | Heads | Cross-consistency | Derivability | Phase | Where in paper |
|---|---|---|---|---|---|
| `qdepth-repro` | depth | — | — | 1 | §5.1 main table |
| `geo-3head-no-xc` | depth, normal, plane | off | — | 2c | §5.3 |
| `geo-3head` (Phase-2 best) | depth, normal, plane | on | — | 2b | §5.1 main table |
| `geo-3head − normal` | depth, plane | on | — | 2c | §5.3 |
| `geo-3head − plane` | depth, normal | on | — | 2c | §5.3 |
| `geo+sup-no-dr` | depth, normal, plane, support | on | off | 3c | §5.4 (does support help w/o derivability? does contact emerge?) |
| `geo+sup` (Phase-3 best) | depth, normal, plane, support | on | on | 3b | §5.1 main table |
| `geo+sup + SF-align` | depth, normal, plane, support + SF loss | on | on | 4 | §5.5 (additivity discussion) |

8 variants total (down from 10 in the original 5-head plan). Each evaluated on LIBERO-Spatial + Long + Simpler-Stack-Block at minimum; the headline three (`qdepth-repro`, `geo-3head`, `geo+sup`) on all 4 LIBERO suites.

---

## §8 · Smoke-test gates (per QDepth-VLA + PSSA workspace pattern)

Before any expensive run, the following must pass:

| Gate | Test | Pass threshold |
|---|---|---|
| `gate-1` | Load OpenVLA-7B-finetuned-libero-spatial → 5 rollouts on `libero_spatial:0` with PSSA's 2 fixes (--libero-action-fix + --libero-image-fix) | 5/5 SR, ≤ 5 min |
| `gate-2` | Pretrained depth VQ-VAE: reconstruct 10 random LIBERO frames | RMSE < 0.05 on normalized depth |
| `gate-3` | Pretrained normal VQ-VAE: reconstruct 10 random Robosuite normal maps | Median angular error < 10° |
| `gate-4` | Pretrained plane VQ-VAE: reconstruct 10 random plane masks | Per-mask IoU > 0.80 |
| `gate-5` | Support extraction (& latent contact-patch GT for L_dr eval): 50 hand-labeled LIBERO-Spatial frames | Support F1 ≥ 0.90, contact-patch IoU ≥ 0.85 |
| `gate-6` | One-step forward + backward of full Phase-3 model (4 experts + 5 loss terms) | No NaN, no OOM at batch 32 / A800 |
| `gate-7` | Phase-1 sanity: 1-epoch training of QDepth-VLA reproduction | Action-loss decreases monotonically; depth-loss decreases monotonically |

Failure of any gate triggers a "halt + diagnose" path before incurring main-run cost.

---

## §9 · Risks & fall-back plans (matched to ideation_summary §7)

| Risk | Trigger metric | Fall-back |
|---|---|---|
| GST-VLA (Mar '26) covers most differentiation | Phase-2 best does not beat GST-VLA's published LIBERO numbers | Reframe as "structured & decomposable supervision (vs. token-stream CoT)" + invest in interpretability case studies |
| Spatial Forcing wins by being simpler | SF beats Phase-3 best on LIBERO single-view by > 1 pp | Add SF loss to ours as additive baseline (Phase 4); reposition as "structured supervision atop SF backbone" |
| Pseudo-label noise (normal / plane on OXE) | Validation normal angular error > 25° on OXE val | Use Robosuite GT only (drop OXE pretraining); accept smaller pretraining set |
| Compute overrun > ¥2,000 | 1.5× projected at end of any phase | Stop at end-of-phase; halt before Phase 4; submit Phase-2 result as workshop paper if Phase 3 cannot complete |
| Support relation provides little gain | Phase-3 Stack-Block improvement < 3 pp | Drop the relational claim, present GeoRel as "multi-task geometric supervision" only — paper still publishable |
| Two-workspace overlap with PSSA-VLA | Reviewer flags duplicate authorship / scope | Clearly cite PSSA workspace as concurrent independent work; emphasize axis difference; do **not** combine in this paper |

---

## §10 · Hardware plan

- **Primary**: AutoDL A800 80 GB PCIe (matches PSSA workspace; same provider, same network workarounds).
- **Topology**: 2 × A800 with task isolation (PSSA documented this works); upgrade to 4 × A800 for Phase 3 if Phase 2 took the budgeted time.
- **Storage**: ≥ 80 GB for HF cache (PSSA cached 28 GB just for OpenVLA + LIBERO sims; we'll need additional VQ-VAE checkpoints, Omnidata-V2 weights, normal/plane label caches).
- **Networking**: `HF_ENDPOINT=https://hf-mirror.com` (huggingface.co blocked from CN per PSSA notes), `git config insteadOf https://gh-proxy.com/https://github.com/` for GitHub clone throttling. **Inherit PSSA's full network workaround set verbatim.**
- **Fallback**: if AutoDL has GPU shortage at run time, fall back to single A100 80 GB (slower wall-clock but same memory footprint).

For real-world Phase-3d:
- 6-DoF Piper arm + RealSense D455 (matches QDepth-VLA real-world setup) — assume access; if not, pivot to LIBERO-Plus only and defer real-world to a follow-up paper.

---

## §11 · Project timeline

| Week | Activity | Artifacts produced | Gate |
|---|---|---|---|
| 1 | Phase 0 + Phase 1 | `experiment/code/`, QDepth-VLA repro numbers in `logs/phase1_repro.json` | gate-1 + Phase-1 gate |
| 2 | Phase 2a + 2b | normal-VQ-VAE, plane-VQ-VAE checkpoints; Phase-2 main result | gate-2/3/4 + Phase-2 gate (RQ1) |
| 3 | Phase 2c + 2d + write §5.1, §5.3 of paper | ablation table; Stack-Block teaser | Phase-2 gate (RQ2) |
| 4 | Phase 3a + 3b | support + contact-patch extractor; Phase-3 main result | gate-5 + Phase-3 gate (RQ3) |
| 5 | Phase 3c + 3d (real Piper) | derivability ablation; Piper SR | Phase-3 gate (RQ4) |
| 6 | Phase 4 — robustness + figures | LIBERO-Plus, occluded-LIBERO, figure ckpt; SF-additive run | — |
| 7-8 | Writing | LaTeX draft, response to PSSA-overlap review | submit-ready |

Slippage budget: ≤ 1 week (push submission target by 1 week if Phase 3 needs re-runs). The 4-head simplification gives back ~1 day vs the 5-head plan, recoverable as slippage cushion.

---

## §12 · Decisions locked at user sign-off (2026-05-11)

1. ✅ **Working name** = `GeoRel-VLA`.
2. ✅ **Cardinality** = 4 heads (depth + normal + plane + support). Down from 5: contact head removed because it would be redundant with the support head's L_dr derivability constraint, which (a) requires a contact patch to exist and (b) requires the contact patch to be plane-aligned + gravity-aligned. Contact representation thus emerges inside the support head's attention, not as an independently supervised target. Decision driven by user feedback "不要感觉是强行模块堆砌."
3. ✅ **Backbone** = open-π₀.
4. ✅ **Benchmark triple** = LIBERO (4 suites) + Simpler-WidowX (incl. Stack-Block) + real Piper. CALVIN ABC-D = follow-up.
5. ✅ **Phased gating** = P1 reproduce QDepth-VLA → P2 +normal +plane → P3 +support w/ derivability. Explicit go/no-go after each.
6. ✅ **Compute envelope** ≈ ¥1,276 / 19 wallclock days at 2× A800 (revised down from ¥1,376 / 20 days after the head simplification).
7. ✅ **HOST mode continues for SETUP onward** unless the user later supplies an API key.

Stored in `manifest.json` `checkpoints.decisions_locked` block.

---

## §13 · What this blueprint does NOT cover

Honest scope disclosure for the user:

- **No code is written yet.** SETUP / CODING produce the actual Python; this blueprint specifies *what* and *why*, not *how*.
- **No environment is provisioned yet.** Inheriting PSSA's environment recipe is planned but not done.
- **No ground-truth label extraction script is written yet.** The Robosuite GT extraction (depth / normal / plane / support — and contact-patch GT only for the L_dr-eval gate, not for direct training supervision) is hand-described in §3 but not implemented.
- **No real Piper arm is reserved yet.** Phase 3d assumes access; if not, deferred to follow-up.
- **GST-VLA paper has only been read via WebSearch summary.** Phase-0 includes "read GST-VLA full text and lock the differentiation claim" as the first item — the differentiation in §2 of `plans/ideation_summary.md` is based on the abstract + WebSearch summary only and may need refinement once the full text is in hand.

These limitations are acknowledged here so the user can decide whether to (a) sign off and let SETUP proceed under a real API key, (b) request another HOST-mode pass with sharper differentiation reading, or (c) abandon and start from a different angle.
