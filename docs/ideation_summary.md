# Ideation Summary — Geometry-Structure Aware VLA

**Workspace:** `20260511121847e9b8`
**Stage:** IDEATION → PLANNING handoff
**Date:** 2026-05-11
**Mode:** Claude Code HOST (Opus 4.7) acting as research engine

> *This document is the single artifact the user reads to decide whether the proposed direction is worth investing compute in. Read this first; `papers/literature_seed.md` is the supporting evidence.*

---

## 1 · Original motivation (verbatim, user-provided)

> 当前 VLA 方法主要依赖二维图像区域进行视觉对齐的局限，通过引入深度、法向量、平面结构等丰富的场景表示信息作为辅助监督，使模型能够学习更具结构感知能力的空间表示。相比传统基于图像注意力区域的视觉对齐方式，我们要做的 VLA 能够显式建模目标物体的空间位置关系、接触关系以及支撑关系，从而提升机器人在遮挡环境、复杂操作任务以及精细操作场景中的鲁棒性和准确性。

Distilled to one English sentence:

> **Replace 2D-region visual grounding in VLA models with three decoupled geometric prediction heads (depth + surface normal + plane) plus two derived relational heads (contact + support), so the policy reasons in a structured, decomposable spatial schema — improving robustness under occlusion and on long-horizon, fine-grained manipulation.**

---

## 2 · Gap statement (after surveying ~30 closest works)

The "auxiliary 3D supervision for VLA" recipe is no longer novel by itself. As of May 2026, **at least nine** parallel works have grafted some form of geometric awareness onto a VLA backbone (QDepth-VLA, GST-VLA, GeoVLA, GraphCoT-VLA, 3D-CAVLA, 4D-VLA, MolmoAct, GeoAware-VLA, Spatial Forcing). What none of them yet do is the following intersection:

> **(a) supervise depth + surface normal + plane segmentation as three explicitly-decoupled heads, and (b) derive symbolic, structured contact + support relations from those primitives, and (c) tie everything back into the action expert via a cross-head consistency loss that forces the geometry heads to agree with each other and with the relational heads.**

Specifically:

| Existing work | Closest design choice | What they do NOT do |
|---|---|---|
| QDepth-VLA (Oct '25) | Depth expert with quantized tokens | Single signal (depth only); no normal / plane / relations |
| GST-VLA (Mar '26) | Gaussian primitives (mean = depth, covariance ≈ orientation), DA-CoT for contact geometry | Orientation entangled in covariance, not an explicit head; contact via token-stream CoT, not symbolic relation; no support reasoning |
| GeoVLA (Aug '25) | Point cloud encoder + 3D-enhanced action expert | 3D used as input not supervision; no relation prediction |
| GraphCoT-VLA (Aug '25) | Spatial-aware reasoning graph for ambiguous instructions | Symbolic graph but **without** geometric supervision producing it — the graph is hand-crafted in CoT |
| Spatial Forcing (ICLR '26) | Implicit alignment of intermediate VLA embeddings to VGGT features | Implicit-only, no decomposable / interpretable outputs; no relation prediction |
| GeoAware-VLA (Sep '25) | Frozen VGGT as visual encoder | Same — implicit, no relational outputs |
| Lift3D Policy (CVPR '25) | Multi-task aux: affordance mask + depth reconstruction (closest in spirit) | **Not VLA**, no normal / plane head, no relation prediction |
| GVMRN (2021), Support Relation Analysis (2020) | Classical: support-relation graph from RGB-D for grasping order | **Not learned end-to-end**, not in a VLA, no language conditioning |

The gap is therefore **defensible**, but narrow — the contribution must be the *combination* and the *cross-head consistency story*, not any single one of (depth aux), (normal aux), (plane aux), (contact head), (support head). Each in isolation is foreseeable (or already published in a non-VLA setting); the package is what is novel.

---

## 3 · Candidate name (locked)

**GeoRel-VLA** — Geometry-and-Relation-aware Vision-Language-Action model. Locked at user sign-off, 2026-05-11.

Rationale: keeps the `<modifier>-VLA` convention (QDepth-VLA, SpatialVLA, GeoVLA, GST-VLA …); "Geo" signals the multi-task geometric supervision; "Rel" signals the symbolic support output that distinguishes us from the depth-only / Gaussian-only crowd.

---

## 4 · Thesis (one paragraph the paper Introduction collapses to)

> Existing VLA models perform visual grounding via 2D image-region attention, which collapses position, orientation, and inter-object relations into the same embedding subspace. We propose **GeoRel-VLA**, an auxiliary-supervision framework in which an open-π₀ backbone is co-trained with **three decoupled geometric prediction heads — depth (scalar field), surface normal (vector field), plane segmentation (region map) — and one functional relational head — pairwise object support** — that share an embedding bus with the action expert. A cross-head consistency loss forces depth, normal, and plane outputs to be locally coherent (e.g., flat-plane normals must agree with plane-mask normals), and a derivability constraint forces the support head to be grounded in the geometric heads (predicted "A supports B" requires a planar contact patch on A directly under B's bottom face along gravity in the depth/normal field). Contact relations are not separately supervised; they emerge as the geometrically-grounded prerequisite that the derivability constraint enforces inside the support head. The support output is exposed to the action expert as a small, structured side-bus so the policy can attend to "what holds the target up" the same way it attends to "the target." We hypothesise — and propose to verify on LIBERO, Simpler-WidowX (especially Stack-Block), and a real Piper arm — that this decomposition yields the largest gains on **occluded scenes, long-horizon multi-stage tasks, and fine-grained placement / stacking** — exactly the regimes where 2D-region grounding fails most visibly.

---

## 5 · Three core contributions (paper-ready bullets)

1. **A structured, decoupled multi-task geometric supervision scheme for VLA.** We are the first to co-train depth + surface normal + plane segmentation as three explicit auxiliary heads inside a VLA, and the first to use a **cross-geometry consistency loss** that forces the three heads to agree at training time. The three heads are not redundant: ablations remove each in isolation and show a different task family degrades for each (LIBERO-Spatial cares most about depth; low-texture / glossy scenes care about normal; placement / stacking cares most about plane).
2. **Geometrically-grounded relational supervision via a single support head with derivability constraint.** Object-support relations are predicted by a dedicated head supervised from simulator ground truth, with a derivability term that requires every predicted "A supports B" edge to (i) admit a planar contact patch on A's surface in the predicted depth + plane field, and (ii) be aligned with gravity in the predicted normal field. Contact relations therefore arise as the geometrically-grounded prerequisite of support, without needing an additional supervised contact head. To our knowledge no prior VLA emits a geometry-grounded support output.
3. **Empirical demonstration on the regimes where 2D-region VLAs visibly fail.** We target three regimes underlined by the user motivation — **occlusion** (LIBERO-Plus visual perturbations + custom occluded-LIBERO variant), **long-horizon multi-stage** (LIBERO-Long), and **fine-grained placement / stacking** (Simpler-WidowX Stack-Block, real-world stack-on-yellow). Targets (provisional, locked in PLANNING):
   - LIBERO single-view average ≥ **87.0** (vs QDepth-VLA 85.4; +1.6 abs)
   - LIBERO-Long single-view ≥ **76.0** (vs QDepth-VLA 72.6; +3.4 abs — long-horizon is exactly where structured relations should pay off)
   - Simpler Stack-Block ≥ **48.0** (vs QDepth-VLA 39.6, SpatialVLA 29.2; +8 abs — stacking is exactly where support-relation supervision matters)
   - Real-world Piper "stack green-on-yellow" ≥ **30.0** (vs QDepth-VLA 10.0; +20 abs — the most striking demo)

These targets are aggressive but not heroic; they are within the spread seen between QDepth-VLA → GeoVLA on LIBERO (~12 pp) and between QDepth-VLA → 3D-CAVLA on Simpler.

---

## 6 · Why now? Why us?

- **Why now**: All structural ingredients are mature in 2026 — open-π₀ released and reproduced (QDepth-VLA, PSSA-VLA workspace both use it), VGGT and Depth-Anything-V2 give clean monocular geometry pseudo-labels, LIBERO + Robosuite provide free contact / support ground truth from MuJoCo, Spatial Forcing's plug-and-play recipe shows ~30-line auxiliary losses already produce SOTA results. Six months earlier, depth pseudo-labels were too noisy (QDepth-VLA cites this); six months later, this gap will likely be closed by someone else.
- **Why us**: Companion workspace `20260509000456c3a8` (PSSA-VLA) has already established our local capability to (i) reproduce open-π₀ baselines (80.2% on LIBERO-Spatial vs paper 84.7%, within replication variance), (ii) run real LIBERO sims at ~190 ms/step on AutoDL A800 with all version pins / network workarounds documented, (iii) operate the AutoDL cluster within ~330 RMB / Phase-1 sweep budget. Reusing that infrastructure (but not literature) means our ramp-up cost on this project is hours, not weeks.

---

## 7 · Threats & honest risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| **GST-VLA (Mar '26) covers most of the differentiation.** Their Gaussian covariance encodes orientation; their DA-CoT covers contact geometry. | Medium-High | Keep our differentiation claim tight: **explicit decoupled heads ≠ entangled covariance**, and **symbolic relation outputs ≠ DA-CoT token stream**. Run a head-to-head against GST-VLA's published numbers on LIBERO. If we cannot show consistent advantage on at least Long + Stack-Block, the project should pivot or stop after the empirical study. |
| **Spatial Forcing (ICLR '26) wins by being simpler.** Implicit VGGT alignment, 3.8× speedup, plug-and-play. | Medium | Frame ours as **complementary**: GeoRel-VLA + Spatial-Forcing alignment loss should be additive (the SF loss aligns embeddings, our heads supervise outputs). Include SF as an additive baseline in ablations. |
| **Pseudo-label noise dominates the auxiliary heads.** QDepth-VLA explicitly motivates quantization because pixel depth regression hurt them; same risk applies to normal / plane. | High | (a) Use **simulator GT in LIBERO** (free) as the primary training signal; (b) use VGGT / Omnidata-V2 / region-growing plane segmentation as pseudo-labels only on real datasets; (c) follow QDepth's design — quantize each head's output to a small codebook so the loss is CE not regression; (d) ablate (regression head) vs (quantized head) for each of depth/normal/plane. |
| **Compute overrun.** 5 auxiliary heads × 4 LIBERO suites × ablations. | Medium | Stage in three phases: Phase 1 (depth-only reproduction of QDepth-VLA, sanity check), Phase 2 (add normal + plane, no relations yet), Phase 3 (add contact + support relational heads). Each phase must pass a go/no-go gate before the next. |
| **The "support relation matters" claim turns out to be small in practice.** Many LIBERO tasks do not involve stacking. | Medium | Front-load Simpler-WidowX Stack-Block and the real-world stack task in the evaluation; if support supervision does not help these specifically, drop the relational claim and reframe as a multi-task geometric supervision paper (still publishable, just narrower). |
| **Two-workspace overlap with PSSA-VLA.** Both target VLA grounding; risk of cannibalization. | Low | The two workspaces start from different premises (temporal consistency vs structured geometry) and have **independent literature reviews per user request**. They can be cross-evaluated as orthogonal lines that may later combine. |

---

## 8 · What changes vs the user's original framing

The user's motivation says "spatial position relations, contact relations, support relations." We take this as the **target output** and add a layer underneath: the geometric primitives (depth, normal, plane) that *cause* those relations to be predictable. This is a strict super-set of the original framing — if the user wants a leaner story, the relational heads can be dropped without breaking the geometric ones, but our recommendation is to keep them because they are the part that nobody else has done.

Two things in the user's original framing that we propose to slightly defer rather than drop:

1. *"Improving robustness in occluded environments"* — we add this to the evaluation (LIBERO-Plus + custom occluded-LIBERO) but do **not** make it a core architectural claim in the first paper, because handling occlusion well requires either active perception or amodal completion, both of which are sufficient for a separate paper. Honest framing: our structured geometry helps under mild occlusion; severe-occlusion is left to future work.
2. *"Fine-grained operation"* — Simpler Stack-Block and the real Piper stack-on-yellow are our fine-grained proxies. We do **not** claim sub-millimetre dexterous manipulation in this paper; the action backbone (open-π₀) limits us to ~5 mm placement accuracy.

---

## 9 · Decisions locked at user sign-off (2026-05-11)

1. ✅ **Name** = `GeoRel-VLA`.
2. ✅ **Cardinality** = 4 heads (depth + normal + plane + support). Down from the initial 5; contact head removed because it would be redundant with the support head's derivability constraint — *contact emerges as the geometric prerequisite the constraint enforces, not as a separate supervised target.* (User's words: "不要感觉是强行模块堆砌").
3. ✅ **Backbone** = open-π₀ (matches QDepth-VLA + PSSA workspace).
4. ✅ **Benchmark triple** = LIBERO + Simpler-WidowX + real-Piper. CALVIN ABC-D deferred to follow-up.
5. ✅ **Phased gating** = P1 reproduce QDepth-VLA → P2 +normal +plane → P3 +support. Each phase has explicit go/no-go.
6. ✅ **Compute envelope** ≈ ¥1260 / 18-19 wallclock days at 2× A800 (revised down from the 5-head ¥1376 estimate; saved ~8% by removing the contact expert).
7. ✅ **HOST mode continues for SETUP onward** unless the user later supplies an API key.

These are now locked in `manifest.json` `checkpoints` block and reflected throughout `plans/experiment_blueprint.{md,json}`.

---

## 10 · Handoff to PLANNING

Inputs PLANNING needs to consume:
- `papers/literature_seed.md` — full annotated bibliography, gap matrix, benchmark inventory
- this file — thesis, contributions, targets, risks
- companion workspace `20260509000456c3a8` infrastructure notes (network workarounds, version pins, AutoDL cost model) — **infrastructure only, no literature reuse**

Outputs PLANNING must produce:
- `plans/experiment_blueprint.md` (human-readable)
- `plans/experiment_blueprint.json` (machine-readable, NanoResearch CLI compatible)

with at minimum: research questions, primary / secondary baselines, full benchmark list with metrics & rollout counts, ablation matrix, compute estimate, hardware plan, smoke-test gate, fall-back plans for each risk above.
