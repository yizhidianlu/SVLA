# Literature Seed — Geometry-Structure Aware VLA (depth + normal + plane → contact + support)

**Workspace:** `20260511121847e9b8`
**Stage:** IDEATION (Claude Code HOST mode)
**Date:** 2026-05-11
**Curated by:** Claude Code (Opus 4.7) via WebSearch + arXiv MCP + paper-search MCP

---

## 1 · Scope of survey

This bibliography is built around three intersecting lines of work that together delimit the design space of the proposed project:

1. **3D / depth integration in Vision-Language-Action (VLA) models** — input-side fusion, projection, and auxiliary-supervision variants.
2. **Multi-task geometric supervision in robot policy learning** — depth, surface normal, plane, and affordance prediction as auxiliary heads.
3. **Symbolic / relational scene reasoning for manipulation** — contact graphs, support relations, and stacking / placement reasoning.

Inclusion bar: 2023-2026 work that is either a direct competitor (auxiliary 3D supervision in VLA) or supplies an experimentally-relevant baseline / dataset / loss design. A small number of pre-2023 anchors are kept where they are still the canonical reference (VoxPoser, GVMRN, classical support-relation analysis).

---

## 2 · Tier 1 — Direct competitors (auxiliary 3D supervision in VLA)

The closest prior art. These works (a) keep the VLA backbone 2D, (b) introduce a geometric prediction head as **auxiliary** supervision (not as input), and (c) report results on LIBERO / Simpler / OXE-derived benchmarks. Their existence forces our story to differentiate on **what is supervised** and **how the supervision is decomposed**, not on the high-level recipe of "auxiliary 3D loss for VLA."

| ID | Paper | Venue / date | Geom signal supervised | Mechanism | Why it matters to us |
|---|---|---|---|---|---|
| **QDepth-VLA** | Li et al., *QDepth-VLA: Quantized Depth Prediction as Auxiliary Supervision for Vision–Language–Action Models* — arXiv:2510.14836 | Oct 2025 (v2 Dec 2025) | **Depth only** (quantized via VQ-VAE codebook K=256) | Dedicated depth expert (transformer, mirrors action expert) attends image+text; co-training loss `L_action + λ_t · L_depth`, λ₀=0.01 with exp decay; hybrid attention mask routes depth into action expert | **Most direct precedent.** Confirms that (i) auxiliary geometric supervision improves LIBERO and Simpler, (ii) **quantized** beats pixel-wise regression, (iii) a separate expert avoids interfering with VLM semantics. We must beat its 85.4 single-view LIBERO average. |
| **GST-VLA** | Sarowar, Tariq, Kim, *GST-VLA: Structured Gaussian Spatial Tokens for 3D Depth-Aware VLA* — arXiv:2603.09079 | Mar 2026 | Depth + **implicit surface orientation** (Gaussian covariance eigenstructure) + opacity confidence | 128 anisotropic 3D Gaussian primitives parameterized by metric residual mean, log-scale covariance, learned opacity; spatial-attention pooling with learned queries; **3D Depth-Aware Chain-of-Thought (DA-CoT)** supervises 4 intermediate thoughts: (a) 3D object grounding, (b) **grasp-affordance contact geometry**, (c) pairwise metric distances, (d) coarse SE(3) waypoints | **Most comprehensive competitor.** Already covers depth + (implicit) orientation + contact geometry. Our differentiation has to come from (a) **explicit & decoupled** normal / plane heads (GST entangles them in covariance), (b) **symbolic** support-relation prediction (DA-CoT does not produce structured relations), (c) cross-head consistency losses. |
| **DreamVLA** | Zhang et al., *DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge* — arXiv:2507.04447 | Jul 2025 | **Future depth** + dynamic / semantic prediction | Predicts future-depth maps as part of dreamed world state; serves as auxiliary visual prediction task | Established that pixel-wise depth prediction CAN be detrimental in VLA training (QDepth-VLA cites this as motivation for quantization). Use as cautionary baseline. |
| **3D-CAVLA** | Bhat et al., *3D CAVLA: Leveraging Depth and 3D Context to Generalize VLA for Unseen Tasks* — arXiv:2505.05800 | May 2025 | Depth + RoI 3D context | RoI pooling with depth embeddings projected into VLM token space; explicit point-cloud / depth at inference | Uses 3D **as input** rather than supervision; LIBERO multi-view 98.1% avg. Strong upper bound to compare against — we should approach it without 3D inputs at inference. |
| **4D-VLA** | Zhang et al., *4D-VLA: Spatiotemporal VLA Pretraining with Cross-Scene Calibration* — arXiv:2506.22242 | Jun 2025 | Depth + temporal alignment via 3D coord. embeddings | 3D coordinate embeddings augment visual tokens for spatial + temporal reasoning | LIBERO multi-view 88.6 avg — weaker than QDepth-VLA / GeoVLA. Time-extension axis we might combine later. |
| **CoT-VLA** | Zhao et al., *CoT-VLA: Visual Chain-of-Thought Reasoning for VLA* — CVPR 2025 / arXiv:2503.22020 | Mar 2025 | RGB sub-goal images (visual CoT, not geometry per se) | Auto-regresses sub-goal RGB then action; high sample cost | Establishes the "auxiliary visual reasoning head" pattern. Pure RGB reasoning lacks geometric grounding — motivates why we add geometric heads instead. |
| **GraphCoT-VLA** | (8/2025) — arXiv:2508.07650 | Aug 2025 | 3D spatial-aware reasoning graph | Graph-structured CoT with explicit object-relation nodes; tackles ambiguous instructions | Closest prior work that emits **structured** spatial relations during reasoning. Symbolic graph but no geometric supervision — **our project = inverse: geometric supervision driving symbolic relations.** |

---

## 3 · Tier 2 — 3D-input VLAs (feed 3D in, no auxiliary supervision)

These works integrate 3D **at the input side**: point clouds, depth maps, voxels, orthographic projections, or position encodings. They establish strong upper bounds when 3D sensors are available at inference. Our auxiliary-supervision approach is intentionally **inference-time-RGB-only**, so these define a comparison axis, not a competitor on the same axis.

| ID | Paper | Date | 3D mechanism | Notes |
|---|---|---|---|---|
| **PointVLA** | Li et al., arXiv:2503.07511 (RA-L 2025) | Mar 2025 | Point cloud injected via lightweight modular block; **vanilla action expert frozen** | Plug-and-play 3D-into-pretrained-VLA. Modality-agnostic semantic alignment exemplar. |
| **GeoVLA** | Sun et al., arXiv:2508.09071 | Aug 2025 | Depth → point cloud → Point Embedding Network → 3D-enhanced action expert; concatenates 2D + 3D embeddings | LIBERO multi-view **97.7 avg**, ManiSkill2; height + scale + viewpoint robustness in real-world. Strong upper bound. |
| **3D-VLA** | Zhen et al., arXiv:2403.09631 (ICML 2024) | Mar 2024 | 3D-LLM backbone + diffusion world model that generates goal images / point clouds | Original "3D-VLA". Generative world modeling axis — different from our discriminative auxiliary heads. |
| **3DS-VLA** | Li et al., PMLR v305:25g (CoRL 2025) | 2025 | 3D-aware end-effector pose prediction from 2D VLM + 3D awareness | Multi-task robust manipulation. |
| **SpatialVLA** | Qu et al., arXiv:2501.15830 | Jan 2025 (v5 mid-2025) | **Ego3D Position Encoding** + **Adaptive Action Grids**; pre-trained on 1.1 M real episodes | Position-encoding axis. Single-view LIBERO 78.1 avg — strong general-VLA but does not supervise geometry. |
| **OG-VLA** | arXiv:2506.01196 | Jun 2025 | **Orthographic** image generation as 3D awareness signal | Project-to-2D paradigm — preserves VLM 2D priors at the cost of projection loss. |
| **MolmoAct** | Allen AI, arXiv:2508.07917 | Aug 2025 | Three-stage autoregressive: **Depth Perception Tokens** → Visual Reasoning Trace Tokens → Action Tokens | Depth as **reasoning token stream** (not auxiliary loss). +23.3% OOD generalization. MolmoAct2 (arXiv:2605.02881) does adaptive depth reasoning only for changed regions. |
| **BridgeVLA** | Li et al., arXiv:2506.07961 | Jun 2025 | Renders 3D inputs into multi-view 2D images for VLM compatibility | "Project 3D → 2D" paradigm. |
| **FP3** | Yang et al., arXiv:2503.08950 | Mar 2025 | 3D foundation policy for manipulation | 3D-encoder upper-bound baseline. |
| **Evo-0** | Lin et al., arXiv:2507.00416 | Jul 2025 | VLA with **implicit** spatial understanding | Implicit-prior axis precursor to Spatial Forcing. |

---

## 4 · Tier 3 — Geometric-prior VLAs (frozen 3D foundation models)

Recent thread (late 2025 → ICLR 2026) that exploits a **pretrained 3D foundation model** (notably VGGT) to inject geometry without sensors. These are not strictly auxiliary supervision in our sense, but they share our motivation: leverage 3D priors without changing inference modality. Their existence raises the bar — anyone proposing geometric VLA improvements has to compare.

| ID | Paper | Date | Mechanism |
|---|---|---|---|
| **GeoAware-VLA** | arXiv:2509.14117 | Sep 2025 | Replace VLA vision encoder with **frozen VGGT** + lightweight projection layer; injects geometric prior; +2× zero-shot novel-view-pose success on LIBERO |
| **Spatial Forcing (SF)** | arXiv:2510.12276 — **ICLR 2026** | Oct 2025 | Aligns intermediate VLA visual embeddings with VGGT geometric representations as supervision signal; no explicit 3D inputs; plug-and-play (~30 LoC); **3.8× training speedup**; SOTA over 2D and 3D VLAs |
| **GeoPredict** | arXiv:2512.16811 | Dec 2025 | Predictive kinematics + 3D Gaussian geometry for precise manipulation |

> **Implication for our project.** Spatial Forcing is the most threatening recent baseline because it (a) gets geometric awareness for free, (b) is plug-and-play, (c) has SOTA numbers, (d) is published at ICLR'26. Our auxiliary-supervision story has to argue that **structured, decomposed, relational** supervision provides capabilities (e.g., explicit support-graph reasoning) that an implicit alignment loss cannot.

---

## 5 · Tier 4 — VLA backbones (the substrate to extend)

These are the action policies on top of which the auxiliary supervision is grafted. We adopt one (likely **open-π₀** following QDepth-VLA, since it is the strongest open VLA action backbone with a published auxiliary-supervision protocol).

| ID | Paper | Date | Notes |
|---|---|---|---|
| **OpenVLA** | Kim et al., arXiv:2406.09246 (CoRL 2024) | Jun 2024 | Open generalist VLA; LIBERO single-view 76.5 avg. Standard low-bar baseline. |
| **π₀** | Black et al., arXiv:2410.24164 | Oct 2024 | Vision-Language-Action **flow** model (CFM action loss); the dominant action-expert formulation. |
| **π₀.5** | Physical Intelligence, arXiv:2504.16054 | Apr 2025 | Open-world generalization extension. |
| **open-π₀** | Allen Z. Ren | 2024-25 | Open re-implementation of π₀; substrate used by QDepth-VLA and likely best fit for ours. |
| **π₀-FAST** | arXiv:2410.24164 (FAST variant) | 2024 | Faster sampling variant; LIBERO multi-view 85.5 avg. |
| **CoT-VLA-7B** | Zhao et al., CVPR 2025 | Mar 2025 | Visual chain-of-thought VLA; strong single-view baseline (81.1). |
| **RT-2 / RT-2-X** | Brohan et al., arXiv:2307.15818 | 2023 | First scaled VLA; architectural ancestor. |
| **RT-1** | Brohan et al., arXiv:2212.06817 | 2022 | Foundational manipulation transformer. |
| **Octo** | Octo Team, arXiv:2405.12213 | May 2024 | Open-source generalist policy; LIBERO multi-view 75.1 avg. |
| **GR-2 / GR-3** | ByteDance, arXiv:2410.06158 / 2507.15493 | 2024-25 | Generative video-language-action; web-scale knowledge transfer. |
| **UniVLA** | Bu et al., arXiv:2505.06111 | May 2025 | Task-centric latent actions; LIBERO multi-view 95.2 avg. |
| **RoboVLM** | Li et al., arXiv:2412.14058 | Dec 2024 | "What matters in building VLA" recipe paper. |
| **Long-VLA** | Fan et al., arXiv:2508.19958 | Aug 2025 | Long-horizon VLA; relevant for our LIBERO-Long target. |
| **ReconVLA** | Song et al., arXiv:2508.10333 | Aug 2025 | **Reconstructive** VLA — reconstruction as auxiliary signal; close-cousin design point. |
| **AgiBot World Colosseo** | AgiBot Contributors, arXiv:2503.06669 | Mar 2025 | Large-scale manipulation platform; latent-action pretraining axis. |
| **ChatVLA-2** | NeurIPS 2025 poster | 2025 | Open-world reasoning VLA. |

---

## 6 · Tier 5 — Adjacent (contact, support, occlusion, force/tactile)

Less direct, but each contributes a building-block for the relational outputs we want to supervise.

### 6.1 Contact / tactile / force in VLA
- **DreamTacVLA** — Learning to Feel the Future for Contact-Rich Manipulation, arXiv:2512.23864 (Dec 2025). Tactile awareness in VLA; addresses contact-rich gap.
- **ForceVLA** — Force as first-class modality with force-aware MoE; significantly improves contact-rich manipulation.
- **CoA-VLA** — *Improving VLA via Visual-Text Chain-of-Affordance*, ICCV 2025 (arXiv:2412.20451). Symbolic affordance reasoning — analogous CoT pattern but for affordances, not relations.

### 6.2 Affordance / scene-graph / relation reasoning (non-VLA but transferable)
- **RoboPoint** — Yuan et al., arXiv:2406.10721 (CoRL 2024). VLM that predicts spatial-affordance keypoints. Beats GPT-4o by 21.8% on spatial affordance, 30.5% on downstream task SR.
- **VoxPoser** — Composable 3D Value Maps, CoRL 2023. LLM-generated Python that composes 3D voxel value maps. Symbolic-spatial reasoning ancestor.
- **ManipLLM** — *Embodied Multimodal LLM for Object-Centric Robotic Manipulation*, CVPR 2024.
- **GVMRN** — *Graph-Based Visual Manipulation Relationship Reasoning Network for Robotic Grasping*, Frontiers Neurorobotics 2021. Considers **three kinds of support relations** in cluttered scenes; pre-deep-VLA but the only canonical reference for "structured support relation prediction → manipulation order."
- **Self-Supervised Scene-Graph Representations for Robotic Sequential Manipulation** — Nguyen et al., CoRL 2020. Direct precursor to graph-based manipulation reasoning.
- **Support Relation Analysis for Objects in Multiple-View RGB-D Images** — Springer 2020. Volumetric "true support" reasoning. Possible source of supervision signals.
- **Planning for Multi-Object Manipulation with Graph Neural Networks** — arXiv:2209.11943. GNN over partial point clouds to predict inter-object relation changes given actions.
- **Skill Composition via Scene-Graph Atomic Skills** — ICRA 2026.

### 6.3 Occlusion / amodal manipulation
- **Vision in Action (ViA)** — arXiv:2506.15666. Active perception for multi-stage bimanual manipulation under visual occlusion. Uses DINOv2 over RGB-D.
- **Lift3D Policy** — Jia et al., CVPR 2025. Lifts 2D foundation models for 3D robotic manipulation; **task-related affordance masking + depth reconstruction as multi-task auxiliary supervision**. Closest prior art to our multi-head auxiliary geometry idea **outside VLA** — strong inspiration.

### 6.4 Geometry foundation models (as prior or supervisor)
- **VGGT — Visual Geometry Grounded Transformer** — basis of GeoAware-VLA and Spatial Forcing. Single forward pass produces camera params, multi-view depth, dense point clouds, point tracking. Likely our default source of pseudo-ground-truth normal / depth signals when LIBERO does not provide them natively.
- **Depth Anything V2** — Yang et al., arXiv:2406.09414. Monocular depth foundation; QDepth-VLA uses Video-Depth-Anything (ViDA) for OXE depth pseudo-labels.
- **Video-Depth-Anything (ViDA)** — Chen et al., CVPR 2025. Temporally-consistent monocular video depth. Likely needed for our depth pseudo-labels at training time.
- **DepthCues** — arXiv:2411.17385. Benchmark of monocular depth perception in large vision models — useful for diagnosing what 2D priors a VLM already knows.

### 6.5 Surface normal / plane estimation foundations
- **Region-Growing Planar Segmentation for Robot Action Planning** — Springer 2015. Classical baseline for plane segmentation as a precursor to action reasoning. Not learned; can supply ground truth on planar scenes.
- **Occlusion-Aware Depth Estimation with Adaptive Normal Constraints** — ECCV 2020. Joint depth + normal supervision improves both — direct evidence for our multi-task hypothesis.

---

## 7 · Benchmarks adopted by Tier 1 competitors

QDepth-VLA, GeoVLA, 3D-CAVLA, SpatialVLA, MolmoAct, GeoAware-VLA all converge on a small core of benchmarks. Adopting the same set is mandatory for direct numerical comparison.

### Simulation
1. **LIBERO** (Liu et al., arXiv:2306.03310) — 4 suites × 10 tasks × 50 rollouts:
   - **LIBERO-Spatial** (10) — spatial reasoning between objects
   - **LIBERO-Object** (10) — object identity reasoning
   - **LIBERO-Goal** (10) — goal-conditioned variation
   - **LIBERO-Long** (10) — long-horizon multi-stage
   - QDepth-VLA single-view: 86.0 / 88.8 / 94.0 / 72.6 (avg 85.4); multi-view: 97.6 / 96.6 / 95.2 / 90.0 (avg 94.9)
   - GeoVLA multi-view: 98.4 / 99.0 / 96.6 / 96.6 (avg 97.7)
   - 3D-CAVLA multi-view: 98.2 / 99.8 / 98.2 / 96.1 (avg 98.1) — **upper bound with 3D inputs at inference**
2. **Simpler** (Li et al., arXiv:2405.05941) — Google Robot tasks (Pick Coke Can / Move Near / Open-Close Drawer / Put Apple In) and WidowX250 tasks (Put Carrot / Put Eggplant / Put Spoon / **Stack Block**). The **Stack Block** task is the canonical "support relation matters" task — QDepth-VLA reaches only 39.6%, SpatialVLA 29.2%, open-π₀ 15.8% — there is large headroom for an explicit support-supervision approach.
3. **CALVIN ABC-D** — long-horizon language-conditioned. Used by PSSA-VLA (companion workspace) and many earlier VLA works; secondary in tier-1 papers.
4. **ManiSkill2** — used by GeoVLA, supplements LIBERO.
5. **LIBERO-Plus** — robustness extension (visual perturbations); used by Long-VLA / ReconVLA.

### Real-world
- **6-DoF Piper arm + RealSense D455** — QDepth-VLA's real setup; 4 pick-and-place / stacking tasks × 10 trials. **Exactly the kind of setup we should reproduce** (small, low-cost, accessible).
- **AutoDL A800 80 GB** — companion workspace's real GPU setup (PSSA-VLA spent ~330 RMB on Phase 1 baseline sweep). Likely available to us.

### Auxiliary signals (where to get geometric ground truth)
- **LIBERO** uses Robosuite + MuJoCo — depth, normals, plane masks **available** by querying the renderer. This makes LIBERO an ideal training-time pseudo-GT source.
- **OXE / Open-X-Embodiment** — RGB only; need pseudo-labels via VGGT or Depth-Anything-V2 for depth, normals via VGGT or Omnidata-V2.
- **Video-Depth-Anything (ViDA)** — used by QDepth-VLA for OXE temporal-consistent depth.
- **Omnidata-V2** — best monocular normal estimator, classical choice when ground-truth normals unavailable.

---

## 8 · Synthesis — what is supervised in prior work, and what is missing

Coverage matrix across the geometric and relational supervision space we care about:

| Method (Tier 1) | Depth (head) | Surface normal (head) | Plane segmentation (head) | Contact relation | Support relation | Symbolic / decomposable output |
|---|---|---|---|---|---|---|
| QDepth-VLA | ✅ quantized | — | — | — | — | partial (codebook) |
| GST-VLA | ✅ Gaussian mean | partial (covariance eigenstruct.) | — | partial (DA-CoT) | — | partial (CoT) |
| GeoVLA | (point cloud, input) | — | — | — | — | — |
| 3D-CAVLA | (depth, input) | — | — | — | — | — |
| 4D-VLA | (3D coord embed, input) | — | — | — | — | — |
| DreamVLA | ✅ future depth (pixel) | — | — | — | — | — |
| MolmoAct | ✅ depth tokens (reasoning) | — | — | — | — | — |
| GraphCoT-VLA | — | — | — | partial (graph CoT) | partial (graph CoT) | ✅ (graph) |
| SpatialVLA | (Ego3D pos enc) | — | — | — | — | — |
| GeoAware-VLA | (frozen VGGT enc) | implicit (in VGGT) | — | — | — | — |
| Spatial Forcing | (VGGT alignment) | implicit (in VGGT) | — | — | — | — |
| **Proposed (this workspace)** | ✅ (head) | ✅ (head) | ✅ (head) | ✅ (head, derived) | ✅ (head, derived) | ✅ (relation graph) |

**The gap.** No prior VLA work supervises (depth, normal, plane) **jointly as decoupled heads** *and* derives **explicit symbolic contact / support relations** from the geometric primitives. GST-VLA comes closest but entangles orientation in Gaussian covariance and produces contact only as DA-CoT token streams. GraphCoT-VLA produces relational graphs but not from geometric supervision. The combination — multi-task structured geometry → structured relations — is **uncovered** as of May 2026.

This is the seed for the candidate name and thesis in `plans/ideation_summary.md`.

---

## 9 · Companion workspace cross-link (information only, no shared content)

Workspace `20260509000456c3a8` (PSSA-VLA — Persistent Temporal Scene-Spatial Alignment for VLA) explores a different angle on the same VLA-grounding bottleneck: **temporally consistent scene-entity tracking** as a cross-frame consistency signal. It overlaps with the present project on (a) backbone choice (open-π₀ + LIBERO) and (b) belief that 2D-region grounding is the bottleneck, but its solution (temporal entity consistency) and ours (multi-task geometric supervision + relational outputs) are orthogonal axes; literature was **deliberately not reused** per user request so each workspace can be evaluated on its own.

The two could in principle be combined later (PSSA's temporal entity tracker × our geometric / relational heads), but that decision is out of scope for this workspace.

---

## 10 · Open questions left for PLANNING stage

1. Which geometric primitives ground-truth do we get **for free** in LIBERO via Robosuite, vs. which need pseudo-labels? (Plane masks are usually computable; surface normals require renderer modification or Omnidata-V2 distillation.)
2. Should the contact / support heads be supervised by (a) Robosuite-extracted contact / support facts (clean but sim-only) or (b) heuristically derived from depth + normal + plane on real datasets (noisier but transfers)? Likely mixed.
3. Do we use a **single** fused expert (action expert sees concatenated geometry-aware embeddings) or **multiple** experts (separate depth / normal / plane / contact / support experts each mirroring QDepth's depth expert)? Architecture trade-off to be settled in PLANNING.
4. What is the ablation matrix? Minimum: −depth, −normal, −plane, −contact, −support, −cross-consistency. Probably 6 ablations × 4 LIBERO suites is the budget envelope.
5. What is the realistic compute budget? PSSA-VLA workspace's Phase 1 baseline sweep cost ≈ 330 RMB on AutoDL dual-A800 (16 + 6 + 6 wallclock h). If we adopt the same, we can target one full sweep + one ablation sweep within ~1000 RMB.
