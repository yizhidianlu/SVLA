# GeoRel-VLA — Phase-1 paper draft

LaTeX source for the Phase-1 implementation-study paper.
Companion to the codebase in this repository.

## Layout

- `main.tex` — paper source
- `refs.bib` — bibliography (~22 references; covers the depth-aware-VLA cluster + VQ-VAE prior art)
- `figures/` — PDF figures generated from training/eval metrics, plus the matplotlib script that produces them
  - `fig_vqvae_curves.pdf` — VQ-VAE pretrain: depth recon RMSE + active-codes-out-of-256
  - `fig_train_curves.pdf` — main training: depth-CE + action-CFM + total losses over the 19,455-step run
  - `fig_codebook_compare.pdf` — vanilla L2 codebook (collapses to 6/256) vs. EMA + dead-code reset (250/256)
  - `gen_figures.py` — re-generate from `*_metrics.json`

## Build

```bash
cd paper
# the .tex falls back to article-class margins if neurips_2024.sty is absent;
# for the official NeurIPS look, drop neurips_2024.sty into this dir first.
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Status (2026-05-12)

GeoRel-VLA Phase-1 paper draft. Target metric: LIBERO-Spatial overall SR ≥ 84%
(matching/approaching QDepth-VLA's reported 86.0 %).

The final numbers in §5.3 (Table~\ref{tab:ablation}) come from the Stage-C
training run currently in flight — 3× A800 DDP, effective batch 1024, bridge_beta
warmstart + 20-epoch LIBERO-90 pretrain + 50-epoch LIBERO-Spatial finetune,
evaluated at 50 rollouts/task under float32 inference. v1 and v2 entries in
the same ablation table come from the abbreviated-recipe predecessor runs
(no warmstart, no LIBERO-90 pretrain) and serve to isolate the contribution
of recipe vs. warmstart-plus-pretrain vs. depth-CE auxiliary head.

Placeholders `XX.X` mark the eval-derived numbers; they are filled in
automatically once `eval_libero.py --rollouts 50 --precision fp32` completes
on the Stage-C checkpoint.
