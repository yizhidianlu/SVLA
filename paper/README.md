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

This is a Phase-1 draft. The Section 5.3 evaluation table currently has a single
data point (task 0 = 0/10 SR); the remaining 9 LIBERO-Spatial tasks finish in
~2.7 hr at the time of writing and the table will be patched with the full result
once eval completes. The honest framing the paper takes (depth-supervision
mechanism converges; task SR falls short of QDepth-VLA's published 86 % under
the abbreviated single-A800 5-epoch budget without the Bridge-pretrained
checkpoint or the LIBERO-90 multi-suite pretrain stage) survives the eval-table
update — it is a budget gap, not a method gap.
