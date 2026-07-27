# Paper — Cross-Attentive Temporal Fusion with Clinical Priors for ASD

Manuscript for submission to **Medical Image Analysis** (Elsevier).

## Files
- `main.tex` — the manuscript (Elsevier `elsarticle` class, which MIA requires).
- `refs.bib` — bibliography (two related-work entries are placeholders — complete them).
- `figures/` — put figures here (see the figure plan in `../pegasus/EXPERIMENTS.md` §五).

## Template / class
Medical Image Analysis uses Elsevier's official **`elsarticle`** document class.
It is **not vendored here** (this machine has no TeX and no internet); it ships
with every TeX Live/MiKTeX install and is built into Overleaf. To get it standalone:
`https://ctan.org/pkg/elsarticle` (or Elsevier "Your Paper Your Way").

## How to compile
**Overleaf (easiest):** create a project, upload `main.tex` + `refs.bib`, set the
compiler to pdfLaTeX. `elsarticle` and `elsarticle-num.bst` are already available.

**Local TeX Live:**
```bash
latexmk -pdf main.tex          # runs pdflatex + bibtex automatically
# or manually: pdflatex main; bibtex main; pdflatex main; pdflatex main
```
Switch `\documentclass[review,5p,times]{elsarticle}` to `[final,5p,times]` for a
camera-ready two-column build.

## Where the numbers come from
All results/conclusions are transcribed from the completed 3-fold experiment
matrix:
- `../analysis/results_summary.md` / `.csv` — per-method accuracy (mean±std) + F1.
- `../analysis/alignment_out/alignment_summary.csv` — interpretability alignment.
- `../pegasus/EXPERIMENTS.md` §六 — the written conclusions this paper is built on.

## TODO before submission
- Complete author list and affiliations (currently `Kaixu Chen` + placeholder).
- Fill the two placeholder references in `refs.bib`.
- Add figures (architecture, fusion-layer curve, gate-bias, ablation,
  confusion matrix, attention-alignment case study).
- Add dataset ethics/IRB statement and exact patient/clip counts.
- Confirm MIA reference style (this draft uses `elsarticle-num`).
