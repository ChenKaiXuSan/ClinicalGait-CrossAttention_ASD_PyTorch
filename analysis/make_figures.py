"""
File: analysis/make_figures.py
Created: 2026-07-27
-----
Generate publication figures for the paper from the completed experiment matrix.
Outputs vector PDFs to paper/figures/:
  fig_layers.pdf     A5: accuracy vs fusion location/depth (single vs multi)
  fig_methods.pdf    A1: fusion-method comparison
  fig_ablation.pdf   A2/A3/A4: gate-bias & loss/component ablations vs full
  fig_confusion.pdf  main model (multi-[0,1]) confusion matrix, 3 folds aggregated
  fig_alignment.pdf  interpretability: attention-vs-ROI correlation by side head

Design: Okabe-Ito colorblind-safe palette; identity is carried by marker/linestyle
+ direct labels as well as colour (grayscale/print/CVD-safe); recessive grid/spines.

Run (asd env, no GPU needed):  python -m analysis.make_figures
"""

from __future__ import annotations

import os
import re
import ast
import glob
import statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "logs/train"
OUT = "paper/figures"

# Okabe-Ito colorblind-safe palette
OI = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "sky": "#56B4E9", "purple": "#CC79A7",
    "yellow": "#F0E442", "grey": "#999999", "ink": "#222222",
}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150,
})


def _accs():
    """method -> list of per-fold POOLED test accuracy (%), from best_preds.

    (The logged test/video_acc is per-batch macro averaged and under-reports; we
    pool the saved per-sample predictions instead — same source as the paper.)
    """
    import torch
    a = defaultdict(list)
    for pf in glob.glob(f"{LOG}/*_f[0-9]/*/*/best_preds/fold_*_pred.pt"):
        meth = re.sub(r"_f[0-9]+$", "", pf.split(f"{LOG}/")[1].split("/")[0])
        try:
            p = torch.load(pf, map_location="cpu").numpy()
            y = torch.load(pf.replace("_pred.pt", "_label.pt"), map_location="cpu").numpy()
        except Exception:
            continue
        pred = p.argmax(1) if p.ndim > 1 else p
        a[meth].append(100 * float((pred == y).mean()))
    return a


def _ms(vals):
    return (sum(vals) / len(vals), st.pstdev(vals) if len(vals) > 1 else 0.0)


def _finish(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{name}")


# ---------------------------------------------------------------- Fig: layers
def fig_layers(a):
    xs = [0, 1, 2, 3, 4]
    single = [_ms(a[f"pose_atn_single_L{i}"]) for i in xs]
    multi_keys = ["pose_atn_single_L0", "pose_atn_multi_P1", "pose_atn_multi_P2",
                  "pose_atn_multi_P3", "pose_gated_full"]  # [0],[0,1],[0-2],[0-3],[0-4]
    multi = [_ms(a[k]) for k in multi_keys]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.errorbar(xs, [m for m, _ in single], yerr=[s for _, s in single],
                marker="o", ls="-", color=OI["blue"], capsize=2, lw=1.6,
                ms=5, label="single $[i]$")
    ax.errorbar(xs, [m for m, _ in multi], yerr=[s for _, s in multi],
                marker="s", ls="--", color=OI["vermillion"], capsize=2, lw=1.6,
                ms=5, label="multi $[0..i]$")
    base = _ms(a["baseline_rgb"])[0]
    ax.axhline(base, color=OI["grey"], ls=":", lw=1.2)
    ax.text(0.0, base + 0.35, "RGB baseline", color=OI["grey"], fontsize=7,
            ha="left", va="bottom")
    ax.set_xlabel("fusion stage index $i$ (0=stem .. 4=layer4)")
    ax.set_ylabel("video accuracy (%)")
    ax.set_xticks(xs)
    ax.legend(frameon=False, loc="lower right", ncol=1)
    ax.set_ylim(88, 97)
    _finish(fig, "fig_layers.pdf")


# --------------------------------------------------------------- Fig: methods
def fig_methods(a):
    rows = [
        ("RGB baseline", a["baseline_rgb"], OI["grey"]),
        ("Early (mul)", a["early_mul"], OI["ink"]),
        ("Cross-attn $[3,4]$", a["cross_atn_L34"], OI["ink"]),
        ("SE (best)", a["se_atn_prefix0"], OI["ink"]),
        ("Cross-attn $[4]$", a["cross_atn_L4"], OI["ink"]),
        ("Early (add)", a["early_add"], OI["ink"]),
        ("Early (concat)", a["early_concat"], OI["ink"]),
        ("PoseGated (ours)", a["pose_atn_multi_P1"], OI["blue"]),
    ]
    rows = [(n, _ms(v), c) for n, v, c in rows if v]
    rows.sort(key=lambda r: r[1][0])
    names = [r[0] for r in rows]
    means = [r[1][0] for r in rows]
    errs = [r[1][1] for r in rows]
    cols = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    y = np.arange(len(names))
    ax.barh(y, means, xerr=errs, color=cols, height=0.66, capsize=2,
            error_kw=dict(lw=0.8))
    for yi, m in zip(y, means):
        ax.text(m + 1.0, yi, f"{m:.1f}", va="center", fontsize=7, color=OI["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("video accuracy (%)")
    ax.set_xlim(86, 97)
    ax.grid(axis="y", visible=False)
    _finish(fig, "fig_methods.pdf")


# -------------------------------------------------------------- Fig: ablation
def fig_ablation(a):
    ref = _ms(a["pose_gated_full"])[0]
    items = [
        ("bias$=0$", a["pose_gated_bias0"]),
        ("bias$=-1$", a["pose_gated_biasneg1"]),
        ("$-$ bg loss", a["pose_gated_nobg"]),
        ("$-$ tmp loss", a["pose_gated_notmp"]),
        ("$-$ side head", a["pose_gated_noside"]),
        ("full (ref.)", a["pose_gated_full"]),
    ]
    items = [(n, _ms(v)) for n, v in items if v]
    names = [n for n, _ in items]
    means = [m for _, (m, _) in items]
    errs = [s for _, (_, s) in items]
    cols = [OI["green"] if m > ref else (OI["grey"] if abs(m - ref) < 1e-6 else OI["vermillion"])
            for m in means]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=errs, color=cols, width=0.66, capsize=2,
           error_kw=dict(lw=0.8))
    ax.axhline(ref, color=OI["grey"], ls="--", lw=1.1)
    ax.text(len(names) - 0.5, ref + 0.3, "full", color=OI["grey"], fontsize=7, ha="right")
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.4, f"{m:.1f}", ha="center", fontsize=7, color=OI["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("video accuracy (%)")
    ax.set_ylim(88, 96)
    ax.grid(axis="x", visible=False)
    _finish(fig, "fig_ablation.pdf")


# ------------------------------------------------------------- Fig: confusion
def fig_confusion():
    import torch
    classes = ["ASD", "DHS", "LCS-HipOA"]
    P, Y = [], []
    for f in (0, 1, 2):
        pd = glob.glob(f"{LOG}/pose_atn_multi_P1_f{f}/*/*/best_preds/fold_{f}_pred.pt")
        yd = glob.glob(f"{LOG}/pose_atn_multi_P1_f{f}/*/*/best_preds/fold_{f}_label.pt")
        if not pd or not yd:
            continue
        p = torch.load(pd[0], map_location="cpu")
        y = torch.load(yd[0], map_location="cpu")
        p = p.numpy() if hasattr(p, "numpy") else np.asarray(p)
        y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
        P.append(p.argmax(1) if p.ndim > 1 else p)
        Y.append(y)
    if not P:
        print("skip confusion: no pred/label dumps found")
        return
    P = np.concatenate(P); Y = np.concatenate(Y)
    cm = np.zeros((3, 3))
    for t, pr in zip(Y, P):
        cm[int(t), int(pr)] += 1
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else OI["ink"], fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("row-normalized (%)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    _finish(fig, "fig_confusion.pdf")


# ------------------------------------------------------------- Fig: alignment
def fig_alignment():
    import csv
    p = "analysis/alignment_out/alignment_summary.csv"
    if not os.path.exists(p):
        print("skip alignment: no summary")
        return
    rows = list(csv.reader(open(p)))
    hdr = rows[0]
    ci = hdr.index("cc")  # first 'cc' column = mean
    cc = defaultdict(dict)
    for r in rows[3:]:
        if len(r) > ci and r[0] in ("ASD", "DHS", "LCS_HipOA") and r[ci]:
            cc[r[0]][int(r[1])] = float(r[ci])
    if not cc:
        print("skip alignment: no parsed rows")
        return
    layers = sorted({L for c in cc.values() for L in c})
    cls = ["ASD", "DHS", "LCS_HipOA"]
    colmap = {"ASD": OI["blue"], "DHS": OI["orange"], "LCS_HipOA": OI["green"]}
    mk = {"ASD": "o", "DHS": "s", "LCS_HipOA": "^"}
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for c in cls:
        ys = [cc[c].get(L, np.nan) for L in layers]
        ax.plot(layers, ys, marker=mk[c], ls="-", color=colmap[c], lw=1.6, ms=5,
                label=c.replace("_", "-"))
    ax.set_xlabel("side-head stage index")
    ax.set_ylabel("attention-vs-ROI correlation (CC)")
    ax.set_xticks(layers)
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0.3, 0.85)
    _finish(fig, "fig_alignment.pdf")


def main():
    a = _accs()
    fig_layers(a)
    fig_methods(a)
    fig_ablation(a)
    fig_confusion()   # best_preds are the correct pooled predictions (now the paper's source)
    fig_alignment()
    print("done")


if __name__ == "__main__":
    main()
