"""
File: analysis/mask_faces_fig.py
Created: 2026-09-06
-----
Post-process the cached clips/cams of make_combined_fig (COMB_CACHE pickle) and
re-render fig_qual_combined with the patient's face pixelated in the underlying
RGB frame. Only the base frame is touched: every heat overlay (Grad-CAM,
supervised side-head map, doctor ROI) is drawn on top unchanged.

Head box = top `head_frac` of the person's silhouette (frames are person-on-black),
found automatically; pass --skip CLASS to leave a row untouched when the face is
not visible (e.g. back view).

Run (repo root, asd env, CPU):
    python -m analysis.mask_faces_fig --cache analysis/figures_out/comb_cache_fold0.pkl \
        [--head-frac 0.14] [--block 6] [--skip DHS]
"""
from __future__ import annotations
import argparse, os, pickle, shutil
import numpy as np


def head_box(rgb: np.ndarray, head_frac: float, pad: int = 2):
    sil = rgb.max(axis=2) > 0.08                      # person on black background
    rows = np.where(sil.any(axis=1))[0]
    if rows.size == 0:
        return None
    y0, y1 = rows[0], rows[-1]
    hy1 = int(y0 + head_frac * (y1 - y0 + 1))
    cols = np.where(sil[y0:hy1 + 1].any(axis=0))[0]
    if cols.size == 0:
        return None
    H, W = sil.shape
    return (max(0, y0 - pad), min(H, hy1 + pad + 1), max(0, cols[0] - pad), min(W, cols[-1] + pad + 1))


def pixelate(rgb: np.ndarray, box, block: int):
    y0, y1, x0, x1 = box
    out = rgb.copy()
    for y in range(y0, y1, block):
        for x in range(x0, x1, block):
            patch = out[y:min(y + block, y1), x:min(x + block, x1)]
            patch[...] = patch.reshape(-1, 3).mean(axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--head-frac", type=float, default=0.14)
    ap.add_argument("--block", type=int, default=6)
    ap.add_argument("--skip", nargs="*", default=[], help="class names whose row is left unmasked")
    ap.add_argument("--out", default="analysis/figures_out/fig_qual_combined_facemask.pdf")
    a = ap.parse_args()

    from analysis.make_combined_fig import render, OUT
    clips, cams = pickle.load(open(a.cache, "rb"))
    for cls, c in clips.items():
        if cls in a.skip:
            print(f"{cls}: skipped (face not visible)"); continue
        box = head_box(c["rgb"], a.head_frac)
        if box is None:
            print(f"{cls}: no silhouette found, skipped"); continue
        c["rgb"] = pixelate(c["rgb"], box, a.block)
        print(f"{cls}: pixelated head box rows {box[0]}-{box[1]}, cols {box[2]}-{box[3]} (block {a.block}px)")
    render(clips, cams)                                # writes OUT/fig_qual_combined.pdf
    shutil.move(f"{OUT}/fig_qual_combined.pdf", a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
