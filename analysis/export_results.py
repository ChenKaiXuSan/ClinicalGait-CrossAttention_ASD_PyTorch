"""
File: analysis/export_results.py
Created Date: 2026-07-26
Author: Kaixu Chen
-----
Comment:
 Aggregate all per-fold test metrics into a paper-ready summary.
 Parses logs/train/<tag>_f<fold>/*/*/metrics/fold_*_metrics.txt, computes
 3-fold mean±std per method, groups by the paper's A0-A5 structure, and writes:
   analysis/results_summary.csv   (flat: group,method,fold0/1/2,mean,std for acc+f1)
   analysis/results_summary.md    (grouped markdown tables)

 Handles both metric key variants: `test/video_acc` and `test/video_acc_epoch`
 (baseline trainer logs the `_epoch` suffix; others don't).

Usage:
    python -m analysis.export_results
-----
"""

from __future__ import annotations

import os
import re
import ast
import glob
import statistics as st
from collections import defaultdict

LOG_ROOT = "logs/train"
OUT_DIR = "analysis"

# paper grouping: (group_label, [(method_tag, display_name), ...])
GROUPS = [
    ("A0 Baseline", [("baseline_rgb", "B1 RGB-only (fuse=none)")]),
    ("A1 Fusion method", [
        ("early_add", "Early add"), ("early_mul", "Early mul"), ("early_concat", "Early concat"),
        ("se_atn_prefix0", "SE [0]"), ("se_atn_prefix1", "SE [0,1]"), ("se_atn_prefix2", "SE [0-2]"),
        ("se_atn_prefix3", "SE [0-3]"), ("se_atn_prefix4", "SE [0-4]"),
        ("cross_atn_L3", "Cross [3]"), ("cross_atn_L4", "Cross [4]"), ("cross_atn_L34", "Cross [3,4]"),
        ("pose_gated_full", "PoseGated full [0-4]"),
    ]),
    ("A2 Gate init bias (full [0-4])", [
        ("pose_gated_bias0", "bias=0.0"), ("pose_gated_biasneg1", "bias=-1.0"),
        ("pose_gated_full", "bias=2.0 (=full)"),
    ]),
    ("A3/A4 Component & loss ablation (full [0-4])", [
        ("pose_gated_full", "full (reference)"), ("pose_gated_noside", "- side head"),
        ("pose_gated_nobg", "- bg loss"), ("pose_gated_notmp", "- tmp loss"),
    ]),
    ("A5 Fusion layers - single", [(f"pose_atn_single_L{i}", f"single L{i}") for i in range(5)]),
    ("A5 Fusion layers - multi", [
        ("pose_atn_multi_P1", "multi [0,1]"), ("pose_atn_multi_P2", "multi [0,1,2]"),
        ("pose_atn_multi_P3", "multi [0-3]"), ("pose_gated_full", "multi [0-4] (=full)"),
    ]),
    ("Extra (post-hoc)", [("pose_gated_bestcombo", "bestcombo: multi[0,1]+bias0+no bg/tmp")]),
]

METRIC_KEYS = ["video_acc", "video_precision", "video_recall", "video_f1_score"]


def _collect():
    """method -> metric -> {fold: value}, computed POOLED from best_preds.

    The logged test/video_acc metric is per-batch macro accuracy averaged over
    batches (average='macro' + logging the batch value), which under-reports by
    ~5-9 points. We instead pool the saved per-sample predictions per fold and
    compute standard pooled accuracy (micro) and macro precision/recall/F1.
    """
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score
    data = defaultdict(lambda: defaultdict(dict))
    for pf in glob.glob(f"{LOG_ROOT}/*_f[0-9]/*/*/best_preds/fold_*_pred.pt"):
        tag = pf.split(f"{LOG_ROOT}/")[1].split("/")[0]
        mo = re.search(r"_f([0-9]+)$", tag)
        if not mo:
            continue
        method, fold = tag[: mo.start()], mo.group(1)
        try:
            p = torch.load(pf, map_location="cpu").numpy()
            y = torch.load(pf.replace("_pred.pt", "_label.pt"), map_location="cpu").numpy()
        except Exception:
            continue
        pred = p.argmax(1) if p.ndim > 1 else p
        data[method]["video_acc"][fold] = float((pred == y).mean())
        data[method]["video_precision"][fold] = float(precision_score(y, pred, average="macro", zero_division=0))
        data[method]["video_recall"][fold] = float(recall_score(y, pred, average="macro", zero_division=0))
        data[method]["video_f1_score"][fold] = float(f1_score(y, pred, average="macro"))
    return data


def _stats(folddict):
    vals = [folddict[f] for f in sorted(folddict)]
    if not vals:
        return None, None, vals
    mean = sum(vals) / len(vals)
    sd = st.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, sd, vals


def main():
    data = _collect()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- CSV (flat) ---- use csv.writer so display names with commas
    # (e.g. "multi [0,1]") are properly quoted.
    import csv
    csv_path = os.path.join(OUT_DIR, "results_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "method", "display", "n_folds",
                    "acc_fold0", "acc_fold1", "acc_fold2",
                    "acc_mean", "acc_std", "f1_mean"])
        for group, methods in GROUPS:
            for tag, disp in methods:
                am = data.get(tag, {}).get("video_acc", {})
                mean, sd, _ = _stats(am)
                if mean is None:
                    w.writerow([group, tag, disp, 0, "", "", "", "", "", ""])
                    continue
                a = [am.get(str(i), "") for i in range(3)]
                fm, _, _ = _stats(data.get(tag, {}).get("video_f1_score", {}))
                w.writerow([group, tag, disp, len(am), *[f"{v:.4f}" if v != "" else "" for v in a],
                            f"{mean:.4f}", f"{sd:.4f}",
                            f"{fm:.4f}" if fm is not None else ""])

    # ---- Markdown (grouped) ----
    md_path = os.path.join(OUT_DIR, "results_summary.md")
    with open(md_path, "w") as f:
        f.write("# ASD 实验矩阵结果汇总（3-fold，有效重跑）\n\n")
        f.write("acc = test/video_acc，mean±std over 3 folds。⧗ 表示未满 3 折。\n\n")
        for group, methods in GROUPS:
            f.write(f"## {group}\n\n| Method | acc (mean±std) | f1 | folds |\n|---|---|---|---|\n")
            for tag, disp in methods:
                am = data.get(tag, {}).get("video_acc", {})
                mean, sd, vals = _stats(am)
                if mean is None:
                    f.write(f"| {disp} | — | — | 0/3 |\n")
                    continue
                fm, _, _ = _stats(data.get(tag, {}).get("video_f1_score", {}))
                warn = "" if len(am) == 3 else f" ⧗"
                folds = "/".join(f"{v:.3f}" for v in vals)
                f.write(f"| {disp} | {mean*100:.1f} ± {sd*100:.1f}{warn} | "
                        f"{fm*100:.1f} | {folds} |\n")
            f.write("\n")

    # ---- console echo ----
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    # quick top-5
    ranked = []
    for tag in {t for _, ms in GROUPS for t, _ in ms}:
        mean, _, _ = _stats(data.get(tag, {}).get("video_acc", {}))
        if mean is not None:
            ranked.append((mean, tag))
    print("Top-5 by acc:")
    for mean, tag in sorted(ranked, reverse=True)[:5]:
        print(f"  {tag:24} {mean*100:.1f}%")


if __name__ == "__main__":
    main()
