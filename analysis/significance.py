"""
File: analysis/significance.py
Created: 2026-07-28
-----
Statistical significance for the head-line comparisons (reviewer ask #3).

The saved best_preds are PAIRED across methods within a fold (shared
StratifiedGroupKFold split, deterministic test order, identical length), so for
any two methods we can compare their predictions clip-by-clip and run:

  * McNemar's test on the pooled paired correct/incorrect table (the standard
    paired test for two classifiers on the same samples), with continuity
    correction; and
  * a clip-level bootstrap 95% CI on the accuracy difference dAcc.

Both are CLIP-level (consistent with the paper's reported metric); we state this
so the CIs are not read as subject-level. We also report the 3 paired per-fold
accuracies (n=3 is small -> descriptive only, no fold-level p-value claimed).

Usage:  python -m analysis.significance
Writes analysis/significance.md (+ prints a LaTeX-ready table).
"""

from __future__ import annotations

import glob
import numpy as np

LOG = "logs/train"
COMPARISONS = [
    ("PoseGated multi-[0,1]", "pose_atn_multi_P1", "RGB baseline",        "baseline_rgb"),
    ("PoseGated multi-[0,1]", "pose_atn_multi_P1", "Early concat",        "early_concat"),
    ("PoseGated multi-[0,1]", "pose_atn_multi_P1", "SE [0]",              "se_atn_prefix0"),
    ("PoseGated multi-[0,1]", "pose_atn_multi_P1", "Cross-attn [4]",      "cross_atn_L4"),
    ("PoseGated multi-[0,1]", "pose_atn_multi_P1", "PoseGated full [0-4]","pose_gated_full"),
]


def _load(tag):
    """pooled (pred, label) across folds; per-fold list of correctness arrays."""
    import torch
    preds, labs, per_fold = [], [], []
    for f in (0, 1, 2):
        pf = glob.glob(f"{LOG}/{tag}_f{f}/*/*/best_preds/fold_{f}_pred.pt")
        lf = glob.glob(f"{LOG}/{tag}_f{f}/*/*/best_preds/fold_{f}_label.pt")
        if not pf or not lf:
            per_fold.append(None); continue
        p = torch.load(pf[0], map_location="cpu").numpy()
        y = torch.load(lf[0], map_location="cpu").numpy()
        p = p.argmax(1) if p.ndim > 1 else p
        preds.append(p); labs.append(y); per_fold.append((p == y).astype(np.int8))
    return np.concatenate(preds), np.concatenate(labs), per_fold


def _mcnemar(correct_a, correct_b):
    """McNemar with continuity correction on paired correctness vectors."""
    from scipy.stats import chi2
    b = int(np.sum((correct_a == 1) & (correct_b == 0)))   # A right, B wrong
    c = int(np.sum((correct_a == 0) & (correct_b == 1)))   # A wrong, B right
    if b + c == 0:
        return 0.0, 1.0, b, c
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(stat), float(chi2.sf(stat, 1)), b, c


def _bootstrap_ci(correct_a, correct_b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(correct_a)
    diff = correct_a.astype(float) - correct_b.astype(float)
    idx = rng.integers(0, N, size=(n, N))
    boot = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    rows = []
    for na, ta, nb, tb in COMPARISONS:
        pa, ya, fa = _load(ta)
        pb, yb, fb = _load(tb)
        assert np.array_equal(ya, yb), f"label mismatch {ta} vs {tb} (unpaired!)"
        ca, cb = (pa == ya).astype(np.int8), (pb == yb).astype(np.int8)
        acc_a, acc_b = ca.mean() * 100, cb.mean() * 100
        stat, p, b, c = _mcnemar(ca, cb)
        d, lo, hi = _bootstrap_ci(ca, cb)
        # per-fold paired accuracies
        pf = [(fa[i].mean() * 100, fb[i].mean() * 100) for i in range(3) if fa[i] is not None and fb[i] is not None]
        rows.append(dict(a=na, b=nb, acc_a=acc_a, acc_b=acc_b, dacc=d * 100,
                         lo=lo * 100, hi=hi * 100, mcnemar=stat, p=p, disc_b=b, disc_c=c, pf=pf))

    with open("analysis/significance.md", "w") as f:
        f.write("# Statistical significance (clip-level, paired)\n\n")
        f.write("McNemar (continuity-corrected) on pooled paired clips; bootstrap 95% CI on "
                "clip-level dAcc (5000 resamples). CLIP-level — not subject-level.\n\n")
        f.write("| Comparison (A vs B) | acc A | acc B | dAcc (A-B) | 95% CI | McNemar chi2 | p |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            sig = "**" if (r["lo"] > 0 or r["hi"] < 0) else ""
            f.write(f"| {r['a']} vs {r['b']} | {r['acc_a']:.1f} | {r['acc_b']:.1f} | "
                    f"{sig}{r['dacc']:+.1f}{sig} | [{r['lo']:+.1f}, {r['hi']:+.1f}] | "
                    f"{r['mcnemar']:.1f} | {r['p']:.2e} |\n")
        f.write("\n**bold** dAcc = 95% CI excludes 0.\n\n")
        f.write("Per-fold paired accuracies (A / B):\n\n")
        for r in rows:
            folds = "  ".join(f"f{i}: {a:.1f}/{b:.1f}" for i, (a, b) in enumerate(r["pf"]))
            f.write(f"- {r['a']} vs {r['b']}: {folds}\n")

    print("wrote analysis/significance.md\n")
    print(f"{'Comparison':42} {'dAcc':>7} {'95% CI':>18} {'p(McNemar)':>12}")
    for r in rows:
        print(f"{r['a']+' vs '+r['b']:42} {r['dacc']:+6.1f}  "
              f"[{r['lo']:+.1f},{r['hi']:+.1f}]   {r['p']:.2e}"
              f"   (b={r['disc_b']},c={r['disc_c']})")


if __name__ == "__main__":
    main()
