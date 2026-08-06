# Clean held-out results summary (2026-08-06)

All numbers under the **leak-free held-out protocol** (`data.heldout_test=True`):
outer StratifiedGroupKFold → true patient-disjoint test (test≠val, no magic_move,
held-out set not over-sampled), inner split → val for early stopping. 3 folds.
Aggregated from saved softmax with `analysis/aggregate_heldout.py` (no re-inference).
Clip level (~1,954 gait clips) is the granularity comparable to the group's prior
3-class work. **Prec/F1 are macro-averaged over the 3 classes.** Supersedes the
archived leaky-protocol numbers (`analysis/archive_leaky_2026-08-05/`,
`logs/archive/leaky_protocol_2026-08-05/`).

## ⚠️ Accuracy vs macro-F1 gap
Under the real 3-class imbalance, **macro-F1 sits far below accuracy** (best model:
Acc 69.2 but F1 48.8). Accuracy is buoyed by the majority (ASD) class; macro-F1 is
the honest headline. Report both; do not headline accuracy alone.

## Main comparison (clip, mean ± std over 3 folds, %)

| method | Acc | macro-Prec | macro-F1 |
|---|---|---|---|
| RGB baseline (fuse=none) | 52.6 ± 9.4 | 31.3 ± 12.7 | 27.9 ± 7.9 |
| early concat | 53.7 ± 8.0 | 40.1 ± 8.5 | 31.0 ± 5.8 |
| PoseGated multi-[0,1] | 60.3 ± 15.9 | 41.4 ± 18.1 | 38.8 ± 14.9 |
| **PoseGated single-L3** | **69.2 ± 5.5** | **51.6 ± 5.3** | **48.8 ± 4.2** |

single-L3 leads on all three metrics and has the lowest variance. Fusion
competitors (early add/mul/avg, late, SE prefix0-4, cross L3/L4/L34) are being
re-run clean in array 892635[]; their clip Acc/Prec/F1 will be added on completion.

## A5 — fusion location / depth (clip, %)

| single | Acc | F1 | multi prefix | Acc | F1 |
|---|---|---|---|---|---|
| [0] | 53.1 ± 16.7 | 30.9 | [0,1]     | 60.3 ± 15.9 | 38.8 |
| [1] | 56.6 ± 9.2  | 36.0 | [0,1,2]   | 61.5 ± 17.4 | 40.8 |
| [2] | 59.8 ± 13.9 | 38.2 | [0,1,2,3] | 58.5 ± 15.6 | 38.7 |
| **[3]** | **69.2 ± 5.5** | **48.8** | [0,1,2,3,4] | 61.1 ± 14.3 | 40.3 |
| [4] | 45.3 ± 6.1  | 24.2 |           |         |      |

Finding: single-layer accuracy rises with depth to a **peak at L3**, then collapses
at L4 (deepest/final stage). All multi-prefix variants cluster ~58-61% (F1 ~39-41)
with high variance and are worse than single-L3. **Inverts** the leaky conclusion
("shallow [0,1] best, deep/full worst"). single-L3 is also lowest-variance (±5.5).

## A2/A3/A4/A6 — gate-bias, loss, gate-mechanism ablations (clip, %)

Reference = single-L3 (b=2.0, all losses, learned gate) = Acc 69.2 / F1 48.8.

| change | Acc | F1 | ΔAcc |
|---|---|---|---|
| gate bias b=0.0 | 60.3 ± 8.4 | 41.3 | −8.9 |
| gate bias b=−1.0 | 62.3 ± 7.2 | 43.6 | −6.9 |
| − bg loss | 57.5 ± 12.6 | 37.0 | −11.7 |
| − tmp loss | 61.7 ± 13.0 | 41.8 | −7.5 |
| − side heads | 59.4 ± 13.6 | 37.5 | −9.8 |
| gate → add (plain injection) | 56.1 ± 12.0 | 32.7 | −13.1 |
| gate → fixed (0.5 mix) | 57.2 ± 9.1 | 40.6 | −12.0 |

Finding: **every** single-variable change lowers Acc and F1, so each default choice
of single-L3 is justified. The **learned gate is essential** (gated 69.2/48.8 ≫
add 56.1/32.7, fixed 57.2/40.6; the leaky "gate is secondary" result also inverts).
The RGB-biased init b=2.0 is best; bg/tmp losses and side heads all help.

## Caveats
- 3 folds; several configs have high std (bg, gate-mech, all multi) — differences
  are directionally consistent but not all individually significant. fold-2 is the
  hardest split across every config (LCS-HipOA–heavy test set).
- necessity (ROI perturbation) and attention–ROI alignment are analysis re-runs on
  the clean single-L3 checkpoints, still pending.

## Reproduce
```
python -m analysis.aggregate_heldout data.root_path=/work/SKIING/chenkaixu/data/asd_dataset
```
