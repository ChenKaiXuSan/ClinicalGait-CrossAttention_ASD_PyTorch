# Clean held-out results summary (2026-08-06)

All numbers under the **leak-free held-out protocol** (`data.heldout_test=True`):
outer StratifiedGroupKFold → true patient-disjoint test (test≠val, no magic_move,
held-out set not over-sampled), inner split → val for early stopping. 3 folds.
Aggregated from saved softmax with `analysis/aggregate_heldout.py` (no re-inference).
Clip level (~1,954 gait clips) is the granularity comparable to the group's prior
3-class work. Supersedes the archived leaky-protocol numbers
(`analysis/archive_leaky_2026-08-05/`, `logs/archive/leaky_protocol_2026-08-05/`).

## Accuracy by granularity (mean ± std over 3 folds, %)

| method | chunk (~42k) | clip (~1954) | patient (81) |
|---|---|---|---|
| RGB baseline (fuse=none) | 53.8 ± 8.0 | 52.6 ± 9.4 | 67.1 ± 9.5 |
| early concat | 56.1 ± 6.4 | 53.7 ± 8.0 | 74.0 ± 4.4 |
| **PoseGated single-L3** | **68.9 ± 5.9** | **69.2 ± 5.5** | **80.2 ± 4.4** |

## A5 — fusion location / depth (clip, %)

| single | Acc | multi prefix | Acc |
|---|---|---|---|
| [0] | 53.1 ± 16.7 | [0,1]     | 60.3 ± 15.9 |
| [1] | 56.6 ± 9.2  | [0,1,2]   | 61.5 ± 17.4 |
| [2] | 59.8 ± 13.9 | [0,1,2,3] | 58.5 ± 15.6 |
| **[3]** | **69.2 ± 5.5** | [0,1,2,3,4] | 61.1 ± 14.3 |
| [4] | 45.3 ± 6.1  |           |         |

Finding: single-layer accuracy rises with depth to a **peak at L3**, then collapses
at L4 (deepest/final stage). All multi-prefix variants cluster at 58–61% with high
variance and are worse than single-L3. **Inverts** the leaky-protocol conclusion
("shallow [0,1] best, deep/full worst"). single-L3 is also the lowest-variance
config (±5.5).

## A2/A3/A4/A6 — gate-bias, loss, gate-mechanism ablations (clip, %)

Reference = single-L3 (b=2.0, all losses, learned gate) = 69.2 ± 5.5.

| change | Acc | Δ |
|---|---|---|
| gate bias b=0.0 | 60.3 ± 8.4 | −8.9 |
| gate bias b=−1.0 | 62.3 ± 7.2 | −6.9 |
| − bg loss | 57.5 ± 12.6 | −11.7 |
| − tmp loss | 61.7 ± 13.0 | −7.5 |
| − side heads | 59.4 ± 13.6 | −9.8 |
| gate → add (plain injection) | 56.1 ± 12.0 | −13.1 |
| gate → fixed (0.5 mix) | 57.2 ± 9.1 | −12.0 |

Finding: **every** single-variable change lowers accuracy, so each default choice of
single-L3 is justified. The **learned gate is essential** (gated 69.2 ≫ add 56.1,
fixed 57.2; a 12–13 point gap — the leaky-protocol "gate is secondary" result also
inverts). The RGB-biased init b=2.0 is best; bg/tmp losses and side heads all help.

## Caveats
- 3 folds; several ablations have high std (bg, gate-mech, some multi) — differences
  are directionally consistent but not all individually significant. fold-2 is the
  hardest split across every config (LCS-HipOA–heavy test set).
- necessity (ROI perturbation) and attention–ROI alignment are analysis re-runs on
  the clean single-L3 checkpoints, still pending.
