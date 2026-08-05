# Accuracy by evaluation granularity (chunk / clip / patient)

## ⭐ CLEAN held-out protocol (job 890178) — USE THESE, supersedes the leaked table below

Re-run of the key methods under the leakage-free protocol (`data.heldout_test=True`):
patient-grouped outer split → true held-out **test ≠ val**, **no `magic_move`**,
hold-out set not over-sampled. Aggregated by `analysis/aggregate_heldout.py`.
mean ± std over 3 folds.

| Method | chunk (~42k) | **clip (~1954)** | patient (81) |
|---|---|---|---|
| RGB baseline            | 53.8 ± 8.0 | 52.6 ± 9.4  | 67.1 ± 9.5 |
| early_concat            | 56.1 ± 6.4 | 53.7 ± 8.0  | 74.0 ± 4.4 |
| PoseGated multi[0,1]    | 59.9 ± 13.3| 60.3 ± 15.9 | 74.4 ± 7.8 |
| **PoseGated single L3** | **68.9 ± 5.9** | **69.2 ± 5.5** | **80.2 ± 4.4** |

**What this settles:**
1. The old ~90% was **inflated by leakage + test==val**, NOT by chunk-vs-clip
   granularity: under the clean protocol clip ≈ chunk (69.2 vs 68.9).
2. Clean **PoseGated (single L3) = 69.2% clip-level** — in range with the group's
   prior 3-class work (PhaseMix 71.4%), but with no patient leakage.
3. **Honest ablation lead: +16.6 clip-level points over the RGB baseline** (69.2 vs 52.6).
4. **The best config changed**: single-L3 now beats the previously-"main" multi[0,1]
   (69.2 vs 60.3) — the manuscript's headline method must switch to single-L3.

The multi-granularity table below was computed on the OLD leaky protocol
(`magic_move`, test==val) and is retained only for the record. **Do not cite it.**

---

## (superseded) Leaky-protocol multi-granularity table

Computed by `analysis/aggregate_levels.py` from the SAVED per-chunk softmax
predictions (best_preds) — no re-inference. Aggregation: mean-probability within
each gait clip (`video_name`) and within each patient key (`video_name` prefix).
mean ± std over the 3 folds.

**Why this matters.** The draft reports **chunk-level** accuracy (~42k 16-frame
units). Prior work of the group (Chen 2023, Front. Neurosci.; PhaseMix, IEEE
Access) evaluates at the **clip level** (one 2–10 s gait clip). The two are not
directly comparable, and the method ranking *changes* with granularity.

| Method | chunk (~42k) | clip (~1954) | patient (81) |
|---|---|---|---|
| RGB baseline            | 90.7 ± 1.9 | 89.4 ± 6.6 | 96.8 ± 2.6 |
| early_concat            | 93.7 ± 2.7 | **94.8 ± 5.0** | **97.0 ± 2.1** |
| early_add               | 91.7 ± 1.5 | 87.3 ± 4.9 | 93.6 ± 1.6 |
| early_mul               | 89.3 ± 5.5 | 88.5 ± 5.6 | 93.8 ± 2.3 |
| se_atn (best prefix[0]) | 92.0 ± 2.4 | 93.9 ± 3.8 | 94.8 ± 0.8 |
| cross_atn L4            | 89.9 ± 3.3 | 91.0 ± 4.2 | 94.0 ± 2.8 |
| cross_atn L3            | 89.9 ± 2.9 | 87.3 ± 8.7 | 93.9 ± 2.3 |
| cross_atn L34           | 90.2 ± 1.4 | 89.3 ± 6.6 | 94.1 ± 3.5 |
| **PoseGated multi[0,1] (main)** | **94.8 ± 1.9** | 90.2 ± 7.4 | 91.5 ± 6.2 |
| PoseGated full [0-4]    | 90.9 ± 3.1 | 89.3 ± 2.2 | 90.2 ± 5.3 |
| PoseGated single L3     | 93.8 ± 1.5 | 93.2 ± 0.3 | 96.1 ± 1.8 |

Chen 2023 (Front. Neurosci.): clip-level, **binary**, 5-fold, **75.53%**.

## Findings that affect the manuscript's claims

1. **The "PoseGated is best" result is chunk-level only.** At chunk level PoseGated
   multi[0,1] leads (94.8 > early_concat 93.7). At the **clip level** it drops to
   90.2 and is *beaten* by early_concat (94.8), SE (93.9) and single-L3 (93.2),
   and is barely above the RGB baseline (89.4). At patient level it is near the
   bottom (91.5 vs early_concat 97.0).

2. **"Shallow multi[0,1] is optimal" also weakens at clip level:** single-L3 (93.2)
   and early_concat (94.8) exceed multi[0,1] (90.2).

3. **High clip-level variance (±5–8, 3 folds)** means most methods are within noise
   of each other; there is no clearly-separated winner at the comparable granularity.

4. **Separate methodological issue:** `data_loader.py:182` sets `test == val`, and
   the checkpoint is selected by `val/video_acc`, so every reported number is
   "best-epoch-on-the-eval-set" (optimistically biased). A true held-out test set
   (train/val/test or nested CV) would require re-training.

5. **What survives regardless of granularity:** the interpretability result
   (supervised attention localises the clinical ROI; the perturbation control shows
   the prior is actually used).
