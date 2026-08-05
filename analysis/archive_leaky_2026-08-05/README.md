# ARCHIVED — leaky-protocol analysis outputs (superseded 2026-08-05)

Analysis artifacts derived from the **old evaluation protocol** (test==val +
magic_move patient leakage + chunk-level counting). **Invalid for reporting.**
Kept for provenance only. The raw training logs they were computed from are in
`logs/archive/leaky_protocol_2026-08-05/` (see its README for the full diagnosis).

## Files here
- `results_summary.{csv,md}`   — old main-comparison result tables (~94.8% headline)
- `attention_perturbation.md`  — old necessity study on multi-[0,1] (real/shuffled/zero)
- `significance.md`            — old McNemar/bootstrap significance
- `attention_similarity.csv`   — old attention-vs-ROI similarity
- `alignment_out/`             — old side-head↔doctor-ROI alignment (from pose_atn_multi_P1)

## NOT archived (still current, left in analysis/)
- `granularity_comparison.md`  — the diagnosis that motivated the clean protocol
- all `*.py` scripts           — reusable; re-run on the clean `heldout_*` logs
- `aggregate_heldout.py`       — clip/patient aggregation for the clean re-run

Superseded by the clean held-out protocol (`data.heldout_test=True`); active
results live under `logs/train/heldout_*`. See `pegasus/EXPERIMENTS.md` §IX.
