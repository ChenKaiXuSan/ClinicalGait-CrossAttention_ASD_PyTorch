"""
File: analysis/aggregate_levels.py
Created: 2026-08-05
-----
Report accuracy at three granularities so results are comparable across the
group's papers, which evaluate at the gait-CLIP level (not per chunk, not per
patient):

  chunk   : per 16-frame chunk (~42k units)  -- what the current draft reports
  clip    : per gait-cycle clip / source video_name (~1,954 units)  -- matches
            Chen 2023 (Front. Neurosci.) & the PhaseMix dataset unit
  patient : per subject (~81 units)

Uses the SAVED softmax predictions (best_preds); no re-inference. We rebuild the
deterministic test index per fold to recover each chunk's (video_name, label),
truncate to the saved-prediction length (test drop_last), then aggregate by
mean-probability within each clip / patient.

Run:
    python -m analysis.aggregate_levels \
        data.root_path=/work/SKIING/chenkaixu/data/asd_dataset
"""

from __future__ import annotations

import glob
import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)
METHODS = [  # main-comparison methods, to recompute at the prior work's clip granularity
    "baseline_rgb",
    "early_concat", "early_add", "early_mul",
    "se_atn_prefix0",
    "cross_atn_L4", "cross_atn_L3", "cross_atn_L34",
    "pose_atn_multi_P1",   # main PoseGated (shallow [0,1])
    "pose_gated_full",     # all-stage
    "pose_atn_single_L3",  # best single
]


def _load_probs(tag, fold):
    p = glob.glob(f"logs/train/{tag}_f{fold}/*/*/best_preds/fold_{fold}_pred.pt")
    y = glob.glob(f"logs/train/{tag}_f{fold}/*/*/best_preds/fold_{fold}_label.pt")
    if not p or not y:
        return None, None
    import torch
    return (torch.load(p[0], map_location="cpu").numpy(),
            torch.load(y[0], map_location="cpu").numpy())


def run(config):
    from pytorch_lightning import seed_everything
    from project.cross_validation import DefineCrossValidation
    from project.dataloader.data_loader import WalkDataModule

    seed_everything(42, workers=True)
    fold_idx = DefineCrossValidation(config)()

    # per fold, recover ordered (video_name, label) once (shared across methods)
    order = {}
    for fk in sorted(fold_idx.keys(), key=int):
        fold = int(fk)
        dm = WalkDataModule(config, fold_idx[fk])
        dm.setup("test")
        im = dm.test_dataloader().dataset._index_map
        order[fold] = [(e["video_name"], int(e["label"])) for e in im]
        logger.info(f"[fold {fold}] index rebuilt: {len(im)} chunks")

    def patient_of(vn):
        return vn.split("-")[0]

    from sklearn.metrics import precision_recall_fscore_support

    def _scores(y_true, y_pred):
        """acc / macro-precision / macro-recall / macro-F1, all in %."""
        y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
        acc = (y_true == y_pred).mean() * 100.0
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)
        return np.array([acc, p * 100.0, r * 100.0, f * 100.0])

    results = {}
    for tag in METHODS:
        lvl = {"chunk": [], "clip": [], "patient": []}     # each holds per-fold [acc,P,R,F1]
        for fold, meta in order.items():
            probs, labels = _load_probs(tag, fold)
            if probs is None:
                continue
            n = len(probs)
            meta_n = meta[:n]                       # test drop_last: saved <= index
            # sanity: chunk labels must match the rebuilt order
            lab_meta = np.array([m[1] for m in meta_n])
            assert np.array_equal(lab_meta, labels[:n]), f"label mismatch {tag} fold{fold}"

            # chunk level
            lvl["chunk"].append(_scores(labels[:n], probs.argmax(1)))

            # aggregate mean-prob by clip (video_name) and by patient
            for key_fn, name in ((lambda i: meta_n[i][0], "clip"),
                                 (lambda i: patient_of(meta_n[i][0]), "patient")):
                acc = defaultdict(lambda: [np.zeros(probs.shape[1]), None])
                for i in range(n):
                    k = key_fn(i)
                    acc[k][0] += probs[i]
                    acc[k][1] = labels[i]
                y_true = [int(v[1]) for v in acc.values()]
                y_pred = [int(v[0].argmax()) for v in acc.values()]
                lvl[name].append(_scores(y_true, y_pred))
        # per level: mean / std / max (best fold) / per-fold rows, over the 4-metric vector
        results[tag] = {k: (np.mean(v, axis=0), np.std(v, axis=0),
                            np.max(v, axis=0), np.asarray(v), len(v))
                        for k, v in lvl.items() if v}

    # ---- print tables ----
    # (1) CLIP-level BEST-FOLD (max over folds): Acc / macro-Prec / macro-F1
    #     NB each metric's max is taken independently and may come from different folds.
    print("\n=== CLIP-level BEST-FOLD (max over folds): Acc / macro-Prec / macro-F1 (%) ===")
    print(f"{'method':22} {'Acc':>7} {'Prec':>7} {'F1':>7}")
    for tag in METHODS:
        r = results.get(tag, {}).get("clip")
        if r is None:
            print(f"{tag:22} {'--':>7}"); continue
        _, _, mx, _, _ = r
        print(f"{tag:22} {mx[0]:7.1f} {mx[1]:7.1f} {mx[3]:7.1f}")

    # (1b) CLIP-level BEST fold by ACCURACY — report THAT fold's Acc/Prec/F1
    #      (all three metrics from the SAME fold; self-consistent). Also show which fold.
    print("\n=== CLIP-level BEST-ACC FOLD (same fold, self-consistent): Acc / Prec / F1 (%) ===")
    print(f"{'method':22} {'fold':>4} {'Acc':>7} {'Prec':>7} {'F1':>7}")
    for tag in METHODS:
        r = results.get(tag, {}).get("clip")
        if r is None:
            print(f"{tag:22} {'--':>4}"); continue
        _, _, _, rows, _ = r          # rows: (n_folds, 4) = [acc, prec, rec, f1]; row i == fold i
        bi = int(np.argmax(rows[:, 0]))
        print(f"{tag:22} {bi:>4} {rows[bi,0]:7.1f} {rows[bi,1]:7.1f} {rows[bi,3]:7.1f}")

    # (2) CLIP-level mean±std (kept for reference)
    print("\n=== CLIP-level (mean±std over folds): Acc / macro-Prec / macro-F1 (%) ===")
    print(f"{'method':22} {'Acc':>13} {'Prec':>13} {'F1':>13}")
    for tag in METHODS:
        r = results.get(tag, {}).get("clip")
        if r is None:
            print(f"{tag:22} {'--':>13}"); continue
        m, s, mx, rows, _ = r
        print(f"{tag:22} {m[0]:5.1f} ± {s[0]:4.1f} {m[1]:5.1f} ± {s[1]:4.1f} {m[3]:5.1f} ± {s[3]:4.1f}")

    # (3) accuracy across granularities (chunk / clip / patient), best fold
    print("\n=== BEST-FOLD accuracy by granularity (max over folds, %) ===")
    print(f"{'method':22} {'chunk (~42k)':>14} {'clip (~1954)':>14} {'patient (81)':>14}")
    for tag in METHODS:
        r = results.get(tag, {})
        cells = []
        for lv in ("chunk", "clip", "patient"):
            if lv in r:
                _, _, mx, _, _ = r[lv]; cells.append(f"{mx[0]:5.1f}")
            else:
                cells.append("  --  ")
        print(f"{tag:22} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14}")
    print("\nChen 2023 (Front. Neurosci.) evaluates at the CLIP level (binary, 5-fold).")
    return results


def _main():
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
    def _run(config):
        OmegaConf.set_struct(config, False)
        run(config)

    _run()


if __name__ == "__main__":
    _main()
