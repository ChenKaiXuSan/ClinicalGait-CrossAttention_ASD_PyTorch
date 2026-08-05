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

    results = {}
    for tag in METHODS:
        lvl = {"chunk": [], "clip": [], "patient": []}
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
            lvl["chunk"].append((probs.argmax(1) == labels[:n]).mean() * 100)

            # aggregate mean-prob by clip (video_name) and by patient
            for key_fn, name in ((lambda i: meta_n[i][0], "clip"),
                                 (lambda i: patient_of(meta_n[i][0]), "patient")):
                acc = defaultdict(lambda: [np.zeros(probs.shape[1]), 0, None])
                for i in range(n):
                    k = key_fn(i)
                    acc[k][0] += probs[i]
                    acc[k][1] += 1
                    acc[k][2] = labels[i]
                correct = sum(int(v[0].argmax() == v[2]) for v in acc.values())
                lvl[name].append(100 * correct / len(acc))
        results[tag] = {k: (np.mean(v), np.std(v), len(v)) for k, v in lvl.items() if v}
        # also report unit counts from the last fold processed
    # ---- print table ----
    print("\n=== Accuracy by evaluation granularity (mean±std over folds) ===")
    print(f"{'method':20} {'chunk (~42k)':>16} {'clip (~1954)':>16} {'patient (81)':>16}")
    for tag in METHODS:
        r = results.get(tag, {})
        cells = []
        for lv in ("chunk", "clip", "patient"):
            if lv in r:
                m, s, _ = r[lv]; cells.append(f"{m:5.1f} ± {s:3.1f}")
            else:
                cells.append("   --   ")
        print(f"{tag:20} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")
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
