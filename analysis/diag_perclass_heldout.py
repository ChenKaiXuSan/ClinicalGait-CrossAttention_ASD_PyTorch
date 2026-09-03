"""
File: analysis/diag_perclass_heldout.py
Created: 2026-09-03
-----
Why does the clean confusion matrix collapse on the minority classes?

For several clean held-out runs, print per fold at CLIP level:
  - the 3x3 confusion matrix, per-class recall, and how often each class is
    PREDICTED at all (a class never predicted = degenerate solution);
and, per fold, how many UNIQUE PATIENTS of each class are in train / val / test
(the held-out protocol's inner split may leave only a handful of LCS-HipOA
subjects to learn from).

No re-inference: saved best_preds + the rebuilt held-out index.
Run (repo root, asd env):
    python -m analysis.diag_perclass_heldout data.root_path=/work/SKIING/chenkaixu/data/asd_dataset
"""

from __future__ import annotations

import glob
import logging
from collections import defaultdict, Counter

import numpy as np

logger = logging.getLogger(__name__)
CLASSES = ["ASD", "DHS", "LCS-HipOA"]
TAGS = ["heldout_baseline", "heldout_early_mul", "heldout_pose_single_L3", "heldout_pose_multi01"]


def _load_probs(tag, fold):
    import torch
    p = sorted(glob.glob(f"logs/train/{tag}_f{fold}/*/*/best_preds/fold_{fold}_pred.pt"))
    y = sorted(glob.glob(f"logs/train/{tag}_f{fold}/*/*/best_preds/fold_{fold}_label.pt"))
    if not p or not y:
        return None, None
    return (torch.load(p[-1], map_location="cpu").numpy(),
            torch.load(y[-1], map_location="cpu").numpy())


def _patients_per_class(index_map):
    """unique patient ids per class from a dataset _index_map."""
    s = defaultdict(set)
    for e in index_map:
        # patient key = leading date[+idx] (first two "_" tokens); the "-"-prefix
        # over-splits patients because video_name spellings are inconsistent.
        s[int(e["label"])].add("_".join(str(e["video_name"]).split("_")[:2]))
    return {c: len(s[c]) for c in range(3)}


def run(config):
    from pytorch_lightning import seed_everything
    from project.cross_validation import DefineCrossValidation
    from project.dataloader.data_loader import WalkDataModule

    seed_everything(42, workers=True)
    fold_idx = DefineCrossValidation(config)()

    # ---- (1) unique patients per class per split, per fold ----
    print("\n=== Unique PATIENTS per class in each split (held-out protocol) ===")
    print(f"{'fold':>4} {'split':>6} {'ASD':>5} {'DHS':>5} {'LCS':>5}")
    order = {}
    for fk in sorted(fold_idx.keys(), key=int):
        fold = int(fk)
        dm = WalkDataModule(config, fold_idx[fk])
        try:
            dm.setup("fit")
            tr = _patients_per_class(dm.train_dataloader().dataset._index_map)
            va = _patients_per_class(dm.val_dataloader().dataset._index_map)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[fold {fold}] fit index unavailable: {e}")
            tr = va = {0: -1, 1: -1, 2: -1}
        dm.setup("test")
        im = dm.test_dataloader().dataset._index_map
        te = _patients_per_class(im)
        order[fold] = [(e["video_name"], int(e["label"])) for e in im]
        for name, d in (("train", tr), ("val", va), ("test", te)):
            print(f"{fold:>4} {name:>6} {d[0]:>5} {d[1]:>5} {d[2]:>5}")

    # ---- (2) per-tag, per-fold clip-level confusion / recall / prediction counts ----
    for tag in TAGS:
        print(f"\n=== {tag} : clip-level, per fold ===")
        pooled = np.zeros((3, 3))
        for fold, meta in order.items():
            probs, labels = _load_probs(tag, fold)
            if probs is None:
                print(f"  fold {fold}: no preds"); continue
            n = len(probs); meta_n = meta[:n]
            assert np.array_equal(np.array([m[1] for m in meta_n]), labels[:n])
            acc = defaultdict(lambda: [np.zeros(probs.shape[1]), None])
            for i in range(n):
                acc[meta_n[i][0]][0] += probs[i]; acc[meta_n[i][0]][1] = int(labels[i])
            cm = np.zeros((3, 3))
            for v in acc.values():
                cm[v[1], int(v[0].argmax())] += 1
            pooled += cm
            rec = [cm[c, c] / cm[c].sum() * 100 if cm[c].sum() else float("nan") for c in range(3)]
            pred_cnt = cm.sum(0).astype(int)
            true_cnt = cm.sum(1).astype(int)
            print(f"  fold {fold}: recall ASD/DHS/LCS = {rec[0]:5.1f} / {rec[1]:5.1f} / {rec[2]:5.1f}   "
                  f"true n = {true_cnt.tolist()}   predicted n = {pred_cnt.tolist()}   "
                  f"acc = {np.trace(cm)/cm.sum()*100:5.1f}")
        rec = [pooled[c, c] / pooled[c].sum() * 100 if pooled[c].sum() else float("nan") for c in range(3)]
        print(f"  POOLED: recall ASD/DHS/LCS = {rec[0]:5.1f} / {rec[1]:5.1f} / {rec[2]:5.1f}   "
              f"predicted n = {pooled.sum(0).astype(int).tolist()}   acc = {np.trace(pooled)/pooled.sum()*100:5.1f}")


def _main():
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
    def _run(config):
        OmegaConf.set_struct(config, False)
        config.data.heldout_test = True
        run(config)

    _run()


if __name__ == "__main__":
    _main()
