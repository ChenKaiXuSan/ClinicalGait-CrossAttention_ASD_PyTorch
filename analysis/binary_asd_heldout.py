"""
File: analysis/binary_asd_heldout.py
Created: 2026-09-03
-----
ASD-vs-non-ASD (screening) metrics DERIVED from the clean 3-class held-out runs.

Narrative B of the paper: the primary clinical question is recognising ASD among
patients with other spine/hip conditions; the 3-class result is secondary. These
numbers come from the SAME trained 3-class models and saved best_preds -- no
re-training, no re-inference, no threshold tuning:
    clip prob  = mean chunk softmax           (clip level, as in the paper)
    score      = P(ASD)                        (threshold-free -> AUROC)
    prediction = argmax == ASD                 (same decision rule as the 3-class)
Metrics per fold: accuracy, sensitivity (ASD recall), specificity (non-ASD
recall), balanced accuracy, F1(ASD), AUROC. Printed: per fold, mean+-std,
best fold (by balanced accuracy, all metrics from that same fold), pooled.

Run (repo root, asd env):
    python -m analysis.binary_asd_heldout data.root_path=/work/SKIING/chenkaixu/data/asd_dataset
"""

from __future__ import annotations

import glob
import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)
ASD = 0   # class index of ASD (labels: 0=ASD, 1=DHS, 2=LCS-HipOA)
TAGS = [
    "heldout_baseline",
    "heldout_early_add", "heldout_early_mul", "heldout_early_concat", "heldout_early_avg",
    "heldout_se_prefix4", "heldout_cross_L4",
    "heldout_pose_single_L3", "heldout_pose_multi01",
]
METRICS = ["acc", "sens", "spec", "bacc", "f1", "auroc"]


def _load_probs(tag, fold):
    import torch
    p = sorted(glob.glob(f"logs/train/{tag}_f{fold}/*/*/best_preds/fold_{fold}_pred.pt"))
    y = sorted(glob.glob(f"logs/train/{tag}_f{fold}/*/*/best_preds/fold_{fold}_label.pt"))
    if not p or not y:
        return None, None
    return (torch.load(p[-1], map_location="cpu").numpy(),
            torch.load(y[-1], map_location="cpu").numpy())


def _binary_metrics(y_true3, score, pred3):
    from sklearn.metrics import roc_auc_score, f1_score
    y = (np.asarray(y_true3) == ASD).astype(int)          # 1 = ASD
    yhat = (np.asarray(pred3) == ASD).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {
        "acc": (tp + tn) / max(len(y), 1) * 100,
        "sens": sens * 100, "spec": spec * 100,
        "bacc": (sens + spec) / 2 * 100,
        "f1": f1_score(y, yhat, zero_division=0) * 100,
        "auroc": roc_auc_score(y, score) * 100 if len(set(y)) > 1 else float("nan"),
        "n": len(y), "n_asd": int(y.sum()),
    }


def run(config):
    from pytorch_lightning import seed_everything
    from project.cross_validation import DefineCrossValidation
    from project.dataloader.data_loader import WalkDataModule

    seed_everything(42, workers=True)
    fold_idx = DefineCrossValidation(config)()
    order = {}
    for fk in sorted(fold_idx.keys(), key=int):
        fold = int(fk)
        dm = WalkDataModule(config, fold_idx[fk]); dm.setup("test")
        order[fold] = [(e["video_name"], int(e["label"])) for e in dm.test_dataloader().dataset._index_map]

    print("\n=== ASD vs non-ASD, CLIP level, derived from the 3-class held-out runs (%) ===")
    hdr = f"{'method':22} {'fold':>4} {'acc':>6} {'sens':>6} {'spec':>6} {'bacc':>6} {'F1':>6} {'AUROC':>6}   n(ASD/all)"
    for tag in TAGS:
        rows = []; pooled_y = []; pooled_s = []; pooled_p = []
        for fold, meta in order.items():
            probs, labels = _load_probs(tag, fold)
            if probs is None:
                continue
            n = len(probs); meta_n = meta[:n]
            assert np.array_equal(np.array([m[1] for m in meta_n]), labels[:n])
            acc = defaultdict(lambda: [np.zeros(probs.shape[1]), 0, None])
            for i in range(n):
                k = meta_n[i][0]; acc[k][0] += probs[i]; acc[k][1] += 1; acc[k][2] = int(labels[i])
            y3 = np.array([v[2] for v in acc.values()])
            pm = np.stack([v[0] / v[1] for v in acc.values()])       # mean prob per clip
            s = pm[:, ASD]; p3 = pm.argmax(1)
            m = _binary_metrics(y3, s, p3); m["fold"] = fold; rows.append(m)
            pooled_y.append(y3); pooled_s.append(s); pooled_p.append(p3)
        if not rows:
            print(f"{tag:22}   -- no preds"); continue
        print("\n" + hdr)
        for m in rows:
            print(f"{tag:22} {m['fold']:>4} {m['acc']:6.1f} {m['sens']:6.1f} {m['spec']:6.1f} "
                  f"{m['bacc']:6.1f} {m['f1']:6.1f} {m['auroc']:6.1f}   {m['n_asd']}/{m['n']}")
        arr = {k: np.array([m[k] for m in rows]) for k in METRICS}
        print(f"{tag:22} {'mean':>4} " + " ".join(f"{arr[k].mean():6.1f}" for k in METRICS))
        print(f"{tag:22} {'±std':>4} " + " ".join(f"{arr[k].std():6.1f}" for k in METRICS))
        bi = int(np.argmax(arr["bacc"]))
        print(f"{tag:22} {'best':>4} " + " ".join(f"{arr[k][bi]:6.1f}" for k in METRICS)
              + f"   (fold {rows[bi]['fold']}, by balanced acc)")
        pm_ = _binary_metrics(np.concatenate(pooled_y), np.concatenate(pooled_s), np.concatenate(pooled_p))
        print(f"{tag:22} {'pool':>4} " + " ".join(f"{pm_[k]:6.1f}" for k in METRICS)
              + f"   {pm_['n_asd']}/{pm_['n']}")


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
