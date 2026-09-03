"""
File: analysis/make_confusion_heldout.py
Created: 2026-08-06
-----
CLIP-level confusion matrix of the clean main model (paper/figures/fig_confusion.pdf).

Replaces make_figures.fig_confusion(), which pooled CHUNK-level argmax predictions
of the archived leaky run (pose_atn_multi_P1). The paper reports gait-CLIP-level
results under the held-out protocol, so this script:
  1. loads the saved softmax best_preds of the clean run (default heldout_pose_multi01),
  2. rebuilds the held-out test index per fold (data.heldout_test=True is forced, so
     the index matches the split the checkpoints were trained/selected on),
  3. maps chunks -> clips (video_name), aggregates mean probability per clip,
  4. pools the clips of all folds into one row-normalised 3x3 matrix.
No re-inference. Style mirrors the original fig_confusion.

Run (repo root, asd env):
    python -m analysis.make_confusion_heldout \
        data.root_path=/work/SKIING/chenkaixu/data/asd_dataset [+cm.tag=heldout_pose_multi01]
"""

from __future__ import annotations

import glob
import logging
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
OUT = "paper/figures"
CLASSES = ["ASD", "DHS", "LCS-HipOA"]


def _load_probs(tag, fold):
    import torch
    p = sorted(glob.glob(f"logs/train/{tag}_f{fold}/*/*/best_preds/fold_{fold}_pred.pt"))
    y = sorted(glob.glob(f"logs/train/{tag}_f{fold}/*/*/best_preds/fold_{fold}_label.pt"))
    if not p or not y:
        return None, None
    # several run dirs (e.g. a re-run): take the most recent
    return (torch.load(p[-1], map_location="cpu").numpy(),
            torch.load(y[-1], map_location="cpu").numpy())


def run(config):
    from pytorch_lightning import seed_everything
    from project.cross_validation import DefineCrossValidation
    from project.dataloader.data_loader import WalkDataModule

    tag = str(config.get("cm", {}).get("tag", "heldout_pose_multi01"))
    seed_everything(42, workers=True)
    fold_idx = DefineCrossValidation(config)()

    cm = np.zeros((3, 3))
    n_clips = 0
    for fk in sorted(fold_idx.keys(), key=int):
        fold = int(fk)
        probs, labels = _load_probs(tag, fold)
        if probs is None:
            logger.warning(f"[fold {fold}] no best_preds for {tag}; skipped")
            continue
        dm = WalkDataModule(config, fold_idx[fk])
        dm.setup("test")
        im = dm.test_dataloader().dataset._index_map
        meta = [(e["video_name"], int(e["label"])) for e in im]
        n = len(probs)
        meta = meta[:n]                                   # test drop_last: saved <= index
        assert np.array_equal(np.array([m[1] for m in meta]), labels[:n]), \
            f"label mismatch {tag} fold{fold}"
        acc = defaultdict(lambda: [np.zeros(probs.shape[1]), None])
        for i in range(n):
            k = meta[i][0]
            acc[k][0] += probs[i]
            acc[k][1] = int(labels[i])
        for v in acc.values():
            cm[v[1], int(v[0].argmax())] += 1
        n_clips += len(acc)
        logger.info(f"[fold {fold}] {len(acc)} clips from {n} chunks")

    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    imh = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "#222222", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.grid(False)
    cb = fig.colorbar(imh, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("row-normalized (%)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    os.makedirs(OUT, exist_ok=True)
    out = f"{OUT}/fig_confusion.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({n_clips} clips pooled, tag={tag}, clip level)")
    print("row-normalized (%):\n", np.round(cmn * 100, 1))
    print("counts:\n", cm.astype(int))
    return cm


def _main():
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
    def _run(config):
        OmegaConf.set_struct(config, False)
        config.data.heldout_test = True   # clean ckpts were trained/selected on the held-out split
        run(config)

    _run()


if __name__ == "__main__":
    _main()
