"""
File: analysis/make_combined_fig.py
Created: 2026-07-28
-----
Single combined qualitative figure (merges the former fig_qualitative +
fig_method_cam into one image), using the SAME held-out clip per class for both
panels:

  (a) Class-discriminative attention (Grad-CAM at the shared SlowR50 blocks[4])
      across fusion methods: RGB | Doctor ROI | Baseline | SE | Cross-attn |
      ClinicalGated. Shows unsupervised baselines are diffuse; ClinicalGated is focused.
  (b) ClinicalGated SUPERVISED side-head (L3) attention vs. the clinical ROI (deep
      variant multi-[0,1,2,3]): RGB | Doctor ROI | ClinicalGated attention. Shows the
      supervised attention lands on the annotated joints (fidelity).

Grad-CAM (a) answers "what drives the class decision"; the supervised map (b) is
trained to match the ROI — two different quantities, one figure.

Run (asd env; CPU ok, ~1-2 min):
    python -m analysis.make_combined_fig \
        data.root_path=/work/SKIING/chenkaixu/data/asd_dataset +qual.fold=0
"""

from __future__ import annotations

import os
import importlib
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from analysis.attention_alignment import _resolve_run_dir, _find_best_ckpt
from analysis.make_qualitative_fig import _upsample, _norm, _overlay
from analysis.make_method_cam_fig import _grad_cam, METHODS

logger = logging.getLogger(__name__)
OUT = "paper/figures"
SUP_TAG = "heldout_pose_multi_P3"   # clean deep variant with side heads through L3 (supervised panel); run with data.heldout_test=True


def collect(config):
    import torch
    from pytorch_lightning import seed_everything
    from project.cross_validation import DefineCrossValidation, class_num_mapping_Dict
    from project.dataloader.data_loader import WalkDataModule
    from project.trainer.mid.train_pose_attn import PoseAttnTrainer

    seed_everything(42, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_map = class_num_mapping_Dict[int(config.model.model_class_num)]
    want_fold = int(config.get("qual", {}).get("fold", 0))

    fold_idx = DefineCrossValidation(config)()
    fold_key = str(want_fold) if str(want_fold) in fold_idx else list(fold_idx.keys())[0]
    fold = int(fold_key)

    dm = WalkDataModule(config, fold_idx[fold_key])
    dm.setup("test")
    ds = dm.test_dataloader().dataset
    index_map = ds._index_map
    by_label = {}
    for i, e in enumerate(index_map):
        by_label.setdefault(int(e["label"]), []).append(i)

    # ---- (b) supervised panel: pick clips via the deep P3 model, keep the clip ----
    run_dir = _resolve_run_dir(f"logs/train/{SUP_TAG}_f*/**", fold)
    ckpt = _find_best_ckpt(run_dir, fold) if run_dir else None
    assert ckpt, f"no {SUP_TAG} checkpoint for fold {fold}"
    p3 = PoseAttnTrainer.load_from_checkpoint(ckpt, map_location=device).eval().to(device)

    clips = {}   # class -> dict(video, attn, label, rgb, roi, sup)
    with torch.no_grad():
        for lab, idxs in sorted(by_label.items()):
            cls = class_map.get(lab, str(lab))
            for p in np.linspace(0.15, 0.85, 8):
                idx = idxs[int(p * (len(idxs) - 1))]
                item = ds[idx]
                video = item["video"].unsqueeze(0).to(device)
                attn = item["attn_map"].unsqueeze(0).to(device)
                out = p3.model(video, attn, return_aux=True)
                logits, aux = out if isinstance(out, tuple) else (out, {"side_preds": []})
                if int(logits.argmax(1)) != lab or not aux.get("side_preds"):
                    continue
                T, H, W = video.shape[2], video.shape[3], video.shape[4]
                t = T // 2
                Sig = torch.sigmoid(aux["side_preds"][-1])[0]          # (Cctx,Ti,Hi,Wi)
                ti = min(t, Sig.shape[1] - 1)
                clips[cls] = {
                    "video": item["video"].unsqueeze(0), "attn": item["attn_map"].unsqueeze(0),
                    "label": lab,
                    "rgb": np.clip(video[0, :, t].cpu().numpy().transpose(1, 2, 0), 0, 1),
                    "roi": _norm(attn[0].max(0).values[t].cpu().numpy()),
                    "sup": _norm(_upsample(Sig.max(0).values[ti].cpu().numpy(), (H, W))),
                }
                logger.info(f"  {cls}: clip idx {idx}")
                break
    del p3

    # ---- (a) Grad-CAM per method on the SAME clips ----
    cams = {}
    for disp, tag, mod, cls_name in METHODS:
        rd = _resolve_run_dir(f"logs/train/{tag}_f*/**", fold)
        ck = _find_best_ckpt(rd, fold) if rd else None
        if not ck:
            logger.warning(f"{disp}: no ckpt; skipped"); continue
        Trainer = getattr(importlib.import_module(mod), cls_name)
        module = Trainer.load_from_checkpoint(ck, map_location=device).eval().to(device)
        net = module.model
        target = net.model.blocks[4]
        cams[disp] = {}
        for cls, c in clips.items():
            heat = _grad_cam(net, c["video"].to(device), c["attn"].to(device), target, c["label"])
            cams[disp][cls] = _norm(_upsample(heat, c["rgb"].shape[:2]))
        logger.info(f"{disp}: Grad-CAM done")
        del module
    return clips, cams


def render(clips, cams):
    os.makedirs(OUT, exist_ok=True)
    classes = [c for c in ["ASD", "DHS", "LCS_HipOA"] if c in clips] or list(clips)
    methods = [m for m, *_ in METHODS if m in cams]
    nrow = len(classes)
    a_cols = ["RGB", "Doctor ROI"] + methods                 # panel (a): 2 + methods
    nca = len(a_cols)

    L, R = 0.07, 0.99
    cellw = (R - L) / nca
    fig = plt.figure(figsize=(1.15 * nca, 2.05 * nrow + 0.7))

    # two explicitly-positioned bands so panel titles never collide
    gsa = fig.add_gridspec(nrow, nca, left=L, right=R, top=0.90, bottom=0.55,
                           hspace=0.05, wspace=0.05)
    bL = (L + R) / 2 - 1.5 * cellw                            # centre the 3 (b)-cells
    gsb = fig.add_gridspec(nrow, 3, left=bL, right=bL + 3 * cellw, top=0.44, bottom=0.05,
                           hspace=0.05, wspace=0.05)

    # ---- panel (a): Grad-CAM across methods ----
    for r, cls in enumerate(classes):
        c = clips[cls]
        row = []
        ax0 = fig.add_subplot(gsa[r, 0]); ax0.imshow(c["rgb"]); ax0.set_ylabel(cls.replace("_", "-"), fontsize=9); row.append(ax0)
        ax1 = fig.add_subplot(gsa[r, 1]); _overlay(ax1, c["rgb"], c["roi"]); row.append(ax1)
        for k, m in enumerate(methods):
            ax = fig.add_subplot(gsa[r, 2 + k]); _overlay(ax, c["rgb"], cams[m][cls]); row.append(ax)
        for cc, ax in enumerate(row):
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(a_cols[cc], fontsize=8)

    # ---- panel (b): ClinicalGated supervised attention (centered 3 cols) ----
    b_cols = ["RGB", "Doctor ROI", "ClinicalGated (supervised)"]
    for r, cls in enumerate(classes):
        c = clips[cls]
        axr = fig.add_subplot(gsb[r, 0]); axr.imshow(c["rgb"]); axr.set_ylabel(cls.replace("_", "-"), fontsize=9)
        axo = fig.add_subplot(gsb[r, 1]); _overlay(axo, c["rgb"], c["roi"])
        axs = fig.add_subplot(gsb[r, 2]); _overlay(axs, c["rgb"], c["sup"])
        for cc, ax in enumerate((axr, axo, axs)):
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(b_cols[cc], fontsize=8)

    fig.text(0.5, 0.955, "(a) Class-discriminative attention (Grad-CAM) across fusion methods",
             ha="center", va="center", fontsize=9)
    fig.text(0.5, 0.495, "(b) ClinicalGated supervised side-head attention vs. clinical ROI",
             ha="center", va="center", fontsize=9)
    fig.savefig(f"{OUT}/fig_qual_combined.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig_qual_combined.pdf ({nrow} classes; (a) {len(methods)} methods, (b) supervised)")


def _main():
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
    def run(config):
        import pickle
        OmegaConf.set_struct(config, False)
        cache = os.environ.get("COMB_CACHE")
        if cache and os.path.exists(cache):
            clips, cams = pickle.load(open(cache, "rb"))
            logger.info(f"loaded cached collect() from {cache}")
        else:
            clips, cams = collect(config)
            if cache:
                pickle.dump((clips, cams), open(cache, "wb"))
        render(clips, cams)

    run()


if __name__ == "__main__":
    _main()
