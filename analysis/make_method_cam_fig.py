"""
File: analysis/make_method_cam_fig.py
Created: 2026-07-28
-----
Cross-method qualitative comparison for the paper (companion to fig_qualitative).
For one correctly-comparable held-out clip per class, overlay Grad-CAM
(class-discriminative saliency, model-agnostic) for EACH fusion method against
the doctor ROI, so a reader can see PoseGated's attention localises on the
clinical ROI while the unsupervised baselines are more diffuse / off-target.

Grad-CAM target = the SlowR50 backbone's last residual stage (blocks[4]),
identical across all methods, so the comparison is apples-to-apples. The same
fold-0 clips are fed to every method's fold-0 checkpoint (the fold split is
shared across methods).

Run (asd env; CPU ok, ~1-2 min):
    python -m analysis.make_method_cam_fig \
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

from analysis.attention_alignment import _resolve_run_dir, _find_best_ckpt
from analysis.make_qualitative_fig import _upsample, _norm, _overlay

logger = logging.getLogger(__name__)
OUT = "paper/figures"

# display name, log tag, trainer module, trainer class
METHODS = [
    ("Baseline",   "baseline_rgb",      "project.trainer.baseline.train_3dcnn", "Res3DCNNTrainer"),
    ("SE",         "se_atn_prefix0",    "project.trainer.mid.train_se_attn",    "SEAttnTrainer"),
    ("Cross-attn", "cross_atn_L4",      "project.trainer.mid.train_cross_attn", "CrossAttentionTrainer"),
    ("PoseGated",  "pose_atn_multi_P1", "project.trainer.mid.train_pose_attn",  "PoseAttnTrainer"),
]


def _grad_cam(model, video, attn, target_layer, cls_idx):
    """Grad-CAM heat map (H',W' upsampled to frame) for class `cls_idx`."""
    import torch
    acts, grads = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.zero_grad(set_to_none=True)
        out = model(video, attn)
        logits = out[0] if isinstance(out, tuple) else out
        score = logits[0, cls_idx]
        score.backward()
        A = acts["a"].detach()            # (1,C,T,H',W')
        G = grads["g"].detach()           # (1,C,T,H',W')
        w = G.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((w * A).sum(dim=1))[0]   # (T,H',W')
        t = cam.shape[0] // 2
        return cam[t].cpu().numpy()
    finally:
        h1.remove(); h2.remove()


def collect(config):
    import torch
    from pytorch_lightning import seed_everything
    from project.cross_validation import DefineCrossValidation, class_num_mapping_Dict
    from project.dataloader.data_loader import WalkDataModule

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

    # one clip per class (indices spread within each class; skip edges)
    by_label = {}
    for i, e in enumerate(index_map):
        by_label.setdefault(int(e["label"]), []).append(i)
    clips = {}   # class_name -> dict(video, attn, rgb, roi, label)
    for lab, idxs in sorted(by_label.items()):
        idx = idxs[int(0.5 * (len(idxs) - 1))]
        item = ds[idx]
        cls = class_map.get(lab, str(lab))
        video = item["video"].unsqueeze(0)
        attn = item["attn_map"].unsqueeze(0)
        T, H, W = video.shape[2], video.shape[3], video.shape[4]
        t = T // 2
        clips[cls] = {
            "video": video, "attn": attn, "label": lab,
            "rgb": np.clip(video[0, :, t].cpu().numpy().transpose(1, 2, 0), 0, 1),
            "roi": _norm(attn[0].max(0).values[t].cpu().numpy()),
        }
    logger.info(f"clips: {list(clips)} (fold {fold})")

    # per method: load ckpt, Grad-CAM each clip for its true class
    cams = {}   # method -> class -> heat(H,W)
    for disp, tag, mod, cls_name in METHODS:
        run_dir = _resolve_run_dir(f"logs/train/{tag}_f*/**", fold)
        if not run_dir:
            logger.warning(f"{disp}: no run dir for fold {fold}; skipped"); continue
        ckpt = _find_best_ckpt(run_dir, fold)
        if not ckpt:
            logger.warning(f"{disp}: no ckpt; skipped"); continue
        Trainer = getattr(importlib.import_module(mod), cls_name)
        module = Trainer.load_from_checkpoint(ckpt, map_location=device).eval().to(device)
        net = module.model
        target = net.model.blocks[4]          # last residual stage (shared backbone)
        cams[disp] = {}
        for cls, c in clips.items():
            heat = _grad_cam(net, c["video"].to(device), c["attn"].to(device), target, c["label"])
            cams[disp][cls] = _norm(_upsample(heat, c["rgb"].shape[:2]))
        logger.info(f"{disp}: Grad-CAM done ({tag})")
    return clips, cams


def render(clips, cams):
    os.makedirs(OUT, exist_ok=True)
    classes = [c for c in ["ASD", "DHS", "LCS_HipOA"] if c in clips] or list(clips)
    methods = [m for m, *_ in METHODS if m in cams]
    cols = ["RGB", "Doctor ROI"] + methods
    nrow, ncol = len(classes), len(cols)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.05 * ncol, 1.15 * nrow))
    if nrow == 1:
        axes = axes[None, :]
    for r, cls in enumerate(classes):
        c = clips[cls]
        axes[r, 0].imshow(c["rgb"])
        _overlay(axes[r, 1], c["rgb"], c["roi"])
        for k, m in enumerate(methods):
            _overlay(axes[r, 2 + k], c["rgb"], cams[m][cls])
        axes[r, 0].set_ylabel(cls.replace("_", "-"), fontsize=9)
        for cc in range(ncol):
            axes[r, cc].set_xticks([]); axes[r, cc].set_yticks([])
            if r == 0:
                axes[r, cc].set_title(cols[cc], fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_method_cam.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig_method_cam.pdf ({nrow} classes x {len(methods)} methods)")


def _main():
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
    def run(config):
        OmegaConf.set_struct(config, False)
        clips, cams = collect(config)
        render(clips, cams)

    run()


if __name__ == "__main__":
    _main()
