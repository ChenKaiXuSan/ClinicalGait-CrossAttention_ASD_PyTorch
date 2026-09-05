"""
File: analysis/make_qualitative_fig.py
Created: 2026-07-28
-----
Qualitative interpretability figure for the paper: for one correctly-classified
held-out clip per diagnostic class, show
    [ RGB frame | RGB + doctor ROI | RGB + model side-head attention ]
so a reader can SEE that the supervised attention lands on the clinical ROI
(the quantitative version is Table `tab:align` / fig_alignment).

Mirrors analysis/attention_alignment.evaluate_alignment for the (proven) data +
checkpoint path, but instead of computing metrics it captures a few example
tensors and renders analysis/figures_out/fig_qualitative.pdf.

Run (asd env; CPU is fine, just slower):
    python -m analysis.make_qualitative_fig \
        data.root_path=/work/SKIING/chenkaixu/data/asd_dataset \
        +align.run_glob='logs/train/pose_atn_multi_P1_f*/**' \
        +qual.fold=0 data.batch_size=4
"""

from __future__ import annotations

import os
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
OUT = "analysis/figures_out"  # neutral output; copy into paper/<version>/figures by hand

# import the proven checkpoint-resolution helpers
from analysis.attention_alignment import _resolve_run_dir, _find_best_ckpt


def _upsample(a2d: np.ndarray, hw) -> np.ndarray:
    """Smoothly upsample a small (h,w) side-head map to (H,W): bilinear zoom + a
    light Gaussian blur, so a low-res (e.g. 14x14) head renders as a clean heat
    blob instead of blocky squares. Falls back to nearest if scipy is absent."""
    H, W = hw
    h, w = a2d.shape
    try:
        from scipy.ndimage import zoom, gaussian_filter
        up = zoom(a2d, (H / h, W / w), order=1)          # bilinear
        up = gaussian_filter(up, sigma=max(H, W) / 40.0)  # soften
        return up
    except Exception:
        yi = (np.arange(H) * h / H).astype(int).clip(0, h - 1)
        xi = (np.arange(W) * w / W).astype(int).clip(0, w - 1)
        return a2d[yi][:, xi]


def _norm(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def collect(config):
    import torch
    import torch.nn.functional as F
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

    run_dir = _resolve_run_dir(config.align.run_glob, fold)
    assert run_dir, f"no run dir with checkpoints for fold {fold} under {config.align.run_glob}"
    ckpt = _find_best_ckpt(run_dir, fold)
    assert ckpt, f"no checkpoint for fold {fold} in {run_dir}"
    logger.info(f"[fold {fold}] loading {ckpt}")

    module = PoseAttnTrainer.load_from_checkpoint(ckpt, map_location=device)
    module.eval().to(device)

    dm = WalkDataModule(config, fold_idx[fold_key])
    dm.setup("test")
    ds = dm.test_dataloader().dataset               # whole_video_dataset (shuffle=False)
    index_map = ds._index_map

    # The test set is ordered by class, so iterating batches would only reach one
    # class before any CPU budget runs out. Instead pick a few candidate chunk
    # indices spread across EACH class and run just those through the model.
    by_label = {}
    for i, e in enumerate(index_map):
        by_label.setdefault(int(e["label"]), []).append(i)
    n_cand = int(config.get("qual", {}).get("candidates", 6))
    cand_idx = []
    for lab, idxs in by_label.items():
        # evenly spaced picks within the class (skip the very first/last chunks)
        picks = np.linspace(0.15, 0.85, n_cand)
        for p in picks:
            cand_idx.append(idxs[int(p * (len(idxs) - 1))])
    logger.info(f"labels present: {sorted(by_label)}; {len(cand_idx)} candidate clips")

    examples = {}          # class_name -> dict(rgb, roi, sal)
    with torch.no_grad():
        for ci, idx in enumerate(cand_idx):
            item = ds[idx]
            lab = int(item["label"])
            cls = class_map.get(lab, str(lab))
            if cls in examples:
                continue
            video = item["video"].unsqueeze(0).to(device)     # (1,3,T,H,W)
            doctor = item["attn_map"].unsqueeze(0).to(device)  # (1,Cctx,T,H,W)
            out = module.model(video, doctor, return_aux=True)
            logits, aux = out if isinstance(out, tuple) else (out, {"side_preds": []})
            pred = int(logits.argmax(1).item())
            side_preds = aux.get("side_preds", [])
            if pred != lab or not side_preds:
                continue                                       # want a correct, supervised example
            Pi = side_preds[-1]                                # deepest included side head
            Sig = torch.sigmoid(Pi)[0]                         # (Cctx,Ti,Hi,Wi)
            T, H, W = video.shape[2], video.shape[3], video.shape[4]
            t = T // 2
            rgb = np.clip(video[0, :, t].cpu().numpy().transpose(1, 2, 0), 0, 1)  # (H,W,3)
            roi = doctor[0].max(0).values[t].cpu().numpy()                        # (H,W)
            ti = min(t, Sig.shape[1] - 1)
            sal = _upsample(Sig.max(0).values[ti].cpu().numpy(), (H, W))          # (H,W)
            examples[cls] = {"rgb": rgb, "roi": _norm(roi), "sal": _norm(sal)}
            logger.info(f"  captured '{cls}' from candidate {ci} (chunk idx {idx})")
            if len(examples) >= len(by_label):
                break
    return examples


def _overlay(ax, rgb, heat, cmap="jet", thr=0.35, amax=0.80):
    """Show rgb, then a heat overlay whose alpha ramps from 0 below `thr` to
    `amax` at the peak — so only the salient region is visible (clean on a dark
    frame, unlike a flat-alpha wash)."""
    from matplotlib import cm
    ax.imshow(rgb)
    m = np.clip(heat, 0, 1)
    alpha = np.clip((m - thr) / (1 - thr + 1e-8), 0, 1) * amax
    rgba = cm.get_cmap(cmap)(m)
    rgba[..., 3] = alpha
    ax.imshow(rgba)


def render(examples):
    os.makedirs(OUT, exist_ok=True)
    classes = [c for c in ["ASD", "DHS", "LCS_HipOA"] if c in examples] or list(examples)
    nrow = len(classes)
    if nrow == 0:
        print("no examples captured — nothing to render")
        return
    fig, axes = plt.subplots(nrow, 3, figsize=(6.2, 2.05 * nrow))
    if nrow == 1:
        axes = axes[None, :]
    col_titles = ["RGB frame", "+ doctor ROI", "+ model attention"]
    for r, cls in enumerate(classes):
        ex = examples[cls]
        axes[r, 0].imshow(ex["rgb"])
        _overlay(axes[r, 1], ex["rgb"], ex["roi"])
        _overlay(axes[r, 2], ex["rgb"], ex["sal"])
        axes[r, 0].set_ylabel(cls.replace("_", "-"), fontsize=10)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_qualitative.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig_qualitative.pdf ({nrow} classes)")


def _main():
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
    def run(config):
        if "align" not in config:
            OmegaConf.set_struct(config, False)
            config.align = OmegaConf.create(
                {"run_glob": "logs/train/pose_atn_multi_P1_f*/**"})
        render(collect(config))

    run()


if __name__ == "__main__":
    _main()
