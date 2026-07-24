"""
File: analysis/attention_alignment.py
Project: analysis
Created Date: 2026-07-24
Author: Kaixu Chen
-----
Comment:
 Quantitative clinical-attention alignment metrics (Ablation "interpretability").

 Measures how well PoseGated's side-head heatmaps (side_preds) match the
 doctor-annotated ROI (doctor_hm = the batch `attn_map`) on the HELD-OUT test
 patients of each fold. This is narrative (A): fidelity of the supervised
 clinical prior, generalizing to unseen patients — NOT a cross-method table.

 Why not cosine/SSIM (the old calc_similarity.py): SSIM measures perceptual
 image quality and cosine is inflated by shared background agreement; neither
 targets saliency localization. We use the standard saliency-evaluation family
 (Bylinskii et al., 2019, "What do different evaluation metrics tell us about
 saliency models?") plus Pointing Game (weak-localization):

   CC   : Pearson correlation of the two maps as densities        (threshold-free, [-1,1], ↑)
   NSS  : mean z-normalized saliency at ROI locations             (penalizes diffuse maps, ↑)
   PG   : Pointing Game — saliency peak lands inside the ROI       (hit rate %, ↑)
   AUC  : AUC-Judd, ROI pixels as positives                       (threshold-free, [0,1], ↑)
   IoU  : overlap of top-k% saliency vs ROI mask                  (ties to the Dice loss, ↑)
   Dice : soft-Dice of the same two masks                         (↑)
   KL   : KL(doctor || saliency) as distributions                (↓)
   SIM  : histogram intersection of the two distributions         (↑)

 Aggregation: per (fold, class, side-head layer, joint) then mean±std over all
 test clips. Per-joint lets you show attention concentrates on the pathological
 joints the surgeons flagged (lumbar/pelvis/head/shoulder).

Usage (on the cluster, after training PoseGated folds):
    python -m analysis.attention_alignment \
        data.root_path=/work/SKIING/chenkaixu/data/asd_dataset \
        +align.run_glob='logs/train/pose_gated_full_f*/**'

Self-test the metric numerics (no data / no checkpoint needed):
    python -m analysis.attention_alignment --selftest
-----
"""

from __future__ import annotations

import os
import sys
import glob
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ROI binarization threshold on the (soft) doctor heatmap, and top-k fraction
# for the overlap metrics. Both are reported so the numbers are reproducible.
ROI_THR = 0.5
TOPK_FRAC = 0.20
EPS = 1e-8


# ============================ metric functions ============================
# All operate on flat float arrays. `sal` = model saliency (prob, [0,1]-ish),
# `doc` = doctor heatmap (soft, [0,1]). Shapes must match.

def _flat(sal: np.ndarray, doc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(sal, np.float64).ravel(), np.asarray(doc, np.float64).ravel()


def cc(sal: np.ndarray, doc: np.ndarray) -> float:
    """Pearson correlation coefficient of the two maps as densities."""
    s, d = _flat(sal, doc)
    s = s - s.mean()
    d = d - d.mean()
    denom = np.sqrt((s * s).sum() * (d * d).sum())
    if denom < EPS:
        return 0.0
    return float((s * d).sum() / denom)


def nss(sal: np.ndarray, doc: np.ndarray, roi_thr: float = ROI_THR) -> float:
    """Normalized Scanpath Saliency: z-normalized saliency averaged over ROI."""
    s, d = _flat(sal, doc)
    std = s.std()
    if std < EPS:
        return 0.0
    s = (s - s.mean()) / std
    roi = d > roi_thr
    if roi.sum() == 0:
        return float("nan")
    return float(s[roi].mean())


def pointing_game(sal: np.ndarray, doc: np.ndarray, roi_thr: float = ROI_THR) -> Optional[float]:
    """1.0 if the saliency peak lands on an ROI pixel, else 0.0 (per-map hit)."""
    s, d = _flat(sal, doc)
    roi = d > roi_thr
    if roi.sum() == 0:
        return None  # no ROI in this frame/joint -> excluded from the hit-rate
    return float(roi[int(np.argmax(s))])


def auc_judd(sal: np.ndarray, doc: np.ndarray, roi_thr: float = ROI_THR) -> float:
    """AUC with ROI pixels as positives; rank-based (Mann-Whitney) exact AUC."""
    s, d = _flat(sal, doc)
    pos = d > roi_thr
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg_rank = (start + cum + 1) / 2.0
    ranks = avg_rank[inv]
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def topk_overlap(sal: np.ndarray, doc: np.ndarray, k: float = TOPK_FRAC,
                 roi_thr: float = ROI_THR) -> Tuple[float, float]:
    """IoU and soft-Dice between the top-k% saliency mask and the ROI mask."""
    s, d = _flat(sal, doc)
    roi = d > roi_thr
    if roi.sum() == 0:
        return float("nan"), float("nan")
    n_top = max(1, int(round(k * s.size)))
    thr = np.partition(s, -n_top)[-n_top]
    pred = s >= thr
    inter = np.logical_and(pred, roi).sum()
    union = np.logical_or(pred, roi).sum()
    iou = inter / (union + EPS)
    dice = 2.0 * inter / (pred.sum() + roi.sum() + EPS)
    return float(iou), float(dice)


def kl_div(sal: np.ndarray, doc: np.ndarray) -> float:
    """KL(doctor || saliency), both L1-normalized to distributions. Lower better."""
    s, d = _flat(sal, doc)
    s = np.clip(s, 0, None); d = np.clip(d, 0, None)
    s = s / (s.sum() + EPS); d = d / (d.sum() + EPS)
    return float(np.sum(d * np.log(d / (s + EPS) + EPS)))


def sim(sal: np.ndarray, doc: np.ndarray) -> float:
    """Histogram intersection of the two distributions. [0,1], higher better."""
    s, d = _flat(sal, doc)
    s = np.clip(s, 0, None); d = np.clip(d, 0, None)
    s = s / (s.sum() + EPS); d = d / (d.sum() + EPS)
    return float(np.minimum(s, d).sum())


def all_metrics(sal: np.ndarray, doc: np.ndarray) -> Dict[str, float]:
    iou, dice = topk_overlap(sal, doc)
    return {
        "cc": cc(sal, doc),
        "nss": nss(sal, doc),
        "pg": pointing_game(sal, doc),
        "auc": auc_judd(sal, doc),
        "iou": iou,
        "dice": dice,
        "kl": kl_div(sal, doc),
        "sim": sim(sal, doc),
    }


# ============================ evaluation runner ============================
# Imports of torch / project modules are deferred so `--selftest` runs without
# a GPU, dataset, or the heavy dependency stack.

def _find_best_ckpt(run_dir: str, fold: int) -> Optional[str]:
    """Best checkpoint for a fold under train.py's layout: <run_dir>/checkpoints/fold_<f>/*.ckpt.

    ModelCheckpoint filename is '{epoch}-{val_loss:.2f}-{val_video_acc:.4f}.ckpt'
    (auto_insert_metric_name=False). Pick the highest trailing accuracy; fall
    back to last.ckpt.
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints", f"fold_{fold}")
    cands = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    scored: List[Tuple[float, str]] = []
    last = None
    for p in cands:
        name = os.path.basename(p)
        if name.lower() == "last.ckpt":
            last = p
            continue
        try:
            acc = float(name.replace(".ckpt", "").split("-")[-1])
            scored.append((acc, p))
        except ValueError:
            continue
    if scored:
        return max(scored, key=lambda t: t[0])[1]
    return last


def _resolve_run_dir(run_glob: str, fold: int) -> Optional[str]:
    """Find the run directory holding this fold's checkpoints.

    `run_glob` may point at a per-fold experiment tree (e.g.
    'logs/train/pose_gated_full_f*/**'); we keep only dirs that actually contain
    checkpoints/fold_<fold>/, then take the most recently modified one.
    """
    matches = []
    for d in glob.glob(run_glob, recursive=True):
        if os.path.isdir(os.path.join(d, "checkpoints", f"fold_{fold}")):
            matches.append(d)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def evaluate_alignment(config) -> "pd.DataFrame":  # noqa: F821
    """Run the alignment eval over all folds; return a tidy per-record DataFrame."""
    import torch
    import torch.nn.functional as F
    import pandas as pd
    from pytorch_lightning import seed_everything

    from project.cross_validation import DefineCrossValidation, class_num_mapping_Dict
    from project.dataloader.data_loader import WalkDataModule
    from project.trainer.mid.train_pose_attn import PoseAttnTrainer

    seed_everything(42, workers=True)

    run_glob = config.align.run_glob
    clinical_joints = list(config.align.get("clinical_joints", []))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_map = class_num_mapping_Dict[int(config.model.model_class_num)]

    fold_dataset_idx = DefineCrossValidation(config)()
    records: List[Dict] = []

    for fold in fold_dataset_idx.keys():
        fold = int(fold)
        run_dir = _resolve_run_dir(run_glob, fold)
        if run_dir is None:
            logger.warning(f"[fold {fold}] no run dir with checkpoints under {run_glob}; skipped")
            continue
        ckpt = _find_best_ckpt(run_dir, fold)
        if ckpt is None:
            logger.warning(f"[fold {fold}] no checkpoint in {run_dir}; skipped")
            continue
        logger.info(f"[fold {fold}] loading {ckpt}")

        module = PoseAttnTrainer.load_from_checkpoint(ckpt, map_location=device)
        module.eval().to(device)

        dm = WalkDataModule(config, fold_dataset_idx[fold])
        dm.setup("test")

        with torch.no_grad():
            for batch in dm.test_dataloader():
                video = batch["video"].to(device)
                doctor = batch["attn_map"].to(device)  # (N,J,T,H,W) soft ROI
                labels = batch["label"].tolist()

                out = module.model(video, doctor, return_aux=True)
                _, aux = out if isinstance(out, tuple) else (out, {"side_preds": []})
                side_preds = aux.get("side_preds", [])

                for layer_i, Pi in enumerate(side_preds):
                    Ti, Hi, Wi = Pi.shape[2:]
                    Ai = F.interpolate(doctor, size=(Ti, Hi, Wi),
                                       mode="trilinear", align_corners=False)
                    Sig = torch.sigmoid(Pi)  # (N,J,Ti,Hi,Wi) saliency prob

                    N, J = Sig.shape[0], Sig.shape[1]
                    for n in range(N):
                        cls = class_map.get(int(labels[n]), str(labels[n]))
                        for j in range(J):
                            # per-joint, averaged over frames of this clip
                            frame_metrics: List[Dict[str, float]] = []
                            for t in range(Ti):
                                m = all_metrics(
                                    Sig[n, j, t].cpu().numpy(),
                                    Ai[n, j, t].cpu().numpy(),
                                )
                                frame_metrics.append(m)
                            rec = {
                                "fold": fold, "cls": cls, "layer": layer_i, "joint": j,
                                "clinical": (j in clinical_joints) if clinical_joints else None,
                            }
                            for key in ("cc", "nss", "pg", "auc", "iou", "dice", "kl", "sim"):
                                vals = [fm[key] for fm in frame_metrics
                                        if fm[key] is not None and not np.isnan(fm[key])]
                                rec[key] = float(np.mean(vals)) if vals else float("nan")
                            records.append(rec)

    return pd.DataFrame.from_records(records)


def summarize(df) -> "pd.DataFrame":  # noqa: F821
    """Aggregate the per-record table to mean±std by (cls, layer)."""
    import pandas as pd  # noqa: F401
    metric_cols = ["cc", "nss", "pg", "auc", "iou", "dice", "kl", "sim"]
    g = df.groupby(["cls", "layer"])[metric_cols]
    return g.agg(["mean", "std"])


# ============================ self-test ============================

def _selftest() -> int:
    """Numeric sanity checks on the metric functions (no data needed)."""
    rng = np.random.RandomState(0)
    H = W = 32
    yy, xx = np.mgrid[0:H, 0:W]

    def gauss(cy, cx, s=4.0):
        return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))

    doc = gauss(8, 8)                       # doctor ROI: peak at (8,8)
    same = doc.copy()                       # perfect match
    shifted = gauss(24, 24)                 # peak elsewhere -> should miss
    uniform = np.ones((H, W)) * 0.5         # diffuse -> low NSS, PG chance
    noise = rng.rand(H, W)                  # unrelated

    fails = []

    def check(name, cond):
        print(f"  [{'OK' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    print("self-test: perfect match")
    check("cc(same,doc) ~ 1", abs(cc(same, doc) - 1.0) < 1e-6)
    check("kl(same,doc) ~ 0", kl_div(same, doc) < 1e-6)
    check("sim(same,doc) ~ 1", sim(same, doc) > 0.99)
    check("pg(same,doc) == 1 (peak in ROI)", pointing_game(same, doc) == 1.0)
    check("auc(same,doc) > 0.99", auc_judd(same, doc) > 0.99)
    check("iou(same,doc) > 0.3", topk_overlap(same, doc)[0] > 0.3)

    print("self-test: shifted (peak outside ROI)")
    check("pg(shifted,doc) == 0", pointing_game(shifted, doc) == 0.0)
    check("cc(shifted,doc) < cc(same,doc)", cc(shifted, doc) < cc(same, doc))

    print("self-test: diffuse / unrelated")
    check("nss(uniform,doc) ~ 0", abs(nss(uniform, doc)) < 1e-6)
    check("nss(same,doc) > 1 (peaked at ROI)", nss(same, doc) > 1.0)
    check("cc(noise,doc) near 0", abs(cc(noise, doc)) < 0.2)
    check("auc(noise,doc) near 0.5", abs(auc_judd(noise, doc) - 0.5) < 0.15)

    print("self-test: anti-correlation")
    check("cc(1-doc, doc) < 0", cc(1.0 - doc, doc) < 0)

    print("self-test: empty ROI handled")
    check("pg(same, zeros) is None", pointing_game(same, np.zeros((H, W))) is None)
    check("nss(same, zeros) is nan", np.isnan(nss(same, np.zeros((H, W)))))

    print(f"\n{'ALL PASSED' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return 1 if fails else 0


def _main():
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
    def run(config):
        if "align" not in config:
            OmegaConf.set_struct(config, False)
            config.align = OmegaConf.create({"run_glob": "logs/train/pose_gated_full_f*/**",
                                             "clinical_joints": []})
        df = evaluate_alignment(config)
        if df.empty:
            logger.error("No records produced — check align.run_glob points at trained folds.")
            return
        out_dir = os.path.join("analysis", "alignment_out")
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, "alignment_per_record.csv"), index=False)
        summarize(df).to_csv(os.path.join(out_dir, "alignment_summary.csv"))
        logger.info(f"Saved alignment tables to {out_dir}")
        print(summarize(df))

    run()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    _main()
