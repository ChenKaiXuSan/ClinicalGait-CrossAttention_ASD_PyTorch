"""
File: analysis/decision_rule_heldout.py
Created: 2026-09-03
-----
VALIDATION-tuned decision rules for the clean held-out runs (post-hoc, no re-training).

The default argmax rule of the 3-class models collapses onto the majority class
(ASD recall ~94%, DHS ~56%, LCS-HipOA 0%). This probes how much of that is the
DECISION RULE rather than the representation, without touching the test set:

  per fold
    1. run the fold's best checkpoint on the INNER VALIDATION split (patient-disjoint
       from test), aggregate chunk softmax -> clip mean probability;
    2. tune on VAL only:
         (a) 3-class prior correction: predict argmax(p * w) with class weights w
             chosen to maximise VAL balanced accuracy (mean per-class recall);
         (b) binary screening threshold t on P(ASD) maximising VAL balanced accuracy;
    3. apply the tuned rules to the SAVED TEST predictions (best_preds) at clip level.

Reports default vs tuned: per-class recall, balanced accuracy, macro-F1, accuracy
(3-class) and sensitivity / specificity / balanced accuracy (binary), per fold and
mean +- std. Writes analysis/decision_rule_heldout.md.

Run (GPU job recommended for the validation inference):
    python -m analysis.decision_rule_heldout \
        data.root_path=/work/SKIING/chenkaixu/data/asd_dataset +dr.tag=heldout_pose_single_L3
"""

from __future__ import annotations

import glob
import logging
from collections import defaultdict

import numpy as np

from analysis.attention_alignment import _resolve_run_dir, _find_best_ckpt

logger = logging.getLogger(__name__)
ASD = 0
CLASSES = ["ASD", "DHS", "LCS-HipOA"]


def _load_test_probs(tag, fold):
    import torch
    p = sorted(glob.glob(f"logs/train/{tag}_f{fold}/*/*/best_preds/fold_{fold}_pred.pt"))
    y = sorted(glob.glob(f"logs/train/{tag}_f{fold}/*/*/best_preds/fold_{fold}_label.pt"))
    if not p or not y:
        return None, None
    return (torch.load(p[-1], map_location="cpu").numpy(),
            torch.load(y[-1], map_location="cpu").numpy())


def _to_clips(probs, labels, index_map):
    """chunk probs -> clip mean prob, using the dataset _index_map order."""
    n = len(probs)
    meta = [(e["video_name"], int(e["label"])) for e in index_map][:n]
    assert np.array_equal(np.array([m[1] for m in meta]), labels[:n]), "label/index mismatch"
    acc = defaultdict(lambda: [np.zeros(probs.shape[1]), 0, None])
    for i in range(n):
        k = meta[i][0]; acc[k][0] += probs[i]; acc[k][1] += 1; acc[k][2] = int(labels[i])
    P = np.stack([v[0] / v[1] for v in acc.values()])
    Y = np.array([v[2] for v in acc.values()])
    return P, Y


def _recalls(y, yhat, k=3):
    return np.array([(yhat[y == c] == c).mean() * 100 if (y == c).any() else np.nan for c in range(k)])


def _m3(y, yhat):
    from sklearn.metrics import f1_score
    r = _recalls(y, yhat)
    return {"acc": (y == yhat).mean() * 100, "bacc": np.nanmean(r),
            "f1": f1_score(y, yhat, average="macro", zero_division=0) * 100,
            "r_asd": r[0], "r_dhs": r[1], "r_lcs": r[2]}


def _m2(y, s, t):
    yb = (y == ASD).astype(int); pb = (s >= t).astype(int)
    tp = ((yb == 1) & (pb == 1)).sum(); fn = ((yb == 1) & (pb == 0)).sum()
    tn = ((yb == 0) & (pb == 0)).sum(); fp = ((yb == 0) & (pb == 1)).sum()
    sens = tp / max(tp + fn, 1) * 100; spec = tn / max(tn + fp, 1) * 100
    return {"sens": sens, "spec": spec, "bacc": (sens + spec) / 2}


def _tune_weights(Pv, Yv):
    """grid over class weights (w_ASD=1) maximising VAL balanced accuracy."""
    grid = np.logspace(-1, 2, 31)
    best, best_w = -1, np.ones(3)
    for w1 in grid:
        for w2 in grid:
            w = np.array([1.0, w1, w2])
            b = np.nanmean(_recalls(Yv, (Pv * w).argmax(1)))
            if b > best + 1e-9:
                best, best_w = b, w
    return best_w, best


def _tune_threshold(Pv, Yv):
    s = Pv[:, ASD]
    cands = np.unique(np.concatenate([[0.0, 1.0], s]))
    best, best_t = -1, 0.5
    for t in cands:
        b = _m2(Yv, s, t)["bacc"]
        if b > best + 1e-9:
            best, best_t = b, t
    return best_t, best


def run(config):
    import torch
    from torch.utils.data import DataLoader
    from pytorch_lightning import seed_everything
    from project.cross_validation import DefineCrossValidation
    from project.dataloader.data_loader import WalkDataModule
    from project.trainer.mid.train_pose_attn import PoseAttnTrainer

    tag = str(config.get("dr", {}).get("tag", "heldout_pose_single_L3"))
    workers = int(config.data.get("num_workers", 4))
    seed_everything(42, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_idx = DefineCrossValidation(config)()

    rows = []
    for fk in sorted(fold_idx.keys(), key=int):
        fold = int(fk)
        run_dir = _resolve_run_dir(f"logs/train/{tag}_f*/**", fold)
        ckpt = _find_best_ckpt(run_dir, fold) if run_dir else None
        if not ckpt:
            logger.warning(f"[fold {fold}] no ckpt for {tag}; skipped"); continue
        module = PoseAttnTrainer.load_from_checkpoint(ckpt, map_location=device).eval().to(device)
        if hasattr(module.model, "use_side"):
            module.model.use_side = False

        dm = WalkDataModule(config, fold_idx[fk])
        dm.setup("fit")
        val_ds = dm.val_dataloader().dataset
        loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=workers)
        vp, vy = [], []
        with torch.no_grad():
            for batch in loader:
                out = module.model(batch["video"].to(device), batch["attn_map"].to(device), return_aux=False)
                logits = out[0] if isinstance(out, tuple) else out
                vp.append(torch.softmax(logits, 1).cpu().numpy()); vy.append(batch["label"].numpy())
        vp = np.concatenate(vp); vy = np.concatenate(vy)
        Pv, Yv = _to_clips(vp, vy, val_ds._index_map)

        dm.setup("test")
        tp_, ty_ = _load_test_probs(tag, fold)
        Pt, Yt = _to_clips(tp_, ty_, dm.test_dataloader().dataset._index_map)

        w, vb = _tune_weights(Pv, Yv)
        t, vb2 = _tune_threshold(Pv, Yv)
        d3 = _m3(Yt, Pt.argmax(1)); t3 = _m3(Yt, (Pt * w).argmax(1))
        d2 = _m2(Yt, Pt[:, ASD], 0.5); tt = _m2(Yt, Pt[:, ASD], t)
        # default binary rule in the paper = argmax==ASD; report that too
        d2a = _m2(Yt, (Pt.argmax(1) == ASD).astype(float), 0.5)
        rows.append({"fold": fold, "w": w, "t": t, "val_bacc3": vb, "val_bacc2": vb2,
                     "d3": d3, "t3": t3, "d2": d2a, "t2": tt,
                     "n_val": len(Yv), "n_test": len(Yt)})
        logger.info(f"[fold {fold}] w={np.round(w,2).tolist()} t={t:.3f} | 3-class bacc {d3['bacc']:.1f} -> {t3['bacc']:.1f}"
                    f" | binary bacc {d2a['bacc']:.1f} -> {tt['bacc']:.1f}")
        del module; torch.cuda.empty_cache() if device.type == "cuda" else None

    if not rows:
        print("no folds processed"); return
    L = [f"# Validation-tuned decision rules ({tag}; clip level; tuned on inner VAL, applied to held-out TEST)\n"]
    L += ["Rules: (a) 3-class argmax(p*w), w tuned on val for balanced accuracy; (b) binary P(ASD)>=t, t tuned on val.",
          "Default = plain argmax (paper). All numbers %.\n"]
    L += ["| fold | w (ASD,DHS,LCS) | t | 3-cls bacc def→tuned | macro-F1 def→tuned | acc def→tuned | recall ASD/DHS/LCS def | recall ASD/DHS/LCS tuned | binary sens/spec/bacc def | binary sens/spec/bacc tuned |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['fold']} | {np.round(r['w'],2).tolist()} | {r['t']:.3f} | {r['d3']['bacc']:.1f}→{r['t3']['bacc']:.1f} | "
                 f"{r['d3']['f1']:.1f}→{r['t3']['f1']:.1f} | {r['d3']['acc']:.1f}→{r['t3']['acc']:.1f} | "
                 f"{r['d3']['r_asd']:.0f}/{r['d3']['r_dhs']:.0f}/{r['d3']['r_lcs']:.0f} | "
                 f"{r['t3']['r_asd']:.0f}/{r['t3']['r_dhs']:.0f}/{r['t3']['r_lcs']:.0f} | "
                 f"{r['d2']['sens']:.0f}/{r['d2']['spec']:.0f}/{r['d2']['bacc']:.1f} | "
                 f"{r['t2']['sens']:.0f}/{r['t2']['spec']:.0f}/{r['t2']['bacc']:.1f} |")

    def ms(key_fn):
        v = np.array([key_fn(r) for r in rows]); return f"{v.mean():.1f} ± {v.std():.1f}"
    L += ["", "**Mean ± std over folds (TEST):**", "",
          "| metric | default | val-tuned |", "|---|---|---|",
          f"| 3-class balanced acc | {ms(lambda r: r['d3']['bacc'])} | {ms(lambda r: r['t3']['bacc'])} |",
          f"| 3-class macro-F1 | {ms(lambda r: r['d3']['f1'])} | {ms(lambda r: r['t3']['f1'])} |",
          f"| 3-class accuracy | {ms(lambda r: r['d3']['acc'])} | {ms(lambda r: r['t3']['acc'])} |",
          f"| recall ASD | {ms(lambda r: r['d3']['r_asd'])} | {ms(lambda r: r['t3']['r_asd'])} |",
          f"| recall DHS | {ms(lambda r: r['d3']['r_dhs'])} | {ms(lambda r: r['t3']['r_dhs'])} |",
          f"| recall LCS-HipOA | {ms(lambda r: r['d3']['r_lcs'])} | {ms(lambda r: r['t3']['r_lcs'])} |",
          f"| binary sensitivity | {ms(lambda r: r['d2']['sens'])} | {ms(lambda r: r['t2']['sens'])} |",
          f"| binary specificity | {ms(lambda r: r['d2']['spec'])} | {ms(lambda r: r['t2']['spec'])} |",
          f"| binary balanced acc | {ms(lambda r: r['d2']['bacc'])} | {ms(lambda r: r['t2']['bacc'])} |"]
    out = "\n".join(L) + "\n"
    open("analysis/decision_rule_heldout.md", "w").write(out)
    print(out); print("wrote analysis/decision_rule_heldout.md")


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
