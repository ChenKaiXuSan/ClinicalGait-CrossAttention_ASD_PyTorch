"""
File: analysis/attention_perturbation.py
Created: 2026-07-28
-----
Clinical-prior NECESSITY control (reviewer ask #1), test-time version.

Does PoseGated actually USE the doctor attention map, or is the gain just extra
capacity? We take the trained PoseGated (multi-[0,1]) and, at test time, replace
the doctor ROI with:
    real     : the true doctor attention map (reference)
    shuffled : another clip's ROI (breaks the video<->ROI correspondence)
    zero     : an all-zero map (no prior at all)
and measure pooled clip accuracy per condition. If accuracy is unchanged under
zero/shuffled, the prior is not being used; a drop shows the prior contributes.

This is a TEST-TIME probe (no retraining) on a stratified subsample of each
fold's test set (CPU-friendly). A complementary TRAIN-time control (train with
shuffled/zero maps) is provided as a Pegasus script for the cluster.

Usage:
    python -m analysis.attention_perturbation \
        data.root_path=/work/SKIING/chenkaixu/data/asd_dataset +pert.per_fold=750
"""

from __future__ import annotations

import logging
import numpy as np

from analysis.attention_alignment import _resolve_run_dir, _find_best_ckpt

logger = logging.getLogger(__name__)
TAG = "pose_atn_multi_P1"


def _derangement(n, gen):
    """A permutation with no fixed point (each clip gets a DIFFERENT clip's ROI)."""
    import torch
    if n == 1:
        return torch.tensor([0])
    perm = torch.randperm(n, generator=gen)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
    return perm


def run(config):
    import torch
    from torch.utils.data import Subset, DataLoader
    from pytorch_lightning import seed_everything
    from project.cross_validation import DefineCrossValidation
    from project.dataloader.data_loader import WalkDataModule
    from project.trainer.mid.train_pose_attn import PoseAttnTrainer

    seed_everything(42, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(0)
    per_fold = int(config.get("pert", {}).get("per_fold", 750))
    tag = str(config.get("pert", {}).get("tag", TAG))   # clean re-run: +pert.tag=heldout_pose_multi01

    fold_idx = DefineCrossValidation(config)()
    conds = ["real", "shuffled", "zero"]
    tally = {c: [] for c in conds}   # per-fold accuracy

    for fold_key in sorted(fold_idx.keys(), key=int):
        fold = int(fold_key)
        run_dir = _resolve_run_dir(f"logs/train/{tag}_f*/**", fold)
        ckpt = _find_best_ckpt(run_dir, fold) if run_dir else None
        if not ckpt:
            logger.warning(f"[fold {fold}] no ckpt; skipped"); continue
        module = PoseAttnTrainer.load_from_checkpoint(ckpt, map_location=device).eval().to(device)
        # accuracy only needs the classification path — skip side-head maps (big
        # intermediate tensors) so CPU eval stays within memory. Logits unchanged.
        if hasattr(module.model, "use_side"):
            module.model.use_side = False
        torch.set_num_threads(4)

        dm = WalkDataModule(config, fold_idx[fold_key])
        dm.setup("test")
        full = dm.test_dataloader().dataset
        # evenly-spaced subsample across the (class-ordered) test set -> stratified-ish
        sub_idx = np.linspace(0, len(full) - 1, min(per_fold, len(full))).astype(int)
        loader = DataLoader(Subset(full, sub_idx.tolist()), batch_size=8, shuffle=False,
                            num_workers=0)

        corr = {c: 0 for c in conds}; tot = 0
        import gc
        with torch.no_grad():
            for batch in loader:
                video = batch["video"].to(device)
                attn = batch["attn_map"].to(device)
                y = batch["label"].to(device)
                n = video.shape[0]; tot += n
                variants = {
                    "real": attn,
                    "shuffled": attn[_derangement(n, gen).to(device)],
                    "zero": torch.zeros_like(attn),
                }
                for c, a in variants.items():
                    out = module.model(video, a, return_aux=False)
                    logits = out[0] if isinstance(out, tuple) else out
                    corr[c] += int((logits.argmax(1) == y).sum())
                    del out, logits
                del batch, video, attn
                gc.collect()
        for c in conds:
            acc = 100 * corr[c] / max(tot, 1)
            tally[c].append(acc)
            logger.info(f"[fold {fold}] {c:9s} acc={acc:.1f}  (n={tot})")
        del module

    # summarize
    lines = ["# Attention-perturbation control (test-time, PoseGated multi-[0,1])\n",
             f"Stratified subsample (~{per_fold}/fold). Pooled clip accuracy (%), mean+-std over folds.\n",
             "| Condition | Accuracy (%) | dAcc vs real |", "|---|---|---|"]
    real_mean = float(np.mean(tally["real"])) if tally["real"] else float("nan")
    for c in conds:
        vals = tally[c]
        if not vals:
            continue
        m, s = float(np.mean(vals)), float(np.std(vals))
        lines.append(f"| {c} | {m:.1f} +- {s:.1f} | {m-real_mean:+.1f} |")
    out = "\n".join(lines) + "\n"
    open("analysis/attention_perturbation.md", "w").write(out)
    print(out)
    print("wrote analysis/attention_perturbation.md")


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
