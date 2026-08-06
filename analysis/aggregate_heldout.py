"""
File: analysis/aggregate_heldout.py
-----
Clip/patient-level aggregation for the CLEAN held-out re-run (job 890178).

Reuses analysis.aggregate_levels.run() but (1) points METHODS at the heldout
experiment tags and (2) forces data.heldout_test=True so the rebuilt test index
is the true held-out set (patient-disjoint, no magic_move leakage). This lets us
report the clean numbers at the same gait-CLIP granularity as the group's prior
3-class work (~62-71%).

Run:
    python -m analysis.aggregate_heldout \
        data.root_path=/work/SKIING/chenkaixu/data/asd_dataset
"""

from __future__ import annotations

import hydra
from omegaconf import OmegaConf

import analysis.aggregate_levels as A

HELDOUT_METHODS = [
    # --- main-comparison anchors ---
    "heldout_baseline",        # RGB-only backbone
    "heldout_early_concat",    # early concat fusion
    # --- A5 fusion location/depth sweep (single Li, multi prefix Pi) ---
    "heldout_pose_single_L0",
    "heldout_pose_single_L1",
    "heldout_pose_single_L2",
    "heldout_pose_single_L3",  # single L3 (clean best so far)
    "heldout_pose_single_L4",
    "heldout_pose_multi01",    # multi [0,1] (P1)
    "heldout_pose_multi_P2",   # multi [0,1,2]
    "heldout_pose_multi_P3",   # multi [0,1,2,3]
    "heldout_pose_multi_P4",   # multi [0,1,2,3,4] (all-stage)
    # --- A2/A3/A4/A6 ablations, anchored on single-L3 ---
    "heldout_ab_bias0",        # gate bias 0.0
    "heldout_ab_biasneg1",     # gate bias -1.0
    "heldout_ab_nobg",         # - bg loss
    "heldout_ab_notmp",        # - tmp loss
    "heldout_ab_noside",       # - side heads
    "heldout_ab_gateadd",      # gate_mode=add (plain injection)
    "heldout_ab_gatefixed",    # gate_mode=fixed (0.5 mix)
    # --- A2/A3/A4/A6 ablations re-anchored on multi-[0,1] (companion set) ---
    "heldout_abm_bias0",
    "heldout_abm_biasneg1",
    "heldout_abm_nobg",
    "heldout_abm_notmp",
    "heldout_abm_noside",
    "heldout_abm_gateadd",
    "heldout_abm_gatefixed",
]


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def _run(config):
    OmegaConf.set_struct(config, False)
    config.data.heldout_test = True
    A.METHODS = HELDOUT_METHODS  # run() reads this module-level global
    A.run(config)


if __name__ == "__main__":
    _run()
