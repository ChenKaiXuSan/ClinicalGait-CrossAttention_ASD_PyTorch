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
    "heldout_baseline",        # RGB-only backbone
    "heldout_early_concat",    # early concat fusion
    "heldout_pose_multi01",    # PoseGated multi [0,1]
    "heldout_pose_single_L3",  # PoseGated single L3 (best)
]


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def _run(config):
    OmegaConf.set_struct(config, False)
    config.data.heldout_test = True
    A.METHODS = HELDOUT_METHODS  # run() reads this module-level global
    A.run(config)


if __name__ == "__main__":
    _run()
