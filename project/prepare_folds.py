"""
File: prepare_folds.py
Project: project
Created Date: 2026-07-24
Author: Kaixu Chen
-----
Comment:
 Pre-build the K-fold index cache (fold split + per-fold video copies)
 without training. Run this ONCE before submitting parallel per-fold PBS
 array jobs, so the sub-jobs only load the cache instead of one of them
 building it while the others wait:

    python -m project.prepare_folds data.root_path=/path/to/data

 Uses the same configs/config.yaml as project.train, so train.fold /
 data.sampling / model.model_class_num overrides work identically.

Have a good code time!
-----
"""

import logging

import hydra
from omegaconf import DictConfig

from project.cross_validation import DefineCrossValidation

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config.yaml",
)
def main(config: DictConfig):
    fold_dataset_idx = DefineCrossValidation(config)()

    for fold, v in fold_dataset_idx.items():
        logger.info(f"fold {fold}: train samples={len(v[0])}, val samples={len(v[1])}")

    logger.info(
        f"fold cache ready: K={config.train.fold}, "
        f"sampling={config.data.sampling}, class_num={config.model.model_class_num}"
    )


if __name__ == "__main__":
    main()
