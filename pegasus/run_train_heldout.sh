#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N heldout_clean                 # CLEAN protocol: true held-out test, no magic_move leakage
#PBS -t 0-11                          # array: 4 configs x 3 folds  (SUBREQNO = cfg*3 + fold)
#PBS -o logs/pegasus/train_heldout_out.log
#PBS -e logs/pegasus/train_heldout_err.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/ checkpoints/
source pegasus/setup_env.sh

cfg=$(( PBS_SUBREQNO / 3 ))
fold=$(( PBS_SUBREQNO % 3 ))
root_path=/work/SKIING/chenkaixu/data/asd_dataset
workers=$(( $(nproc) / 4 ))

# Re-run the key methods under the CLEAN evaluation protocol:
#   data.heldout_test=True  -> patient-grouped train/val/TEST, test is a true
#   held-out set (test != val, no magic_move, holdout not over-sampled). Builds a
#   separate  <sampling>_K3_heldout  fold cache (pre-build once with prepare_folds).
# Compare these numbers to the ~62-71% of the group's prior 3-class work
# (KnowledgeGuided) and aggregate to clip level with analysis/aggregate_levels.py.
case ${cfg} in
  0) fuse=none;     layers=0; abl=single; tag=heldout_baseline ;;
  1) fuse=pose_atn; layers=1; abl=multi;  tag=heldout_pose_multi01 ;;
  2) fuse=concat;   layers=0; abl=single; tag=heldout_early_concat ;;
  3) fuse=pose_atn; layers=3; abl=single; tag=heldout_pose_single_L3 ;;
  *) echo "bad cfg ${cfg}"; exit 1 ;;
esac

echo "PBS ${PBS_JOBID} sub ${PBS_SUBREQNO} -> ${tag}, fold ${fold} (CLEAN held-out protocol)"

python -m project.train data.root_path=${root_path} \
    data.heldout_test=True \
    model.fuse_method=${fuse} model.fusion_layers=${layers} model.ablation_study=${abl} \
    train.fold=3 train.fold_idx=${fold} train.gpu=1 \
    train.experiment=${tag}_f${fold} \
    data.num_workers=${workers}

###############################################################################
# Fixes the two inflators found by comparing to the prior 3-class work:
#   (1) test == val  -> now a separate patient-grouped val for early stopping.
#   (2) magic_move leaked ~20/43 test patients into train -> removed; the outer
#       StratifiedGroupKFold split is now patient-disjoint (verified: tr/va/te
#       share 0 patients, each patient tested once).
# Pre-build the clean cache once before submitting the array:
#   python -m project.prepare_folds data.root_path=... data.heldout_test=True
###############################################################################
