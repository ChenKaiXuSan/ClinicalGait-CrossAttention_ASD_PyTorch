#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N heldout_location              # A5 fusion location/depth sweep — CLEAN held-out protocol
#PBS -t 0-20                          # array: 7 configs x 3 folds  (SUBREQNO = cfg*3 + fold)
#PBS -o logs/pegasus/train_heldout_loc_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_heldout_loc_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/ checkpoints/
source pegasus/setup_env.sh

cfg=$(( PBS_SUBREQNO / 3 ))
fold=$(( PBS_SUBREQNO % 3 ))
root_path=/work/SKIING/chenkaixu/data/asd_dataset
workers=$(( $(nproc) / 4 ))          # match run_train_heldout.sh (fold-0 held-out split is largest)

# --- Re-run the WHERE-TO-FUSE sweep under the CLEAN held-out protocol -----------
# The leaky-protocol conclusion (shallow [0,1] best, deep/full weak) inverted once
# test!=val and magic_move leakage were removed: single-L3 is now the strongest
# point (69.2% clip). This sweep re-establishes tab:layers cleanly.
#   already done elsewhere:  single L3 = heldout_pose_single_L3
#                            multi  [0,1] (P1) = heldout_pose_multi01
# cfg -> (ablation_study, fusion_layers, tag)
case ${cfg} in
  0) abl=single; layers=0; tag=heldout_pose_single_L0 ;;   # [0]
  1) abl=single; layers=1; tag=heldout_pose_single_L1 ;;   # [1]
  2) abl=single; layers=2; tag=heldout_pose_single_L2 ;;   # [2]
  3) abl=single; layers=4; tag=heldout_pose_single_L4 ;;   # [4]
  4) abl=multi;  layers=2; tag=heldout_pose_multi_P2 ;;     # [0,1,2]
  5) abl=multi;  layers=3; tag=heldout_pose_multi_P3 ;;     # [0,1,2,3]
  6) abl=multi;  layers=4; tag=heldout_pose_multi_P4 ;;     # [0,1,2,3,4] (all-stage)
  *) echo "bad cfg ${cfg}"; exit 1 ;;
esac

echo "PBS ${PBS_JOBID} sub ${PBS_SUBREQNO} -> ${tag} (${abl} ${layers}), fold ${fold} (CLEAN held-out)"

python -m project.train data.root_path=${root_path} \
    data.heldout_test=True \
    model.fuse_method=pose_atn \
    model.fusion_layers=${layers} model.ablation_study=${abl} \
    train.fold=3 train.fold_idx=${fold} train.gpu=1 \
    train.experiment=${tag}_f${fold} \
    data.num_workers=${workers}

###############################################################################
# Combined with the two already-clean points, this yields the full clean sweep:
#   single: L0 L1 L2 [L3*] L4        multi: [L0]=single-L0  [0,1]* P2 P3 P4
#   (* = already trained: heldout_pose_single_L3, heldout_pose_multi01)
# Aggregate to clip level with:
#   python -m analysis.aggregate_heldout data.root_path=... (extend HELDOUT_METHODS)
# Pre-built clean cache required (already present): <sampling>_K3_heldout.
###############################################################################
