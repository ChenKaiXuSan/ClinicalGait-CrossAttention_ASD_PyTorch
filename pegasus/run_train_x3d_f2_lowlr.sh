#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N x3d_f2_lowlr                  # option B diagnostic: rescue the stuck X3D fold-2 with lower LR
#PBS -t 0-1                           # 0=x3d_baseline f2, 1=x3d_pose_multi01 f2
#PBS -o logs/pegasus/train_x3d_f2lowlr_out.log
#PBS -e logs/pegasus/train_x3d_f2lowlr_err.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/ checkpoints/
source pegasus/setup_env.sh

root_path=/work/SKIING/chenkaixu/data/asd_dataset
X3D_CKPT=checkpoints/X3D_M.pyth
fold=2
workers=4
# Option A (train longer) did NOT help: both folds still plateaued at ~0.71.
# Seed is fixed, so this changes the optimisation trajectory: Adam LR 1e-4 -> 5e-5.
# DIAGNOSTIC on the 2 stuck folds only. If they now reach ~0.95 -> rerun the whole
# X3D matrix at 5e-5 for a fair (same-hyperparameter) comparison. If still stuck
# at ~0.71 -> give up on X3D (option C, backbone generality -> future work).
case ${PBS_SUBREQNO} in
  0) fuse=none;     layers=0; abl=single; tag=x3d_baseline ;;
  1) fuse=pose_atn; layers=1; abl=multi;  tag=x3d_pose_multi01 ;;
  *) echo "bad ${PBS_SUBREQNO}"; exit 1 ;;
esac

echo "LOWLR(B) ${tag} fold ${fold}: loss.lr=5e-5, patience=20, max_epochs=80, workers=${workers}"

python -m project.train data.root_path=${root_path} \
    model.backbone_net=x3d_m model.ckpt_path=${X3D_CKPT} \
    model.fuse_method=${fuse} model.fusion_layers=${layers} model.ablation_study=${abl} \
    loss.lr=5e-5 \
    train.fold=3 train.fold_idx=${fold} train.gpu=1 \
    train.max_epochs=80 +train.early_stop_patience=20 \
    train.experiment=${tag}_f${fold} \
    data.num_workers=${workers}
