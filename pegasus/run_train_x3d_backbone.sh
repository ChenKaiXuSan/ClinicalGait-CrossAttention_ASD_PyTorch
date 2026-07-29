#!/bin/bash
#PBS -A SKIING                        # 项目名
#PBS -q gpu                           # 队列
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00
#PBS -N x3d_backbone_train            # #4 backbone-generality + #5 alt published backbone (X3D-M)
#PBS -t 0-8                           # array: 3 configs × 3 folds  (SUBREQNO = cfg*3 + fold)
#PBS -o logs/pegasus/train_x3d_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_x3d_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/ checkpoints/
source pegasus/setup_env.sh

# combined array: outer config index * 3 + fold
cfg=$(( PBS_SUBREQNO / 3 ))
fold=$(( PBS_SUBREQNO % 3 ))
root_path=/work/SKIING/chenkaixu/data/asd_dataset
workers=$(( $(nproc) / 6 ))   # lower: X3D PoseGated fold-0 (largest train split) OOM at nproc/3
X3D_CKPT=checkpoints/X3D_M.pyth       # init_x3d downloads Kinetics X3D-M here if missing

echo "PBS ${PBS_JOBID} sub ${PBS_SUBREQNO} → cfg ${cfg}, fold ${fold} / 3-fold  (X3D-M backbone)"

# --- three configs to test whether the SlowR50 findings transfer to X3D ---
# cfg 0: X3D RGB baseline (#5, published-architecture baseline)
# cfg 1: X3D PoseGated shallow multi-[0,1] (#4, our best config on the new backbone)
# cfg 2: X3D PoseGated full   multi-[0-4] (#4, the "fuse-everywhere" point)
case ${cfg} in
  0) fuse=none;     layers=0; abl=single; tag=x3d_baseline ;;
  1) fuse=pose_atn; layers=1; abl=multi;  tag=x3d_pose_multi01 ;;
  2) fuse=pose_atn; layers=4; abl=multi;  tag=x3d_pose_full ;;
  *) echo "bad cfg ${cfg}"; exit 1 ;;
esac

python -m project.train data.root_path=${root_path} \
    model.backbone_net=x3d_m model.ckpt_path=${X3D_CKPT} \
    model.fuse_method=${fuse} model.fusion_layers=${layers} model.ablation_study=${abl} \
    train.fold=3 train.fold_idx=${fold} train.gpu=1 \
    train.experiment=${tag}_f${fold} \
    data.num_workers=${workers}

###############################################################################
# #4 backbone generality + #5 alternative published backbone (X3D-M, Feichtenhofer 2020).
# Reads whether "shallow fusion > full fusion" and "PoseGated > RGB baseline"
# transfer off SlowR50. Compare within-backbone:
#   x3d_baseline  vs  x3d_pose_multi01  -> does the clinical prior still help?
#   x3d_pose_multi01 vs x3d_pose_full   -> is shallow still better than all-stage?
# Note: X3D-M uses 16-frame clips (== project default uniform_temporal_subsample_num).
# Only fuse=none/pose_atn are wired for x3d_m; input-level fusion (concat) stays SlowR50.
###############################################################################
