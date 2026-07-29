#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N x3d_pose_f0_rerun             # rerun the 2 fold-0 X3D-PoseGated jobs that OOM'd
#PBS -t 0-1                           # 0=multi[0,1], 1=full[0-4]; fold fixed = 0
#PBS -o logs/pegasus/train_x3d_rerun_out.log
#PBS -e logs/pegasus/train_x3d_rerun_err.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/ checkpoints/
source pegasus/setup_env.sh

root_path=/work/SKIING/chenkaixu/data/asd_dataset
X3D_CKPT=checkpoints/X3D_M.pyth
fold=0
# fewer workers: the fold-0 train split is the largest (~30k chunks) and the X3D
# PoseGated path (side heads + fusion) is memory-heavy — 16 workers x per-worker
# video cache blew the job RAM cap. 4 workers keeps it under the limit.
workers=4

case ${PBS_SUBREQNO} in
  0) layers=1; abl=multi; tag=x3d_pose_multi01 ;;
  1) layers=4; abl=multi; tag=x3d_pose_full ;;
  *) echo "bad ${PBS_SUBREQNO}"; exit 1 ;;
esac

echo "RERUN ${tag} fold ${fold} with num_workers=${workers}"

python -m project.train data.root_path=${root_path} \
    model.backbone_net=x3d_m model.ckpt_path=${X3D_CKPT} \
    model.fuse_method=pose_atn model.fusion_layers=${layers} model.ablation_study=multi \
    train.fold=3 train.fold_idx=${fold} train.gpu=1 \
    train.experiment=${tag}_f${fold} \
    data.num_workers=${workers}
