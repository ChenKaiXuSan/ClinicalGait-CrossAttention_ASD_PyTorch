#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -b 1
#PBS -l elapstim_req=24:00:00
#PBS -N x3d_f2_rerun                  # option A: rerun the 2 non-converged X3D fold-2 jobs, train longer
#PBS -t 0-1                           # 0=x3d_baseline f2, 1=x3d_pose_multi01 f2
#PBS -o logs/pegasus/train_x3d_f2rerun_out.log
#PBS -e logs/pegasus/train_x3d_f2rerun_err.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch
mkdir -p logs/pegasus/ checkpoints/
source pegasus/setup_env.sh

root_path=/work/SKIING/chenkaixu/data/asd_dataset
X3D_CKPT=checkpoints/X3D_M.pyth
fold=2
workers=4
# fold 2 of x3d_baseline (val 0.71) and x3d_pose_multi01 (val 0.72) got stuck /
# early-stopped while the other folds hit 0.95+. Seed is fixed (42), so we only
# train LONGER: patience 10->20, max_epochs 50->80 (x3d_pose_full f2 needed
# epoch 48 to reach 0.966). If they still plateau at ~0.71 -> genuine bad basin,
# escalate to option B (LR/warmup).
case ${PBS_SUBREQNO} in
  0) fuse=none;     layers=0; abl=single; tag=x3d_baseline ;;
  1) fuse=pose_atn; layers=1; abl=multi;  tag=x3d_pose_multi01 ;;
  *) echo "bad ${PBS_SUBREQNO}"; exit 1 ;;
esac

echo "RERUN(A) ${tag} fold ${fold}: patience=20, max_epochs=80, workers=${workers}"

python -m project.train data.root_path=${root_path} \
    model.backbone_net=x3d_m model.ckpt_path=${X3D_CKPT} \
    model.fuse_method=${fuse} model.fusion_layers=${layers} model.ablation_study=${abl} \
    train.fold=3 train.fold_idx=${fold} train.gpu=1 \
    train.max_epochs=80 +train.early_stop_patience=20 \
    train.experiment=${tag}_f${fold} \
    data.num_workers=${workers}
