#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                           # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N pose_gated_notmploss_train    # 🏷 Ablation A4b: No tmp_loss
#PBS -t 0-2                           # job array: fold index (3-fold, 每折一个 node)
#PBS -o logs/pegasus/train_pose_gated_notmp_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_pose_gated_notmp_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

fold=${PBS_SUBREQNO}

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: ${PBS_SUBREQNO} → fold ${fold} / 3-fold"
echo "Ablation A4b: loss_selection=[cls,attn_loss,bg] → tmp_loss removed"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn \
    model.fusion_layers=4 model.ablation_study=multi \
    'loss.selection=["cls","attn_loss","bg"]' \
    train.fold=3 train.fold_idx=${fold} \
    train.gpu=1 \
    train.experiment=pose_gated_notmp_f${fold} \
    data.num_workers=$(( $(nproc) / 3 ))


# Experiment notes
###############################################################################
# 实验编号: A4b — 损失消融: 去掉 temporal smoothness loss (w_temp 置 0)
# 其余同 pose_gated_full。
###############################################################################
