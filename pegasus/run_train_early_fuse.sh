#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                           # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N early_fuse_train              # 🏷 A1a-c: Early fusion (add/mul/concat)
#PBS -t 0-8                           # job array: method(3) × fold(3)
#PBS -o logs/pegasus/train_early_fuse_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_early_fuse_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

# array 展开: SUBREQNO = method*3 + fold
methods=(add mul concat)
method_idx=$(( PBS_SUBREQNO / 3 ))
fold=$(( PBS_SUBREQNO % 3 ))
method=${methods[$method_idx]}

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: ${PBS_SUBREQNO} → early fusion '${method}', fold ${fold} / 3-fold"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=${method} \
    train.fold=3 train.fold_idx=${fold} \
    train.gpu=1 \
    train.experiment=early_${method}_f${fold} \
    data.num_workers=$(( $(nproc) / 3 ))


# Experiment notes
###############################################################################
# 实验编号: A1a-c — Early fusion 对比方法: 输入端 add / mul / concat attn map
# 走 EarlyFusion3DCNNTrainer + Res3DCNN，仅 cls loss。
# 9 个 sub-job = 3 method × 3 fold。补齐 EXPERIMENTS.md "待补脚本" 的三条腿。
###############################################################################
