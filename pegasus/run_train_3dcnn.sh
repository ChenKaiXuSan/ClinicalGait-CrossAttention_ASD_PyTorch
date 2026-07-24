#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                           # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N baseline_rgb_train            # 🏷 B1: RGB-only baseline
#PBS -t 0-2                           # job array: fold index (3-fold, 每折一个 node)
#PBS -o logs/pegasus/train_3dcnn_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_3dcnn_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

fold=${PBS_SUBREQNO}

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: ${PBS_SUBREQNO} → fold ${fold} / 3-fold"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=none \
    train.fold=3 train.fold_idx=${fold} \
    train.gpu=1 \
    train.experiment=baseline_rgb_f${fold} \
    data.num_workers=$(( $(nproc) / 3 ))


# Experiment notes
###############################################################################
# 实验编号: B1 — Baseline (Table X Row 1)
# 纯 RGB SlowR50，无 skeleton prior，无 side head，仅 cls loss。
# array 维度 = fold (0-2)，每折独占一个 node，24h walltime 内单折稳定完成。
# 提交前先构建 fold 缓存: python -m project.prepare_folds data.root_path=...
###############################################################################
