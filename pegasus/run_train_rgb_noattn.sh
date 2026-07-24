#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                           # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N baseline_rgb_noattn_train     # 🏷 B2: RGB-only, attn_map=False
#PBS -t 0-2                           # job array: fold index (3-fold, 每折一个 node)
#PBS -o logs/pegasus/train_rgb_noattn_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_rgb_noattn_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

fold=${PBS_SUBREQNO}

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: ${PBS_SUBREQNO} → fold ${fold} / 3-fold"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=none train.attn_map=False \
    train.fold=3 train.fold_idx=${fold} \
    train.gpu=1 \
    train.experiment=baseline_rgb_noattn_f${fold} \
    data.num_workers=$(( $(nproc) / 3 ))


# Experiment notes
###############################################################################
# 实验编号: B2 — RGB-only, no attn map (Table X Row 2)
# 与 B1 的差异: dataloader 不加载 attention map (train.attn_map=False)，
# 验证 prior 输入通路本身是否引入信息。
# 注意: 这不是 skeleton-only。若论文需要 skeleton-only，需要新增
# dataloader/model 支持仅骨架输入。
###############################################################################
