#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                           # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N cross_atn_train               # 🏷 A1e: QKV Cross-Attention fusion
#PBS -t 0-8                           # job array: config(3: L3/L4/L34) × fold(3)
#PBS -o logs/pegasus/train_cross_atn_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_cross_atn_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

# array 展开: SUBREQNO = cfg*3 + fold
# cross-attention 的 THW×THW 注意力矩阵在浅层不可行:
#   stem/layer1 输出 56×56×16 → THW≈50k → attn 矩阵 ~10GB/样本，必然 OOM
#   layer2 (28×28) 也在边缘 (~630MB/样本)
# 因此只扫深层配置: [3], [4], [3,4]（显式层列表,绕过 int 的 prefix 映射）
cfgs=("[3]" "[4]" "[3,4]")
names=(L3 L4 L34)
cfg_idx=$(( PBS_SUBREQNO / 3 ))
fold=$(( PBS_SUBREQNO % 3 ))
cfg=${cfgs[$cfg_idx]}
name=${names[$cfg_idx]}

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: ${PBS_SUBREQNO} → cross_atn fusion_layers=${cfg}, fold ${fold} / 3-fold"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=cross_atn \
    "model.fusion_layers=${cfg}" \
    train.fold=3 train.fold_idx=${fold} \
    train.gpu=1 \
    train.experiment=cross_atn_${name}_f${fold} \
    data.num_workers=$(( $(nproc) / 3 ))


# Experiment notes
###############################################################################
# 实验编号: A1e — QKV Cross-Attention fusion 对比方法
# CrossAttentionFusion: Q=Conv3d(RGB feat), K/V=Conv3d(attn map)，
# softmax(QK^T/√C) 作用于 V 后残差加回 RGB。attention 在时空维度 (THW×THW)。
# trainer 已接回 project/train.py (CrossAttentionTrainer, 仅 cls loss)。
# 论文 A1 表取 {L3, L4, L3+4} 中的最优点。9 个 sub-job = 3 config × 3 fold。
###############################################################################
