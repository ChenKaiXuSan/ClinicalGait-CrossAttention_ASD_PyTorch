#!/bin/bash
###############################################################################
# 实验编号: Ablation A1 — Cross-Attention Fusion (Table X Row 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 对比目标: "QKV cross-attention" vs "PoseGated gated mixing" — 融合策略对比
#
# CrossAttentionRes3DCNN 的融合机制：在每个 SlowR50 block 后，将 skeleton
# attention map 作为 Key/Value，RGB feature 作为 Query，通过 QK^T 计算
# frame-to-frame self-attention matrix (THW×THW)，然后将注意力权重应用到
# skeleton features 上再加回原始 RGB feature。
#
# CrossAttentionFusion(x, context):
#   Q = Conv3d(RGB_feat)        → [B, THW, C]      ← "what to attend"
#   K = Conv3d(skeleton_map)    → [B, C, THW]       ← "where to attend" (key)
#   V = Conv3d(skeleton_map)    → [B, THW, C]       ← 被检索的内容
#   attn = softmax(QK^T / √C)  → [B, THW, THW]     ← frame-to-frame attention
#   out = bmm(attn, V) + x     → residual
#
# ✗ vs PoseGated 的核心差异:
#   CrossAttention: attention 在时空维度 (THW×THW)，所有通道共享同一注意力图
#                  → "模型关注哪些帧，但每个通道的关注方式一样"
#   PoseGated:      gate 在通道维度 (C维)，空间自适应
#                  → "模型为每个特征通道独立决定信视频还是信骨架"
#
# PBS_SUBREQNO 含义 (job array 0~4):
#   0 = fusion at blocks[0] only (stem, 64ch)
#   1 = fusion at blocks[1] only (layer1, 256ch)   ← cross_atn 原始配置只跑这个
#   2 = fusion at blocks[2] only (layer2, 512ch)
#   3 = fusion at blocks[3] only (layer3, 1024ch)
#   4 = fusion at blocks[4] only (layer4, 2048ch)
#
# ✗ vs PoseGated: pose_gated 的 multi=4 表示 fusion ALL layers [0,1,2,3,4]
#                  cross_atn 目前只跑 single layer，建议也补一个 multi 版本
#
# 消融矩阵:
#   fuse_method      = cross_atn
#   fusion_layers    = ${PBS_SUBREQNO}  (遍历 5 层)
#   use_side_heads   = False
#   loss_selection   = ["cls"]
#   gate_init_bias   = N/A
#
# 论文报告方式:
#   Table I Row 2 — "Cross-Attention Fusion" (Ablation A1)
#   Figure X — fusion layer 选择 vs performance (单层消融图)
###############################################################################

#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gen_S                         # ✅ 队列名（gpu / debug / gen_S）
#PBS -l elapstim_req=24:00:00          # ⏱ 运行时间限制（最多 24 小时）
#PBS -N cross_atn_train               # 🏷 作业名称
#PBS -t 0-4                           # job array 0-4 (5 folds, 遍历融合层)
#PBS -o logs/pegasus/train_cross_atn_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_cross_atn_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
source /work/SKIING/chenkaixu/code/med_atn/bin/activate

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: $PBS_SUBREQNO (fusion layer index)"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

# Ablation A1: Cross-Attention — fusion at block ${PBS_SUBREQNO}
python -m project.train data.root_path=${root_path} \
    model.fuse_method=cross_atn train.fold=5 \
    data.num_workers=$(( $(nproc) / 3 )) \
    model.fusion_layers=${PBS_SUBREQNO}
