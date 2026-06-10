#!/bin/bash
###############################################################################
# 实验编号: Ablation A1 — Squeeze-and-Excitation Fusion (Table X Row 3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 对比目标: "SE channel-wise scale" vs "PoseGated gate mixing" — 两种
#           通道级融合策略哪个更好？
#
# SEFusionRes3DCNN 的融合机制：在每个 SlowR50 block 后，将 skeleton attention
# map squeeze 成 1×1×1 向量（全局池化），然后通过两层 Conv3d (实质是全连接)
# 映射出每个通道的 scale 系数 [0,1]，乘以 RGB feature。
#
# SEFusion(x, context):
#   squeezed = AdaptiveAvgPool3d(context_map)  → [B, C_ctx, 1, 1, 1]
#   scale    = Sigmoid(MLP(squeezed))           → [B, C_rgb, 1, 1, 1]
#   out      = x * scale + x                    ← channel-wise scaling
#
# ✗ vs PoseGated 的核心差异:
#   SE-Fusion:    skeleton map squeeze 成 1×1×1 → "丢失空间信息"
#                 scale 系数对整帧同一通道是常数 → 无法自适应位置
#   PoseGated:    skeleton map 保留 THW 维度 → Gate 输出 (B,C,T,H,W)
#                 每通道不同时空位置有不同权重 → "空间自适应融合"
#
# ✗ vs CrossAttention:
#   SE-Fusion:     fusion 是 channel-wise scalar，无跨帧建模能力
#   CrossAttn:     fusion 是 frame-to-frame attention matrix (THW×THW)
#   PoseGated:     fusion 是 spatially-adaptive gate weights (C维)
#
# PBS_SUBREQNO 含义 (job array 0~4):
#   0 = fusion at blocks[0] only (stem, 64ch)
#   1 = fusion at blocks[1] only (layer1, 256ch)
#   ... → 逐个 ablation layer
#
# 消融矩阵:
#   fuse_method      = se_atn
#   fusion_layers    = ${PBS_SUBREQNO}
#   use_side_heads   = False
#   loss_selection   = ["cls"]
#   gate_init_bias   = N/A
#
# 论文报告方式:
#   Table I Row 3 — "SE-Fusion" (Ablation A1)
#   → 证明 PoseGated 的空间自适应优于 SE 的全局 channel scale
###############################################################################

#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                         # ✅ 队列名（gpu / debug / gen_S）
#PBS -l elapstim_req=24:00:00          # ⏱ 运行时间限制（最多 24 小时）
#PBS -N se_atn_single_train           # 🏷 作业名称
#PBS -t 0-4                           # job array 0-4 (5 folds, 遍历融合层)
#PBS -o logs/pegasus/train_se_atn_single_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_se_atn_single_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: $PBS_SUBREQNO (SE fusion layer index)"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

# Ablation A1: SE-Fusion — fusion at block ${PBS_SUBREQNO}
python -m project.train data.root_path=${root_path} \
    model.fuse_method=se_atn train.fold=5 \
    train.experiment=se_atn_prefix_${PBS_SUBREQNO} \
    data.batch_size=64 \
    data.num_workers=$(( $(nproc) / 3 )) \
    model.fusion_layers=${PBS_SUBREQNO} model.ablation_study=single
