#!/bin/bash
###############################################################################
# 实验编号: Ablation A5 — PoseGated Single Layer (Table X Row 4-8)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 对比目标: "PoseGated fusion 应该在哪个层注入？" — ablation study of
#           fusion layer index for the proposed method.
#
# PoseAttnFusion 机制（论文核心贡献）:
#   out = Norm( RGB_feat × g + skeleton_feat × (1-g) )
#   where g = sigmoid(GateNetwork(cat(RGB_feat, skeleton_feat)) / temp)
#
# Gate 网络结构:
#   cat([RGB_feat, skeleton_feat]) → Conv3d(2C→C, 1x1x1)
#                               → ReLU → GroupNorm → Conv3d(C→C, 1x1x1)
#                               → sigmoid (g ∈ [0, 1], per-channel)
#   gate_init_bias = 2.0 → sigmoid(2.0) ≈ 0.88 → 训练初期优先信赖 RGB
#
# PBS_SUBREQNO 含义:
#   0 = fusion at blocks[0] ONLY (stem, 64ch)    ← A5a
#   1 = fusion at blocks[1] ONLY (layer1, 256ch)  ← A5b
#   2 = fusion at blocks[2] ONLY (layer2, 512ch)  ← A5c
#   3 = fusion at blocks[3] ONLY (layer3, 1024ch) ← A5d
#   4 = fusion at blocks[4] ONLY (layer4, 2048ch) ← A5e
#
# 消融矩阵 (single layer = 一个脚本跑一个 PBS job):
#   fuse_method      = pose_atn
#   ablation_study   = single                    ← 单层融合
#   fusion_layers    = ${PBS_SUBREQNO}           ← 0~4
#   use_side_heads   = True                      ← 有 side head (默认)
#   loss_selection   = ["cls","attn_loss","bg","tmp"] ← 完整多任务损失
#   gate_init_bias   = 2.0                       ← 默认值
#
# ✗ vs multi: multi=[0,1,2,3,4] 是 "所有层同时融合"，single 只选一层做 ablation
#
# 论文报告方式:
#   Table X Row A5a-e — "PoseGated fusion at layer i (single)"
#   Figure Y — line chart: x=layer index, y=accuracy → 观察最佳 fusion 层
#   → 证明多层融合优于单层（后续 multi 脚本验证）
###############################################################################

#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                         # ✅ 队列名（gpu / debug / gen_S）
#PBS -l elapstim_req=24:00:00          # ⏱ 运行时间限制（最多 24 小时）
#PBS -N pose_atn_single_train         # 🏷 作业名称 — PoseGated Single Layer
#PBS -t 0-4                           # job array 0-4 (各跑一个 layer)
#PBS -o logs/pegasus/train_pose_atn_single_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_pose_atn_single_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: $PBS_SUBREQNO (fusion layer index for PoseGated single)"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

# Ablation A5a-e: PoseGated single layer fusion at block ${PBS_SUBREQNO}
python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn train.fold=5 \
    train.experiment=pose_atn_single_${PBS_SUBREQNO} \
    data.batch_size=64 \
    data.num_workers=$(( $(nproc) / 3 )) \
    model.fusion_layers=${PBS_SUBREQNO} model.ablation_study=single
