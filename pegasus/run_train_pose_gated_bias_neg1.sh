#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                         # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00          # ⏱ 运行时间限制（最多 24 小时）
#PBS -N pose_gated_bias_neg1_train    # 🏷 Ablation A2b: gate_init_bias = -1.0
#PBS -t 0-4                           # job array 0-4 (5 folds, 遍历融合层)
#PBS -o logs/pegasus/train_pose_gated_biasneg1_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_pose_gated_biasneg1_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: $PBS_SUBREQNO (fusion layer index)"
echo "gate_init_bias = -1.0 (skeleton-biased init: g≈0.27 → trust skeleton at start)"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn train.fold=5 \
    train.gpu=1 \
    train.experiment=pose_atn_bias_neg1_single_${PBS_SUBREQNO} \
    data.batch_size=32 \
    data.num_workers=$(( $(nproc) / 3 )) \
    model.fusion_layers=${PBS_SUBREQNO} model.ablation_study=single \
    model.gate_init_bias=-1.0


# Script notes
# Ablation A2b: PoseGated gate init bias = -1.0 (skeleton-biased: g≈0.27 at t=0)

# Experiment notes
###############################################################################
# 实验编号: Ablation A2 — Gate Init Bias = -1.0 (Table X Row A2b)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 对比目标: "Gate 初始化偏 skeleton" vs "默认偏 RGB(2.0)" vs "无偏(0.0)"
#           — 三组 gate_init_bias 的完整消融。
#
# gate_init_bias = -1.0 的物理含义:
#   g = sigmoid(-1.0) ≈ 0.27 at t=0
#   PoseGated = RGB × 0.27 + Skeleton × 0.73    ← "模型从信任 skeleton 开始"
#
# 临床动机:
#   骨科医生标注的 ROI (腰椎、骨盆、头部、肩部) 是强先验，在早期数据噪声大时，
#   更信赖这些医学标注可能比像素空间特征更有助于收敛方向。
#
# ✗ vs bias=0.0: 本脚本偏 skeleton（g≈0.27），bias=0.0 无偏（g=0.5）
# ✓ vs bias=2.0: 完整三组 (2.0/0.0/-1.0) → 画一张 "gate_init_bias → accuracy" 曲线
#
# 消融矩阵:
#   fuse_method      = pose_atn
#   ablation_study   = single
#   fusion_layers    = ${PBS_SUBREQNO}           ← 遍历 0~4
#   use_side_heads   = True
#   loss_selection   = ["cls","attn_loss","bg","tmp"]
#   gate_init_bias   = -1.0      ← ★ THIS IS THE ABLATION ★
#                                 (偏 skeleton: g≈0.27 at initialization)
#
# 论文报告方式:
#   Table X Row A2b — "PoseGated gate_init_bias=-1.0" (Ablation A2)
#   Figure W — line chart: x=gates_init_bias=[-1, 0, 2], y=accuracy → 三组对比
#   → 预期：bias=2.0 最优（RGB 特征已足够表达，医学先验作为引导而非主导）
###############################################################################
