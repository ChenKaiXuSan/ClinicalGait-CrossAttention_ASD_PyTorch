#!/bin/bash
###############################################################################
# 实验编号: Ablation A2 — Gate Init Bias = 0.0 (Table X Row A2a)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 对比目标: "Gate 无偏初始化 vs 默认偏 RGB" — 验证 gate_init_bias 的初始化策略
#           是否影响最终收敛，还是训练足够后初始化不再重要。
#
# Gate Init Bias 物理含义:
#   gate_conv2.bias = init_bias
#   g = sigmoid(init_bias)  at t=0 (训练刚开始, conv weights ≈ 0)
#   → bias=2.0 (默认):  g≈0.88 → "模型从信任 RGB 开始"
#   → bias=0.0 (本脚本): g=0.50 → "模型从平等对待两者开始"
#   → bias=-1.0:        g=0.27 → "模型从信任 skeleton 开始"
#
# ✗ 为什么初始化重要?
#   PoseGated = RGB × g + Skeleton × (1-g)
#   如果 g→1 (bias>>0): 模型认为"骨架都是噪声，别信" → early overfit to RGB
#   如果 g→0 (bias<<0): 模型认为"视频不可靠，全信骨架" → early overfit to skeleton
#   如果 g≈0.5:          模型逐步学会融合 → 更平滑的训练曲线
#
# 消融矩阵:
#   fuse_method      = pose_atn                    ← PoseGated 融合
#   ablation_study   = single                      ← 单点 fusion
#   fusion_layers    = ${PBS_SUBREQNO}             ← 遍历 0~4
#   use_side_heads   = True                        ← side head 保持开启
#   loss_selection   = ["cls","attn_loss","bg","tmp"]
#   gate_init_bias   = 0.0       ← ★ THIS IS THE ABLATION ★
#                                  (与默认 2.0、本脚本对比 -1.0 形成三组)
#
# 论文报告方式:
#   Table X Row A2a — "PoseGated gate_init_bias=0.0" (Ablation A2)
#   Figure W — line chart: x=gates_init_bias, y=accuracy → 看曲线是否平坦或有无最优值
#   → 预期：bias=0.0 和 bias=2.0 最终接近（训练后 gate self-adjusts），但收敛速度不同
###############################################################################

#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gen_S                         # ✅ 队列名（gpu / debug / gen_S）
#PBS -l elapstim_req=24:00:00          # ⏱ 运行时间限制（最多 24 小时）
#PBS -N pose_gated_bias0_train        # 🏷 Ablation A2a: gate_init_bias = 0.0
#PBS -t 0-4                           # job array 0-4 (5 folds, 遍历融合层)
#PBS -o logs/pegasus/train_pose_gated_bias0_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_pose_gated_bias0_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
source /work/SKIING/chenkaixu/code/med_atn/bin/activate

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: $PBS_SUBREQNO (fusion layer index)"
echo "gate_init_bias = 0.0 (neutral init: g≈0.5 → equal RGB/skeleton trust)"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

# Ablation A2a: PoseGated gate init bias = 0.0 (neutral, equal trust in both modalities)
python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn train.fold=5 \
    train.experiment=pose_atn_bias0_single_${PBS_SUBREQNO} \
    data.num_workers=$(( $(nproc) / 3 )) \
    model.fusion_layers=${PBS_SUBREQNO} model.ablation_study=single \
    model.gate_init_bias=0.0
