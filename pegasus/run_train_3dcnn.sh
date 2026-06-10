#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                         # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00          # ⏱ 运行时间限制（最多 24 小时）
#PBS -N run_3dcnn_train               # 🏷 作业名称
#PBS -o logs/pegasus/train_3dcnn_out.log
#PBS -e logs/pegasus/train_3dcnn_err.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

echo "Current working directory: $(pwd)"
echo "Total CPU cores: $(nproc), use $(( $(nproc) / 3 )) for data loading"
echo "Python version: $(python --version)"
echo "Virtual environment: $(which python)"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=none train.fold=5 \
    train.gpu=1 \
    train.experiment=baseline_rgb_3dcnn \
    data.batch_size=32 \
    data.num_workers=$(( $(nproc) / 3 ))


# Script notes
# === 切换到作业提交目录 ===
# === 下载预训练模型（如果需要） ===
# wget -O checkpoints/SLOW_8x8_R50.pyth https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW_8x8_R50.pyth
# === 加载 Python + 激活 Conda 环境 ===
# === 打印环境信息 ===
# params
# === Baseline Res3DCNN: fuse=none → 纯 RGB，无 skeleton prior ===

# Experiment notes
###############################################################################
# 实验编号: Baseline (Table X Row 1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 对比目标: "纯视频" vs "带临床先验的方法" — 所有消融实验的最低基线
#
# 这个脚本训练的是一个没有注入 skeleton attention map 的标准 SlowR50 Res3DCNN：
#   - fuse_method = none     → attn_map 在 backbone 前端不做任何融合操作
#   - use_side_heads  = False → 无 side head，无中间层监督
#   - loss_selection = ["cls"] → 仅交叉熵分类损失
#
# 对比逻辑:
#   Baseline accuracy ≈ 78.5%  (README 原始数据)
#   PoseGated       ≈ 86.1%   ← 需要证明比 baseline 显著更高
#
# 消融矩阵 (Ablation Matrix):
#   fuse_method      = none
#   use_side_heads   = False  (默认关闭)
#   loss_selection   = ["cls"]
#   gate_init_bias   = N/A    (PoseGated 专用)
#   fusion_layers    = N/A    (无 fusion)
#
# 论文报告方式:
#   Table I Row 1 — "Baseline: Standard Res3DCNN (RGB only)"
###############################################################################
