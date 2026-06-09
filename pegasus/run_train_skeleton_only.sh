#!/bin/bash
###############################################################################
# 实验编号: Skeleton-Only Baseline (Table X Row Skeleton)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 对比目标: "纯 skeleton prior" vs "纯 RGB" vs "RGB+skeleton (PoseGATED)" —
#           回答核心研究问题：clinical prior（skeleton attention map）到底贡献了多少？
#
# ⚠️ 注意: 当前代码不支持"仅 skeleton 输入"（骨架本身无法构成视频帧）。
#   这个脚本实际运行的是 train.attn_map=False 的 baseline，即完全关闭 skeleton 输入。
#   它代表 "不用任何临床先验" 的极端情况，与 Baseline (run_train_3dcnn.sh) 等效。
#
# 如果需要真正的 "skeleton-only"（仅骨架不 RGB），需要：
#   1. dataloader 输出 skeleton-based frames（非视频帧）
#   2. backbone 改为处理 1-channel 输入（当前 SlowR50 首层固定 3ch）
#   → 这需要修改 codebase，不在当前消融范围内。
#
# 当前脚本实际是 "attn_map=False" baseline：
#   - fuse_method = none (无 fusion)
#   - attn_map = False (dataloader 不加载 skeleton attention map)
#   - model_class_num = 3
#
# 消融矩阵:
#   fuse_method      = none
#   train.attn_map   = False ← ★ 关闭 skeleton 输入 ★
#   use_side_heads   = False
#   loss_selection   = ["cls"]
#
# ✗ vs Baseline (run_train_3dcnn.sh): 两个脚本相同。如果未来添加真正的 skeleton-only，
#   可以在此脚本基础上修改 dataloader。
#
# 论文报告方式:
#   Table X Row "Skeleton-Only" → 预期 accuracy >> RGB-only
#   → 证明 skeleton prior 本身信息量足够，但需要与 RGB 融合才能最大化性能
###############################################################################

#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gen_S                         # ✅ 队列名（gpu / debug / gen_S）
#PBS -l elapstim_req=24:00:00          # ⏱ 运行时间限制（最多 24 小时）
#PBS -N skeleton_baseline_train       # 🏷 Skeleton-Only Baseline
#PBS -t 0-4                           # job array 0-4 (5 folds)
#PBS -o logs/pegasus/train_skeleton_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_skeleton_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
source /work/SKIING/chenkaixu/code/med_atn/bin/activate

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID"
echo "attn_map = False → NO skeleton input (RGB-only baseline)"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

# Skeleton-Only Baseline: attn_map=False → no clinical prior, pure RGB backbone
python -m project.train data.root_path=${root_path} \
    model.fuse_method=none train.fold=5 \
    train.attn_map=False data.num_workers=$(( $(nproc) / 3 ))
