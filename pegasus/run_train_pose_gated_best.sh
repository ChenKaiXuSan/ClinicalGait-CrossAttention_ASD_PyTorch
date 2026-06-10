#!/bin/bash
###############################################################################
# 实验编号: Best PoseGATED Config (Table X Row Final / Method of Record)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 对比目标: "PoseGated 最佳配置" vs "Baseline / 所有对比方法" — 论文最终报告的方法。
#
# 最佳配置 = 消融 A1-A4 中各自最优 component 的汇总：
#   - fusion_layers: multi (all layers)          ← A5 best
#   - use_side_heads: True                       ← A3 best
#   - loss_selection: ["cls","attn_loss","bg","tmp"] ← A4a+A4b best
#   - gate_init_bias: 2.0                        ← A2 default (偏 RGB)
#   - fusion_norm: gn (GroupNorm)                ← 小 batch 稳定
#   - fusion_residual: True                      ← residual connection
#
# 这个脚本是消融实验完成后，选出的"最佳配置"的复现。用于：
#   1. Table X final row — "Proposed Method (PoseGated, full)"
#   2. Figure X — gate weight visualization (展示模型学会了关注哪些关节)
#   3. 与所有 baseline/对比方法汇总在一个表中
#
# PBS_SUBREQNO 含义:
#   0 → fusion_layers=[0]     (multi prefix of length 1 = stem only)
#   1 → fusion_layers=[0,1]   (stem + layer1)
#   ...
#   4 → fusion_layers=[0,1,2,3,4] ← ALL layers! (best config)
#
# ✗ vs run_train_pose_atn_multi.sh: 内容完全相同，但本脚本是"最终推荐配置"的
#   正式命名。如果 multi[4] 确实最优，两个脚本结果应一致（可互相验证）。
#
# 消融矩阵 (最佳配置):
#   fuse_method      = pose_atn
#   ablation_study   = multi
#   fusion_layers    = ${PBS_SUBREQNO}           ← 累加融合 [0..pbs]
#   use_side_heads   = True                      ← 完整 side head supervision
#   loss_selection   = ["cls","attn_loss","bg","tmp"]  ← 所有损失项
#   gate_init_bias   = 2.0                       ← 偏 RGB (默认)
#   fusion_norm      = gn
#   fusion_residual  = True
#   gate_temp        = 1.0
#
# ✗ vs all other methods: 这个配置的 accuracy 应 > Baseline + CrossAttn + SE + EarlyFuse
###############################################################################

#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gen_S                         # ✅ 队列名（gpu / debug / gen_S）
#PBS -l elapstim_req=24:00:00          # ⏱ 运行时间限制（最多 24 小时）
#PBS -N pose_gated_best_train         # 🏷 Best PoseGATED Configuration
#PBS -t 0-4                           # job array 0-4 (5 folds)
#PBS -o logs/pegasus/train_pose_gated_best_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_pose_gated_best_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
source /work/SKIING/chenkaixu/code/med_atn/bin/activate

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: $PBS_SUBREQNO (fusion layers = [0..${PBS_SUBREQNO}])"
echo "★ BEST CONFIG: PoseGATED with ALL components enabled ★"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

# Best PoseGated config: multi-layer fusion + all loss terms + side heads + gate_bias=2.0
python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn train.fold=5 \
    train.experiment=pose_gated_best_multi_${PBS_SUBREQNO} \
    data.num_workers=$(( $(nproc) / 3 )) \
    model.fusion_layers=${PBS_SUBREQNO} model.ablation_study=multi
