#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                         # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00          # ⏱ 运行时间限制（最多 24 小时）
#PBS -N pose_atn_multi_train          # 🏷 作业名称 — PoseGated Multi Layer
#PBS -t 0-4                           # job array 0-4 (多层层叠 fusion)
#PBS -o logs/pegasus/train_pose_atn_multi_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_pose_atn_multi_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: $PBS_SUBREQNO (multi fusion layers prefix length)"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn train.fold=5 \
    train.gpu=1 \
    train.experiment=pose_atn_multi_${PBS_SUBREQNO} \
    data.num_workers=$(( $(nproc) / 3 )) \
    model.fusion_layers=${PBS_SUBREQNO} model.ablation_study=multi


# Script notes
# Ablation A5 multi: PoseGated with {0..${PBS_SUBREQNO}} layers fused

# Experiment notes
###############################################################################
# 实验编号: Ablation A5 — PoseGated Multi Layers (Table X Row 9)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 对比目标: "单点 fusion vs 多层融合" — 验证在多个层注入 gate 是否优于单层。
#
# multi 的 fusion_layers mapping (由 FUSE_LAYERS_MAPPING 定义):
#   PBS_SUBREQNO=0 → fusion_layers=[0]     (only stem)
#   PBS_SUBREQNO=1 → fusion_layers=[0,1]   (stem + layer1)
#   PBS_SUBREQNO=2 → fusion_layers=[0,1,2] (stem + layer1 + layer2)
#   PBS_SUBREQNO=3 → fusion_layers=[0,1,2,3]
#   PBS_SUBREQNO=4 → fusion_layers=[0,1,2,3,4]  ← 所有层同时融合！
#
# ✗ vs single: single=[i] 只在一层注入，multi=i 从 0 累加到 i 共 (i+1) 层
#
# 消融矩阵 (multi = all components enabled):
#   fuse_method      = pose_atn
#   ablation_study   = multi
#   fusion_layers    = ${PBS_SUBREQNO}  (映射为 FUSE_LAYERS_MAPPING[pbs_id])
#   use_side_heads   = True              ← side head supervision
#   loss_selection   = ["cls","attn_loss","bg","tmp"]
#   gate_init_bias   = 2.0               ← 默认偏 RGB
#   fusion_norm      = gn                ← GroupNorm (小 batch 稳定)
#   fusion_residual  = True              ← residual connection
#   gate_temp        = 1.0
#
# ✗ best_config: run_train_pose_gated_best.sh 跑 multi[4] + all layers
#                这个脚本是 multi 的逐层 ablation（用来选 best layer config）
#
# 论文报告方式:
#   Table X Row A5_multi — "PoseGated with multi-layer fusion" (Ablation A5)
#   Figure Z — bar chart: x=fusion_layers count, y=accuracy → 证明越多越好或存在拐点
###############################################################################
