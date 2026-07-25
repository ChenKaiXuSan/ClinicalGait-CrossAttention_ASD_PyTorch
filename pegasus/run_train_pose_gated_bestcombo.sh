#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                           # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N pose_gated_bestcombo_train    # 🏷 经验最优组合（消融各自最优的叠加）
#PBS -t 0-2                           # job array: fold index (3-fold, 每折一个 node)
#PBS -o logs/pegasus/train_pose_gated_bestcombo_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_pose_gated_bestcombo_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

fold=${PBS_SUBREQNO}

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: ${PBS_SUBREQNO} → fold ${fold} / 3-fold"
echo "★ 经验最优组合: multi[0,1] + gate_bias=0.0 + loss=[cls,attn_loss] (去 bg/tmp)"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn \
    model.fusion_layers=1 model.ablation_study=multi \
    model.gate_init_bias=0.0 \
    'loss.selection=["cls","attn_loss"]' \
    train.fold=3 train.fold_idx=${fold} \
    train.gpu=1 \
    train.experiment=pose_gated_bestcombo_f${fold} \
    data.num_workers=$(( $(nproc) / 3 ))


# Experiment notes
###############################################################################
# 经验最优组合 (post-hoc)：把 3-fold 消融里各自最优的选择叠加，验证是否协同增益。
#   - fusion_layers: multi [0,1]      ← A5 最优（88.3% > full 81.5%）
#   - gate_init_bias: 0.0             ← A2 最优（84.8% > bias2.0 的 81.5%）
#   - loss.selection: [cls,attn_loss] ← A3/A4：去 bg(87.8%)、去 tmp(84.6%) 均优于完整
#   - use_side_heads: True（默认）      ← A3 显示 side head 近中性，保留
# 注意：这是 post-hoc 组合，不能替代独立测试；仅验证叠加效果，报告时需说明。
###############################################################################
