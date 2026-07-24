#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                           # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N pose_atn_multi_train          # 🏷 A5 multi-prefix: PoseGated 多层累加注入
#PBS -t 3-11                          # job array: prefix(3: P1-P3) × fold(3); P0 dropped (== single L0)
#PBS -o logs/pegasus/train_pose_atn_multi_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_pose_atn_multi_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

# array 展开: SUBREQNO = prefix*3 + fold；prefix 只扫 1-3 (SUBREQNO 3-11)。
# P0 (= [0]) 与 pose_atn_single 的 L0 完全等价, 已删除避免重复。
# P4 (= full [0..4]) 由 run_train_pose_gated_best.sh 提供，避免重复训练。
prefix=$(( PBS_SUBREQNO / 3 ))
fold=$(( PBS_SUBREQNO % 3 ))

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: ${PBS_SUBREQNO} → multi prefix [0..${prefix}], fold ${fold} / 3-fold"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn \
    model.fusion_layers=${prefix} model.ablation_study=multi \
    train.fold=3 train.fold_idx=${fold} \
    train.gpu=1 \
    train.experiment=pose_atn_multi_P${prefix}_f${fold} \
    data.num_workers=$(( $(nproc) / 3 ))


# Experiment notes
###############################################################################
# 实验编号: A5g-i — 融合层数量消融 (multi prefix): [0,1], [0,1,2], [0,1,2,3]
# 曲线起点 [0] 复用 single L0, 终点 [0,1,2,3,4] 复用 pose_gated_full (均勿重复跑)。
# 9 个 sub-job = 3 prefix × 3 fold。
###############################################################################
