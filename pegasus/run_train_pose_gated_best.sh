#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                           # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N pose_gated_full_train         # 🏷 主结果: PoseGated full [0..4]
#PBS -t 0-2                           # job array: fold index (3-fold, 每折一个 node)
#PBS -o logs/pegasus/train_pose_gated_full_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_pose_gated_full_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

fold=${PBS_SUBREQNO}

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: ${PBS_SUBREQNO} → fold ${fold} / 3-fold"
echo "★ MAIN RESULT: PoseGated full config, fusion_layers=[0,1,2,3,4] ★"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn \
    model.fusion_layers=4 model.ablation_study=multi \
    train.fold=3 train.fold_idx=${fold} \
    train.gpu=1 \
    train.experiment=pose_gated_full_f${fold} \
    data.num_workers=$(( $(nproc) / 3 ))


# Experiment notes
###############################################################################
# 实验编号: Final — PoseGated full config (Table X final row / Method of Record)
# 配置: multi prefix [0,1,2,3,4] + side heads + 全部损失项 + gate_init_bias=2.0
# 本实验同时充当:
#   - A2a (bias=2.0 对照)、A3/A4 的完整模型对照
#   - A5 multi 曲线的最后一个点 (P4)，run_train_pose_atn_multi.sh 只扫 P0-P3
# 不再用 array 扫 prefix — 旧版行为与 pose_atn_multi 完全重复，已移除。
###############################################################################
