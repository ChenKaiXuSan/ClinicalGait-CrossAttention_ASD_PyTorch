#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                           # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N pose_gated_bias0_train        # 🏷 Ablation A2b: gate_init_bias = 0.0
#PBS -t 0-2                           # job array: fold index (3-fold, 每折一个 node)
#PBS -o logs/pegasus/train_pose_gated_bias0_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_pose_gated_bias0_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

fold=${PBS_SUBREQNO}

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: ${PBS_SUBREQNO} → fold ${fold} / 3-fold"
echo "Ablation A2b: gate_init_bias=0.0, full multi [0..4]"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn \
    model.fusion_layers=4 model.ablation_study=multi \
    model.gate_init_bias=0.0 \
    train.fold=3 train.fold_idx=${fold} \
    train.gpu=1 \
    train.experiment=pose_gated_bias0_f${fold} \
    data.num_workers=$(( $(nproc) / 3 ))


# Experiment notes
###############################################################################
# 实验编号: A2b — gate init bias 消融 (中性起点)
# 对照: A2a = pose_gated_full (bias=2.0)，A2c = bias_neg1 (bias=-1.0)
###############################################################################
