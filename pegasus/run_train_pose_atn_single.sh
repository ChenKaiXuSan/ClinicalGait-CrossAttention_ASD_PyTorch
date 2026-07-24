#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                           # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N pose_atn_single_train         # 🏷 A5 single-layer: PoseGated 单层注入
#PBS -t 0-14                          # job array: layer(5) × fold(3)
#PBS -o logs/pegasus/train_pose_atn_single_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_pose_atn_single_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

# array 展开: SUBREQNO = layer*3 + fold
layer=$(( PBS_SUBREQNO / 3 ))
fold=$(( PBS_SUBREQNO % 3 ))

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: ${PBS_SUBREQNO} → single layer [${layer}], fold ${fold} / 3-fold"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn \
    model.fusion_layers=${layer} model.ablation_study=single \
    train.fold=3 train.fold_idx=${fold} \
    train.gpu=1 \
    train.experiment=pose_atn_single_L${layer}_f${fold} \
    data.num_workers=$(( $(nproc) / 3 ))


# Experiment notes
###############################################################################
# 实验编号: A5a-e — 融合层位置消融 (single): 只在第 layer 层融合 [layer]
# layer 含义: 0=stem(64ch) 1=layer1(256) 2=layer2(512) 3=layer3(1024) 4=layer4(2048)
# 15 个 sub-job = 5 layer × 3 fold，每个 sub-job 独占一个 node 只跑一折。
###############################################################################
