#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                           # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N se_atn_train                  # 🏷 A1d: SE fusion (prefix scan)
#PBS -t 0-14                          # job array: prefix(5) × fold(3)
#PBS -o logs/pegasus/train_se_atn_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_se_atn_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

# array 展开: SUBREQNO = prefix*3 + fold
# 注意: SE 模型的 int fusion_layers 是 prefix 映射 (0→[0], 1→[0,1], ... 4→[0..4])，
# 与 pose_atn 的 single/multi 不同，ablation_study 对 SE 无效。
prefix=$(( PBS_SUBREQNO / 3 ))
fold=$(( PBS_SUBREQNO % 3 ))

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: ${PBS_SUBREQNO} → SE prefix [0..${prefix}], fold ${fold} / 3-fold"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=se_atn \
    model.fusion_layers=${prefix} \
    train.fold=3 train.fold_idx=${fold} \
    train.gpu=1 \
    train.experiment=se_atn_prefix${prefix}_f${fold} \
    data.num_workers=$(( $(nproc) / 3 ))


# Experiment notes
###############################################################################
# 实验编号: A1d — SE (Squeeze-and-Excitation) fusion 对比方法
# 论文 A1 表取 prefix 扫描中的最优点。15 个 sub-job = 5 prefix × 3 fold。
# (原脚本名 run_train_se_atn_single.sh 有误导性——SE 实际是 prefix 融合，已更名。)
###############################################################################
