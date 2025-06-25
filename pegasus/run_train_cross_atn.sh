#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gen_S                           # ✅ 队列名（gpu / debug / gen_S）
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N cross_atn_train                     # 🏷 作业名称
#PBS -o logs/pegasus/train_cross_atn_out.log            # 📤 标准输出日志
#PBS -e logs/pegasus/train_cross_atn_err.log            # ❌ 错误输出日志

# === 切换到作业提交目录 ===
cd /home/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

# mkdir -p logs/pegasus
mkdir -p logs/pegasus/

# === 加载 Python + 激活 Conda 环境 ===
module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
# conda activate base               # ✅ 替换为你的环境名
source /home/SKIING/chenkaixu/code/med_atn/bin/activate

# === 可选：打印 GPU 状态 ===
nvidia-smi

# 输出当前环境信息
echo "Current Python version: $(python --version)"
echo "Current Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Current working directory: $(pwd)"

# params 
root_path = /work/SKIING/chenkaixu/data/asd_dataset/pose_attn_map_dataset

# === 运行你的训练脚本（Hydra 参数可以加在后面）===
python -m project.main data.root_path=${root_path} model.fuse_method=cross_atn train.fold=10 