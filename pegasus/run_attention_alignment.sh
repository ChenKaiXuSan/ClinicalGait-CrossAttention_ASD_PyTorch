#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名
#PBS -q gpu                           # ✅ 队列
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=06:00:00         # ⏱ test 推理很快,给 6h 富余
#PBS -N attn_align                    # 🏷 可解释性对齐分析
#PBS -o logs/pegasus/attn_align_out.log
#PBS -e logs/pegasus/attn_align_err.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/

source pegasus/setup_env.sh

echo "Current working directory: $(pwd)"
echo "★ 可解释性对齐: 主推方法 pose_atn_multi_P1 的 side-head 热图 vs 医生标注"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

# attention_alignment.py 内部循环 3 折:加载各折最优 ckpt → 跑测试集 →
# 抽 side_preds → 对 doctor_hm 算 CC/NSS/PointingGame/AUC/IoU/Dice/KL/SIM。
# 结果写 analysis/alignment_out/{alignment_per_record.csv, alignment_summary.csv}
python -m analysis.attention_alignment \
    data.root_path=${root_path} \
    data.num_workers=$(( $(nproc) / 3 )) \
    "+align.run_glob=logs/train/pose_atn_multi_P1_f*/**"
