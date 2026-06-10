#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                         # ✅ 队列名（gpu / debug / gen_S）
#PBS -b 1                             # GPU 数量
#PBS -l elapstim_req=24:00:00          # ⏱ 运行时间限制（最多 24 小时）
#PBS -N pose_gated_no_sidehead_train  # 🏷 Ablation A3: No Side Head
#PBS -t 0-4                           # job array 0-4 (5 folds, 遍历融合层)
#PBS -o logs/pegasus/train_pose_gated_noside_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_pose_gated_noside_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: $PBS_SUBREQNO (fusion layer index)"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn train.fold=5 \
    train.gpu=1 \
    train.experiment=pose_atn_noside_single_${PBS_SUBREQNO} \
    data.batch_size=32 \
    data.num_workers=$(( $(nproc) / 3 )) \
    model.fusion_layers=${PBS_SUBREQNO} model.ablation_study=single \
    model.use_side_heads=False


# Script notes
# Ablation A3: PoseGated WITHOUT side heads → no intermediate heatmap supervision

# Experiment notes
###############################################################################
# 实验编号: Ablation A3 — No Side Head (Table X Row A3a)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 对比目标: "有 side head 辅助监督" vs "无 side head" — 验证中间层 heatmap
#           监督（doctor_hm）是否对训练有帮助。
#
# Side Head 机制：在 layer1~layer4 的每个 block 后挂一个 1×1×1 Conv3d，
# 将该层特征投影回 skeleton space (ctx_channels)，输出 per-joint heatmap logits。
# trainer 用 doctor_hm（医生标注的注意力图）和 side_pred 计算 attn_loss (BCE+Dice)。
#
# Side Head on 的效果:
#   - side_preds[0]: block1 feature → projected to [B, ctx_ch, T1, H1, W1]
#   - side_preds[1]: block2 feature → projected to [B, ctx_ch, T2, H2, W2]
#   - side_preds[2]: block3 feature → projected to [B, ctx_ch, T3, H3, W3]
#   - side_preds[3]: block4 feature → projected to [B, ctx_ch, T4, H4, W4]
#   attn_loss = Σ λ_i · BCE_Dice(side_preds[i], doctor_hm_resized_to_Ti,Hi,Wi)
#
# 消融矩阵:
#   fuse_method      = pose_atn                    ← PoseGated 融合
#   ablation_study   = single                      ← 单点 fusion
#   fusion_layers    = ${PBS_SUBREQNO}             ← 遍历 0~4
#   use_side_heads   = False  ← ★ THIS IS THE ABLATION ★
#                            → side heads 不输出，attn_loss = 0
#                            → lambda_list 被强制设为 [0,0,0,0]（trick：在代码中检测到无 attn_loss 时清空权重）
#   loss_selection   = ["cls","bg","tmp"]          ← 只剩 cls + bg + tmp
#   gate_init_bias   = 2.0
#
# 论文报告方式:
#   Table X Row A3a — "PoseGated without side head supervision" (Ablation A3)
#   → 预期：accuracy 下降（side head 提供了额外的监督信号，尤其在小数据集上）
###############################################################################
