#!/bin/bash
###############################################################################
# 实验编号: Ablation A4b — No Temporal Smoothness Loss (Table X Row A4b)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 对比目标: "有 tmp_loss" vs "无 tmp_loss" — 验证时间平滑约束是否对骨架动画
#           heatmap 质量有益。
#
# tmp_loss (时间平滑损失, Total Variation L1) 的原理:
#   stride_gait = 30fps, gait cycle ≈ 1-2s
#   skeleton attention map 在相邻帧之间应该连续变化（人走路是连续运动）
#   tmp_loss penalize frame-to-frame abrupt changes in predicted heatmap:
#
#   P_sig = sigmoid(side_pred)     → [B, C, T, H, W] ∈ [0,1] probability maps
#   tmp_loss = |P_sig[:, :, 1:] - P_sig[:, :, :-1]|_L1.mean()
#              ↑ 相邻帧概率差的 L1 范数 → 值越小，越平滑
#
# ✗ 为什么需要 tmp_loss?
#   skeleton keypoint（YOLOv8 估计的）有噪声，frame-by-frame 预测可能在同一关节上
#   出现"跳动"——前一帧关注腰椎、后一帧突然跳到膝盖。tmp_loss 强制 temporal coherence。
# ✗ vs bg_loss: bg_loss 是空间约束（背景区域不能关注），tmp_loss 是时间约束（相邻帧要连贯）。
#   一个管"看哪"，一个管"怎么动"。
#
# 消融矩阵:
#   fuse_method      = pose_atn
#   ablation_study   = single
#   fusion_layers    = ${PBS_SUBREQNO}           ← 遍历 0~4
#   use_side_heads   = True                      ← side head 保留（仍需监督）
#   loss_selection   = ["cls","attn_loss","bg"]  ← ★ tmp_loss 被移除 ★
#                                                    (w_temp 被 trainer 设为 0)
#   gate_init_bias   = 2.0
#
# 论文报告方式:
#   Table X Row A4b — "PoseGated w/o temporal smoothness" (Ablation A4b)
#   → 预期：gate weights 跳变增多，heatmap 可视化出现帧间闪烁
###############################################################################

#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                         # ✅ 队列名（gpu / debug / gen_S）
#PBS -l elapstim_req=24:00:00          # ⏱ 运行时间限制（最多 24 小时）
#PBS -N pose_gated_notmploss_train    # 🏷 Ablation A4b: No tmp_loss
#PBS -t 0-4                           # job array 0-4 (5 folds, 遍历融合层)
#PBS -o logs/pegasus/train_pose_gated_notmp_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_pose_gated_notmp_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: $PBS_SUBREQNO (fusion layer index)"
echo "loss_selection = [cls, attn_loss, bg]  → tmp_loss (temporal smoothness) removed"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

# Ablation A4b: PoseGATED w/o tmp_loss (temporal smoothness loss)
python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn train.fold=5 \
    train.experiment=pose_atn_notmp_single_${PBS_SUBREQNO} \
    data.batch_size=64 \
    data.num_workers=$(( $(nproc) / 3 )) \
    model.fusion_layers=${PBS_SUBREQNO} model.ablation_study=single \
    loss.selection=["cls","attn_loss","bg"]
