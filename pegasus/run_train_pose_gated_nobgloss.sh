#!/bin/bash
###############################################################################
# 实验编号: Ablation A4a — No Background Suppression Loss (Table X Row A4a)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 对比目标: "有 bg_loss" vs "无 bg_loss" — 验证背景抑制是否对 heatmap 质量有益。
#
# bg_loss (背景抑制损失) 的原理:
#   doctor_hm (医生标注的 attention map) 只在关节 ROI 区域有值，其他区域是 0。
#   side head 输出可能在这些 "非关注" 区域也有激活 → 假阳性。
#   bg_loss 强制 side head 在背景区域的预测接近零：
#
#   A_union = max(side_pred across joints)     → 所有关节关注的并集
#   A_bg    = (1 - A_union).clamp(0, 1)        → 取反 → 背景区域 mask
#   bg_loss = WeightedBCE(P_max, zeros, weight=A_bg)
#                                          ↑ 只在背景区域 penalize
#
# ✗ 如果不加 bg_loss: side head 可能"瞎关注"（在非 ROI 区域也激活），导致
#   heatmap 可视化不精确，gate weights 无法准确反映真实关注区域。
# ✗ 与 attn_loss 区别: attn_loss 监督 "在关注区域预测要高"；
#   bg_loss 监督 "在非关注区域预测要低" — 两者互补。
#
# 消融矩阵:
#   fuse_method      = pose_atn
#   ablation_study   = single
#   fusion_layers    = ${PBS_SUBREQNO}           ← 遍历 0~4
#   use_side_heads   = True                      ← side head 保留（仍需监督）
#   loss_selection   = ["cls","attn_loss","tmp"] ← ★ bg_loss 被移除 ★
#                                                    (w_bg 被 trainer 设为 0)
#   gate_init_bias   = 2.0
#
# 论文报告方式:
#   Table X Row A4a — "PoseGated w/o background suppression" (Ablation A4a)
#   → 预期：attn_loss 升高（heatmap 质量下降），accuracy 可能轻微下降
###############################################################################

#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gpu                         # ✅ 队列名（gpu / debug / gen_S）
#PBS -l elapstim_req=24:00:00          # ⏱ 运行时间限制（最多 24 小时）
#PBS -N pose_gated_nobgloss_train     # 🏷 Ablation A4a: No bg_loss
#PBS -t 0-4                           # job array 0-4 (5 folds, 遍历融合层)
#PBS -o logs/pegasus/train_pose_gated_nobg_out_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/train_pose_gated_nobg_err_${PBS_SUBREQNO}.log

cd /work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch

mkdir -p logs/pegasus/ checkpoints/

source pegasus/setup_env.sh

echo "Current working directory: $(pwd)"
echo "PBS job id: $PBS_JOBID, sub-request: $PBS_SUBREQNO (fusion layer index)"
echo "loss_selection = [cls, attn_loss, tmp]  → bg_loss removed"
echo "Total CPU cores: $(nproc), workers = $(( $(nproc) / 3 ))"

root_path=/work/SKIING/chenkaixu/data/asd_dataset

# Ablation A4a: PoseGATED w/o bg_loss (background suppression loss)
python -m project.train data.root_path=${root_path} \
    model.fuse_method=pose_atn train.fold=5 \
    train.experiment=pose_atn_nobg_single_${PBS_SUBREQNO} \
    data.batch_size=64 \
    data.num_workers=$(( $(nproc) / 3 )) \
    model.fusion_layers=${PBS_SUBREQNO} model.ablation_study=single \
    loss.selection=["cls","attn_loss","tmp"]
