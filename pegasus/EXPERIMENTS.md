# Pegasus 实验总览表 (Ablation Matrix)

## 一、所有实验一览

```
┌──────────────────────────────────────┬─────────────────────────┬──────────────┬──────────────────────────────────────────┐
│ 脚本                                │ Ablation              │ 对比什么      │ 论文表格                                   │
├──────────────────────────────────────┼─────────────────────────┼──────────────┼──────────────────────────────────────────┤
│                                      │                         │              │                                          │
│ BASELINE                            │                         │              │                                          │
│ ─────────────────────────────────  │                         │              │                                          │
│ run_train_3dcnn.sh                  │ Baseline               │ "纯 RGB"    │ Table I Row 1                             │
│ run_train_skeleton_only.sh          │ Skeleton-only          │ "无 prior"   │ Table I Row Skeleton                      │
│                                      │                         │              │                                          │
│ ABLATION A1: FUSION METHOD (5 methods)                    │              │                                          │
│ ────────────────────────           │                         │              │                                          │
│ run_train_early_fuse_add.sh         │ A1a  early add         │ "逐像素加"   │ Table I Row A1a                           │
│ run_train_early_fuse_concat.sh      │ A1b  early concat      │ "通道拼接"   │ Table I Row A1b                           │
│ run_train_early_fuse_mul.sh         │ A1c  early mul         │ "逐像素乘"   │ Table I Row A1c                           │
│ run_train_cross_atn.sh              │ A1d  cross-attn        │ "QKV self-    │ Table I Row A1d                           │
│                                      │                        │    attn (THW) │                                          │
│ run_train_se_atn_single.sh          │ A1e  se-fusion         │ "Squeeze-excit. │ Table I Row A1e                          │
│                                      │                        │    global ch."│                                          │
│                                      │                         │              │                                          │
│ ABLATION A2: GATE INIT BIAS (3 values)                  │              │                                          │
│ ──────────────────────────         │                         │              │                                          │
│ run_train_3dcnn.sh + default(2.0)   │ A2a  gate_bias=2.0     │ "偏 RGB       │ Table I Row A2a (default = reference)    │
│ run_train_pose_gated_bias0.sh       │ A2b  gate_bias=0.0     │ "无偏         │ Table I Row A2b                           │
│ run_train_pose_gated_bias_neg1.sh   │ A2c  gate_bias=-1.0    │ "偏 skeleton  │ Table I Row A2c                           │
│                                      │                         │              │                                          │
│ ABLATION A3: SIDE HEAD (yes/no)     │                         │              │                                          │
│ ──────────────────────             │                         │              │                                          │
│ run_train_3dcnn.sh + default(True)  │ A3a  side_head=True    │ "有中间监督"   │ Table I Row A3a (default = reference)    │
│ run_train_pose_gated_nosidehead.sh  │ A3b  side_head=False   │ "无中间监督"   │ Table I Row A3b                           │
│                                      │                         │              │                                          │
│ ABLATION A4: LOSS COMPONENTS (2 ablation)                 │              │                                          │
│ ───────────────────────────        │                         │              │                                          │
│ run_train_3dcnn.sh + default(all)   │ A4a  all losses        │ "bg+tmp       │ Table I Row A4a (default = reference)    │
│ run_train_pose_gated_nobgloss.sh    │ A4b  w/o bg_loss       │ "去掉背景抑制" │ Table I Row A4b                           │
│ run_train_pose_gated_notmploss.sh   │ A4c  w/o tmp_loss      │ "去掉时间平滑" │ Table I Row A4c                           │
│                                      │                         │              │                                          │
│ ABLATION A5: FUSION LAYERS          │                         │              │                                          │
│ ────────────────────────           │                         │              │                                          │
│ run_train_pose_atn_single.sh (×5)   │ A5a-e  single[i]       │ "单点 fusion  │ Table I Row A5a-e; Fig: line chart       │
│                                      │                        │   at layer i" │   of layers vs accuracy                   │
│ run_train_pose_atn_multi.sh (×5)    │ A5f-j  multi[0..4]     │ "多层融合      │ Table I Row A5f-j; Fig: bar chart        │
│                                      │                        │   prefix i"  │   of fusion count vs accuracy             │
│ run_train_pose_gated_best.sh        │ A5-ultimate best       │ "所有层+       │ Table I Row Best / Method of Record      │
│                                      │                        │    all losses │                                          │
└──────────────────────────────────────┴─────────────────────────┴──────────────┴──────────────────────────────────────────┘
```

## 二、消融实验设计逻辑图

```
核心问题: PoseGated 为什么比 Baseline 好？
         ↓
    ┌────────────────────────────────────┐
    │ A1: 哪个 fusion 策略最好？          │
    │   - early add/concat/mul           │ ← 在输入端融合
    │   - cross-attention                │ ← QKV self-attn on frames
    │   - SE-fusion                      │ ← global channel scaling
    │   - pose-gated (ours)              │ ← spatially-adaptive gate per channel
    │                                    │
    │ A5: fusion 在哪层注入最优？         │
    │   - single[i] vs multi[0..4]       │ ← layer ablation
    │                                    │
    │ A2: Gate 初始偏向 RGB 还是 skeleton?│
    │   - bias=2.0 → g≈0.88 (偏 RGB)    │ ← default
    │   - bias=0.0 → g=0.50 (无偏)      │
    │   - bias=-1.0 → g=0.27 (偏 skeleton)│
    │                                    │
    │ A3: Side head 辅助监督有用吗？     │
    │   - True vs False                  │ ← side head on/off
    │                                    │
    │ A4: bg_loss / tmp_loss 各自贡献？  │
    │   - bg_loss (背景抑制)             │ ← spatial constraint
    │   - tmp_loss (时间平滑)            │ ← temporal constraint
    └────────────────────────────────────┘
         ↓
    Best config = multi[4] + side_head + all losses + bias=2.0
```

## 三、消融矩阵完整对照表 (Paper Table)

```
┌─────┬──────────────┬──────────────┬──────────────┬───────────────────┬─────────────┐
│ Row │ Method       │ fusion_layers│ side_heads   │ loss_selection    │ gate_bias   │
├─────┼──────────────┼──────────────┼──────────────┼───────────────────┼─────────────┤
│  1  │ Baseline     │ none         │ False        │ [cls]             │ N/A         │
│ A1a │ Early (+)    │ none         │ False        │ [cls]             │ N/A         │
│ A1b │ Early (cat)  │ none         │ False        │ [cls]             │ N/A         │
│ A1c │ Early (mul)  │ none         │ False        │ [cls]             │ N/A         │
│ A1d │ Cross-Attn   │ single[i]    │ False        │ [cls]             │ N/A         │
│ A1e │ SE-Fusion    │ single[i]    │ False        │ [cls]             │ N/A         │
│ A2a │ PoseGated(★) │ single[i]    │ True         │ all               │ 2.0 (default)│
│ A2b │ gate_bias=0  │ single[i]    │ True         │ all               │ 0.0         │
│ A2c │ gate_bias=-1 │ single[i]    │ True         │ all               │ -1.0        │
│ A3a │ side_head=T  │ single[i]    │ True (★)     │ all               │ 2.0         │
│ A3b │ side_head=F  │ single[i]    │ False        │ [cls, bg, tmp]    │ 2.0         │
│ A4a │ all_losses   │ single[i]    │ True         │ [all] (★)        │ 2.0         │
│ A4b │ w/o bg       │ single[i]    │ True         │ [cls, attn, tmp]  │ 2.0         │
│ A4c │ w/o tmp      │ single[i]    │ True         │ [cls, attn, bg]   │ 2.0         │
│ A5a │ single[0]    │ [0]          │ True         │ all               │ 2.0         │
│ A5b │ single[1]    │ [1]          │ True         │ all               │ 2.0         │
│ ... │ ...          │ ...          │ ...          │ ...               │ ...         │
│ A5e │ single[4]    │ [4]          │ True         │ all               │ 2.0         │
│ A5f │ multi[0]     │ [0]          │ True         │ all               │ 2.0         │
│ A5g │ multi[1]     │ [0,1]        │ True         │ all               │ 2.0         │
│ ... │ ...          │ ...          │ ...          │ ...               │ ...         │
│ A5j │ multi[4]★    │ [0..4]       │ True         │ all               │ 2.0         │
│ ★   │ BEST (Final) │ [0..4]       │ True         │ all               │ 2.0         │
└─────┴──────────────┴──────────────┴──────────────┴───────────────────┴─────────────┘

★ = 默认配置 / 本文推荐方法
```

## 四、每个实验的论文 Figure 引用建议

```
Figure 1: Overview diagram — PoseGated fusion block architecture
Figure 2: Gate weight visualization (per class, per layer)
Figure 3: Ablation results
         ├── Subplot A: fuse_method comparison (A1) — bar chart
         ├── Subplot B: gate_init_bias comparison (A2) — line chart [-1, 0, 2]
         ├── Subplot C: side head impact (A3) — bar chart (T/F)
         ├── Subplot D: loss components (A4a,b) — two bars each
         └── Subplot E: fusion layers (A5a-e + A5f-j) — two line charts
Figure 4: Gate weight analysis by disease class (correct vs wrong cases)
Figure 5: Skeleton fidelity → accuracy correlation scatter plot
Figure 6: Per-class ROC curves
Figure 7: Case study — per-frame gate weights on representative videos
```
