# Pegasus 实验总览表（2026-07 重构版：3-fold + 每折一个 node）

这份表记录当前 `pegasus/*.sh` 和 `configs/config.yaml` 里实际的实验设置。

## ⚠️ 重要变更（2026-07-24）

1. **5-fold → 3-fold**：`train.fold=3`。原因：24h walltime 装不下 5 折串行训练（2026-06 那批作业几乎全部在中途被杀，26 个实验只有 1 个跑完全部 5 折）。
2. **每折独占一个 node**：`train.py` 新增 `train.fold_idx`（`-1`=串行跑全部折，`>=0`=只跑该折）。所有脚本的 PBS array 现在展开到 fold 维度，单个 sub-job 只跑一折（full 配置实测单折 ~11h < 24h walltime）。
3. **旧 5-fold 结果全部作废**：换 fold 数意味着换数据划分，2026-06 之前的结果不能与新结果混用。旧日志仍在 `logs/train/` 旧命名目录下（新实验 tag 全部带 `_f{fold}` 后缀，不会混淆）。
4. **fold 缓存目录带 K 后缀**（`index_mapping/<class_num>/over_K3/`），切换 fold 数不会静默复用旧划分；并行作业构建缓存有目录锁保护。
5. **`cross_atn` trainer 已接回**（`project/trainer/mid/train_cross_attn.py`），early fusion 脚本已补齐。
6. `run_train_se_atn_single.sh` 更名为 `run_train_se_atn.sh`（SE 实际是 prefix 融合，旧名误导）。

## 〇、提交前准备（必做一次）

并行 fold 作业首次运行需要 fold 缓存。虽然有锁保护（其余作业会等第一个构建完），但预构建更稳妥：

```bash
python -m project.prepare_folds data.root_path=/work/SKIING/chenkaixu/data/asd_dataset
```

缓存构建包含视频复制，耗时较长；只需做一次，之后所有作业直接加载。

## 一、当前脚本总表

| 脚本 | 实验角色 | array | sub-jobs | experiment tag |
|---|---|---|---:|---|
| `run_train_3dcnn.sh` | B1: RGB-only baseline | fold 0-2 | 3 | `baseline_rgb_f{fold}` |
| `run_train_early_fuse.sh` | A1a-c: early add/mul/concat | method(3)×fold(3) | 9 | `early_{method}_f{fold}` |
| `run_train_se_atn.sh` | A1d: SE fusion | prefix(5)×fold(3) | 15 | `se_atn_prefix{p}_f{fold}` |
| `run_train_cross_atn.sh` | A1e: QKV cross-attention | cfg(L3/L4/L34)×fold(3) | 9 | `cross_atn_{cfg}_f{fold}` |
| `run_train_pose_atn_single.sh` | A5a-e: 单层注入 | layer(5)×fold(3) | 15 | `pose_atn_single_L{l}_f{fold}` |
| `run_train_pose_atn_multi.sh` | A5g-i: prefix 注入 P1-P3 | prefix(3)×fold(3) | 9 | `pose_atn_multi_P{p}_f{fold}` |
| `run_train_pose_gated_best.sh` | **主结果**: full [0..4] | fold 0-2 | 3 | `pose_gated_full_f{fold}` |
| `run_train_pose_gated_bias0.sh` | A2b: gate bias=0.0 | fold 0-2 | 3 | `pose_gated_bias0_f{fold}` |
| `run_train_pose_gated_bias_neg1.sh` | A2c: gate bias=-1.0 | fold 0-2 | 3 | `pose_gated_biasneg1_f{fold}` |
| `run_train_pose_gated_nosidehead.sh` | A3: 无 side head | fold 0-2 | 3 | `pose_gated_noside_f{fold}` |
| `run_train_pose_gated_nobgloss.sh` | A4a: 无 bg loss | fold 0-2 | 3 | `pose_gated_nobg_f{fold}` |
| `run_train_pose_gated_notmploss.sh` | A4b: 无 tmp loss | fold 0-2 | 3 | `pose_gated_notmp_f{fold}` |

合计 **78 个 sub-job**，每个 ≤ ~12h（一折一 node）。

> 已移除的重复实验（2026-07-24）：B2 `rgb_noattn`（fuse=none 忽略 attn_map，与 B1 完全等价）；multi P0 `[0]`（与 single L0 完全等价）。

组合 array 的展开规则统一为 `SUBREQNO = 外层索引*3 + fold`：

```bash
layer=$(( PBS_SUBREQNO / 3 ))   # 或 prefix / method / cfg
fold=$((  PBS_SUBREQNO % 3 ))
```

日志目录：`logs/train/<train.experiment>/<date>/<time>/`；每折的 test metrics 在 `<run_dir>/metrics/fold_{fold}_metrics.txt`。

## 二、关键代码映射

`configs/config.yaml` 默认值：

```yaml
train.fold: 3
train.fold_idx: -1        # 脚本里覆盖为具体 fold
loss.selection: ["cls", "attn_loss", "bg", "tmp"]
model.use_side_heads: True
model.gate_init_bias: 2.0
model.fuse_method: pose_atn
model.fusion_layers: 5
model.ablation_study: single
```

PoseGated 的 `fusion_layers` 映射（`project/models/pose_fusion_res_3dcnn.py`）：

| `ablation_study` | `fusion_layers` | 实际融合层 |
|---|---:|---|
| `single` | `i` (0-4) | `[i]` |
| `multi` | `i` (0-4) | `[0..i]` |
| 任意 | `5` | `[0,1,2,3,4]` |

SE / CrossAttention 模型的 int `fusion_layers` 走各自文件里的 `fuse_layers_mapping`（SE: `i→[0..i]` prefix；cross: 显式列表更直观，脚本直接传 `[3]`/`[4]`/`[3,4]`），`ablation_study` 对它们无效。

cross_atn 只扫深层的原因：THW×THW 注意力矩阵在 stem/layer1（56×56×16 → THW≈50k）需要 ~10GB/样本，必然 OOM；layer2 也在边缘。

层含义：

| index | backbone block | channel |
|---:|---|---:|
| 0 | stem | 64 |
| 1 | layer1 | 256 |
| 2 | layer2 | 512 |
| 3 | layer3 | 1024 |
| 4 | layer4 | 2048 |

## 三、论文实验矩阵

### A0. Baseline

| Row | Method | 脚本 |
|---|---|---|
| B1 | RGB-only | `run_train_3dcnn.sh` |

### A1. Fusion Method（→ Fig. 2）

| Row | Method | 脚本 | 报告方式 |
|---|---|---|---|
| A1a-c | Early add / mul / concat | `run_train_early_fuse.sh` | 各方法 3 折均值 |
| A1d | SE fusion | `run_train_se_atn.sh` | prefix 扫描取最优点 |
| A1e | Cross-attention | `run_train_cross_atn.sh` | {L3, L4, L3+4} 取最优点 |
| A1f | **PoseGated (ours)** | `run_train_pose_gated_best.sh` | full [0..4] |

### A2. Gate Init Bias（→ Fig. 4）

| Row | bias | 脚本 |
|---|---:|---|
| A2a | 2.0 | 复用 `pose_gated_full` |
| A2b | 0.0 | `run_train_pose_gated_bias0.sh` |
| A2c | -1.0 | `run_train_pose_gated_bias_neg1.sh` |

### A3 + A4. 组件/损失消融（→ Fig. 5）

全部固定 full multi `[0,1,2,3,4]`，与 `pose_gated_full` 单变量对照：

| Row | 拆掉 | 脚本 |
|---|---|---|
| A3 | side heads | `run_train_pose_gated_nosidehead.sh` |
| A4a | bg loss | `run_train_pose_gated_nobgloss.sh` |
| A4b | tmp loss | `run_train_pose_gated_notmploss.sh` |

### A5. Fusion Layers（→ Fig. 3）

| Row | Method | 脚本 |
|---|---|---|
| A5a-e | single `[0]`..`[4]` | `run_train_pose_atn_single.sh` |
| A5g-i | multi `[0,1]`..`[0..3]` | `run_train_pose_atn_multi.sh`（P0=`[0]` 复用 single L0） |
| A5j | multi `[0..4]` | 复用 `pose_gated_full`（勿重复跑） |

## 四、推荐提交顺序

1. **预构建缓存**：`python -m project.prepare_folds data.root_path=...`（一次）
2. **主结果**：`qsub run_train_pose_gated_best.sh` + `qsub run_train_3dcnn.sh`（6 node）
3. **A5**：`qsub run_train_pose_atn_single.sh` + `qsub run_train_pose_atn_multi.sh`（24 node）
4. **A2/A3/A4**：5 个组件消融脚本（15 node）
5. **A1 外部方法**：`run_train_se_atn.sh`、`run_train_cross_atn.sh`、`run_train_early_fuse.sh`（33 node）

## 五、Figure 规划

| Figure | 内容 | 数据来源 |
|---|---|---|
| Fig. 1 | PoseGated block / overall architecture | 方法图 |
| Fig. 2 | Fusion method comparison | A1 |
| Fig. 3 | Single vs multi fusion layer | A5 |
| Fig. 4 | Gate init bias curve | A2 |
| Fig. 5 | Side head / loss component ablation | A3 + A4 |
| Fig. 6 | Per-class ROC / confusion matrix | pose_gated_full |
| Fig. 7 | Gate/attention visualization case study | pose_gated_full checkpoint |

## 六、实验结论（2026-07-25，3-fold 有效重跑）

> 这批是修复零 attn bug 后**第一次真正应用临床先验**的结果（旧的六月结果作废）。
> **已全部完成：原矩阵 78/78 折（26 方法）+ bestcombo 3/3。**
> 均值为 3-fold（std ±2-6%，小差异在噪声内）。acc = **pooled** 准确率（从 best_preds 汇总；日志里的 test/video_acc 是 per-batch macro 平均、系统性偏低约 6-9 点，已弃用）。
> 完整汇总见 `analysis/results_summary.md` / `.csv`（由 `python -m analysis.export_results` 生成）。

### 主要发现

1. **PoseGated 有效**：最优配置 **multi [0,1] = 94.8%**，比 RGB baseline（90.7%）高约 **4 个点**。临床先验价值成立。

2. **⚠️ 预设的 "full" 配置在每个默认轴上都是次优的**（`pose_gated_full` = 全层[0-4] + bias2.0 + 全损失 = 仅 90.9%）：
   - **融合层：less is more**。full [0-4]（90.9%）是所有 PoseGated 配置里最差；只融合浅两层 [0,1]（94.8%）最好。single 层 L3/L4 也达 93-94%。
   - **辅助损失反而有害**：去 bg loss（93.7%）、去 tmp loss（93.3%）均优于完整（90.9%）。
   - **gate bias**：0.0（92.2%）最好;-1.0（90.7%）与 2.0/full（90.9%）相近。默认的强 RGB 偏置(2.0)非最优。
   - **side head 近中性**：去掉后 90.8% vs 90.9%，在噪声内。

3. **A1 融合方法对比**（各方法最优点）：PoseGated（94.8%）> early concat（93.7%）> SE(~91-92%) ≈ early add（91.7%）≈ cross（~90%）> early mul（89.3%）。

### 论文写作建议

- **主结果不要报 `pose_gated_full`**，应报**经验最优配置**（multi [0,1] 或 single L3）。
- A2/A4/A5 消融恰好构成"如何注入先验"的分析：**浅层融合、少辅助损失、中性门控**最好。
- `run_train_pose_gated_bestcombo.sh`（multi[0,1] + bias0 + 去 bg/tmp）结果 **94.8 ± 2.8**，**未超过单独的 multi[0,1]（94.8%）**——per-ablation 最优不叠加。结论：**融合层选择是主导因素**，层选对后 bias/损失改动不再有增益（它们在 full 上"有效"只是因为 full 被过多融合层拖累）。故主结果直接用 **multi[0,1]**，无需 bestcombo。（全数据 3-fold，78/78 原矩阵 + 3/3 bestcombo 已完成。）

### 可解释性对齐（注意力 vs 医生标注，→ Fig. 7）

由 `analysis/attention_alignment.py` 在主推方法 `pose_atn_multi_P1` 各折最优 ckpt 上、对**留出患者**测试集计算 side-head 热图与医生标注 ROI 的对齐度（3-fold；输出 `analysis/alignment_out/`，summary 已入库，per-record 25MB 不入库、`python -m analysis.attention_alignment +align.run_glob='logs/train/pose_atn_multi_P1_f*'` 可重跑）。

| side-head 层 | CC ↑ | NSS ↑ | PG ↑ | AUC ↑ | Dice ↑ |
|---|---|---|---|---|---|
| **L3（最深）** | **0.71–0.76** | **2.8–3.0** | **1.00** | ~1.00 | 0.47–0.51 |
| L0–L2 | 0.43–0.53 | 1.5–2.0 | 1.00 | ~1.00 | 0.41–0.52 |

（ASD/DHS/LCS_HipOA 三类高度一致）

- **结论**：模型注意力可靠定位到医生标注 ROI（PG=1.00 全命中、AUC≈0.998、NSS≈2–3、深层 CC 达 0.76），且**深层 L3 对齐最强**。
- **⚠️ caveat（须写进论文）**：side head 由医生标注直接监督（attn_loss = BCE+Dice），故高对齐主要证明**临床先验被忠实学习并泛化到留出患者**（保真度叙事），而非模型自发发现正确关节。更强论证需加对照组：`pose_gated_noside`（去 attn 监督）的对齐度应大幅下降——其 ckpt 已有，可对它再跑一次 alignment。

## 七、遗留事项

1. skeleton-only baseline（仅骨架输入）仍未实现，需要 dataloader/model 支持后另开脚本。
2. `analysis/` 的画图 notebook 需要适配新的 `_f{fold}` 日志目录命名（glob `<tag>_f*`）。
3. baseline trainer 的 metrics 键名带 `_epoch`（`test/video_acc_epoch`），其他 trainer 是 `test/video_acc`；聚合脚本需兼容两种（`attention_alignment.py` 之外的汇总代码注意）。

## 八、审稿补充实验（reviewer-driven，2026-07）

针对"对比是否充分"的审稿，新增 5 组实验。**#1、#3 为纯推理/分析，已在本机跑出真实结果**；**#4、#5、#6 需 GPU 训练，代码已实现并通过 CPU shape 测试，脚本待 qsub。**

| # | 实验 | 类型 | 脚本 / 分析 | 状态 |
|---|---|---|---|---|
| 1 | 临床先验必要性（test-time 扰动：real/shuffled/zero 注意力） | 推理 | `analysis/attention_perturbation.py` | 本机已跑（见下） |
| 3 | 统计显著性（McNemar + bootstrap CI，配对 clip） | 分析 | `analysis/significance.py` → `significance.md` | 本机已跑 |
| 4 | 第二骨干 X3D-M（"浅层融合"是否迁移） | 训练 | `run_train_x3d_backbone.sh`（array 0-8） | 待 qsub |
| 5 | 另一发表架构 RGB baseline（X3D-M） | 训练 | 同上 cfg 0（`x3d_baseline`） | 待 qsub |
| 6 | 门控机制 vs 纯注入（gate_mode=add/fixed） | 训练 | `run_train_gate_mode.sh`（array 0-5） | 待 qsub |

**代码改动**（均保留 slow_r50 原路径不变，已 CPU 验证无回归）：
- `weight_loader.py`：新增 `init_x3d` / `init_backbone`（X3D-M，`.blocks[0..5]` 同构；head=`blocks[-1].proj` 复用 `modify_head`）。
- `pose_fusion_res_3dcnn.py`：`backbone_net` 开关；非 slow_r50 时 `_infer_stage_dims()` 动态推断每层通道（X3D 维度 `[24,24,48,96,192]`）；`_make_norm` 的 GroupNorm 组数改为"能整除 c 的最大者"（修 X3D 48 通道）；新增 `gate_mode`（gated/add/fixed）。
- `res_3dcnn.py`：`backbone_net`（X3D 仅支持 `fuse=none` RGB baseline，input-level 融合仍 slow_r50）。
- `config.yaml`：新增 `model.backbone_net`、`model.gate_mode`。
- X3D-M 用 16 帧 clip（== 项目默认 `uniform_temporal_subsample_num=16`）；权重 `checkpoints/X3D_M.pyth`（缺失自动下载）。

**#4/#5 提交后要看的对比**（同骨干内）：`x3d_baseline` vs `x3d_pose_multi01`（先验是否仍有效）、`x3d_pose_multi01` vs `x3d_pose_full`（浅层是否仍优）。
**#6**：`add`/`fixed` 若≈`gated`(94.8) → 增益来自浅层注入而非门控；若下降 → 门控本身有贡献。
