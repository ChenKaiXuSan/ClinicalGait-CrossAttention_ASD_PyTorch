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

## 六、遗留事项

1. skeleton-only baseline（仅骨架输入）仍未实现，需要 dataloader/model 支持后另开脚本。
2. `analysis/` 的画图 notebook 需要适配新的 `_f{fold}` 日志目录命名（glob `<tag>_f*`）。
