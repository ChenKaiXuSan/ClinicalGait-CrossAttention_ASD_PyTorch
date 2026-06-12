# Pegasus 实验总览表（按当前脚本/代码状态）

这份表只记录当前 `pegasus/*.sh` 和 `configs/config.yaml` 里实际能看到的实验设置。旧版里有些条目是论文设计稿，例如 `run_train_early_fuse_add.sh` / `concat.sh` / `mul.sh`，但当前目录里没有这些脚本，所以这里拆成“已有脚本”和“待补实验”。

## 一、当前已有脚本

| 脚本 | 实验角色 | 实际 override | PBS array 含义 | 当前状态 |
|---|---|---|---|---|
| `run_train_3dcnn.sh` | RGB-only baseline | `model.fuse_method=none`, `train.fold=5` | 无 array | 可作为无 fusion 基线 |
| `run_train_skeleton_only.sh` | no-prior baseline | `model.fuse_method=none`, `train.attn_map=False`, `train.fold=5` | `0-4`，但脚本没有使用 `$PBS_SUBREQNO` | 当前工作区已删除；若恢复，建议加 `train.experiment=baseline_rgb_noattn` |
| `run_train_pose_atn_single.sh` | PoseGated 单层注入 | `model.fuse_method=pose_atn`, `model.ablation_study=single`, `model.fusion_layers=$PBS_SUBREQNO` | `0..4` = 只在对应 block 融合 | 可跑 |
| `run_train_pose_atn_multi.sh` | PoseGated 多层累加注入 | `model.fuse_method=pose_atn`, `model.ablation_study=multi`, `model.fusion_layers=$PBS_SUBREQNO` | `0..4` = `[0]` 到 `[0,1,2,3,4]` | 可跑 |
| `run_train_pose_gated_best.sh` | PoseGated full / final config | `model.fuse_method=pose_atn`, `model.ablation_study=multi`, `model.fusion_layers=$PBS_SUBREQNO` | `0..4`，其中 `4` 是 full `[0..4]` | 名字是 best，但实际仍扫 multi prefix |
| `run_train_pose_gated_bias0.sh` | gate bias 消融 | `model.ablation_study=multi`, `model.fusion_layers=4`, `model.gate_init_bias=0.0` | 无 array，固定 full `[0..4]` | 可跑 |
| `run_train_pose_gated_bias_neg1.sh` | gate bias 消融 | `model.ablation_study=multi`, `model.fusion_layers=4`, `model.gate_init_bias=-1.0` | 无 array，固定 full `[0..4]` | 可跑 |
| `run_train_pose_gated_nosidehead.sh` | side head 消融 | `model.ablation_study=multi`, `model.fusion_layers=4`, `model.use_side_heads=False` | 无 array，固定 full `[0..4]` | 可跑 |
| `run_train_pose_gated_nobgloss.sh` | loss 消融 | `model.ablation_study=multi`, `model.fusion_layers=4`, `loss.selection=["cls","attn_loss","tmp"]` | 无 array，固定 full `[0..4]` | 可跑 |
| `run_train_pose_gated_notmploss.sh` | loss 消融 | `model.ablation_study=multi`, `model.fusion_layers=4`, `loss.selection=["cls","attn_loss","bg"]` | 无 array，固定 full `[0..4]` | 可跑 |
| `run_train_se_atn_single.sh` | SE fusion 对比 | `model.fuse_method=se_atn`, `model.ablation_study=single`, `model.fusion_layers=$PBS_SUBREQNO` | 当前代码里 `fusion_layers=0..4` 映射为 prefix `[0]` 到 `[0..4]`，不是严格 single | 可跑，但脚本文案需要注意 |
| `run_train_cross_atn.sh` | Cross-attention 对比 | `model.fuse_method=cross_atn`, `model.fusion_layers=$PBS_SUBREQNO` | 当前代码里 `fusion_layers=0..4` 映射为 prefix `[0]` 到 `[0..4]` | 当前工作区已删除；若恢复，还需要接回 `cross_atn` trainer，并建议加 `train.experiment=cross_atn_prefix_$PBS_SUBREQNO` |

## 二、日志目录命名

当前 Pegasus 脚本已显式传入 `train.experiment=...`，因此日志会落到：

```text
logs/train/<train.experiment>/<date>/<time>/
```

已设置的 experiment tag：

| 脚本 | `train.experiment` |
|---|---|
| `run_train_3dcnn.sh` | `baseline_rgb_3dcnn` |
| `run_train_se_atn_single.sh` | `se_atn_prefix_${PBS_SUBREQNO}` |
| `run_train_pose_atn_single.sh` | `pose_atn_single_${PBS_SUBREQNO}` |
| `run_train_pose_atn_multi.sh` | `pose_atn_multi_${PBS_SUBREQNO}` |
| `run_train_pose_gated_best.sh` | `pose_gated_best_multi_${PBS_SUBREQNO}` |
| `run_train_pose_gated_bias0.sh` | `pose_atn_bias0_multi_4` |
| `run_train_pose_gated_bias_neg1.sh` | `pose_atn_bias_neg1_multi_4` |
| `run_train_pose_gated_nosidehead.sh` | `pose_atn_noside_multi_4` |
| `run_train_pose_gated_nobgloss.sh` | `pose_atn_nobg_multi_4` |
| `run_train_pose_gated_notmploss.sh` | `pose_atn_notmp_multi_4` |

这样 `bias0`、`bias_neg1`、`noside`、`nobg`、`notmp` 不会再混在默认目录名里，并且都固定在 full multi `[0,1,2,3,4]` 上做组件消融。

## 三、关键代码映射

`configs/config.yaml` 默认值：

```yaml
loss.selection: ["cls", "attn_loss", "bg", "tmp"]
model.use_side_heads: True
model.gate_init_bias: 2.0
model.fuse_method: pose_atn
model.fusion_layers: 5
model.ablation_study: single
train.fold: 5
```

PoseGated 的 `fusion_layers` 映射在 `project/models/pose_fusion_res_3dcnn.py`：

| `ablation_study` | `fusion_layers` | 实际融合层 |
|---|---:|---|
| `single` | `0` | `[0]` |
| `single` | `1` | `[1]` |
| `single` | `2` | `[2]` |
| `single` | `3` | `[3]` |
| `single` | `4` | `[4]` |
| `multi` | `0` | `[0]` |
| `multi` | `1` | `[0,1]` |
| `multi` | `2` | `[0,1,2]` |
| `multi` | `3` | `[0,1,2,3]` |
| `multi` | `4` | `[0,1,2,3,4]` |
| 任意 | `5` | `[0,1,2,3,4]` |

层含义：

| index | backbone block | channel |
|---:|---|---:|
| 0 | stem | 64 |
| 1 | layer1 | 256 |
| 2 | layer2 | 512 |
| 3 | layer3 | 1024 |
| 4 | layer4 | 2048 |

注意：多数 Pegasus 脚本里 `#PBS -t 0-4` 不是 fold index，而是传给 `model.fusion_layers` 的融合层/融合层前缀索引。`train.fold=5` 是训练配置里的 fold 数或 fold 参数，不能和 `$PBS_SUBREQNO` 混在一起解释。

## 四、建议的论文/实验对比矩阵

### A0. Baseline

| Row | Method | 脚本 | 参数 | 目的 |
|---|---|---|---|---|
| B1 | RGB-only | `run_train_3dcnn.sh` | `fuse_method=none` | 主基线 |
| B2 | RGB-only, no attn map | `run_train_skeleton_only.sh` | `fuse_method=none`, `attn_map=False` | 检查 dataloader 不加载 prior 的情况 |

说明：`run_train_skeleton_only.sh` 当前不是 skeleton-only。若论文需要 “skeleton-only”，需要新增 dataloader/model 支持仅骨架输入。

### A1. Fusion Method

| Row | Method | 当前脚本/代码 | 建议状态 |
|---|---|---|---|
| A1a | Early add | 代码支持 `model.fuse_method=add`，无 Pegasus 脚本 | 待补脚本 |
| A1b | Early concat | 代码支持 `model.fuse_method=concat`，无 Pegasus 脚本 | 待补脚本 |
| A1c | Early mul | 代码支持 `model.fuse_method=mul`，无 Pegasus 脚本 | 待补脚本 |
| A1d | SE fusion | `run_train_se_atn_single.sh` | 可跑；注意实际是 prefix mapping |
| A1e | Cross-attention | `run_train_cross_atn.sh` | 先恢复 `project/train.py` 的 `cross_atn` trainer 入口 |
| A1f | PoseGated | `run_train_pose_atn_single.sh` / `run_train_pose_atn_multi.sh` | 主方法 |

### A2. Gate Init Bias

| Row | Method | 脚本 | 参数 |
|---|---|---|---|
| A2a | bias = 2.0 | `run_train_pose_gated_best.sh` with `PBS_SUBREQNO=4` | full multi `[0..4]`, 默认 `model.gate_init_bias=2.0` |
| A2b | bias = 0.0 | `run_train_pose_gated_bias0.sh` | full multi `[0..4]`, `model.gate_init_bias=0.0` |
| A2c | bias = -1.0 | `run_train_pose_gated_bias_neg1.sh` | full multi `[0..4]`, `model.gate_init_bias=-1.0` |

说明：这组三个实验固定 full multi `[0,1,2,3,4]`，只比较 gate 初始化。

### A3. Side Head

| Row | Method | 脚本 | 参数 |
|---|---|---|---|
| A3a | with side head | `run_train_pose_gated_best.sh` with `PBS_SUBREQNO=4` | full multi `[0..4]`, 默认 `model.use_side_heads=True` |
| A3b | without side head | `run_train_pose_gated_nosidehead.sh` | full multi `[0..4]`, `model.use_side_heads=False` |

说明：固定 full multi `[0,1,2,3,4]` 后对比 side head，避免和 layer ablation 混在一起。

### A4. Loss Components

| Row | Method | 脚本 | `loss.selection` |
|---|---|---|---|
| A4a | all losses | `run_train_pose_gated_best.sh` with `PBS_SUBREQNO=4` | full multi `[0..4]`, `["cls","attn_loss","bg","tmp"]` |
| A4b | w/o bg loss | `run_train_pose_gated_nobgloss.sh` | full multi `[0..4]`, `["cls","attn_loss","tmp"]` |
| A4c | w/o tmp loss | `run_train_pose_gated_notmploss.sh` | full multi `[0..4]`, `["cls","attn_loss","bg"]` |

### A5. Fusion Layers

| Row | Method | 脚本 | 对比内容 |
|---|---|---|---|
| A5a-e | single layer | `run_train_pose_atn_single.sh` | `[0]`, `[1]`, `[2]`, `[3]`, `[4]` |
| A5f-j | multi prefix | `run_train_pose_atn_multi.sh` | `[0]`, `[0,1]`, `[0,1,2]`, `[0,1,2,3]`, `[0,1,2,3,4]` |
| Final | full PoseGated | `run_train_pose_gated_best.sh` with `PBS_SUBREQNO=4` | `[0,1,2,3,4]` + side head + all losses + default bias |

## 五、当前最该补/修的地方

1. 修正文案：只有 fusion-layer 脚本里的 `#PBS -t 0-4` 表示 fusion layer / prefix index；组件消融脚本已固定 full multi `[0,1,2,3,4]`。
2. 决定是否恢复 `cross_atn`：`project/train.py` 里 `cross_atn` 分支目前被注释；若要跑 `run_train_cross_atn.sh`，需要接回对应 trainer。
3. 补 early fusion 脚本：`add` / `mul` / `concat` 在代码里支持，但 Pegasus 目录没有对应脚本。
4. 重新命名或修正 `run_train_skeleton_only.sh`：当前它是 `attn_map=False` baseline，不是真正 skeleton-only。
5. 如果需要节省算力，`run_train_pose_gated_best.sh` 可只提交 `PBS_SUBREQNO=4` 作为 full multi 主结果和 A2/A3/A4 的默认对照。

## 六、推荐跑实验顺序

1. 先跑主结果：`run_train_3dcnn.sh`、`run_train_pose_gated_best.sh` 的 `PBS_SUBREQNO=4`。
2. 再跑 A5：`run_train_pose_atn_single.sh` 和 `run_train_pose_atn_multi.sh`，报告层位置/层数消融。
3. 跑 A2/A3/A4：这些脚本已经固定 full multi `[0,1,2,3,4]`，得到干净组件消融。
4. 最后补 A1 的外部方法：SE、cross-attn、early add/mul/concat。

## 七、Figure 建议

| Figure | 内容 | 数据来源 |
|---|---|---|
| Fig. 1 | PoseGated block / overall architecture | 方法图 |
| Fig. 2 | Fusion method comparison | A1 |
| Fig. 3 | Single vs multi fusion layer | A5 |
| Fig. 4 | Gate init bias curve | A2 |
| Fig. 5 | Side head / loss component ablation | A3 + A4 |
| Fig. 6 | Per-class ROC / confusion matrix | final model |
| Fig. 7 | Gate/attention visualization case study | final model checkpoint |
