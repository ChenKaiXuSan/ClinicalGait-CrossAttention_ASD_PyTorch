# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ClinicalGait-CrossAttention ASD — PyTorch 实现：使用视频步态分析进行成人脊柱畸形（ASD）分类的深度学习项目。模型融合 RGB 视频与从关键点提取的临床注意力图（doctor-annotated attention maps）。核心架构基于 PyTorch Lightning + Hydra 配置系统，主干使用 pytorchvideo 的 SlowFast R50 3D CNN（SlowR50），并在此基础上注入不同类型的临床先验注意力机制。主推方案是 PoseGated（`pose_atn`）channel-wise gated fusion。

## 目录结构

```
project/                 # 核心代码
├── train.py             # Hydra entry point：K-fold 训练循环入口
├── eval.py              # Hydra entry point：评估入口（自动从 log 目录挑选每 fold 最优 ckpt）
├── cross_validation.py  # StratifiedGroupKFold 交叉验证 + over/under sampler + magic_move
├── dataloader/          # WalkDataModule (LightningDataModule)
│   ├── data_loader.py   # 数据模块，collate_fn 做 label mapping (disease_to_num_mapping_Dict)
│   ├── whole_video_dataset.py
│   └── med_attn_map.py
├── models/              # 模型定义（纯 nn.Module，与 trainer 分离）
│   ├── weight_loader.py # init_slow_r50()：SlowR50 权重下载/加载 + 首层/头部修改
│   ├── res_3dcnn.py     # Res3DCNN（基础 backbone，early fusion: add/mul/concat/avg + late）
│   ├── cross_attn_res_3dcnn.py   # Cross-Attention Fusion (QKV Conv3d)
│   ├── se_attn_res_3dcnn.py      # Squeeze-and-Excitation Fusion
│   ├── pose_fusion_res_3dcnn.py  # PoseFusionRes3DCNN（主推）：channel-wise gated mixing + side heads
│   └── make_model.py    # select_model()：根据 hparams 选择模型的工厂函数
├── trainer/             # LightningModule trainers
│   ├── baseline/train_3dcnn.py   # Res3DCNNTrainer (fuse_method=none)
│   ├── early/train_early_fusion.py  # EarlyFusion3DCNNTrainer (add/mul/concat/avg)
│   ├── late/train_late_fusion.py    # LateFusion3DCNNTrainer (late)
│   └── mid/             # PoseAttnTrainer (pose_atn), SEAttnTrainer (se_atn)
└── utils/               # helper.py (save_helper), save_CAM.py (feature map dump)
configs/                 # Hydra YAML 配置
├── config.yaml          # 训练主配置（loss/data/model/train/hydra）
├── eval.yaml            # 评估配置
└── prepare_*.yaml       # 数据准备配置
pegasus/                 # PBS 作业脚本（HPC 集群提交）
├── EXPERIMENTS.md       # ⭐ 实验总览表：每个脚本的 override、PBS array 含义、experiment tag
└── run_train_*.sh       # 各实验脚本
prepare_skeleton_dataset/ # 姿态关键点提取（YOLOv8）
tests/                   # pytest 模型 shape 测试
```

## 关键架构理解

### 1. 入口与配置流
- `python -m project.train` 通过 Hydra 加载 `configs/config.yaml`（`python -m project.eval` 加载 `eval.yaml`）
- `@hydra.main` → `init_params()` → `DefineCrossValidation()` 生成 K-fold 索引 → 循环每 fold 调用 `train()`
- `train()` 根据 `hparams.model.fuse_method` 路由到不同 trainer（见下表）
- 每 fold 结束后自动 `trainer.test(ckpt_path="best")`，metrics 写入 `<log_path>/metrics/fold_N_metrics.txt`
- Checkpoint/EarlyStopping 均监控 `val/video_acc`（mode=max，patience 默认 10，保存 top-2 + last）

### 2. fuse_method → trainer/model 路由

| fuse_method | model (make_model.py) | trainer (train.py) |
|---|---|---|
| `none` | Res3DCNN | Res3DCNNTrainer |
| `add` / `mul` / `concat` / `avg` | Res3DCNN | EarlyFusion3DCNNTrainer |
| `late` | Res3DCNN (LateFusionBlock, 可学习 alpha) | LateFusion3DCNNTrainer |
| `pose_atn`（主推） | PoseFusionRes3DCNN | PoseAttnTrainer |
| `se_atn` | SEFusionRes3DCNN | SEAttnTrainer |
| `cross_atn` | CrossAttentionRes3DCNN | CrossAttentionTrainer（仅 cls loss；THW×THW 注意力在浅层会 OOM，只在深层融合，如 `model.fusion_layers=[3,4]`） |

### 3. fusion_layers 语义（pose_fusion_res_3dcnn.py）
`fusion_layers` 为 int 时的解析规则（0=stem, 1-4=layer1-4）：
- `ablation_study=single` + `fusion_layers=i`（0-4）→ 仅在第 i 层融合 `[i]`
- `ablation_study=multi` + `fusion_layers=i`（0-4）→ prefix 融合 `[0..i]`
- `fusion_layers=5` → 全层 `[0,1,2,3,4]`，**无视 ablation_study**（config 默认值）

### 4. PoseAttn 多任务损失（trainer/mid/train_pose_attn.py）
```
total = cls_loss + Σ_i lambda_list[i]*attn_loss_i(BCE+Dice) + w_bg*bg_loss + w_temp*tv_l1
```
- side heads（`model.use_side_heads`）输出 per-joint heatmap logits，与医生标注 doctor_hm 做监督
- `loss.selection`（默认 `["cls","attn_loss","bg","tmp"]`）控制启用哪些损失项：移除某项即把对应权重置 0；`"cls"` 必须保留（否则 assert）
- `model.gate_init_bias`（默认 2.0）：PoseGated 门控初始偏置，正值使早期训练偏向 RGB 分支

### 5. Cross-Validation（cross_validation.py）
`DefineCrossValidation` 使用 `StratifiedGroupKFold`（按患者分组，防数据泄漏）；`data.sampling` 控制 over/under-sampling（imblearn）；`magic_move()` 在 train/val 间交换非 ASD 样本以平衡分布。类标签映射：`cross_validation.py:class_num_mapping_Dict` 与 `data_loader.py:disease_to_num_mapping_Dict`。

fold 索引缓存在 `data.index_mapping/<class_num>/<sampling>_K<fold>/`（目录名带 K，换 fold 数不会复用旧划分；有目录锁支持并行作业）。首次构建会复制视频、耗时较长，提交并行作业前先跑一次：
```bash
python -m project.prepare_folds data.root_path=/path/to/data
```

单折模式：`train.fold_idx=k`（>=0）只训练第 k 折，供 PBS array 每折一个节点使用；`-1`（默认）串行跑全部折。

### 6. 权重加载（models/weight_loader.py）
`init_slow_r50(weight_path, class_num)`：若 `ckpt_path` 指定但文件不存在，自动从 pytorchvideo model zoo 下载 SLOW_8x8_R50 到该路径；加载后替换 stem conv（kernel 7³, stride (1,2,2)）和最后 FC 层（匹配 class_num）。`ckpt_path` 为空/None 则随机初始化。

## 常用命令

### 安装依赖
```bash
pip install -r requirements.txt
pip install -r tests/requirements.txt   # 测试依赖
```

### 运行训练（Hydra 参数覆盖）
```bash
# PoseGated 训练（single layer i，5-fold）
python -m project.train data.root_path=/path/to/data model.fuse_method=pose_atn train.fold=5 model.ablation_study=single model.fusion_layers=0

# PoseGated full（全层融合，config 默认 fusion_layers=5 即全层）
python -m project.train data.root_path=/path/to/data model.fuse_method=pose_atn model.ablation_study=multi model.fusion_layers=4

# SE Attention
python -m project.train data.root_path=/path/to/data model.fuse_method=se_atn model.ablation_study=single model.fusion_layers=0

# Baseline（RGB only）
python -m project.train data.root_path=/path/to/data model.fuse_method=none

# 损失消融（去掉 bg loss）
python -m project.train model.fuse_method=pose_atn 'loss.selection=["cls","attn_loss","tmp"]'
```

### 评估
```bash
python -m project.eval   # 使用 configs/eval.yaml；按 ckpt 文件名中的 val/video_acc 自动挑每 fold 最优权重
```

### HPC 集群提交 (PBS / Pegasus)
实验矩阵、每个脚本的 array 展开和 `train.experiment` tag 见 `pegasus/EXPERIMENTS.md`（权威记录）。**所有脚本的 PBS array 均展开到 fold 维度，每个 sub-job 用 `train.fold_idx` 只跑一折、独占一个节点**；组合 array 的规则是 `SUBREQNO = 外层索引*3 + fold`。
```bash
python -m project.prepare_folds data.root_path=...   # 提交前先构建 fold 缓存（一次）
qsub pegasus/run_train_pose_gated_best.sh     # 主结果 full [0..4]，array 0-2 (fold)
qsub pegasus/run_train_pose_atn_single.sh     # 单层消融，array 0-14 (layer×fold)
qsub pegasus/run_train_pose_atn_multi.sh      # prefix 消融 P0-P3，array 0-11
qsub pegasus/run_train_se_atn.sh              # SE fusion，array 0-14 (prefix×fold)
qsub pegasus/run_train_cross_atn.sh           # Cross-attention {L3,L4,L34}，array 0-8
qsub pegasus/run_train_early_fuse.sh          # early add/mul/concat，array 0-8
qsub pegasus/run_train_3dcnn.sh               # RGB baseline，array 0-2 (fold)
```
日志落在 `logs/train/<train.experiment>/<date>/<time>/`，新实验 tag 均带 `_f{fold}` 后缀。

### 运行测试
```bash
cd tests/
pytest -xvs model/                            # 模型 shape 单元测试
pytest -xvs model/test_pose_attn_res_3dcnn.py # 单个测试文件
```
pytest 配置在 `setup.cfg`（`--strict --doctest-modules`）；flake8 配置同文件（max-line-length=120）。每个模型测试用 dummy input `(2, 3, 8, 224, 224)` video + `(2, 1, 8, 224, 224)` attn_map，验证输出 `(2, num_classes)`。

## 配置关键字段 (configs/config.yaml)

| 字段 | 默认值 | 说明 |
|---|---|---|
| `data.root_path` | `/mnt/data/xchen/asd_data` | 数据根目录（Pegasus 上为 `/work/SKIING/chenkaixu/data/asd_dataset`） |
| `data.sampling` | `over` | 采样策略：`over`/`under`/`none` |
| `model.model_class_num` | 3 | 分类数: 2(ASD/non-ASD), 3(ASD/DHS/LCS_HipOA), 4(+normal) |
| `model.fuse_method` | `pose_atn` | 融合方式（见路由表） |
| `model.fusion_layers` | 5 | 融合层索引：0-4 单层/prefix，5=全层 |
| `model.ablation_study` | `single` | `single`(单层) / `multi`(prefix)；fusion_layers=5 时无效 |
| `model.use_side_heads` | True | side head 辅助监督开关 |
| `model.gate_init_bias` | 2.0 | PoseGated 门控初始偏置 |
| `model.ckpt_path` | `checkpoints/SLOW_8x8_R50.pyth` | SlowR50 预训练权重（缺失时自动下载） |
| `loss.selection` | `["cls","attn_loss","bg","tmp"]` | 启用的损失项（`cls` 必选） |
| `loss.lambda_list` | [0.25, 0.5, 0.75, 1.0] | 各层 side head attn loss 权重 |
| `loss.w_bg` / `loss.w_temp` | 0.2 / 0.05 | 背景抑制 / 时间平滑损失权重 |
| `train.max_epochs` | 50 | 训练轮数（EarlyStopping patience=10） |
| `train.gpu` | 2 | GPU 数量（Pegasus 脚本覆盖为 1） |
| `train.attn_map` | True | 是否使用 attention map 输入 |
| `train.fold` | 3 | K-fold 折数（2026-07 从 5 改为 3，旧 5-fold 结果作废） |
| `train.fold_idx` | -1 | -1=串行跑全部折；>=0=只跑该折（PBS 每折一节点用） |

## 数据格式

数据集由 JSON 文件描述（`data.data_info_path` 下），每个 JSON 包含：
- `video_name`, `video_path`, `disease` (ASD/DHS/LCS_HipOA)
- skeleton path → `.pkl` 关键点序列（`data.skeleton_path`）
- doctor results → 医生标注注意力热图，用于 side head 监督（`data.doctor_results_path`）

## 注意事项

- README.md 的 Project Structure / Quick Start 部分已过时（描述的是旧版目录结构），以本文件和实际代码为准。
- `pegasus/EXPERIMENTS.md` 是实验设计的权威记录：改动 pegasus 脚本或新增实验时应同步更新它。
- trainer 与 model 严格分离：新增融合方式需要同时在 `make_model.py`（模型选择）和 `train.py`（trainer 路由）注册。
