# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ClinicalAttention AS — PyTorch 实现：使用视频步态分析进行成人脊柱畸形（ASD）分类的深度学习项目。模型融合 RGB 视频与从关键点提取的临床注意力图（skeleton/pose attention maps）。核心架构基于 PyTorch Lightning + Hydra 配置系统，主干使用 pytorchvideo 提供的 SlowFast R50 3D CNN（SlowR50），并在此基础上注入不同类型的临床先验注意力机制。

## 目录结构

```
project/                 # 核心代码
├── main.py              # Hydra entry point：K-fold 训练循环入口
├── cross_validation.py  # StratifiedGroupKFold 交叉验证 + over/under sampler
├── dataloader/          # WalkDataModule (LightningDataModule)
│   ├── data_loader.py   # 数据模块，collate_fn 做 label mapping
│   ├── whole_video_dataset.py / batch_video_dataset.py
│   └── med_attn_map.py
├── models/              # 模型定义
│   ├── base_model.py    # BaseModel + SlowR50 权重加载
│   ├── res_3dcnn.py     # Res3DCNN（基础 backbone，支持 add/mul/concat/avg/late fuse）
│   ├── cross_attn_res_3dcnn.py   # Cross-Attention Fusion (QKV Conv3d)
│   ├── se_attn_res_3dcnn.py      # Squeeze-and-Excitation Fusion
│   ├── pose_fusion_res_3dcnn.py  # Pose-Attn Fusion (channel-wise gated)
│   └── make_model.py    # 根据 hparams 选择模型的工厂函数
├── trainer/             # LightningModule trainers
│   ├── baseline/        # Res3DCNNTrainer, train_cnn, train_two_stream 等
│   ├── mid/             # PoseAttnTrainer, SEAttnTrainer, CrossAttentionTrainer
│   └── early/late/      # Early/Late Fusion trainers
└── utils/               # helper.py (save_helper), save_CAM.py (feature map dump)
configs/                 # Hydra YAML 配置
├── config.yaml          # 主配置（data/model/train/hydra/loss）
├── eval.yaml            # 评估配置
└── prepare_*.yaml       # 数据准备配置
pegasus/                 # PBS 作业脚本，用于 HPC 集群提交训练
prepare_skeleton_dataset/ # 姿态关键点提取（YOLOv8）
tests/                   # pytest 测试
```

## 关键架构理解

### 1. 入口与配置流
- `python -m project.main` 通过 Hydra 加载 `configs/config.yaml`
- `@hydra.main` 装饰器调用 `init_params()` → `DefineCrossValidation()` 生成 K-fold 索引 → 循环每 fold 调用 `train()`
- `train()` 根据 `hparams.model.fuse_method` 路由到不同 trainer

### 2. 模型融合方式 (`fuse_method`)
| fuse_method | 说明 |
|---|---|
| `none` / `add` / `mul` / `avg` / `concat` | Early fusion：在输入端与 attn_map 组合后送入 ResNet |
| `late` | Late fusion：主干特征与注意力图通过 `LateFusionBlock`（可学习 alpha）融合 |
| `cross_atn` | CrossAttentionRes3DCNN：在每个 res block 后用 QKV 做 cross-attention |
| `se_atn` | SEFusionRes3DCNN：在每个 res block 后用 SE 机制生成 channel-wise scale |
| `pose_atn` | PoseFusionRes3DCNN（推荐）：channel-wise gated mixing，支持 side heads 辅助监督 |

### 3. Pose-Attn Fusion（主推方案）
PoseAttnTrainer 使用多任务损失：
```
total = lambda[0]*cls_loss + lambda[1]*attn_loss(BCE+Dice) + w_bg*bg_loss + w_temp*tv_l1
```
- side heads 输出 per-joint heatmap logits，与医生标注的 doctor_hm (attention map) 做监督
- `ablation_study` 控制融合层：`single`(仅一层) / `multi`(多层的 prefix)

### 4. Cross-Validation
`DefineCrossValidation` 使用 `StratifiedGroupKFold`（按患者分组，防数据泄漏），支持 over-sampling / under-sampling。`magic_move()` 在 train/val 间交换非 ASD 样本以平衡分布。

## 常用命令

### 安装依赖
```bash
pip install -r requirements.txt
pip install -r tests/requirements.txt   # 测试依赖
```

### 运行训练（Hydra 参数覆盖）
```bash
# Pose Attention 训练 (single layer, 5-fold)
python -m project.main data.root_path=/path/to/data model.fuse_method=pose_atn train.fold=5 model.ablation_study=single

# SE Attention 训练 (multi layers)
python -m project.main data.root_path=/path/to/data model.fuse_method=se_atn train.fold=5 model.ablation_study=multi model.fusion_layers=0,1,2,3,4

# Cross-Attention 训练
python -m project.main data.root_path=/path/to/data model.fuse_method=cross_atn train.fold=5

# Base Res3DCNN
python -m project.main data.root_path=/path/to/data model.fuse_method=none
```

### HPC 集群提交 (PBS)
```bash
qsub pegasus/run_train_pose_atn_single.sh    # Pose Attention
qsub pegasus/run_train_cross_atn.sh           # Cross Attention
qsub pegasus/run_train_se_atn_single.sh       # SE Attention
qsub pegasus/run_train_3dcnn.sh               # Baseline
```

### 运行测试
```bash
cd tests/
pytest -xvs . --flake8                        # flake8 lint
pytest -xvs model/                           # 模型单元测试
```

## 配置关键字段 (configs/config.yaml)

| 字段 | 默认值 | 说明 |
|---|---|---|
| `data.root_path` | `/workspace/data` | 数据根目录 |
| `data.sampling` | `over` | 过采样/欠采样策略：`over`/`under`/`none` |
| `model.model_class_num` | 3 | 分类数: 2(ASD/non-ASD), 3(DHS/LCS), 4(normal) |
| `model.fusion_layers` | 5 | Fusion 层索引（0=stem..4=layer4）或快捷值 |
| `model.ckpt_path` | `checkpoints/SLOW_8x8_R50.pyth` | SlowR50 Kinetics 预训练权重 |
| `train.max_epochs` | 50 | 训练轮数 |
| `train.attn_map` | True | 是否使用 attention map 输入 |
| `loss.lambda_list` | [0.25, 0.5, 0.75, 1.0] | side head 各层 attn loss 权重 |

## 数据格式

数据集由 JSON 文件描述，每个 JSON 包含：
- `video_name`, `video_path`, `disease` (ASD/DHS/LCS_HipOA)
- skeleton path → `.pkl` 中的关键点序列
- doctor results → 医生标注的注意力热图（用于 side head 监督）

类标签映射见 `cross_validation.py:class_num_mapping_Dict` 和 `data_loader.py:disease_to_num_mapping_Dict`。

## 模型权重加载

`base_model.init_resnet()` 从 pytorchvideo hub 下载 SlowR50 Kinetics-400 预训练权重，自动修改首层卷积（支持输入通道变化）和最后一层 FC（匹配分类数）。若 `ckpt_path` 为空则随机初始化。

## 测试文件说明

- `tests/model/test_res_3dcnn.py` — Res3DCNN fuse_method shape 测试
- `tests/model/test_pose_attn_res_3dcnn.py` — PoseAttn model shape test
- `tests/model/test_SE_3dcnn.py` — SE Attention model shape test
- `tests/model/test_cross_attention_3dcnn.py` — CrossAttention model shape test
- 每个测试创建 dummy input `(2, 3, 8, 224, 224)` video + `(2, 1, 8, 224, 224)` attn_map，验证输出为 `(2, num_classes)`
