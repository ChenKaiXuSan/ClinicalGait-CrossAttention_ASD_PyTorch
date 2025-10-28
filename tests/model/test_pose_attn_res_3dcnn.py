#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/tests/model/test_pose_fusion_res_3dcnn.py
Project: /workspace/code/tests/model
Created Date: Friday June 27th 2025
Author: Kaixu Chen
-----
Comment:
This file contains unit tests for the PoseFusionRes3DCNN model.
It tests the model's behavior under different fusion layer configurations
and ensures that the output shapes are as expected.

Have a good code time :)
-----
Last Modified: Friday July 18th 2025 1:20:56 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
import pytest
import torch
from omegaconf import OmegaConf
from project.models.pose_fusion_res_3dcnn import PoseFusionRes3DCNN


@pytest.mark.parametrize(
    "fuse_layer_idx,expected_blocks",
    [
        (0, [0]),
        (1, [1]),
        (2, [2]),
        (3, [3]),
        (4, [4]),
    ],
)
def test_single_fusion_layers(fuse_layer_idx, expected_blocks):
    """Test 'single' ablation: only one fusion point active."""
    cfg = OmegaConf.create(
        {
            "model": {
                "model_class_num": 4,
                "fusion_layers": fuse_layer_idx,
                "ckpt_path": "",
                "ablation_study": "single",
                "attn_channels": 1,
                "use_side_heads": True,
                "fusion_norm": "gn",
                "fusion_residual": True,
                "gate_init_bias": 2.0,
                "gate_temp": 1.0,
            }
        }
    )

    model = PoseFusionRes3DCNN(cfg)
    assert model.fusion_layers == expected_blocks

    rgb = torch.randn(2, 3, 8, 224, 224)
    attn = torch.randn(2, 1, 8, 224, 224)

    logits, aux = model(rgb, attn, return_aux=True)

    assert logits.shape == (2, 4)
    assert aux is not None


@pytest.mark.parametrize(
    "fuse_layer_idx,expected_blocks",
    [
        (0, []),
        (1, [0]),
        (2, [0, 1]),
        (3, [0, 1, 2]),
        (4, [0, 1, 2, 3]),
        (5, [0, 1, 2, 3, 4]),
    ],
)
def test_multi_fusion_layers(fuse_layer_idx, expected_blocks):
    """Test 'multi' ablation: cumulative fusion up to N-th layer."""
    cfg = OmegaConf.create(
        {
            "model": {
                "model_class_num": 4,
                "fusion_layers": fuse_layer_idx,
                "ckpt_path": "",
                "ablation_study": "multi",
                "attn_channels": 1,
                "use_side_heads": True,
                "fusion_norm": "gn",
                "fusion_residual": True,
                "gate_init_bias": 2.0,
                "gate_temp": 1.0,
            }
        }
    )

    model = PoseFusionRes3DCNN(cfg)
    assert model.fusion_layers == expected_blocks

    rgb = torch.randn(2, 3, 8, 224, 224)
    attn = torch.randn(2, 1, 8, 224, 224)

    logits, aux = model(rgb, attn, return_aux=True)

    assert logits.shape == (2, 4)
    assert aux is not None


@pytest.mark.parametrize("tmp_path", ["tests/temp_test_dir"], indirect=True)
def test_save_attention_and_side_maps(tmp_path):
    cfg = OmegaConf.create(
        {
            "model": {
                "model_class_num": 3,
                "fusion_layers": [1, 2],  # 至少一个融合层以产生 gate_scales
                "ckpt_path": "",
                "ablation_study": "multi",
                "attn_channels": 1,
                "use_side_heads": True,  # 以生成 side_preds
            }
        }
    )
    model = PoseFusionRes3DCNN(cfg)
    rgb = torch.randn(2, 3, 8, 224, 224)
    attn = torch.randn(2, 1, 8, 224, 224)

    logits, aux = model(rgb, attn, return_aux=True)
    assert logits.shape == (2, 3)

    # 保存 gate 柱状图
    save_dir_g = tmp_path / "gates"
    model.save_attention_maps(str(save_dir_g))
    pngs_g = list(save_dir_g.glob("block*_gate.png"))
    assert len(pngs_g) >= 1

    # 保存侧头 2D 网格
    save_dir_s = tmp_path / "side_maps"
    model.save_side_feature_maps(
        aux["side_preds"],
        save_dir=str(save_dir_s),
        aggregate="mean",
        max_channels=8,
        ncols=4,
    )
    # 任意一层应有输出
    found = any(
        layer_dir.is_dir() and list(layer_dir.glob("*.png"))
        for layer_dir in save_dir_s.glob("layer*")
    )
    assert found is True
