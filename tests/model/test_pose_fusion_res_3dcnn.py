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
            }
        }
    )

    model = PoseFusionRes3DCNN(cfg)
    assert model.fusion_layers == expected_blocks

    rgb = torch.randn(2, 3, 8, 224, 224)
    attn = torch.randn(2, 1, 8, 224, 224)

    out = model(rgb, attn)
    assert out.shape == (2, 4)


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
            }
        }
    )

    model = PoseFusionRes3DCNN(cfg)
    assert model.fusion_layers == expected_blocks

    rgb = torch.randn(2, 3, 8, 224, 224)
    attn = torch.randn(2, 1, 8, 224, 224)

    out = model(rgb, attn)
    assert out.shape == (2, 4)
