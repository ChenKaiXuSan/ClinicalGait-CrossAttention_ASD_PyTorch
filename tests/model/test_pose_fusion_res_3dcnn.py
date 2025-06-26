#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/tests/model/test_pose_fusion_res_3dcnn.py
Project: /workspace/code/tests/model
Created Date: Friday June 27th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday June 27th 2025 1:25:20 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import pytest
import torch
from types import SimpleNamespace
from project.models.pose_fusion_res_3dcnn import PoseFusionRes3DCNN

@pytest.mark.parametrize(
    "fusion_layers",
    [
        [],
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4],
    ],
)
def test_sefusion_res3dcnn_forward(fusion_layers):
    hparams = SimpleNamespace(
        model=SimpleNamespace(
            model_class_num=3, fusion_layers=fusion_layers,
            ckpt_path="checkpoints/SLOW_8x8_R50"
        )
    )

    model = PoseFusionRes3DCNN(hparams)
    video = torch.randn(2, 3, 8, 224, 224)
    attn_map = torch.randn(2, 1, 8, 224, 224)

    out = model(video, attn_map)

    assert out.shape == (2, 3), f"Expected output shape (2, 3), but got {out.shape}"
