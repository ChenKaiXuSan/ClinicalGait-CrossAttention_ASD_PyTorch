#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/tests/model/test_res_3dcnn.py
Project: /workspace/code/tests/model
Created Date: Tuesday June 24th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday June 24th 2025 3:21:32 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import torch
from omegaconf import OmegaConf
import pytest

from project.models.res_3dcnn import Res3DCNN


@pytest.mark.parametrize("fuse_method", ["late", "mul", "add", "none", "avg"])
def test_res3dcnn_fuse_method_output_shape(fuse_method):
    hparams = OmegaConf.create(
        {
            "model": {
                "model_class_num": 3,
                "fuse_method": fuse_method,
            }
        }
    )

    model = Res3DCNN(hparams)
    video = torch.randn(2, 3, 8, 224, 224)
    attn_map = torch.randn(2, 1, 8, 224, 224)

    output = model(video, attn_map)

    assert output.shape == (
        2,
        3,
    ), f"[{fuse_method}] Expected shape (2, 3), but got {output.shape}"
