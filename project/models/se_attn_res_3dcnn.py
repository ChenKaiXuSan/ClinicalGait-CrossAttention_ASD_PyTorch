#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/models/se_att_res_3dcnn.py
Project: /workspace/code/project/models
Created Date: Thursday June 19th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday June 19th 2025 5:36:16 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

fuse_layers_mapping = {
    0: [],
    1: [0],
    2: [0, 1],
    3: [0, 1, 2],
    4: [0, 1, 2, 3],
    5: [0, 1, 2, 3, 4],
}


class SEFusion(nn.Module):
    def __init__(self, in_channels, context_channels=1, reduction=8, name=None):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(context_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.save_attn = False
        self.name = name or "unnamed"
        self.last_scale = None  # 存储最后一次 scale（[B, C, 1, 1, 1]）

    def forward(self, x, context):
        scale = self.fc(self.pool(context))  # [B, C, 1, 1, 1]
        if self.save_attn:
            self.last_scale = scale.detach().cpu()
        return x * scale + x


class SEFusionRes3DCNN(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.model_class_num = hparams.model.model_class_num
        fusion_layers = hparams.model.fusion_layers
        if isinstance(fusion_layers, int):
            fusion_layers = fuse_layers_mapping[fusion_layers]

        self.fusion_layers = fusion_layers
        logger.info(f"Using SEFusionRes3DCNN with fusion layers: {self.fusion_layers}")
        self.model = self.init_resnet(self.model_class_num)

        self.attn_fusions = nn.ModuleList()
        dim_list = [64, 256, 512, 1024, 2048]
        for i, dim in enumerate(dim_list):
            if i in self.fusion_layers:
                fusion = SEFusion(dim, context_channels=1)
                # fusion.save_attn = True  # If needed, can be toggled
                self.attn_fusions.append(fusion)
            else:
                self.attn_fusions.append(nn.Identity())

    @staticmethod
    def init_resnet(class_num: int = 3) -> nn.Module:
        slow = torch.hub.load(
            "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
        )
        slow.blocks[0].conv = nn.Conv3d(
            3,
            64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )
        slow.blocks[-1].proj = nn.Linear(2048, class_num)
        return slow

    def forward(self, video: torch.Tensor, attn_map: torch.Tensor) -> torch.Tensor:
        x = self.model.blocks[0](video)
        if not isinstance(self.attn_fusions[0], nn.Identity):
            x = self.attn_fusions[0](
                x, F.interpolate(attn_map, size=x.shape[-3:], mode="trilinear")
            )

        x = self.model.blocks[1](x)
        if not isinstance(self.attn_fusions[1], nn.Identity):
            x = self.attn_fusions[1](
                x, F.interpolate(attn_map, size=x.shape[-3:], mode="trilinear")
            )

        x = self.model.blocks[2](x)
        if not isinstance(self.attn_fusions[2], nn.Identity):
            x = self.attn_fusions[2](
                x, F.interpolate(attn_map, size=x.shape[-3:], mode="trilinear")
            )

        x = self.model.blocks[3](x)
        if not isinstance(self.attn_fusions[3], nn.Identity):
            x = self.attn_fusions[3](
                x, F.interpolate(attn_map, size=x.shape[-3:], mode="trilinear")
            )

        x = self.model.blocks[4](x)
        if not isinstance(self.attn_fusions[4], nn.Identity):
            x = self.attn_fusions[4](
                x, F.interpolate(attn_map, size=x.shape[-3:], mode="trilinear")
            )

        x = self.model.blocks[5](x)
        return x

    def save_attention_maps(self, save_dir="se_attn_vis"):
        os.makedirs(save_dir, exist_ok=True)
        for i, fusion in enumerate(self.se_fusions):
            if fusion.last_scale is not None:
                scale = fusion.last_scale.squeeze().mean(
                    dim=0
                )  # 平均所有 batch，只保留 channel 维度
                scale_np = scale.numpy()  # [C]
                plt.figure(figsize=(10, 2))
                plt.bar(range(len(scale_np)), scale_np)
                plt.title(f"SE Fusion Scale - Layer {i} ({fusion.name})")
                plt.xlabel("Channel Index")
                plt.ylabel("Scale Value")
                plt.tight_layout()
                plt.savefig(f"{save_dir}/layer{i}_{fusion.name}_scale.png")
                plt.close()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    hparams = OmegaConf.create(
        {
            "model": {
                "model_class_num": 3,
                "fusion_layers": [
                    0,
                    2,
                    4,
                ],  # example: only apply SE fusion at res1, res3, res5
            }
        }
    )

    model = SEFusionRes3DCNN(hparams)
    video = torch.randn(2, 3, 8, 224, 224)
    attn_map = torch.randn(2, 1, 8, 224, 224)
    output = model(video, attn_map)
    print(output.shape)  # Expected output shape: [2, class_num]
