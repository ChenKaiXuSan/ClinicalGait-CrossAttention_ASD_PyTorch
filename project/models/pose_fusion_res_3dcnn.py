#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: pose_fusion_res_3dcnn.py
Author: Kaixu Chen <chenkaixusan@gmail.com>
-------------------------------------------------
A 3-D CNN backbone (default R(2+1)D-18) that fuses RGB clips and
key-point–derived attention maps with a lightweight **Pose-Gate Fusion**
module.  The gate learns channel-wise weights conditioned on both RGB and
pose cues – an inexpensive yet effective alternative to cross-attention.

· `fusion_layers` lets you choose at which ResNet stages you fuse
· `save_attention_maps()` dumps the learned gate weights for inspection

Input shapes
------------
RGB  : **(N, 3, T, H, W)**
Attn : **(N, 1, T, H, W)** – same temporal length as RGB, single channel

Output
------
Logits: **(N, num_classes)**
"""
import logging
import os
from typing import List, Sequence, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from project.models.base_model import BaseModel

logger = logging.getLogger(__name__)


# -------------------------- Pose-Gate Fusion Block ---------------------------
class PoseGateFusion(nn.Module):
    def __init__(self, in_channels: int, context_channels: int = 1):
        super().__init__()
        self.rgb_conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.attn_conv = nn.Conv3d(context_channels, in_channels, kernel_size=1)

        self.gate = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.last_scale: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_conv(x)
        attn_feat = self.attn_conv(attn)

        gate_input = torch.cat([rgb_feat, attn_feat], dim=1)
        gate_weight = self.gate(gate_input)
        self.last_scale = gate_weight.detach()

        return rgb_feat * gate_weight + attn_feat * (1.0 - gate_weight)


# -------------------------- Fusion Config Mapping ----------------------------
FUSE_LAYERS_MAPPING = {
    "single": {i: [i] for i in range(5)},
    "multi": {
        0: [],
        1: [0],
        2: [0, 1],
        3: [0, 1, 2],
        4: [0, 1, 2, 3],
        5: [0, 1, 2, 3, 4],
    },
}


# ---------------------------- Main Model Class -------------------------------
class PoseFusionRes3DCNN(BaseModel):
    def __init__(self, hparams: OmegaConf) -> None:
        super().__init__(hparams)

        ablation = hparams.model.get("ablation_study", "multi")
        fusion_layers = hparams.model.fusion_layers
        if isinstance(fusion_layers, int):
            fusion_layers = FUSE_LAYERS_MAPPING[ablation].get(fusion_layers, [])
        self.fusion_layers: List[int] = fusion_layers
        logger.info(f"Fusion at blocks: {self.fusion_layers}")

        self.ckpt = hparams.model.ckpt_path
        self.model_class_num = hparams.model.model_class_num
        self.model = self.init_resnet(self.model_class_num, self.ckpt)

        self.blocks = nn.ModuleList([self.model.blocks[i] for i in range(6)])

        dim_list = [64, 256, 512, 1024, 2048]
        self.attn_fusions = nn.ModuleList([
            PoseGateFusion(dim) if i in self.fusion_layers else nn.Identity()
            for i, dim in enumerate(dim_list)
        ])

    def forward(self, video: torch.Tensor, attn_map: torch.Tensor) -> torch.Tensor:
        x = video
        for idx in range(5):
            x = self.blocks[idx](x)
            if not isinstance(self.attn_fusions[idx], nn.Identity):
                attn_resized = F.interpolate(
                    attn_map, size=x.shape[-3:], mode="trilinear", align_corners=False
                )
                x = self.attn_fusions[idx](x, attn_resized)
        x = self.blocks[5](x)

        # TODO: 这里应该返回 logits，各个网络层生成的attn map
        return x

    def save_attention_maps(self, save_dir: str = "fusion_vis") -> None:
        os.makedirs(save_dir, exist_ok=True)
        for idx, fusion in enumerate(self.attn_fusions):
            if isinstance(fusion, PoseGateFusion) and fusion.last_scale is not None:
                scale = fusion.last_scale.mean(dim=(0, 2, 3, 4)).cpu().numpy()
                plt.figure(figsize=(12, 3))
                plt.bar(range(len(scale)), scale)
                plt.title(f"Gate Weights – Block {idx}")
                plt.xlabel("Channel")
                plt.ylabel("Weight")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"block{idx}_gate.png"))
                plt.close()


# ---------------------------- Quick Test Entry -------------------------------
if __name__ == "__main__":
    cfg = OmegaConf.create(
        {
            "model": {
                "model_class_num": 3,
                "fusion_layers": 3,
                "ckpt_path": "",
                "ablation_study": "multi",  # "single" or "multi"
            }
        }
    )
    model = PoseFusionRes3DCNN(cfg)
    rgb = torch.randn(2, 3, 8, 112, 112)
    pose = torch.randn(2, 1, 8, 112, 112)
    output = model(rgb, pose)
    print("Output shape:", output.shape)