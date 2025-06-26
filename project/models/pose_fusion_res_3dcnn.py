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
from __future__ import annotations

import logging
import os
from typing import List, Sequence, Union

import matplotlib.pyplot as plt  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision.models.video import r3d_18
from project.models.base_model import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Pose-Gate Fusion block (channel-wise gating -> cheap & stable)
# ---------------------------------------------------------------------------


class PoseGateFusion(nn.Module):
    """Fuse RGB features with a 1-channel pose/attention map via gating."""

    def __init__(self, in_channels: int, context_channels: int = 1):
        super().__init__()
        # Bring both inputs to the same #channels ➜ concatenate ➜ predict gate
        self.rgb_conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.attn_conv = nn.Conv3d(context_channels, in_channels, kernel_size=1)

        self.gate = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.last_scale: torch.Tensor | None = None  # saved for visualisation

    def forward(
        self, x: torch.Tensor, attn: torch.Tensor
    ) -> torch.Tensor:  # noqa: D401, N802
        rgb_feat = self.rgb_conv(x)
        attn_feat = self.attn_conv(attn)

        gate_input = torch.cat([rgb_feat, attn_feat], dim=1)
        gate_weight = self.gate(gate_input)  # (N, C, T, H, W) in [0,1]
        self.last_scale = gate_weight.detach()

        return rgb_feat * gate_weight + attn_feat * (1.0 - gate_weight)


# Handy aliases when users give an *int* instead of a list
fuse_layers_mapping = {
    0: [],
    1: [0],
    2: [0, 1],
    3: [0, 1, 2],
    4: [0, 1, 2, 3],
    5: [0, 1, 2, 3, 4],
}


# ---------------------------------------------------------------------------
#  Main network
# ---------------------------------------------------------------------------


class PoseFusionRes3DCNN(BaseModel):
    """RGB ✕ Pose attention fusion on top of a 3-D ResNet backbone."""

    def __init__(self, hparams: OmegaConf) -> None:  # noqa: D401
        super().__init__(hparams)

        # -------------------------------------------------- hyper-params ----
        fusion_layers: Union[int, Sequence[int]] = hparams.model.fusion_layers
        if isinstance(fusion_layers, int):
            fusion_layers = fuse_layers_mapping[fusion_layers]

        self.fusion_layers: List[int] = fusion_layers
        logger.info("PoseFusionRes3DCNN | fusion at blocks: %s", self.fusion_layers)

        self.ckpt = hparams.model.ckpt_path
        self.model_class_num = hparams.model.model_class_num
        self.model = self.init_resnet(self.model_class_num, self.ckpt)

        # Expose backbone stages as a list for easy indexing
        self.blocks = nn.ModuleList(
            [
                self.model.block[0],  # stem
                self.model.block[1],  # res2
                self.model.block[2],  # res3
                self.model.block[3],  # res4
                self.model.block[4],  # res5
                self.model.block[5],  # global pool + flatten + classifier
            ]
        )

        self.attn_fusions = nn.ModuleList()
        dim_list = [64, 256, 512, 1024, 2048]
        for i, dim in enumerate(dim_list):
            if i in self.fusion_layers:
                fusion = PoseGateFusion(dim, context_channels=1)
                fusion.save_attn = True
                self.attn_fusions.append(fusion)
            else:
                self.attn_fusions.append(nn.Identity())

    # ---------------------------------------------------------------- forward
    def forward(
        self, video: torch.Tensor, attn_map: torch.Tensor
    ) -> torch.Tensor:  # noqa: D401, N802
        """Forward pass.

        Parameters
        ----------
        video : (N, 3, T, H, W)
        attn_map : (N, 1, T, H, W)
        """
        x = video
        for idx in range(5):  # stem + 4 res stages
            x = self.blocks[idx](x)
            if not isinstance(self.attn_fusions[idx], nn.Identity):
                attn_resized = F.interpolate(
                    attn_map, size=x.shape[-3:], mode="trilinear", align_corners=False
                )
                x = self.attn_fusions[idx](x, attn_resized)

        # global pool ➜ flatten ➜ classifier
        x = self.blocks[5](x)
        return x

    # --------------------------------------------------------- visualisation
    def save_attention_maps(self, save_dir: str = "fusion_vis") -> None:  # noqa: D401
        os.makedirs(save_dir, exist_ok=True)
        for idx, fusion in enumerate(self.attn_fusions):
            if isinstance(fusion, PoseGateFusion) and fusion.last_scale is not None:
                # average over batch & time/spatial dims ➜ channel-wise weight
                scale = fusion.last_scale.mean(dim=(0, 2, 3, 4)).cpu().numpy()
                plt.figure(figsize=(12, 3))
                plt.bar(range(len(scale)), scale)
                plt.title(f"Gate Weights – Block {idx}")
                plt.xlabel("Channel")
                plt.ylabel("Weight")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"block{idx}_gate.png"))
                plt.close()


# ---------------------------------------------------------------------------
#  Quick smoke-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":

    cfg = OmegaConf.create(
        {
            "model": {
                "model_class_num": 3,
                "fusion_layers": [0, 2, 4],  # fuse at stem, layer2, layer4
                "ckpt_path": "",  # optional path to pretrained r3d_18 weights
            }
        }
    )

    net = PoseFusionRes3DCNN(cfg)
    rgb = torch.randn(2, 3, 8, 112, 112)  # small input for CI sanity-checks
    pose = torch.randn(2, 1, 8, 112, 112)
    logits = net(rgb, pose)
    print("Output shape:", logits.shape)  # should be (2, 3) for 3 classes
