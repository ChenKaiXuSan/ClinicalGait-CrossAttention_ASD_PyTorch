import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

import logging

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_q, dim_kv, dim_out, name=None):
        super().__init__()
        self.query_proj = nn.Conv3d(dim_q, dim_out, kernel_size=1)
        self.key_proj = nn.Conv3d(dim_kv, dim_out, kernel_size=1)
        self.value_proj = nn.Conv3d(dim_kv, dim_out, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.name = name or "unnamed"
        self.save_attn = False
        self.attn_map = None

    def forward(self, x, context):
        B, C, T, H, W = x.shape
        q = self.query_proj(x).flatten(2).transpose(1, 2)  # [B, THW, C]
        k = self.key_proj(context).flatten(2)  # [B, C, THW]
        v = self.value_proj(context).flatten(2).transpose(1, 2)  # [B, THW, C]
        attn = torch.bmm(q, k) / (k.shape[1] ** 0.5)  # [B, THW, THW]
        attn = self.softmax(attn)

        if self.save_attn:
            self.attn_map = attn.detach().cpu()

        out = torch.bmm(attn, v).transpose(1, 2).view(B, -1, T, H, W)
        return out + x


class CrossAttentionRes3DCNN(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.model_class_num = hparams.model.model_class_num
        self.fusion_layers = hparams.model.fusion_layers  # e.g., [0, 2, 4]
        logger.info(
            f"Using CrossAttentionRes3DCNN with fusion layers: {self.fusion_layers}"
        )
        self.model = self.init_resnet(self.model_class_num)

        self.attn_fusions = nn.ModuleList()
        dim_list = [64, 256, 512, 1024, 2048]
        for i, dim in enumerate(dim_list):
            if i in self.fusion_layers:
                fusion = CrossAttentionFusion(dim, 1, dim, name=f"res{i+1}")
                fusion.save_attn = True  # 如果需要可切换
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

    def save_attention_maps(self, save_dir="attn_vis"):
        os.makedirs(save_dir, exist_ok=True)
        for i, fusion in enumerate(self.attn_fusions):
            if isinstance(fusion, CrossAttentionFusion) and fusion.attn_map is not None:
                B, N, _ = fusion.attn_map.shape
                for b in range(min(1, B)):
                    plt.imshow(fusion.attn_map[b].mean(0).view(int(N**0.5), -1))
                    plt.colorbar()
                    plt.title(f"{fusion.name}_sample{b}_attn")
                    plt.savefig(f"{save_dir}/layer{i}_sample{b}.png")
                    plt.close()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    hparams = OmegaConf.create(
        {
            "model": {
                "model_class_num": 3,
                "fusion_layers": [0, 1, 2, 3, 4],  # 控制哪些层启用 cross attention
            }
        }
    )
    model = CrossAttentionRes3DCNN(hparams)
    video = torch.randn(2, 3, 8, 224, 224)  # [B, C, T, H, W]
    attn_map = torch.randn(2, 1, 8, 224, 224)  # [B, C, T, H, W]
    output = model(video, attn_map)
    print(output.shape)  # 应该是 [B, C, T, H, W]，即 [2, 3, 16, 112, 112]
