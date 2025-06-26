#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/dataloader/utils.py
Project: /workspace/code/project/dataloader
Created Date: Wednesday April 23rd 2025
Author: Kaixu Chen
-----
Comment:

Copy from pytorchvideo.

Have a good code time :)
-----
Last Modified: Wednesday June 25th 2025 5:38:56 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from typing import Any, Callable, Dict, Optional


import torch
import torch
import torch.nn.functional as F
from torch import Tensor


# class UniformTemporalSubsample:
#     """Uniformly subsample ``num_samples`` indices from the temporal dimension of the video.

#     Videos are expected to be of shape ``[..., T, C, H, W]`` where ``T`` denotes the temporal dimension.

#     When ``num_samples`` is larger than the size of temporal dimension of the video, it
#     will sample frames based on nearest neighbor interpolation.

#     Args:
#         num_samples (int): The number of equispaced samples to be selected
#     """

#     _transformed_types = (torch.Tensor,)

#     def __init__(self, num_samples: int):
#         super().__init__()
#         self.num_samples = num_samples


#     def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
#         inpt = inpt.permute(1, 0, 2, 3)  # [C, T, H, W] -> [T, C, H, W]
#         return F.uniform_temporal_subsample, inpt, self.num_samples

class UniformTemporalSubsample:

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: Tensor of shape (T, C, H, W)
        returns: Subsampled video of shape (num_samples, C, H, W)
        """
        # Reference: https://github.com/facebookresearch/pytorchvideo/blob/a0a131e/pytorchvideo/transforms/functional.py#L19
        t_max = video.shape[-4] - 1
        indices = torch.linspace(0, t_max, self.num_samples, device=video.device).long()
        return torch.index_select(video, -4, indices)

# class UniformTemporalSubsample:
#     """
#     使用线性插值将输入视频在时间维（T）上统一为固定帧数。
#     支持输入形状为 (T, C, H, W) 或 (B, T, C, H, W)。
#     """

#     def __init__(self, num_samples: int, mode: str = "linear", align_corners: bool = False):
#         self.num_samples = num_samples
#         self.mode = mode
#         self.align_corners = align_corners

#     def __call__(self, video: Tensor) -> Tensor:
#         """
#         Args:
#             video: (T, C, H, W) or (B, T, C, H, W)

#         Returns:
#             video_interp: (num_samples, C, H, W) or (B, num_samples, C, H, W)
#         """
#         if video.ndim == 4:
#             video = video.unsqueeze(0)  # [1, T, C, H, W]
#             squeeze_back = True
#         elif video.ndim == 5:
#             squeeze_back = False
#         else:
#             raise ValueError("Expected 4D or 5D input tensor")

#         # 输入为 [B, T, C, H, W] → 转为 [B, C, T, H, W] 以便 interpolate
#         video = video.permute(0, 2, 1, 3, 4)
#         # 在 temporal 维（即 dim=2）上做插值
#         video_interp = F.interpolate(
#             video,
#             size=self.num_samples,
#             mode=self.mode,
#             align_corners=self.align_corners if self.mode in ["linear", "bilinear", "trilinear"] else None,
#         )
#         # 转回原始形状 [B, T, C, H, W]
#         video_interp = video_interp.permute(0, 2, 1, 3, 4)

#         if squeeze_back:
#             return video_interp.squeeze(0)  # [num_samples, C, H, W]
#         else:
#             return video_interp  # [B, num_samples, C, H, W]

class ApplyTransformToKey:
    """
    Applies transform to key of dictionary input.

    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied

    Example:
        >>>   transforms.ApplyTransformToKey(
        >>>       key='video',
        >>>       transform=UniformTemporalSubsample(num_video_samples),
        >>>   )
    """

    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x[self._key] = self._transform(x[self._key])
        return x


class Div255(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.div_255``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale clip frames from [0, 255] to [0, 1].
        Args:
            x (Tensor): A tensor of the clip's RGB frames with shape:
                (C, T, H, W).
        Returns:
            x (Tensor): Scaled tensor by dividing 255.
        """
        return x / 255.0
