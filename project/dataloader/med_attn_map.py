#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/dataloader/med_attn_map.py
Project: /workspace/code/project/dataloader
Created Date: Wednesday April 23rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday April 23rd 2025 6:11:19 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import numpy as np
import os
import torch
from torchvision.utils import save_image

import pandas as pd

COCO_KEYPOINTS = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

region_to_keypoints = {
    "foot": [15, 16],
    "wrist": [9, 10],
    "shoulder": [5, 6],
    "lumbar_pelvis": [11, 12],
    "head": [0, 1, 2, 3, 4],
}


class MedAttnMap:

    def __init__(
        self,
        doctor_res_path: str,
        skeleton_path: str,
    ) -> None:

        self.doctor_res = self.load_doctor_res(doctor_res_path)
        self.skeleton = pd.read_pickle(skeleton_path + "/whole_annotations.pkl")

        # pre-build per-video skeleton cache (avoid O(N) scan each __getitem__)
        self._skeleton_by_video: Dict[str, dict] = {}
        for entry in self.skeleton["annotations"]:
            video_name = entry["frame_dir"].split("/")[-1]
            # Convert numpy arrays to torch.Tensor for convenience
            keypoint_data = torch.tensor(entry["keypoint"]) if isinstance(entry.get("keypoint"), np.ndarray) else entry["keypoint"]
            score_data = torch.tensor(entry["keypoint_score"]) if isinstance(entry.get("keypoint_score"), np.ndarray) else entry["keypoint_score"]
            self._skeleton_by_video[video_name] = {
                "keypoint": keypoint_data,
                "keypoint_score": score_data,
                "total_frames": entry["total_frames"],
            }

    def load_doctor_res(self, docker_res_path: str) -> list[pd.DataFrame]:
        """
        Load the doctor result from the given video path.
        """
        doctor_1 = pd.read_csv(docker_res_path + "/doctor1.csv")
        doctor_2 = pd.read_csv(docker_res_path + "/doctor2.csv")

        return doctor_1, doctor_2

    def find_doctor_res(self, video_name: str) -> list[list[str]]:
        """
        Find the doctor result for the given video path.
        """

        doctor_attn = []
        keypoint_num = []

        for one_doctor in self.doctor_res:
            for idx, row in one_doctor.iterrows():
                if row["video file name"] in video_name:
                    doctor_attn.append(row["attention"][2:-6])
                    for i in region_to_keypoints[row["attention"][2:-6]]:
                        keypoint_num.append(i)

        return set(doctor_attn), set(keypoint_num)

    def find_skeleton(self, video_name: str) -> list[dict[str, Any]]:
        """
        Find the skeleton for the given video path.
        """
        res = []

        # Find the skeleton for the given video path
        for one in self.skeleton["annotations"]:

            # keypoint = one["keypoint"]
            # keypoint_score = one["keypoint_score"]
            # total_frame = one["total_frames"]
            _video_name = one["frame_dir"].split("/")[-1]

            if video_name in _video_name:

                res.append(one)

        return res

    def generate_attention_map(
        self,
        vframes: torch.Tensor,
        mapped_keypoint: list,
        keypoint: torch.Tensor,
        confidence_score,
    ) -> torch.Tensor:
        """Generate attention map for the full video [T, H, W].

        Calls generate_attention_map_chunk with start_frame=0 for backward compatibility.
        """
        return self.generate_attention_map_chunk(
            vframes=vframes,
            mapped_keypoint=mapped_keypoint,
            keypoint=keypoint,
            confidence_score=confidence_score,
            start_frame=0,
        )

    def generate_attention_map_chunk(
        self,
        vframes: torch.Tensor,
        mapped_keypoint: list,
        keypoint: torch.Tensor,
        confidence_score,
        start_frame: int = 0,
    ) -> torch.Tensor:
        """Generate attention map for a chunk of frames [T_chunk, H, W].

        Uses skeleton/keypoints from `start_frame` onwards, avoiding full-video iteration.
        vframes shape: (chunk_size, C, H, W)
        Returns:       (chunk_size, H, W)
        """
        t_chunk = vframes.shape[0]
        h = vframes.shape[-2]
        w = vframes.shape[-1]

        sigma = 0.1 * min(h, w)  # standard deviation for Gaussian kernel

        y_grid, x_grid = torch.meshgrid(
            torch.arange(h), torch.arange(w), indexing="ij"
        )  # shape: [H, W]

        res = []

        for idx in range(t_chunk):
            frame_abs = start_frame + idx

            attn_maps = []

            for i in mapped_keypoint:
                x = keypoint[0, frame_abs, i, 0] * w if (keypoint.size if hasattr(keypoint, "size") else keypoint.numel()) > 0 else -1
                y = keypoint[0, frame_abs, i, 1] * h if (keypoint.size if hasattr(keypoint, "size") else keypoint.numel()) > 0 else -1

                # none keypoint
                if x < 0 or y < 0:
                    attn_maps.append(torch.zeros((h, w)))
                    continue

                dist_squared = (x_grid - x) ** 2 + (y_grid - y) ** 2
                heatmap = torch.exp(-dist_squared / (2 * sigma**2))

                curr_confidence = confidence_score[0, frame_abs, i] if confidence_score is not None else 0.0
                if curr_confidence > 0.8:
                    heatmap *= curr_confidence

                attn_maps.append(heatmap)

            # TODO: 这里可以将不同关键的信息都保存下来
            attn_stack = torch.stack(attn_maps, dim=0) if attn_maps else torch.zeros((0, h, w))
            # Keep per-frame map 2D (H, W) so stacked chunk is (T_chunk, H, W).
            attn_mean = torch.mean(attn_stack, dim=0) if attn_stack.numel() > 0 else torch.zeros((h, w))

            res.append(attn_mean)

        return torch.stack(res, dim=0)  # [T_chunk, H, W]

    def save_attention_map(
        self, attention_map: torch.Tensor, save_path: str, video_name: str
    ) -> None:
        """
        Save the generated attention map to the specified path.
        """
        # Save the attention map
        t, *_ = attention_map.shape

        save_pth = os.path.join(save_path, "attention_map", video_name)
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        for i in range(t):

            save_image(attention_map[i], save_pth + f"/attn_{i}.png", normalize=True)

    def __call__(self, video_path, disease, vframes, video_name) -> torch.Tensor:

        # for one video file
        # * 1 find the doctor result
        doctor_attn, mapped_keypoint = self.find_doctor_res(video_name)

        # * 2 find the skeleton
        # FIXME: 为什么会有两个skeleton被找出来？
        skeleton = self.find_skeleton(video_name)

        # * 3 generate the attention map
        attn_map = self.generate_attention_map(
            vframes,
            mapped_keypoint,
            skeleton[0]["keypoint"],
            confidence_score=skeleton[0]["keypoint_score"],
        )

        # self.save_attention_map(attn_map, "logs", video_name)

        return attn_map
