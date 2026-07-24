#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/dataloader/gait_video_dataset.py
Project: /workspace/code/project/dataloader
Created Date: Tuesday April 22nd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday April 22nd 2025 11:18:09 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

10-06-2026	Kaixu Chen	chunk-based indexing + improvements:
                         - discard incomplete last chunk (only exact chunk_size frames are included)
                         - separate spatial-only transform for attn_map (Div255+Resize, no temporal ops)
                         - per-worker lru_cache on video reads to avoid repeated decode

04-05-2025	Kaixu Chen	load the video as batch, this will save the CPU memory.

23-04-2025	Kaixu Chen	init the code.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from project.dataloader.med_attn_map import MedAttnMap

logger = logging.getLogger(__name__)


def _read_video_with_torchcodec(
    video_path: str, start_frame: int, end_frame: int
) -> Optional[torch.Tensor]:
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError:
        return None

    try:
        decoder = VideoDecoder(video_path, device="cpu")
        frame_batch = decoder.get_frames_at(indices=list(range(start_frame, end_frame)))
    except Exception as exc:
        logger.debug("torchcodec VideoDecoder failed for %s: %s", video_path, exc)
        return None
    return frame_batch.data.contiguous()


def _frame_to_tchw(frame: Any) -> torch.Tensor:
    frame_tensor = torch.from_numpy(frame)
    if frame_tensor.ndim == 2:
        frame_tensor = frame_tensor.unsqueeze(-1).expand(-1, -1, 3)
    if frame_tensor.shape[-1] > 3:
        frame_tensor = frame_tensor[..., :3]
    return frame_tensor.permute(2, 0, 1)


def _read_video_with_cv2(
    video_path: str, start_frame: int, end_frame: int
) -> Optional[torch.Tensor]:
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(end_frame - start_frame):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(_frame_to_tchw(frame))
    finally:
        cap.release()

    if not frames:
        return None
    return torch.stack(frames, dim=0).contiguous()


def _read_video_with_imageio(
    video_path: str, start_frame: int, end_frame: int
) -> Optional[torch.Tensor]:
    try:
        import imageio.v3 as iio
    except ImportError:
        return None

    frames = []
    try:
        for frame_idx, frame in enumerate(iio.imiter(video_path)):
            if frame_idx < start_frame:
                continue
            if frame_idx >= end_frame:
                break
            frames.append(_frame_to_tchw(frame))
    except Exception:
        return None

    if not frames:
        return None
    return torch.stack(frames, dim=0).contiguous()


@lru_cache(maxsize=16)
def _read_video_cached(
    video_path: str, start_frame: int, end_frame: int, fps: int
) -> torch.Tensor:
    """Cache interval-based video reads by (path, start_frame, end_frame, fps).

    Prefer TorchCodec when available, then fall back to OpenCV and imageio.
    torchvision video IO was removed from newer builds, so avoid that API here.
    """
    frames = _read_video_with_torchcodec(video_path, start_frame, end_frame)
    if frames is None:
        frames = _read_video_with_cv2(video_path, start_frame, end_frame)
    if frames is None:
        frames = _read_video_with_imageio(video_path, start_frame, end_frame)
    if frames is None:
        raise RuntimeError(
            "Could not read video frames. Install torchcodec, opencv-python, "
            "or imageio with an FFmpeg-capable backend."
        )
    return frames


class LabeledGaitVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        experiment: str,
        labeled_video_paths: list[Tuple[str, Optional[dict]]],
        transform: Optional[Callable[[dict], Any]] = None,
        attn_transform: Optional[Any] = None,
        doctor_res_path: str = "",
        skeleton_path: str = "",
        chunk_size: int = 8,
        load_attn: bool = True,
    ) -> None:
        super().__init__()

        self._video_transform = (
            transform  # full pipeline: UniformTemporalSubsample + Div255 + Resize
        )
        self._attn_transform = attn_transform  # spatial only: Div255 + Resize
        self._chunk_size = (
            chunk_size  # frames per chunk (== uniform_temporal_subsample_num)
        )
        self._experiment = experiment
        self._shape_debug_logged = False

        # Load doctor-annotated attention maps only when explicitly requested.
        # Previously gated on '"True" in experiment_name', which silently fed
        # ZERO attention maps whenever the experiment tag lacked the substring
        # "True" (e.g. custom tags like "pose_gated_full_f0").
        if load_attn:
            self.attn_map = MedAttnMap(doctor_res_path, skeleton_path)

        # pre-build index_map: one entry per chunk, spanning all videos
        self._index_map = self.prepare_chunk_index(labeled_video_paths)

    def prepare_chunk_index(
        self, labeled_video_paths: list[Tuple[str, Optional[dict]]]
    ) -> List[dict[str, Any]]:
        """Return a flat list of {video_path, start_frame, end_frame, label, disease, video_name}
        for every chunk across all videos.  Only chunks with EXACTLY chunk_size frames are included;
        the last incomplete chunk is discarded."""

        index_map: List[dict[str, Any]] = []

        for json_path in labeled_video_paths:
            with open(json_path) as f:
                info = json.load(f)

            video_path = info["video_path"]
            frame_count = info["frame_count"]

            fps = int(info.get("video_fps", 30))  # store fps for interval reading

            # only keep full chunks (discard remainder frames at end)
            full_chunks = frame_count // self._chunk_size
            for start in range(0, full_chunks * self._chunk_size, self._chunk_size):
                index_map.append(
                    {
                        "video_path": video_path,
                        "video_name": info["video_name"],
                        "start_frame": start,
                        "end_frame": start + self._chunk_size,
                        "label": info["label"],
                        "disease": info["disease"],
                        "fps": fps,  # cache fps so __getitem__ doesn't re-read JSON
                    }
                )

        logger.info(
            f"chunk index prepared: {len(index_map)} chunks from {len(labeled_video_paths)} videos"
        )
        return index_map

    def __len__(self) -> int:
        return len(self._index_map)

    def _apply_video_transform(self, data_tensor: torch.Tensor) -> torch.Tensor:
        """Apply full pipeline (UniformTemporalSubsample + Div255 + Resize) to video frames.

        Args:
            data_tensor: shape (T, C, H, W).

        Returns:
            (C, T', H, W).
        """
        if not self._video_transform:
            logger.warning("no video transform provided")
            return data_tensor
        result = self._video_transform(
            data_tensor
        )  # → (T', C, H, W) with temporal permute
        return result.permute(1, 0, 2, 3)  # → (C, T', H, W) for model input

    def _apply_attn_transform(self, data_tensor: torch.Tensor) -> torch.Tensor:
        """Apply spatial-only transform (Div255 + Resize) to attention maps.

        Attention maps are per-frame heatmaps — no temporal subsampling needed.
        The temporal dimension is preserved so it stays aligned with video frames.

        Args:
            data_tensor: shape (T, H, W).

        Returns:
            (1, T, H', W') — adds channel dimension for model compatibility.
        """
        if not self._attn_transform:
            logger.warning("no attn transform provided")
            return data_tensor
        result = self._attn_transform(data_tensor)  # → (T, H', W')
        return result.unsqueeze(0)  # (1, T, H', W')

    def __getitem__(self, index) -> dict[str, Any]:
        chunk_info = self._index_map[index]
        video_path = chunk_info["video_path"]
        start = chunk_info["start_frame"]
        end = chunk_info["end_frame"]

        # fps is cached in chunk_info from prepare_chunk_index — no disk I/O needed
        fps_int = chunk_info["fps"]

        # load only the specific frame range (cached per-worker to avoid repeated decode)
        vframes: torch.Tensor = _read_video_cached(video_path, start, end, fps_int)

        chunk_frames = vframes[: end - start]  # (chunk_size, C, H, W)

        label = chunk_info["label"]
        disease = chunk_info["disease"]
        video_name = chunk_info["video_name"]

        # attention map — use the new generate_attention_map_chunk to avoid full-video iteration
        if hasattr(self, "attn_map"):
            skeleton = self.attn_map.find_skeleton(video_name)
            doctor_attn, mapped_keypoint = self.attn_map.find_doctor_res(video_name)
            keypoint = skeleton[0]["keypoint"]
            confidence_score = skeleton[0]["keypoint_score"]

            chunk_attn = self.attn_map.generate_attention_map_chunk(
                vframes=chunk_frames,
                mapped_keypoint=mapped_keypoint,
                keypoint=keypoint,
                confidence_score=confidence_score,
                start_frame=start,
            )
        else:
            # fallback: empty attention map — keep (T_chunk, H, W) with proper spatial dims
            h = vframes.shape[-2] if vframes.numel() > 0 else 224
            w = vframes.shape[-1] if vframes.numel() > 0 else 224
            chunk_attn = torch.zeros(self._chunk_size, h, w, dtype=torch.float32)

        transformed_vframes = self._apply_video_transform(
            chunk_frames
        )  # (C, chunk_size, H, W)
        transformed_attn_map = self._apply_attn_transform(
            chunk_attn
        )  # (1, chunk_size, H, W)

        return {
            "video": transformed_vframes,  # [C, chunk_size, H, W]
            "label": label,  # int
            "attn_map": transformed_attn_map,  # [1, chunk_size, H, W]
            "disease": disease,
            "video_name": video_name,
            "video_index": index,
            "start_frame": start,
            "end_frame": end,
        }


def whole_video_dataset(
    experiment: str,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    attn_transform: Optional[Any] = None,
    dataset_idx: list = [],
    doctor_res_path: str = "",
    skeleton_path: str = "",
    clip_duration: int = 1,
    chunk_size: int = 8,
    load_attn: bool = True,
) -> LabeledGaitVideoDataset:
    dataset = LabeledGaitVideoDataset(
        experiment=experiment,
        transform=transform,
        attn_transform=attn_transform,
        labeled_video_paths=dataset_idx,
        doctor_res_path=doctor_res_path,
        skeleton_path=skeleton_path,
        chunk_size=chunk_size,
        load_attn=load_attn,
    )

    return dataset
