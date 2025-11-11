import os
import math
import logging
from typing import Dict, List, Optional, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------- 基础工具 ----------


def _resize_u8(
    img: np.ndarray,
    size: Optional[Tuple[int, int]] = None,  # (H_out, W_out)
    mode: str = "nearest",
) -> np.ndarray:
    """
    将 uint8 图像放缩到指定尺寸
    """
    if size is None:
        return img
    h, w = img.shape[:2]
    if (h, w) == size:
        return img
    pil = Image.fromarray(img)
    resample = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
    }.get(mode, Image.NEAREST)
    pil = pil.resize((size[1], size[0]), resample=resample)
    return np.asarray(pil)


def _to_uint8_gray(x: torch.Tensor, eps: float = 1e-6) -> np.ndarray:
    """(H,W) -> uint8 灰度"""
    x = x.detach().float().cpu()
    x_min, x_max = x.min(), x.max()
    if (x_max - x_min) < eps:
        x = torch.zeros_like(x)
    else:
        x = (x - x_min) / (x_max - x_min + eps)
    return (x * 255.0).clamp(0, 255).byte().numpy()


def _save_grid_gray(
    images: List[np.ndarray], save_path: str, ncols: int = 8, pad: int = 2
) -> None:
    """灰度网格拼图 (H,W) 列表"""
    if len(images) == 0:
        return
    h, w = images[0].shape[:2]
    n = len(images)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    grid_h = nrows * h + (nrows - 1) * pad
    grid_w = ncols * w + (ncols - 1) * pad
    canvas = np.full((grid_h, grid_w), 0, dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, ncols)
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        canvas[y0 : y0 + h, x0 : x0 + w] = img
    Image.fromarray(canvas, mode="L").save(save_path)


def _colormap_jet(gray_u8: np.ndarray) -> np.ndarray:
    """简易 JET（无额外依赖） -> (H,W,3) uint8"""
    g = gray_u8.astype(np.float32) / 255.0
    c = np.zeros((g.shape[0], g.shape[1], 3), dtype=np.float32)
    c[..., 2] = np.clip(1.5 - np.abs(4 * g - 3), 0, 1)  # R
    c[..., 1] = np.clip(1.5 - np.abs(4 * g - 2), 0, 1)  # G
    c[..., 0] = np.clip(1.5 - np.abs(4 * g - 1), 0, 1)  # B
    return (c * 255.0 + 0.5).astype(np.uint8)


def _to_uint8_img3c(x: torch.Tensor, in_range: str = "auto") -> np.ndarray:
    """(3,H,W) -> (H,W,3) uint8"""
    x = x.detach().float().cpu()
    if in_range == "0,1":
        x = x.clamp(0, 1)
    elif in_range == "-1,1":
        x = (x.clamp(-1, 1) + 1.0) / 2.0
    else:
        x_min, x_max = x.min(), x.max()
        x = (
            torch.zeros_like(x)
            if (x_max - x_min) < 1e-6
            else (x - x_min) / (x_max - x_min + 1e-6)
        )
    return (x * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()


# ---------- 主功能：保存每层特征图 ----------
class _FeatureHook:
    def __init__(self, name: str, module: nn.Module, feats: Dict[str, torch.Tensor]):
        self.name = name
        self.module = module
        self.feats = feats
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, m, inp, out):
        # 只保留 tensor 输出；多值输出时取第一个 tensor
        if isinstance(out, torch.Tensor):
            self.feats[self.name] = out
        elif isinstance(out, (list, tuple)) and len(out) > 0:
            for v in out:
                if isinstance(v, torch.Tensor):
                    self.feats[self.name] = v
                    break

    def remove(self):
        try:
            self.handle.remove()
        except Exception:
            pass


def _list_match(names: Iterable[str], patterns: Iterable[str]) -> bool:
    """任意名字包含任意 pattern 即匹配"""
    ps = list(patterns) if patterns else []
    if not ps:
        return True
    for n in names:
        s = str(n)
        for p in ps:
            if p in s:
                return True
    return False


@torch.no_grad()
def dump_all_feature_maps(
    model: nn.Module,
    video: torch.Tensor,  # (B,3,T,H,W)
    *,
    attn_map: Optional[torch.Tensor] = None,  # (B,1,T,H,W) 若你的前向需要
    save_root: str = "fusion_vis/all_features",
    time_select: str = "mean",  # "mean" | "max" | "t=<int>"
    max_channels: int = 16,
    ncols: int = 8,
    color: bool = True,  # True: 伪彩色；False: 灰度
    overlay: bool = False,  # 把特征图叠到原帧
    overlay_alpha: float = 0.45,
    video_range: str = "auto",  # "auto" | "0,1" | "-1,1"
    include_types: Tuple[type, ...] = (
        nn.Conv3d,
        nn.Conv2d,
        nn.ReLU,
        nn.BatchNorm3d,
        nn.MaxPool3d,
        nn.AvgPool3d,
    ),
    include_name_contains: Tuple[str, ...] = (),  # 名称包含关键字才抓取；空则不过滤
    exclude_name_contains: Tuple[str, ...] = (
        "proj",
    ),  # 名称包含关键字则跳过（默认跳过分类 head 如 "proj"/"head" 可按需改）
    resize_to: Optional[Tuple[int, int]] = None,  # 统一放大到指定尺寸 (H, W)
    resize_mode: str = "nearest",  # 最近邻/双线性/双三次
) -> Dict[str, List[str]]:
    """
    对一个前向过程抓取并保存**每一层**输出特征图。
    - 仅抓取 4D/5D 张量（(B,C,H,W) 或 (B,C,T,H,W)）。
    - 每层按 batch 分别保存到目录：save_root/<layer_name>/b{k}/...
    返回：{ layer_name: [保存的文件路径...] }
    """
    os.makedirs(save_root, exist_ok=True)

    # 1) 注册 hooks
    feats: Dict[str, torch.Tensor] = {}
    hooks: List[_FeatureHook] = []

    for name, mod in model.named_modules():
        if not isinstance(mod, include_types):
            continue
        parts = name.split(".")
        if not _list_match([name] + parts, include_name_contains):
            continue
        if _list_match([name] + parts, exclude_name_contains):
            continue
        hooks.append(_FeatureHook(name, mod, feats))

    if not hooks:
        logger.warning(
            "没有匹配到任何模块，检查 include_types/include_name_contains 过滤条件。"
        )

    # 2) 前向一次抓取
    model_was_training = model.training
    model.eval()
    model = model.to(video.device)

    try:
        _ = model(video, attn_map)
    finally:
        if model_was_training:
            model.train()
        for h in hooks:
            h.remove()

    # 3) 保存每层
    saved: Dict[str, List[str]] = {}
    B = video.size(0)
    _, _, T_video, H_video, W_video = video.shape

    # 选帧函数
    def pick_t(T: int, mode: str) -> Optional[int]:
        if isinstance(mode, str) and mode.startswith("t="):
            try:
                idx = int(mode.split("=", 1)[1])
            except Exception as e:
                raise ValueError(
                    f"Invalid time_select '{mode}', expect 't=<int>'"
                ) from e
            if idx < 0:
                idx = T + idx
            return max(0, min(T - 1, idx))
        return None

    for lname, tensor in feats.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim not in (4, 5):
            # 跳过非 (B,C,H,W)/(B,C,T,H,W)
            continue

        layer_dir = os.path.join(save_root, lname.replace(".", "_"))
        os.makedirs(layer_dir, exist_ok=True)
        saved[layer_dir] = []

        # 统一到 5D: (B,C,T,H,W)
        if tensor.ndim == 4:
            Bx, C, H, W = tensor.shape
            T = 1
            P = tensor.reshape(Bx, C, 1, H, W)
        else:
            Bx, C, T, H, W = tensor.shape
            P = tensor

        # 数值范围到 [0,1]（按通道独立归一化再绘制）
        # 这里不直接归一化 P，逐图像绘制时做归一化可避免跨通道相互影响

        # 时间聚合或单帧
        t_idx = pick_t(T, time_select)
        if t_idx is not None:
            # 单帧 (B,C,H,W)
            maps_4d = P[:, :, max(0, min(T - 1, t_idx))]
        elif time_select == "max":
            maps_4d = P.amax(dim=2)  # (B,C,H,W)
        else:
            maps_4d = P.mean(dim=2)  # (B,C,H,W)

        # 每个样本分别保存
        C_use = min(C, max_channels)
        for b in range(min(B, maps_4d.size(0))):
            subdir = os.path.join(layer_dir, f"b{b}")
            os.makedirs(subdir, exist_ok=True)

            # 收集前 K 个通道的图像（灰度或伪彩）
            imgs_gray: List[np.ndarray] = []
            imgs_color: List[np.ndarray] = []

            for ch in range(C_use):
                g = _to_uint8_gray(maps_4d[b, ch])  # 逐图像归一化
                if resize_to is not None:
                    g = _resize_u8(g, size=resize_to, mode=resize_mode)

                imgs_gray.append(g)
                if color:
                    imgs_color.append(_colormap_jet(g))

            # 保存网格
            if color:
                col = _colormap_jet(g)
                if resize_to is not None:
                    col = _resize_u8(col, size=resize_to, mode=resize_mode)
                imgs_color.append(col)

                # 伪彩色网格：先转灰度网格（复用已有），或自定义彩色网格函数
                # 这里同时保存单张拼图（灰度）和通道彩色样本示例
                grid_gray_path = os.path.join(subdir, "grid_gray.png")
                _save_grid_gray(imgs_gray, grid_gray_path, ncols=ncols, pad=2)
                saved[layer_dir].append(grid_gray_path)

                # 也逐通道保存彩色（可选，避免太多文件可以只保存前几张）
                for i, col in enumerate(imgs_color):
                    p = os.path.join(subdir, f"ch{i:02d}.png")
                    Image.fromarray(col).save(p)
                    saved[layer_dir].append(p)
            else:
                grid_path = os.path.join(subdir, "grid.png")
                _save_grid_gray(imgs_gray, grid_path, ncols=ncols, pad=2)
                saved[layer_dir].append(grid_path)

            # 可选：叠加到原视频帧
            if overlay:
                mid_t = (
                    T_video // 2 if t_idx is None else max(0, min(T_video - 1, t_idx))
                )
                frame = _to_uint8_img3c(video[b, :, mid_t], in_range=video_range)
                # 用“平均后的整层响应”做一张热力叠加
                layer_mean = maps_4d[b, :C_use].mean(dim=0)
                g = _to_uint8_gray(layer_mean)
                heat = _colormap_jet(g)
                over = overlay_alpha * heat.astype(np.float32) + (
                    1 - overlay_alpha
                ) * frame.astype(np.float32)
                over = np.clip(over, 0, 255).astype(np.uint8)
                over_path = os.path.join(subdir, "overlay.png")
                Image.fromarray(over).save(over_path)
                saved[layer_dir].append(over_path)

        logger.info(f"[Features] saved: {layer_dir}")

    return saved
