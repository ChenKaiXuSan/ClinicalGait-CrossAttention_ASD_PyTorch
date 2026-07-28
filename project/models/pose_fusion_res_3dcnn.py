#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: pose_fusion_res_3dcnn.py
Author: Kaixu Chen <chenkaixusan@gmail.com>

A 3D-CNN backbone that fuses RGB clips and keypoint-derived attention maps
via a lightweight Pose-Attn Fusion module (channel-wise gated mixing).
It supports:
- Selecting fusion stages (fusion_layers)
- Side heads for per-joint heatmap supervision (optional)
- Saving gate weights and side feature maps for interpretability

Inputs
------
RGB  : (N, 3, T, H, W)
Attn : (N, C_ctx, T, H, W)  # C_ctx=1 or num_joints

Output
------
- logits: (N, num_classes)
- (optional) aux: {"side_preds": List[Tensor], "gate_scales": List[Tensor]}
"""
from __future__ import annotations

import os
import math
import logging
from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from project.models.weight_loader import init_slow_r50, init_backbone

logger = logging.getLogger(__name__)

# -------------------------- Pose-Attention Fusion Block --------------------------
class PoseAttnFusion(nn.Module):
    """
    Channel-wise gated fusion of RGB feature and pose-attention feature.

    out = Norm( RGB_feat * g + Pose_feat * (1 - g) [+ residual x] )
    where g = sigmoid(Conv(ReLU(Norm(Conv([RGB_feat, Pose_feat])))) / temp)

    Notes:
    - Use GroupNorm as default for stability in small-batch settings.
    - gate_init_bias > 0 biases early training toward RGB branch.
    - Automatically interpolates attn spatial-temporal size to x.
    - context_channels can be > 1 (e.g., num_joints). In that case, the gate
      is computed as the mean across joint channels, so gate weight remains
      per-channel but reflects per-joint attention signals independently.

    Args:
        in_channels: RGB feature channel dimension
        context_channels: skeleton/attention map channels (1 = single map, > 1 = per-joint)
        norm: normalization type ("bn" | "gn" | "ln" | "none")
        use_residual: whether to add residual x after fusion
        gate_init_bias: initial bias for gate_conv2 (sigmoid(2.0) ≈ 0.88 → biased to RGB early)
        gate_temp: temperature scaling for gate sigmoid (higher = softer / more uniform gates)
        per_joint_gate: if True, compute gate mean across joint channels when context_channels > 1
    """
    def __init__(
        self,
        in_channels: int,
        context_channels: int = 1,
        norm: str = "gn",           # "bn" | "gn" | "ln" | "none"
        use_residual: bool = True,
        gate_init_bias: float = 2.0,
        gate_temp: float = 1.0,
        per_joint_gate: bool = False,  # P2: support per-joint fusion gates
        chunk_size: int = 8,          # temporal dim from dataloader — used to disambiguate attn dims
        gate_mode: str = "gated",     # "gated" | "add" | "fixed" — ablation of the gating mechanism
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.gate_temp = gate_temp
        self.per_joint_gate = per_joint_gate
        self._chunk_size: int = int(chunk_size)
        assert gate_mode in ("gated", "add", "fixed"), f"bad gate_mode {gate_mode}"
        self.gate_mode = gate_mode

        self.rgb_conv  = nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False)
        self.attn_conv = nn.Conv3d(context_channels, in_channels, 1, bias=False)

        self.norm_rgb  = self._make_norm(norm, in_channels)
        self.norm_attn = self._make_norm(norm, in_channels)
        self.norm_out  = self._make_norm(norm, in_channels) if norm != "none" else nn.Identity()

        self.gate_conv1 = nn.Conv3d(in_channels * 2, in_channels, 1, bias=False)
        self.gate_norm1 = self._make_norm(norm, in_channels)
        self.gate_conv2 = nn.Conv3d(in_channels, in_channels, 1, bias=True)
        nn.init.constant_(self.gate_conv2.bias, gate_init_bias)

        self.act = nn.ReLU(inplace=True)
        self.last_scale: Optional[torch.Tensor] = None          # (N,C,T,H,W) channel-mean gate
        self.last_joint_scales: Optional[torch.Tensor] = None   # (N,C_ctx,C,T,H,W) per-joint gates

    @staticmethod
    def _make_norm(kind: str, c: int) -> nn.Module:
        if kind == "bn":
            return nn.BatchNorm3d(c)
        if kind == "gn":
            # largest group count in {32,16,8,4,2,1} that divides c. For SlowR50
            # widths (all divisible by 32) this is 32 (unchanged); for X3D widths
            # like 48 it falls back to a valid divisor (16) so GroupNorm is legal.
            g = next(gg for gg in (32, 16, 8, 4, 2, 1) if c % gg == 0)
            return nn.GroupNorm(g, c)
        if kind == "ln":
            return nn.GroupNorm(1, c)  # LN-like
        return nn.Identity()

    def forward(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        # Align dtype/device & THW
        attn = attn.to(dtype=x.dtype, device=x.device)

        # Normalize accidental extra singleton dimensions so we always work with
        # (N, C_ctx, T, H, W).
        while attn.dim() > 5:
            squeezed = False
            for dim in range(1, attn.dim() - 3):
                if attn.size(dim) == 1:
                    attn = attn.squeeze(dim)
                    squeezed = True
                    break
            if not squeezed:
                raise ValueError(
                    f"Expected attn to be 5D after squeezing singleton dims, got shape {tuple(attn.shape)}"
                )

        # Ensure attn has 5D (N, C_ctx, T_attn, H, W) for trilinear interpolation.
        # When B == chunk_size pytorchvideo tensor becomes ambiguous — batch and temporal
        # have the same value so F.interpolate can't tell which is spatial. We ensure
        # context_channels > max(B, chunk_size) at data prep time; here we just validate
        # that attn has a distinct enough C_attn to disambiguate from any potential C_out
        # in x.shape[-3:].
        if attn.dim() == 4:
            batch_size: int = int(attn.shape[0])
            context_channels: int = int(attn.shape[1])
            if context_channels == 1 and batch_size <= self._chunk_size:
                # B >= chunk_size (at minimum). Pad so C_attn > max(B, T_x) to ensure
                # no collision with any layer's C_out in x.shape[-3:].
                pad = torch.zeros(batch_size, 1, *attn.shape[2:], device=attn.device, dtype=attn.dtype)
                attn = torch.cat([attn, pad], dim=1)

        if x.shape[-3:] != attn.shape[-3:]:
            Tx, Hx, Wx = x.shape[-3:]  # target spatial-temporal size
            Tattn = attn.size(-3)
            # Handle temporal dimension mismatch.
            # Trilinear interpolation fails when the input temporal dim is 1 (can't interpolate
            # across a single spatial location). We repeat the singleton frame to match Tx,
            # then let trilinear handle the remaining H/W resize.
            if Tattn == 1:
                attn = attn.repeat(1, 1, Tx, 1, 1)
                Tattn = Tx  # update for subsequent check below

            # After handling singleton temporal, interpolate remaining mismatch (H/W or T if both > 1).
            if x.shape[-3:] != attn.shape[-3:]:
                size = list(x.shape[-3:])
                # Only set target T if source T is valid (>1), otherwise keep as-is.
                if Tattn == 1:
                    size[-3] = 1
                elif Tx != 1 and Tattn != Tx:
                    size[-3] = min(Tx, Tattn)  # avoid growing singleton beyond its original range
                attn = F.interpolate(attn, size=size, mode="trilinear", align_corners=False)

        N = x.size(0)

        # Two-stream encode
        rgb_feat  = self.norm_rgb(self.rgb_conv(x))
        attn_feat = self.norm_attn(self.attn_conv(attn))

        # Gate (channel-wise, shared across joint channels)
        g = self.gate_conv1(torch.cat([rgb_feat, attn_feat], dim=1))
        g = self.act(self.gate_norm1(g))
        g = self.gate_conv2(g)
        if self.gate_temp != 1.0:
            g = g / self.gate_temp
        g = torch.sigmoid(g)

        self.last_scale = g.detach()

        # Optionally also store per-joint gates for interpretability
        if self.per_joint_gate and attn.size(1) > 1:
            # Expand attn_feat to (N, C_ctx, C_in, T, H, W), then gate each joint
            attn_expanded = attn_feat.unsqueeze(1).expand(-1, N, -1, -1, -1, -1)
            joint_gates_list = []
            for j in range(attn.size(1)):  # per-joint loop
                joint_attn = self.norm_attn(
                    self.attn_conv(attn[:, j:j+1])
                ).unsqueeze(1).expand(-1, N, -1, -1, -1, -1)
                g_joint = self.gate_conv1(torch.cat([rgb_feat.unsqueeze(1), joint_attn], dim=2).squeeze(1))
                g_joint = self.act(self.gate_norm1(g_joint))
                g_joint = self.gate_conv2(g_joint)
                if self.gate_temp != 1.0:
                    g_joint = g_joint / self.gate_temp
                joint_gates_list.append(torch.sigmoid(g_joint).unsqueeze(1))
            self.last_joint_scales = torch.cat(joint_gates_list, dim=1).detach()

        # Fuse (+ optional residual). gate_mode ablates the LEARNED gate:
        #   gated : learned channel-wise gate g (default, the proposed mechanism)
        #   add   : plain additive injection of the projected prior (no gate)
        #   fixed : frozen equal mix (gate hard-set to 0.5) — isolates "shallow
        #           injection" from "learned gating" at the SAME layers/inputs.
        if self.gate_mode == "add":
            fused = rgb_feat + attn_feat
        elif self.gate_mode == "fixed":
            fused = 0.5 * rgb_feat + 0.5 * attn_feat
        else:
            fused = rgb_feat * g + attn_feat * (1.0 - g)
        out = fused + x if self.use_residual else fused
        out = self.norm_out(out)
        return out


# -------------------------- Fusion Config Mapping ----------------------------
FUSE_LAYERS_MAPPING = {
    "single": {i: [i] for i in range(5)},
    "multi":  {
        0: [0],
        1: [0, 1],
        2: [0, 1, 2],
        3: [0, 1, 2, 3],
        4: [0, 1, 2, 3, 4],
    },
}

# blocks[0]: stem          →  64ch
# blocks[1]: layer1 (x3)   → 256ch
# blocks[2]: layer2 (x4)   → 512ch
# blocks[3]: layer3 (x6)   → 1024ch
# blocks[4]: layer4 (x3)   → 2048ch
# blocks[5]: head (GAP+FC) → logits
DIM_LIST = [64, 256, 512, 1024, 2048]


# ---------------------------- Skeleton-Aware Side Head -------------------------
class BoneSegmentConv(nn.Module):
    """Convolution that respects bone segment topology (skeleton-aware)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 3x3x1 kernel along the spatial dimensions preserves temporal context
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (N, C, T, H, W) → conv per frame → concat across frames
        N, C, T = feat.size(0), feat.size(1), feat.size(2)
        out_list = []
        for t in range(T):
            out_list.append(self.conv(feat[:, :, t]))  # (N, out_ch, H, W)
        return torch.stack(out_list, dim=2)  # (N, out_ch, T, H, W)


class SkeletonAwareSideHead(nn.Module):
    """
    Side head that fuses bone-segment features with per-joint Conv3d.

    - For each joint: apply a 1x1x1 Conv3d to get per-joint heatmap logits
    - Fuse: concatenate bone-segment (spatial structure) + joint-specific features
      → produces per-joint attention maps that respect skeleton topology.

    Args:
        in_channels: input feature channel dimension
        num_joints: number of keypoints (context_channels)
    """

    def __init__(self, in_channels: int, context_channels: int = 12) -> None:
        super().__init__()
        self.num_joints = context_channels
        # Bone-segment pathway: spatial structure encoding
        self.bone_conv = BoneSegmentConv(in_channels // 2, in_channels // 2)
        # Joint-specific pathway: standard Conv3d per joint
        self.joint_conv = nn.Conv3d(in_channels, context_channels, kernel_size=1)
        # Fusion: concatenate bone + joint features → reduce back to ctx_ch
        self.fusion = nn.Conv3d(in_channels, context_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, H, W) feature map from backbone block
        Returns:
            side_pred: (N, num_joints, T, H, W) per-joint heatmap logits
        """
        B, C, T, H, W = x.shape
        # Split: half for bone-segment, half for joint features
        x_bone, x_joint = x[:, : C // 2], x[:, C // 2:]

        # Bone-segment pathway: capture spatial relationships between joints
        x_bone_fused = self.bone_conv(x_bone)  # (B, C//2, T, H, W)

        # Joint-specific pathway
        x_joint_logits = self.joint_conv(x_joint)  # (B, num_joints, T, H, W)

        # Fuse: concatenate and reduce back to context channels
        fused = torch.cat([x_bone_fused.expand(-1, C // 2, -1, -1, -1), x_joint_logits], dim=1)
        side_pred = self.fusion(fused)  # (B, num_joints, T, H, W)

        return side_pred


# ---------------------------- Helpers: image saving --------------------------
def _to_uint8(x: torch.Tensor, eps: float = 1e-6) -> np.ndarray:
    x = x.detach().float().cpu()
    x_min, x_max = x.min(), x.max()
    if (x_max - x_min) < eps:
        x = torch.zeros_like(x)
    else:
        x = (x - x_min) / (x_max - x_min + eps)
    x = (x * 255.0).clamp(0, 255).byte().numpy()
    return x


def _save_grid(images: List[np.ndarray], save_path: str, ncols: int = 4, pad: int = 2) -> None:
    if len(images) == 0:
        return
    h, w = images[0].shape[:2]
    n = len(images)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    grid_h = nrows * h + (nrows - 1) * pad
    grid_w = ncols * w + (ncols - 1) * pad
    canvas = np.full((grid_h, grid_w), 0, dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, ncols)
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        canvas[y0:y0 + h, x0:x0 + w] = img
    Image.fromarray(canvas, mode="L").save(save_path)


# ---------------------------- Fusion Config Mapping ----------------------------
FUSE_LAYERS_MAPPING = {
    "single": {i: [i] for i in range(5)},
    "multi":  {
        0: [0],
        1: [0, 1],
        2: [0, 1, 2],
        3: [0, 1, 2, 3],
        4: [0, 1, 2, 3, 4],
    },
}

# blocks[0]: stem          →  64ch
# blocks[1]: layer1 (x3)   → 256ch
# blocks[2]: layer2 (x4)   → 512ch
# blocks[3]: layer3 (x6)   → 1024ch
# blocks[4]: layer4 (x3)   → 2048ch
# blocks[5]: head (GAP+FC) → logits
DIM_LIST = [64, 256, 512, 1024, 2048]


# ---------------------------- Main Model Class -------------------------------
class PoseFusionRes3DCNN(nn.Module):
    def __init__(self, hparams: OmegaConf) -> None:
        super().__init__()

        m = hparams.model
        ablation = m.get("ablation_study", "multi")
        fusion_layers = m.fusion_layers
        if isinstance(fusion_layers, int):
            # Special value 5 = all layers [0..4], regardless of ablation mode
            if fusion_layers == 5:
                fusion_layers = [0, 1, 2, 3, 4]
            else:
                fusion_layers = FUSE_LAYERS_MAPPING[ablation].get(fusion_layers, [])
        self.fusion_layers: List[int] = list(fusion_layers)
        logger.info(f"Fusion at blocks (0=stem..4=layer4): {self.fusion_layers}")

        self.num_classes = int(m.model_class_num)
        self.ckpt = m.get("ckpt_path", "None")
        self.attn_channels = int(m.get("attn_channels", 1))
        self.use_side = bool(m.get("use_side_heads", False))

        # P2: per-joint gate (context_channels > 1) → each joint has independent gate
        self.per_joint_gate = bool(m.get("per_joint_gate", False))
        self.skeleton_topology_aware = bool(m.get("skeleton_topology_aware", False))

        # Chunk size from dataloader (uniform_temporal_subsample_num).
        # We need this to detect collisions between batch/temporal and C_out in fusion layers.
        # pytorchvideo returns inherently ambiguous tensors (N, C, T, H, W) — when B == chunk_size,
        # batch and temporal both have value V. F.interpolate resizes all spatial dims uniformly so
        # we MUST ensure chunk_size != any potential C_out in fusion layers.
        chunk_size = getattr(m, "uniform_temporal_subsample_num", None)
        self._chunk_size = int(chunk_size) if chunk_size is not None \
            else getattr(hparams.train, "uniform_temporal_subsample_num", 8)

        # Build backbone (Kinetics-pretrained). backbone_net selects the family
        #   slow_r50 (default, proven) | x3d_m  — see weight_loader.init_backbone.
        # Per-stage channel dims: hardcoded DIM_LIST for slow_r50 (unchanged path),
        # inferred by a dummy forward for any other backbone (#4 backbone-agnostic).
        self.backbone_net = str(m.get("backbone_net", "slow_r50"))
        self.model = init_backbone(self.backbone_net, self.ckpt, self.num_classes)
        n_blocks = len(self.model.blocks)
        self.blocks = nn.ModuleList([self.model.blocks[i] for i in range(n_blocks)])
        self.dim_list = (DIM_LIST
                         if self.backbone_net.lower() in ("slow_r50", "slowr50", "3dcnn")
                         else self._infer_stage_dims())

        # Guard: chunk_size must not collide with any stage channel width (the
        # (N,C,T,H,W) ambiguity when B == chunk_size == C_out). See notes above.
        colliding = [d for d in set(self.dim_list) if d == self._chunk_size]
        if colliding:
            raise ValueError(
                f"Collision: uniform_temporal_subsample_num={self._chunk_size} equals a "
                f"stage width {colliding}. Set it to a value not in {sorted(set(self.dim_list))}."
            )

        # Fusion modules per stage
        self.attn_fusions = nn.ModuleList([
            PoseAttnFusion(
                in_channels=dim,
                context_channels=self.attn_channels,
                norm=m.get("fusion_norm", "gn"),
                use_residual=bool(m.get("fusion_residual", True)),
                gate_init_bias=float(m.get("gate_init_bias", 2.0)),
                gate_temp=float(m.get("gate_temp", 1.0)),
                per_joint_gate=self.per_joint_gate,  # P2
                chunk_size=self._chunk_size,
                gate_mode=str(m.get("gate_mode", "gated")),  # #6 ablation
            ) if i in self.fusion_layers else nn.Identity()
            for i, dim in enumerate(self.dim_list)
        ])

        # Side heads for per-joint (or 1ch) maps at chosen stages
        # P2: skeleton_topology_aware → use bone-segment convolution instead of plain Conv3d
        self.side_heads = nn.ModuleList()
        for i, dim in enumerate(self.dim_list):
            if self.use_side and i in {1,2,3,4}:
                if self.skeleton_topology_aware:
                    # Skeleton-aware side head: bone segments conv + concat
                    self.side_heads.append(SkeletonAwareSideHead(dim, self.attn_channels))
                else:
                    self.side_heads.append(nn.Conv3d(dim, self.attn_channels, kernel_size=1))
            else:
                self.side_heads.append(nn.Identity())

    # ------------------------ backbone-agnostic dims -------------------------
    @torch.no_grad()
    def _infer_stage_dims(self) -> List[int]:
        """Per-stage output channel width for stages 0..4, by a dummy forward
        through the backbone blocks. Lets fusion/side-heads adapt to a non-
        SlowR50 backbone (e.g. X3D) without hardcoded DIM_LIST."""
        was_training = self.model.training
        self.model.eval()
        x = torch.zeros(1, 3, self._chunk_size, 224, 224)
        dims: List[int] = []
        for i in range(5):                       # stem + layer1..4 (blocks 0..4)
            x = self.blocks[i](x)
            dims.append(int(x.shape[1]))
        if was_training:
            self.model.train()
        return dims

    # ---------------------------- Forward ------------------------------------
    def forward(
        self,
        video: torch.Tensor,
        attn_map: torch.Tensor,
        return_aux: bool = False
    ):
        """
        video   : (N,3,T,H,W)
        attn_map: (N,C_ctx,T,H,W)
        """
        aux: Optional[Dict[str, List[torch.Tensor]]] = {"side_preds": [], "gate_scales": []} if (return_aux or self.use_side) else None

        x = video
        for idx in range(5):  # 0..4 stages
            # First run the corresponding backbone stage so x channel dims match
            # the fusion module configured for this stage.
            x = self.blocks[idx](x)

            # side prediction logits
            if (return_aux or self.use_side) and not isinstance(self.side_heads[idx], nn.Identity):
                side_pred = self.side_heads[idx](x)  # (N, C_ctx, Ti, Hi, Wi)
                aux["side_preds"].append(side_pred)

            # fusion
            if not isinstance(self.attn_fusions[idx], nn.Identity):
                # ensure THW alignment is done inside fusion
                x = self.attn_fusions[idx](x, attn_map)
                if (return_aux or self.use_side) and hasattr(self.attn_fusions[idx], "last_scale") and self.attn_fusions[idx].last_scale is not None:
                    # (N, C_i) channel-mean gate weights for logging
                    g = self.attn_fusions[idx].last_scale.mean(dim=(2,3,4)).detach()
                    aux["gate_scales"].append(g)

        # head -> logits
        logits = self.blocks[5](x)  # expected to return (N, num_classes)

        return (logits, aux) if (return_aux or self.use_side) else logits

    # ---------------------------- Visualization ------------------------------
    def get_gate_scales(self) -> List[Optional[torch.Tensor]]:
        """Return per-stage channel-mean gate scales (CPU) if available."""
        out: List[Optional[torch.Tensor]] = []
        for fusion in self.attn_fusions:
            if isinstance(fusion, PoseAttnFusion) and fusion.last_scale is not None:
                out.append(fusion.last_scale.mean(dim=(0,2,3,4)).detach().cpu())  # (C,)
            else:
                out.append(None)
        return out

    def save_attention_maps(self, save_dir: str = "fusion_vis") -> None:
        """
        Save bar charts of channel-mean gate weights per fused stage.
        """
        os.makedirs(save_dir, exist_ok=True)
        for idx, scale in enumerate(self.get_gate_scales()):
            if scale is None:
                continue
            arr = scale.numpy()
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 3))
            plt.bar(range(len(arr)), arr)
            plt.title(f"Gate Weights – Block {idx}")
            plt.xlabel("Channel")
            plt.ylabel("Weight")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"block{idx}_gate.png"))
            plt.close()
        logger.info(f"Gate weight figures saved to: {save_dir}")

    def save_side_feature_maps(
        self,
        side_preds: List[torch.Tensor],
        save_dir: str = "fusion_vis/side_maps",
        aggregate: str = "mean",     # "mean" | "max" | "t=<int>"
        max_channels: int = 16,
        ncols: int = 4
    ) -> None:
        """
        Convert side head 3D logits (B,C,T,H,W) to 2D grids and save as PNG.
        """
        os.makedirs(save_dir, exist_ok=True)

        # parse aggregate option
        t_idx = None
        if aggregate.startswith("t="):
            try:
                t_idx = int(aggregate.split("=", 1)[1])
            except Exception as e:
                raise ValueError(f"Invalid aggregate '{aggregate}', expect 't=<int>'") from e

        for li, P in enumerate(side_preds):
            P = torch.sigmoid(P)       # visualize probabilities
            B, C, T, H, W = P.shape
            layer_dir = os.path.join(save_dir, f"layer{li}")
            os.makedirs(layer_dir, exist_ok=True)

            for b in range(B):
                if t_idx is not None:
                    t_sel = t_idx if t_idx >= 0 else (T + t_idx)
                    t_sel = max(0, min(T - 1, t_sel))
                    M = P[b, :, t_sel]           # (C,H,W)
                elif aggregate == "max":
                    M = P[b].amax(dim=1)         # (C,H,W)
                else:
                    M = P[b].mean(dim=1)         # (C,H,W)

                C_use = min(C, max_channels)
                imgs = [_to_uint8(M[ch]) for ch in range(C_use)]
                save_path = os.path.join(layer_dir, f"b{b}_grid.png")
                _save_grid(imgs, save_path, ncols=ncols, pad=2)

        logger.info(f"Side feature maps saved to: {save_dir}")


# ---------------------------- Quick Test Entry -------------------------------
if __name__ == "__main__":
    cfg = OmegaConf.create({
        "model": {
            "model_class_num": 3,
            "fusion_layers": [2, 3, 4],     # fuse at layer2..4
            "ckpt_path": "",
            "ablation_study": "multi",
            "attn_channels": 1,             # or J
            "use_side_heads": True,
            "fusion_norm": "gn",
            "fusion_residual": True,
            "gate_init_bias": 2.0,
            "gate_temp": 1.0,
        }
    })
    model = PoseFusionRes3DCNN(cfg)
    rgb  = torch.randn(2, 3, 8, 224, 224)
    attn = torch.randn(2, 1, 8, 224, 224)

    logits, aux = model(rgb, attn, return_aux=True)
    print("logits:", logits.shape)               # (2, 3)
    print("side heads:", len(aux["side_preds"])) # == #stages with side heads
    model.save_attention_maps("test_fusion_vis/gates")
    model.save_side_feature_maps(aux["side_preds"], save_dir="test_fusion_vis/side", aggregate="mean")
