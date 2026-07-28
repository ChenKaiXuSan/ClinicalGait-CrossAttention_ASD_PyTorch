"""
File: weight_loader.py
Project: project/models
Description: Standalone SlowR50 weight loading and model initialization.

Usage:
    model = init_slow_r50(weight_path, class_num)
"""

from pathlib import Path

import torch
import torch.nn as nn
from pytorchvideo.models.hub.resnet import slow_r50

SLOW_R50_URL = (
    "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/"
    "SLOW_8x8_R50.pyth"
)
X3D_M_URL = (
    "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/"
    "X3D_M.pyth"
)


def modify_first_layer(model: nn.Module) -> None:
    """Replace stem conv with custom kernel/stride/padding."""
    model.blocks[0].conv = nn.Conv3d(
        3,
        model.blocks[0].conv.out_channels,
        kernel_size=(7, 7, 7),
        stride=(1, 2, 2),
        padding=(3, 3, 3),
        bias=False,
    )


def modify_head(model: nn.Module, class_num: int) -> None:
    """Replace classification head with new class_num."""
    model.blocks[-1].proj = nn.Linear(
        model.blocks[-1].proj.in_features, class_num
    )


def init_slow_r50(weight_path: str | None, class_num: int) -> nn.Module:
    """
    Build a SlowR50 model and load pretrained weights if available.

    Parameters
    ----------
    weight_path : str | None
        Path to a local .pth / .pyth checkpoint. If the path is set
        but does not exist, the official Kinetics SlowR50 checkpoint is
        downloaded there. Empty string or *None* skips loading.
    class_num : int
        Number of output classes (classification head will be replaced).

    Returns
    -------
    nn.Module
        Modified SlowR50 instance ready for further customization.
    """
    weight_path = Path(weight_path) if weight_path else None  # type: ignore[assignment]

    model = slow_r50(pretrained=False)

    if weight_path is not None and not weight_path.exists():
        print(f"[INFO] Downloading SlowR50 weights to: {weight_path}")
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(SLOW_R50_URL, str(weight_path), progress=True)

    # Load weights from local file if it exists.
    if weight_path is not None and weight_path.exists():
        print(f"[INFO] Loading local weights: {weight_path}")
        state = torch.load(weight_path, map_location="cpu")
        model_state = state.get("model_state", state)
        model.load_state_dict(model_state)
    else:
        print("[INFO] No valid weight path - model will be random.")

    # Modify first layer and classification head
    modify_first_layer(model)
    modify_head(model, class_num)

    return model


def init_x3d(weight_path: str | None, class_num: int) -> nn.Module:
    """Build an X3D-M backbone (a second, published 3D-CNN family) with the same
    ``.blocks[0..5]`` layout as SlowR50, for the backbone-generality study (#4)
    and as an alternative published-architecture RGB baseline (#5).

    Unlike SlowR50 we keep X3D's native stem (only the classification head is
    replaced). If ``weight_path`` is set but missing, the Kinetics X3D-M
    checkpoint is downloaded there; empty/None → random init.
    """
    from pytorchvideo.models.hub.x3d import x3d_m

    weight_path = Path(weight_path) if weight_path else None  # type: ignore[assignment]
    model = x3d_m(pretrained=False)

    if weight_path is not None and not weight_path.exists():
        print(f"[INFO] Downloading X3D-M weights to: {weight_path}")
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(X3D_M_URL, str(weight_path), progress=True)

    if weight_path is not None and weight_path.exists():
        print(f"[INFO] Loading local X3D-M weights: {weight_path}")
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state.get("model_state", state))
    else:
        print("[INFO] No valid X3D weight path - model will be random.")

    # X3D head is blocks[-1].proj (Linear), same as SlowR50 -> reuse modify_head.
    modify_head(model, class_num)
    return model


def init_backbone(name: str, weight_path: str | None, class_num: int) -> nn.Module:
    """Dispatch to a backbone builder by name ('slow_r50' | 'x3d_m')."""
    name = (name or "slow_r50").lower()
    if name in ("slow_r50", "slowr50", "3dcnn"):
        return init_slow_r50(weight_path, class_num)
    if name in ("x3d_m", "x3d", "x3dm"):
        return init_x3d(weight_path, class_num)
    raise ValueError(f"Unknown backbone_net: {name!r} (use 'slow_r50' or 'x3d_m')")
