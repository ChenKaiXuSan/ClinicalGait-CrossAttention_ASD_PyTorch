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
        Path to a local ``.pth`` / ``.pyth`` checkpoint.
        Empty string or *None* → skip loading, keep random init.
    class_num : int
        Number of output classes (classification head will be replaced).

    Returns
    -------
    nn.Module
        Modified SlowR50 instance ready for further customization.
    """
    weight_path = Path(weight_path) if weight_path else None  # type: ignore[assignment]

    model = slow_r50(pretrained=False)

    # Load weights from local file if it exists
    if weight_path is not None and weight_path.exists():
        print(f"[INFO] Loading local weights: {weight_path}")
        state = torch.load(weight_path, map_location="cpu")
        model_state = state.get("model_state", state)
        model.load_state_dict(model_state)
    else:
        print("[INFO] No valid weight path — model will be random.")

    # Modify first layer and classification head
    modify_first_layer(model)
    modify_head(model, class_num)

    return model
