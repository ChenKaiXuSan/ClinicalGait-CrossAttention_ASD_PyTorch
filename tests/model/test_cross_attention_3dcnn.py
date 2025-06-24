import pytest
import torch
from omegaconf import OmegaConf

from project.models.cross_attn_res_3dcnn import CrossAttentionRes3DCNN


@pytest.mark.parametrize(
    "fusion_layers",
    [
        [],  # 不启用任何 cross-attention
        [0],  # 仅第一层
        [0, 1],  # 第一和第三层
        [0, 1, 2],  # 前三层融合
        [0, 1, 2, 3],  # 前四层融合
        [0, 1, 2, 3, 4],  # 所有层均融合
    ],
)
def test_cross_attention_layers(fusion_layers):
    # 构建超参数配置
    hparams = OmegaConf.create(
        {"model": {"model_class_num": 3, "fusion_layers": fusion_layers}}
    )

    # 构建模型
    model = CrossAttentionRes3DCNN(hparams)

    # 构造随机输入
    B, C, T, H, W = 2, 3, 8, 224, 224
    video = torch.randn(B, C, T, H, W)
    attn_map = torch.randn(B, 1, T, H, W)

    # 前向传播
    output = model(video, attn_map)

    # 验证输出类型与形状
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == B
    assert output.shape[1] == hparams.model.model_class_num
