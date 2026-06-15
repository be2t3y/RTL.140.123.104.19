"""ViT CARE ReLU6 浮點 backbone、embed_dim=32（無 Q8.8 to_fixed_point）。

預訓練載入邏輯與 vit_CARE_relu6_dim32 相同（192 / 768 等 → 32 維投影）。
"""
from lib.models.sglatrack.vit_CARE_relu6_dim32 import (
    _create_vision_transformer_dim32,
    vit_tiny32_care_patch16_224,
)


def vit_care_relu6_shared_trunk_dim32_base_patch16_224(pretrained=False, **kwargs):
    """CARE ReLU6 浮點，embed_dim=32；供 vit_mae_teacher_shared_trunk_32dim 等 yaml 的 BACKBONE.TYPE。"""
    return _create_vision_transformer_dim32(pretrained=pretrained, **kwargs)


# 舊鍵名相容（仍為同一浮點 backbone，不含 fixed-point）
vit_care_relu6_fixed_shared_trunk_dim32_base_patch16_224 = (
    vit_care_relu6_shared_trunk_dim32_base_patch16_224
)
