"""dim192 CARE ReLU6 定點 test-time 變體（Qm.8）。

各變體沿用對應 ``vit_CARE_relu6_fixed_q**`` 的 ``VisionTransformer`` 前向（位寬已寫死），
僅 factory 維度為 embed_dim=192 / num_heads=3。權重與浮點 ``vit_CARE_relu6_dim192`` 相容。
"""
from __future__ import annotations

from typing import Type

from lib.models.sglatrack.vit_CARE_relu6_dim192 import (
    MLP_RATIO,
    STUDENT_DEPTH,
    STUDENT_EMBED_DIM,
    STUDENT_NUM_HEADS,
    load_vit_tiny_pretrained_dim192,
)
from lib.models.sglatrack.vit_CARE_relu6_fixed_q48 import VisionTransformer as _VT_q48
from lib.models.sglatrack.vit_CARE_relu6_fixed_q58 import VisionTransformer as _VT_q58
from lib.models.sglatrack.vit_CARE_relu6_fixed_q68 import VisionTransformer as _VT_q68
from lib.models.sglatrack.vit_CARE_relu6_fixed_q76 import VisionTransformer as _VT_q76
from lib.models.sglatrack.vit_CARE_relu6_fixed_q77 import VisionTransformer as _VT_q77
from lib.models.sglatrack.vit_CARE_relu6_fixed_q78 import VisionTransformer as _VT_q78
from lib.models.sglatrack.vit_CARE_relu6_fixed_q88 import VisionTransformer as _VT_q88
from lib.models.sglatrack.vit_CARE_relu6_fixed_q98 import VisionTransformer as _VT_q98
from lib.models.sglatrack.vit_CARE_relu6_fixed_q108 import VisionTransformer as _VT_q108
from lib.models.sglatrack.vit_CARE_relu6_fixed_q118 import VisionTransformer as _VT_q118
from lib.models.sglatrack.vit_CARE_relu6_fixed_q128 import VisionTransformer as _VT_q128


def _create_dim192_fixed_q(
    vision_transformer_cls: Type,
    label: str,
    pretrained=False,
    **kwargs,
):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=STUDENT_EMBED_DIM,
        depth=STUDENT_DEPTH,
        num_heads=STUDENT_NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        qkv_bias=True,
    )
    model_kwargs.update(kwargs)
    model = vision_transformer_cls(**model_kwargs)

    if pretrained:
        if isinstance(pretrained, str) and pretrained.endswith(".pth"):
            load_vit_tiny_pretrained_dim192(model, pretrained)
        elif isinstance(pretrained, str) and (
            pretrained.endswith(".pth.tar") or "sglatrack" in pretrained
        ):
            pass
        elif pretrained:
            raise ValueError(
                f"dim192 fixed {label}：pretrained 請為 ViT-Tiny .pth 或 sglatrack checkpoint 路徑"
            )
    return model


def vit_tiny192_care_fixed_q48_patch16_224(pretrained=False, **kwargs):
    return _create_dim192_fixed_q(_VT_q48, "Q4.8", pretrained=pretrained, **kwargs)


def vit_tiny192_care_fixed_q58_patch16_224(pretrained=False, **kwargs):
    return _create_dim192_fixed_q(_VT_q58, "Q5.8", pretrained=pretrained, **kwargs)


def vit_tiny192_care_fixed_q68_patch16_224(pretrained=False, **kwargs):
    return _create_dim192_fixed_q(_VT_q68, "Q6.8", pretrained=pretrained, **kwargs)


def vit_tiny192_care_fixed_q76_patch16_224(pretrained=False, **kwargs):
    return _create_dim192_fixed_q(_VT_q76, "Q7.6", pretrained=pretrained, **kwargs)


def vit_tiny192_care_fixed_q77_patch16_224(pretrained=False, **kwargs):
    return _create_dim192_fixed_q(_VT_q77, "Q7.7", pretrained=pretrained, **kwargs)


def vit_tiny192_care_fixed_q78_patch16_224(pretrained=False, **kwargs):
    return _create_dim192_fixed_q(_VT_q78, "Q7.8", pretrained=pretrained, **kwargs)


def vit_tiny192_care_fixed_q88_patch16_224(pretrained=False, **kwargs):
    return _create_dim192_fixed_q(_VT_q88, "Q8.8", pretrained=pretrained, **kwargs)


def vit_tiny192_care_fixed_q98_patch16_224(pretrained=False, **kwargs):
    return _create_dim192_fixed_q(_VT_q98, "Q9.8", pretrained=pretrained, **kwargs)


def vit_tiny192_care_fixed_q108_patch16_224(pretrained=False, **kwargs):
    return _create_dim192_fixed_q(_VT_q108, "Q10.8", pretrained=pretrained, **kwargs)


def vit_tiny192_care_fixed_q118_patch16_224(pretrained=False, **kwargs):
    return _create_dim192_fixed_q(_VT_q118, "Q11.8", pretrained=pretrained, **kwargs)


def vit_tiny192_care_fixed_q128_patch16_224(pretrained=False, **kwargs):
    return _create_dim192_fixed_q(_VT_q128, "Q12.8", pretrained=pretrained, **kwargs)
