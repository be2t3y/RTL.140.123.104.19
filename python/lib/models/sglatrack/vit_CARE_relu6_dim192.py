"""ViT CARE ReLU6 變體：embed_dim=192（ViT-Tiny），直接載入 timm ViT-Tiny 預訓練（不壓維投影）。

預設對齊 timm `vit_tiny_patch16_224.augreg_in21k_ft_in1k`（192 / 12 layers / 3 heads）。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from lib.models.sglatrack.vit_CARE_relu6 import VisionTransformer, resize_pos_embed


STUDENT_EMBED_DIM = 192
STUDENT_NUM_HEADS = 3
STUDENT_DEPTH = 12
MLP_RATIO = 4


def _unwrap_checkpoint(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """支援裸 state_dict、{'state_dict':...}、{'model':...}。"""
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            sd = checkpoint["model"]
        else:
            sd = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        key = k
        if key.startswith("module."):
            key = key[len("module.") :]
        out[key] = v
    return out


def _align_pretrained_to_student(
    teacher_sd: Dict[str, torch.Tensor],
    model: VisionTransformer,
) -> Dict[str, torch.Tensor]:
    """僅載入與 student 鍵名、shape 一致之權重；pos_embed 尺寸不同時做 resize。"""
    student_sd = model.state_dict()
    out: Dict[str, torch.Tensor] = {}

    for k, vs in student_sd.items():
        if k not in teacher_sd:
            continue
        vt = teacher_sd[k]
        if k == "pos_embed" and vt.shape != vs.shape:
            vt = resize_pos_embed(
                vt,
                model.pos_embed,
                getattr(model, "num_tokens", 1),
                model.patch_embed.grid_size,
            )
        if vt.shape != vs.shape:
            continue
        out[k] = vt.clone()

    return out


def load_vit_tiny_pretrained_dim192(model: VisionTransformer, checkpoint_path: str) -> Tuple[list, list]:
    """從 timm ViT-Tiny .pth 直接載入 192 維 CARE backbone（strict=False，不投影）。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    teacher_sd = _unwrap_checkpoint(checkpoint)
    aligned = _align_pretrained_to_student(teacher_sd, model)
    incomp = model.load_state_dict(aligned, strict=False)
    missing = getattr(incomp, "missing_keys", incomp[0])
    unexpected = getattr(incomp, "unexpected_keys", incomp[1])
    print(
        f"[vit_CARE_relu6_dim192] Loaded direct pretrained from: {checkpoint_path} "
        f"(embed_dim={STUDENT_EMBED_DIM}, aligned keys={len(aligned)})"
    )
    print(f"  load_state_dict strict=False -> missing: {len(missing)} keys, unexpected: {len(unexpected)} keys")
    if missing:
        print(f"  missing (first 20): {missing[:20]}")
    if unexpected:
        print(f"  unexpected (first 20): {unexpected[:20]}")
    return missing, unexpected


def _create_vision_transformer_dim192(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=STUDENT_EMBED_DIM,
        depth=STUDENT_DEPTH,
        num_heads=STUDENT_NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        qkv_bias=True,
    )
    model_kwargs.update(kwargs)
    model = VisionTransformer(**model_kwargs)

    if pretrained:
        if isinstance(pretrained, str) and pretrained.endswith(".pth"):
            load_vit_tiny_pretrained_dim192(model, pretrained)
        else:
            raise ValueError(
                "dim192 預訓練請傳入 ViT-Tiny .pth 路徑"
                "（例如 pretrained_models/vit_tiny_patch16_224_augreg_in21k_ft_in1k.pth）"
            )
    return model


def vit_tiny192_care_patch16_224(pretrained=False, **kwargs):
    """
    CARE ReLU6，embed_dim=192、depth=12、num_heads=3。
    若 `pretrained` 為字串路徑，則從 timm ViT-Tiny checkpoint 直接載入（不壓維）。
    """
    return _create_vision_transformer_dim192(pretrained=pretrained, **kwargs)
