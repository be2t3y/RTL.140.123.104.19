"""下載 timm vit_tiny_patch16_224.augreg_in21k_ft_in1k 並存至 pretrained_models/。

用法（在 python/ 目錄下）：
    python scripts/download_vit_tiny_augreg_pretrained.py
"""
from __future__ import annotations

import os

import torch

OUT_NAME = "vit_tiny_patch16_224_augreg_in21k_ft_in1k.pth"
TIMM_NAME = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"


def main() -> None:
    import timm

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "pretrained_models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, OUT_NAME)

    print(f"Loading timm model: {TIMM_NAME}")
    model = timm.create_model(TIMM_NAME, pretrained=True)
    torch.save(model.state_dict(), out_path)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Saved: {out_path}")
    print(f"  size: {os.path.getsize(out_path) / 1e6:.1f} MB")
    print(f"  embed_dim: {model.embed_dim}, params: {n_params:.2f}M")


if __name__ == "__main__":
    main()
