"""Shared-trunk CenterPredictor：兩層 3x3（in→96→48）+ 三分支 1x1 tail。

訓練（yaml 無 FIXED_INT_BITS）：浮點 forward。
Test-time 定點（yaml 設 FIXED_INT_BITS / FIXED_FRAC_BITS）：在 tail 輸出、sigmoid 後與 bbox 插入
``to_fixed_point``，節點對齊 ``head_shared_trunk_dump.py``。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lib.models.layers.head import conv
from lib.module import to_fixed_point

SHARED_CH1 = 96
SHARED_CH2 = 48


class CenterPredictorFixedSharedTrunk(nn.Module):
    """Shared trunk + per-branch 1x1 tails，與 CenterPredictor 相同 forward 介面。"""

    def __init__(
        self,
        inplanes=64,
        channel=256,
        feat_sz=20,
        stride=16,
        freeze_bn=False,
        int_bits: int = 8,
        frac_bits: int = 8,
        enable_fixed_quant: bool = False,
    ):
        super().__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.enable_fixed_quant = bool(enable_fixed_quant)
        self.shared_conv1 = conv(
            inplanes, SHARED_CH1, kernel_size=3, stride=1, padding=1, freeze_bn=freeze_bn
        )
        self.shared_conv2 = conv(
            SHARED_CH1, SHARED_CH2, kernel_size=3, stride=1, padding=1, freeze_bn=freeze_bn
        )

        self.tail_ctr = nn.Conv2d(SHARED_CH2, 1, kernel_size=1)
        self.tail_size = nn.Conv2d(SHARED_CH2, 2, kernel_size=1)
        self.tail_offset = nn.Conv2d(SHARED_CH2, 2, kernel_size=1)

        for m in (self.tail_ctr, self.tail_size, self.tail_offset):
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        return to_fixed_point(x, self.int_bits, self.frac_bits)

    def forward(self, x, gt_score_map=None):
        score_map_ctr, size_map, offset_map = self.get_score_map(x)
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)
        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        bbox = torch.cat(
            [
                (idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                size.squeeze(-1),
            ],
            dim=1,
        )
        if self.enable_fixed_quant:
            bbox = self._q(bbox)

        if return_score:
            return bbox, max_score
        return bbox

    def get_score_map(self, x):
        if not self.enable_fixed_quant:
            def _sigmoid(t):
                return torch.clamp(t.sigmoid_(), min=1e-4, max=1 - 1e-4)

            trunk = self.shared_conv2(self.shared_conv1(x))
            score_map_ctr = self.tail_ctr(trunk)
            score_map_size = self.tail_size(trunk)
            score_map_offset = self.tail_offset(trunk)
            return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset

        def _sigmoid_q(t: torch.Tensor) -> torch.Tensor:
            y = torch.clamp(t.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return self._q(y)

        trunk = self.shared_conv2(self.shared_conv1(x))
        q_ctr = self._q(self.tail_ctr(trunk))
        q_size = self._q(self.tail_size(trunk))
        q_off = self._q(self.tail_offset(trunk))
        return _sigmoid_q(q_ctr.clone()), _sigmoid_q(q_size.clone()), q_off
