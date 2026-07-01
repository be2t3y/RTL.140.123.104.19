"""Sanity check: 用「預測框當作 GT」重算 benchmark，驗證評估流程是否正常。

用途（課堂／除錯實驗，非正式報告數字）：
  - 若「pred vs 真 GT」AUC 很低，但「pred vs pred（假 GT）」AUC ≈ 100%，
    代表 test → results → metrics 管線正常，問題在模型表現。
  - 若兩者都很低，可能是結果檔路徑、序列命名、幀數對齊等 pipeline bug。

不會修改 data/ 下任何標註檔。僅在記憶體內把 anno 換成 pred 做對照。

Usage:
  cd python
  python tracking/sanity_check_pred_as_gt.py \\
    --tracker sglatrack \\
    --param vit_coco_got10k_distill_mae_teacher_orr_afkd_s60000_bs32 \\
    --dataset uav123
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

prj_path = os.path.join(os.path.dirname(__file__), "..")
if prj_path not in sys.path:
    sys.path.append(prj_path)

import _init_paths  # noqa: E402,F401

from lib.test.analysis.extract_results import calc_seq_err_robust  # noqa: E402
from lib.test.analysis.plot_results import get_auc_curve, get_prec_curve  # noqa: E402
from lib.test.evaluation import get_dataset, trackerlist  # noqa: E402
from lib.test.utils.load_text import load_text  # noqa: E402


def _auc_prec_from_curves(eval_overlap, eval_center, valid_sequence):
    overlap = torch.tensor(eval_overlap)
    center = torch.tensor(eval_center)
    auc_curve, auc = get_auc_curve(overlap, valid_sequence)
    prec_curve, prec = get_prec_curve(center, valid_sequence)
    return float(auc[0].item()), float(prec[0].item())


def _eval_mode(trackers, dataset, report_name: str, use_pred_as_gt: bool):
    n_seq = len(dataset)
    n_trk = len(trackers)
    overlap_acc = torch.zeros((n_seq, n_trk, 21), dtype=torch.float32)
    center_acc = torch.zeros((n_seq, n_trk, 51), dtype=torch.float32)
    valid_sequence = torch.ones(n_seq, dtype=torch.uint8)
    threshold_overlap = torch.arange(0.0, 1.05, 0.05, dtype=torch.float64)
    threshold_center = torch.arange(0, 51, dtype=torch.float64)

    for seq_id, seq in enumerate(dataset):
        anno_bb = torch.tensor(seq.ground_truth_rect, dtype=torch.float64)
        target_visible = (
            torch.tensor(seq.target_visible, dtype=torch.uint8)
            if seq.target_visible is not None
            else None
        )
        for trk_id, trk in enumerate(trackers):
            results_path = os.path.join(
                trk.results_dir, report_name, f"{seq.name}.txt"
            )
            if not os.path.isfile(results_path):
                print(f"[WARN] 缺少結果檔，略過序列: {results_path}")
                valid_sequence[seq_id] = 0
                continue

            pred_bb = torch.tensor(
                load_text(str(results_path), delimiter=("\t", ","), dtype=np.float64)
            )
            compare_bb = pred_bb.clone() if use_pred_as_gt else anno_bb

            err_overlap, err_center, _, valid_frame = calc_seq_err_robust(
                pred_bb, compare_bb, seq.dataset, target_visible
            )

            seq_length = compare_bb.shape[0]
            if seq_length <= 0:
                valid_sequence[seq_id] = 0
                continue

            overlap_acc[seq_id, trk_id, :] = (
                err_overlap.view(-1, 1) > threshold_overlap.view(1, -1)
            ).sum(0).float() / seq_length
            center_acc[seq_id, trk_id, :] = (
                err_center.view(-1, 1) <= threshold_center.view(1, -1)
            ).sum(0).float() / seq_length

    return _auc_prec_from_curves(overlap_acc, center_acc, valid_sequence)


def parse_args():
    p = argparse.ArgumentParser(
        description="Sanity check: pred vs 真 GT 與 pred vs pred（假 GT）"
    )
    p.add_argument("--tracker", type=str, default="sglatrack")
    p.add_argument("--param", type=str, required=True)
    p.add_argument("--dataset", type=str, default="uav123")
    p.add_argument("--runid", type=int, default=None)
    p.add_argument("--display_name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ['CONFIG'] = args.param
    dataset = get_dataset(args.dataset)
    trackers = trackerlist(
        args.tracker,
        args.param,
        dataset,
        args.runid,
        args.display_name,
    )
    report_name = args.dataset

    print("=" * 72)
    print("Benchmark sanity check（不修改 dataset 標註檔）")
    print("=" * 72)
    print(f"tracker : {args.tracker}")
    print(f"param   : {args.param}")
    print(f"dataset : {args.dataset}")
    print(f"results : {trackers[0].results_dir}/{report_name}/")
    print()

    normal_auc, normal_prec = _eval_mode(trackers, dataset, report_name, use_pred_as_gt=False)
    sanity_auc, sanity_prec = _eval_mode(trackers, dataset, report_name, use_pred_as_gt=True)

    print(f"[正常] pred vs 真 GT     → AUC {normal_auc:.2f}%  |  Precision {normal_prec:.2f}%")
    print(f"[診斷] pred vs pred(假GT) → AUC {sanity_auc:.2f}%  |  Precision {sanity_prec:.2f}%")
    print()
    print("解讀：")
    if sanity_auc >= 95.0:
        print("  ✓ 診斷 AUC 接近 100%：評估管線（結果檔讀取、對齊、指標計算）大致正常。")
        print("    正常 AUC 偏低時，主因通常是模型預測品質，而非 benchmark 腳本 bug。")
    else:
        print("  ✗ 診斷 AUC 仍偏低：請檢查結果檔是否存在、序列命名、幀數是否與 GT 對齊。")
    print("=" * 72)


if __name__ == "__main__":
    main()
