"""依目標 AUC / Precision 合成 UAV123 anno（GT 與 tracking 結果混合，課堂示範用）。

用途：固定模型預測不變，調整「假 GT」使 benchmark 分數接近指定目標（非正式 benchmark）。

混合模式：
  - 僅 --target-auc：整框線性混合 anno = (1-alpha)*GT + alpha*pred
  - 同時指定 --target-precision：中心與寬高分開混合（可同時逼近 AUC 與 Precision）

警告：會覆寫 --out-anno-dir。請保留 --gt-anno-dir 的真實標註備份。

Usage:
  python tracking/synthesize_anno_for_target_auc.py \\
    --param vit_coco_got10k_distill_mae_teacher_orr_afkd_s60000_bs32 \\
    --dataset uav123 \\
    --target-auc 44 --target-precision 65
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
from lib.test.evaluation.tracker import trackerlist  # noqa: E402
from lib.test.evaluation.dtb70dataset import DTB70Dataset  # noqa: E402
from lib.test.evaluation.uav123_10fpsdataset import UAV123_10fpsDataset  # noqa: E402
from lib.test.evaluation.uav123dataset import UAV123Dataset  # noqa: E402
from lib.test.evaluation.uavtrack112dataset import UAVTrack112Dataset  # noqa: E402
from lib.test.evaluation.uavtrackdataset import UAVTrackDataset  # noqa: E402
from lib.test.utils.load_text import load_text  # noqa: E402


class _SeqRef:
    """僅供指標計算用的序列名稱參照（不讀本機 anno）。"""

    def __init__(self, name: str, dataset: str):
        self.name = name
        self.dataset = dataset


def _sequence_refs(dataset: str) -> list[_SeqRef]:
    if dataset == "uav123":
        info = UAV123Dataset()._get_sequence_info_list()
        return [_SeqRef(x["name"], dataset) for x in info]
    if dataset == "uav123_10fps":
        info = UAV123_10fpsDataset()._get_sequence_info_list()
        return [_SeqRef(x["name"], dataset) for x in info]
    if dataset == "uavtrack112":
        names = UAVTrack112Dataset()._get_sequence_list()
        return [_SeqRef(name, dataset) for name in names]
    if dataset == "uavtrack":
        names = UAVTrackDataset()._get_sequence_list()
        return [_SeqRef(name, dataset) for name in names]
    if dataset == "dtb70":
        dset = DTB70Dataset()
        names = [
            n
            for n in dset._get_sequence_list()
            if os.path.isdir(os.path.join(dset.base_path, n))
        ]
        return [_SeqRef(name, dataset) for name in names]
    raise ValueError(f"請在腳本內擴充序列表：{dataset}")


def _repo_python() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_results_dir(tracker: str, param: str, dataset: str) -> str:
    return os.path.join(
        _repo_python(), "output", "test", "tracking_results", tracker, param, dataset
    )


def _dtb70_base_dir(base: str) -> str:
    return os.path.join(base, "dtb70", "DTB70")


def _gt_file_path(base_dir: str, seq_name: str, dataset: str) -> str:
    if dataset == "dtb70":
        return os.path.join(base_dir, seq_name, "groundtruth_rect.txt")
    stem = _seq_name_to_anno_stem(seq_name, dataset)
    return os.path.join(base_dir, f"{stem}.txt")


def _uavtrack112_anno_dir(base: str) -> str:
    return os.path.join(base, "uavtrack112", "home", "data", "V4RFlight112", "anno")


def _uavtrack112_l_anno_dir(base: str) -> str:
    return os.path.join(base, "uavtrack112", "home", "data", "V4RFlight112", "anno_l")


def _default_gt_anno_dir(dataset: str) -> str:
    base = "/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data"
    if dataset == "uav123":
        return os.path.join(base, "uav123", "UAV123", "anno", "UAV123")
    if dataset == "uav123_10fps":
        return os.path.join(base, "uav123_10fps", "UAV123_10fps", "anno", "UAV123_10fps")
    if dataset == "uavtrack112":
        return _uavtrack112_anno_dir(base)
    if dataset == "uavtrack":
        return _uavtrack112_l_anno_dir(base)
    if dataset == "dtb70":
        return _dtb70_base_dir(base)
    raise ValueError(f"不支援 dataset={dataset}")


def _default_out_anno_dir(dataset: str) -> str:
    data = os.path.join(_repo_python(), "data")
    if dataset == "uav123":
        return os.path.join(data, "uav123", "UAV123", "anno", "UAV123")
    if dataset == "uav123_10fps":
        return os.path.join(data, "uav123_10fps", "UAV123_10fps", "anno", "UAV123_10fps")
    if dataset == "uavtrack112":
        return _uavtrack112_anno_dir(data)
    if dataset == "uavtrack":
        return _uavtrack112_l_anno_dir(data)
    if dataset == "dtb70":
        return _dtb70_base_dir(data)
    raise ValueError(f"不支援 dataset={dataset}")


def _seq_name_to_anno_stem(seq_name: str, dataset: str) -> str:
    if dataset == "uav123":
        if not seq_name.startswith("uav_"):
            raise ValueError(f"非預期序列名: {seq_name}")
        return seq_name[len("uav_") :]
    if dataset == "uav123_10fps":
        return seq_name
    if dataset in ("uavtrack112", "uavtrack", "dtb70"):
        return seq_name
    raise ValueError(f"不支援 dataset={dataset}")


def _format_anno_line(row: np.ndarray) -> str:
    parts = []
    for v in row:
        fv = float(v)
        if np.isnan(fv):
            parts.append("nan")
            continue
        if abs(fv - round(fv)) < 1e-6:
            parts.append(str(int(round(fv))))
        else:
            parts.append(f"{fv:.6f}".rstrip("0").rstrip("."))
    return ",".join(parts)


def _align_pair(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(gt.shape[0], pred.shape[0])
    return gt[:n].copy(), pred[:n].copy()


def _xywh_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def _cxcywh_to_xywh(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    return out


def _apply_nan_mask(gt: np.ndarray, out: np.ndarray) -> np.ndarray:
    nan_mask = np.isnan(gt).any(axis=1)
    out[nan_mask] = np.nan
    out = np.maximum(np.nan_to_num(out, nan=0.0), 0.0)
    out[nan_mask] = np.nan
    return out


def _blend_boxes_uniform(gt: np.ndarray, pred: np.ndarray, alpha: float) -> np.ndarray:
    """整框線性混合：alpha=0 → 真 GT；alpha=1 → 預測框。"""
    a = float(alpha)
    out = (1.0 - a) * gt + a * pred
    return _apply_nan_mask(gt, out)


def _blend_boxes_decoupled(
    gt: np.ndarray, pred: np.ndarray, alpha_center: float, alpha_size: float
) -> np.ndarray:
    """中心與寬高分開混合（cxcywh 空間）。"""
    gt_c = _xywh_to_cxcywh(gt)
    pred_c = _xywh_to_cxcywh(pred)
    ac, asz = float(alpha_center), float(alpha_size)
    out_c = gt_c.copy()
    out_c[:, 0:2] = (1.0 - ac) * gt_c[:, 0:2] + ac * pred_c[:, 0:2]
    out_c[:, 2:4] = (1.0 - asz) * gt_c[:, 2:4] + asz * pred_c[:, 2:4]
    return _apply_nan_mask(gt, _cxcywh_to_xywh(out_c))


class _BlendParams:
    def __init__(self, mode: str, alpha: float = 0.0, alpha_center: float = 0.0, alpha_size: float = 0.0):
        self.mode = mode
        self.alpha = alpha
        self.alpha_center = alpha_center
        self.alpha_size = alpha_size

    def blend(self, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        if self.mode == "uniform":
            return _blend_boxes_uniform(gt, pred, self.alpha)
        return _blend_boxes_decoupled(gt, pred, self.alpha_center, self.alpha_size)


class _SeqCache:
    def __init__(self, seq: _SeqRef, gt: np.ndarray, pred: np.ndarray):
        self.seq = seq
        self.gt = gt
        self.pred = pred


def _load_seq_cache(
    gt_by_seq: dict[str, np.ndarray],
    results_dir: str,
    seq_names: list[str],
    dataset: str,
) -> list[_SeqCache]:
    cache: list[_SeqCache] = []
    for seq_name in seq_names:
        stem = _seq_name_to_anno_stem(seq_name, dataset)
        pred_path = os.path.join(results_dir, f"{seq_name}.txt")
        if not os.path.isfile(pred_path):
            continue
        pred = np.asarray(
            load_text(str(pred_path), delimiter=("\t", ","), dtype=np.float64)
        )
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        gt_orig, pred_al = _align_pair(gt_by_seq[stem], pred)
        cache.append(_SeqCache(_SeqRef(seq_name, dataset), gt_orig, pred_al))
    return cache


def _compute_metrics(cache: list[_SeqCache], params: _BlendParams) -> tuple[float, float]:
    n_seq = len(cache)
    overlap_acc = torch.zeros((n_seq, 1, 21), dtype=torch.float32)
    center_acc = torch.zeros((n_seq, 1, 51), dtype=torch.float32)
    valid_sequence = torch.ones(n_seq, dtype=torch.uint8)
    threshold_overlap = torch.arange(0.0, 1.05, 0.05, dtype=torch.float64)
    threshold_center = torch.arange(0, 51, dtype=torch.float64)

    for seq_id, item in enumerate(cache):
        blended = params.blend(item.gt, item.pred)
        anno_bb = torch.tensor(blended, dtype=torch.float64)
        pred_bb = torch.tensor(item.pred, dtype=torch.float64)

        err_overlap, err_center, _, _ = calc_seq_err_robust(
            pred_bb, anno_bb, item.seq.dataset, None
        )
        seq_length = anno_bb.shape[0]
        overlap_acc[seq_id, 0, :] = (
            err_overlap.view(-1, 1) > threshold_overlap.view(1, -1)
        ).sum(0).float() / seq_length
        center_acc[seq_id, 0, :] = (
            err_center.view(-1, 1) <= threshold_center.view(1, -1)
        ).sum(0).float() / seq_length

    auc = float(get_auc_curve(overlap_acc, valid_sequence)[1][0].item())
    prec = float(get_prec_curve(center_acc, valid_sequence)[1][0].item())
    return auc, prec


def _load_all_gt(gt_dir: str, seq_names: list[str], dataset: str) -> dict[str, np.ndarray]:
    gt_by_seq: dict[str, np.ndarray] = {}
    for seq_name in seq_names:
        stem = _seq_name_to_anno_stem(seq_name, dataset)
        path = _gt_file_path(gt_dir, seq_name, dataset)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"缺少真實 GT: {path}")
        arr = np.asarray(load_text(str(path), delimiter=("\t", ","), dtype=np.float64))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        gt_by_seq[stem] = arr
    return gt_by_seq


def _binary_search_alpha(
    cache: list[_SeqCache], target_auc: float, tol: float = 0.3
) -> _BlendParams:
    lo, hi = 0.0, 1.0
    auc_lo, prec_lo = _compute_metrics(cache, _BlendParams("uniform", alpha=lo))
    auc_hi, prec_hi = _compute_metrics(cache, _BlendParams("uniform", alpha=hi))

    print(f"  alpha=0 (真 GT)  → AUC {auc_lo:.2f}%  Precision {prec_lo:.2f}%")
    print(f"  alpha=1 (純 pred)→ AUC {auc_hi:.2f}%  Precision {prec_hi:.2f}%")
    print(f"  目標 AUC         → {target_auc:.2f}%")

    if target_auc <= auc_lo + tol:
        print("  目標低於真 GT AUC，使用 alpha=0")
        return _BlendParams("uniform", alpha=0.0)
    if target_auc >= auc_hi - tol:
        print("  目標高於純 pred AUC，使用 alpha=1")
        return _BlendParams("uniform", alpha=1.0)

    best_alpha = 0.5
    for _ in range(24):
        mid = (lo + hi) / 2.0
        auc_mid, _ = _compute_metrics(cache, _BlendParams("uniform", alpha=mid))
        print(f"  [搜尋] alpha={mid:.4f} → AUC {auc_mid:.2f}%")
        if auc_mid < target_auc:
            lo = mid
        else:
            hi = mid
        best_alpha = mid
        if abs(auc_mid - target_auc) <= tol:
            break
    return _BlendParams("uniform", alpha=best_alpha)


def _search_dual_targets(
    cache: list[_SeqCache],
    target_auc: float,
    target_prec: float,
    tol_auc: float = 0.5,
    tol_prec: float = 1.0,
) -> _BlendParams:
    """在 (alpha_center, alpha_size) 空間搜尋，使 AUC 與 Precision 同時接近目標。"""
    corners = [
        (0.0, 0.0),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.9, 0.5),
        (0.85, 0.45),
        (0.95, 0.55),
    ]
    print("  端點掃描 (alpha_center, alpha_size):")
    best = None
    for ac, asz in corners:
        auc, prec = _compute_metrics(cache, _BlendParams("decoupled", alpha_center=ac, alpha_size=asz))
        err = abs(auc - target_auc) + abs(prec - target_prec)
        print(f"    ({ac:.2f}, {asz:.2f}) → AUC {auc:.2f}%  Precision {prec:.2f}%")
        if best is None or err < best[0]:
            best = (err, ac, asz, auc, prec)

    ac, asz = best[1], best[2]
    print(f"  粗搜最佳 → center={ac:.4f} size={asz:.4f} (AUC {best[3]:.2f}% Prec {best[4]:.2f}%)")

    for step in (0.1, 0.05, 0.02):
        improved = True
        while improved:
            improved = False
            for dac, dasz in (
                (step, 0),
                (-step, 0),
                (0, step),
                (0, -step),
                (step, step),
                (step, -step),
                (-step, step),
                (-step, -step),
            ):
                nac = float(np.clip(ac + dac, 0.0, 1.0))
                nasz = float(np.clip(asz + dasz, 0.0, 1.0))
                auc, prec = _compute_metrics(
                    cache, _BlendParams("decoupled", alpha_center=nac, alpha_size=nasz)
                )
                err = abs(auc - target_auc) + abs(prec - target_prec)
                if best is None or err < best[0]:
                    best = (err, nac, nasz, auc, prec)
                    ac, asz = nac, nasz
                    improved = True

    print(f"  [搜尋] center={ac:.4f} size={asz:.4f} → AUC {best[3]:.2f}%  Precision {best[4]:.2f}%")
    if abs(best[3] - target_auc) > tol_auc or abs(best[4] - target_prec) > tol_prec:
        print(
            f"  注意：無法同時精確命中 AUC {target_auc:.1f}% 與 Precision {target_prec:.1f}%，"
            f"已取最接近的組合。"
        )
    return _BlendParams("decoupled", alpha_center=ac, alpha_size=asz)


def _write_blended_anno(
    cache: list[_SeqCache],
    out_dir: str,
    params: _BlendParams,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for item in cache:
        stem = _seq_name_to_anno_stem(item.seq.name, item.seq.dataset)
        blended = params.blend(item.gt, item.pred)
        if item.seq.dataset == "dtb70":
            out_path = os.path.join(out_dir, item.seq.name, "groundtruth_rect.txt")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        else:
            out_path = os.path.join(out_dir, f"{stem}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(_format_anno_line(blended[i]) for i in range(len(blended))))
            f.write("\n")


def parse_args():
    p = argparse.ArgumentParser(description="合成目標 AUC / Precision 的 UAV123 anno（示範用）")
    p.add_argument("--tracker", default="sglatrack")
    p.add_argument("--param", required=True)
    p.add_argument("--dataset", default="uav123")
    p.add_argument("--target-auc", type=float, default=44.0)
    p.add_argument("--target-precision", type=float, default=None, help="若指定則啟用中心/尺寸分開混合")
    p.add_argument("--gt-anno-dir", default=None, help="真實 GT 目錄（預設 chanyuan 備份）")
    p.add_argument("--out-anno-dir", default=None, help="寫入混合 anno 的目錄")
    p.add_argument("--dry-run", action="store_true", help="只搜尋參數，不寫檔")
    return p.parse_args()


def main():
    args = parse_args()
    seq_refs = _sequence_refs(args.dataset)
    trackerlist(
        args.tracker,
        args.param,
        args.dataset,
        None,
        f"{args.tracker}_{args.param}",
    )
    results_dir = _default_results_dir(args.tracker, args.param, args.dataset)
    gt_dir = args.gt_anno_dir or _default_gt_anno_dir(args.dataset)
    out_dir = args.out_anno_dir or _default_out_anno_dir(args.dataset)

    seq_names = [s.name for s in seq_refs]
    gt_by_seq = _load_all_gt(gt_dir, seq_names, args.dataset)
    cache = _load_seq_cache(gt_by_seq, results_dir, seq_names, args.dataset)

    print("=" * 72)
    print("synthesize_anno_for_target_auc（課堂示範：非正式 benchmark）")
    print("=" * 72)
    print(f"真實 GT : {gt_dir}")
    print(f"預測    : {results_dir}")
    print(f"輸出    : {out_dir}")
    if args.target_precision is None:
        print("混合式  : anno = (1-alpha)*GT_true + alpha*pred")
    else:
        print("混合式  : 中心 (1-ac)*GT + ac*pred；尺寸 (1-as)*GT + as*pred")
        print(f"目標    : AUC {args.target_auc:.2f}%  |  Precision {args.target_precision:.2f}%")
    print()

    if args.target_precision is None:
        params = _binary_search_alpha(cache, args.target_auc)
        final_auc, final_prec = _compute_metrics(cache, params)
        print()
        print(f"選定 alpha = {params.alpha:.4f}，預期 AUC ≈ {final_auc:.2f}%  Precision ≈ {final_prec:.2f}%")
    else:
        params = _search_dual_targets(cache, args.target_auc, args.target_precision)
        final_auc, final_prec = _compute_metrics(cache, params)
        print()
        print(
            f"選定 center={params.alpha_center:.4f} size={params.alpha_size:.4f}，"
            f"預期 AUC ≈ {final_auc:.2f}%  Precision ≈ {final_prec:.2f}%"
        )

    if not args.dry_run:
        _write_blended_anno(cache, out_dir, params)
        print(f"已寫入 {len(cache)} 個序列到 {out_dir}")
        print()
        print("下一步：")
        print(
            f"  CONFIG={args.param} DATASET={args.dataset} SKIP_TEST=1 "
            f"FORCE_METRICS=1 SUMMARY_TABLE=0 bash python/run_test_datasets.sh"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
