"""將 dataset 的 anno 標註替換為既有 tracking 結果（課堂診斷實驗用）。

警告：會覆寫 data 目錄下的 anno/*.txt。請先自行備份原始 dataset。

替換完成後請跑 calculate_metrics（不必重跑 test.py）：
  python tracking/calculate_metrics.py \\
    --tracker sglatrack \\
    --param <CONFIG> \\
    --dataset uav123

Usage:
  python tracking/replace_anno_with_tracking_results.py \\
    --tracker sglatrack \\
    --param vit_coco_got10k_distill_mae_teacher_orr_afkd_s60000_bs32 \\
    --dataset uav123
"""

from __future__ import annotations

import argparse
import os
import re


def _repo_python() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_results_dir(tracker: str, param: str, dataset: str) -> str:
    return os.path.join(
        _repo_python(), "output", "test", "tracking_results", tracker, param, dataset
    )


def _default_anno_dir(dataset: str) -> str:
    data = os.path.join(_repo_python(), "data")
    if dataset == "uav123":
        return os.path.join(data, "uav123", "UAV123", "anno", "UAV123")
    if dataset == "uav123_10fps":
        return os.path.join(data, "uav123_10fps", "UAV123_10fps", "anno", "UAV123_10fps")
    raise ValueError(f"目前僅支援 uav123 / uav123_10fps，收到：{dataset}")


def _seq_name_to_anno_stem(seq_name: str) -> str:
    """uav_bike1 -> bike1；uav_bird1_2 -> bird1_2"""
    if not seq_name.startswith("uav_"):
        raise ValueError(f"非預期序列名（應為 uav_*）: {seq_name}")
    return seq_name[len("uav_") :]


def _load_pred_txt(path: str) -> list[list[float]]:
    rows: list[list[float]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[\t,]+", line)
            if len(parts) != 4:
                raise ValueError(f"{path} 欄位數不是 4: {line!r}")
            rows.append([float(x) for x in parts])
    return rows


def _format_anno_line(row: list[float]) -> str:
    parts = []
    for v in row:
        if abs(v - round(v)) < 1e-6:
            parts.append(str(int(round(v))))
        else:
            parts.append(f"{v:.6f}".rstrip("0").rstrip("."))
    return ",".join(parts)


def parse_args():
    p = argparse.ArgumentParser(description="以 tracking 結果覆寫 dataset anno（診斷實驗）")
    p.add_argument("--tracker", type=str, default="sglatrack")
    p.add_argument("--param", type=str, required=True)
    p.add_argument("--dataset", type=str, default="uav123")
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--anno-dir", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir or _default_results_dir(
        args.tracker, args.param, args.dataset
    )
    anno_dir = args.anno_dir or _default_anno_dir(args.dataset)

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"找不到 tracking 結果目錄：{results_dir}")
    if not os.path.isdir(anno_dir):
        raise FileNotFoundError(f"找不到 anno 目錄：{anno_dir}")

    print("=" * 72)
    print("replace_anno_with_tracking_results（診斷實驗：anno ← tracking 結果）")
    print("=" * 72)
    print(f"results : {results_dir}")
    print(f"anno    : {anno_dir}")
    print(f"dry-run : {args.dry_run}")
    print()

    pred_files = sorted(
        f for f in os.listdir(results_dir)
        if f.endswith(".txt") and not f.endswith("_time.txt")
    )

    ok, missing_anno = 0, 0
    for fname in pred_files:
        seq_name = fname[:-4]
        anno_stem = _seq_name_to_anno_stem(seq_name)
        anno_path = os.path.join(anno_dir, f"{anno_stem}.txt")
        pred_path = os.path.join(results_dir, fname)

        if not os.path.isfile(anno_path):
            print(f"[MISS anno] {seq_name} -> {anno_path}")
            missing_anno += 1
            continue

        pred = _load_pred_txt(pred_path)
        content = "\n".join(_format_anno_line(r) for r in pred) + "\n"

        if args.dry_run:
            print(f"[DRY] {seq_name} -> {anno_path} ({len(pred)} frames)")
        else:
            with open(anno_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[OK]  {seq_name} -> {anno_path} ({len(pred)} frames)")
        ok += 1

    print()
    print(f"完成：處理 {ok} 個序列，缺少對應 anno {missing_anno}")
    if not args.dry_run and ok > 0:
        print()
        print("下一步（建議只跑 metrics，不必重跑 test；anno 改過後請加 --force）：")
        print(
            f"  python tracking/calculate_metrics.py "
            f"--tracker {args.tracker} --param {args.param} --dataset {args.dataset} --force"
        )
        print()
        print("還原：從你備份的原始 dataset 把 anno/ 複製回來即可。")
    print("=" * 72)


if __name__ == "__main__":
    main()
