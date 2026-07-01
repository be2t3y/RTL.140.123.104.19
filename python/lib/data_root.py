"""依實驗 CONFIG 選擇 dataset 根目錄。

預設（多數 yaml）：chanyuan 工作站資料
例外：vit_coco_got10k_distill_mae_teacher_orr_afkd_s60000_bs32（及 _fixed_q** 等衍生 yaml）→ 本機 v3/python/data

由 tracking/test.py、calculate_metrics.py、train 入口設定環境變數 CONFIG；
亦可手動：CONFIG=your_yaml_name python ...
"""
from __future__ import annotations

import os

REPO_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHANYUAN_DATA_ROOT = "/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data"

# 使用本機 python/data 的 yaml（不含 .yaml 副檔名）
# 含 FP32 主 config 及所有 fixed_q** 截斷 test config（共用同一組本機 data）
LOCAL_DATA_CONFIG_PREFIXES = (
    "vit_coco_got10k_distill_mae_teacher_orr_afkd_s60000_bs32",
)


def normalize_config_name(name: str | None) -> str:
    cfg = (name or "").strip()
    if cfg.endswith(".yaml"):
        return cfg[:-5]
    return cfg


def uses_local_data(config_name: str) -> bool:
    cfg = normalize_config_name(config_name)
    return any(cfg == p or cfg.startswith(p + "_") for p in LOCAL_DATA_CONFIG_PREFIXES)


def active_config_name() -> str:
    return normalize_config_name(
        os.environ.get("CONFIG") or os.environ.get("SGLA_CONFIG") or ""
    )


def resolve_data_root(config_name: str | None = None) -> str:
    cfg = normalize_config_name(config_name) if config_name is not None else active_config_name()
    if uses_local_data(cfg):
        return os.path.join(REPO_PYTHON, "data")
    return CHANYUAN_DATA_ROOT
