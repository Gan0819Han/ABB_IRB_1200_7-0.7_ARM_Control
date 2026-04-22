#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


DEFAULT_NAMING_CONFIG: Dict[str, object] = {
    "project_tag": "",
    "robot_name": "abb_irb",
    "task_name": "fk",
    "sampling_name": "uniform_random",
    "default_split_ratios": [0.7, 0.15, 0.15],
}


def load_naming_config(config_path: str | Path) -> Dict[str, object]:
    cfg = dict(DEFAULT_NAMING_CONFIG)
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        if not isinstance(user_cfg, dict):
            raise ValueError(f"Invalid naming config format: {p}")
        for k in DEFAULT_NAMING_CONFIG:
            if k in user_cfg:
                cfg[k] = user_cfg[k]
    return cfg


def make_base_name(n_samples: int, seed: int, cfg: Dict[str, object]) -> str:
    prefix = f"{cfg['project_tag']}_" if str(cfg["project_tag"]).strip() else ""
    return (
        f"{prefix}{cfg['robot_name']}_{cfg['task_name']}_"
        f"{cfg['sampling_name']}_{n_samples}_seed{seed}"
    )


def make_full_filenames(base_name: str) -> Dict[str, str]:
    return {
        "csv": f"{base_name}_full.csv",
        "npz": f"{base_name}_full.npz",
        "meta": f"{base_name}_full_meta.json",
    }


def ratio_token(r: float) -> str:
    pct = r * 100.0
    rounded = round(pct)
    if abs(pct - rounded) < 1e-9:
        return str(int(rounded))
    s = f"{pct:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def make_split_tag(ratios: Iterable[float]) -> str:
    return "_".join(ratio_token(float(r)) for r in ratios)


def make_split_filenames(base_name: str, ratios: Tuple[float, float, float]) -> Dict[str, str]:
    split_tag = make_split_tag(ratios)
    return {
        "train": f"{base_name}_split{split_tag}_train.npz",
        "val": f"{base_name}_split{split_tag}_val.npz",
        "test": f"{base_name}_split{split_tag}_test.npz",
        "meta": f"{base_name}_split{split_tag}_meta.json",
    }
