#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


def fit_normalizer(x: np.ndarray, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    mean = np.mean(x, axis=0, keepdims=True).astype(np.float32)
    std = np.std(x, axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    return {"mean": mean, "std": std}


def apply_normalizer(x: np.ndarray, normalizer: Dict[str, np.ndarray]) -> np.ndarray:
    return (x - normalizer["mean"]) / normalizer["std"]


def save_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
