#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fk_model import fk_abb_irb, fk_abb_irb_joint_points, rot_to_zyx_euler_rad


def parse_q_deg(text: str) -> np.ndarray:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) != 6:
        raise ValueError("Expected 6 comma-separated joint angles.")
    return np.asarray([float(item) for item in parts], dtype=float)


def python_mm_to_unity_m(position_mm: Sequence[float]) -> np.ndarray:
    px, py, pz = [float(v) for v in position_mm]
    return np.asarray([-py, pz, px], dtype=float) / 1000.0


def python_rotation_to_unity_rotation(rotation_matrix: np.ndarray) -> np.ndarray:
    transform = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return transform @ rotation_matrix @ transform.T


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Python FK reference for Unity-side ABB validation.")
    parser.add_argument("--q", required=True, help='Six comma-separated joint angles in degrees, for example "0,0,0,0,0,0".')
    parser.add_argument("--out_json", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    q_deg = parse_q_deg(args.q)
    T06, p_mm, R06 = fk_abb_irb(q_deg, input_unit="deg")
    euler_zyx_rad = rot_to_zyx_euler_rad(R06)
    joint_points_mm = fk_abb_irb_joint_points(q_deg, input_unit="deg")
    unity_pos_m = python_mm_to_unity_m(p_mm)
    unity_rot = python_rotation_to_unity_rotation(R06)

    result = {
        "q_deg": q_deg.tolist(),
        "python_position_mm": p_mm.tolist(),
        "python_zyx_euler_rad": euler_zyx_rad.tolist(),
        "python_rotation_matrix": R06.reshape(-1).tolist(),
        "unity_expected_world_position_m": unity_pos_m.tolist(),
        "unity_expected_world_rotation_matrix": unity_rot.reshape(-1).tolist(),
        "unity_position_mapping": "Unity(m) = [-Python_y, Python_z, Python_x] / 1000",
        "unity_rotation_mapping": "R_unity = S * R_python * S^T, S=[[0,-1,0],[0,0,1],[1,0,0]]",
        "joint_points_mm": joint_points_mm.tolist(),
        "T06": T06.tolist(),
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
