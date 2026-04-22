#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fk_model import fk_abb_irb_joint_points, pose6_from_q, wrist_center_from_q
from robot_config import (
    DH_A_MM,
    DH_ALPHA_DEG,
    DH_D_MM,
    JOINT_LIMITS_DEG,
    OFFICIAL_WRIST_CENTER_REFERENCES,
    ROBOT_MODEL,
    ROBOT_NAME,
    THETA_OFFSETS_DEG,
    ZERO_POSE_Q_DEG,
)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def evaluate_workspace_references(theta2_offset_deg: float) -> Dict[str, object]:
    theta_offsets_deg = THETA_OFFSETS_DEG.copy()
    theta_offsets_deg[1] = theta2_offset_deg

    samples: List[dict] = []
    errors: List[float] = []

    for ref in OFFICIAL_WRIST_CENTER_REFERENCES:
        q_deg = np.asarray(ref["q_deg"], dtype=float)
        wrist = wrist_center_from_q(q_deg, input_unit="deg", theta_offsets_deg=theta_offsets_deg)
        model_x = float(wrist[0])
        model_z = float(wrist[2])
        ref_x = float(ref["official_wrist_center_x_mm"])
        ref_z = float(ref["official_wrist_center_z_mm"])
        err = float(np.hypot(model_x - ref_x, model_z - ref_z))
        errors.append(err)
        samples.append(
            {
                "name": ref["name"],
                "q_deg": q_deg.tolist(),
                "official_wrist_center_xz_mm": [ref_x, ref_z],
                "model_wrist_center_xyz_mm": wrist.tolist(),
                "model_wrist_center_xz_mm": [model_x, model_z],
                "xz_error_mm": err,
                "source": ref["source"],
            }
        )

    return {
        "theta2_offset_deg": float(theta2_offset_deg),
        "mean_xz_error_mm": float(np.mean(errors)),
        "max_xz_error_mm": float(np.max(errors)),
        "samples": samples,
    }


def build_report() -> dict:
    zero_points = fk_abb_irb_joint_points(ZERO_POSE_Q_DEG, input_unit="deg")
    zero_pose6 = pose6_from_q(ZERO_POSE_Q_DEG, input_unit="deg")

    hypotheses = [
        evaluate_workspace_references(theta2_offset_deg=-90.0),
        evaluate_workspace_references(theta2_offset_deg=0.0),
        evaluate_workspace_references(theta2_offset_deg=90.0),
    ]
    current_result = next(item for item in hypotheses if item["theta2_offset_deg"] == float(THETA_OFFSETS_DEG[1]))

    return {
        "robot_name": ROBOT_NAME,
        "robot_model": ROBOT_MODEL,
        "joint_limits_deg": JOINT_LIMITS_DEG.tolist(),
        "dh_parameters": {
            "a_mm": DH_A_MM.tolist(),
            "alpha_deg": DH_ALPHA_DEG.tolist(),
            "d_mm": DH_D_MM.tolist(),
            "theta_offsets_deg": THETA_OFFSETS_DEG.tolist(),
        },
        "zero_pose": {
            "q_deg": ZERO_POSE_Q_DEG.tolist(),
            "joint_points_mm": zero_points.tolist(),
            "pose6": {
                "x_mm": float(zero_pose6[0]),
                "y_mm": float(zero_pose6[1]),
                "z_mm": float(zero_pose6[2]),
                "phi_rad": float(zero_pose6[3]),
                "theta_rad": float(zero_pose6[4]),
                "psi_rad": float(zero_pose6[5]),
            },
            "mechanical_wrist_center_mm": zero_points[4].tolist(),
            "end_effector_mm": zero_points[6].tolist(),
        },
        "official_workspace_validation": {
            "metric": "IRB1200产品规格.pdf p54 mechanical wrist center XZ error",
            "current_model_result": current_result,
            "theta2_offset_hypotheses": hypotheses,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ABB_IRB FK model against official workspace references.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/fk_validation",
        help="Directory for validation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (ROOT / args.out_dir).resolve()
    report = build_report()
    out_path = out_dir / "fk_validation_report.json"
    save_json(report, out_path)

    current = report["official_workspace_validation"]["current_model_result"]
    print(f"Saved validation report: {out_path}")
    print(f"theta2_offset_deg={current['theta2_offset_deg']:.1f}")
    print(f"mean_xz_error_mm={current['mean_xz_error_mm']:.6f}")
    print(f"max_xz_error_mm={current['max_xz_error_mm']:.6f}")

    for sample in current["samples"]:
        name = sample["name"]
        err = sample["xz_error_mm"]
        model_x, model_z = sample["model_wrist_center_xz_mm"]
        ref_x, ref_z = sample["official_wrist_center_xz_mm"]
        print(
            f"[{name}] model=({model_x:.3f}, {model_z:.3f}) "
            f"official=({ref_x:.3f}, {ref_z:.3f}) err={err:.6f} mm"
        )


if __name__ == "__main__":
    main()
