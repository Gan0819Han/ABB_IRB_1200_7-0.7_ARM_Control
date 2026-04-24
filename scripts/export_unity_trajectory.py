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

from fk_model import fk_abb_irb, fk_abb_irb_joint_points


def parse_q_deg(text: str) -> np.ndarray:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) != 6:
        raise ValueError("Expected 6 comma-separated joint angles.")
    return np.asarray([float(item) for item in parts], dtype=float)


def python_mm_to_unity_m(position_mm: Sequence[float]) -> np.ndarray:
    px, py, pz = [float(v) for v in position_mm]
    return np.asarray([-py, pz, px], dtype=float) / 1000.0


def build_frames(q_start_deg: np.ndarray, q_goal_deg: np.ndarray, steps: int) -> list[dict]:
    frames: list[dict] = []
    total_steps = max(2, int(steps))

    for frame_idx in range(total_steps):
        t = frame_idx / float(total_steps - 1)
        q_deg = (1.0 - t) * q_start_deg + t * q_goal_deg
        _, p_mm, _ = fk_abb_irb(q_deg, input_unit="deg")
        joint_points_mm = fk_abb_irb_joint_points(q_deg, input_unit="deg")

        frames.append(
            {
                "index": frame_idx,
                "t": t,
                "q_deg": q_deg.tolist(),
                "python_tool_position_mm": p_mm.tolist(),
                "unity_expected_tool_world_position_m": python_mm_to_unity_m(p_mm).tolist(),
                "joint_points_mm": joint_points_mm.tolist(),
            }
        )

    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a simple ABB joint-space trajectory JSON for Unity playback.")
    parser.add_argument("--q_start", required=True, help='Six comma-separated start joint angles in degrees, for example "0,0,0,0,0,0".')
    parser.add_argument("--q_goal", required=True, help='Six comma-separated goal joint angles in degrees, for example "20,30,-40,10,20,0".')
    parser.add_argument("--steps", type=int, default=120, help="Number of trajectory frames.")
    parser.add_argument("--duration", type=float, default=3.0, help="Suggested playback duration in seconds.")
    parser.add_argument("--name", default="abb_joint_linear_traj", help="Trajectory name stored in JSON.")
    parser.add_argument("--out_json", required=True, help="Output JSON path.")
    args = parser.parse_args()

    q_start_deg = parse_q_deg(args.q_start)
    q_goal_deg = parse_q_deg(args.q_goal)
    steps = max(2, int(args.steps))
    duration = max(0.001, float(args.duration))

    frames = build_frames(q_start_deg, q_goal_deg, steps)

    result = {
        "schema": "abb_unity_joint_trajectory_v1",
        "robot_name": "ABB_IRB",
        "trajectory_name": args.name,
        "trajectory_mode": "joint_linear_interpolation",
        "playback_duration_seconds": duration,
        "trajectory_steps": steps,
        "q_start_deg": q_start_deg.tolist(),
        "q_goal_deg": q_goal_deg.tolist(),
        "frames": frames,
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
