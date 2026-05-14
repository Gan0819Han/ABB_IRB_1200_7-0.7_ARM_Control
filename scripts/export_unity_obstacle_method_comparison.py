#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def python_mm_to_unity_m(position_mm: Sequence[float]) -> np.ndarray:
    px, py, pz = [float(v) for v in position_mm]
    return np.asarray([-py, pz, px], dtype=float) / 1000.0


def python_size_mm_to_unity_scale_m(size_mm: Sequence[float]) -> np.ndarray:
    sx, sy, sz = [float(v) for v in size_mm]
    return np.asarray([sy, sz, sx], dtype=float) / 1000.0


def vec3_payload(values: Iterable[float]) -> dict:
    x, y, z = [float(v) for v in values]
    return {"x": x, "y": y, "z": z}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Unity-friendly obstacle method comparison JSON.")
    parser.add_argument("--comparison_json", required=True, help="Input obstacle method comparison JSON.")
    parser.add_argument("--out_json", required=True, help="Output Unity JSON path.")
    parser.add_argument("--demo_name", default="", help="Optional Unity demo name override.")
    return parser.parse_args()


def build_unity_frames_from_planning_frames(frames: list[dict]) -> list[dict]:
    unity_frames = []
    for frame in frames:
        tool_mm = np.asarray(frame["joint_points_mm"], dtype=float)[-1]
        unity_frames.append(
            {
                "frame_index": int(frame["frame_index"]),
                "q_deg": [float(v) for v in frame["q_deg"]],
                "tool_world_m": vec3_payload(python_mm_to_unity_m(tool_mm)),
                "collision": bool(frame["collision"]),
                "min_clearance_mm": float(frame["min_clearance_mm"]),
            }
        )
    return unity_frames


def build_method_payload(method: dict, color_hint: str) -> dict:
    selected = method["selected_solution"]
    summary = selected["trajectory_summary"]
    frames = summary.get("frames", [])
    if not frames:
        raise ValueError(f"Method {method.get('label', '-')} does not contain saved frames.")
    return {
        "method_id": str(method["method_id"]),
        "label": str(method["label"]),
        "color_hint": color_hint,
        "selected_solution_collision_free": bool(method["selected_solution_collision_free"]),
        "trajectory_mode": str(method["trajectory_mode"]),
        "final_pos_err_mm": float(method["final_pos_err_mm"]),
        "final_ori_err_rad": float(method["final_ori_err_rad"]),
        "min_clearance_mm": float(method["min_clearance_mm"]),
        "frames": build_unity_frames_from_planning_frames(frames),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.comparison_json).read_text(encoding="utf-8"))
    scene = payload["scene"]
    target_pose6 = payload["target_pose6"]

    unity_obstacles = []
    for obstacle in scene.get("obstacles", []):
        center_mm = obstacle["center_mm"]
        size_mm = obstacle["size_mm"]
        unity_obstacles.append(
            {
                "name": obstacle["name"],
                "center_world_m": vec3_payload(python_mm_to_unity_m(center_mm)),
                "size_world_m": vec3_payload(python_size_mm_to_unity_scale_m(size_mm)),
                "center_mm": [float(v) for v in center_mm],
                "size_mm": [float(v) for v in size_mm],
            }
        )

    color_map = {
        "nn_nr": "#2563EB",
        "dls": "#F97316",
        "lbfgsb": "#10B981",
    }
    methods = [
        build_method_payload(method, color_map.get(str(method.get("method_id")), "#64748B"))
        for method in payload.get("methods", [])
    ]

    result = {
        "schema": "abb_unity_obstacle_method_compare_v1",
        "demo_name": args.demo_name.strip() or payload.get("comparison_name", "obstacle_method_compare"),
        "scene_name": scene.get("scene_name", ""),
        "target_pose6": [float(v) for v in target_pose6],
        "target_world_m": vec3_payload(python_mm_to_unity_m(target_pose6[:3])),
        "q_start_deg": [float(v) for v in payload.get("q_start_deg", [0, 0, 0, 0, 0, 0])],
        "obstacles": unity_obstacles,
        "methods": methods,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
