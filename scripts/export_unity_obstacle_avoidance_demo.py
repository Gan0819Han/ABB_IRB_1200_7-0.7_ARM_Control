#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from obstacle_avoidance.collision import ObstacleScene
from obstacle_avoidance.planning import evaluate_trajectory_against_scene


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
    parser = argparse.ArgumentParser(description="Export a Unity-friendly obstacle avoidance demo JSON.")
    parser.add_argument("--plan_json", required=True, help="Input planning result JSON from plan_collision_free_ik.py")
    parser.add_argument("--out_json", required=True, help="Output Unity-friendly JSON path")
    parser.add_argument("--demo_name", default="", help="Optional Unity demo name override")
    parser.add_argument(
        "--collision_candidate_mode",
        choices=["none", "best_collision"],
        default="best_collision",
        help="Whether to also export one colliding candidate trajectory for Unity comparison.",
    )
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


def build_solution_payload_from_summary(solution: dict, trajectory_summary: dict) -> dict:
    frames = trajectory_summary.get("frames", [])
    if not frames:
        raise ValueError("Trajectory summary does not contain per-frame data.")
    return {
        "subspace_id": int(solution["subspace_id"]),
        "q0_deg": [float(v) for v in solution["q0_deg"]],
        "q_goal_deg": [float(v) for v in solution["q_goal_deg"]],
        "nr_iters": int(solution["nr_iters"]),
        "nr_converged": bool(solution["nr_converged"]),
        "final_pose6": [float(v) for v in solution["final_pose6"]],
        "final_pos_err_mm": float(solution["final_pos_err_mm"]),
        "final_ori_err_rad": float(solution["final_ori_err_rad"]),
        "collision": bool(trajectory_summary["collision"]),
        "collision_frame_count": int(trajectory_summary["collision_frame_count"]),
        "first_collision_frame": int(trajectory_summary["first_collision_frame"]),
        "min_clearance_mm": float(trajectory_summary["min_clearance_mm"]),
        "trajectory_steps": int(trajectory_summary["trajectory_steps"]),
        "joint_path_length_deg": float(trajectory_summary["joint_path_length_deg"]),
        "max_joint_step_deg": float(trajectory_summary["max_joint_step_deg"]),
        "frames": build_unity_frames_from_planning_frames(frames),
    }


def select_collision_candidate(payload: dict) -> dict | None:
    for item in payload.get("evaluated_candidates", []):
        if bool(item["trajectory_summary"]["collision"]):
            return item
    return None


def main() -> None:
    args = parse_args()
    plan_path = Path(args.plan_json)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))

    target_pose6 = payload["target_pose6"]
    target_unity_world_m = python_mm_to_unity_m(target_pose6[:3])
    scene = payload["scene"]
    selected = payload["selected_solution"]
    trajectory_summary = selected["trajectory_summary"]
    frames = trajectory_summary.get("frames", [])
    if not frames:
        raise ValueError("The selected solution does not contain per-frame data. Re-run planning with --save_selected_frames.")

    unity_obstacles = []
    for obstacle in scene["obstacles"]:
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

    selected_solution_payload = build_solution_payload_from_summary(
        solution=selected,
        trajectory_summary=trajectory_summary,
    )

    collision_solution_payload = None
    if args.collision_candidate_mode == "best_collision":
        collision_candidate = select_collision_candidate(payload)
        if collision_candidate is not None:
            collision_summary = evaluate_trajectory_against_scene(
                q_start_deg=payload["q_start_deg"],
                q_goal_deg=collision_candidate["q_goal_deg"],
                scene=ObstacleScene.from_dict(scene),
                steps=int(collision_candidate["trajectory_summary"]["trajectory_steps"]),
                include_frames=True,
            )
            collision_solution_payload = build_solution_payload_from_summary(
                solution=collision_candidate,
                trajectory_summary=collision_summary,
            )

    result = {
        "schema": "abb_unity_obstacle_demo_v2",
        "demo_name": args.demo_name.strip() or scene.get("scene_name", "obstacle_demo"),
        "planner_name": payload.get("planner_name", ""),
        "scene_name": scene.get("scene_name", ""),
        "target_pose6": [float(v) for v in target_pose6],
        "target_world_m": vec3_payload(target_unity_world_m),
        "q_start_deg": [float(v) for v in payload.get("q_start_deg", [0, 0, 0, 0, 0, 0])],
        "obstacles": unity_obstacles,
        "selected_solution": selected_solution_payload,
        "comparison_collision_solution": collision_solution_payload,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
