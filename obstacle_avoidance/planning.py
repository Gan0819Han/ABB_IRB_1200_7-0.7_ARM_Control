#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from fk_model import fk_abb_irb_joint_points
from .collision import ObstacleScene, evaluate_robot_aabb_collision


@dataclass(frozen=True)
class TrajectorySelectionWeights:
    collision_flag_weight: float = 1.0e6
    collision_frame_weight: float = 1.0e4
    collision_violation_weight: float = 100.0
    accuracy_violation_weight: float = 1.0e3
    joint_path_weight: float = 1.0
    max_joint_step_weight: float = 0.25
    clearance_reward_weight: float = 0.1
    clearance_reward_cap_mm: float = 100.0
    pos_tol_mm: float = 1.0
    ori_tol_rad: float = 1.0e-2


def build_joint_trajectory_deg(
    q_start_deg: Iterable[float],
    q_goal_deg: Iterable[float],
    steps: int,
) -> np.ndarray:
    q_start = np.asarray(list(q_start_deg), dtype=float).reshape(6)
    q_goal = np.asarray(list(q_goal_deg), dtype=float).reshape(6)
    total_steps = max(2, int(steps))
    ts = np.linspace(0.0, 1.0, total_steps, dtype=float)
    return (1.0 - ts.reshape(-1, 1)) * q_start.reshape(1, 6) + ts.reshape(-1, 1) * q_goal.reshape(1, 6)


def trajectory_joint_path_length_deg(q_traj_deg: np.ndarray) -> float:
    q = np.asarray(q_traj_deg, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError("q_traj_deg must have shape (N, 6).")
    diffs = np.abs(np.diff(q, axis=0))
    return float(np.sum(diffs))


def trajectory_max_joint_step_deg(q_traj_deg: np.ndarray) -> float:
    q = np.asarray(q_traj_deg, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError("q_traj_deg must have shape (N, 6).")
    diffs = np.abs(np.diff(q, axis=0))
    if diffs.size == 0:
        return 0.0
    return float(np.max(diffs))


def evaluate_trajectory_against_scene(
    q_start_deg: Iterable[float],
    q_goal_deg: Iterable[float],
    scene: ObstacleScene,
    steps: int,
    include_frames: bool = False,
) -> dict:
    q_traj = build_joint_trajectory_deg(q_start_deg=q_start_deg, q_goal_deg=q_goal_deg, steps=steps)
    frame_records: list[dict] = []
    collision_frame_count = 0
    first_collision_frame = -1
    min_clearance_mm = float("inf")
    colliding_links: set[str] = set()
    colliding_obstacles: set[str] = set()

    for frame_index, q_deg in enumerate(q_traj):
        joint_points_mm = fk_abb_irb_joint_points(q_deg, input_unit="deg")
        collision_info = evaluate_robot_aabb_collision(joint_points_mm=joint_points_mm, scene=scene)

        if collision_info["collision"]:
            collision_frame_count += 1
            if first_collision_frame < 0:
                first_collision_frame = int(frame_index)
            colliding_links.update(collision_info["colliding_links"])
            colliding_obstacles.update(collision_info["colliding_obstacles"])

        min_clearance_mm = min(min_clearance_mm, float(collision_info["min_clearance_mm"]))

        if include_frames:
            frame_records.append(
                {
                    "frame_index": int(frame_index),
                    "q_deg": q_deg.tolist(),
                    "joint_points_mm": joint_points_mm.tolist(),
                    "collision": bool(collision_info["collision"]),
                    "min_clearance_mm": float(collision_info["min_clearance_mm"]),
                    "colliding_links": list(collision_info["colliding_links"]),
                    "colliding_obstacles": list(collision_info["colliding_obstacles"]),
                }
            )

    if not np.isfinite(min_clearance_mm):
        min_clearance_mm = float("inf")

    summary = {
        "trajectory_steps": int(q_traj.shape[0]),
        "collision": bool(collision_frame_count > 0),
        "collision_frame_count": int(collision_frame_count),
        "first_collision_frame": int(first_collision_frame),
        "min_clearance_mm": float(min_clearance_mm),
        "joint_path_length_deg": float(trajectory_joint_path_length_deg(q_traj)),
        "max_joint_step_deg": float(trajectory_max_joint_step_deg(q_traj)),
        "colliding_links": sorted(colliding_links),
        "colliding_obstacles": sorted(colliding_obstacles),
    }
    if include_frames:
        summary["frames"] = frame_records
    return summary


def compute_selection_cost(
    final_pos_err_mm: float,
    final_ori_err_rad: float,
    trajectory_summary: dict,
    weights: TrajectorySelectionWeights,
) -> float:
    collision_flag = 1.0 if trajectory_summary["collision"] else 0.0
    collision_frames = float(max(0, int(trajectory_summary["collision_frame_count"])))
    collision_violation_mm = float(max(0.0, -float(trajectory_summary["min_clearance_mm"])))
    pos_violation = max(0.0, float(final_pos_err_mm) - float(weights.pos_tol_mm)) / max(float(weights.pos_tol_mm), 1e-9)
    ori_violation = max(0.0, float(final_ori_err_rad) - float(weights.ori_tol_rad)) / max(float(weights.ori_tol_rad), 1e-9)
    accuracy_violation = pos_violation + ori_violation
    clearance_reward = min(max(0.0, float(trajectory_summary["min_clearance_mm"])), float(weights.clearance_reward_cap_mm))

    return float(
        weights.collision_flag_weight * collision_flag
        + weights.collision_frame_weight * collision_frames
        + weights.collision_violation_weight * collision_violation_mm
        + weights.accuracy_violation_weight * accuracy_violation
        + weights.joint_path_weight * float(trajectory_summary["joint_path_length_deg"])
        + weights.max_joint_step_weight * float(trajectory_summary["max_joint_step_deg"])
        - weights.clearance_reward_weight * clearance_reward
    )


def summarize_candidate_rank(
    final_pos_err_mm: float,
    final_ori_err_rad: float,
    trajectory_summary: dict,
    selection_cost: float,
    weights: TrajectorySelectionWeights,
) -> dict:
    accurate = bool(
        float(final_pos_err_mm) <= float(weights.pos_tol_mm)
        and float(final_ori_err_rad) <= float(weights.ori_tol_rad)
    )
    collision_free = not bool(trajectory_summary["collision"])
    feasible = bool(accurate and collision_free)
    return {
        "accurate": accurate,
        "collision_free": collision_free,
        "feasible": feasible,
        "selection_cost": float(selection_cost),
        "rank_key": [
            0 if feasible else 1,
            0 if collision_free else 1,
            float(selection_cost),
            float(final_pos_err_mm),
            float(final_ori_err_rad),
        ],
    }
