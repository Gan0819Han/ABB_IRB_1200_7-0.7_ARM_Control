#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from fk_model import JOINT_LIMITS_DEG, fk_abb_irb_joint_points
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


def clip_q_to_joint_limits_deg(q_deg: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(q_deg), dtype=float).reshape(6)
    return np.clip(q, JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1])


def build_waypoint_joint_trajectory_candidates_deg(
    q_start_deg: Iterable[float],
    q_goal_deg: Iterable[float],
) -> list[dict]:
    q_start = np.asarray(list(q_start_deg), dtype=float).reshape(6)
    q_goal = np.asarray(list(q_goal_deg), dtype=float).reshape(6)
    q_mid = 0.5 * (q_start + q_goal)
    delta = q_goal - q_start

    templates: list[tuple[str, np.ndarray]] = [
        ("direct", q_goal.reshape(6)),
        ("midpoint", q_mid),
        ("lift_shoulder", q_mid + np.array([0.0, 25.0, -20.0, 10.0, 0.0, 0.0], dtype=float)),
        ("drop_shoulder", q_mid + np.array([0.0, -25.0, 20.0, -10.0, 0.0, 0.0], dtype=float)),
        ("lift_elbow", q_mid + np.array([0.0, 10.0, -30.0, 12.0, 0.0, 0.0], dtype=float)),
        ("drop_elbow", q_mid + np.array([0.0, -10.0, 30.0, -12.0, 0.0, 0.0], dtype=float)),
        ("swing_wrist_pos", q_mid + np.array([0.0, 0.0, 0.0, 20.0, 0.0, 15.0], dtype=float)),
        ("swing_wrist_neg", q_mid + np.array([0.0, 0.0, 0.0, -20.0, 0.0, -15.0], dtype=float)),
        ("start_biased", 0.7 * q_start + 0.3 * q_goal + np.array([0.0, 18.0, -15.0, 8.0, 0.0, 0.0], dtype=float)),
        ("goal_biased", 0.3 * q_start + 0.7 * q_goal + np.array([0.0, -18.0, 15.0, -8.0, 0.0, 0.0], dtype=float)),
    ]

    candidates: list[dict] = []
    seen_signatures: set[tuple[float, ...]] = set()
    for mode, waypoint in templates:
        if mode == "direct":
            q_points = np.vstack([q_start, q_goal])
        else:
            q_wp = clip_q_to_joint_limits_deg(waypoint)
            q_points = np.vstack([q_start, q_wp, q_goal])
        signature = tuple(np.round(q_points.reshape(-1), 4).tolist())
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        item = {
            "trajectory_mode": mode,
            "q_points_deg": q_points,
            "waypoint_deg": None if mode == "direct" else q_points[1].tolist(),
        }
        candidates.append(item)
    return candidates


def build_joint_trajectory_deg(
    q_start_deg: Iterable[float],
    q_goal_deg: Iterable[float],
    steps: int,
) -> np.ndarray:
    return build_piecewise_joint_trajectory_deg([q_start_deg, q_goal_deg], steps)


def build_piecewise_joint_trajectory_deg(
    q_points_deg: Iterable[Iterable[float]],
    steps: int,
) -> np.ndarray:
    points = [np.asarray(list(point), dtype=float).reshape(6) for point in q_points_deg]
    if len(points) < 2:
        raise ValueError("q_points_deg must contain at least two waypoints.")

    total_steps = max(int(steps), len(points))
    total_points = total_steps + len(points) - 2
    segment_lengths = []
    for idx in range(len(points) - 1):
        segment_lengths.append(float(np.sum(np.abs(points[idx + 1] - points[idx]))))
    total_length = float(np.sum(segment_lengths))

    if total_length <= 0.0:
        return np.repeat(points[0].reshape(1, 6), total_steps, axis=0)

    raw_alloc = [max(2, int(round(total_points * (length / total_length)))) for length in segment_lengths]
    allocated = int(np.sum(raw_alloc))
    while allocated > total_points:
        idx = int(np.argmax(raw_alloc))
        if raw_alloc[idx] > 2:
            raw_alloc[idx] -= 1
            allocated -= 1
        else:
            break
    while allocated < total_points:
        idx = int(np.argmax(segment_lengths))
        raw_alloc[idx] += 1
        allocated += 1

    traj_parts: list[np.ndarray] = []
    for seg_idx, seg_steps in enumerate(raw_alloc):
        start = points[seg_idx]
        end = points[seg_idx + 1]
        ts = np.linspace(0.0, 1.0, max(2, int(seg_steps)), dtype=float)
        seg = (1.0 - ts.reshape(-1, 1)) * start.reshape(1, 6) + ts.reshape(-1, 1) * end.reshape(1, 6)
        if seg_idx > 0:
            seg = seg[1:]
        traj_parts.append(seg)
    return np.vstack(traj_parts)


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
    return evaluate_joint_trajectory_against_scene(q_traj_deg=q_traj, scene=scene, include_frames=include_frames)


def evaluate_joint_trajectory_against_scene(
    q_traj_deg: np.ndarray,
    scene: ObstacleScene,
    include_frames: bool = False,
) -> dict:
    q_traj = np.asarray(q_traj_deg, dtype=float)
    if q_traj.ndim != 2 or q_traj.shape[1] != 6:
        raise ValueError("q_traj_deg must have shape (N, 6).")

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
