#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from abb_nn.optimization import (
    DLSOptions,
    LBFGSBOptions,
    NROptions,
    dls_refine,
    evaluate_solution_metrics,
    lbfgsb_refine,
    newton_raphson_refine,
)
from obstacle_avoidance.collision import ObstacleScene
from obstacle_avoidance.planning import (
    TrajectorySelectionWeights,
    build_piecewise_joint_trajectory_deg,
    build_waypoint_joint_trajectory_candidates_deg,
    compute_selection_cost,
    evaluate_joint_trajectory_against_scene,
    summarize_candidate_rank,
)
from predict_ik import (
    apply_normalizer,
    generate_hierarchical_candidates,
    load_json,
    load_prediction_pair,
    position_l2_norm,
    predict_q_deg,
    safe_torch_load,
)
from robot_config import JOINT_LIMITS_DEG


def parse_pose(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 6:
        raise ValueError("--pose must be x_mm,y_mm,z_mm,phi_rad,theta_rad,psi_rad")
    return np.asarray(values, dtype=np.float32)


def parse_q_deg(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 6:
        raise ValueError("--q_start must contain 6 joint angles in degrees.")
    return np.asarray(values, dtype=float)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def is_duplicate_solution(
    q_deg: np.ndarray,
    existing: list[np.ndarray],
    atol_deg: float,
) -> bool:
    for prev in existing:
        if np.allclose(q_deg, prev, atol=atol_deg, rtol=0.0):
            return True
    return False


def is_duplicate_guess(q_deg: np.ndarray, existing: list[np.ndarray]) -> bool:
    for prev in existing:
        if np.allclose(q_deg, prev, atol=1e-9, rtol=0.0):
            return True
    return False


def clamp_joint_guess(q_deg: np.ndarray) -> np.ndarray:
    arr = np.asarray(q_deg, dtype=float).reshape(6)
    return np.clip(arr, JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1])


def build_numeric_initial_guesses(
    q_start_deg: np.ndarray,
    *,
    max_guesses: int,
) -> list[np.ndarray]:
    base = np.asarray(q_start_deg, dtype=float).reshape(6)
    offset_library = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 40.0, -60.0, 0.0, 0.0, 0.0],
        [0.0, -40.0, 60.0, 0.0, 0.0, 0.0],
        [0.0, 60.0, -120.0, 0.0, 0.0, 0.0],
        [0.0, -60.0, 120.0, 0.0, 0.0, 0.0],
        [60.0, 20.0, -40.0, 0.0, 0.0, 0.0],
        [-60.0, 20.0, -40.0, 0.0, 0.0, 0.0],
        [90.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-90.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 30.0, -90.0, 90.0, 0.0, 0.0],
        [0.0, 30.0, -90.0, -90.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 180.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -180.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 180.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -180.0],
    ]
    guesses: list[np.ndarray] = []
    for offset in offset_library:
        candidate = clamp_joint_guess(base + np.asarray(offset, dtype=float))
        if is_duplicate_guess(candidate, guesses):
            continue
        guesses.append(candidate)
        if len(guesses) >= max_guesses:
            break
    return guesses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare obstacle-aware planning across NN + NR, DLS, and L-BFGS-B.")
    parser.add_argument("--pose", required=True, help="x_mm,y_mm,z_mm,phi_rad,theta_rad,psi_rad")
    parser.add_argument("--q_start", default="0,0,0,0,0,0", help="Six comma-separated start joint angles in degrees.")
    parser.add_argument("--scene_json", required=True, help="Obstacle scene JSON path.")
    parser.add_argument("--pred_meta", default="artifacts/prediction_system_formal/metadata.json")
    parser.add_argument("--branch_meta", default="artifacts/branch_classification_system/metadata.json")
    parser.add_argument("--fine_meta", default="artifacts/fine_classification_system/metadata.json")
    parser.add_argument("--topk_shoulder", type=int, default=2)
    parser.add_argument("--topk_elbow", type=int, default=1)
    parser.add_argument("--topk_wrist", type=int, default=2)
    parser.add_argument("--max_branch_candidates", type=int, default=6)
    parser.add_argument("--fine_topk_per_branch", type=int, default=3)
    parser.add_argument("--max_subspace_candidates", type=int, default=18)
    parser.add_argument("--dedupe_tol_deg", type=float, default=0.5)
    parser.add_argument("--nr_max_iters", type=int, default=40)
    parser.add_argument("--nr_tol_pos_mm", type=float, default=1e-3)
    parser.add_argument("--nr_tol_ori_rad", type=float, default=1e-3)
    parser.add_argument("--nr_damping", type=float, default=1e-5)
    parser.add_argument("--nr_step_scale", type=float, default=1.0)
    parser.add_argument("--dls_max_iters", type=int, default=80)
    parser.add_argument("--dls_tol_pos_mm", type=float, default=1.0)
    parser.add_argument("--dls_tol_ori_rad", type=float, default=1e-2)
    parser.add_argument("--dls_damping", type=float, default=1e-2)
    parser.add_argument("--dls_orientation_weight", type=float, default=200.0)
    parser.add_argument("--dls_multistart_guesses", type=int, default=12)
    parser.add_argument("--lbfgsb_max_iters", type=int, default=200)
    parser.add_argument("--lbfgsb_tol_pos_mm", type=float, default=1.0)
    parser.add_argument("--lbfgsb_tol_ori_rad", type=float, default=1e-2)
    parser.add_argument("--lbfgsb_orientation_weight", type=float, default=200.0)
    parser.add_argument("--lbfgsb_multistart_guesses", type=int, default=12)
    parser.add_argument("--trajectory_steps", type=int, default=120)
    parser.add_argument("--cost_collision_flag_weight", type=float, default=1.0e6)
    parser.add_argument("--cost_collision_frame_weight", type=float, default=1.0e4)
    parser.add_argument("--cost_collision_violation_weight", type=float, default=100.0)
    parser.add_argument("--cost_accuracy_violation_weight", type=float, default=1.0e3)
    parser.add_argument("--cost_joint_path_weight", type=float, default=1.0)
    parser.add_argument("--cost_max_joint_step_weight", type=float, default=0.25)
    parser.add_argument("--cost_clearance_reward_weight", type=float, default=0.1)
    parser.add_argument("--cost_clearance_reward_cap_mm", type=float, default=100.0)
    parser.add_argument("--selection_pos_tol_mm", type=float, default=1.0)
    parser.add_argument("--selection_ori_tol_rad", type=float, default=1.0e-2)
    parser.add_argument("--comparison_name", default="abb_obstacle_method_compare")
    parser.add_argument("--save_selected_frames", action="store_true")
    parser.add_argument("--out_json", required=True, help="Output JSON path.")
    return parser


def solve_nn_nr_goal(
    target_pose: np.ndarray,
    pred_meta_path: Path,
    branch_meta_path: Path,
    fine_meta_path: Path,
    topk_shoulder: int,
    topk_elbow: int,
    topk_wrist: int,
    max_branch_candidates: int,
    fine_topk_per_branch: int,
    max_subspace_candidates: int,
    dedupe_tol_deg: float,
    nr_options: NROptions,
) -> dict:
    t0 = time.perf_counter()
    pred_meta = load_json(pred_meta_path)
    branch_meta = load_json(branch_meta_path)
    fine_meta = load_json(fine_meta_path)

    pred_profile = pred_meta.get("segment_profile", "abb_strict")
    branch_profile = branch_meta.get("segment_profile", "abb_strict")
    fine_profile = fine_meta.get("segment_profile", "abb_strict")
    if pred_profile != branch_profile or pred_profile != fine_profile:
        raise ValueError(
            "Segment profile mismatch among prediction / branch / fine artifacts. "
            f"prediction={pred_profile}, branch={branch_profile}, fine={fine_profile}"
        )

    pred_mean = np.array(pred_meta["normalizer"]["mean"], dtype=np.float32).reshape(1, -1)
    pred_std = np.array(pred_meta["normalizer"]["std"], dtype=np.float32).reshape(1, -1)
    x_pred = apply_normalizer(target_pose.reshape(1, -1).astype(np.float32), pred_mean, pred_std)

    candidate_labels, candidate_generation_info, timing_generation = generate_hierarchical_candidates(
        target_pose,
        branch_meta_path,
        branch_meta,
        fine_meta_path,
        fine_meta,
        topk_shoulder,
        topk_elbow,
        topk_wrist,
        max_branch_candidates,
        fine_topk_per_branch,
        max_subspace_candidates,
    )

    model_index = {
        int(item["subspace_id"]): item for item in pred_meta["trained_subspaces"]
    }
    available = [int(sid) for sid in candidate_labels if int(sid) in model_index]
    if not available:
        available = sorted(model_index.keys())
    if not available:
        raise RuntimeError("No trained candidate subspaces available for NN + NR obstacle comparison.")

    goal_candidates: list[dict] = []
    unique_refined_solutions: list[np.ndarray] = []

    for rank_index, sid in enumerate(available, start=1):
        ckpt = safe_torch_load(pred_meta_path.parent / "subspace_models" / model_index[int(sid)]["model_file"])
        m15, m6 = load_prediction_pair(ckpt)
        q0 = predict_q_deg(m15, m6, x_pred)
        l2 = position_l2_norm(q0, target_pose)

        nr = newton_raphson_refine(
            q0_deg=q0,
            target_pose6=target_pose,
            options=nr_options,
        )
        q_goal_deg = np.asarray(nr["q_deg"], dtype=float).reshape(6)

        if is_duplicate_solution(q_deg=q_goal_deg, existing=unique_refined_solutions, atol_deg=dedupe_tol_deg):
            continue
        unique_refined_solutions.append(q_goal_deg.copy())

        metrics = evaluate_solution_metrics(q_goal_deg, target_pose)
        goal_candidates.append(
            {
                "candidate_rank": int(rank_index),
                "subspace_id": int(sid),
                "q0_deg": q0.tolist(),
                "q_goal_deg": q_goal_deg.tolist(),
                "position_l2_mm": float(l2),
                "e_max": float(ckpt.get("e_max", np.inf)),
                "converged": bool(nr["converged"]),
                "iters": int(nr["iters"]),
                "final_pose6": metrics["final_pose6"],
                "final_pos_err_mm": float(metrics["final_pos_err_mm"]),
                "final_ori_err_rad": float(metrics["final_ori_err_rad"]),
            }
        )

    if not goal_candidates:
        raise RuntimeError("NN + NR 未生成任何可评估的终点候选。")

    total_ms = float((time.perf_counter() - t0) * 1000.0)
    best_initial = min(goal_candidates, key=lambda item: item["position_l2_mm"])
    return {
        "method_id": "nn_nr",
        "label": "NN + NR",
        "solver_family": "nn",
        "ik_time_ms": total_ms,
        "planning_time_ms": total_ms,
        "q0_deg": best_initial["q0_deg"],
        "q_goal_deg": best_initial["q_goal_deg"],
        "converged": bool(best_initial["converged"]),
        "iters": int(best_initial["iters"]),
        "final_pose6": best_initial["final_pose6"],
        "final_pos_err_mm": float(best_initial["final_pos_err_mm"]),
        "final_ori_err_rad": float(best_initial["final_ori_err_rad"]),
        "candidate_source": "hierarchical_predictions",
        "candidate_subspaces": [int(x) for x in available],
        "candidate_generation": candidate_generation_info,
        "candidate_generation_timing_ms": timing_generation,
        "initial_solution": {
            "subspace_id": int(best_initial["subspace_id"]),
            "q0_deg": best_initial["q0_deg"],
            "position_l2_mm": float(best_initial["position_l2_mm"]),
            "e_max": float(best_initial["e_max"]),
        },
        "goal_candidates": goal_candidates,
    }


def solve_dls_goal(target_pose: np.ndarray, q_start_deg: np.ndarray, options: DLSOptions) -> dict:
    return solve_numeric_goal_multistart(
        method_id="dls",
        label="DLS",
        target_pose=target_pose,
        initial_guesses=build_numeric_initial_guesses(
            q_start_deg,
            max_guesses=int(getattr(options, "multistart_guesses", 12)),
        ),
        dedupe_tol_deg=float(getattr(options, "dedupe_tol_deg", 0.5)),
        solver_func=lambda guess: dls_refine(
            q0_deg=guess,
            target_pose6=target_pose,
            options=options,
        ),
    )


def solve_lbfgsb_goal(target_pose: np.ndarray, q_start_deg: np.ndarray, options: LBFGSBOptions) -> dict:
    return solve_numeric_goal_multistart(
        method_id="lbfgsb",
        label="L-BFGS-B",
        target_pose=target_pose,
        initial_guesses=build_numeric_initial_guesses(
            q_start_deg,
            max_guesses=int(getattr(options, "multistart_guesses", 12)),
        ),
        dedupe_tol_deg=float(getattr(options, "dedupe_tol_deg", 0.5)),
        solver_func=lambda guess: lbfgsb_refine(
            q0_deg=guess,
            target_pose6=target_pose,
            options=options,
        ),
    )


def solve_numeric_goal_multistart(
    *,
    method_id: str,
    label: str,
    target_pose: np.ndarray,
    initial_guesses: list[np.ndarray],
    dedupe_tol_deg: float,
    solver_func,
) -> dict:
    t0 = time.perf_counter()
    goal_candidates: list[dict] = []
    unique_refined_solutions: list[np.ndarray] = []
    evaluated_starts = 0

    for start_index, guess in enumerate(initial_guesses, start=1):
        out = solver_func(guess)
        evaluated_starts += 1
        q_goal_deg = np.asarray(out["q_deg"], dtype=float).reshape(6)
        if is_duplicate_solution(q_deg=q_goal_deg, existing=unique_refined_solutions, atol_deg=dedupe_tol_deg):
            continue
        unique_refined_solutions.append(q_goal_deg.copy())
        goal_candidates.append(
            {
                "candidate_rank": int(start_index),
                "subspace_id": None,
                "q0_deg": [float(x) for x in np.asarray(guess, dtype=float).tolist()],
                "q_goal_deg": [float(x) for x in q_goal_deg.tolist()],
                "converged": bool(out["converged"]),
                "iters": int(out["iters"]),
                "final_pose6": out["final_pose6"],
                "final_pos_err_mm": float(out["final_pos_err_mm"]),
                "final_ori_err_rad": float(out["final_ori_err_rad"]),
                "weighted_cost": float(out.get("weighted_cost", float("nan"))),
                "solver_output": {
                    key: value
                    for key, value in out.items()
                    if key not in {"q_deg", "converged", "iters", "final_pose6", "final_pos_err_mm", "final_ori_err_rad"}
                },
            }
        )

    if not goal_candidates:
        raise RuntimeError(f"{label} 未生成任何可评估的终点候选。")

    total_ms = float((time.perf_counter() - t0) * 1000.0)
    best_initial = min(
        goal_candidates,
        key=lambda item: (
            0 if item["converged"] else 1,
            item["weighted_cost"],
            item["final_pos_err_mm"],
            item["final_ori_err_rad"],
        ),
    )
    return {
        "method_id": method_id,
        "label": label,
        "solver_family": "numeric",
        "ik_time_ms": total_ms,
        "planning_time_ms": total_ms,
        "q0_deg": best_initial["q0_deg"],
        "q_goal_deg": best_initial["q_goal_deg"],
        "converged": bool(best_initial["converged"]),
        "iters": int(best_initial["iters"]),
        "final_pose6": best_initial["final_pose6"],
        "final_pos_err_mm": float(best_initial["final_pos_err_mm"]),
        "final_ori_err_rad": float(best_initial["final_ori_err_rad"]),
        "initial_guess_count": int(evaluated_starts),
        "unique_goal_candidate_count": int(len(goal_candidates)),
        "initial_guess_library": [[float(x) for x in guess.tolist()] for guess in initial_guesses],
        "goal_candidates": goal_candidates,
    }


def build_method_record(
    *,
    method: dict,
    q_start_deg: np.ndarray,
    scene: ObstacleScene,
    selection_weights: TrajectorySelectionWeights,
    trajectory_steps: int,
    include_frames: bool,
) -> dict:
    t0 = time.perf_counter()
    variants: list[dict] = []

    goal_candidates = method.get("goal_candidates")
    if goal_candidates:
        candidate_items: Iterable[dict] = goal_candidates
    else:
        candidate_items = [
            {
                "candidate_rank": 1,
                "subspace_id": method.get("subspace_id"),
                "q0_deg": method["q0_deg"],
                "q_goal_deg": method["q_goal_deg"],
                "converged": method["converged"],
                "iters": method["iters"],
                "final_pose6": method["final_pose6"],
                "final_pos_err_mm": method["final_pos_err_mm"],
                "final_ori_err_rad": method["final_ori_err_rad"],
            }
        ]

    for candidate in candidate_items:
        q_goal_deg = np.asarray(candidate["q_goal_deg"], dtype=float).reshape(6)
        for traj_variant in build_waypoint_joint_trajectory_candidates_deg(q_start_deg=q_start_deg, q_goal_deg=q_goal_deg):
            q_points_deg = np.asarray(traj_variant["q_points_deg"], dtype=float)
            q_traj_deg = build_piecewise_joint_trajectory_deg(q_points_deg=q_points_deg, steps=trajectory_steps)
            trajectory_summary = evaluate_joint_trajectory_against_scene(
                q_traj_deg=q_traj_deg,
                scene=scene,
                include_frames=False,
            )
            selection_cost = compute_selection_cost(
                final_pos_err_mm=float(candidate["final_pos_err_mm"]),
                final_ori_err_rad=float(candidate["final_ori_err_rad"]),
                trajectory_summary=trajectory_summary,
                weights=selection_weights,
            )
            rank_summary = summarize_candidate_rank(
                final_pos_err_mm=float(candidate["final_pos_err_mm"]),
                final_ori_err_rad=float(candidate["final_ori_err_rad"]),
                trajectory_summary=trajectory_summary,
                selection_cost=selection_cost,
                weights=selection_weights,
            )
            variants.append(
                {
                    "candidate_rank": int(candidate.get("candidate_rank", 1)),
                    "subspace_id": candidate.get("subspace_id"),
                    "q0_deg": candidate["q0_deg"],
                    "q_goal_deg": candidate["q_goal_deg"],
                    "nr_converged": bool(candidate["converged"]),
                    "nr_iters": int(candidate["iters"]),
                    "final_pose6": candidate["final_pose6"],
                    "final_pos_err_mm": float(candidate["final_pos_err_mm"]),
                    "final_ori_err_rad": float(candidate["final_ori_err_rad"]),
                    "trajectory_mode": str(traj_variant["trajectory_mode"]),
                    "trajectory_waypoint_deg": traj_variant["waypoint_deg"],
                    "trajectory_points_deg": q_points_deg.tolist(),
                    "trajectory_summary": trajectory_summary,
                    "selection": rank_summary,
                }
            )
    variants.sort(
        key=lambda item: (
            item["selection"]["rank_key"][0],
            item["selection"]["rank_key"][1],
            item["selection"]["rank_key"][2],
            item["selection"]["rank_key"][3],
            item["selection"]["rank_key"][4],
        )
    )
    if not variants:
        raise RuntimeError(f"Method {method['label']} produced no trajectory variants.")
    selected_solution = dict(variants[0])
    if include_frames:
        selected_points = np.asarray(selected_solution["trajectory_points_deg"], dtype=float)
        selected_traj = build_piecewise_joint_trajectory_deg(q_points_deg=selected_points, steps=trajectory_steps)
        selected_solution["trajectory_summary"] = evaluate_joint_trajectory_against_scene(
            q_traj_deg=selected_traj,
            scene=scene,
            include_frames=True,
        )

    selected_summary = selected_solution["trajectory_summary"]
    total_planning_ms = float(method["planning_time_ms"]) + float((time.perf_counter() - t0) * 1000.0)
    return {
        "method_id": method["method_id"],
        "label": method["label"],
        "solver_family": method["solver_family"],
        "planning_time_ms": total_planning_ms,
        "ik_time_ms": float(method["ik_time_ms"]),
        "final_pos_err_mm": float(selected_solution["final_pos_err_mm"]),
        "final_ori_err_rad": float(selected_solution["final_ori_err_rad"]),
        "converged": bool(selected_solution["nr_converged"]),
        "iters": int(selected_solution["nr_iters"]),
        "initial_guess_count": int(method.get("initial_guess_count", 1)),
        "unique_goal_candidate_count": int(method.get("unique_goal_candidate_count", 1)),
        "trajectory_mode": str(selected_solution["trajectory_mode"]),
        "trajectory_waypoint_deg": selected_solution.get("trajectory_waypoint_deg"),
        "selected_solution_collision_free": bool(not selected_summary["collision"]),
        "collision_frame_count": int(selected_summary["collision_frame_count"]),
        "min_clearance_mm": float(selected_summary["min_clearance_mm"]),
        "joint_path_length_deg": float(selected_summary["joint_path_length_deg"]),
        "max_joint_step_deg": float(selected_summary["max_joint_step_deg"]),
        "selected_solution": {
            "candidate_rank": int(selected_solution.get("candidate_rank", 1)),
            "subspace_id": selected_solution.get("subspace_id"),
            "q0_deg": selected_solution["q0_deg"],
            "q_goal_deg": selected_solution["q_goal_deg"],
            "final_pose6": selected_solution["final_pose6"],
            "trajectory_mode": str(selected_solution["trajectory_mode"]),
            "trajectory_waypoint_deg": selected_solution.get("trajectory_waypoint_deg"),
            "trajectory_points_deg": selected_solution["trajectory_points_deg"],
            "trajectory_summary": selected_summary,
        },
        "evaluated_trajectory_variants": variants,
        "solver_details": {
            key: value
            for key, value in method.items()
            if key
            not in {
                "method_id",
                "label",
                "solver_family",
                "planning_time_ms",
                "ik_time_ms",
                "q0_deg",
                "q_goal_deg",
                "goal_candidates",
                "initial_guess_count",
                "unique_goal_candidate_count",
                "initial_guess_library",
                "converged",
                "iters",
                "final_pose6",
                "final_pos_err_mm",
                "final_ori_err_rad",
            }
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    target_pose = parse_pose(args.pose)
    q_start_deg = parse_q_deg(args.q_start)
    scene = ObstacleScene.from_json(args.scene_json)

    nr_options = NROptions(
        max_iters=args.nr_max_iters,
        tol_pos_mm=args.nr_tol_pos_mm,
        tol_ori_rad=args.nr_tol_ori_rad,
        damping=args.nr_damping,
        step_scale=args.nr_step_scale,
    )
    dls_options = DLSOptions(
        max_iters=args.dls_max_iters,
        tol_pos_mm=args.dls_tol_pos_mm,
        tol_ori_rad=args.dls_tol_ori_rad,
        damping=args.dls_damping,
        orientation_weight=args.dls_orientation_weight,
    )
    dls_options.multistart_guesses = int(args.dls_multistart_guesses)
    dls_options.dedupe_tol_deg = float(args.dedupe_tol_deg)
    lbfgsb_options = LBFGSBOptions(
        max_iters=args.lbfgsb_max_iters,
        tol_pos_mm=args.lbfgsb_tol_pos_mm,
        tol_ori_rad=args.lbfgsb_tol_ori_rad,
        orientation_weight=args.lbfgsb_orientation_weight,
    )
    lbfgsb_options.multistart_guesses = int(args.lbfgsb_multistart_guesses)
    lbfgsb_options.dedupe_tol_deg = float(args.dedupe_tol_deg)
    selection_weights = TrajectorySelectionWeights(
        collision_flag_weight=args.cost_collision_flag_weight,
        collision_frame_weight=args.cost_collision_frame_weight,
        collision_violation_weight=args.cost_collision_violation_weight,
        accuracy_violation_weight=args.cost_accuracy_violation_weight,
        joint_path_weight=args.cost_joint_path_weight,
        max_joint_step_weight=args.cost_max_joint_step_weight,
        clearance_reward_weight=args.cost_clearance_reward_weight,
        clearance_reward_cap_mm=args.cost_clearance_reward_cap_mm,
        pos_tol_mm=args.selection_pos_tol_mm,
        ori_tol_rad=args.selection_ori_tol_rad,
    )

    raw_methods = [
        solve_nn_nr_goal(
            target_pose=target_pose,
            pred_meta_path=Path(args.pred_meta),
            branch_meta_path=Path(args.branch_meta),
            fine_meta_path=Path(args.fine_meta),
            topk_shoulder=args.topk_shoulder,
            topk_elbow=args.topk_elbow,
            topk_wrist=args.topk_wrist,
            max_branch_candidates=args.max_branch_candidates,
            fine_topk_per_branch=args.fine_topk_per_branch,
            max_subspace_candidates=args.max_subspace_candidates,
            dedupe_tol_deg=args.dedupe_tol_deg,
            nr_options=nr_options,
        ),
        solve_dls_goal(target_pose=target_pose, q_start_deg=q_start_deg, options=dls_options),
        solve_lbfgsb_goal(target_pose=target_pose, q_start_deg=q_start_deg, options=lbfgsb_options),
    ]

    methods = [
        build_method_record(
            method=item,
            q_start_deg=q_start_deg,
            scene=scene,
            selection_weights=selection_weights,
            trajectory_steps=args.trajectory_steps,
            include_frames=args.save_selected_frames,
        )
        for item in raw_methods
    ]
    ranking = sorted(
        [
            {
                "method_id": item["method_id"],
                "label": item["label"],
                "selected_solution_collision_free": item["selected_solution_collision_free"],
                "selection_cost": float(item["selected_solution"]["trajectory_summary"].get("selection_cost", item["evaluated_trajectory_variants"][0]["selection"]["selection_cost"])),
                "planning_time_ms": float(item["planning_time_ms"]),
                "min_clearance_mm": float(item["min_clearance_mm"]),
            }
            for item in methods
        ],
        key=lambda item: (
            0 if item["selected_solution_collision_free"] else 1,
            -item["min_clearance_mm"],
            item["planning_time_ms"],
        ),
    )

    result = {
        "schema": "abb_obstacle_method_compare_v1",
        "comparison_name": args.comparison_name,
        "target_pose6": target_pose.tolist(),
        "q_start_deg": q_start_deg.tolist(),
        "scene_name": scene.scene_name,
        "scene": scene.to_dict(),
        "trajectory_steps": int(args.trajectory_steps),
        "methods": methods,
        "selected_method_ranking": ranking,
    }
    out_path = Path(args.out_json)
    save_json(out_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
