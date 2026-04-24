#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from abb_nn.optimization import NROptions, newton_raphson_refine
from obstacle_avoidance.collision import ObstacleScene
from obstacle_avoidance.planning import (
    TrajectorySelectionWeights,
    compute_selection_cost,
    evaluate_trajectory_against_scene,
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
from abb_nn.optimization import evaluate_solution_metrics


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


def build_model_cache(pred_meta_path: Path, pred_meta: dict) -> tuple[dict[int, dict], dict[int, dict]]:
    model_index: Dict[int, Dict[str, object]] = {
        int(item["subspace_id"]): item for item in pred_meta["trained_subspaces"]
    }
    model_cache: dict[int, dict] = {}
    for sid, item in model_index.items():
        ckpt = safe_torch_load(pred_meta_path.parent / "subspace_models" / item["model_file"])
        m15, m6 = load_prediction_pair(ckpt)
        model_cache[int(sid)] = {
            "m15": m15,
            "m6": m6,
            "e_max": float(ckpt.get("e_max", np.inf)),
        }
    return model_index, model_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan collision-aware IK for ABB_IRB in a fixed AABB scene.")
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
    parser.add_argument("--max_evaluated_candidates", type=int, default=18)
    parser.add_argument("--nr_max_iters", type=int, default=40)
    parser.add_argument("--nr_tol_pos_mm", type=float, default=1e-3)
    parser.add_argument("--nr_tol_ori_rad", type=float, default=1e-3)
    parser.add_argument("--nr_damping", type=float, default=1e-5)
    parser.add_argument("--nr_step_scale", type=float, default=1.0)
    parser.add_argument("--trajectory_steps", type=int, default=120)
    parser.add_argument("--save_selected_frames", action="store_true", help="Store per-frame data for the selected solution.")
    parser.add_argument("--dedupe_tol_deg", type=float, default=0.5, help="Treat refined solutions within this joint-angle tolerance as duplicates.")
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
    parser.add_argument("--out_json", default="", help="Optional output JSON path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    t_total_start = time.perf_counter()

    target_pose = parse_pose(args.pose)
    q_start_deg = parse_q_deg(args.q_start)
    scene = ObstacleScene.from_json(args.scene_json)

    pred_meta_path = Path(args.pred_meta)
    branch_meta_path = Path(args.branch_meta)
    fine_meta_path = Path(args.fine_meta)
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
        args.topk_shoulder,
        args.topk_elbow,
        args.topk_wrist,
        args.max_branch_candidates,
        args.fine_topk_per_branch,
        args.max_subspace_candidates,
    )

    model_index, model_cache = build_model_cache(pred_meta_path=pred_meta_path, pred_meta=pred_meta)
    candidate_labels = [int(sid) for sid in candidate_labels if int(sid) in model_index]
    candidate_labels = candidate_labels[: max(1, int(args.max_evaluated_candidates))]
    if not candidate_labels:
        raise RuntimeError("No trained candidate subspaces available for collision-aware planning.")

    nr_options = NROptions(
        max_iters=args.nr_max_iters,
        tol_pos_mm=args.nr_tol_pos_mm,
        tol_ori_rad=args.nr_tol_ori_rad,
        damping=args.nr_damping,
        step_scale=args.nr_step_scale,
    )
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

    t_eval_start = time.perf_counter()
    evaluated_candidates: list[dict] = []
    unique_refined_solutions: list[np.ndarray] = []

    for rank_index, sid in enumerate(candidate_labels, start=1):
        model_item = model_cache[int(sid)]
        q0_deg = predict_q_deg(model_item["m15"], model_item["m6"], x_pred)
        initial_pos_err_mm = position_l2_norm(q0_deg, target_pose)

        nr = newton_raphson_refine(
            q0_deg=q0_deg,
            target_pose6=target_pose,
            options=nr_options,
        )
        q_goal_deg = np.asarray(nr["q_deg"], dtype=float).reshape(6)

        if is_duplicate_solution(q_deg=q_goal_deg, existing=unique_refined_solutions, atol_deg=args.dedupe_tol_deg):
            continue
        unique_refined_solutions.append(q_goal_deg.copy())

        metrics = evaluate_solution_metrics(q_goal_deg, target_pose)
        trajectory_summary = evaluate_trajectory_against_scene(
            q_start_deg=q_start_deg,
            q_goal_deg=q_goal_deg,
            scene=scene,
            steps=args.trajectory_steps,
            include_frames=False,
        )
        selection_cost = compute_selection_cost(
            final_pos_err_mm=float(metrics["final_pos_err_mm"]),
            final_ori_err_rad=float(metrics["final_ori_err_rad"]),
            trajectory_summary=trajectory_summary,
            weights=selection_weights,
        )
        rank_summary = summarize_candidate_rank(
            final_pos_err_mm=float(metrics["final_pos_err_mm"]),
            final_ori_err_rad=float(metrics["final_ori_err_rad"]),
            trajectory_summary=trajectory_summary,
            selection_cost=selection_cost,
            weights=selection_weights,
        )

        evaluated_candidates.append(
            {
                "candidate_rank": int(rank_index),
                "subspace_id": int(sid),
                "q0_deg": q0_deg.tolist(),
                "q_goal_deg": q_goal_deg.tolist(),
                "initial_pos_err_mm": float(initial_pos_err_mm),
                "initial_e_max_mm": float(model_item["e_max"]),
                "nr_iters": int(nr["iters"]),
                "nr_converged": bool(nr["converged"]),
                "final_pose6": metrics["final_pose6"],
                "final_pos_err_mm": float(metrics["final_pos_err_mm"]),
                "final_ori_err_rad": float(metrics["final_ori_err_rad"]),
                "trajectory_summary": trajectory_summary,
                "selection": rank_summary,
            }
        )

    if not evaluated_candidates:
        raise RuntimeError("All evaluated candidates collapsed to duplicates; no candidate remained for planning.")

    evaluated_candidates.sort(
        key=lambda item: (
            item["selection"]["rank_key"][0],
            item["selection"]["rank_key"][1],
            item["selection"]["rank_key"][2],
            item["selection"]["rank_key"][3],
            item["selection"]["rank_key"][4],
        )
    )

    selected_solution = dict(evaluated_candidates[0])
    if args.save_selected_frames:
        selected_solution["trajectory_summary"] = evaluate_trajectory_against_scene(
            q_start_deg=q_start_deg,
            q_goal_deg=np.asarray(selected_solution["q_goal_deg"], dtype=float),
            scene=scene,
            steps=args.trajectory_steps,
            include_frames=True,
        )

    t_total_ms = (time.perf_counter() - t_total_start) * 1000.0
    t_eval_ms = (time.perf_counter() - t_eval_start) * 1000.0

    result = {
        "planner_name": "collision_aware_nn_nr_selector",
        "target_pose6": target_pose.tolist(),
        "q_start_deg": q_start_deg.tolist(),
        "scene": scene.to_dict(),
        "candidate_generation": candidate_generation_info,
        "candidate_subspaces": candidate_labels,
        "selection_weights": {
            "collision_flag_weight": float(selection_weights.collision_flag_weight),
            "collision_frame_weight": float(selection_weights.collision_frame_weight),
            "collision_violation_weight": float(selection_weights.collision_violation_weight),
            "accuracy_violation_weight": float(selection_weights.accuracy_violation_weight),
            "joint_path_weight": float(selection_weights.joint_path_weight),
            "max_joint_step_weight": float(selection_weights.max_joint_step_weight),
            "clearance_reward_weight": float(selection_weights.clearance_reward_weight),
            "clearance_reward_cap_mm": float(selection_weights.clearance_reward_cap_mm),
            "pos_tol_mm": float(selection_weights.pos_tol_mm),
            "ori_tol_rad": float(selection_weights.ori_tol_rad),
        },
        "evaluated_candidates": evaluated_candidates,
        "selected_solution": selected_solution,
        "planning_time_ms": float(t_total_ms),
        "timing_breakdown_ms": {
            "candidate_generation_ms": float(timing_generation["candidate_generation_ms"]),
            "classification_ms": float(timing_generation["classification_ms"]),
            "branch_classification_ms": float(timing_generation["branch_classification_ms"]),
            "fine_classification_ms": float(timing_generation["fine_classification_ms"]),
            "candidate_evaluation_ms": float(t_eval_ms),
            "total_ms": float(t_total_ms),
        },
    }

    if args.out_json.strip():
        save_json(Path(args.out_json), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
