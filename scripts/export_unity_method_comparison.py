#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Sequence

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
from fk_model import JOINT_LIMITS_DEG, fk_abb_irb, fk_abb_irb_joint_points
from predict_ik import (
    apply_normalizer,
    generate_flat_candidates,
    generate_hierarchical_candidates,
    load_json,
    load_prediction_pair,
    position_l2_norm,
    predict_q_deg,
    safe_torch_load,
)


def parse_pose(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 6:
        raise ValueError("--pose must be x_mm,y_mm,z_mm,phi_rad,theta_rad,psi_rad")
    return np.asarray(values, dtype=np.float32)


def parse_q_deg(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 6:
        raise ValueError("--q_start must contain 6 joint angles in degrees.")
    q_deg = np.asarray(values, dtype=float)
    return np.clip(q_deg, JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1])


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


def build_frames(q_start_deg: np.ndarray, q_goal_deg: np.ndarray, steps: int) -> list[dict]:
    frames: list[dict] = []
    total_steps = max(2, int(steps))

    for frame_idx in range(total_steps):
        t = frame_idx / float(total_steps - 1)
        q_deg = (1.0 - t) * q_start_deg + t * q_goal_deg
        _, p_mm, r06 = fk_abb_irb(q_deg, input_unit="deg")
        joint_points_mm = fk_abb_irb_joint_points(q_deg, input_unit="deg")
        unity_pos_m = python_mm_to_unity_m(p_mm)
        unity_rot = python_rotation_to_unity_rotation(r06)

        frames.append(
            {
                "index": int(frame_idx),
                "t": float(t),
                "q_deg": q_deg.tolist(),
                "python_tool_position_mm": p_mm.tolist(),
                "unity_expected_tool_world_position_m": unity_pos_m.tolist(),
                "python_tool_rotation_matrix": r06.reshape(-1).tolist(),
                "unity_expected_tool_world_rotation_matrix": unity_rot.reshape(-1).tolist(),
                "joint_points_mm": joint_points_mm.tolist(),
            }
        )

    return frames


def choose_prediction_initial_solution(
    target_pose: np.ndarray,
    x_pred: np.ndarray,
    pred_meta_path: Path,
    pred_meta: dict,
    candidate_labels: Sequence[int],
    force_all_subspaces: bool,
) -> tuple[dict, list[int], str, bool]:
    model_index: Dict[int, Dict[str, object]] = {
        int(item["subspace_id"]): item for item in pred_meta["trained_subspaces"]
    }
    trained_all = sorted(model_index.keys())
    candidate_source = "hierarchical_predictions"

    if force_all_subspaces:
        candidate_labels = trained_all
        candidate_source = "force_all_subspaces"
    else:
        available_candidates = [int(sid) for sid in candidate_labels if int(sid) in model_index]
        if not available_candidates:
            candidate_labels = trained_all
            candidate_source = "hierarchical_empty_fallback_to_trained"
        else:
            candidate_labels = available_candidates

    best = None
    for sid in candidate_labels:
        ckpt = safe_torch_load(pred_meta_path.parent / "subspace_models" / model_index[int(sid)]["model_file"])
        m15, m6 = load_prediction_pair(ckpt)
        q0 = predict_q_deg(m15, m6, x_pred)
        l2 = position_l2_norm(q0, target_pose)
        item = {
            "subspace_id": int(sid),
            "q0_deg": q0.tolist(),
            "position_l2_mm": float(l2),
            "e_max": float(ckpt.get("e_max", np.inf)),
        }
        if (best is None) or (item["position_l2_mm"] < best["position_l2_mm"]):
            best = item

    if best is None:
        raise RuntimeError("No candidate subspace has trained model.")

    fallback_triggered = False
    if best["position_l2_mm"] > best["e_max"]:
        fallback_triggered = True
        fallback_best = None
        for sid in trained_all:
            ckpt = safe_torch_load(pred_meta_path.parent / "subspace_models" / model_index[int(sid)]["model_file"])
            m15, m6 = load_prediction_pair(ckpt)
            q0 = predict_q_deg(m15, m6, x_pred)
            l2 = position_l2_norm(q0, target_pose)
            item = {
                "subspace_id": int(sid),
                "q0_deg": q0.tolist(),
                "position_l2_mm": float(l2),
                "e_max": float(ckpt.get("e_max", np.inf)),
            }
            if (fallback_best is None) or (item["position_l2_mm"] < fallback_best["position_l2_mm"]):
                fallback_best = item
        if fallback_best is not None:
            best = fallback_best

    return best, list(candidate_labels), candidate_source, fallback_triggered


def solve_nn_nr(
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

    candidate_labels, candidate_generation_info, _ = generate_hierarchical_candidates(
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

    initial, candidate_labels, candidate_source, fallback_triggered = choose_prediction_initial_solution(
        target_pose=target_pose,
        x_pred=x_pred,
        pred_meta_path=pred_meta_path,
        pred_meta=pred_meta,
        candidate_labels=candidate_labels,
        force_all_subspaces=False,
    )

    nr = newton_raphson_refine(
        q0_deg=initial["q0_deg"],
        target_pose6=target_pose,
        options=nr_options,
    )
    q_goal_deg = np.asarray(nr["q_deg"], dtype=float)
    metrics = evaluate_solution_metrics(q_goal_deg, target_pose)

    return {
        "method_id": "nn_nr",
        "label": "NN + NR",
        "solve_time_ms": float((time.perf_counter() - t0) * 1000.0),
        "q_goal_deg": q_goal_deg.tolist(),
        "converged": bool(nr["converged"]),
        "iters": int(nr["iters"]),
        "final_pose6": metrics["final_pose6"],
        "final_pos_err_mm": float(metrics["final_pos_err_mm"]),
        "final_ori_err_rad": float(metrics["final_ori_err_rad"]),
        "candidate_source": candidate_source,
        "candidate_subspaces": [int(x) for x in candidate_labels],
        "candidate_generation": candidate_generation_info,
        "initial_solution": initial,
        "fallback_full_scan_triggered": bool(fallback_triggered),
    }


def solve_dls(target_pose: np.ndarray, q_start_deg: np.ndarray, options: DLSOptions) -> dict:
    t0 = time.perf_counter()
    out = dls_refine(
        q0_deg=q_start_deg,
        target_pose6=target_pose,
        options=options,
    )
    q_goal_deg = np.asarray(out["q_deg"], dtype=float)
    return {
        "method_id": "dls",
        "label": "DLS",
        "solve_time_ms": float((time.perf_counter() - t0) * 1000.0),
        "q_goal_deg": q_goal_deg.tolist(),
        "converged": bool(out["converged"]),
        "iters": int(out["iters"]),
        "final_pose6": out["final_pose6"],
        "final_pos_err_mm": float(out["final_pos_err_mm"]),
        "final_ori_err_rad": float(out["final_ori_err_rad"]),
        "start_q_deg": [float(x) for x in q_start_deg.tolist()],
        "weighted_cost": float(out.get("weighted_cost", float("nan"))),
    }


def solve_lbfgsb(target_pose: np.ndarray, q_start_deg: np.ndarray, options: LBFGSBOptions) -> dict:
    t0 = time.perf_counter()
    out = lbfgsb_refine(
        q0_deg=q_start_deg,
        target_pose6=target_pose,
        options=options,
    )
    q_goal_deg = np.asarray(out["q_deg"], dtype=float)
    return {
        "method_id": "lbfgsb",
        "label": "L-BFGS-B",
        "solve_time_ms": float((time.perf_counter() - t0) * 1000.0),
        "q_goal_deg": q_goal_deg.tolist(),
        "converged": bool(out["converged"]),
        "iters": int(out["iters"]),
        "final_pose6": out["final_pose6"],
        "final_pos_err_mm": float(out["final_pos_err_mm"]),
        "final_ori_err_rad": float(out["final_ori_err_rad"]),
        "start_q_deg": [float(x) for x in q_start_deg.tolist()],
        "weighted_cost": float(out.get("weighted_cost", float("nan"))),
        "eval_count": int(out.get("eval_count", 0)),
        "optimizer_success": bool(out.get("optimizer_success", False)),
        "optimizer_message": str(out.get("optimizer_message", "")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ABB_IRB multi-method IK trajectories for Unity playback."
    )
    parser.add_argument("--pose", required=True, help="x_mm,y_mm,z_mm,phi_rad,theta_rad,psi_rad")
    parser.add_argument("--q_start", default="0,0,0,0,0,0", help="Six comma-separated start joint angles in degrees.")
    parser.add_argument("--steps", type=int, default=120, help="Number of playback frames.")
    parser.add_argument("--duration", type=float, default=3.0, help="Suggested playback duration in seconds.")
    parser.add_argument("--name", default="abb_ik_method_compare", help="Comparison trajectory name stored in JSON.")
    parser.add_argument("--pred_meta", default="artifacts/prediction_system_formal/metadata.json")
    parser.add_argument("--branch_meta", default="artifacts/branch_classification_system/metadata.json")
    parser.add_argument("--fine_meta", default="artifacts/fine_classification_system/metadata.json")
    parser.add_argument("--topk_shoulder", type=int, default=2)
    parser.add_argument("--topk_elbow", type=int, default=1)
    parser.add_argument("--topk_wrist", type=int, default=2)
    parser.add_argument("--max_branch_candidates", type=int, default=4)
    parser.add_argument("--fine_topk_per_branch", type=int, default=3)
    parser.add_argument("--max_subspace_candidates", type=int, default=15)
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
    parser.add_argument("--lbfgsb_max_iters", type=int, default=200)
    parser.add_argument("--lbfgsb_tol_pos_mm", type=float, default=1.0)
    parser.add_argument("--lbfgsb_tol_ori_rad", type=float, default=1e-2)
    parser.add_argument("--lbfgsb_orientation_weight", type=float, default=200.0)
    parser.add_argument("--out_json", required=True, help="Output JSON path.")
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    target_pose = parse_pose(args.pose)
    q_start_deg = parse_q_deg(args.q_start)
    steps = max(2, int(args.steps))
    duration = max(0.001, float(args.duration))

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
    lbfgsb_options = LBFGSBOptions(
        max_iters=args.lbfgsb_max_iters,
        tol_pos_mm=args.lbfgsb_tol_pos_mm,
        tol_ori_rad=args.lbfgsb_tol_ori_rad,
        orientation_weight=args.lbfgsb_orientation_weight,
    )

    nn_nr = solve_nn_nr(
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
        nr_options=nr_options,
    )
    dls = solve_dls(target_pose=target_pose, q_start_deg=q_start_deg, options=dls_options)
    lbfgsb = solve_lbfgsb(target_pose=target_pose, q_start_deg=q_start_deg, options=lbfgsb_options)

    methods = []
    for item in [nn_nr, dls, lbfgsb]:
        q_goal_deg = np.asarray(item["q_goal_deg"], dtype=float)
        method_payload = dict(item)
        method_payload["q_start_deg"] = q_start_deg.tolist()
        method_payload["frames"] = build_frames(q_start_deg=q_start_deg, q_goal_deg=q_goal_deg, steps=steps)
        methods.append(method_payload)

    target_unity_world_position_m = python_mm_to_unity_m(target_pose[:3])
    result = {
        "schema": "abb_unity_ik_method_compare_v1",
        "robot_name": "ABB_IRB",
        "comparison_name": args.name,
        "trajectory_mode": "joint_linear_interpolation",
        "playback_duration_seconds": duration,
        "trajectory_steps": steps,
        "target_pose6": target_pose.tolist(),
        "target_unity_world_position_m": target_unity_world_position_m.tolist(),
        "q_start_deg": q_start_deg.tolist(),
        "unity_position_mapping": "Unity(m) = [-Python_y, Python_z, Python_x] / 1000",
        "unity_rotation_mapping": "R_unity = S * R_python * S^T, S=[[0,-1,0],[0,0,1],[1,0,0]]",
        "methods": methods,
    }

    out_path = Path(args.out_json)
    save_json(out_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
