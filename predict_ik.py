#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Windows + conda local workaround for duplicate OpenMP runtime initialization
# observed during lightweight inference in this environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from abb_nn.models import MLPRegressor, build_classifier_variant
from abb_nn.optimization import NROptions, newton_raphson_refine
from fk_model import JOINT_LIMITS_DEG, pose6_from_q


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def safe_torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_pose(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 6:
        raise ValueError("--pose must be x_mm,y_mm,z_mm,phi_rad,theta_rad,psi_rad")
    return np.array(vals, dtype=np.float32).reshape(1, 6)


def apply_normalizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def load_prediction_pair(ckpt: dict) -> tuple[MLPRegressor, MLPRegressor]:
    m15 = MLPRegressor(input_dim=6, output_dim=5, hidden_dims=ckpt["hidden_dims_q15"])
    m6 = MLPRegressor(input_dim=6, output_dim=1, hidden_dims=ckpt["hidden_dims_q6"])
    m15.load_state_dict(ckpt["state_q15"])
    m6.load_state_dict(ckpt["state_q6"])
    m15.eval()
    m6.eval()
    return m15, m6


def predict_q_deg(m15: MLPRegressor, m6: MLPRegressor, x_norm: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xt = torch.from_numpy(x_norm.astype(np.float32))
        q15 = m15(xt).numpy().reshape(-1)
        q6 = m6(xt).numpy().reshape(-1)
    q = np.concatenate([q15, q6], axis=0)
    return np.clip(q, JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1])


def position_l2_norm(q_deg: np.ndarray, target_pose6: np.ndarray) -> float:
    pose_pred = pose6_from_q(q_deg, input_unit="deg")
    return float(np.linalg.norm(pose_pred[:3] - target_pose6[:3]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABB_IRB IK inference.")
    parser.add_argument("--pose", type=str, required=True, help="x_mm,y_mm,z_mm,phi_rad,theta_rad,psi_rad")
    parser.add_argument("--pred_meta", type=str, default="artifacts/prediction_system/metadata.json")
    parser.add_argument("--cls_meta", type=str, default="artifacts/classification_system/metadata.json")
    parser.add_argument(
        "--cls_topk",
        type=int,
        default=1,
        help="Use top-k predicted subspaces from each classifier as candidates.",
    )
    parser.add_argument("--enable_nr", action="store_true", help="Apply Newton-Raphson refinement.")
    parser.add_argument("--nr_max_iters", type=int, default=40)
    parser.add_argument("--nr_tol_pos_mm", type=float, default=1e-3)
    parser.add_argument("--nr_tol_ori_rad", type=float, default=1e-3)
    parser.add_argument("--nr_damping", type=float, default=1e-5)
    parser.add_argument("--nr_step_scale", type=float, default=1.0)
    parser.add_argument("--force_all_subspaces", action="store_true")
    parser.add_argument("--out_json", type=str, default="", help="Optional path to save inference result JSON.")
    return parser.parse_args()


def main() -> None:
    t0_total = time.perf_counter()
    args = parse_args()
    target_pose = parse_pose(args.pose).reshape(-1)

    pred_meta_path = Path(args.pred_meta)
    cls_meta_path = Path(args.cls_meta)
    pred_meta = load_json(pred_meta_path)
    cls_meta = load_json(cls_meta_path)
    if args.cls_topk <= 0:
        raise ValueError("--cls_topk must be >= 1.")
    pred_profile = pred_meta.get("segment_profile", "abb_strict")
    cls_profile = cls_meta.get("segment_profile", "abb_strict")
    if pred_profile != cls_profile:
        raise ValueError(
            f"Segment profile mismatch: prediction={pred_profile}, classification={cls_profile}. "
            "Please use matched artifacts."
        )

    pred_mean = np.array(pred_meta["normalizer"]["mean"], dtype=np.float32).reshape(1, -1)
    pred_std = np.array(pred_meta["normalizer"]["std"], dtype=np.float32).reshape(1, -1)
    cls_mean = np.array(cls_meta["normalizer"]["mean"], dtype=np.float32).reshape(1, -1)
    cls_std = np.array(cls_meta["normalizer"]["std"], dtype=np.float32).reshape(1, -1)

    x_pred = apply_normalizer(target_pose.reshape(1, -1).astype(np.float32), pred_mean, pred_std)
    x_cls = apply_normalizer(target_pose.reshape(1, -1).astype(np.float32), cls_mean, cls_std)

    t_cls_start = time.perf_counter()
    candidate_labels: List[int] = []
    for item in cls_meta["models"]:
        ckpt = safe_torch_load(cls_meta_path.parent / item["file"])
        num_classes = int(ckpt["num_classes"])
        cls = build_classifier_variant(
            variant=int(item["variant"]),
            input_dim=6,
            num_classes=num_classes,
        )
        cls.load_state_dict(ckpt["state_dict"])
        cls.eval()
        with torch.no_grad():
            logits = cls(torch.from_numpy(x_cls))
            topk = min(args.cls_topk, num_classes)
            pred_ids = torch.topk(logits, k=topk, dim=1).indices.reshape(-1).tolist()
        candidate_labels.extend(int(x) for x in pred_ids)
    candidate_labels = sorted(set(candidate_labels))
    t_cls_end = time.perf_counter()

    model_index: Dict[int, Dict[str, object]] = {
        int(x["subspace_id"]): x for x in pred_meta["trained_subspaces"]
    }
    trained_all = sorted(model_index.keys())
    candidate_source = "classification_predictions"

    if args.force_all_subspaces:
        candidate_labels = trained_all
        candidate_source = "force_all_subspaces"
    else:
        available_candidates = [sid for sid in candidate_labels if sid in model_index]
        if not available_candidates:
            candidate_labels = trained_all
            candidate_source = "classification_empty_fallback_to_trained"
        else:
            candidate_labels = available_candidates

    t_init_start = time.perf_counter()
    best = None
    for sid in candidate_labels:
        if sid not in model_index:
            continue
        ck = safe_torch_load(pred_meta_path.parent / "subspace_models" / model_index[sid]["model_file"])
        m15, m6 = load_prediction_pair(ck)
        q0 = predict_q_deg(m15, m6, x_pred)
        l2 = position_l2_norm(q0, target_pose)
        item = {
            "subspace_id": sid,
            "q0_deg": q0.tolist(),
            "position_l2_mm": l2,
            "e_max": float(ck.get("e_max", np.inf)),
        }
        if (best is None) or (item["position_l2_mm"] < best["position_l2_mm"]):
            best = item
    t_init_end = time.perf_counter()

    if best is None:
        raise RuntimeError("No candidate subspace has trained model.")

    t_fallback_ms = 0.0
    fallback_triggered = False
    if best["position_l2_mm"] > best["e_max"]:
        fallback_triggered = True
        t_fb_start = time.perf_counter()
        fallback_best = None
        for sid in trained_all:
            ck = safe_torch_load(pred_meta_path.parent / "subspace_models" / model_index[sid]["model_file"])
            m15, m6 = load_prediction_pair(ck)
            q0 = predict_q_deg(m15, m6, x_pred)
            l2 = position_l2_norm(q0, target_pose)
            item = {
                "subspace_id": sid,
                "q0_deg": q0.tolist(),
                "position_l2_mm": l2,
                "e_max": float(ck.get("e_max", np.inf)),
            }
            if (fallback_best is None) or (item["position_l2_mm"] < fallback_best["position_l2_mm"]):
                fallback_best = item
        if fallback_best is not None:
            best = fallback_best
        t_fallback_ms = (time.perf_counter() - t_fb_start) * 1000.0

    result = {
        "target_pose6": target_pose.tolist(),
        "cls_topk": int(args.cls_topk),
        "candidate_subspaces": candidate_labels,
        "candidate_source": candidate_source,
        "fallback_full_scan_triggered": fallback_triggered,
        "initial_solution": best,
    }

    t_nr_ms = 0.0
    if args.enable_nr:
        t_nr_start = time.perf_counter()
        nr = newton_raphson_refine(
            q0_deg=best["q0_deg"],
            target_pose6=target_pose,
            options=NROptions(
                max_iters=args.nr_max_iters,
                tol_pos_mm=args.nr_tol_pos_mm,
                tol_ori_rad=args.nr_tol_ori_rad,
                damping=args.nr_damping,
                step_scale=args.nr_step_scale,
            ),
        )
        q_opt = np.asarray(nr["q_deg"], dtype=float)
        pose_opt = pose6_from_q(q_opt, input_unit="deg")
        result["refined_solution"] = {
            "q_deg": q_opt.tolist(),
            "final_pose6": pose_opt.tolist(),
            "nr_iters": int(nr["iters"]),
            "nr_converged": bool(nr["converged"]),
            "final_pos_err_mm": float(nr["final_pos_err_mm"]),
            "final_ori_err_rad": float(nr["final_ori_err_rad"]),
        }
        t_nr_ms = (time.perf_counter() - t_nr_start) * 1000.0

    t_total_ms = (time.perf_counter() - t0_total) * 1000.0
    result["ik_solve_time_ms"] = float(t_total_ms)
    result["timing_breakdown_ms"] = {
        "classification_ms": float((t_cls_end - t_cls_start) * 1000.0),
        "initial_selection_ms": float((t_init_end - t_init_start) * 1000.0),
        "fallback_full_scan_ms": float(t_fallback_ms),
        "nr_refinement_ms": float(t_nr_ms),
        "total_ms": float(t_total_ms),
    }

    if args.out_json.strip():
        save_json(Path(args.out_json), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
