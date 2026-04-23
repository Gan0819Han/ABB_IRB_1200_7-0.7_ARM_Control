#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Windows + conda local workaround for duplicate OpenMP runtime initialization
# observed during lightweight inference in this environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from abb_nn.branch_models import build_branch_classifier_variant
from abb_nn.branching import (
    BRANCH_COUNT,
    BRANCH_HEAD_DIMS,
    branch_fine_to_subspace_label,
    branch_label_to_name,
    encode_branch_index,
)
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


def build_conditioned_features(x_pose_norm: np.ndarray, branch_label: int) -> np.ndarray:
    onehot = np.zeros((1, BRANCH_COUNT), dtype=np.float32)
    onehot[0, int(branch_label)] = 1.0
    return np.concatenate([x_pose_norm.astype(np.float32), onehot], axis=1)


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


def generate_flat_candidates(
    pose: np.ndarray,
    cls_meta_path: Path,
    cls_meta: dict,
    cls_topk: int,
) -> Tuple[List[int], Dict[str, object], Dict[str, float]]:
    if cls_topk <= 0:
        raise ValueError("--cls_topk must be >= 1.")

    cls_mean = np.array(cls_meta["normalizer"]["mean"], dtype=np.float32).reshape(1, -1)
    cls_std = np.array(cls_meta["normalizer"]["std"], dtype=np.float32).reshape(1, -1)
    x_cls = apply_normalizer(pose.reshape(1, -1).astype(np.float32), cls_mean, cls_std)
    xt_cls = torch.from_numpy(x_cls)

    t0 = time.perf_counter()
    candidate_labels: List[int] = []
    per_model_predictions = []

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
            logits = cls(xt_cls)
            topk = min(cls_topk, num_classes)
            pred_ids = torch.topk(logits, k=topk, dim=1).indices.reshape(-1).tolist()
        pred_ids = [int(x) for x in pred_ids]
        candidate_labels.extend(pred_ids)
        per_model_predictions.append(
            {
                "variant": int(item["variant"]),
                "topk_subspaces": pred_ids,
            }
        )

    candidate_labels = sorted(set(candidate_labels))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return (
        candidate_labels,
        {
            "mode": "flat",
            "cls_topk": int(cls_topk),
            "per_model_predictions": per_model_predictions,
        },
        {
            "candidate_generation_ms": float(elapsed_ms),
            "classification_ms": float(elapsed_ms),
            "branch_classification_ms": 0.0,
            "fine_classification_ms": 0.0,
        },
    )


def generate_hierarchical_candidates(
    pose: np.ndarray,
    branch_meta_path: Path,
    branch_meta: dict,
    fine_meta_path: Path,
    fine_meta: dict,
    topk_shoulder: int,
    topk_elbow: int,
    topk_wrist: int,
    max_branch_candidates: int,
    fine_topk_per_branch: int,
    max_subspace_candidates: int,
) -> Tuple[List[int], Dict[str, object], Dict[str, float]]:
    if min(
        topk_shoulder,
        topk_elbow,
        topk_wrist,
        max_branch_candidates,
        fine_topk_per_branch,
        max_subspace_candidates,
    ) <= 0:
        raise ValueError("Hierarchical top-k / max-candidate arguments must be >= 1.")

    branch_mean = np.array(branch_meta["normalizer"]["mean"], dtype=np.float32).reshape(1, -1)
    branch_std = np.array(branch_meta["normalizer"]["std"], dtype=np.float32).reshape(1, -1)
    fine_mean = np.array(fine_meta["normalizer"]["mean"], dtype=np.float32).reshape(1, -1)
    fine_std = np.array(fine_meta["normalizer"]["std"], dtype=np.float32).reshape(1, -1)
    x_branch = apply_normalizer(pose.reshape(1, -1).astype(np.float32), branch_mean, branch_std)
    x_fine = apply_normalizer(pose.reshape(1, -1).astype(np.float32), fine_mean, fine_std)
    xt_branch = torch.from_numpy(x_branch)

    branch_models = []
    for item in branch_meta["models"]:
        ckpt = safe_torch_load(branch_meta_path.parent / item["file"])
        model = build_branch_classifier_variant(
            variant=int(item["variant"]),
            input_dim=6,
            head_dims=ckpt["branch_head_dims"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        branch_models.append((int(item["variant"]), model))

    fine_models = []
    for item in fine_meta["models"]:
        ckpt = safe_torch_load(fine_meta_path.parent / item["file"])
        model = build_classifier_variant(
            variant=int(item["variant"]),
            input_dim=int(ckpt["input_dim"]),
            num_classes=int(ckpt["num_classes"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        fine_models.append((int(item["variant"]), model))

    t_branch_start = time.perf_counter()
    branch_scores: Dict[int, float] = {}
    branch_per_model = []
    for variant, model in branch_models:
        with torch.no_grad():
            logits_shoulder, logits_elbow, logits_wrist = model(xt_branch)
            logp_shoulder = torch.log_softmax(logits_shoulder, dim=1)
            logp_elbow = torch.log_softmax(logits_elbow, dim=1)
            logp_wrist = torch.log_softmax(logits_wrist, dim=1)

        k_shoulder = min(topk_shoulder, BRANCH_HEAD_DIMS[0])
        k_elbow = min(topk_elbow, BRANCH_HEAD_DIMS[1])
        k_wrist = min(topk_wrist, BRANCH_HEAD_DIMS[2])
        shoulder_ids = torch.topk(logp_shoulder, k=k_shoulder, dim=1).indices.reshape(-1).tolist()
        elbow_ids = torch.topk(logp_elbow, k=k_elbow, dim=1).indices.reshape(-1).tolist()
        wrist_ids = torch.topk(logp_wrist, k=k_wrist, dim=1).indices.reshape(-1).tolist()

        local_candidates = []
        for shoulder, elbow, wrist in itertools.product(shoulder_ids, elbow_ids, wrist_ids):
            branch_label = encode_branch_index((shoulder, elbow, wrist))
            score = float(
                logp_shoulder[0, shoulder].item()
                + logp_elbow[0, elbow].item()
                + logp_wrist[0, wrist].item()
            )
            branch_scores[branch_label] = max(branch_scores.get(branch_label, float("-inf")), score)
            local_candidates.append(
                {
                    "branch_label": int(branch_label),
                    "branch_name": branch_label_to_name(branch_label),
                    "score": score,
                }
            )
        local_candidates.sort(key=lambda x: x["score"], reverse=True)
        branch_per_model.append(
            {
                "variant": int(variant),
                "candidates": local_candidates,
            }
        )

    branch_candidates = sorted(branch_scores.items(), key=lambda kv: kv[1], reverse=True)
    branch_candidates = branch_candidates[:max_branch_candidates]
    t_branch_ms = (time.perf_counter() - t_branch_start) * 1000.0

    t_fine_start = time.perf_counter()
    fine_num_classes = int(fine_meta["num_fine_classes"])
    fine_per_branch = []
    fine_scores_all: Dict[int, float] = {}

    for branch_label, branch_score in branch_candidates:
        conditioned = build_conditioned_features(x_fine, int(branch_label))
        xt_fine = torch.from_numpy(conditioned)
        agg_logits = np.zeros((fine_num_classes,), dtype=np.float64)

        for _, model in fine_models:
            with torch.no_grad():
                logits = model(xt_fine)
                logp = torch.log_softmax(logits, dim=1).cpu().numpy().reshape(-1)
            agg_logits += logp

        local_topk = min(fine_topk_per_branch, fine_num_classes)
        top_ids = np.argsort(-agg_logits)[:local_topk]
        local_items = []
        for fine_label in top_ids.tolist():
            global_score = float(branch_score + agg_logits[fine_label])
            subspace_id = branch_fine_to_subspace_label(
                int(branch_label),
                int(fine_label),
                segment_profile=branch_meta["segment_profile"],
            )
            fine_scores_all[subspace_id] = max(fine_scores_all.get(subspace_id, float("-inf")), global_score)
            local_items.append(
                {
                    "fine_label": int(fine_label),
                    "subspace_id": int(subspace_id),
                    "score": global_score,
                }
            )
        fine_per_branch.append(
            {
                "branch_label": int(branch_label),
                "branch_name": branch_label_to_name(int(branch_label)),
                "branch_score": float(branch_score),
                "fine_candidates": local_items,
            }
        )

    final_subspaces = sorted(fine_scores_all.items(), key=lambda kv: kv[1], reverse=True)
    final_subspaces = final_subspaces[:max_subspace_candidates]
    t_fine_ms = (time.perf_counter() - t_fine_start) * 1000.0

    return (
        [int(subspace_id) for subspace_id, _ in final_subspaces],
        {
            "mode": "hierarchical",
            "topk": {
                "shoulder": int(topk_shoulder),
                "elbow": int(topk_elbow),
                "wrist": int(topk_wrist),
                "fine_per_branch": int(fine_topk_per_branch),
            },
            "max_branch_candidates": int(max_branch_candidates),
            "max_subspace_candidates": int(max_subspace_candidates),
            "branch_candidates": [
                {
                    "branch_label": int(branch_label),
                    "branch_name": branch_label_to_name(int(branch_label)),
                    "score": float(score),
                }
                for branch_label, score in branch_candidates
            ],
            "fine_per_branch": fine_per_branch,
            "per_model_branch_candidates": branch_per_model,
            "candidate_subspaces_with_scores": [
                {
                    "subspace_id": int(subspace_id),
                    "score": float(score),
                }
                for subspace_id, score in final_subspaces
            ],
        },
        {
            "candidate_generation_ms": float(t_branch_ms + t_fine_ms),
            "classification_ms": float(t_branch_ms + t_fine_ms),
            "branch_classification_ms": float(t_branch_ms),
            "fine_classification_ms": float(t_fine_ms),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABB_IRB IK inference.")
    parser.add_argument("--pose", type=str, required=True, help="x_mm,y_mm,z_mm,phi_rad,theta_rad,psi_rad")
    parser.add_argument("--pred_meta", type=str, default="artifacts/prediction_system/metadata.json")
    parser.add_argument(
        "--candidate_mode",
        type=str,
        default="flat",
        choices=["flat", "hierarchical"],
        help="Subspace candidate generation strategy.",
    )
    parser.add_argument(
        "--cls_meta",
        type=str,
        default="artifacts/classification_system/metadata.json",
        help="Flat classifier metadata, used when candidate_mode=flat.",
    )
    parser.add_argument(
        "--cls_topk",
        type=int,
        default=1,
        help="Use top-k predicted subspaces from each flat classifier as candidates.",
    )
    parser.add_argument(
        "--branch_meta",
        type=str,
        default="artifacts/branch_classification_system/metadata.json",
        help="Hierarchical stage-1 branch classifier metadata.",
    )
    parser.add_argument(
        "--fine_meta",
        type=str,
        default="artifacts/fine_classification_system/metadata.json",
        help="Hierarchical stage-2 fine classifier metadata.",
    )
    parser.add_argument("--topk_shoulder", type=int, default=1)
    parser.add_argument("--topk_elbow", type=int, default=1)
    parser.add_argument("--topk_wrist", type=int, default=2)
    parser.add_argument("--max_branch_candidates", type=int, default=6)
    parser.add_argument("--fine_topk_per_branch", type=int, default=2)
    parser.add_argument("--max_subspace_candidates", type=int, default=12)
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
    pred_meta = load_json(pred_meta_path)
    pred_profile = pred_meta.get("segment_profile", "abb_strict")

    pred_mean = np.array(pred_meta["normalizer"]["mean"], dtype=np.float32).reshape(1, -1)
    pred_std = np.array(pred_meta["normalizer"]["std"], dtype=np.float32).reshape(1, -1)
    x_pred = apply_normalizer(target_pose.reshape(1, -1).astype(np.float32), pred_mean, pred_std)
    candidate_generation_info: Dict[str, object]
    timing_generation: Dict[str, float]
    if args.candidate_mode == "flat":
        cls_meta_path = Path(args.cls_meta)
        cls_meta = load_json(cls_meta_path)
        cls_profile = cls_meta.get("segment_profile", "abb_strict")
        if pred_profile != cls_profile:
            raise ValueError(
                f"Segment profile mismatch: prediction={pred_profile}, classification={cls_profile}. "
                "Please use matched artifacts."
            )
        candidate_labels, candidate_generation_info, timing_generation = generate_flat_candidates(
            target_pose,
            cls_meta_path,
            cls_meta,
            args.cls_topk,
        )
    else:
        branch_meta_path = Path(args.branch_meta)
        fine_meta_path = Path(args.fine_meta)
        branch_meta = load_json(branch_meta_path)
        fine_meta = load_json(fine_meta_path)
        branch_profile = branch_meta.get("segment_profile", "abb_strict")
        fine_profile = fine_meta.get("segment_profile", "abb_strict")
        if pred_profile != branch_profile or pred_profile != fine_profile:
            raise ValueError(
                "Segment profile mismatch among prediction / branch / fine artifacts. "
                f"prediction={pred_profile}, branch={branch_profile}, fine={fine_profile}"
            )
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

    model_index: Dict[int, Dict[str, object]] = {
        int(x["subspace_id"]): x for x in pred_meta["trained_subspaces"]
    }
    trained_all = sorted(model_index.keys())
    candidate_source = f"{args.candidate_mode}_predictions"

    if args.force_all_subspaces:
        candidate_labels = trained_all
        candidate_source = "force_all_subspaces"
    else:
        available_candidates = [sid for sid in candidate_labels if sid in model_index]
        if not available_candidates:
            candidate_labels = trained_all
            candidate_source = f"{args.candidate_mode}_empty_fallback_to_trained"
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
        "candidate_mode": args.candidate_mode,
        "candidate_generation": candidate_generation_info,
        "candidate_subspaces": candidate_labels,
        "candidate_source": candidate_source,
        "fallback_full_scan_triggered": fallback_triggered,
        "initial_solution": best,
    }
    if args.candidate_mode == "flat":
        result["cls_topk"] = int(args.cls_topk)

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
        "candidate_generation_ms": float(timing_generation["candidate_generation_ms"]),
        "classification_ms": float(timing_generation["classification_ms"]),
        "branch_classification_ms": float(timing_generation["branch_classification_ms"]),
        "fine_classification_ms": float(timing_generation["fine_classification_ms"]),
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
