#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abb_nn.branch_models import build_branch_classifier_variant
from abb_nn.models import MLPRegressor, build_classifier_variant
from abb_nn.optimization import (
    DLSOptions,
    LBFGSBOptions,
    NROptions,
    dls_refine,
    evaluate_solution_metrics,
    lbfgsb_refine,
    multistart_dls_refine,
    multistart_lbfgsb_refine,
    newton_raphson_refine,
)
from fk_model import JOINT_LIMITS_DEG, pose6_from_q_torch_batch
from predict_ik import build_conditioned_features

FIGURE_DIR = ROOT / "figure"
DATA_DIR = FIGURE_DIR / "data"
FIGURES_DIR = FIGURE_DIR / "figures"

SUCCESS_POS_MM = 1.0
SUCCESS_ORI_RAD = 1e-2
MULTISTART_COUNT = 10
HOME_Q_DEG = np.zeros(6, dtype=float)


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
    metrics = evaluate_solution_metrics(q_deg, target_pose6)
    return float(metrics["final_pos_err_mm"])


class HierarchicalNNSolver:
    def __init__(
        self,
        pred_meta_path: Path,
        branch_meta_path: Path,
        fine_meta_path: Path,
        topk_shoulder: int = 2,
        topk_elbow: int = 1,
        topk_wrist: int = 2,
        max_branch_candidates: int = 6,
        fine_topk_per_branch: int = 3,
        max_subspace_candidates: int = 18,
    ) -> None:
        self.pred_meta_path = pred_meta_path
        self.branch_meta_path = branch_meta_path
        self.fine_meta_path = fine_meta_path
        self.pred_meta = load_json(pred_meta_path)
        self.branch_meta = load_json(branch_meta_path)
        self.fine_meta = load_json(fine_meta_path)
        self.topk_shoulder = topk_shoulder
        self.topk_elbow = topk_elbow
        self.topk_wrist = topk_wrist
        self.max_branch_candidates = max_branch_candidates
        self.fine_topk_per_branch = fine_topk_per_branch
        self.max_subspace_candidates = max_subspace_candidates

        self.pred_mean = np.array(self.pred_meta["normalizer"]["mean"], dtype=np.float32).reshape(1, -1)
        self.pred_std = np.array(self.pred_meta["normalizer"]["std"], dtype=np.float32).reshape(1, -1)
        self.branch_mean = np.array(self.branch_meta["normalizer"]["mean"], dtype=np.float32).reshape(1, -1)
        self.branch_std = np.array(self.branch_meta["normalizer"]["std"], dtype=np.float32).reshape(1, -1)
        self.fine_mean = np.array(self.fine_meta["normalizer"]["mean"], dtype=np.float32).reshape(1, -1)
        self.fine_std = np.array(self.fine_meta["normalizer"]["std"], dtype=np.float32).reshape(1, -1)
        self.branch_count = int(self.branch_meta["num_combined_branches"])
        self.branch_head_dims = tuple(int(x) for x in self.branch_meta["branch_head_dims"])
        self.fine_num_classes = int(self.fine_meta["num_fine_classes"])
        self.model_index: Dict[int, Dict[str, object]] = {
            int(x["subspace_id"]): x for x in self.pred_meta["trained_subspaces"]
        }
        self.trained_all = sorted(self.model_index.keys())
        self.branch_name_map = {
            int(x["branch_label"]): x["branch_name"] for x in self.branch_meta["branch_to_subspaces"]
        }
        self.branch_fine_map = {
            int(item["branch_label"]): {
                int(x["fine_label"]): int(x["subspace_id"]) for x in item["fine_to_subspace"]
            }
            for item in self.fine_meta["branch_fine_to_subspace"]
        }

        self.branch_models = []
        for item in self.branch_meta["models"]:
            ckpt = safe_torch_load(branch_meta_path.parent / item["file"])
            model = build_branch_classifier_variant(
                variant=int(item["variant"]),
                input_dim=6,
                head_dims=ckpt["branch_head_dims"],
            )
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self.branch_models.append((int(item["variant"]), model))

        self.fine_models = []
        for item in self.fine_meta["models"]:
            ckpt = safe_torch_load(fine_meta_path.parent / item["file"])
            model = build_classifier_variant(
                variant=int(item["variant"]),
                input_dim=int(ckpt["input_dim"]),
                num_classes=int(ckpt["num_classes"]),
            )
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self.fine_models.append((int(item["variant"]), model))

        self.prediction_models = {}
        for sid, item in self.model_index.items():
            ckpt = safe_torch_load(pred_meta_path.parent / "subspace_models" / item["model_file"])
            m15, m6 = load_prediction_pair(ckpt)
            self.prediction_models[sid] = {
                "m15": m15,
                "m6": m6,
                "e_max": float(ckpt.get("e_max", np.inf)),
            }

    def generate_candidates(self, pose: np.ndarray) -> tuple[list[int], dict]:
        x_branch = apply_normalizer(pose.reshape(1, -1).astype(np.float32), self.branch_mean, self.branch_std)
        x_fine = apply_normalizer(pose.reshape(1, -1).astype(np.float32), self.fine_mean, self.fine_std)
        xt_branch = torch.from_numpy(x_branch)

        branch_scores: Dict[int, float] = {}
        for _, model in self.branch_models:
            with torch.no_grad():
                logits_shoulder, logits_elbow, logits_wrist = model(xt_branch)
                logp_shoulder = torch.log_softmax(logits_shoulder, dim=1)
                logp_elbow = torch.log_softmax(logits_elbow, dim=1)
                logp_wrist = torch.log_softmax(logits_wrist, dim=1)

            k_shoulder = min(self.topk_shoulder, self.branch_head_dims[0])
            k_elbow = min(self.topk_elbow, self.branch_head_dims[1])
            k_wrist = min(self.topk_wrist, self.branch_head_dims[2])
            shoulder_ids = torch.topk(logp_shoulder, k=k_shoulder, dim=1).indices.reshape(-1).tolist()
            elbow_ids = torch.topk(logp_elbow, k=k_elbow, dim=1).indices.reshape(-1).tolist()
            wrist_ids = torch.topk(logp_wrist, k=k_wrist, dim=1).indices.reshape(-1).tolist()

            for shoulder in shoulder_ids:
                for elbow in elbow_ids:
                    for wrist in wrist_ids:
                        branch_label = shoulder * self.branch_head_dims[1] * self.branch_head_dims[2] + elbow * self.branch_head_dims[2] + wrist
                        score = float(
                            logp_shoulder[0, shoulder].item()
                            + logp_elbow[0, elbow].item()
                            + logp_wrist[0, wrist].item()
                        )
                        branch_scores[branch_label] = max(branch_scores.get(branch_label, float("-inf")), score)

        branch_candidates = sorted(branch_scores.items(), key=lambda kv: kv[1], reverse=True)[: self.max_branch_candidates]
        fine_scores_all: Dict[int, float] = {}
        for branch_label, branch_score in branch_candidates:
            conditioned = build_conditioned_features(x_fine, int(branch_label))
            xt_fine = torch.from_numpy(conditioned)
            agg_logits = np.zeros((self.fine_num_classes,), dtype=np.float64)
            for _, model in self.fine_models:
                with torch.no_grad():
                    logits = model(xt_fine)
                    logp = torch.log_softmax(logits, dim=1).cpu().numpy().reshape(-1)
                agg_logits += logp

            top_ids = np.argsort(-agg_logits)[: min(self.fine_topk_per_branch, self.fine_num_classes)]
            mapping = self.branch_fine_map[int(branch_label)]
            for fine_label in top_ids.tolist():
                subspace_id = mapping[int(fine_label)]
                score = float(branch_score + agg_logits[fine_label])
                fine_scores_all[subspace_id] = max(fine_scores_all.get(subspace_id, float("-inf")), score)

        final_subspaces = sorted(fine_scores_all.items(), key=lambda kv: kv[1], reverse=True)[: self.max_subspace_candidates]
        candidate_labels = [int(x[0]) for x in final_subspaces]
        return candidate_labels, {
            "candidate_count": int(len(candidate_labels)),
            "candidate_subspaces": candidate_labels,
            "branch_candidates": [
                {
                    "branch_label": int(label),
                    "branch_name": self.branch_name_map[int(label)],
                    "score": float(score),
                }
                for label, score in branch_candidates
            ],
        }

    def select_initial_solution(self, pose: np.ndarray, candidate_labels: list[int]) -> dict:
        x_pred = apply_normalizer(pose.reshape(1, -1).astype(np.float32), self.pred_mean, self.pred_std)
        best = None
        for sid in candidate_labels:
            model_pair = self.prediction_models[sid]
            q0 = predict_q_deg(model_pair["m15"], model_pair["m6"], x_pred)
            l2 = position_l2_norm(q0, pose)
            item = {
                "subspace_id": int(sid),
                "q0_deg": q0.tolist(),
                "position_l2_mm": float(l2),
                "e_max": float(model_pair["e_max"]),
            }
            if (best is None) or (item["position_l2_mm"] < best["position_l2_mm"]):
                best = item

        if best is None:
            raise RuntimeError("No trained candidate subspace available.")

        fallback_triggered = False
        if best["position_l2_mm"] > best["e_max"]:
            fallback_triggered = True
            fallback_best = None
            for sid in self.trained_all:
                model_pair = self.prediction_models[sid]
                q0 = predict_q_deg(model_pair["m15"], model_pair["m6"], x_pred)
                l2 = position_l2_norm(q0, pose)
                item = {
                    "subspace_id": int(sid),
                    "q0_deg": q0.tolist(),
                    "position_l2_mm": float(l2),
                    "e_max": float(model_pair["e_max"]),
                }
                if (fallback_best is None) or (item["position_l2_mm"] < fallback_best["position_l2_mm"]):
                    fallback_best = item
            if fallback_best is not None:
                best = fallback_best
        best["fallback_triggered"] = fallback_triggered
        return best

    def solve(self, pose: np.ndarray, enable_nr: bool) -> dict:
        t0 = time.perf_counter()
        candidate_labels, candidate_info = self.generate_candidates(pose)
        initial = self.select_initial_solution(pose, candidate_labels)
        if enable_nr:
            nr = newton_raphson_refine(
                q0_deg=initial["q0_deg"],
                target_pose6=pose,
                options=NROptions(
                    max_iters=40,
                    tol_pos_mm=1e-3,
                    tol_ori_rad=1e-3,
                    damping=1e-5,
                    step_scale=1.0,
                ),
            )
            final_q_deg = np.asarray(nr["q_deg"], dtype=float)
            metrics = evaluate_solution_metrics(final_q_deg, pose)
            result = {
                "method": "nn_nr",
                "q_deg": final_q_deg.tolist(),
                "iters": int(nr["iters"]),
                "converged": bool(nr["converged"]),
            }
        else:
            final_q_deg = np.asarray(initial["q0_deg"], dtype=float)
            metrics = evaluate_solution_metrics(final_q_deg, pose)
            result = {
                "method": "nn_only",
                "q_deg": final_q_deg.tolist(),
                "iters": 0,
                "converged": False,
            }

        result.update(metrics)
        result.update(
            {
                "candidate_count": int(candidate_info["candidate_count"]),
                "candidate_subspaces": candidate_info["candidate_subspaces"],
                "initial_subspace_id": int(initial["subspace_id"]),
                "initial_pos_err_mm": float(initial["position_l2_mm"]),
                "fallback_triggered": bool(initial["fallback_triggered"]),
                "starts_used": 1,
                "solve_time_ms": float((time.perf_counter() - t0) * 1000.0),
            }
        )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 6-method IK benchmark for ABB_IRB.")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def build_pose_batch(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.uniform(JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1], size=(n_samples, 6)).astype(np.float32)
    with torch.no_grad():
        pose = pose6_from_q_torch_batch(torch.from_numpy(q), input_unit="deg").cpu().numpy()
    return pose.astype(np.float32)


def solve_single_method(method: str, pose: np.ndarray, nn_solver: HierarchicalNNSolver, rng: np.random.Generator) -> dict:
    if method == "nn_only":
        return nn_solver.solve(pose=pose, enable_nr=False)
    if method == "nn_nr":
        return nn_solver.solve(pose=pose, enable_nr=True)
    if method == "dls":
        t0 = time.perf_counter()
        out = dls_refine(
            q0_deg=HOME_Q_DEG,
            target_pose6=pose,
            options=DLSOptions(max_iters=80, tol_pos_mm=SUCCESS_POS_MM, tol_ori_rad=SUCCESS_ORI_RAD, damping=1e-2, orientation_weight=200.0),
        )
        out["solve_time_ms"] = float((time.perf_counter() - t0) * 1000.0)
        out["candidate_count"] = 0
        out["candidate_subspaces"] = []
        out["initial_subspace_id"] = -1
        out["initial_pos_err_mm"] = float("nan")
        out["fallback_triggered"] = False
        out["starts_used"] = 1
        return out
    if method == "multistart_dls":
        t0 = time.perf_counter()
        out = multistart_dls_refine(
            target_pose6=pose,
            n_starts=MULTISTART_COUNT,
            rng=rng,
            options=DLSOptions(max_iters=80, tol_pos_mm=SUCCESS_POS_MM, tol_ori_rad=SUCCESS_ORI_RAD, damping=1e-2, orientation_weight=200.0),
            include_zero=True,
        )
        out["solve_time_ms"] = float((time.perf_counter() - t0) * 1000.0)
        out["candidate_count"] = 0
        out["candidate_subspaces"] = []
        out["initial_subspace_id"] = -1
        out["initial_pos_err_mm"] = float("nan")
        out["fallback_triggered"] = False
        return out
    if method == "lbfgsb":
        t0 = time.perf_counter()
        out = lbfgsb_refine(
            q0_deg=HOME_Q_DEG,
            target_pose6=pose,
            options=LBFGSBOptions(max_iters=200, tol_pos_mm=SUCCESS_POS_MM, tol_ori_rad=SUCCESS_ORI_RAD, orientation_weight=200.0),
        )
        out["solve_time_ms"] = float((time.perf_counter() - t0) * 1000.0)
        out["candidate_count"] = 0
        out["candidate_subspaces"] = []
        out["initial_subspace_id"] = -1
        out["initial_pos_err_mm"] = float("nan")
        out["fallback_triggered"] = False
        out["starts_used"] = 1
        return out
    if method == "multistart_lbfgsb":
        t0 = time.perf_counter()
        out = multistart_lbfgsb_refine(
            target_pose6=pose,
            n_starts=MULTISTART_COUNT,
            rng=rng,
            options=LBFGSBOptions(max_iters=200, tol_pos_mm=SUCCESS_POS_MM, tol_ori_rad=SUCCESS_ORI_RAD, orientation_weight=200.0),
            include_zero=True,
        )
        out["solve_time_ms"] = float((time.perf_counter() - t0) * 1000.0)
        out["candidate_count"] = 0
        out["candidate_subspaces"] = []
        out["initial_subspace_id"] = -1
        out["initial_pos_err_mm"] = float("nan")
        out["fallback_triggered"] = False
        return out
    raise ValueError(f"Unknown method: {method}")


def summarize_results(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method"):
        rows.append(
            {
                "method": method,
                "success_rate": float(g["success"].mean()),
                "converged_rate": float(g["converged"].mean()),
                "mean_final_pos_err_mm": float(g["final_pos_err_mm"].mean()),
                "median_final_pos_err_mm": float(g["final_pos_err_mm"].median()),
                "p95_final_pos_err_mm": float(g["final_pos_err_mm"].quantile(0.95)),
                "mean_final_ori_err_rad": float(g["final_ori_err_rad"].mean()),
                "median_final_ori_err_rad": float(g["final_ori_err_rad"].median()),
                "p95_final_ori_err_rad": float(g["final_ori_err_rad"].quantile(0.95)),
                "mean_solve_time_ms": float(g["solve_time_ms"].mean()),
                "median_solve_time_ms": float(g["solve_time_ms"].median()),
                "p95_solve_time_ms": float(g["solve_time_ms"].quantile(0.95)),
                "mean_iters": float(g["iters"].mean()),
                "mean_starts_used": float(g["starts_used"].mean()),
                "mean_candidate_count": float(g["candidate_count"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def plot_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    labels = {
        "nn_only": "NN only",
        "nn_nr": "NN + NR",
        "dls": "DLS",
        "multistart_dls": "Multi-start DLS",
        "lbfgsb": "L-BFGS-B",
        "multistart_lbfgsb": "Multi-start L-BFGS-B",
    }
    df = summary_df.copy()
    df["label"] = df["method"].map(labels)
    order = [labels[x] for x in ["nn_only", "nn_nr", "dls", "multistart_dls", "lbfgsb", "multistart_lbfgsb"]]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)

    sns.barplot(data=df, x="label", y="success_rate", order=order, hue="label", dodge=False, palette="Set2", ax=axes[0])
    axes[0].set_title("工程成功率 / Engineering Success Rate")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("成功率 / Success Rate")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].set_ylim(0.0, 1.05)
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()

    sns.barplot(data=df, x="label", y="median_final_pos_err_mm", order=order, hue="label", dodge=False, palette="Set2", ax=axes[1])
    axes[1].set_title("位置误差中位数 / Median Position Error")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("误差 (mm) / Error")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].set_yscale("log")
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()

    sns.barplot(data=df, x="label", y="median_solve_time_ms", order=order, hue="label", dodge=False, palette="Set2", ax=axes[2])
    axes[2].set_title("求解时间中位数 / Median Solve Time")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("时间 (ms) / Time")
    axes[2].tick_params(axis="x", rotation=18)
    if axes[2].get_legend() is not None:
        axes[2].get_legend().remove()

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_distribution(df: pd.DataFrame, out_path: Path) -> None:
    labels = {
        "nn_only": "NN only",
        "nn_nr": "NN + NR",
        "dls": "DLS",
        "multistart_dls": "Multi-start DLS",
        "lbfgsb": "L-BFGS-B",
        "multistart_lbfgsb": "Multi-start L-BFGS-B",
    }
    data = df.copy()
    data["label"] = data["method"].map(labels)
    order = [labels[x] for x in ["nn_only", "nn_nr", "dls", "multistart_dls", "lbfgsb", "multistart_lbfgsb"]]
    methods = ["nn_only", "nn_nr", "dls", "multistart_dls", "lbfgsb", "multistart_lbfgsb"]
    palette = dict(zip(order, sns.color_palette("tab10", n_colors=len(order))))
    pos_thresholds = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0], dtype=float)
    ori_thresholds = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.2], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 7.2), constrained_layout=True)
    ax_pos_ecdf = axes[0, 0]
    ax_ori_ecdf = axes[0, 1]
    ax_pos_thr = axes[1, 0]
    ax_ori_thr = axes[1, 1]

    for method in methods:
        group = data[data["method"] == method]
        label = labels[method]
        color = palette[label]

        pos_vals = np.sort(group["final_pos_err_mm"].to_numpy(dtype=float))
        pos_cdf = np.arange(1, len(pos_vals) + 1, dtype=float) / len(pos_vals)
        ax_pos_ecdf.step(np.clip(pos_vals, 1e-12, None), pos_cdf, where="post", linewidth=1.9, color=color, label=label)

        ori_vals = np.sort(group["final_ori_err_rad"].to_numpy(dtype=float))
        ori_cdf = np.arange(1, len(ori_vals) + 1, dtype=float) / len(ori_vals)
        ax_ori_ecdf.step(np.clip(ori_vals, 1e-12, None), ori_cdf, where="post", linewidth=1.9, color=color, label=label)

        pos_pass_rate = [(group["final_pos_err_mm"].to_numpy(dtype=float) <= thr).mean() for thr in pos_thresholds]
        ax_pos_thr.plot(
            pos_thresholds,
            pos_pass_rate,
            marker="o",
            markersize=3.8,
            linewidth=1.7,
            color=color,
            label=label,
        )

        ori_pass_rate = [(group["final_ori_err_rad"].to_numpy(dtype=float) <= thr).mean() for thr in ori_thresholds]
        ax_ori_thr.plot(
            ori_thresholds,
            ori_pass_rate,
            marker="o",
            markersize=3.8,
            linewidth=1.7,
            color=color,
            label=label,
        )

    ax_pos_ecdf.axvline(SUCCESS_POS_MM, color="#666666", linestyle="--", linewidth=1.0)
    ax_pos_ecdf.set_title("位置误差累计分布 / Position Error ECDF")
    ax_pos_ecdf.set_xlabel("位置误差 (mm) / Position Error")
    ax_pos_ecdf.set_ylabel("累计概率 / Cumulative Probability")
    ax_pos_ecdf.set_xscale("log")
    ax_pos_ecdf.set_xlim(1e-8, 2e3)
    ax_pos_ecdf.set_ylim(0.0, 1.02)

    ax_ori_ecdf.axvline(SUCCESS_ORI_RAD, color="#666666", linestyle="--", linewidth=1.0)
    ax_ori_ecdf.set_title("姿态误差累计分布 / Orientation Error ECDF")
    ax_ori_ecdf.set_xlabel("姿态误差 (rad) / Orientation Error")
    ax_ori_ecdf.set_ylabel("累计概率 / Cumulative Probability")
    ax_ori_ecdf.set_xscale("log")
    ax_ori_ecdf.set_xlim(1e-8, 3.2)
    ax_ori_ecdf.set_ylim(0.0, 1.02)

    ax_pos_thr.axvline(SUCCESS_POS_MM, color="#666666", linestyle="--", linewidth=1.0)
    ax_pos_thr.set_title("位置误差阈值达标率 / Position Error Pass Rate")
    ax_pos_thr.set_xlabel("阈值 (mm) / Threshold")
    ax_pos_thr.set_ylabel("达标率 / Pass Rate")
    ax_pos_thr.set_xscale("log")
    ax_pos_thr.set_xlim(pos_thresholds.min(), pos_thresholds.max())
    ax_pos_thr.set_ylim(0.0, 1.02)

    ax_ori_thr.axvline(SUCCESS_ORI_RAD, color="#666666", linestyle="--", linewidth=1.0)
    ax_ori_thr.set_title("姿态误差阈值达标率 / Orientation Error Pass Rate")
    ax_ori_thr.set_xlabel("阈值 (rad) / Threshold")
    ax_ori_thr.set_ylabel("达标率 / Pass Rate")
    ax_ori_thr.set_xscale("log")
    ax_ori_thr.set_xlim(1e-8, 3.2)
    ax_ori_thr.set_ylim(0.0, 1.02)

    handles, legend_labels = ax_pos_ecdf.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=True,
        columnspacing=1.4,
        handlelength=2.4,
        handletextpad=0.6,
        borderpad=0.8,
        prop={"size": 10},
    )

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cdf(df: pd.DataFrame, out_path: Path) -> None:
    labels = {
        "nn_only": "NN only",
        "nn_nr": "NN + NR",
        "dls": "DLS",
        "multistart_dls": "Multi-start DLS",
        "lbfgsb": "L-BFGS-B",
        "multistart_lbfgsb": "Multi-start L-BFGS-B",
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.2), constrained_layout=True)
    for method, group in df.groupby("method"):
        vals = np.sort(group["final_pos_err_mm"].to_numpy(dtype=float))
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, linewidth=1.8, label=labels[method])
    ax.set_title("位置误差累计分布 / CDF of Position Error")
    ax.set_xlabel("位置误差 (mm) / Position Error")
    ax.set_ylabel("累计概率 / Cumulative Probability")
    ax.set_xscale("log")
    ax.legend(frameon=True, ncol=2)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_iterations(df: pd.DataFrame, out_path: Path) -> None:
    labels = {
        "nn_nr": "NN + NR",
        "dls": "DLS",
        "multistart_dls": "Multi-start DLS",
        "lbfgsb": "L-BFGS-B",
        "multistart_lbfgsb": "Multi-start L-BFGS-B",
    }
    data = df[df["method"].isin(labels.keys())].copy()
    data["label"] = data["method"].map(labels)
    methods = ["nn_nr", "dls", "multistart_dls", "lbfgsb", "multistart_lbfgsb"]
    order = [labels[x] for x in methods]
    bucket_specs = [
        ("<=5", lambda x: x <= 5),
        ("6-10", lambda x: (x >= 6) & (x <= 10)),
        ("11-20", lambda x: (x >= 11) & (x <= 20)),
        ("21-40", lambda x: (x >= 21) & (x <= 40)),
        ("41-80", lambda x: (x >= 41) & (x <= 80)),
        (">80", lambda x: x > 80),
    ]
    bucket_colors = ["#DCEAF7", "#B8D8BA", "#F6D7A7", "#F2B880", "#E07A5F", "#8E5A7A"]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    ax_stack, ax_stats = axes

    y_pos = np.arange(len(methods))
    left = np.zeros(len(methods), dtype=float)
    for (bucket_label, bucket_fn), bucket_color in zip(bucket_specs, bucket_colors):
        widths = []
        for method in methods:
            vals = data.loc[data["method"] == method, "iters"].to_numpy(dtype=float)
            widths.append(float(bucket_fn(vals).mean()) * 100.0)
        widths = np.asarray(widths, dtype=float)
        ax_stack.barh(
            y_pos,
            widths,
            left=left,
            height=0.65,
            color=bucket_color,
            edgecolor="white",
            linewidth=0.8,
            label=bucket_label,
        )
        left += widths

    ax_stack.set_title("迭代分档占比 / Iteration Bucket Proportion")
    ax_stack.set_xlabel("样本占比 (%) / Sample Ratio")
    ax_stack.set_ylabel("")
    ax_stack.set_yticks(y_pos)
    ax_stack.set_yticklabels(order)
    ax_stack.set_xlim(0.0, 100.0)
    ax_stack.invert_yaxis()

    medians = []
    p95_values = []
    for method in methods:
        vals = data.loc[data["method"] == method, "iters"].to_numpy(dtype=float)
        medians.append(float(np.median(vals)))
        p95_values.append(float(np.quantile(vals, 0.95)))

    x = np.arange(len(methods), dtype=float)
    bar_width = 0.34
    bars_median = ax_stats.bar(
        x - bar_width / 2.0,
        medians,
        width=bar_width,
        color="#7AA6DC",
        edgecolor="white",
        linewidth=0.8,
        label="Median",
    )
    bars_p95 = ax_stats.bar(
        x + bar_width / 2.0,
        p95_values,
        width=bar_width,
        color="#E7A96B",
        edgecolor="white",
        linewidth=0.8,
        label="P95",
    )

    ax_stats.set_title("关键统计量对比 / Median and P95 of Iterations")
    ax_stats.set_xlabel("")
    ax_stats.set_ylabel("迭代次数 / Iterations")
    ax_stats.set_xticks(x)
    ax_stats.set_xticklabels(order, rotation=18)
    ax_stats.set_ylim(0.0, max(p95_values) * 1.28)
    ax_stats.legend(frameon=True, loc="upper left")

    for bars in [bars_median, bars_p95]:
        for bar in bars:
            value = float(bar.get_height())
            ax_stats.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 1.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    handles, legend_labels = ax_stack.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.32, -0.12),
        ncol=3,
        frameon=True,
        columnspacing=1.1,
        handlelength=1.6,
        handletextpad=0.5,
        borderpad=0.7,
        prop={"size": 9},
        title="迭代区间 / Iteration Buckets",
        title_fontsize=9,
    )

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tag = args.tag.strip() or f"n{args.n_samples}"
    pose_batch = build_pose_batch(n_samples=args.n_samples, seed=args.seed)
    nn_solver = HierarchicalNNSolver(
        pred_meta_path=ROOT / "artifacts" / "prediction_system_formal" / "metadata.json",
        branch_meta_path=ROOT / "artifacts" / "branch_classification_system" / "metadata.json",
        fine_meta_path=ROOT / "artifacts" / "fine_classification_system" / "metadata.json",
        topk_shoulder=2,
        topk_elbow=1,
        topk_wrist=2,
        max_branch_candidates=6,
        fine_topk_per_branch=3,
        max_subspace_candidates=18,
    )

    methods = [
        "nn_only",
        "nn_nr",
        "dls",
        "multistart_dls",
        "lbfgsb",
        "multistart_lbfgsb",
    ]

    rows = []
    for idx, pose in enumerate(pose_batch):
        print(f"[benchmark-6] sample {idx + 1}/{args.n_samples}")
        for method in methods:
            method_rng = np.random.default_rng(args.seed * 100000 + idx * 100 + methods.index(method))
            out = solve_single_method(method=method, pose=pose, nn_solver=nn_solver, rng=method_rng)
            rows.append(
                {
                    "sample_id": int(idx),
                    "method": method,
                    "target_pose6": ",".join(f"{float(x):.9f}" for x in pose.tolist()),
                    "q_result_deg": ",".join(f"{float(x):.9f}" for x in np.asarray(out["q_deg"], dtype=float).tolist()),
                    "final_pos_err_mm": float(out["final_pos_err_mm"]),
                    "final_ori_err_rad": float(out["final_ori_err_rad"]),
                    "success": int(
                        bool(out["within_joint_limits"])
                        and float(out["final_pos_err_mm"]) <= SUCCESS_POS_MM
                        and float(out["final_ori_err_rad"]) <= SUCCESS_ORI_RAD
                    ),
                    "converged": int(bool(out.get("converged", False))),
                    "solve_time_ms": float(out["solve_time_ms"]),
                    "iters": int(out.get("iters", 0)),
                    "starts_used": int(out.get("starts_used", 1)),
                    "candidate_count": int(out.get("candidate_count", 0)),
                    "within_joint_limits": int(bool(out["within_joint_limits"])),
                    "initial_subspace_id": int(out.get("initial_subspace_id", -1)),
                    "initial_pos_err_mm": float(out.get("initial_pos_err_mm", float("nan"))),
                    "fallback_triggered": int(bool(out.get("fallback_triggered", False))),
                }
            )

    df = pd.DataFrame(rows)
    detail_path = DATA_DIR / f"ik_benchmark_six_methods_detailed_{tag}.csv"
    summary_path = DATA_DIR / f"ik_benchmark_six_methods_summary_{tag}.csv"
    fig_summary_path = FIGURES_DIR / f"ik_benchmark_six_methods_summary_{tag}.png"
    fig_distribution_path = FIGURES_DIR / f"ik_benchmark_six_methods_distribution_{tag}.png"
    fig_cdf_path = FIGURES_DIR / f"ik_benchmark_six_methods_cdf_{tag}.png"
    fig_iters_path = FIGURES_DIR / f"ik_benchmark_six_methods_iterations_{tag}.png"

    df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df = summarize_results(df, summary_path)
    plot_summary(summary_df, fig_summary_path)
    plot_distribution(df, fig_distribution_path)
    plot_cdf(df, fig_cdf_path)
    plot_iterations(df, fig_iters_path)

    print(f"Saved detailed benchmark to: {detail_path}")
    print(f"Saved summary benchmark to: {summary_path}")
    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
