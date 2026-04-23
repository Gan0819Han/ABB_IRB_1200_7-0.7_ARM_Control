#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
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

from fk_model import pose6_from_q_torch_batch
from robot_config import JOINT_LIMITS_DEG

FIGURE_DIR = ROOT / "figure"
DATA_DIR = FIGURE_DIR / "data"
FIGURES_DIR = FIGURE_DIR / "figures"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batched IK benchmark for flat vs hierarchical pipelines.")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of random target poses used in benchmark.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--success_pos_mm",
        type=float,
        default=1.0,
        help="Engineering success threshold for final position error.",
    )
    parser.add_argument(
        "--success_ori_rad",
        type=float,
        default=1e-2,
        help="Engineering success threshold for final orientation error.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix tag for output files. Empty means n{n_samples}.",
    )
    return parser.parse_args()


def build_pose_batch(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.uniform(JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1], size=(n_samples, 6)).astype(np.float32)
    with torch.no_grad():
        pose = pose6_from_q_torch_batch(torch.from_numpy(q), input_unit="deg").cpu().numpy()
    return pose.astype(np.float32)


def run_predict_ik(pose6: np.ndarray, mode: str, enable_nr: bool) -> Dict[str, object]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_json = Path(tmp.name)

    pose_text = ",".join(f"{float(x):.9f}" for x in pose6.tolist())
    cmd: List[str] = [
        sys.executable,
        "-X",
        "utf8",
        "predict_ik.py",
        f"--pose={pose_text}",
        "--pred_meta",
        "artifacts/prediction_system_formal/metadata.json",
        "--out_json",
        str(out_json),
    ]
    if mode == "flat":
        cmd.extend(
            [
                "--candidate_mode",
                "flat",
                "--cls_meta",
                "artifacts/classification_system_formal/metadata.json",
                "--cls_topk",
                "2",
            ]
        )
    elif mode == "hierarchical":
        cmd.extend(
            [
                "--candidate_mode",
                "hierarchical",
                "--branch_meta",
                "artifacts/branch_classification_system/metadata.json",
                "--fine_meta",
                "artifacts/fine_classification_system/metadata.json",
                "--topk_shoulder",
                "2",
                "--topk_elbow",
                "1",
                "--topk_wrist",
                "2",
                "--max_branch_candidates",
                "4",
                "--fine_topk_per_branch",
                "3",
                "--max_subspace_candidates",
                "15",
            ]
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if enable_nr:
        cmd.append("--enable_nr")

    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        result = json.loads(out_json.read_text(encoding="utf-8"))
    finally:
        if out_json.exists():
            out_json.unlink()
    return result


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def evaluate_final_pose_error(target_pose6: np.ndarray, q_deg: np.ndarray) -> Dict[str, float]:
    q_t = torch.from_numpy(np.asarray(q_deg, dtype=np.float32).reshape(1, 6))
    with torch.no_grad():
        final_pose = pose6_from_q_torch_batch(q_t, input_unit="deg").cpu().numpy().reshape(6)
    delta = np.asarray(target_pose6, dtype=float).reshape(6) - final_pose
    delta[3:] = wrap_to_pi(delta[3:])
    return {
        "final_pos_err_mm": float(np.linalg.norm(delta[:3])),
        "final_ori_err_rad": float(np.linalg.norm(delta[3:])),
    }


def summarize_results(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    for (mode, nr_flag), g in df.groupby(["mode", "enable_nr"]):
        rows.append(
            {
                "mode": mode,
                "enable_nr": nr_flag,
                "success_rate_tol": float(g["success_tol"].mean()),
                "nr_converged_rate": float(g["nr_converged"].mean()),
                "mean_final_pos_err_mm": float(g["final_pos_err_mm"].mean()),
                "median_final_pos_err_mm": float(g["final_pos_err_mm"].median()),
                "mean_final_ori_err_rad": float(g["final_ori_err_rad"].mean()),
                "median_final_ori_err_rad": float(g["final_ori_err_rad"].median()),
                "mean_total_ms": float(g["total_ms"].mean()),
                "median_total_ms": float(g["total_ms"].median()),
                "mean_candidate_count": float(g["candidate_count"].mean()),
                "mean_nr_iters": float(g["nr_iters"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["mode", "enable_nr"]).reset_index(drop=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def plot_benchmark_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    summary_df = summary_df.copy()
    summary_df["label"] = summary_df.apply(
        lambda r: f"{r['mode']} + {'NR' if r['enable_nr'] else 'No NR'}",
        axis=1,
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), constrained_layout=True)

    sns.barplot(
        data=summary_df,
        x="label",
        y="success_rate_tol",
        hue="label",
        dodge=False,
        palette="Set2",
        ax=axes[0],
    )
    axes[0].set_title("工程成功率 / Engineering Success Rate")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("成功率 / Success Rate")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set_ylim(0.0, 1.05)
    leg = axes[0].get_legend()
    if leg is not None:
        leg.remove()

    sns.barplot(
        data=summary_df,
        x="label",
        y="median_final_pos_err_mm",
        hue="label",
        dodge=False,
        palette="Set2",
        ax=axes[1],
    )
    axes[1].set_title("最终位置误差中位数 / Median Final Position Error")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("误差 (mm) / Error")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_yscale("log")
    leg = axes[1].get_legend()
    if leg is not None:
        leg.remove()

    sns.barplot(
        data=summary_df,
        x="label",
        y="median_total_ms",
        hue="label",
        dodge=False,
        palette="Set2",
        ax=axes[2],
    )
    axes[2].set_title("总时间中位数 / Median Total Time")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("时间 (ms) / Time")
    axes[2].tick_params(axis="x", rotation=15)
    leg = axes[2].get_legend()
    if leg is not None:
        leg.remove()

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_benchmark_error_box(df: pd.DataFrame, out_path: Path) -> None:
    df = df.copy()
    df["label"] = df.apply(lambda r: f"{r['mode']} + {'NR' if r['enable_nr'] else 'No NR'}", axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)

    sns.boxplot(
        data=df,
        x="label",
        y="final_pos_err_mm",
        hue="label",
        dodge=False,
        palette="Set3",
        ax=axes[0],
        showfliers=False,
    )
    axes[0].set_title("最终位置误差分布 / Final Position Error Distribution")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("误差 (mm) / Error")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set_yscale("log")
    leg = axes[0].get_legend()
    if leg is not None:
        leg.remove()

    sns.boxplot(
        data=df,
        x="label",
        y="total_ms",
        hue="label",
        dodge=False,
        palette="Set3",
        ax=axes[1],
        showfliers=False,
    )
    axes[1].set_title("推理总时间分布 / Total Inference Time Distribution")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("时间 (ms) / Time")
    axes[1].tick_params(axis="x", rotation=15)
    leg = axes[1].get_legend()
    if leg is not None:
        leg.remove()

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def update_manifest(paths: List[str]) -> None:
    readme = FIGURE_DIR / "README.md"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else "# Figure Outputs\n"
    additions = [f"- `{p}`" for p in paths]
    lines = existing.rstrip().splitlines()
    for item in additions:
        if item not in lines:
            lines.append(item)
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.n_samples <= 0:
        raise ValueError("--n_samples must be positive.")
    if args.success_pos_mm <= 0 or args.success_ori_rad <= 0:
        raise ValueError("Success thresholds must be positive.")

    configure_style()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    n_samples = args.n_samples
    tag = args.tag.strip() or f"n{n_samples}"
    poses = build_pose_batch(n_samples=n_samples, seed=args.seed)
    records = []
    configs = [
        ("flat", False),
        ("flat", True),
        ("hierarchical", False),
        ("hierarchical", True),
    ]
    for idx, pose6 in enumerate(poses):
        print(f"[benchmark] sample {idx + 1}/{n_samples}")
        for mode, enable_nr in configs:
            result = run_predict_ik(pose6, mode=mode, enable_nr=enable_nr)
            refined = result.get("refined_solution")
            if refined is not None:
                q_final = np.asarray(refined["q_deg"], dtype=np.float32)
                nr_converged = bool(refined["nr_converged"])
                nr_iters = int(refined["nr_iters"])
            else:
                q_final = np.asarray(result["initial_solution"]["q0_deg"], dtype=np.float32)
                nr_converged = False
                nr_iters = 0
            errs = evaluate_final_pose_error(pose6, q_final)
            success_tol = (
                errs["final_pos_err_mm"] <= args.success_pos_mm
                and errs["final_ori_err_rad"] <= args.success_ori_rad
            )

            records.append(
                {
                    "sample_id": idx,
                    "mode": mode,
                    "enable_nr": enable_nr,
                    "candidate_count": len(result["candidate_subspaces"]),
                    "initial_pos_err_mm": float(result["initial_solution"]["position_l2_mm"]),
                    "final_pos_err_mm": errs["final_pos_err_mm"],
                    "final_ori_err_rad": errs["final_ori_err_rad"],
                    "success_tol": float(success_tol),
                    "nr_converged": float(nr_converged),
                    "nr_iters": nr_iters,
                    "total_ms": float(result["timing_breakdown_ms"]["total_ms"]),
                }
            )

    df = pd.DataFrame(records)
    detail_path = DATA_DIR / f"ik_benchmark_detailed_{tag}.csv"
    summary_path = DATA_DIR / f"ik_benchmark_summary_{tag}.csv"
    fig_summary_path = FIGURES_DIR / f"ik_benchmark_summary_{tag}.png"
    fig_distribution_path = FIGURES_DIR / f"ik_benchmark_distribution_{tag}.png"

    df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df = summarize_results(df, summary_path)
    plot_benchmark_summary(summary_df, fig_summary_path)
    plot_benchmark_error_box(df, fig_distribution_path)
    update_manifest(
        [
            f"figures/{fig_summary_path.name}",
            f"figures/{fig_distribution_path.name}",
        ]
    )
    print(f"Saved benchmark results to: {DATA_DIR}")
    print(f"Saved benchmark figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
