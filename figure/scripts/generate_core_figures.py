#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURE_DIR = ROOT / "figure"
DATA_DIR = FIGURE_DIR / "data"
FIGURES_DIR = FIGURE_DIR / "figures"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def save_figure(fig: plt.Figure, stem: str) -> None:
    png_path = FIGURES_DIR / f"{stem}.png"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_fk_validation_tables() -> pd.DataFrame:
    report = load_json(ARTIFACTS_DIR / "fk_validation" / "fk_validation_report.json")
    rows = []
    for item in report["official_workspace_validation"]["theta2_offset_hypotheses"]:
        rows.append(
            {
                "theta2_offset_deg": item["theta2_offset_deg"],
                "mean_xz_error_mm": item["mean_xz_error_mm"],
                "max_xz_error_mm": item["max_xz_error_mm"],
            }
        )
    df = pd.DataFrame(rows).sort_values("theta2_offset_deg").reset_index(drop=True)
    df.to_csv(DATA_DIR / "fk_theta2_offset_validation.csv", index=False, encoding="utf-8-sig")
    return df


def plot_fk_theta2_offset_validation(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), constrained_layout=True)
    color_main = "#2A5CAA"
    color_aux = "#C95F46"

    sns.barplot(
        data=df,
        x="theta2_offset_deg",
        y="mean_xz_error_mm",
        ax=axes[0],
        color=color_main,
    )
    axes[0].set_title("关节2偏置假设平均误差 / Mean XZ Error by $\\theta_2$ Offset")
    axes[0].set_xlabel("关节2偏置 (deg) / Joint-2 Offset")
    axes[0].set_ylabel("平均XZ误差 (mm) / Mean XZ Error")

    sns.barplot(
        data=df,
        x="theta2_offset_deg",
        y="max_xz_error_mm",
        ax=axes[1],
        color=color_aux,
    )
    axes[1].set_title("关节2偏置假设最大误差 / Max XZ Error by $\\theta_2$ Offset")
    axes[1].set_xlabel("关节2偏置 (deg) / Joint-2 Offset")
    axes[1].set_ylabel("最大XZ误差 (mm) / Max XZ Error")

    save_figure(fig, "fk_theta2_offset_validation")


def export_subspace_prediction_metrics() -> pd.DataFrame:
    meta = load_json(ARTIFACTS_DIR / "prediction_system_formal" / "metadata.json")
    rows = []
    for item in meta["trained_subspaces"]:
        rows.append(
            {
                "subspace_id": item["subspace_id"],
                "val_loss_q15_deg2": item["val_loss_q15"],
                "val_loss_q6_deg2": item["val_loss_q6"],
                "test_pos_l2_mean_mm": item["test_pos_l2_mean_mm"],
                "e_max_mm": item["e_max"],
            }
        )
    df = pd.DataFrame(rows).sort_values("subspace_id").reset_index(drop=True)
    df.to_csv(DATA_DIR / "prediction_subspace_metrics.csv", index=False, encoding="utf-8-sig")
    return df


def export_subspace_profiles() -> pd.DataFrame:
    obj = load_json(ARTIFACTS_DIR / "subspace_validation" / "subspace_profiles.json")
    rows = []
    joint_names = ["q1", "q2", "q3", "q4", "q5", "q6"]
    for profile in obj["profiles"]:
        for joint_name, bins in zip(joint_names, profile["joint_bins"]):
            rows.append(
                {
                    "profile": profile["profile"],
                    "joint": joint_name,
                    "bins": bins,
                    "subspace_count": profile["subspace_count"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "subspace_profile_summary.csv", index=False, encoding="utf-8-sig")
    return df


def plot_subspace_profile_comparison(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)

    sns.barplot(data=df, x="joint", y="bins", hue="profile", ax=axes[0], palette="Set2")
    axes[0].set_title("关节分段数对比 / Joint Bin Count by Profile")
    axes[0].set_xlabel("关节 / Joint")
    axes[0].set_ylabel("分段数 / Number of Bins")
    axes[0].legend(title="Profile", frameon=True)

    total_df = df[["profile", "subspace_count"]].drop_duplicates().reset_index(drop=True)
    sns.barplot(
        data=total_df,
        x="profile",
        y="subspace_count",
        hue="profile",
        dodge=False,
        ax=axes[1],
        palette=["#E4B363", "#2A5CAA"],
    )
    axes[1].set_title("总子空间数量 / Total Number of Subspaces")
    axes[1].set_xlabel("划分方案 / Profile")
    axes[1].set_ylabel("子空间总数 / Total Subspaces")
    leg = axes[1].get_legend()
    if leg is not None:
        leg.remove()

    save_figure(fig, "subspace_profile_comparison")


def plot_prediction_error_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)

    sns.histplot(
        data=df,
        x="test_pos_l2_mean_mm",
        bins=18,
        kde=True,
        color="#2A5CAA",
        ax=axes[0],
    )
    axes[0].set_title("子空间平均位置误差分布 / Distribution of Mean Position Error")
    axes[0].set_xlabel("测试集平均位置误差 (mm) / Mean Test Position Error")
    axes[0].set_ylabel("子空间数量 / Number of Subspaces")

    sns.boxplot(
        y=df["test_pos_l2_mean_mm"],
        color="#E4B363",
        width=0.35,
        ax=axes[1],
    )
    axes[1].set_title("子空间平均位置误差箱线图 / Boxplot of Mean Position Error")
    axes[1].set_ylabel("测试集平均位置误差 (mm) / Mean Test Position Error")
    axes[1].set_xlabel("")

    save_figure(fig, "prediction_subspace_error_distribution")


def plot_prediction_error_rank(df: pd.DataFrame) -> None:
    ranked = df.sort_values("test_pos_l2_mean_mm").reset_index(drop=True).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    fig, ax = plt.subplots(figsize=(8.4, 3.5), constrained_layout=True)
    ax.plot(
        ranked["rank"],
        ranked["test_pos_l2_mean_mm"],
        color="#2A5CAA",
        linewidth=1.8,
    )
    ax.fill_between(
        ranked["rank"],
        ranked["test_pos_l2_mean_mm"],
        color="#2A5CAA",
        alpha=0.15,
    )
    ax.set_title("192个子空间平均位置误差排序 / Ranked Mean Position Error Across 192 Subspaces")
    ax.set_xlabel("子空间排序 / Ranked Subspace Index")
    ax.set_ylabel("测试集平均位置误差 (mm) / Mean Test Position Error")
    save_figure(fig, "prediction_subspace_error_rank")


def export_classification_metrics() -> pd.DataFrame:
    flat = load_json(ARTIFACTS_DIR / "classification_system_formal" / "metadata.json")
    branch = load_json(ARTIFACTS_DIR / "branch_classification_system" / "metadata.json")
    fine = load_json(ARTIFACTS_DIR / "fine_classification_system" / "metadata.json")

    rows = []
    for item in flat["models"]:
        rows.append(
            {
                "family": "flat_192",
                "variant": item["variant"],
                "metric": "top1_acc",
                "value": item["best_val_acc"],
            }
        )
    for item in branch["models"]:
        rows.extend(
            [
                {
                    "family": "branch",
                    "variant": item["variant"],
                    "metric": "joint_acc",
                    "value": item["best_val_acc_joint"],
                },
                {
                    "family": "branch",
                    "variant": item["variant"],
                    "metric": "shoulder_acc",
                    "value": item["best_val_acc_shoulder"],
                },
                {
                    "family": "branch",
                    "variant": item["variant"],
                    "metric": "elbow_acc",
                    "value": item["best_val_acc_elbow"],
                },
                {
                    "family": "branch",
                    "variant": item["variant"],
                    "metric": "wrist_acc",
                    "value": item["best_val_acc_wrist"],
                },
            ]
        )
    for item in fine["models"]:
        rows.extend(
            [
                {
                    "family": "fine_16",
                    "variant": item["variant"],
                    "metric": "top1_acc",
                    "value": item["best_val_acc_top1"],
                },
                {
                    "family": "fine_16",
                    "variant": item["variant"],
                    "metric": "top3_acc",
                    "value": item["best_val_acc_top3"],
                },
            ]
        )

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "classification_metrics_summary.csv", index=False, encoding="utf-8-sig")
    return df


def plot_classification_comparison(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), constrained_layout=True)

    flat_df = df[df["family"] == "flat_192"].copy()
    sns.barplot(data=flat_df, x="variant", y="value", color="#C95F46", ax=axes[0])
    axes[0].set_title("旧版192类分类器 / Flat 192-Class Classifier")
    axes[0].set_xlabel("模型变体 / Variant")
    axes[0].set_ylabel("验证准确率 / Validation Accuracy")
    axes[0].set_ylim(0.0, max(0.2, flat_df["value"].max() * 1.2))

    branch_df = df[df["family"] == "branch"].copy()
    branch_map = {
        "joint_acc": "Joint",
        "shoulder_acc": "Shoulder",
        "elbow_acc": "Elbow",
        "wrist_acc": "Wrist",
    }
    branch_df["metric_label"] = branch_df["metric"].map(branch_map)
    sns.barplot(data=branch_df, x="metric_label", y="value", hue="variant", palette="Blues", ax=axes[1])
    axes[1].set_title("第一层粗分类器 / Branch Classifier")
    axes[1].set_xlabel("指标 / Metric")
    axes[1].set_ylabel("验证准确率 / Validation Accuracy")
    axes[1].set_ylim(0.0, 0.8)
    axes[1].legend(title="Variant", frameon=True)
    fine_df = df[df["family"] == "fine_16"].copy()
    fine_map = {"top1_acc": "Top-1", "top3_acc": "Top-3"}
    fine_df["metric_label"] = fine_df["metric"].map(fine_map)
    sns.barplot(data=fine_df, x="metric_label", y="value", hue="variant", palette="Greens", ax=axes[2])
    axes[2].set_title("第二层细分类器 / Fine Classifier")
    axes[2].set_xlabel("指标 / Metric")
    axes[2].set_ylabel("验证准确率 / Validation Accuracy")
    axes[2].set_ylim(0.0, 0.9)
    axes[2].legend(title="Variant", frameon=True)
    save_figure(fig, "classification_hierarchical_comparison")


def export_single_case_metrics() -> pd.DataFrame:
    obj = load_json(ARTIFACTS_DIR / "fine_classification_system" / "test_pose_001_full_ik.json")
    rows = [
        {"group": "timing", "item": "candidate_generation_ms", "value": obj["timing_breakdown_ms"]["candidate_generation_ms"]},
        {"group": "timing", "item": "branch_classification_ms", "value": obj["timing_breakdown_ms"]["branch_classification_ms"]},
        {"group": "timing", "item": "fine_classification_ms", "value": obj["timing_breakdown_ms"]["fine_classification_ms"]},
        {"group": "timing", "item": "initial_selection_ms", "value": obj["timing_breakdown_ms"]["initial_selection_ms"]},
        {"group": "timing", "item": "nr_refinement_ms", "value": obj["timing_breakdown_ms"]["nr_refinement_ms"]},
        {"group": "timing", "item": "total_ms", "value": obj["timing_breakdown_ms"]["total_ms"]},
        {"group": "error", "item": "initial_pos_err_mm", "value": obj["initial_solution"]["position_l2_mm"]},
        {"group": "error", "item": "refined_pos_err_mm", "value": obj["refined_solution"]["final_pos_err_mm"]},
        {"group": "error", "item": "refined_ori_err_rad", "value": obj["refined_solution"]["final_ori_err_rad"]},
        {"group": "iters", "item": "nr_iters", "value": obj["refined_solution"]["nr_iters"]},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "single_case_ik_metrics.csv", index=False, encoding="utf-8-sig")
    return df


def plot_single_case_metrics(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)

    timing_df = df[df["group"] == "timing"].copy()
    timing_df = timing_df[timing_df["item"] != "total_ms"]
    timing_map = {
        "candidate_generation_ms": "候选生成 / Candidate",
        "branch_classification_ms": "粗分类 / Branch",
        "fine_classification_ms": "细分类 / Fine",
        "initial_selection_ms": "初值筛选 / Initial Select",
        "nr_refinement_ms": "NR修正 / NR Refine",
    }
    timing_df["label"] = timing_df["item"].map(timing_map)
    sns.barplot(data=timing_df, y="label", x="value", color="#2A5CAA", ax=axes[0])
    axes[0].set_title("单样本推理时间分解 / Timing Breakdown of One IK Case")
    axes[0].set_xlabel("时间 (ms) / Time")
    axes[0].set_ylabel("")

    err_items = [
        ("initial_pos_err_mm", "初始位置误差(mm)\nInitial Pos Error"),
        ("refined_pos_err_mm", "修正后位置误差(mm)\nRefined Pos Error"),
        ("refined_ori_err_rad", "修正后姿态误差(rad)\nRefined Ori Error"),
    ]
    err_vals = [float(df.loc[df["item"] == key, "value"].iloc[0]) for key, _ in err_items]
    err_labels = [label for _, label in err_items]
    axes[1].bar(err_labels, err_vals, color=["#C95F46", "#4E9F71", "#B585D6"])
    axes[1].set_title("单样本误差结果 / Error Metrics of One IK Case")
    axes[1].set_ylabel("数值 / Value")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].set_yscale("log")

    save_figure(fig, "single_case_ik_metrics")


def write_manifest(created_files: Iterable[str]) -> None:
    lines = [
        "# Figure Outputs",
        "",
        "本目录由 `figure/scripts/generate_core_figures.py` 自动生成。",
        "",
        "## 已生成图表",
    ]
    lines.extend([f"- `{name}`" for name in created_files])
    (FIGURE_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_style()

    fk_df = export_fk_validation_tables()
    plot_fk_theta2_offset_validation(fk_df)

    subspace_df = export_subspace_profiles()
    plot_subspace_profile_comparison(subspace_df)

    pred_df = export_subspace_prediction_metrics()
    plot_prediction_error_distribution(pred_df)
    plot_prediction_error_rank(pred_df)

    cls_df = export_classification_metrics()
    plot_classification_comparison(cls_df)

    case_df = export_single_case_metrics()
    plot_single_case_metrics(case_df)

    write_manifest(
        [
            "figures/fk_theta2_offset_validation.png",
            "figures/subspace_profile_comparison.png",
            "figures/prediction_subspace_error_distribution.png",
            "figures/prediction_subspace_error_rank.png",
            "figures/classification_hierarchical_comparison.png",
            "figures/single_case_ik_metrics.png",
        ]
    )

    print(f"Saved figure data to: {DATA_DIR}")
    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
