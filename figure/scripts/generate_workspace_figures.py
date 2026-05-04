#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abb_nn.branching import branch_label_to_name, subspace_to_branch_label

FIGURE_DIR = ROOT / "figure"
DATA_DIR = Path(os.environ.get("ABB_FIGURE_DATA_DIR", str(FIGURE_DIR / "data"))).resolve()
FIGURES_DIR = Path(os.environ.get("ABB_FIGURE_OUTPUT_DIR", str(FIGURE_DIR / "figures"))).resolve()
DEFAULT_REFERENCE_DIR = ROOT / "data" / "subspace_reference_abb_strict_samples512_seed2026"


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
            "legend.fontsize": 7,
        }
    )


def load_reference_points(reference_dir: Path, max_points_per_subspace: int = 128) -> pd.DataFrame:
    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference data directory not found: {reference_dir}")

    rows = []
    rng = np.random.default_rng(2026)
    for file_path in sorted(reference_dir.glob("subspace_*_reference.npz")):
        sid = int(file_path.stem.split("_")[1])
        branch = subspace_to_branch_label(sid, segment_profile="abb_strict")
        arr = np.load(file_path)
        pose6 = arr["pose6"]
        if pose6.shape[0] > max_points_per_subspace:
            idx = rng.choice(pose6.shape[0], size=max_points_per_subspace, replace=False)
            pose6 = pose6[idx]
        for p in pose6:
            rows.append(
                {
                    "subspace_id": sid,
                    "branch_label": branch,
                    "branch_name": branch_label_to_name(branch),
                    "x_mm": float(p[0]),
                    "y_mm": float(p[1]),
                    "z_mm": float(p[2]),
                }
            )
    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "workspace_reference_points_sampled.csv", index=False, encoding="utf-8-sig")
    return df


def plot_workspace_projections(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    palette = sns.color_palette("tab20", n_colors=df["branch_label"].nunique())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    projection_specs = [
        ("x_mm", "z_mm", "X-Z 投影 / X-Z Projection", "X (mm)", "Z (mm)"),
        ("x_mm", "y_mm", "X-Y 投影 / X-Y Projection", "X (mm)", "Y (mm)"),
        ("y_mm", "z_mm", "Y-Z 投影 / Y-Z Projection", "Y (mm)", "Z (mm)"),
    ]

    for ax, (x_col, y_col, title, xlabel, ylabel) in zip(axes, projection_specs):
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue="branch_label",
            palette=palette,
            s=5,
            linewidth=0,
            alpha=0.45,
            legend=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("子空间参考样本末端位置投影 / Workspace Projections of Reference Samples", fontsize=11)
    fig.savefig(FIGURES_DIR / "workspace_reference_projections.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_workspace_3d(df: pd.DataFrame, max_points: int = 12000) -> None:
    if len(df) > max_points:
        df_plot = df.sample(n=max_points, random_state=2026).copy()
    else:
        df_plot = df.copy()

    fig = plt.figure(figsize=(7.2, 5.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    colors = [cmap(int(b) % 20) for b in df_plot["branch_label"].to_numpy()]
    ax.scatter(
        df_plot["x_mm"],
        df_plot["y_mm"],
        df_plot["z_mm"],
        c=colors,
        s=3,
        alpha=0.45,
        linewidths=0,
    )
    ax.set_title("样本可达空间 / Reachable Workspace of Samples")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=24, azim=-48)
    fig.savefig(FIGURES_DIR / "workspace_reference_3d.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def update_manifest() -> None:
    readme = FIGURE_DIR / "README.md"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else "# Figure Outputs\n"
    additions = [
        "- `figures/workspace_reference_projections.png`",
        "- `figures/workspace_reference_3d.png`",
    ]
    lines = existing.rstrip().splitlines()
    for item in additions:
        if item not in lines:
            lines.append(item)
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate workspace figures from saved reference samples.")
    parser.add_argument("--reference_dir", default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--max_points_per_subspace", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_style()
    reference_dir = Path(args.reference_dir)
    df = load_reference_points(reference_dir=reference_dir, max_points_per_subspace=int(args.max_points_per_subspace))
    plot_workspace_projections(df)
    plot_workspace_3d(df)
    update_manifest()
    print(f"Saved workspace figures to: {FIGURES_DIR}")
    print(f"Reference sample directory: {reference_dir.resolve()}")


if __name__ == "__main__":
    main()
