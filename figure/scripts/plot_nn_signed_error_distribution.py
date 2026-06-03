#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fk_model import pose6_from_q


FIGURE_DIR = ROOT / "figure"
DATA_DIR = Path(os.environ.get("ABB_FIGURE_DATA_DIR", str(FIGURE_DIR / "data"))).resolve()
FIGURES_DIR = Path(os.environ.get("ABB_FIGURE_OUTPUT_DIR", str(FIGURE_DIR / "figures"))).resolve()
DEFAULT_DETAIL_CSV = DATA_DIR / "ik_benchmark_six_methods_detailed_n100.csv"


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def parse_vec(text: str) -> np.ndarray:
    return np.asarray([float(x) for x in str(text).split(",")], dtype=float)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    sns.set_theme(style="white")
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
        }
    )


def build_component_error_table(detail_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(detail_csv)
    df = df[df["method"].isin(["nn_only", "nn_nr"])].copy()
    if df.empty:
        raise RuntimeError("No nn_only / nn_nr rows found in benchmark detail CSV.")

    rows = []
    for row in df.itertuples(index=False):
        target_pose6 = parse_vec(row.target_pose6)
        q_result_deg = parse_vec(row.q_result_deg)
        pred_pose6 = pose6_from_q(q_result_deg, input_unit="deg")
        delta = pred_pose6 - target_pose6
        delta[3:] = wrap_to_pi(delta[3:])

        rows.append(
            {
                "sample_id": int(row.sample_id),
                "method": str(row.method),
                "dx_mm": float(delta[0]),
                "dy_mm": float(delta[1]),
                "dz_mm": float(delta[2]),
                "dphi_deg": float(np.rad2deg(delta[3])),
                "dtheta_deg": float(np.rad2deg(delta[4])),
                "dpsi_deg": float(np.rad2deg(delta[5])),
            }
        )

    out = pd.DataFrame(rows).sort_values(["method", "sample_id"]).reset_index(drop=True)
    return out


def plot_method_boxplots(ax: plt.Axes, data: pd.DataFrame, columns: list[str], labels: list[str]) -> None:
    values = [data[col].to_numpy(dtype=float) for col in columns]
    flierprops = dict(marker="+", markerfacecolor="red", markeredgecolor="red", markersize=4, linestyle="none")
    boxprops = dict(color="blue", linewidth=0.8)
    whiskerprops = dict(color="black", linewidth=0.8, linestyle="--")
    capprops = dict(color="black", linewidth=0.8)
    medianprops = dict(color="red", linewidth=0.8)
    ax.boxplot(
        values,
        tick_labels=labels,
        widths=0.45,
        patch_artist=False,
        flierprops=flierprops,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
    )
    ax.tick_params(axis="x", rotation=0)


def render_figure(component_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 11.2), constrained_layout=True)

    nn_only = component_df[component_df["method"] == "nn_only"].copy()
    nn_nr = component_df[component_df["method"] == "nn_nr"].copy()

    plot_method_boxplots(axes[0, 0], nn_only, ["dx_mm", "dy_mm", "dz_mm"], ["X", "Y", "Z"])
    axes[0, 0].set_title("Signed position errors of NN initial solutions")
    axes[0, 0].set_xlabel("Degree of freedom")
    axes[0, 0].set_ylabel("Signed position error (mm)")

    plot_method_boxplots(axes[0, 1], nn_only, ["dphi_deg", "dtheta_deg", "dpsi_deg"], ["phi", "theta", "psi"])
    axes[0, 1].set_title("Signed orientation errors of NN initial solutions")
    axes[0, 1].set_xlabel("Degree of freedom")
    axes[0, 1].set_ylabel("Signed orientation error (deg)")

    plot_method_boxplots(axes[1, 0], nn_nr, ["dx_mm", "dy_mm", "dz_mm"], ["X", "Y", "Z"])
    axes[1, 0].set_title("Signed position errors of NN+NR solutions")
    axes[1, 0].set_xlabel("Degree of freedom")
    axes[1, 0].set_ylabel("Signed position error (mm)")

    plot_method_boxplots(axes[1, 1], nn_nr, ["dphi_deg", "dtheta_deg", "dpsi_deg"], ["phi", "theta", "psi"])
    axes[1, 1].set_title("Signed orientation errors of NN+NR solutions")
    axes[1, 1].set_xlabel("Degree of freedom")
    axes[1, 1].set_ylabel("Signed orientation error (deg)")

    for ax in axes.flat:
        ax.grid(False)

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot signed error distributions for nn_only and nn_nr.")
    parser.add_argument("--detail_csv", default=str(DEFAULT_DETAIL_CSV))
    parser.add_argument(
        "--out_png",
        default=str(FIGURES_DIR / "ik_benchmark_nn_signed_error_distribution_n100.png"),
    )
    parser.add_argument(
        "--out_csv",
        default=str(DATA_DIR / "ik_benchmark_nn_signed_error_components_n100.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dirs()
    configure_style()

    component_df = build_component_error_table(Path(args.detail_csv))
    component_df.to_csv(Path(args.out_csv), index=False, encoding="utf-8-sig")
    render_figure(component_df, Path(args.out_png))

    print(f"Saved signed component data to: {Path(args.out_csv).resolve()}")
    print(f"Saved figure to: {Path(args.out_png).resolve()}")


if __name__ == "__main__":
    main()
