#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obstacle_avoidance.collision import ObstacleScene
from obstacle_avoidance.planning import evaluate_trajectory_against_scene

FIGURE_DIR = ROOT / "figure"
FIGURES_DIR = FIGURE_DIR / "figures"
ARTIFACTS_DIR = ROOT / "artifacts"


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def box_faces(box_min: np.ndarray, box_max: np.ndarray) -> list[list[list[float]]]:
    x0, y0, z0 = box_min.tolist()
    x1, y1, z1 = box_max.tolist()
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    return [
        vertices[[0, 1, 2, 3]].tolist(),
        vertices[[4, 5, 6, 7]].tolist(),
        vertices[[0, 1, 5, 4]].tolist(),
        vertices[[2, 3, 7, 6]].tolist(),
        vertices[[1, 2, 6, 5]].tolist(),
        vertices[[0, 3, 7, 4]].tolist(),
    ]


def pick_collision_candidate(payload: dict) -> dict:
    collision_items = [
        item
        for item in payload.get("evaluated_candidates", [])
        if bool(item["trajectory_summary"]["collision"])
    ]
    if not collision_items:
        raise ValueError("No colliding candidate found in planning result.")
    return min(collision_items, key=lambda item: float(item["selection"]["selection_cost"]))


def extract_joint_series(frames: list[dict]) -> np.ndarray:
    series = []
    for frame in frames:
        series.append(np.asarray(frame["joint_points_mm"], dtype=float))
    return np.asarray(series, dtype=float)


def extract_tool_path(frames: list[dict]) -> np.ndarray:
    return extract_joint_series(frames)[:, -1, :]


def load_planning_payload() -> tuple[dict, dict, dict]:
    payload = load_json(ARTIFACTS_DIR / "obstacle_avoidance" / "open_space_reselect_demo_plan.json")
    selected = payload["selected_solution"]
    selected_summary = selected["trajectory_summary"]
    if "frames" not in selected_summary:
        raise ValueError("Selected solution does not contain frame data.")

    collision_candidate = pick_collision_candidate(payload)
    collision_summary = evaluate_trajectory_against_scene(
        q_start_deg=payload["q_start_deg"],
        q_goal_deg=collision_candidate["q_goal_deg"],
        scene=ObstacleScene.from_dict(payload["scene"]),
        steps=int(collision_candidate["trajectory_summary"]["trajectory_steps"]),
        include_frames=True,
    )
    return payload, selected, {"candidate": collision_candidate, "summary": collision_summary}


def plot_scene_overview(payload: dict, selected: dict, collision_case: dict) -> None:
    selected_frames = selected["trajectory_summary"]["frames"]
    collision_summary = collision_case["summary"]
    collision_candidate = collision_case["candidate"]

    selected_path = extract_tool_path(selected_frames)
    collision_path = extract_tool_path(collision_summary["frames"])
    selected_joints = extract_joint_series(selected_frames)
    collision_joints = extract_joint_series(collision_summary["frames"])
    obstacles = payload["scene"]["obstacles"]
    start_point = selected_path[0]
    target = np.asarray(payload["target_pose6"][:3], dtype=float)

    fig = plt.figure(figsize=(10.6, 5.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    for obstacle in obstacles:
        faces = box_faces(np.asarray(obstacle["min_mm"], dtype=float), np.asarray(obstacle["max_mm"], dtype=float))
        poly = Poly3DCollection(
            faces,
            facecolors="#F4A261",
            edgecolors="#8C4F1F",
            linewidths=0.9,
            alpha=0.26,
        )
        ax.add_collection3d(poly)
        center = np.asarray(obstacle["center_mm"], dtype=float)
        ax.text(center[0], center[1], center[2], obstacle["name"], fontsize=8, color="#7C3E12")

    ax.plot(
        collision_path[:, 0],
        collision_path[:, 1],
        collision_path[:, 2],
        color="#D1495B",
        linewidth=2.0,
        label=f"Colliding candidate (subspace {collision_candidate['subspace_id']})",
    )
    ax.plot(
        selected_path[:, 0],
        selected_path[:, 1],
        selected_path[:, 2],
        color="#2A9D8F",
        linewidth=2.4,
        label=f"Selected trajectory (subspace {selected['subspace_id']})",
    )

    snapshot_indices = [0, len(selected_joints) // 2, len(selected_joints) - 1]
    snapshot_colors = ["#264653", "#3A86FF", "#1D6F63"]
    for idx, color in zip(snapshot_indices, snapshot_colors):
        joints = selected_joints[idx]
        ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], color=color, linewidth=1.2, alpha=0.95)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=8, alpha=0.95)

    colliding_joints = collision_joints[min(max(collision_summary["first_collision_frame"], 0), len(collision_joints) - 1)]
    ax.plot(
        colliding_joints[:, 0],
        colliding_joints[:, 1],
        colliding_joints[:, 2],
        color="#B22222",
        linewidth=1.2,
        linestyle="--",
        alpha=0.9,
    )

    ax.scatter([start_point[0]], [start_point[1]], [start_point[2]], color="#264653", s=42, label="Start point")
    ax.scatter(
        [target[0]],
        [target[1]],
        [target[2]],
        color="#E9C46A",
        edgecolors="#7A5C00",
        s=52,
        label="Target point",
    )

    ax.set_title("Obstacle scene and candidate trajectory comparison")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=23, azim=-58)
    ax.legend(loc="upper left", frameon=True)

    all_points = np.vstack([selected_path, collision_path, target.reshape(1, 3)])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0)
    margin = 0.12 * spans
    ax.set_xlim(mins[0] - margin[0], maxs[0] + margin[0])
    ax.set_ylim(mins[1] - margin[1], maxs[1] + margin[1])
    ax.set_zlim(mins[2] - margin[2], maxs[2] + margin[2])

    note = (
        f"Selected trajectory minimum clearance: {selected['trajectory_summary']['min_clearance_mm']:.2f} mm; "
        f"colliding candidate collision frames: {collision_summary['collision_frame_count']}"
    )
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=8.5, color="#4B5563")
    fig.savefig(FIGURES_DIR / "chapter4_scene_overview.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_clearance_curve(payload: dict, selected: dict, collision_case: dict) -> None:
    selected_frames = selected["trajectory_summary"]["frames"]
    collision_summary = collision_case["summary"]
    collision_candidate = collision_case["candidate"]

    selected_clearance = np.asarray([frame["min_clearance_mm"] for frame in selected_frames], dtype=float)
    collision_clearance = np.asarray([frame["min_clearance_mm"] for frame in collision_summary["frames"]], dtype=float)
    selected_steps = np.arange(len(selected_clearance))
    collision_steps = np.arange(len(collision_clearance))

    fig, ax = plt.subplots(figsize=(9.8, 4.5), constrained_layout=True)
    ax.plot(
        collision_steps,
        collision_clearance,
        color="#D1495B",
        linewidth=2.0,
        label=f"Colliding candidate (subspace {collision_candidate['subspace_id']})",
    )
    ax.plot(
        selected_steps,
        selected_clearance,
        color="#2A9D8F",
        linewidth=2.2,
        label=f"Selected trajectory (subspace {selected['subspace_id']})",
    )
    ax.axhline(0.0, color="#6B7280", linestyle="--", linewidth=1.0, label="Collision threshold")

    collision_frame = int(collision_summary["first_collision_frame"])
    if collision_frame >= 0:
        ax.axvline(collision_frame, color="#B22222", linestyle=":", linewidth=1.1)
        ax.text(
            collision_frame,
            np.nanmin(collision_clearance) + 10.0,
            f"First collision frame: {collision_frame}",
            color="#B22222",
            fontsize=8,
            ha="left",
            va="bottom",
        )

    ax.set_title("Minimum clearance variation along trajectory")
    ax.set_xlabel("Interpolation frame index")
    ax.set_ylabel("Minimum clearance (mm)")
    ax.legend(loc="best", frameon=True)

    text = (
        f"Selected minimum clearance = {selected['trajectory_summary']['min_clearance_mm']:.2f} mm, "
        f"collision candidate minimum clearance = {collision_summary['min_clearance_mm']:.2f} mm"
    )
    fig.text(0.5, 0.02, text, ha="center", va="bottom", fontsize=8.5, color="#4B5563")
    fig.savefig(FIGURES_DIR / "chapter4_clearance_curve.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_candidate_reselection_metrics(payload: dict, selected: dict) -> None:
    candidates = payload["evaluated_candidates"]
    labels = [f"S{item['subspace_id']}" for item in candidates]
    selection_cost = np.asarray([item["selection"]["selection_cost"] for item in candidates], dtype=float)
    min_clearance = np.asarray([item["trajectory_summary"]["min_clearance_mm"] for item in candidates], dtype=float)
    collision_frames = np.asarray([item["trajectory_summary"]["collision_frame_count"] for item in candidates], dtype=float)
    joint_path = np.asarray([item["trajectory_summary"]["joint_path_length_deg"] for item in candidates], dtype=float)
    selected_sid = int(selected["subspace_id"])
    colors = ["#2A9D8F" if item["subspace_id"] == selected_sid else "#A8B0B9" for item in candidates]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    axes = axes.reshape(-1)

    axes[0].bar(labels, selection_cost, color=colors, edgecolor="#4B5563", linewidth=0.4)
    axes[0].set_title("Selection cost")
    axes[0].set_ylabel("Cost value")
    axes[0].set_yscale("log")

    axes[1].bar(labels, min_clearance, color=colors, edgecolor="#4B5563", linewidth=0.4)
    axes[1].axhline(0.0, color="#6B7280", linestyle="--", linewidth=1.0)
    axes[1].set_title("Minimum clearance")
    axes[1].set_ylabel("Clearance (mm)")

    axes[2].bar(labels, collision_frames, color=colors, edgecolor="#4B5563", linewidth=0.4)
    axes[2].set_title("Collision frame count")
    axes[2].set_ylabel("Frame count")
    axes[2].set_xlabel("Candidate subspace")

    axes[3].bar(labels, joint_path, color=colors, edgecolor="#4B5563", linewidth=0.4)
    axes[3].set_title("Joint-space path length")
    axes[3].set_ylabel("Accumulated angle change (deg)")
    axes[3].set_xlabel("Candidate subspace")

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    selected_patch = mpatches.Patch(color="#2A9D8F", label=f"Selected subspace S{selected_sid}")
    other_patch = mpatches.Patch(color="#A8B0B9", label="Other candidates")
    fig.legend(handles=[selected_patch, other_patch], loc="upper center", ncol=2, frameon=True)
    fig.suptitle("Candidate reselection metric comparison", fontsize=12, y=1.02)

    fig.savefig(FIGURES_DIR / "chapter4_candidate_reselection_metrics.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def update_manifest() -> None:
    readme = FIGURE_DIR / "README.md"
    lines = readme.read_text(encoding="utf-8").rstrip().splitlines() if readme.exists() else ["# Figure Outputs", ""]
    additions = [
        "- `figures/chapter4_scene_overview.png`",
        "- `figures/chapter4_clearance_curve.png`",
        "- `figures/chapter4_candidate_reselection_metrics.png`",
    ]
    for item in additions:
        if item not in lines:
            lines.append(item)
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_style()
    payload, selected, collision_case = load_planning_payload()
    plot_scene_overview(payload, selected, collision_case)
    plot_clearance_curve(payload, selected, collision_case)
    plot_candidate_reselection_metrics(payload, selected)
    update_manifest()
    print(f"Saved chapter-4 figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
