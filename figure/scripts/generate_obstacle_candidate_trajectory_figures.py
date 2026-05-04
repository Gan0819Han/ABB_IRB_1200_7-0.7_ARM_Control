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
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fk_model import fk_abb_irb_joint_points
from obstacle_avoidance.collision import LINK_SEGMENTS, ObstacleScene
from obstacle_avoidance.planning import evaluate_trajectory_against_scene

FIGURE_DIR = ROOT / "figure"
FIGURES_DIR = Path(os.environ.get("ABB_FIGURE_OUTPUT_DIR", str(FIGURE_DIR / "figures"))).resolve()
ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_PLAN_JSON = ARTIFACTS_DIR / "obstacle_avoidance" / "open_space_reselect_demo_plan.json"


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


def configure_thesis_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "SimSun", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
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


def build_candidate_cases(payload: dict) -> list[dict]:
    scene = ObstacleScene.from_dict(payload["scene"])
    q_start_deg = payload["q_start_deg"]
    candidates = []
    for item in payload["evaluated_candidates"]:
        summary = evaluate_trajectory_against_scene(
            q_start_deg=q_start_deg,
            q_goal_deg=item["q_goal_deg"],
            scene=scene,
            steps=int(item["trajectory_summary"]["trajectory_steps"]),
            include_frames=True,
        )
        candidates.append(
            {
                "subspace_id": int(item["subspace_id"]),
                "collision": bool(summary["collision"]),
                "first_collision_frame": int(summary["first_collision_frame"]),
                "frames": summary["frames"],
                "min_clearance_mm": float(summary["min_clearance_mm"]),
                "joint_path_length_deg": float(summary["joint_path_length_deg"]),
                "selection_cost": float(item["selection"]["selection_cost"]),
                "colliding_links": list(summary["colliding_links"]),
            }
        )
    return candidates


def initial_robot_points(q_start_deg: list[float]) -> np.ndarray:
    return fk_abb_irb_joint_points(q_start_deg, input_unit="deg")


def tool_path_from_frames(frames: list[dict]) -> np.ndarray:
    return np.asarray([frame["joint_points_mm"][-1] for frame in frames], dtype=float)


def first_collision_point(case: dict) -> np.ndarray | None:
    idx = case["first_collision_frame"]
    if idx < 0 or idx >= len(case["frames"]):
        return None
    return np.asarray(case["frames"][idx]["joint_points_mm"][-1], dtype=float)


def collision_link_midpoints(case: dict) -> list[np.ndarray]:
    idx = case["first_collision_frame"]
    if idx < 0 or idx >= len(case["frames"]):
        return []
    joints = np.asarray(case["frames"][idx]["joint_points_mm"], dtype=float)
    name_to_indices = {name: (i0, i1) for name, i0, i1 in LINK_SEGMENTS}
    midpoints: list[np.ndarray] = []
    for link_name in case.get("colliding_links", []):
        if link_name not in name_to_indices:
            continue
        i0, i1 = name_to_indices[link_name]
        midpoints.append(0.5 * (joints[i0] + joints[i1]))
    return midpoints


def draw_obstacles(ax: plt.Axes, payload: dict, draw_inflated: bool) -> None:
    inflate_mm = float(payload["scene"]["link_radius_mm"]) + float(payload["scene"]["safety_margin_mm"])
    for obstacle in payload["scene"]["obstacles"]:
        if draw_inflated:
            inflated_min = np.asarray(obstacle["min_mm"], dtype=float) - inflate_mm
            inflated_max = np.asarray(obstacle["max_mm"], dtype=float) + inflate_mm
            inflated_faces = box_faces(inflated_min, inflated_max)
            inflated_poly = Poly3DCollection(
                inflated_faces,
                facecolors="#E76F51",
                edgecolors="#A83B2A",
                linewidths=0.7,
                alpha=0.10,
            )
            ax.add_collection3d(inflated_poly)

        faces = box_faces(np.asarray(obstacle["min_mm"], dtype=float), np.asarray(obstacle["max_mm"], dtype=float))
        poly = Poly3DCollection(
            faces,
            facecolors="#D9822B",
            edgecolors="#7B4314",
            linewidths=0.8,
            alpha=0.25,
        )
        ax.add_collection3d(poly)


def draw_initial_robot(ax: plt.Axes, q_start_deg: list[float]) -> np.ndarray:
    joints = initial_robot_points(q_start_deg)
    ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], color="#334155", linewidth=2.0, zorder=5)
    ax.scatter(
        joints[:, 0],
        joints[:, 1],
        joints[:, 2],
        color="#475569",
        edgecolors="#F8FAFC",
        linewidths=0.5,
        s=28,
        zorder=6,
    )
    return joints


def draw_target(ax: plt.Axes, target_xyz: np.ndarray) -> None:
    ax.scatter(
        [target_xyz[0]],
        [target_xyz[1]],
        [target_xyz[2]],
        marker="*",
        s=160,
        color="#E9C46A",
        edgecolors="#7A5C00",
        linewidths=0.8,
        zorder=8,
    )
    ax.text(
        target_xyz[0] - 34.0,
        target_xyz[1] - 26.0,
        target_xyz[2] + 14.0,
        "Target",
        fontsize=8,
        color="#7A5C00",
    )


def set_axes_limits(ax: plt.Axes, payload: dict, cases: list[dict], q_start_deg: list[float]) -> None:
    points = [initial_robot_points(q_start_deg)]
    points.append(np.asarray(payload["target_pose6"][:3], dtype=float).reshape(1, 3))
    for obstacle in payload["scene"]["obstacles"]:
        points.append(np.asarray(obstacle["min_mm"], dtype=float).reshape(1, 3))
        points.append(np.asarray(obstacle["max_mm"], dtype=float).reshape(1, 3))
    for case in cases:
        points.append(tool_path_from_frames(case["frames"]))

    all_points = np.vstack(points)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.58 * np.max(maxs - mins)
    radius = max(radius, 300.0)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.9))
    except Exception:
        pass


def finalize_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=23, azim=-56)
    ax.grid(True, alpha=0.25)


def finalize_axes_thesis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, pad=8)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)", labelpad=10)
    ax.view_init(elev=22, azim=-56)
    ax.grid(True, alpha=0.12, linewidth=0.5)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))


def add_case_label(ax: plt.Axes, path: np.ndarray, subspace_id: int, color: str) -> None:
    end_point = path[-1]
    ax.text(end_point[0], end_point[1], end_point[2] + 10.0, f"S{subspace_id}", fontsize=8, color=color)


def draw_intermediate_robot_snapshots(
    ax: plt.Axes,
    case: dict,
    color: str,
    count: int = 3,
) -> None:
    frames = case["frames"]
    if len(frames) < 4:
        return
    sample_indices = np.linspace(0, len(frames) - 1, count + 2, dtype=int)[1:-1]
    for idx in sample_indices:
        joints = np.asarray(frames[int(idx)]["joint_points_mm"], dtype=float)
        ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], color=color, linewidth=1.0, alpha=0.24)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=8, alpha=0.24)


def draw_collision_snapshot(ax: plt.Axes, case: dict, color: str) -> None:
    idx = case["first_collision_frame"]
    if idx < 0 or idx >= len(case["frames"]):
        return

    joints = np.asarray(case["frames"][idx]["joint_points_mm"], dtype=float)
    ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], color=color, linewidth=1.2, linestyle="--", alpha=0.9)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=10, alpha=0.9)

    colliding = set(case.get("colliding_links", []))
    for link_name, i0, i1 in LINK_SEGMENTS:
        if link_name not in colliding:
            continue
        seg = joints[[i0, i1]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#7F1D1D", linewidth=3.0, alpha=0.95)


def draw_paths(
    ax: plt.Axes,
    cases: list[dict],
    palette: list[str],
    mark_collision_cross: bool,
) -> None:
    for case, color in zip(cases, palette):
        path = tool_path_from_frames(case["frames"])
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=2.0, alpha=0.95)
        if not case["collision"]:
            draw_intermediate_robot_snapshots(ax, case, color, count=3)
        if mark_collision_cross and case["collision"]:
            draw_collision_snapshot(ax, case, color)
            for cross_point in collision_link_midpoints(case):
                ax.scatter(
                    [cross_point[0]],
                    [cross_point[1]],
                    [cross_point[2]],
                    marker="x",
                    s=90,
                    linewidths=2.0,
                    color="#7F1D1D",
                    zorder=10,
                )


def make_legend(ax: plt.Axes, include_collision_cross: bool, thesis_style: bool = False) -> None:
    if thesis_style:
        handles = [
            plt.Line2D([0], [0], color="#334155", lw=2.0, marker="o", markersize=5, label="Initial robot"),
            plt.Line2D([0], [0], color="#E9C46A", lw=0, marker="*", markersize=10, label="Target pose"),
            plt.Line2D([0], [0], color="#D9822B", lw=6, alpha=0.25, label="Obstacle"),
        ]
        if include_collision_cross:
            handles.append(
                plt.Line2D([0], [0], color="#A83B2A", lw=6, alpha=0.15, label="Inflated safety box")
            )
            handles.append(
                plt.Line2D([0], [0], color="#B23A48", lw=1.2, linestyle="--", label="Collision-frame robot")
            )
            handles.append(
                plt.Line2D([0], [0], color="#7F1D1D", lw=3.0, label="Colliding link")
            )
            handles.append(
                plt.Line2D([0], [0], color="#7F1D1D", lw=0, marker="x", markersize=8, label="First collision point")
            )
        else:
            handles.append(
                plt.Line2D([0], [0], color="#2A9D8F", lw=1.0, alpha=0.3, marker="o", markersize=4, label="Intermediate robot pose")
            )
    else:
        handles = [
            plt.Line2D([0], [0], color="#334155", lw=2.0, marker="o", markersize=5, label="Initial robot / 初始机械臂"),
            plt.Line2D([0], [0], color="#E9C46A", lw=0, marker="*", markersize=10, label="Target pose / 目标位姿"),
            plt.Line2D([0], [0], color="#D9822B", lw=6, alpha=0.25, label="Obstacle / 障碍物"),
        ]
        if include_collision_cross:
            handles.append(
                plt.Line2D([0], [0], color="#A83B2A", lw=6, alpha=0.15, label="Inflated safety box / 膨胀安全包络")
            )
            handles.append(
                plt.Line2D([0], [0], color="#B23A48", lw=1.2, linestyle="--", label="Collision-frame robot / 碰撞时刻机械臂")
            )
            handles.append(
                plt.Line2D([0], [0], color="#7F1D1D", lw=3.0, label="Colliding link / 碰撞连杆")
            )
            handles.append(
                plt.Line2D([0], [0], color="#7F1D1D", lw=0, marker="x", markersize=8, label="First collision / 首次碰撞点")
            )
        else:
            handles.append(
                plt.Line2D([0], [0], color="#2A9D8F", lw=1.0, alpha=0.3, marker="o", markersize=4, label="Intermediate robot / 中间机械臂姿态")
            )
    ax.legend(handles=handles, loc="upper left", frameon=True, borderaxespad=0.8)


def create_figure(
    payload: dict,
    cases: list[dict],
    title: str,
    out_name: str,
    palette: list[str],
    mark_collision_cross: bool,
) -> None:
    fig = plt.figure(figsize=(9.6, 7.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    draw_obstacles(ax, payload, draw_inflated=mark_collision_cross)
    draw_initial_robot(ax, payload["q_start_deg"])
    draw_target(ax, np.asarray(payload["target_pose6"][:3], dtype=float))
    draw_paths(ax, cases, palette, mark_collision_cross=mark_collision_cross)
    set_axes_limits(ax, payload, cases, payload["q_start_deg"])
    finalize_axes(ax, title)
    make_legend(ax, include_collision_cross=mark_collision_cross)

    fig.savefig(FIGURES_DIR / out_name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_thesis_figure(
    payload: dict,
    cases: list[dict],
    title: str,
    out_name: str,
    palette: list[str],
    mark_collision_cross: bool,
) -> None:
    fig = plt.figure(figsize=(8.6, 6.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    draw_obstacles(ax, payload, draw_inflated=mark_collision_cross)
    draw_initial_robot(ax, payload["q_start_deg"])
    draw_target(ax, np.asarray(payload["target_pose6"][:3], dtype=float))
    draw_paths(ax, cases, palette, mark_collision_cross=mark_collision_cross)
    set_axes_limits(ax, payload, cases, payload["q_start_deg"])
    finalize_axes_thesis(ax, title)
    make_legend(ax, include_collision_cross=mark_collision_cross, thesis_style=True)

    fig.savefig(FIGURES_DIR / out_name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def update_manifest() -> None:
    readme = FIGURE_DIR / "README.md"
    lines = readme.read_text(encoding="utf-8").rstrip().splitlines() if readme.exists() else ["# Figure Outputs", ""]
    additions = [
        "- `figures/obstacle_candidates_free_only.png`",
        "- `figures/obstacle_candidates_colliding_only.png`",
        "- `figures/obstacle_candidates_overview.png`",
        "- `figures/obstacle_candidates_free_only_thesis.png`",
        "- `figures/obstacle_candidates_colliding_only_thesis.png`",
        "- `figures/obstacle_candidates_overview_thesis.png`",
    ]
    for item in additions:
        if item not in lines:
            lines.append(item)
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate obstacle-avoidance trajectory figures from one planning result JSON.")
    parser.add_argument("--plan_json", default=str(DEFAULT_PLAN_JSON))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dirs()
    configure_style()

    plan_json = Path(args.plan_json)
    payload = load_json(plan_json)
    cases = build_candidate_cases(payload)
    free_cases = [case for case in cases if not case["collision"]]
    colliding_cases = [case for case in cases if case["collision"]]

    create_figure(
        payload=payload,
        cases=free_cases,
        title="Collision-free candidate trajectories / 无碰撞候选轨迹",
        out_name="obstacle_candidates_free_only.png",
        palette=["#1D7A6F", "#2A9D8F", "#63C5B8"],
        mark_collision_cross=False,
    )
    create_figure(
        payload=payload,
        cases=colliding_cases,
        title="Colliding candidate trajectories / 碰撞候选轨迹",
        out_name="obstacle_candidates_colliding_only.png",
        palette=["#B23A48", "#D1495B", "#E76F51", "#F4A261"],
        mark_collision_cross=True,
    )
    create_figure(
        payload=payload,
        cases=cases,
        title="All candidate trajectories / 候选轨迹总览",
        out_name="obstacle_candidates_overview.png",
        palette=["#1D7A6F", "#2A9D8F", "#63C5B8", "#B23A48", "#D1495B", "#E76F51", "#F4A261"],
        mark_collision_cross=True,
    )

    configure_thesis_style()
    create_thesis_figure(
        payload=payload,
        cases=free_cases,
        title="Collision-free candidate trajectories",
        out_name="obstacle_candidates_free_only_thesis.png",
        palette=["#15616D", "#1E847F", "#5BA89A"],
        mark_collision_cross=False,
    )
    create_thesis_figure(
        payload=payload,
        cases=colliding_cases,
        title="Colliding candidate trajectories",
        out_name="obstacle_candidates_colliding_only_thesis.png",
        palette=["#9A031E", "#BB3E03", "#CA6702", "#EE9B00"],
        mark_collision_cross=True,
    )
    create_thesis_figure(
        payload=payload,
        cases=cases,
        title="Overall candidate trajectory comparison",
        out_name="obstacle_candidates_overview_thesis.png",
        palette=["#15616D", "#1E847F", "#5BA89A", "#9A031E", "#BB3E03", "#CA6702", "#EE9B00"],
        mark_collision_cross=True,
    )
    update_manifest()
    print(f"Saved obstacle candidate figures to: {FIGURES_DIR}")
    print(f"Obstacle plan source JSON: {plan_json.resolve()}")


if __name__ == "__main__":
    main()
