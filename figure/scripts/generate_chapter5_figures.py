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
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def draw_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, fc: str) -> None:
    x0, y0 = xy
    patch = mpatches.FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#2F3A4A",
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x0 + width / 2.0,
        y0 + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=9,
        linespacing=1.35,
    )


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], text: str = "") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#4B5563", shrinkA=4, shrinkB=4),
    )
    if text:
        mx = 0.5 * (start[0] + end[0])
        my = 0.5 * (start[1] + end[1])
        ax.text(mx, my + 0.025, text, ha="center", va="bottom", fontsize=8, color="#374151")


def plot_system_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.6), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(
        ax,
        (0.04, 0.58),
        0.19,
        0.22,
        "IRB-1200参数建模\nStandard D-H + FK验证",
        "#DCEBFA",
    )
    draw_box(
        ax,
        (0.30, 0.58),
        0.19,
        0.22,
        "样本生成与\n子空间划分",
        "#E7F6E7",
    )
    draw_box(
        ax,
        (0.56, 0.58),
        0.19,
        0.22,
        "分类模型 + 局部回归\n候选逆解初值生成",
        "#FBE6D4",
    )
    draw_box(
        ax,
        (0.81, 0.58),
        0.15,
        0.22,
        "FK回代筛选\nNR精修",
        "#F7DCE2",
    )

    draw_box(
        ax,
        (0.30, 0.17),
        0.19,
        0.22,
        "固定障碍物场景\nAABB环境建模",
        "#EEE8FB",
    )
    draw_box(
        ax,
        (0.56, 0.17),
        0.19,
        0.22,
        "轨迹插值与\n逐帧碰撞检测",
        "#FFF0C9",
    )
    draw_box(
        ax,
        (0.81, 0.17),
        0.15,
        0.22,
        "候选解重选\nUnity可视化验证",
        "#D8F2EC",
    )

    draw_arrow(ax, (0.23, 0.69), (0.30, 0.69))
    draw_arrow(ax, (0.49, 0.69), (0.56, 0.69))
    draw_arrow(ax, (0.75, 0.69), (0.81, 0.69))
    draw_arrow(ax, (0.655, 0.58), (0.655, 0.39), "候选终点解")
    draw_arrow(ax, (0.49, 0.28), (0.56, 0.28))
    draw_arrow(ax, (0.75, 0.28), (0.81, 0.28))
    draw_arrow(ax, (0.885, 0.58), (0.885, 0.39), "轨迹候选")

    ax.text(
        0.5,
        0.93,
        "面向列车车底检测任务的系统仿真与验证流程",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.07,
        "上层完成逆运动学求解，下层完成受限环境轨迹可执行性验证，并由Unity端进行可视化闭环核对。",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#4B5563",
    )

    fig.savefig(FIGURES_DIR / "system_simulation_pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


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


def extract_tool_path(frames: list[dict]) -> np.ndarray:
    points = []
    for frame in frames:
        joint_points = np.asarray(frame["joint_points_mm"], dtype=float)
        points.append(joint_points[-1])
    return np.asarray(points, dtype=float)


def plot_obstacle_avoidance_comparison() -> None:
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

    selected_path = extract_tool_path(selected_summary["frames"])
    collision_path = extract_tool_path(collision_summary["frames"])
    obstacles = payload["scene"]["obstacles"]
    target = np.asarray(payload["target_pose6"][:3], dtype=float)

    fig = plt.figure(figsize=(9.6, 4.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    for obstacle in obstacles:
        faces = box_faces(np.asarray(obstacle["min_mm"], dtype=float), np.asarray(obstacle["max_mm"], dtype=float))
        pc = Poly3DCollection(
            faces,
            facecolors="#F4A261",
            edgecolors="#8C4F1F",
            linewidths=0.8,
            alpha=0.24,
        )
        ax.add_collection3d(pc)
        center = np.asarray(obstacle["center_mm"], dtype=float)
        ax.text(center[0], center[1], center[2], obstacle["name"], fontsize=8, color="#7C3E12")

    ax.plot(
        collision_path[:, 0],
        collision_path[:, 1],
        collision_path[:, 2],
        color="#D1495B",
        linewidth=2.2,
        label="碰撞候选轨迹",
    )
    ax.plot(
        selected_path[:, 0],
        selected_path[:, 1],
        selected_path[:, 2],
        color="#2A9D8F",
        linewidth=2.4,
        label="最终无碰撞轨迹",
    )
    ax.scatter(
        [selected_path[0, 0]],
        [selected_path[0, 1]],
        [selected_path[0, 2]],
        color="#264653",
        s=36,
        label="起始点",
    )
    ax.scatter(
        [target[0]],
        [target[1]],
        [target[2]],
        color="#E9C46A",
        edgecolors="#7A5C00",
        s=48,
        label="目标点",
    )

    ax.set_title("固定障碍物场景下候选轨迹对比")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=22, azim=-55)
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
        f"最终选中子空间: {selected['subspace_id']}，最小安全间隙: {selected_summary['min_clearance_mm']:.2f} mm\n"
        f"对比碰撞子空间: {collision_candidate['subspace_id']}，碰撞帧数: {collision_summary['collision_frame_count']}"
    )
    fig.text(0.5, 0.03, note, ha="center", va="bottom", fontsize=8.5, color="#4B5563")

    fig.savefig(
        FIGURES_DIR / "obstacle_avoidance_trajectory_comparison.png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def update_manifest() -> None:
    readme = FIGURE_DIR / "README.md"
    lines = readme.read_text(encoding="utf-8").rstrip().splitlines() if readme.exists() else ["# Figure Outputs", ""]
    additions = [
        "- `figures/system_simulation_pipeline.png`",
        "- `figures/obstacle_avoidance_trajectory_comparison.png`",
    ]
    for item in additions:
        if item not in lines:
            lines.append(item)
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_style()
    plot_system_pipeline()
    plot_obstacle_avoidance_comparison()
    update_manifest()
    print(f"Saved chapter-5 figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
