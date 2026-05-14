#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import subprocess
import sys
from io import BytesIO
from html import escape
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fk_model import fk_abb_irb_joint_points
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PYTHON = sys.executable
UNITY_DIR = Path(r"E:\Software\Unity\Project\ABB_IRB_Demo1")
GUI_DEFAULTS_PATH = ROOT / "gui" / "gui_defaults.json"
DEFAULT_OBSTACLE_NAME = "demo_box_1"
DEFAULT_OBSTACLE_CENTER_MM = [221.7, 274.53, 493.57]
DEFAULT_OBSTACLE_SIZE_MM = [127.62, 90.45, 175.06]
NEW_OBSTACLE_CENTER_MM = [360.0, -180.0, 540.0]
NEW_OBSTACLE_SIZE_MM = [110.0, 90.0, 160.0]
PREVIEW_BG = "#F8FAFC"
PREVIEW_AXIS = "#CBD5E1"
PREVIEW_OBSTACLE_FILL = "#E2E8F0"
PREVIEW_OBSTACLE_OUTLINE = "#64748B"
PREVIEW_SELECTED_FILL = "#FDE68A"
PREVIEW_SELECTED_OUTLINE = "#B45309"
PREVIEW_ROBOT = "#2563EB"
PREVIEW_TARGET = "#DC2626"


@dataclass
class CommandResult:
    title: str
    cmd: list[str]
    exit_code: int
    log_text: str
    summary_text: str
    output_path: Path | None = None


def load_path_defaults() -> dict[str, str]:
    if not GUI_DEFAULTS_PATH.exists():
        return {}
    try:
        payload = json.loads(GUI_DEFAULTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    path_defaults = payload.get("path_defaults", {})
    if not isinstance(path_defaults, dict):
        return {}
    return {str(key): str(value) for key, value in path_defaults.items()}


def default_path_value(path_defaults: dict[str, str], key: str, fallback: Path) -> str:
    return path_defaults.get(key, str(fallback))


def ensure_parent(path_text: str) -> None:
    path = Path(path_text)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def build_figure_env(figure_output_dir: str, figure_data_dir: str) -> dict[str, str]:
    figures_dir = Path(figure_output_dir)
    data_dir = Path(figure_data_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["ABB_FIGURE_OUTPUT_DIR"] = str(figures_dir)
    env["ABB_FIGURE_DATA_DIR"] = str(data_dir)
    return env


def load_scene_payload(scene_json: str) -> tuple[Path, dict]:
    scene_path = Path(scene_json)
    if not scene_path.exists():
        raise FileNotFoundError(f"未找到 scene_json：{scene_path}")
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    return scene_path, payload


def write_scene_payload(scene_path: Path, payload: dict) -> None:
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_obstacle_selector_label(index: int, name: str) -> str:
    return f"{index + 1}. {name}"


def obstacle_payload_to_center_size(obstacle: dict) -> tuple[list[float], list[float]]:
    if "center_mm" in obstacle and "size_mm" in obstacle:
        center = [float(v) for v in obstacle["center_mm"]]
        size = [float(v) for v in obstacle["size_mm"]]
        return center, size
    if "min_mm" in obstacle and "max_mm" in obstacle:
        box_min = [float(v) for v in obstacle["min_mm"]]
        box_max = [float(v) for v in obstacle["max_mm"]]
        center = [(a + b) * 0.5 for a, b in zip(box_min, box_max)]
        size = [b - a for a, b in zip(box_min, box_max)]
        return center, size
    raise ValueError("obstacle 缺少 center_mm/size_mm 或 min_mm/max_mm")


def parse_vec3_text(text: str, *, label: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError(f"{label} 必须为 3 个逗号分隔数值")
    return values


def parse_float_value(text: str, *, label: str) -> float:
    try:
        return float(text.strip())
    except Exception as exc:
        raise ValueError(f"{label} 必须是数值") from exc


def format_q_deg(values: list[float]) -> str:
    return ",".join(f"{float(v):.6f}".rstrip("0").rstrip(".") for v in values)


def build_obstacle_payload(name: str, center_text: str, size_text: str, *, fallback_name: str) -> dict:
    center = parse_vec3_text(center_text, label="center_mm")
    size = parse_vec3_text(size_text, label="size_mm")
    if any(v <= 0.0 for v in size):
        raise ValueError("size_mm 的三个分量都必须大于 0")
    return {
        "name": name.strip() or fallback_name,
        "center_mm": center,
        "size_mm": size,
    }


def get_obstacles(scene_json: str) -> list[dict]:
    _, payload = load_scene_payload(scene_json)
    obstacles = payload.get("obstacles", [])
    return obstacles if isinstance(obstacles, list) else []


def load_selected_obstacle(scene_json: str, index: int) -> tuple[list[dict], int, dict]:
    obstacles = get_obstacles(scene_json)
    if not obstacles:
        raise ValueError("scene_json 中没有 obstacles")
    index = max(0, min(index, len(obstacles) - 1))
    obstacle = obstacles[index]
    center, size = obstacle_payload_to_center_size(obstacle)
    return obstacles, index, {
        "name": str(obstacle.get("name", f"obstacle_{index + 1}")),
        "center_mm": format_q_deg(center),
        "size_mm": format_q_deg(size),
    }


def save_selected_obstacle(scene_json: str, index: int, name: str, center_text: str, size_text: str) -> str:
    scene_path, payload = load_scene_payload(scene_json)
    obstacles = payload.setdefault("obstacles", [])
    obstacle_payload = build_obstacle_payload(name, center_text, size_text, fallback_name=f"obstacle_{index + 1}")
    if obstacles:
        index = max(0, min(index, len(obstacles) - 1))
        obstacles[index] = obstacle_payload
    else:
        obstacles.append(obstacle_payload)
        index = 0
    write_scene_payload(scene_path, payload)
    return (
        f"已写回 scene_json 当前障碍物：{build_obstacle_selector_label(index, obstacle_payload['name'])}\n"
        f"center_mm：{obstacle_payload['center_mm']}\n"
        f"size_mm：{obstacle_payload['size_mm']}\n"
        f"scene_json：{scene_path}"
    )


def reset_default_obstacle() -> dict[str, str]:
    return {
        "name": DEFAULT_OBSTACLE_NAME,
        "center_mm": format_q_deg(DEFAULT_OBSTACLE_CENTER_MM),
        "size_mm": format_q_deg(DEFAULT_OBSTACLE_SIZE_MM),
    }


def make_new_obstacle_name(obstacles: list[dict]) -> str:
    used = {str(obstacle.get("name", "")).strip() for obstacle in obstacles}
    idx = 1
    while True:
        candidate = f"demo_box_{idx}"
        if candidate not in used:
            return candidate
        idx += 1


def add_obstacle(scene_json: str) -> tuple[str, int, dict]:
    scene_path, payload = load_scene_payload(scene_json)
    obstacles = payload.setdefault("obstacles", [])
    new_name = make_new_obstacle_name(obstacles)
    obstacle_payload = {
        "name": new_name,
        "center_mm": list(NEW_OBSTACLE_CENTER_MM),
        "size_mm": list(NEW_OBSTACLE_SIZE_MM),
    }
    obstacles.append(obstacle_payload)
    new_index = len(obstacles) - 1
    write_scene_payload(scene_path, payload)
    return (
        f"已新增障碍物：{build_obstacle_selector_label(new_index, new_name)}\n"
        f"center_mm：{obstacle_payload['center_mm']}\n"
        f"size_mm：{obstacle_payload['size_mm']}\n"
        f"scene_json：{scene_path}",
        new_index,
        {
            "name": new_name,
            "center_mm": format_q_deg(NEW_OBSTACLE_CENTER_MM),
            "size_mm": format_q_deg(NEW_OBSTACLE_SIZE_MM),
        },
    )


def delete_obstacle(scene_json: str, index: int) -> tuple[str, int, dict]:
    scene_path, payload = load_scene_payload(scene_json)
    obstacles = payload.get("obstacles", [])
    if not obstacles:
        raise ValueError("scene_json 中没有可删除的障碍物")
    if len(obstacles) == 1:
        raise ValueError("当前至少需要保留 1 个障碍物；如需空场景，请手动编辑 scene_json。")
    index = max(0, min(index, len(obstacles) - 1))
    removed = obstacles.pop(index)
    next_index = min(index, len(obstacles) - 1)
    payload["obstacles"] = obstacles
    write_scene_payload(scene_path, payload)
    _, final_index, final_values = load_selected_obstacle(scene_json, next_index)
    return (
        f"已删除障碍物：{removed.get('name', f'obstacle_{index + 1}')}\n"
        f"已加载障碍物：{build_obstacle_selector_label(final_index, final_values['name'])}\n"
        f"scene_json：{scene_path}",
        final_index,
        final_values,
    )


def nudge_obstacle_value(values_text: str, *, step_text: str, index: int, direction: int, is_size: bool) -> str:
    label = "尺寸步长(mm)" if is_size else "位置步长(mm)"
    value_label = "size_mm" if is_size else "center_mm"
    step = parse_float_value(step_text, label=label)
    if step <= 0.0:
        raise ValueError(f"{label} 必须大于 0")
    values = parse_vec3_text(values_text, label=value_label)
    values[index] += direction * step
    if is_size:
        values[index] = max(1.0, values[index])
    return format_q_deg(values)


def build_preview_scene_payload(
    scene_json: str,
    *,
    selected_index: int,
    obstacle_name: str,
    center_text: str,
    size_text: str,
) -> tuple[dict, int]:
    _, payload = load_scene_payload(scene_json)
    obstacles = payload.get("obstacles", [])
    if not isinstance(obstacles, list):
        obstacles = []
    selected_index = max(0, selected_index)
    obstacle_payload = build_obstacle_payload(
        obstacle_name,
        center_text,
        size_text,
        fallback_name=f"obstacle_{selected_index + 1}",
    )
    cloned = dict(payload)
    cloned_obstacles = [dict(item) if isinstance(item, dict) else item for item in obstacles]
    if cloned_obstacles:
        selected_index = min(selected_index, len(cloned_obstacles) - 1)
        cloned_obstacles[selected_index] = obstacle_payload
    else:
        cloned_obstacles = [obstacle_payload]
        selected_index = 0
    cloned["obstacles"] = cloned_obstacles
    return cloned, selected_index


def preview_target_xyz_mm(pose6_text: str) -> list[float]:
    values = [float(item.strip()) for item in pose6_text.split(",") if item.strip()]
    if len(values) != 6:
        raise ValueError("pose6 必须为 6 个逗号分隔数值")
    return [values[0], values[1], values[2]]


def preview_q_start_deg(q_start_text: str) -> list[float]:
    values = [float(item.strip()) for item in q_start_text.split(",") if item.strip()]
    if len(values) != 6:
        raise ValueError("q_start 必须为 6 个逗号分隔数值")
    return values


def preview_axes_bounds(
    joint_points: list[list[float]],
    target_xyz: list[float],
    obstacles: list[dict],
) -> dict[str, tuple[float, float]]:
    points = list(joint_points) + [target_xyz]
    for obstacle in obstacles:
        center, size = obstacle_payload_to_center_size(obstacle)
        box_min = [c - 0.5 * s for c, s in zip(center, size)]
        box_max = [c + 0.5 * s for c, s in zip(center, size)]
        points.append(box_min)
        points.append(box_max)
    mins = [min(point[i] for point in points) for i in range(3)]
    maxs = [max(point[i] for point in points) for i in range(3)]
    center = [(a + b) * 0.5 for a, b in zip(mins, maxs)]
    radius = max(max(maxs[i] - mins[i] for i in range(3)) * 0.58, 260.0)
    return {
        "x": (center[0] - radius, center[0] + radius),
        "y": (center[1] - radius, center[1] + radius),
        "z": (max(0.0, center[2] - radius), center[2] + radius),
    }


def _project_point(
    value_a: float,
    value_b: float,
    bounds: dict[str, tuple[float, float]],
    axis_a: str,
    axis_b: str,
    *,
    width: int,
    height: int,
    margin: int,
) -> tuple[float, float]:
    min_a, max_a = bounds[axis_a]
    min_b, max_b = bounds[axis_b]
    span_a = max(max_a - min_a, 1.0)
    span_b = max(max_b - min_b, 1.0)
    x = margin + (value_a - min_a) / span_a * (width - 2 * margin)
    y = height - margin - (value_b - min_b) / span_b * (height - 2 * margin)
    return x, y


def build_obstacle_preview_svg(
    payload: dict,
    *,
    selected_index: int,
    pose6_text: str,
    q_start_text: str,
    width: int = 900,
    height: int = 260,
) -> str:
    obstacles = payload.get("obstacles", [])
    if not isinstance(obstacles, list):
        obstacles = []
    target_xyz = preview_target_xyz_mm(pose6_text)
    q_start = preview_q_start_deg(q_start_text)
    joint_points = fk_abb_irb_joint_points(q_start, input_unit="deg").tolist()
    bounds = preview_axes_bounds(joint_points, target_xyz, obstacles)
    margin = 26
    gap = 20
    panel_width = max(220, (width - gap * 2) // 3)
    titles = [("XY 俯视", "x", "y"), ("XZ 正视", "x", "z"), ("YZ 侧视", "y", "z")]
    pieces: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    for view_index, (title, axis_a, axis_b) in enumerate(titles):
        origin_x = view_index * (panel_width + gap)
        pieces.append(
            f'<g transform="translate({origin_x},0)">'
            f'<rect x="0.5" y="0.5" width="{panel_width - 1}" height="{height - 1}" rx="12" fill="{PREVIEW_BG}" stroke="{PREVIEW_AXIS}"/>'
            f'<text x="{margin}" y="24" font-size="14" font-family="Microsoft YaHei UI, Segoe UI, sans-serif" fill="#334155">{escape(title)}</text>'
            f'<line x1="{margin}" y1="{height - margin}" x2="{panel_width - margin}" y2="{height - margin}" stroke="{PREVIEW_AXIS}"/>'
            f'<line x1="{margin}" y1="{margin + 12}" x2="{margin}" y2="{height - margin}" stroke="{PREVIEW_AXIS}"/>'
        )
        for idx, obstacle in enumerate(obstacles):
            center, size = obstacle_payload_to_center_size(obstacle)
            half_size = [value * 0.5 for value in size]
            axis_map = {"x": 0, "y": 1, "z": 2}
            a_idx = axis_map[axis_a]
            b_idx = axis_map[axis_b]
            a0 = center[a_idx] - half_size[a_idx]
            a1 = center[a_idx] + half_size[a_idx]
            b0 = center[b_idx] - half_size[b_idx]
            b1 = center[b_idx] + half_size[b_idx]
            x0, y1 = _project_point(a0, b0, bounds, axis_a, axis_b, width=panel_width, height=height, margin=margin)
            x1, y0 = _project_point(a1, b1, bounds, axis_a, axis_b, width=panel_width, height=height, margin=margin)
            fill = PREVIEW_SELECTED_FILL if idx == selected_index else PREVIEW_OBSTACLE_FILL
            outline = PREVIEW_SELECTED_OUTLINE if idx == selected_index else PREVIEW_OBSTACLE_OUTLINE
            pieces.append(
                f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{max(4.0, x1 - x0):.2f}" height="{max(4.0, y1 - y0):.2f}" '
                f'fill="{fill}" fill-opacity="0.88" stroke="{outline}" stroke-width="2"/>'
            )
        path_points: list[str] = []
        for joint in joint_points:
            px, py = _project_point(
                joint[{"x": 0, "y": 1, "z": 2}[axis_a]],
                joint[{"x": 0, "y": 1, "z": 2}[axis_b]],
                bounds,
                axis_a,
                axis_b,
                width=panel_width,
                height=height,
                margin=margin,
            )
            path_points.append(f"{px:.2f},{py:.2f}")
        pieces.append(
            f'<polyline points="{" ".join(path_points)}" fill="none" stroke="{PREVIEW_ROBOT}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for point_text in path_points:
            px_text, py_text = point_text.split(",")
            pieces.append(
                f'<circle cx="{px_text}" cy="{py_text}" r="2.8" fill="{PREVIEW_ROBOT}"/>'
            )
        tx, ty = _project_point(
            target_xyz[{"x": 0, "y": 1, "z": 2}[axis_a]],
            target_xyz[{"x": 0, "y": 1, "z": 2}[axis_b]],
            bounds,
            axis_a,
            axis_b,
            width=panel_width,
            height=height,
            margin=margin,
        )
        pieces.append(
            f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="5" fill="{PREVIEW_TARGET}"/>'
            f'<text x="{tx + 9:.2f}" y="{ty - 8:.2f}" font-size="13" font-family="Microsoft YaHei UI, Segoe UI, sans-serif" fill="{PREVIEW_TARGET}">T</text>'
            f"</g>"
        )
    pieces.append("</svg>")
    return "".join(pieces)


def _box_faces(box_min: list[float], box_max: list[float]) -> list[list[list[float]]]:
    x0, y0, z0 = [float(v) for v in box_min]
    x1, y1, z1 = [float(v) for v in box_max]
    vertices = [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]
    return [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [0, 3, 7, 4]],
    ]


def _draw_3d_obstacle_box(ax, obstacle: dict, *, selected: bool, inflate_mm: float = 0.0) -> None:
    center, size = obstacle_payload_to_center_size(obstacle)
    box_min = [c - 0.5 * s - inflate_mm for c, s in zip(center, size)]
    box_max = [c + 0.5 * s + inflate_mm for c, s in zip(center, size)]
    faces = _box_faces(box_min, box_max)
    if inflate_mm > 0.0:
        poly = Poly3DCollection(
            faces,
            facecolors="#A83B2A",
            edgecolors="#A83B2A",
            linewidths=0.8,
            alpha=0.10,
        )
    else:
        poly = Poly3DCollection(
            faces,
            facecolors="#D9822B" if not selected else "#F59E0B",
            edgecolors="#7B4314" if not selected else "#B45309",
            linewidths=1.0 if not selected else 1.6,
            alpha=0.28,
        )
    ax.add_collection3d(poly)


def _set_obstacle_3d_axes_limits(ax, joint_points: list[list[float]], target_xyz: list[float], obstacles: list[dict]) -> None:
    points = list(joint_points) + [target_xyz]
    for obstacle in obstacles:
        center, size = obstacle_payload_to_center_size(obstacle)
        box_min = [c - 0.5 * s for c, s in zip(center, size)]
        box_max = [c + 0.5 * s for c, s in zip(center, size)]
        points.append(box_min)
        points.append(box_max)
    mins = [min(point[i] for point in points) for i in range(3)]
    maxs = [max(point[i] for point in points) for i in range(3)]
    center = [(a + b) * 0.5 for a, b in zip(mins, maxs)]
    radius = max(max(maxs[i] - mins[i] for i in range(3)) * 0.62, 320.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.9))
    except Exception:
        pass


def build_obstacle_preview_3d_png(
    payload: dict,
    *,
    selected_index: int,
    pose6_text: str,
    q_start_text: str,
    elev: float = 22.0,
    azim: float = -56.0,
) -> bytes:
    obstacles = payload.get("obstacles", [])
    if not isinstance(obstacles, list):
        obstacles = []
    target_xyz = preview_target_xyz_mm(pose6_text)
    q_start = preview_q_start_deg(q_start_text)
    joint_points = fk_abb_irb_joint_points(q_start, input_unit="deg").tolist()
    inflate_mm = float(payload.get("link_radius_mm", 35.0)) + float(payload.get("safety_margin_mm", 5.0))

    figure = Figure(figsize=(6.8, 5.4), dpi=120)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111, projection="3d")
    ax.set_title("Obstacle scene preview", pad=12)

    for idx, obstacle in enumerate(obstacles):
        _draw_3d_obstacle_box(ax, obstacle, selected=False, inflate_mm=inflate_mm)
        _draw_3d_obstacle_box(ax, obstacle, selected=(idx == selected_index), inflate_mm=0.0)

    xs = [point[0] for point in joint_points]
    ys = [point[1] for point in joint_points]
    zs = [point[2] for point in joint_points]
    ax.plot(xs, ys, zs, color="#334155", linewidth=2.8, marker="o", markersize=4.8)
    ax.scatter(xs, ys, zs, color="#0F766E", s=22)

    ax.scatter(
        [target_xyz[0]],
        [target_xyz[1]],
        [target_xyz[2]],
        marker="*",
        s=240,
        color="#E9C46A",
        edgecolors="#7A5C00",
        linewidths=0.9,
    )
    ax.text(target_xyz[0], target_xyz[1], target_xyz[2] + 14.0, "Target", fontsize=9, color="#7A5C00")

    _set_obstacle_3d_axes_limits(ax, joint_points, target_xyz, obstacles)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.22)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    figure.tight_layout()

    output = BytesIO()
    figure.savefig(output, format="png", bbox_inches="tight")
    return output.getvalue()


def extract_goal_q_from_json(json_path: Path) -> tuple[list[float] | None, str]:
    if not json_path.exists():
        return None, f"未找到输出文件：{json_path}"

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    refined = payload.get("refined_solution", {})
    q_deg = refined.get("q_deg")
    if q_deg:
        return [float(v) for v in q_deg], "来源：refined_solution.q_deg"

    selected = payload.get("selected_solution", {})
    q_deg = selected.get("q_goal_deg")
    if q_deg:
        return [float(v) for v in q_deg], "来源：selected_solution.q_goal_deg"

    initial = payload.get("initial_solution", {})
    q_deg = initial.get("q0_deg")
    if q_deg:
        return [float(v) for v in q_deg], "来源：initial_solution.q0_deg"

    return None, "未在结果 JSON 中找到可用于轨迹终点的关节角"


def build_sync_q_targets_text(json_path: Path) -> str:
    q_deg, source = extract_goal_q_from_json(json_path)
    if not q_deg:
        return f"未同步 Unity 关节目标：{source}"
    text = format_q_deg(q_deg)
    return (
        f"已同步到 Unity 页 q_goal：{text}\n"
        f"已同步到 FK参考导出 q：{text}\n"
        f"{source}"
    )


def build_predict_summary(json_path: Path) -> str:
    if not json_path.exists():
        return f"未找到输出文件：{json_path}"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    lines = ["任务：predict_ik", ""]
    lines.append(f"候选来源：{payload.get('candidate_source', '-')}")
    initial = payload.get("initial_solution", {})
    refined = payload.get("refined_solution", {})
    if initial:
        lines.append(f"初始子空间：{initial.get('subspace_id', '-')}")
        lines.append(f"初始位置误差：{initial.get('position_l2_mm', '-')}")
    if refined:
        lines.append(f"NR 收敛：{refined.get('nr_converged', '-')}")
        lines.append(f"NR 迭代次数：{refined.get('nr_iters', '-')}")
        lines.append(f"最终位置误差(mm)：{refined.get('final_pos_err_mm', '-')}")
        lines.append(f"最终姿态误差(rad)：{refined.get('final_ori_err_rad', '-')}")
    if "ik_solve_time_ms" in payload:
        lines.append(f"总耗时(ms)：{payload['ik_solve_time_ms']}")
    lines.append("")
    lines.append(build_sync_q_targets_text(json_path))
    lines.append("")
    lines.append(f"结果文件：{json_path}")
    return "\n".join(lines)


def build_obstacle_summary(json_path: Path) -> str:
    if not json_path.exists():
        return f"未找到输出文件：{json_path}"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_solution", {})
    traj = selected.get("trajectory_summary", {})
    has_collision_free = bool(payload.get("has_collision_free_solution", False))
    selected_collision_free = bool(payload.get("selected_solution_collision_free", not bool(traj.get("collision", False))))
    lines = ["任务：plan_collision_free_ik", ""]
    lines.append(f"候选数量：{len(payload.get('evaluated_candidates', []))}")
    lines.append(f"是否找到无碰撞候选：{has_collision_free}")
    lines.append(f"选中子空间：{selected.get('subspace_id', '-')}")
    lines.append(f"选中轨迹模式：{selected.get('trajectory_mode', '-')}")
    lines.append(f"NR 收敛：{selected.get('nr_converged', '-')}")
    lines.append(f"NR 迭代次数：{selected.get('nr_iters', '-')}")
    lines.append(f"最终位置误差(mm)：{selected.get('final_pos_err_mm', '-')}")
    lines.append(f"最终姿态误差(rad)：{selected.get('final_ori_err_rad', '-')}")
    lines.append(f"选中轨迹是否无碰撞：{selected_collision_free}")
    lines.append(f"是否碰撞：{traj.get('collision', '-')}")
    lines.append(f"碰撞帧数：{traj.get('collision_frame_count', '-')}")
    lines.append(f"最小净空(mm)：{traj.get('min_clearance_mm', '-')}")
    lines.append(f"轨迹步数：{traj.get('trajectory_steps', '-')}")
    if selected.get("trajectory_waypoint_deg") is not None:
        lines.append(f"waypoint q_mid(deg)：{selected.get('trajectory_waypoint_deg')}")
    lines.append("")
    lines.append(build_sync_q_targets_text(json_path))
    lines.append("")
    lines.append(f"结果文件：{json_path}")
    return "\n".join(lines)


def build_fk_summary(json_path: Path) -> str:
    if not json_path.exists():
        return f"未找到输出文件：{json_path}"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    lines = ["任务：export_unity_fk_reference", ""]
    lines.append(f"关节角 q(deg)：{payload.get('q_deg', [])}")
    lines.append(f"Python 末端位置(mm)：{payload.get('python_position_mm', [])}")
    lines.append(f"Unity 期望位置(m)：{payload.get('unity_expected_world_position_m', [])}")
    lines.append("")
    lines.append(f"结果文件：{json_path}")
    return "\n".join(lines)


def build_traj_summary(json_path: Path) -> str:
    if not json_path.exists():
        return f"未找到输出文件：{json_path}"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    lines = ["任务：export_unity_trajectory", ""]
    lines.append(f"轨迹名称：{payload.get('trajectory_name', '-')}")
    lines.append(f"轨迹步数：{payload.get('trajectory_steps', '-')}")
    lines.append(f"播放时长(s)：{payload.get('playback_duration_seconds', '-')}")
    lines.append(f"起点 q_start：{payload.get('q_start_deg', [])}")
    lines.append(f"终点 q_goal：{payload.get('q_goal_deg', [])}")
    lines.append("")
    lines.append(f"结果文件：{json_path}")
    return "\n".join(lines)


def build_obstacle_unity_summary(json_path: Path) -> str:
    if not json_path.exists():
        return f"未找到输出文件：{json_path}"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_solution", {})
    compare = payload.get("comparison_collision_solution")
    lines = ["任务：export_unity_obstacle_avoidance_demo", ""]
    lines.append(f"demo_name：{payload.get('demo_name', '-')}")
    lines.append(f"scene_name：{payload.get('scene_name', '-')}")
    lines.append(f"障碍物数量：{len(payload.get('obstacles', []))}")
    lines.append(f"是否存在无碰撞轨迹：{payload.get('has_collision_free_solution', False)}")
    lines.append(f"选中轨迹是否无碰撞：{payload.get('selected_solution_collision_free', False)}")
    lines.append(f"选中轨迹子空间：{selected.get('subspace_id', '-')}")
    lines.append(f"选中轨迹模式：{selected.get('trajectory_mode', '-')}")
    lines.append(f"选中轨迹步数：{selected.get('trajectory_steps', '-')}")
    if compare:
        lines.append(f"对比碰撞轨迹子空间：{compare.get('subspace_id', '-')}")
        lines.append(f"对比碰撞轨迹模式：{compare.get('trajectory_mode', '-')}")
        lines.append(f"对比碰撞帧数：{compare.get('collision_frame_count', '-')}")
    lines.append("")
    lines.append(f"结果文件：{json_path}")
    return "\n".join(lines)


def build_obstacle_figure_summary(plan_json: str, output_dir: Path) -> str:
    preview_path = find_latest_obstacle_preview_path(str(output_dir))
    lines = ["任务：generate_obstacle_figures", ""]
    lines.append(f"规划结果 JSON：{plan_json}")
    lines.append(f"图像输出目录：{output_dir}")
    if preview_path:
        lines.append(f"预览图：{preview_path}")
    else:
        lines.append("预览图：未找到已生成 PNG")
    return "\n".join(lines)


def get_obstacle_preview_candidates(figure_output_dir: str) -> list[Path]:
    figures_dir = Path(figure_output_dir)
    return [
        figures_dir / "obstacle_candidates_overview_thesis.png",
        figures_dir / "obstacle_candidates_overview.png",
        figures_dir / "obstacle_candidates_free_only_thesis.png",
        figures_dir / "obstacle_candidates_colliding_only_thesis.png",
    ]


def find_latest_obstacle_preview_path(figure_output_dir: str) -> Path | None:
    for path in get_obstacle_preview_candidates(figure_output_dir):
        if path.exists():
            return path
    return None


def run_command_sync(
    title: str,
    cmd: list[str],
    *,
    summary_builder: Callable[[Path], str] | None = None,
    summary_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    lines = [f"[{title}]", "CMD: " + " ".join(cmd)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip())
    exit_code = proc.wait()
    lines.append(f"[exit code] {exit_code}")
    if exit_code == 0 and summary_builder is not None and summary_path is not None:
        summary_text = summary_builder(summary_path)
    elif exit_code == 0:
        summary_text = f"任务：{title}\n\n执行成功。"
    else:
        summary_text = f"任务：{title}\n\n执行失败，退出码：{exit_code}\n请查看原始日志。"
    return CommandResult(
        title=title,
        cmd=cmd,
        exit_code=exit_code,
        log_text="\n".join(lines),
        summary_text=summary_text,
        output_path=summary_path,
    )


def build_predict_command(form: dict[str, str], *, enable_nr: bool) -> tuple[list[str], Path]:
    out_path = Path(form["predict_out_json"])
    ensure_parent(str(out_path))
    cmd = [
        PYTHON, "-X", "utf8", "predict_ik.py",
        "--candidate_mode", "hierarchical",
        f"--pose={form['pose6']}",
        "--pred_meta", form["pred_meta"],
        "--branch_meta", form["branch_meta"],
        "--fine_meta", form["fine_meta"],
        "--topk_shoulder", form["topk_shoulder"],
        "--topk_elbow", form["topk_elbow"],
        "--topk_wrist", form["topk_wrist"],
        "--max_branch_candidates", form["max_branch_candidates"],
        "--fine_topk_per_branch", form["fine_topk_per_branch"],
        "--max_subspace_candidates", form["max_subspace_candidates"],
        "--nr_max_iters", form["nr_max_iters"],
        "--nr_tol_pos_mm", form["nr_tol_pos_mm"],
        "--nr_tol_ori_rad", form["nr_tol_ori_rad"],
        "--nr_damping", form["nr_damping"],
        "--nr_step_scale", form["nr_step_scale"],
        "--out_json", str(out_path),
    ]
    if enable_nr:
        cmd.append("--enable_nr")
    return cmd, out_path


def build_obstacle_command(form: dict[str, str]) -> tuple[list[str], Path]:
    out_path = Path(form["obstacle_out_json"])
    ensure_parent(str(out_path))
    cmd = [
        PYTHON, "-X", "utf8", "scripts\\plan_collision_free_ik.py",
        f"--pose={form['pose6']}",
        f"--q_start={form['q_start']}",
        "--scene_json", form["scene_json"],
        "--pred_meta", form["pred_meta"],
        "--branch_meta", form["branch_meta"],
        "--fine_meta", form["fine_meta"],
        "--topk_shoulder", form["topk_shoulder"],
        "--topk_elbow", form["topk_elbow"],
        "--topk_wrist", form["topk_wrist"],
        "--max_branch_candidates", form["max_branch_candidates"],
        "--fine_topk_per_branch", form["fine_topk_per_branch"],
        "--max_subspace_candidates", form["max_subspace_candidates"],
        "--max_evaluated_candidates", form["max_evaluated_candidates"],
        "--nr_max_iters", form["nr_max_iters"],
        "--nr_tol_pos_mm", form["nr_tol_pos_mm"],
        "--nr_tol_ori_rad", form["nr_tol_ori_rad"],
        "--nr_damping", form["nr_damping"],
        "--nr_step_scale", form["nr_step_scale"],
        "--trajectory_steps", form["trajectory_steps"],
        "--dedupe_tol_deg", form["dedupe_tol_deg"],
        "--save_selected_frames",
        "--out_json", str(out_path),
    ]
    return cmd, out_path


def build_fk_export_command(form: dict[str, str]) -> tuple[list[str], Path]:
    out_path = Path(form["fk_out_json"])
    ensure_parent(str(out_path))
    cmd = [
        PYTHON, "-X", "utf8", "scripts\\export_unity_fk_reference.py",
        f"--q={form['fk_q']}",
        "--out_json", str(out_path),
    ]
    return cmd, out_path


def build_trajectory_export_command(form: dict[str, str]) -> tuple[list[str], Path]:
    out_path = Path(form["traj_out_json"])
    ensure_parent(str(out_path))
    cmd = [
        PYTHON, "-X", "utf8", "scripts\\export_unity_trajectory.py",
        f"--q_start={form['traj_q_start']}",
        f"--q_goal={form['traj_q_goal']}",
        "--steps", form["traj_steps"],
        "--duration", form["traj_duration"],
        "--name", form["traj_name"],
        "--out_json", str(out_path),
    ]
    return cmd, out_path


def build_obstacle_unity_export_command(form: dict[str, str]) -> tuple[list[str], Path]:
    out_path = Path(form["obstacle_unity_out_json"])
    ensure_parent(str(out_path))
    cmd = [
        PYTHON, "-X", "utf8", "scripts\\export_unity_obstacle_avoidance_demo.py",
        "--plan_json", form["obstacle_plan_json"],
        "--demo_name", form["obstacle_demo_name"],
        "--out_json", str(out_path),
    ]
    return cmd, out_path


def build_core_figures_command(single_case_json: str) -> list[str]:
    return [
        PYTHON, "-X", "utf8", "figure\\scripts\\generate_core_figures.py",
        "--single_case_json", single_case_json,
    ]


def build_workspace_figures_command(reference_dir: str) -> list[str]:
    return [
        PYTHON, "-X", "utf8", "figure\\scripts\\generate_workspace_figures.py",
        "--reference_dir", reference_dir,
    ]


def build_obstacle_figures_command(plan_json: str) -> list[str]:
    return [
        PYTHON, "-X", "utf8", "figure\\scripts\\generate_obstacle_candidate_trajectory_figures.py",
        "--plan_json", plan_json,
    ]
