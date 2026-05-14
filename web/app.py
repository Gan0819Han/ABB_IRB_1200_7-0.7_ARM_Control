#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import mimetypes
import sys
from base64 import b64encode
from datetime import datetime
from pathlib import Path

try:
    from flask import Flask, render_template, request, send_file, url_for
except ImportError as exc:  # pragma: no cover
    raise SystemExit("未安装 Flask。请先在 arm_nn 环境中执行：pip install flask") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import services

app = Flask(__name__, template_folder="templates", static_folder="static")


PREDICT_LABELS = [
    ("topk_shoulder", "肩部分支候选数"),
    ("topk_elbow", "肘部分支候选数"),
    ("topk_wrist", "腕部分支候选数"),
    ("max_branch_candidates", "粗分支最大候选数"),
    ("fine_topk_per_branch", "每个粗分支的细分类候选数"),
    ("max_subspace_candidates", "子空间最大候选数"),
    ("nr_max_iters", "NR 最大迭代次数"),
    ("nr_tol_pos_mm", "NR 位置收敛阈值(mm)"),
    ("nr_tol_ori_rad", "NR 姿态收敛阈值(rad)"),
    ("nr_damping", "NR 阻尼系数"),
    ("nr_step_scale", "NR 步长缩放"),
]

OBSTACLE_LABELS = [
    ("obstacle_topk_shoulder", "肩部分支候选数"),
    ("obstacle_topk_elbow", "肘部分支候选数"),
    ("obstacle_topk_wrist", "腕部分支候选数"),
    ("obstacle_max_branch_candidates", "粗分支最大候选数"),
    ("obstacle_fine_topk_per_branch", "每个粗分支的细分类候选数"),
    ("obstacle_max_subspace_candidates", "子空间最大候选数"),
    ("obstacle_max_evaluated_candidates", "实际评估候选数上限"),
    ("obstacle_nr_max_iters", "NR 最大迭代次数"),
    ("obstacle_nr_tol_pos_mm", "NR 位置收敛阈值(mm)"),
    ("obstacle_nr_tol_ori_rad", "NR 姿态收敛阈值(rad)"),
    ("obstacle_nr_damping", "NR 阻尼系数"),
    ("obstacle_nr_step_scale", "NR 步长缩放"),
    ("trajectory_steps", "轨迹离散步数"),
    ("dedupe_tol_deg", "候选去重容差(deg)"),
]

TAB_TARGETS = {
    "predict": "推理",
    "obstacle": "避障",
    "unity": "Unity",
    "figures": "图表",
}


def build_defaults() -> dict[str, str]:
    path_defaults = services.load_path_defaults()
    return {
        "pose6": "100,200,800,0.1,-0.2,0.3",
        "pred_meta": services.default_path_value(path_defaults, "pred_meta", ROOT / "artifacts" / "prediction_system_formal" / "metadata.json"),
        "branch_meta": services.default_path_value(path_defaults, "branch_meta", ROOT / "artifacts" / "branch_classification_system" / "metadata.json"),
        "fine_meta": services.default_path_value(path_defaults, "fine_meta", ROOT / "artifacts" / "fine_classification_system" / "metadata.json"),
        "predict_out_json": services.default_path_value(path_defaults, "predict_out_json", ROOT / "artifacts" / "gui_outputs" / "predict_result.json"),
        "topk_shoulder": "2",
        "topk_elbow": "1",
        "topk_wrist": "2",
        "max_branch_candidates": "4",
        "fine_topk_per_branch": "3",
        "max_subspace_candidates": "15",
        "enable_nr": "on",
        "nr_max_iters": "40",
        "nr_tol_pos_mm": "1e-3",
        "nr_tol_ori_rad": "1e-3",
        "nr_damping": "1e-5",
        "nr_step_scale": "1.0",
        "q_start": "0,0,0,0,0,0",
        "scene_json": services.default_path_value(path_defaults, "scene_json", ROOT / "data" / "obstacles" / "open_space_reselect_demo.json"),
        "obstacle_out_json": services.default_path_value(path_defaults, "obstacle_out_json", ROOT / "artifacts" / "obstacle_avoidance" / "gui_plan.json"),
        "obstacle_topk_shoulder": "2",
        "obstacle_topk_elbow": "1",
        "obstacle_topk_wrist": "2",
        "obstacle_max_branch_candidates": "6",
        "obstacle_fine_topk_per_branch": "3",
        "obstacle_max_subspace_candidates": "18",
        "obstacle_max_evaluated_candidates": "18",
        "obstacle_nr_max_iters": "40",
        "obstacle_nr_tol_pos_mm": "1e-3",
        "obstacle_nr_tol_ori_rad": "1e-3",
        "obstacle_nr_damping": "1e-5",
        "obstacle_nr_step_scale": "1.0",
        "trajectory_steps": "120",
        "dedupe_tol_deg": "0.5",
        "selected_obstacle_index": "0",
        "obstacle_name": services.DEFAULT_OBSTACLE_NAME,
        "obstacle_center_mm": services.format_q_deg(services.DEFAULT_OBSTACLE_CENTER_MM),
        "obstacle_size_mm": services.format_q_deg(services.DEFAULT_OBSTACLE_SIZE_MM),
        "obstacle_move_step_mm": "10",
        "obstacle_size_step_mm": "10",
        "preview_3d_elev": "22",
        "preview_3d_azim": "-56",
        "fk_q": "20,30,-40,10,20,0",
        "fk_out_json": services.default_path_value(path_defaults, "fk_out_json", services.UNITY_DIR / "Assets" / "ReferenceData" / "gui_fk_reference.json"),
        "traj_q_start": "0,0,0,0,0,0",
        "traj_q_goal": "20,30,-40,10,20,0",
        "traj_steps": "120",
        "traj_duration": "3.0",
        "traj_name": "abb_gui_demo_traj",
        "traj_out_json": services.default_path_value(path_defaults, "traj_out_json", services.UNITY_DIR / "Assets" / "TrajectoryData" / "abb_gui_demo_traj.json"),
        "obstacle_plan_json": services.default_path_value(path_defaults, "obstacle_plan_json", ROOT / "artifacts" / "obstacle_avoidance" / "gui_plan.json"),
        "obstacle_demo_name": "gui_obstacle_demo",
        "obstacle_unity_out_json": services.default_path_value(path_defaults, "obstacle_unity_out_json", services.UNITY_DIR / "Assets" / "PlanningData" / "gui_obstacle_demo_unity.json"),
        "figure_output_dir": services.default_path_value(path_defaults, "figure_output_dir", ROOT / "figure" / "figures"),
        "figure_data_dir": services.default_path_value(path_defaults, "figure_data_dir", ROOT / "figure" / "data"),
        "figure_core_case_json": services.default_path_value(path_defaults, "figure_core_case_json", ROOT / "artifacts" / "gui_outputs" / "predict_result.json"),
        "figure_workspace_ref_dir": services.default_path_value(path_defaults, "figure_workspace_ref_dir", ROOT / "data" / "workspace_reference_samples"),
        "figure_obstacle_plan_json": services.default_path_value(path_defaults, "figure_obstacle_plan_json", ROOT / "artifacts" / "obstacle_avoidance" / "gui_plan.json"),
    }


def merge_form(defaults: dict[str, str]) -> dict[str, str]:
    merged = dict(defaults)
    for key in request.form:
        merged[key] = request.form.get(key, "").strip()
    return merged


def require_text(value: str, *, label: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{label} 不能为空")
    return text


def validate_predict_form(defaults: dict[str, str]) -> None:
    require_text(defaults["pose6"], label="pose6")
    services.preview_target_xyz_mm(defaults["pose6"])
    for key, label in [
        ("pred_meta", "prediction metadata"),
        ("branch_meta", "branch metadata"),
        ("fine_meta", "fine metadata"),
        ("predict_out_json", "输出 JSON"),
    ]:
        require_text(defaults[key], label=label)
    for key, label in PREDICT_LABELS:
        require_text(defaults[key], label=label)


def validate_obstacle_form(defaults: dict[str, str]) -> None:
    require_text(defaults["pose6"], label="pose6")
    require_text(defaults["q_start"], label="q_start")
    require_text(defaults["scene_json"], label="scene_json")
    services.preview_target_xyz_mm(defaults["pose6"])
    services.preview_q_start_deg(defaults["q_start"])
    for key, label in [
        ("pred_meta", "prediction metadata"),
        ("branch_meta", "branch metadata"),
        ("fine_meta", "fine metadata"),
        ("obstacle_out_json", "输出 JSON"),
        ("figure_output_dir", "图像输出目录"),
        ("figure_data_dir", "数据输出目录"),
    ]:
        require_text(defaults[key], label=label)
    for key, label in OBSTACLE_LABELS:
        require_text(defaults[key], label=label)


def validate_fk_form(defaults: dict[str, str]) -> None:
    require_text(defaults["fk_q"], label="关节角 q")
    services.preview_q_start_deg(defaults["fk_q"])
    require_text(defaults["fk_out_json"], label="输出 JSON")


def validate_traj_form(defaults: dict[str, str]) -> None:
    require_text(defaults["traj_q_start"], label="q_start")
    require_text(defaults["traj_q_goal"], label="q_goal")
    services.preview_q_start_deg(defaults["traj_q_start"])
    services.preview_q_start_deg(defaults["traj_q_goal"])
    for key, label in [
        ("traj_steps", "轨迹步数"),
        ("traj_duration", "播放时长(s)"),
        ("traj_name", "轨迹名称"),
        ("traj_out_json", "输出 JSON"),
    ]:
        require_text(defaults[key], label=label)


def validate_obstacle_unity_form(defaults: dict[str, str]) -> None:
    require_text(defaults["obstacle_plan_json"], label="plan_json")
    require_text(defaults["obstacle_demo_name"], label="demo_name")
    require_text(defaults["obstacle_unity_out_json"], label="输出 JSON")


def validate_figures_form(defaults: dict[str, str], *, mode: str) -> None:
    require_text(defaults["figure_output_dir"], label="图像输出目录")
    require_text(defaults["figure_data_dir"], label="数据输出目录")
    if mode == "core":
        require_text(defaults["figure_core_case_json"], label="单案例 IK JSON")
    elif mode == "workspace":
        require_text(defaults["figure_workspace_ref_dir"], label="参考样本目录")
    elif mode == "obstacle":
        require_text(defaults["figure_obstacle_plan_json"], label="避障规划 JSON")


def latest_figure_artifacts(figure_output_dir: str, limit: int = 6) -> list[dict[str, str]]:
    figures_dir = Path(figure_output_dir)
    if not figures_dir.exists():
        return []
    items = sorted(
        [path for path in figures_dir.iterdir() if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]
    return [
        {
            "name": path.name,
            "path": str(path),
            "modified_text": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        }
        for path in items
    ]


def build_sync_hints(defaults: dict[str, str]) -> list[str]:
    hints: list[str] = []
    if defaults.get("predict_out_json"):
        hints.append(f"推理输出默认写入：{defaults['predict_out_json']}")
    if defaults.get("obstacle_out_json"):
        hints.append(f"避障输出默认写入：{defaults['obstacle_out_json']}")
    if defaults.get("obstacle_plan_json") == defaults.get("figure_obstacle_plan_json"):
        hints.append("Unity 页 plan_json 与图表页避障规划 JSON 当前已对齐。")
    if defaults.get("traj_q_goal") == defaults.get("fk_q"):
        hints.append("Unity 页 q_goal 与 FK 参考 q 当前已同步。")
    return hints


def build_obstacle_editor_state(defaults: dict[str, str]) -> tuple[dict, str | None]:
    try:
        selected_index = int(defaults.get("selected_obstacle_index", "0") or "0")
    except ValueError:
        selected_index = 0
    state = {
        "options": [],
        "selected_index": selected_index,
        "status_text": "等待加载 scene_json。",
        "preview_status_text": "等待手动刷新三视图。",
        "preview_svg": "",
        "preview_error": "",
        "preview_3d_status_text": "等待手动刷新 3D 预览。",
        "preview_3d_data_url": "",
        "preview_3d_error": "",
        "preview_3d_elev": defaults.get("preview_3d_elev", "22"),
        "preview_3d_azim": defaults.get("preview_3d_azim", "-56"),
    }
    scene_json = defaults.get("scene_json", "").strip()
    if not scene_json:
        state["status_text"] = "未设置 scene_json。"
        return state, None
    try:
        obstacles, selected_index, values = services.load_selected_obstacle(scene_json, selected_index)
        state["selected_index"] = selected_index
        state["options"] = [
            {"value": str(idx), "label": services.build_obstacle_selector_label(idx, str(obstacle.get("name", f"obstacle_{idx + 1}")))}
            for idx, obstacle in enumerate(obstacles)
        ]
        defaults["selected_obstacle_index"] = str(selected_index)
        defaults["obstacle_name"] = defaults.get("obstacle_name") or values["name"]
        defaults["obstacle_center_mm"] = defaults.get("obstacle_center_mm") or values["center_mm"]
        defaults["obstacle_size_mm"] = defaults.get("obstacle_size_mm") or values["size_mm"]
        state["status_text"] = f"已加载 {len(obstacles)} 个障碍物。当前为 {state['options'][selected_index]['label']}"
        return state, None
    except Exception as exc:
        state["status_text"] = f"读取 scene_json 失败：{exc}"
        return state, str(exc)


def build_preview_state(defaults: dict[str, str], state: dict) -> None:
    try:
        payload, selected_index = services.build_preview_scene_payload(
            defaults["scene_json"],
            selected_index=int(defaults.get("selected_obstacle_index", "0") or "0"),
            obstacle_name=defaults["obstacle_name"],
            center_text=defaults["obstacle_center_mm"],
            size_text=defaults["obstacle_size_mm"],
        )
        state["preview_svg"] = services.build_obstacle_preview_svg(
            payload,
            selected_index=selected_index,
            pose6_text=defaults["pose6"],
            q_start_text=defaults["q_start"],
        )
        current_name = defaults["obstacle_name"].strip() or f"obstacle_{selected_index + 1}"
        state["preview_status_text"] = f"已刷新三视图：当前高亮 {services.build_obstacle_selector_label(selected_index, current_name)}"
        state["preview_error"] = ""
    except Exception as exc:
        state["preview_svg"] = ""
        state["preview_status_text"] = f"三视图刷新失败：{exc}"
        state["preview_error"] = str(exc)


def build_preview_3d_state(defaults: dict[str, str], state: dict) -> None:
    try:
        elev = float(defaults.get("preview_3d_elev", "22") or "22")
        azim = float(defaults.get("preview_3d_azim", "-56") or "-56")
        payload, selected_index = services.build_preview_scene_payload(
            defaults["scene_json"],
            selected_index=int(defaults.get("selected_obstacle_index", "0") or "0"),
            obstacle_name=defaults["obstacle_name"],
            center_text=defaults["obstacle_center_mm"],
            size_text=defaults["obstacle_size_mm"],
        )
        image_bytes = services.build_obstacle_preview_3d_png(
            payload,
            selected_index=selected_index,
            pose6_text=defaults["pose6"],
            q_start_text=defaults["q_start"],
            elev=elev,
            azim=azim,
        )
        state["preview_3d_data_url"] = "data:image/png;base64," + b64encode(image_bytes).decode("ascii")
        current_name = defaults["obstacle_name"].strip() or f"obstacle_{selected_index + 1}"
        state["preview_3d_status_text"] = (
            f"已刷新 3D 预览：当前高亮 {services.build_obstacle_selector_label(selected_index, current_name)}，"
            f"视角 elev={elev:.1f}, azim={azim:.1f}"
        )
        state["preview_3d_error"] = ""
    except Exception as exc:
        state["preview_3d_data_url"] = ""
        state["preview_3d_status_text"] = f"3D 预览刷新失败：{exc}"
        state["preview_3d_error"] = str(exc)


def sync_targets_from_json(defaults: dict[str, str], json_path_text: str) -> None:
    q_deg, _ = services.extract_goal_q_from_json(Path(json_path_text))
    if not q_deg:
        return
    text = services.format_q_deg(q_deg)
    defaults["fk_q"] = text
    defaults["traj_q_goal"] = text


def build_result_payload(command_result: services.CommandResult, *, preview_url: str | None = None) -> dict:
    return {
        "title": command_result.title,
        "exit_code": command_result.exit_code,
        "summary_text": command_result.summary_text,
        "log_text": command_result.log_text,
        "output_path": str(command_result.output_path) if command_result.output_path else "",
        "preview_url": preview_url,
    }


def figure_preview_url(defaults: dict[str, str]) -> str | None:
    preview_path = services.find_latest_obstacle_preview_path(defaults["figure_output_dir"])
    return url_for("preview_image", figure_output_dir=defaults["figure_output_dir"]) if preview_path else None


def render_page(*, active_tab: str, defaults: dict[str, str], result: dict | None = None, obstacle_state: dict | None = None):
    if obstacle_state is None:
        obstacle_state, _ = build_obstacle_editor_state(defaults)
    return render_template(
        "index.html",
        active_tab=active_tab,
        defaults=defaults,
        result=result,
        obstacle_state=obstacle_state,
        predict_labels=PREDICT_LABELS,
        obstacle_labels=OBSTACLE_LABELS,
        sync_hints=build_sync_hints(defaults),
        latest_artifacts=latest_figure_artifacts(defaults.get("figure_output_dir", "")),
    )


def rerender(
    active_tab: str,
    defaults: dict[str, str],
    *,
    result: dict | None = None,
    obstacle_message: str | None = None,
    refresh_preview: bool = False,
    refresh_preview_3d: bool = False,
):
    obstacle_state, error_text = build_obstacle_editor_state(defaults)
    if obstacle_message:
        obstacle_state["status_text"] = obstacle_message
    if refresh_preview:
        build_preview_state(defaults, obstacle_state)
    if refresh_preview_3d:
        build_preview_3d_state(defaults, obstacle_state)
    elif error_text:
        obstacle_state["preview_status_text"] = "当前 scene_json 无法用于预览。"
        obstacle_state["preview_3d_status_text"] = "当前 scene_json 无法用于 3D 预览。"
    return render_page(active_tab=active_tab, defaults=defaults, result=result, obstacle_state=obstacle_state)


@app.get("/")
def index():
    defaults = build_defaults()
    return rerender("predict", defaults)


@app.post("/predict")
def run_predict():
    defaults = merge_form(build_defaults())
    try:
        validate_predict_form(defaults)
    except Exception as exc:
        result = {
            "title": "predict_ik",
            "exit_code": 1,
            "summary_text": f"任务：predict_ik\n\n输入校验失败：{exc}",
            "log_text": f"[predict_ik]\n输入校验失败：{exc}",
            "output_path": defaults.get("predict_out_json", ""),
            "preview_url": None,
        }
        return rerender("predict", defaults, result=result)
    form = {
        "pose6": defaults["pose6"],
        "pred_meta": defaults["pred_meta"],
        "branch_meta": defaults["branch_meta"],
        "fine_meta": defaults["fine_meta"],
        "predict_out_json": defaults["predict_out_json"],
        "topk_shoulder": defaults["topk_shoulder"],
        "topk_elbow": defaults["topk_elbow"],
        "topk_wrist": defaults["topk_wrist"],
        "max_branch_candidates": defaults["max_branch_candidates"],
        "fine_topk_per_branch": defaults["fine_topk_per_branch"],
        "max_subspace_candidates": defaults["max_subspace_candidates"],
        "nr_max_iters": defaults["nr_max_iters"],
        "nr_tol_pos_mm": defaults["nr_tol_pos_mm"],
        "nr_tol_ori_rad": defaults["nr_tol_ori_rad"],
        "nr_damping": defaults["nr_damping"],
        "nr_step_scale": defaults["nr_step_scale"],
    }
    cmd, out_path = services.build_predict_command(form, enable_nr=defaults.get("enable_nr") == "on")
    result = services.run_command_sync("predict_ik", cmd, summary_builder=services.build_predict_summary, summary_path=out_path)
    if result.exit_code == 0:
        sync_targets_from_json(defaults, defaults["predict_out_json"])
        defaults["figure_core_case_json"] = defaults["predict_out_json"]
    return rerender("predict", defaults, result=build_result_payload(result))


@app.post("/obstacle")
def run_obstacle():
    defaults = merge_form(build_defaults())
    save_message = ""
    try:
        validate_obstacle_form(defaults)
        selected_index = int(defaults.get("selected_obstacle_index", "0") or "0")
        save_message = services.save_selected_obstacle(
            defaults["scene_json"],
            selected_index,
            defaults["obstacle_name"],
            defaults["obstacle_center_mm"],
            defaults["obstacle_size_mm"],
        )
    except Exception as exc:
        result = {
            "title": "plan_collision_free_ik",
            "exit_code": 1,
            "summary_text": f"任务：plan_collision_free_ik\n\n输入校验失败：{exc}",
            "log_text": f"[plan_collision_free_ik]\n输入校验失败：{exc}",
            "output_path": defaults.get("obstacle_out_json", ""),
            "preview_url": None,
        }
        return rerender("obstacle", defaults, result=result, refresh_preview=True)
    form = {
        "pose6": defaults["pose6"],
        "q_start": defaults["q_start"],
        "scene_json": defaults["scene_json"],
        "pred_meta": defaults["pred_meta"],
        "branch_meta": defaults["branch_meta"],
        "fine_meta": defaults["fine_meta"],
        "obstacle_out_json": defaults["obstacle_out_json"],
        "topk_shoulder": defaults["obstacle_topk_shoulder"],
        "topk_elbow": defaults["obstacle_topk_elbow"],
        "topk_wrist": defaults["obstacle_topk_wrist"],
        "max_branch_candidates": defaults["obstacle_max_branch_candidates"],
        "fine_topk_per_branch": defaults["obstacle_fine_topk_per_branch"],
        "max_subspace_candidates": defaults["obstacle_max_subspace_candidates"],
        "max_evaluated_candidates": defaults["obstacle_max_evaluated_candidates"],
        "nr_max_iters": defaults["obstacle_nr_max_iters"],
        "nr_tol_pos_mm": defaults["obstacle_nr_tol_pos_mm"],
        "nr_tol_ori_rad": defaults["obstacle_nr_tol_ori_rad"],
        "nr_damping": defaults["obstacle_nr_damping"],
        "nr_step_scale": defaults["obstacle_nr_step_scale"],
        "trajectory_steps": defaults["trajectory_steps"],
        "dedupe_tol_deg": defaults["dedupe_tol_deg"],
    }
    cmd, out_path = services.build_obstacle_command(form)
    result = services.run_command_sync("plan_collision_free_ik", cmd, summary_builder=services.build_obstacle_summary, summary_path=out_path)
    preview_url = None
    if result.exit_code == 0:
        sync_targets_from_json(defaults, defaults["obstacle_out_json"])
        defaults["obstacle_plan_json"] = defaults["obstacle_out_json"]
        defaults["figure_obstacle_plan_json"] = defaults["obstacle_out_json"]
        env = services.build_figure_env(defaults["figure_output_dir"], defaults["figure_data_dir"])
        fig_cmd = services.build_obstacle_figures_command(defaults["figure_obstacle_plan_json"])
        figure_result = services.run_command_sync(
            "generate_obstacle_figures",
            fig_cmd,
            summary_builder=lambda _path: services.build_obstacle_figure_summary(defaults["figure_obstacle_plan_json"], Path(defaults["figure_output_dir"])),
            summary_path=Path(defaults["figure_output_dir"]),
            env=env,
        )
        preview_url = figure_preview_url(defaults)
        result = services.CommandResult(
            title=result.title,
            cmd=result.cmd,
            exit_code=result.exit_code,
            log_text=("[obstacle editor]\n" + save_message + "\n\n" if save_message else "") + result.log_text + "\n\n" + figure_result.log_text,
            summary_text=result.summary_text + ("\n\n当前障碍物已写回 scene_json。" if save_message else "") + "\n\n" + figure_result.summary_text,
            output_path=result.output_path,
        )
    return rerender(
        "figures" if result.exit_code == 0 else "obstacle",
        defaults,
        result=build_result_payload(result, preview_url=preview_url),
        refresh_preview=True,
        refresh_preview_3d=False,
    )


@app.post("/obstacle/editor")
def obstacle_editor_action():
    defaults = merge_form(build_defaults())
    action = request.form.get("editor_action", "").strip()
    selected_index = int(defaults.get("selected_obstacle_index", "0") or "0")
    message = None
    refresh_preview = False
    refresh_preview_3d = False
    try:
        if action == "select":
            obstacles, selected_index, values = services.load_selected_obstacle(defaults["scene_json"], selected_index)
            defaults["selected_obstacle_index"] = str(selected_index)
            defaults["obstacle_name"] = values["name"]
            defaults["obstacle_center_mm"] = values["center_mm"]
            defaults["obstacle_size_mm"] = values["size_mm"]
            message = f"已切换到 {services.build_obstacle_selector_label(selected_index, values['name'])}"
            refresh_preview = True
            refresh_preview_3d = True
        elif action == "load":
            obstacles, selected_index, values = services.load_selected_obstacle(defaults["scene_json"], selected_index)
            defaults["selected_obstacle_index"] = str(selected_index)
            defaults["obstacle_name"] = values["name"]
            defaults["obstacle_center_mm"] = values["center_mm"]
            defaults["obstacle_size_mm"] = values["size_mm"]
            message = f"已从 scene_json 读取 {services.build_obstacle_selector_label(selected_index, values['name'])}"
            refresh_preview = True
            refresh_preview_3d = True
        elif action == "save":
            message = services.save_selected_obstacle(
                defaults["scene_json"],
                selected_index,
                defaults["obstacle_name"],
                defaults["obstacle_center_mm"],
                defaults["obstacle_size_mm"],
            )
            refresh_preview = True
            refresh_preview_3d = True
        elif action == "reset":
            values = services.reset_default_obstacle()
            defaults["obstacle_name"] = values["name"]
            defaults["obstacle_center_mm"] = values["center_mm"]
            defaults["obstacle_size_mm"] = values["size_mm"]
            message = "已恢复默认障碍物参数。"
            refresh_preview = True
            refresh_preview_3d = True
        elif action == "add":
            message, new_index, values = services.add_obstacle(defaults["scene_json"])
            defaults["selected_obstacle_index"] = str(new_index)
            defaults["obstacle_name"] = values["name"]
            defaults["obstacle_center_mm"] = values["center_mm"]
            defaults["obstacle_size_mm"] = values["size_mm"]
            refresh_preview = True
            refresh_preview_3d = True
        elif action == "delete":
            message, next_index, values = services.delete_obstacle(defaults["scene_json"], selected_index)
            defaults["selected_obstacle_index"] = str(next_index)
            defaults["obstacle_name"] = values["name"]
            defaults["obstacle_center_mm"] = values["center_mm"]
            defaults["obstacle_size_mm"] = values["size_mm"]
            refresh_preview = True
            refresh_preview_3d = True
        elif action.startswith("nudge:"):
            _, mode, axis_name, direction_text = action.split(":")
            axis_index = {"x": 0, "y": 1, "z": 2}[axis_name]
            direction = int(direction_text)
            if mode == "center":
                defaults["obstacle_center_mm"] = services.nudge_obstacle_value(
                    defaults["obstacle_center_mm"],
                    step_text=defaults["obstacle_move_step_mm"],
                    index=axis_index,
                    direction=direction,
                    is_size=False,
                )
                message = f"已微调位置 {axis_name.upper()} 轴。"
            else:
                defaults["obstacle_size_mm"] = services.nudge_obstacle_value(
                    defaults["obstacle_size_mm"],
                    step_text=defaults["obstacle_size_step_mm"],
                    index=axis_index,
                    direction=direction,
                    is_size=True,
                )
                message = f"已微调尺寸 d{axis_name}。"
            refresh_preview = True
            refresh_preview_3d = True
        elif action == "preview":
            refresh_preview = True
            message = "已请求刷新三视图。"
        elif action == "preview3d":
            refresh_preview_3d = True
            message = "已请求刷新 3D 预览。"
        else:
            message = "未识别的障碍物编辑操作。"
    except Exception as exc:
        message = f"障碍物编辑失败：{exc}"
    return rerender(
        "obstacle",
        defaults,
        obstacle_message=message,
        refresh_preview=refresh_preview,
        refresh_preview_3d=refresh_preview_3d,
    )


@app.post("/unity/fk")
def run_fk():
    defaults = merge_form(build_defaults())
    try:
        validate_fk_form(defaults)
    except Exception as exc:
        result = {
            "title": "export_unity_fk_reference",
            "exit_code": 1,
            "summary_text": f"任务：export_unity_fk_reference\n\n输入校验失败：{exc}",
            "log_text": f"[export_unity_fk_reference]\n输入校验失败：{exc}",
            "output_path": defaults.get("fk_out_json", ""),
            "preview_url": None,
        }
        return rerender("unity", defaults, result=result)
    form = {"fk_q": defaults["fk_q"], "fk_out_json": defaults["fk_out_json"]}
    cmd, out_path = services.build_fk_export_command(form)
    result = services.run_command_sync("export_unity_fk_reference", cmd, summary_builder=services.build_fk_summary, summary_path=out_path)
    return rerender("unity", defaults, result=build_result_payload(result))


@app.post("/unity/traj")
def run_traj():
    defaults = merge_form(build_defaults())
    try:
        validate_traj_form(defaults)
    except Exception as exc:
        result = {
            "title": "export_unity_trajectory",
            "exit_code": 1,
            "summary_text": f"任务：export_unity_trajectory\n\n输入校验失败：{exc}",
            "log_text": f"[export_unity_trajectory]\n输入校验失败：{exc}",
            "output_path": defaults.get("traj_out_json", ""),
            "preview_url": None,
        }
        return rerender("unity", defaults, result=result)
    form = {
        "traj_q_start": defaults["traj_q_start"],
        "traj_q_goal": defaults["traj_q_goal"],
        "traj_steps": defaults["traj_steps"],
        "traj_duration": defaults["traj_duration"],
        "traj_name": defaults["traj_name"],
        "traj_out_json": defaults["traj_out_json"],
    }
    cmd, out_path = services.build_trajectory_export_command(form)
    result = services.run_command_sync("export_unity_trajectory", cmd, summary_builder=services.build_traj_summary, summary_path=out_path)
    return rerender("unity", defaults, result=build_result_payload(result))


@app.post("/unity/obstacle")
def run_obstacle_unity():
    defaults = merge_form(build_defaults())
    try:
        validate_obstacle_unity_form(defaults)
    except Exception as exc:
        result = {
            "title": "export_unity_obstacle_avoidance_demo",
            "exit_code": 1,
            "summary_text": f"任务：export_unity_obstacle_avoidance_demo\n\n输入校验失败：{exc}",
            "log_text": f"[export_unity_obstacle_avoidance_demo]\n输入校验失败：{exc}",
            "output_path": defaults.get("obstacle_unity_out_json", ""),
            "preview_url": None,
        }
        return rerender("unity", defaults, result=result)
    form = {
        "obstacle_plan_json": defaults["obstacle_plan_json"],
        "obstacle_demo_name": defaults["obstacle_demo_name"],
        "obstacle_unity_out_json": defaults["obstacle_unity_out_json"],
    }
    cmd, out_path = services.build_obstacle_unity_export_command(form)
    result = services.run_command_sync("export_unity_obstacle_avoidance_demo", cmd, summary_builder=services.build_obstacle_unity_summary, summary_path=out_path)
    return rerender("unity", defaults, result=build_result_payload(result))


@app.post("/figures/core")
def run_core_figures():
    defaults = merge_form(build_defaults())
    try:
        validate_figures_form(defaults, mode="core")
    except Exception as exc:
        result = {
            "title": "generate_core_figures",
            "exit_code": 1,
            "summary_text": f"任务：generate_core_figures\n\n输入校验失败：{exc}",
            "log_text": f"[generate_core_figures]\n输入校验失败：{exc}",
            "output_path": defaults.get("figure_output_dir", ""),
            "preview_url": figure_preview_url(defaults),
        }
        return rerender("figures", defaults, result=result)
    env = services.build_figure_env(defaults["figure_output_dir"], defaults["figure_data_dir"])
    cmd = services.build_core_figures_command(defaults["figure_core_case_json"])
    result = services.run_command_sync("generate_core_figures", cmd, env=env)
    return rerender("figures", defaults, result=build_result_payload(result, preview_url=figure_preview_url(defaults)))


@app.post("/figures/workspace")
def run_workspace_figures():
    defaults = merge_form(build_defaults())
    try:
        validate_figures_form(defaults, mode="workspace")
    except Exception as exc:
        result = {
            "title": "generate_workspace_figures",
            "exit_code": 1,
            "summary_text": f"任务：generate_workspace_figures\n\n输入校验失败：{exc}",
            "log_text": f"[generate_workspace_figures]\n输入校验失败：{exc}",
            "output_path": defaults.get("figure_output_dir", ""),
            "preview_url": figure_preview_url(defaults),
        }
        return rerender("figures", defaults, result=result)
    env = services.build_figure_env(defaults["figure_output_dir"], defaults["figure_data_dir"])
    cmd = services.build_workspace_figures_command(defaults["figure_workspace_ref_dir"])
    result = services.run_command_sync("generate_workspace_figures", cmd, env=env)
    return rerender("figures", defaults, result=build_result_payload(result, preview_url=figure_preview_url(defaults)))


@app.post("/figures/obstacle")
def run_obstacle_figures():
    defaults = merge_form(build_defaults())
    try:
        validate_figures_form(defaults, mode="obstacle")
    except Exception as exc:
        result = {
            "title": "generate_obstacle_figures",
            "exit_code": 1,
            "summary_text": f"任务：generate_obstacle_figures\n\n输入校验失败：{exc}",
            "log_text": f"[generate_obstacle_figures]\n输入校验失败：{exc}",
            "output_path": defaults.get("figure_output_dir", ""),
            "preview_url": figure_preview_url(defaults),
        }
        return rerender("figures", defaults, result=result)
    defaults["obstacle_plan_json"] = defaults["figure_obstacle_plan_json"]
    env = services.build_figure_env(defaults["figure_output_dir"], defaults["figure_data_dir"])
    cmd = services.build_obstacle_figures_command(defaults["figure_obstacle_plan_json"])
    result = services.run_command_sync(
        "generate_obstacle_figures",
        cmd,
        summary_builder=lambda _path: services.build_obstacle_figure_summary(defaults["figure_obstacle_plan_json"], Path(defaults["figure_output_dir"])),
        summary_path=Path(defaults["figure_output_dir"]),
        env=env,
    )
    return rerender("figures", defaults, result=build_result_payload(result, preview_url=figure_preview_url(defaults)))


@app.get("/preview-image")
def preview_image():
    figure_output_dir = request.args.get("figure_output_dir", "").strip()
    if not figure_output_dir:
        defaults = build_defaults()
        figure_output_dir = defaults["figure_output_dir"]
    preview_path = services.find_latest_obstacle_preview_path(figure_output_dir)
    if preview_path is None:
        return "<p>暂无预览</p>", 404
    return send_file(preview_path, mimetype=mimetypes.guess_type(str(preview_path))[0] or "image/png")


@app.get("/healthz")
def healthz():
    return {"ok": True, "root": str(ROOT)}


def main() -> None:
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
