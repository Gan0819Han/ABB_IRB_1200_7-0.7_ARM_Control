#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from abb_nn.optimization import DLSOptions, LBFGSBOptions, NROptions
from fk_model import pose6_from_q
from obstacle_avoidance.collision import ObstacleScene
from obstacle_avoidance.planning import TrajectorySelectionWeights, build_piecewise_joint_trajectory_deg
from scripts.compare_obstacle_avoidance_methods import (
    build_method_record,
    parse_pose as parse_obstacle_pose,
    parse_q_deg as parse_obstacle_q_deg,
    solve_dls_goal,
    solve_lbfgsb_goal,
    solve_nn_nr_goal,
)
from scripts.export_unity_method_comparison import (
    parse_pose as parse_ik_pose,
    parse_q_deg as parse_ik_q_deg,
    solve_dls,
    solve_lbfgsb,
    solve_nn_nr,
)

ENGINEERING_SCHEMA = "abb_engineering_case_stats_v1"
IK_MODULE = "ik_single"
OBSTACLE_MODULE = "obstacle_single"
METHOD_COLORS = {
    "nn_nr": "#2563EB",
    "dls": "#F59E0B",
    "lbfgsb": "#16A34A",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export single-case engineering statistics packages for GUI and thesis workflows.")
    parser.add_argument("--module", choices=[IK_MODULE, OBSTACLE_MODULE], required=True)
    parser.add_argument("--action", choices=["run", "plot"], required=True)
    parser.add_argument("--case_root", required=True)
    parser.add_argument("--case_tag", required=True)
    parser.add_argument("--pose", default="100,200,800,0.1,-0.2,0.3")
    parser.add_argument("--q_start", default="0,0,0,0,0,0")
    parser.add_argument("--scene_json", default="")
    parser.add_argument("--pred_meta", default="artifacts/prediction_system_formal/metadata.json")
    parser.add_argument("--branch_meta", default="artifacts/branch_classification_system/metadata.json")
    parser.add_argument("--fine_meta", default="artifacts/fine_classification_system/metadata.json")
    parser.add_argument("--topk_shoulder", type=int, default=2)
    parser.add_argument("--topk_elbow", type=int, default=1)
    parser.add_argument("--topk_wrist", type=int, default=2)
    parser.add_argument("--max_branch_candidates", type=int, default=6)
    parser.add_argument("--fine_topk_per_branch", type=int, default=3)
    parser.add_argument("--max_subspace_candidates", type=int, default=18)
    parser.add_argument("--nr_max_iters", type=int, default=40)
    parser.add_argument("--nr_tol_pos_mm", type=float, default=1e-3)
    parser.add_argument("--nr_tol_ori_rad", type=float, default=1e-3)
    parser.add_argument("--nr_damping", type=float, default=1e-5)
    parser.add_argument("--nr_step_scale", type=float, default=1.0)
    parser.add_argument("--trajectory_steps", type=int, default=120)
    return parser


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bool_zh(value: object) -> str:
    return "是" if bool(value) else "否"


def fmt_num(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if np.isnan(number) or np.isinf(number):
        return "-"
    return f"{number:.{digits}f}"


def markdown_table(headers: list[str], rows: list[dict[str, str]]) -> str:
    if not rows:
        return "| 项 | 值 |\n| --- | --- |\n| 状态 | 暂无数据 |"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "-")) for header in headers) + " |")
    return "\n".join(lines)


def write_summary_files(module_dir: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    csv_path = module_dir / "summary_zh.csv"
    md_path = module_dir / "summary_zh.md"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    md_title = "## 中文汇总表"
    md_content = markdown_table(headers, rows)
    md_path.write_text(f"{md_title}\n\n{md_content}\n", encoding="utf-8")


def write_field_dictionary(case_dir: Path) -> None:
    entries = [
        {"field_key": "method_id", "field_name_zh": "方法ID", "unit": "-", "meaning": "脚本内部使用的方法标识", "module": "ik_single, obstacle_single"},
        {"field_key": "label", "field_name_zh": "方法名称", "unit": "-", "meaning": "用于界面和论文展示的中文/英文方法名", "module": "ik_single, obstacle_single"},
        {"field_key": "solve_time_ms", "field_name_zh": "求解时间", "unit": "ms", "meaning": "普通逆解阶段总耗时", "module": "ik_single"},
        {"field_key": "pure_inverse_compute_ms", "field_name_zh": "纯求逆时间", "unit": "ms", "meaning": "去除模型加载与恢复后，仅保留逆解计算本身的时间", "module": "ik_single, obstacle_single"},
        {"field_key": "model_load_restore_ms", "field_name_zh": "模型加载与恢复时间", "unit": "ms", "meaning": "神经网络模型加载与状态恢复耗时", "module": "ik_single, obstacle_single"},
        {"field_key": "metadata_load_ms", "field_name_zh": "元数据读取时间", "unit": "ms", "meaning": "读取 metadata、归一化参数等辅助数据的耗时", "module": "ik_single, obstacle_single"},
        {"field_key": "candidate_generation_ms", "field_name_zh": "候选分类总时间", "unit": "ms", "meaning": "候选子空间召回、粗分类和细分类总耗时", "module": "ik_single, obstacle_single"},
        {"field_key": "prediction_ms", "field_name_zh": "网络预测时间", "unit": "ms", "meaning": "神经网络前向预测耗时", "module": "ik_single, obstacle_single"},
        {"field_key": "nr_refine_ms", "field_name_zh": "NR精修时间", "unit": "ms", "meaning": "Newton-Raphson 精修阶段耗时", "module": "ik_single, obstacle_single"},
        {"field_key": "uninstrumented_overhead_ms", "field_name_zh": "未单独计时开销", "unit": "ms", "meaning": "当前脚本未拆成独立计时段的 Python 调度、数据转换与胶水逻辑耗时", "module": "ik_single, obstacle_single"},
        {"field_key": "final_pos_err_mm", "field_name_zh": "最终位置误差", "unit": "mm", "meaning": "末端位姿解的位置误差", "module": "ik_single, obstacle_single"},
        {"field_key": "final_ori_err_rad", "field_name_zh": "最终姿态误差", "unit": "rad", "meaning": "末端位姿解的姿态误差", "module": "ik_single, obstacle_single"},
        {"field_key": "iters", "field_name_zh": "迭代次数", "unit": "-", "meaning": "数值修正或优化器迭代次数", "module": "ik_single, obstacle_single"},
        {"field_key": "converged", "field_name_zh": "是否收敛", "unit": "-", "meaning": "求解过程是否满足收敛条件", "module": "ik_single, obstacle_single"},
        {"field_key": "within_joint_limits", "field_name_zh": "满足关节限位", "unit": "-", "meaning": "结果关节角是否位于机械臂限位范围内", "module": "ik_single"},
        {"field_key": "planning_time_ms", "field_name_zh": "总规划时间", "unit": "ms", "meaning": "避障单解从末端逆解到轨迹筛选的总耗时", "module": "obstacle_single"},
        {"field_key": "ik_time_ms", "field_name_zh": "末端逆解时间", "unit": "ms", "meaning": "避障链路里仅求出末端关节解的时间", "module": "obstacle_single"},
        {"field_key": "trajectory_generation_time_ms", "field_name_zh": "轨迹生成时间", "unit": "ms", "meaning": "构造候选关节轨迹的时间", "module": "obstacle_single"},
        {"field_key": "trajectory_evaluation_time_ms", "field_name_zh": "轨迹评估时间", "unit": "ms", "meaning": "碰撞检测与净空评估耗时", "module": "obstacle_single"},
        {"field_key": "selection_time_ms", "field_name_zh": "方案筛选时间", "unit": "ms", "meaning": "候选轨迹计算代价并排序的时间", "module": "obstacle_single"},
        {"field_key": "selected_solution_collision_free", "field_name_zh": "是否无碰撞", "unit": "-", "meaning": "最终选中的轨迹是否无碰撞", "module": "obstacle_single"},
        {"field_key": "collision_frame_count", "field_name_zh": "碰撞帧数", "unit": "帧", "meaning": "轨迹离散帧中发生碰撞的帧数", "module": "obstacle_single"},
        {"field_key": "min_clearance_mm", "field_name_zh": "最小净空", "unit": "mm", "meaning": "轨迹全过程的最小障碍物净空", "module": "obstacle_single"},
        {"field_key": "joint_path_length_deg", "field_name_zh": "路径长度", "unit": "deg", "meaning": "六个关节累计路径长度", "module": "obstacle_single"},
        {"field_key": "max_joint_step_deg", "field_name_zh": "最大单步关节变化", "unit": "deg", "meaning": "离散轨迹中单步最大的关节变化量", "module": "obstacle_single"},
        {"field_key": "trajectory_mode", "field_name_zh": "轨迹模式", "unit": "-", "meaning": "选中的轨迹模式，例如 direct 或 via_waypoint", "module": "obstacle_single"},
        {"field_key": "initial_guess_count", "field_name_zh": "初值数量", "unit": "-", "meaning": "数值法尝试的初始关节解数量", "module": "obstacle_single"},
        {"field_key": "unique_goal_candidate_count", "field_name_zh": "唯一终点候选数", "unit": "-", "meaning": "数值法去重后的终点关节候选数量", "module": "obstacle_single"},
    ]
    for idx in range(6):
        entries.append(
            {
                "field_key": f"q{idx + 1}_deg",
                "field_name_zh": f"关节 q{idx + 1}",
                "unit": "deg",
                "meaning": "该帧对应的关节角度",
                "module": "obstacle_single motion_trace",
            }
        )
    for key, label, unit in [
        ("frame_index", "帧序号", "-"),
        ("time_index", "归一化时间", "-"),
        ("x_mm", "末端 X", "mm"),
        ("y_mm", "末端 Y", "mm"),
        ("z_mm", "末端 Z", "mm"),
        ("phi_rad", "末端 phi", "rad"),
        ("theta_rad", "末端 theta", "rad"),
        ("psi_rad", "末端 psi", "rad"),
        ("target_x_mm", "目标 X", "mm"),
        ("target_y_mm", "目标 Y", "mm"),
        ("target_z_mm", "目标 Z", "mm"),
        ("target_phi_rad", "目标 phi", "rad"),
        ("target_theta_rad", "目标 theta", "rad"),
        ("target_psi_rad", "目标 psi", "rad"),
        ("collision", "该帧是否碰撞", "-"),
        ("min_clearance_mm", "该帧最小净空", "mm"),
    ]:
        entries.append(
            {
                "field_key": key,
                "field_name_zh": label,
                "unit": unit,
                "meaning": "单解轨迹逐帧记录字段",
                "module": "obstacle_single motion_trace",
            }
        )
    headers = ["field_key", "field_name_zh", "unit", "meaning", "module"]
    csv_path = case_dir / "field_dictionary_zh.csv"
    md_path = case_dir / "field_dictionary_zh.md"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(entries)
    md_rows = [{header: item[header] for header in headers} for item in entries]
    md_path.write_text("## 字段中文说明\n\n" + markdown_table(headers, md_rows) + "\n", encoding="utf-8")


def write_shared_snapshot(case_dir: Path, args: argparse.Namespace, *, scene_name: str) -> None:
    payload = {
        "schema": ENGINEERING_SCHEMA,
        "case_tag": args.case_tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pose6": args.pose,
        "q_start": args.q_start,
        "scene_json": args.scene_json,
        "scene_name": scene_name,
        "prediction_metadata": args.pred_meta,
        "branch_metadata": args.branch_meta,
        "fine_metadata": args.fine_meta,
    }
    save_json(case_dir / "shared_input_snapshot.json", payload)


def save_case_metadata(case_dir: Path, args: argparse.Namespace, *, scene_name: str) -> None:
    write_field_dictionary(case_dir)
    write_shared_snapshot(case_dir, args, scene_name=scene_name)


def build_case_dirs(case_root: str, case_tag: str) -> tuple[Path, Path, Path]:
    case_dir = ensure_dir(Path(case_root) / case_tag)
    ik_dir = ensure_dir(case_dir / IK_MODULE)
    obstacle_dir = ensure_dir(case_dir / OBSTACLE_MODULE)
    return case_dir, ik_dir, obstacle_dir


def annotate_bar_values(ax: plt.Axes, *, scale: float = 1.0, fmt: str = "{:.3f}") -> None:
    ymax = max((patch.get_height() for patch in ax.patches), default=0.0)
    offset = ymax * 0.03 if ymax > 0 else 0.05
    ymin, cur_ymax = ax.get_ylim()
    top_needed = cur_ymax
    for patch in ax.patches:
        value = patch.get_height()
        text_y = value + offset
        top_needed = max(top_needed, text_y + offset)
        ax.text(
            patch.get_x() + patch.get_width() * 0.5,
            text_y,
            fmt.format(value * scale),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#334155",
            clip_on=False,
        )
    if top_needed > cur_ymax:
        ax.set_ylim(ymin, top_needed)


def write_timing_breakdown_files(
    module_dir: Path,
    filename_prefix: str,
    headers: list[str],
    rows: list[dict[str, str]],
    *,
    title: str,
) -> None:
    csv_path = module_dir / f"{filename_prefix}.csv"
    md_path = module_dir / f"{filename_prefix}.md"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(f"## {title}\n\n" + markdown_table(headers, rows) + "\n", encoding="utf-8")


def get_timing_breakdown(method: dict, key: str, default: object = None) -> object:
    timing = method.get("timing_breakdown_ms", {})
    if key in timing:
        return timing.get(key)
    solver_timing = method.get("solver_details", {}).get("ik_timing_breakdown_ms", {})
    if key in solver_timing:
        return solver_timing.get(key)
    return default


def build_ik_timing_rows(methods: list[dict]) -> tuple[list[str], list[dict[str, str]]]:
    headers = [
        "方法",
        "总逆解时间(ms)",
        "去除模型加载后的纯求逆时间(ms)",
        "模型加载与恢复(ms)",
        "元数据读取(ms)",
        "候选分类总时间(ms)",
        "分支粗分类(ms)",
        "分支细分类(ms)",
        "网络预测(ms)",
        "初值打分(ms)",
        "NR精修(ms)",
        "未单独计时开销(ms)",
        "优化器求解(ms)",
        "结果评估(ms)",
    ]
    rows: list[dict[str, str]] = []
    for method in methods:
        timing = method.get("timing_breakdown_ms", {})
        rows.append(
            {
                "方法": str(method.get("label", method.get("method_id", "-"))),
                "总逆解时间(ms)": fmt_num(method.get("solve_time_ms"), 3),
                "去除模型加载后的纯求逆时间(ms)": fmt_num(timing.get("pure_inverse_compute_ms", method.get("solve_time_ms")), 3),
                "模型加载与恢复(ms)": fmt_num(timing.get("model_load_restore_ms"), 3),
                "元数据读取(ms)": fmt_num(timing.get("metadata_load_ms"), 3),
                "候选分类总时间(ms)": fmt_num(timing.get("candidate_generation_ms"), 3),
                "分支粗分类(ms)": fmt_num(timing.get("branch_classification_ms"), 3),
                "分支细分类(ms)": fmt_num(timing.get("fine_classification_ms"), 3),
                "网络预测(ms)": fmt_num(timing.get("prediction_ms"), 3),
                "初值打分(ms)": fmt_num(timing.get("initial_pick_scoring_ms"), 3),
                "NR精修(ms)": fmt_num(timing.get("nr_refine_ms"), 3),
                "未单独计时开销(ms)": fmt_num(timing.get("uninstrumented_overhead_ms"), 3),
                "优化器求解(ms)": fmt_num(timing.get("optimizer_solve_ms"), 3),
                "结果评估(ms)": fmt_num(timing.get("metrics_eval_ms"), 3),
            }
        )
    return headers, rows


def build_obstacle_ik_timing_rows(methods: list[dict]) -> tuple[list[str], list[dict[str, str]]]:
    headers = [
        "方法",
        "末端逆解总时间(ms)",
        "去除模型加载后的纯求逆时间(ms)",
        "模型加载与恢复(ms)",
        "元数据读取(ms)",
        "候选分类总时间(ms)",
        "分支粗分类(ms)",
        "分支细分类(ms)",
        "网络预测(ms)",
        "初值打分(ms)",
        "NR精修(ms)",
        "未单独计时开销(ms)",
        "多初值总时间(ms)",
        "单起点平均(ms)",
        "选中起点求解(ms)",
        "初值数量",
        "唯一终点候选数",
    ]
    rows: list[dict[str, str]] = []
    for method in methods:
        rows.append(
            {
                "方法": str(method.get("label", method.get("method_id", "-"))),
                "末端逆解总时间(ms)": fmt_num(method.get("ik_time_ms"), 3),
                "去除模型加载后的纯求逆时间(ms)": fmt_num(
                    method.get("pure_inverse_compute_ms", get_timing_breakdown(method, "pure_inverse_compute_ms", method.get("ik_time_ms"))),
                    3,
                ),
                "模型加载与恢复(ms)": fmt_num(
                    method.get("model_load_restore_ms", get_timing_breakdown(method, "model_load_restore_ms")),
                    3,
                ),
                "元数据读取(ms)": fmt_num(get_timing_breakdown(method, "metadata_load_ms"), 3),
                "候选分类总时间(ms)": fmt_num(get_timing_breakdown(method, "candidate_generation_ms"), 3),
                "分支粗分类(ms)": fmt_num(get_timing_breakdown(method, "branch_classification_ms"), 3),
                "分支细分类(ms)": fmt_num(get_timing_breakdown(method, "fine_classification_ms"), 3),
                "网络预测(ms)": fmt_num(get_timing_breakdown(method, "prediction_ms"), 3),
                "初值打分(ms)": fmt_num(get_timing_breakdown(method, "initial_pick_scoring_ms"), 3),
                "NR精修(ms)": fmt_num(get_timing_breakdown(method, "nr_refine_ms"), 3),
                "未单独计时开销(ms)": fmt_num(get_timing_breakdown(method, "uninstrumented_overhead_ms"), 3),
                "多初值总时间(ms)": fmt_num(get_timing_breakdown(method, "multistart_total_ms"), 3),
                "单起点平均(ms)": fmt_num(
                    method.get("mean_per_start_ms", get_timing_breakdown(method, "mean_per_start_ms", method.get("ik_time_ms"))),
                    3,
                ),
                "选中起点求解(ms)": fmt_num(
                    method.get("selected_start_solve_ms", get_timing_breakdown(method, "selected_start_solve_ms", method.get("ik_time_ms"))),
                    3,
                ),
                "初值数量": str(method.get("initial_guess_count", "-")),
                "唯一终点候选数": str(method.get("unique_goal_candidate_count", "-")),
            }
        )
    return headers, rows


def build_motion_trace_rows(method: dict, *, target_pose6: list[float]) -> list[dict[str, object]]:
    frames = method.get("selected_solution", {}).get("trajectory_summary", {}).get("frames", [])
    if not frames:
        q_points = np.asarray(method["selected_solution"]["trajectory_points_deg"], dtype=float)
        q_traj_deg = build_piecewise_joint_trajectory_deg(q_points_deg=q_points, steps=int(method["selected_solution"]["trajectory_summary"]["trajectory_steps"]))
        frames = [
            {
                "frame_index": int(idx),
                "q_deg": q_deg.tolist(),
                "collision": False,
                "min_clearance_mm": float("nan"),
            }
            for idx, q_deg in enumerate(q_traj_deg)
        ]

    target = [float(v) for v in target_pose6]
    total_frames = max(1, len(frames) - 1)
    rows: list[dict[str, object]] = []
    for frame in frames:
        q_deg = [float(v) for v in frame["q_deg"]]
        pose6 = [float(v) for v in pose6_from_q(q_deg, input_unit="deg").tolist()]
        row: dict[str, object] = {
            "frame_index": int(frame["frame_index"]),
            "time_index": float(frame["frame_index"]) / float(total_frames),
            "x_mm": pose6[0],
            "y_mm": pose6[1],
            "z_mm": pose6[2],
            "phi_rad": pose6[3],
            "theta_rad": pose6[4],
            "psi_rad": pose6[5],
            "target_x_mm": target[0],
            "target_y_mm": target[1],
            "target_z_mm": target[2],
            "target_phi_rad": target[3],
            "target_theta_rad": target[4],
            "target_psi_rad": target[5],
            "collision": int(bool(frame.get("collision", False))),
            "min_clearance_mm": float(frame.get("min_clearance_mm", float("nan"))),
        }
        for idx, value in enumerate(q_deg, start=1):
            row[f"q{idx}_deg"] = value
        rows.append(row)
    return rows


def write_motion_trace_csv(module_dir: Path, method: dict, *, target_pose6: list[float]) -> Path:
    rows = build_motion_trace_rows(method, target_pose6=target_pose6)
    csv_path = module_dir / f"obstacle_motion_trace_{method['method_id']}.csv"
    headers = [
        "frame_index",
        "time_index",
        "q1_deg",
        "q2_deg",
        "q3_deg",
        "q4_deg",
        "q5_deg",
        "q6_deg",
        "x_mm",
        "y_mm",
        "z_mm",
        "phi_rad",
        "theta_rad",
        "psi_rad",
        "target_x_mm",
        "target_y_mm",
        "target_z_mm",
        "target_phi_rad",
        "target_theta_rad",
        "target_psi_rad",
        "collision",
        "min_clearance_mm",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot_motion_pose6(module_dir: Path, method: dict, *, target_pose6: list[float]) -> Path:
    rows = build_motion_trace_rows(method, target_pose6=target_pose6)
    frame_index = [int(row["frame_index"]) for row in rows]
    labels = [
        ("x_mm", "target_x_mm", "X 方向末端运动", "X (mm)"),
        ("y_mm", "target_y_mm", "Y 方向末端运动", "Y (mm)"),
        ("z_mm", "target_z_mm", "Z 方向末端运动", "Z (mm)"),
        ("phi_rad", "target_phi_rad", "phi 姿态分量变化", "phi (rad)"),
        ("theta_rad", "target_theta_rad", "theta 姿态分量变化", "theta (rad)"),
        ("psi_rad", "target_psi_rad", "psi 姿态分量变化", "psi (rad)"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.4))
    axes_flat = axes.reshape(-1)
    for ax, (value_key, target_key, title, ylabel) in zip(axes_flat, labels):
        values = [float(row[value_key]) for row in rows]
        target_val = float(rows[0][target_key]) if rows else 0.0
        ax.plot(frame_index, values, color="#F59E0B", linewidth=1.8, label="实际轨迹")
        ax.axhline(target_val, color="#2563EB", linewidth=1.1, linestyle="--", label="目标位姿")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("帧序号")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    handles, labels_text = axes_flat[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels_text,
        loc="upper left",
        bbox_to_anchor=(0.11, 0.975),
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="#64748B",
        fontsize=9,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.95)
    fig.suptitle(f"{method['label']}：末端位姿随轨迹推进的变化", fontsize=13, y=0.94)
    fig.subplots_adjust(top=0.88)
    output_path = module_dir / f"obstacle_motion_pose6_{method['method_id']}.png"
    save_figure(fig, output_path)
    return output_path


def build_nr_options(args: argparse.Namespace) -> NROptions:
    return NROptions(
        max_iters=args.nr_max_iters,
        tol_pos_mm=args.nr_tol_pos_mm,
        tol_ori_rad=args.nr_tol_ori_rad,
        damping=args.nr_damping,
        step_scale=args.nr_step_scale,
    )


def build_dls_options() -> DLSOptions:
    options = DLSOptions(
        max_iters=80,
        tol_pos_mm=1.0,
        tol_ori_rad=1e-2,
        damping=1e-2,
        orientation_weight=200.0,
    )
    return options


def build_lbfgsb_options() -> LBFGSBOptions:
    return LBFGSBOptions(
        max_iters=200,
        tol_pos_mm=1.0,
        tol_ori_rad=1e-2,
        orientation_weight=200.0,
    )


def build_selection_weights() -> TrajectorySelectionWeights:
    return TrajectorySelectionWeights(
        collision_flag_weight=1.0e6,
        collision_frame_weight=1.0e4,
        collision_violation_weight=100.0,
        accuracy_violation_weight=1.0e3,
        joint_path_weight=1.0,
        max_joint_step_weight=0.25,
        clearance_reward_weight=0.1,
        clearance_reward_cap_mm=100.0,
        pos_tol_mm=1.0,
        ori_tol_rad=1.0e-2,
    )


def export_ik_single(case_dir: Path, args: argparse.Namespace) -> Path:
    module_dir = ensure_dir(case_dir / IK_MODULE)
    target_pose = parse_ik_pose(args.pose)
    q_start_deg = parse_ik_q_deg(args.q_start)
    nr_options = build_nr_options(args)
    dls_options = build_dls_options()
    lbfgsb_options = build_lbfgsb_options()

    methods = [
        solve_nn_nr(
            target_pose=target_pose,
            pred_meta_path=Path(args.pred_meta),
            branch_meta_path=Path(args.branch_meta),
            fine_meta_path=Path(args.fine_meta),
            topk_shoulder=args.topk_shoulder,
            topk_elbow=args.topk_elbow,
            topk_wrist=args.topk_wrist,
            max_branch_candidates=args.max_branch_candidates,
            fine_topk_per_branch=args.fine_topk_per_branch,
            max_subspace_candidates=args.max_subspace_candidates,
            nr_options=nr_options,
        ),
        solve_dls(target_pose=target_pose, q_start_deg=q_start_deg, options=dls_options),
        solve_lbfgsb(target_pose=target_pose, q_start_deg=q_start_deg, options=lbfgsb_options),
    ]

    for method in methods:
        method["solver_family"] = "nn" if method["method_id"] == "nn_nr" else "numeric"
        method["within_joint_limits"] = True
        method["q_result_deg"] = method.pop("q_goal_deg")

    raw_payload = {
        "schema": ENGINEERING_SCHEMA,
        "module": IK_MODULE,
        "case_tag": args.case_tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_pose6": target_pose.tolist(),
        "q_start_deg": q_start_deg.tolist(),
        "methods": methods,
    }
    raw_path = module_dir / "raw_result.json"
    save_json(raw_path, raw_payload)

    headers = ["方法", "求解时间(ms)", "最终位置误差(mm)", "最终姿态误差(rad)", "迭代次数", "是否收敛", "是否满足关节限位"]
    rows = [
        {
            "方法": method["label"],
            "求解时间(ms)": fmt_num(method.get("solve_time_ms"), 3),
            "最终位置误差(mm)": fmt_num(method.get("final_pos_err_mm"), 6),
            "最终姿态误差(rad)": fmt_num(method.get("final_ori_err_rad"), 8),
            "迭代次数": str(method.get("iters", "-")),
            "是否收敛": bool_zh(method.get("converged")),
            "是否满足关节限位": bool_zh(method.get("within_joint_limits")),
        }
        for method in methods
    ]
    write_summary_files(module_dir, headers, rows)
    timing_headers, timing_rows = build_ik_timing_rows(methods)
    write_timing_breakdown_files(
        module_dir,
        "ik_timing_breakdown_zh",
        timing_headers,
        timing_rows,
        title="普通逆解单解时间细分对比",
    )
    save_case_metadata(case_dir, args, scene_name="")
    return raw_path


def export_obstacle_single(case_dir: Path, args: argparse.Namespace) -> Path:
    module_dir = ensure_dir(case_dir / OBSTACLE_MODULE)
    target_pose = parse_obstacle_pose(args.pose)
    q_start_deg = parse_obstacle_q_deg(args.q_start)
    scene = ObstacleScene.from_json(args.scene_json)
    nr_options = build_nr_options(args)
    dls_options = build_dls_options()
    dls_options.multistart_guesses = 12
    dls_options.dedupe_tol_deg = 0.5
    lbfgsb_options = build_lbfgsb_options()
    lbfgsb_options.multistart_guesses = 12
    lbfgsb_options.dedupe_tol_deg = 0.5

    raw_methods = [
        solve_nn_nr_goal(
            target_pose=target_pose,
            pred_meta_path=Path(args.pred_meta),
            branch_meta_path=Path(args.branch_meta),
            fine_meta_path=Path(args.fine_meta),
            topk_shoulder=args.topk_shoulder,
            topk_elbow=args.topk_elbow,
            topk_wrist=args.topk_wrist,
            max_branch_candidates=args.max_branch_candidates,
            fine_topk_per_branch=args.fine_topk_per_branch,
            max_subspace_candidates=args.max_subspace_candidates,
            dedupe_tol_deg=0.5,
            nr_options=nr_options,
        ),
        solve_dls_goal(target_pose=target_pose, q_start_deg=q_start_deg, options=dls_options),
        solve_lbfgsb_goal(target_pose=target_pose, q_start_deg=q_start_deg, options=lbfgsb_options),
    ]
    methods = [
        build_method_record(
            method=item,
            q_start_deg=q_start_deg,
            scene=scene,
            selection_weights=build_selection_weights(),
            trajectory_steps=args.trajectory_steps,
            include_frames=True,
        )
        for item in raw_methods
    ]
    raw_payload = {
        "schema": ENGINEERING_SCHEMA,
        "module": OBSTACLE_MODULE,
        "case_tag": args.case_tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_pose6": target_pose.tolist(),
        "q_start_deg": q_start_deg.tolist(),
        "scene_name": scene.scene_name,
        "scene_json": args.scene_json,
        "trajectory_steps": args.trajectory_steps,
        "methods": methods,
    }
    raw_path = module_dir / "raw_result.json"
    save_json(raw_path, raw_payload)

    headers = ["方法", "总耗时(ms)", "末端逆解耗时(ms)", "轨迹评估耗时(ms)", "最终位置误差(mm)", "最终姿态误差(rad)", "是否成功", "碰撞帧数", "最小净空(mm)", "路径长度(deg)", "最大单步关节变化(deg)", "轨迹模式"]
    rows = [
        {
            "方法": method["label"],
            "总耗时(ms)": fmt_num(method.get("planning_time_ms"), 3),
            "末端逆解耗时(ms)": fmt_num(method.get("ik_time_ms"), 3),
            "轨迹评估耗时(ms)": fmt_num(method.get("trajectory_evaluation_time_ms"), 3),
            "最终位置误差(mm)": fmt_num(method.get("final_pos_err_mm"), 6),
            "最终姿态误差(rad)": fmt_num(method.get("final_ori_err_rad"), 8),
            "是否成功": bool_zh(method.get("selected_solution_collision_free")),
            "碰撞帧数": str(method.get("collision_frame_count", "-")),
            "最小净空(mm)": fmt_num(method.get("min_clearance_mm"), 4),
            "路径长度(deg)": fmt_num(method.get("joint_path_length_deg"), 3),
            "最大单步关节变化(deg)": fmt_num(method.get("max_joint_step_deg"), 4),
            "轨迹模式": str(method.get("trajectory_mode", "-")),
        }
        for method in methods
    ]
    write_summary_files(module_dir, headers, rows)
    timing_headers, timing_rows = build_obstacle_ik_timing_rows(methods)
    write_timing_breakdown_files(
        module_dir,
        "obstacle_ik_timing_breakdown_zh",
        timing_headers,
        timing_rows,
        title="避障单解末端逆解时间细分对比",
    )
    save_case_metadata(case_dir, args, scene_name=scene.scene_name)
    return raw_path


def configure_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def method_labels_and_colors(methods: list[dict]) -> tuple[list[str], list[str]]:
    labels = [str(item.get("label", item.get("method_id", "-"))) for item in methods]
    colors = [METHOD_COLORS.get(str(item.get("method_id", "")), "#64748B") for item in methods]
    return labels, colors


def save_figure(fig: plt.Figure, path: Path) -> None:
    if fig._suptitle is not None:
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ik_single(case_dir: Path) -> list[Path]:
    configure_plot_style()
    module_dir = case_dir / IK_MODULE
    payload = load_json(module_dir / "raw_result.json")
    methods = payload.get("methods", [])
    labels, colors = method_labels_and_colors(methods)
    outputs: list[Path] = []

    x = np.arange(len(labels), dtype=float)
    width = 0.36
    total_vals = np.asarray([float(item.get("solve_time_ms", 0.0)) for item in methods], dtype=float)
    pure_vals = np.asarray([
        float(item.get("timing_breakdown_ms", {}).get("pure_inverse_compute_ms", item.get("solve_time_ms", 0.0)))
        for item in methods
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - width * 0.5, total_vals, width=width, color=colors, alpha=0.95, label="总逆解时间")
    ax.bar(x + width * 0.5, pure_vals, width=width, color="#93C5FD", edgecolor="#1D4ED8", label="去除模型加载后的纯求逆时间")
    ax.set_xticks(x, labels)
    ax.set_title("普通逆解单解：总时间与纯求逆时间对比")
    ax.set_ylabel("时间 (ms)")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    annotate_bar_values(ax, scale=1.0, fmt="{:.1f}")
    time_path = module_dir / "ik_time_compare.png"
    save_figure(fig, time_path)
    outputs.append(time_path)

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    nn_method = next((item for item in methods if item.get("method_id") == "nn_nr"), None)
    if nn_method is not None:
        timing = nn_method.get("timing_breakdown_ms", {})
        names = ["候选分类", "网络预测", "初值打分", "NR精修", "模型加载与恢复", "未单独计时开销"]
        values = np.asarray(
            [
                float(timing.get("candidate_generation_ms", 0.0)),
                float(timing.get("prediction_ms", 0.0)),
                float(timing.get("initial_pick_scoring_ms", 0.0)),
                float(timing.get("nr_refine_ms", 0.0)),
                float(timing.get("model_load_restore_ms", 0.0)),
                float(timing.get("uninstrumented_overhead_ms", 0.0)),
            ],
            dtype=float,
        )
        color_list = ["#2563EB", "#38BDF8", "#F59E0B", "#16A34A", "#7C3AED", "#94A3B8"]
        ax.bar(names, values, color=color_list)
        ax.set_title("NN + NR：时间细分")
        ax.set_ylabel("时间 (ms)")
        annotate_bar_values(ax, scale=1.0, fmt="{:.1f}")
    detail_path = module_dir / "ik_timing_detail_compare.png"
    save_figure(fig, detail_path)
    outputs.append(detail_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].bar(labels, [float(item.get("final_pos_err_mm", 0.0)) for item in methods], color=colors)
    axes[0].set_title("位置误差对比")
    axes[0].set_ylabel("误差 (mm)")
    annotate_bar_values(axes[0], scale=1.0, fmt="{:.4f}")
    axes[1].bar(labels, [float(item.get("final_ori_err_rad", 0.0)) for item in methods], color=colors)
    axes[1].set_title("姿态误差对比")
    axes[1].set_ylabel("误差 (rad)")
    annotate_bar_values(axes[1], scale=1.0, fmt="{:.5f}")
    accuracy_path = module_dir / "ik_accuracy_compare.png"
    save_figure(fig, accuracy_path)
    outputs.append(accuracy_path)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(labels, [float(item.get("iters", 0.0)) for item in methods], color=colors)
    ax.set_title("普通逆解单解：迭代次数对比")
    ax.set_ylabel("迭代次数")
    annotate_bar_values(ax, scale=1.0, fmt="{:.0f}")
    iters_path = module_dir / "ik_iters_compare.png"
    save_figure(fig, iters_path)
    outputs.append(iters_path)
    return outputs


def plot_obstacle_single(case_dir: Path) -> list[Path]:
    configure_plot_style()
    module_dir = case_dir / OBSTACLE_MODULE
    payload = load_json(module_dir / "raw_result.json")
    methods = payload.get("methods", [])
    labels, colors = method_labels_and_colors(methods)
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 4.6))
    total_secs = [float(item.get("planning_time_ms", 0.0)) / 1000.0 for item in methods]
    ax.bar(labels, total_secs, color=colors)
    ax.set_title("避障单解：总耗时对比")
    ax.set_ylabel("总耗时 (s)")
    annotate_bar_values(ax, scale=1.0, fmt="{:.2f}s")
    total_path = module_dir / "obstacle_total_time_compare.png"
    save_figure(fig, total_path)
    outputs.append(total_path)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    ik_vals = np.asarray([float(item.get("ik_time_ms", 0.0)) for item in methods], dtype=float)
    pure_ik_vals = np.asarray([
        float(item.get("pure_inverse_compute_ms", get_timing_breakdown(item, "pure_inverse_compute_ms", item.get("ik_time_ms", 0.0))))
        for item in methods
    ], dtype=float)
    eval_vals_ms = np.asarray([float(item.get("trajectory_evaluation_time_ms", 0.0)) for item in methods], dtype=float)
    other_vals_ms = np.asarray([
        max(
            0.0,
            float(item.get("planning_time_ms", 0.0)) - float(item.get("ik_time_ms", 0.0)) - float(item.get("trajectory_evaluation_time_ms", 0.0)),
        )
        for item in methods
    ], dtype=float)
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    axes[0].bar(x - width * 0.5, ik_vals, width=width, color=colors, alpha=0.95, label="末端逆解总时间")
    axes[0].bar(x + width * 0.5, pure_ik_vals, width=width, color="#93C5FD", edgecolor="#1D4ED8", label="去除模型加载后的纯求逆时间")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("末端逆解耗时对比")
    axes[0].set_ylabel("时间 (ms)")
    axes[0].legend(frameon=False, fontsize=9)
    annotate_bar_values(axes[0], scale=1.0, fmt="{:.1f}")

    eval_vals_sec = eval_vals_ms / 1000.0
    other_vals_sec = other_vals_ms / 1000.0
    axes[1].bar(labels, eval_vals_sec, label="轨迹评估", color="#F59E0B")
    axes[1].bar(labels, other_vals_sec, bottom=eval_vals_sec, label="其余后处理", color="#94A3B8")
    axes[1].set_title("轨迹评估与后处理耗时")
    axes[1].set_ylabel("时间 (s)")
    axes[1].legend(frameon=False)
    total_stack = eval_vals_sec + other_vals_sec
    ymax = float(np.max(total_stack)) if len(total_stack) else 0.0
    offset = ymax * 0.03 if ymax > 0.0 else 0.05
    axes[1].set_ylim(0.0, max(float(axes[1].get_ylim()[1]), ymax + offset * 3.0))
    for idx, total_val in enumerate(total_stack):
        axes[1].text(
            idx,
            float(total_val) + offset,
            f"{total_val:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#334155",
            clip_on=False,
        )
    detail_path = module_dir / "obstacle_timing_detail_compare.png"
    save_figure(fig, detail_path)
    outputs.append(detail_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].bar(labels, [float(item.get("min_clearance_mm", 0.0)) for item in methods], color=colors)
    axes[0].set_title("最小净空对比")
    axes[0].set_ylabel("净空 (mm)")
    annotate_bar_values(axes[0], scale=1.0, fmt="{:.2f}")
    axes[1].bar(labels, [float(item.get("joint_path_length_deg", 0.0)) for item in methods], color=colors)
    axes[1].set_title("路径长度对比")
    axes[1].set_ylabel("累计角度 (deg)")
    annotate_bar_values(axes[1], scale=1.0, fmt="{:.1f}")
    quality_path = module_dir / "obstacle_quality_compare.png"
    save_figure(fig, quality_path)
    outputs.append(quality_path)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    success_vals = [1.0 if item.get("selected_solution_collision_free") else 0.0 for item in methods]
    ax.bar(labels, success_vals, color=colors)
    ax.set_title("避障单解：是否成功")
    ax.set_ylabel("成功=1, 失败=0")
    ax.set_ylim(0, 1.15)
    for idx, value in enumerate(success_vals):
        ax.text(idx, value + 0.03, "成功" if value > 0.5 else "失败", ha="center", va="bottom", fontsize=10, color="#334155")
    success_path = module_dir / "obstacle_success_compare.png"
    save_figure(fig, success_path)
    outputs.append(success_path)

    target_pose6 = [float(v) for v in payload.get("target_pose6", [0.0] * 6)]
    for method in methods:
        outputs.append(write_motion_trace_csv(module_dir, method, target_pose6=target_pose6))
        outputs.append(plot_motion_pose6(module_dir, method, target_pose6=target_pose6))
    return outputs


def main() -> None:
    args = build_parser().parse_args()
    case_dir, _, _ = build_case_dirs(args.case_root, args.case_tag)

    if args.action == "run":
        if args.module == IK_MODULE:
            raw_path = export_ik_single(case_dir, args)
        else:
            if not args.scene_json.strip():
                raise ValueError("obstacle_single 模式需要提供 --scene_json")
            raw_path = export_obstacle_single(case_dir, args)
        print(raw_path)
        return

    if args.module == IK_MODULE:
        outputs = plot_ik_single(case_dir)
    else:
        outputs = plot_obstacle_single(case_dir)
    print(json.dumps([str(path) for path in outputs], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
