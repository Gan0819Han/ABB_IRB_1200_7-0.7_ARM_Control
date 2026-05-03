#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
UNITY_DIR = Path(r"E:\Software\Unity\Project\ABB_IRB_Demo1")


class ScrollText(ttk.Frame):
    def __init__(self, master: tk.Widget, height: int = 12, font: tuple[str, int] = ("Consolas", 10)) -> None:
        super().__init__(master)
        self.text = tk.Text(self, wrap="word", height=height, font=font)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=bar.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        bar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def set_text(self, msg: str) -> None:
        self.text.delete("1.0", "end")
        self.text.insert("end", msg)
        self.text.see("1.0")

    def append(self, msg: str) -> None:
        self.text.insert("end", msg)
        self.text.see("end")

    def clear(self) -> None:
        self.text.delete("1.0", "end")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ABB_IRB Control GUI")
        self.geometry("1380x860")
        self.minsize(1200, 780)

        self.status_var = tk.StringVar(value="Ready")

        self.pose_var = tk.StringVar(value="100,200,800,0.1,-0.2,0.3")
        self.pred_meta_var = tk.StringVar(value=str(ROOT / "artifacts" / "prediction_system_formal" / "metadata.json"))
        self.branch_meta_var = tk.StringVar(value=str(ROOT / "artifacts" / "branch_classification_system" / "metadata.json"))
        self.fine_meta_var = tk.StringVar(value=str(ROOT / "artifacts" / "fine_classification_system" / "metadata.json"))
        self.predict_out_json_var = tk.StringVar(value=str(ROOT / "artifacts" / "gui_outputs" / "predict_result.json"))

        self.obstacle_pose_var = tk.StringVar(value="100,200,800,0.1,-0.2,0.3")
        self.q_start_var = tk.StringVar(value="0,0,0,0,0,0")
        self.scene_var = tk.StringVar(value=str(ROOT / "data" / "obstacles" / "open_space_reselect_demo.json"))
        self.obstacle_out_json_var = tk.StringVar(value=str(ROOT / "artifacts" / "obstacle_avoidance" / "gui_plan.json"))

        self.fk_q_var = tk.StringVar(value="20,30,-40,10,20,0")
        self.fk_out_json_var = tk.StringVar(value=str(UNITY_DIR / "Assets" / "ReferenceData" / "gui_fk_reference.json"))

        self.traj_q_start_var = tk.StringVar(value="0,0,0,0,0,0")
        self.traj_q_goal_var = tk.StringVar(value="20,30,-40,10,20,0")
        self.traj_steps_var = tk.StringVar(value="120")
        self.traj_duration_var = tk.StringVar(value="3.0")
        self.traj_name_var = tk.StringVar(value="abb_gui_demo_traj")
        self.traj_out_json_var = tk.StringVar(value=str(UNITY_DIR / "Assets" / "TrajectoryData" / "abb_gui_demo_traj.json"))

        self.obstacle_plan_json_var = tk.StringVar(value=str(ROOT / "artifacts" / "obstacle_avoidance" / "gui_plan.json"))
        self.obstacle_demo_name_var = tk.StringVar(value="gui_obstacle_demo")
        self.obstacle_unity_out_var = tk.StringVar(value=str(UNITY_DIR / "Assets" / "PlanningData" / "gui_obstacle_demo_unity.json"))

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=8)
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="工程路径").grid(row=0, column=0, sticky="w")
        ttk.Label(top, text=str(ROOT)).grid(row=0, column=1, sticky="w")
        ttk.Label(top, textvariable=self.status_var, foreground="#0B5394").grid(row=0, column=2, sticky="e")

        left = ttk.Frame(self, padding=8)
        left.grid(row=1, column=0, sticky="nsw")
        self.notebook = ttk.Notebook(left)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self._build_predict_tab()
        self._build_obstacle_tab()
        self._build_unity_tab()
        self._build_figure_tab()

        right = ttk.Frame(self, padding=8)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.rowconfigure(3, weight=2)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="结果总结").grid(row=0, column=0, sticky="w")
        self.summary = ScrollText(right, height=12, font=("Consolas", 10))
        self.summary.grid(row=1, column=0, sticky="nsew", pady=(0, 8))

        ttk.Label(right, text="原始日志").grid(row=2, column=0, sticky="w")
        self.log = ScrollText(right, height=24, font=("Consolas", 10))
        self.log.grid(row=3, column=0, sticky="nsew")

        bottom = ttk.Frame(self, padding=8)
        bottom.grid(row=2, column=0, columnspan=2, sticky="ew")
        for i in range(8):
            bottom.columnconfigure(i, weight=1)
        ttk.Button(bottom, text="运行逆解", command=self.run_predict).grid(row=0, column=0, sticky="ew", padx=4)
        ttk.Button(bottom, text="运行避障", command=self.run_obstacle).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(bottom, text="导出FK参考", command=self.run_fk_export).grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(bottom, text="导出轨迹", command=self.run_trajectory_export).grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Button(bottom, text="导出避障回放", command=self.run_obstacle_unity_export).grid(row=0, column=4, sticky="ew", padx=4)
        ttk.Button(bottom, text="打开结果目录", command=self.open_result_dir).grid(row=0, column=5, sticky="ew", padx=4)
        ttk.Button(bottom, text="打开Unity目录", command=self.open_unity_dir).grid(row=0, column=6, sticky="ew", padx=4)
        ttk.Button(bottom, text="清空输出", command=self.clear_outputs).grid(row=0, column=7, sticky="ew", padx=4)

    def _make_labeled_entry(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=2)

    def _make_labeled_path_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        var: tk.StringVar,
        *,
        save: bool,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Button(
            parent,
            text="选择...",
            command=lambda: self._browse_json_path(var, save=save),
            width=10,
        ).grid(row=row, column=2, sticky="ew", padx=(6, 0), pady=2)

    def _browse_json_path(self, var: tk.StringVar, *, save: bool) -> None:
        initial = Path(var.get()) if var.get().strip() else ROOT
        initial_dir = initial.parent if initial.suffix else initial
        dialog_kwargs = {
            "title": "选择 JSON 文件" if not save else "选择输出 JSON 路径",
            "initialdir": str(initial_dir if initial_dir.exists() else ROOT),
            "initialfile": initial.name if initial.suffix else "",
            "filetypes": [("JSON files", "*.json"), ("All files", "*.*")],
        }
        if save:
            selected = filedialog.asksaveasfilename(**dialog_kwargs, defaultextension=".json")
        else:
            selected = filedialog.askopenfilename(**dialog_kwargs)
        if selected:
            var.set(selected)

    def _make_note(self, parent: ttk.Frame, row: int, text: str, *, columnspan: int = 3) -> None:
        ttk.Label(
            parent,
            text=text,
            foreground="#666666",
            wraplength=860,
            justify="left",
        ).grid(row=row, column=0, columnspan=columnspan, sticky="w", pady=(4, 0))

    def _build_predict_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="推理")
        tab.columnconfigure(1, weight=1)
        self._make_labeled_entry(tab, 0, "目标位姿 pose6", self.pose_var)
        self._make_labeled_path_entry(tab, 1, "prediction metadata", self.pred_meta_var, save=False)
        self._make_labeled_path_entry(tab, 2, "branch metadata", self.branch_meta_var, save=False)
        self._make_labeled_path_entry(tab, 3, "fine metadata", self.fine_meta_var, save=False)
        self._make_labeled_path_entry(tab, 4, "输出 JSON", self.predict_out_json_var, save=True)
        self._make_note(tab, 5, "说明：本页只做逆解推理，q_start 不参与 predict_ik。输出结果会保存为完整 JSON，右侧同时给出摘要。")
        ttk.Button(tab, text="运行 predict_ik", command=self.run_predict).grid(row=6, column=0, columnspan=3, sticky="ew", pady=(12, 0))

    def _build_obstacle_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="避障")
        tab.columnconfigure(1, weight=1)
        self._make_labeled_entry(tab, 0, "目标位姿 pose6", self.obstacle_pose_var)
        self._make_labeled_entry(tab, 1, "起始关节 q_start", self.q_start_var)
        self._make_labeled_path_entry(tab, 2, "scene_json", self.scene_var, save=False)
        self._make_labeled_path_entry(tab, 3, "prediction metadata", self.pred_meta_var, save=False)
        self._make_labeled_path_entry(tab, 4, "branch metadata", self.branch_meta_var, save=False)
        self._make_labeled_path_entry(tab, 5, "fine metadata", self.fine_meta_var, save=False)
        self._make_labeled_path_entry(tab, 6, "输出 JSON", self.obstacle_out_json_var, save=True)
        self._make_note(tab, 7, "说明：本页会执行候选逆解评估、轨迹碰撞检测与自动换解。q_start 会参与整条轨迹的碰撞分析。")
        ttk.Button(tab, text="运行 plan_collision_free_ik", command=self.run_obstacle).grid(row=8, column=0, columnspan=3, sticky="ew", pady=(12, 0))

    def _build_unity_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Unity")
        tab.columnconfigure(1, weight=1)

        fk_group = ttk.LabelFrame(tab, text="FK参考导出", padding=8)
        fk_group.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        fk_group.columnconfigure(1, weight=1)
        self._make_labeled_entry(fk_group, 0, "关节角 q", self.fk_q_var)
        self._make_labeled_path_entry(fk_group, 1, "输出 JSON", self.fk_out_json_var, save=True)
        self._make_note(fk_group, 2, "作用：把单组关节角对应的 FK 末端位置、姿态和关节点数据导出给 Unity，用于做单姿态校验。")
        ttk.Button(fk_group, text="导出 FK 参考", command=self.run_fk_export).grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        traj_group = ttk.LabelFrame(tab, text="轨迹导出", padding=8)
        traj_group.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        traj_group.columnconfigure(1, weight=1)
        self._make_labeled_entry(traj_group, 0, "q_start", self.traj_q_start_var)
        self._make_labeled_entry(traj_group, 1, "q_goal", self.traj_q_goal_var)
        self._make_labeled_entry(traj_group, 2, "steps", self.traj_steps_var)
        self._make_labeled_entry(traj_group, 3, "duration", self.traj_duration_var)
        self._make_labeled_entry(traj_group, 4, "name", self.traj_name_var)
        self._make_labeled_path_entry(traj_group, 5, "输出 JSON", self.traj_out_json_var, save=True)
        self._make_note(traj_group, 6, "作用：将 q_start 到 q_goal 的关节空间插值轨迹导出为 Unity 可直接播放的 JSON。")
        ttk.Button(traj_group, text="导出轨迹 JSON", command=self.run_trajectory_export).grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        obs_group = ttk.LabelFrame(tab, text="避障结果转 Unity 回放", padding=8)
        obs_group.grid(row=2, column=0, columnspan=2, sticky="ew")
        obs_group.columnconfigure(1, weight=1)
        self._make_labeled_path_entry(obs_group, 0, "plan_json", self.obstacle_plan_json_var, save=False)
        self._make_labeled_entry(obs_group, 1, "demo_name", self.obstacle_demo_name_var)
        self._make_labeled_path_entry(obs_group, 2, "输出 JSON", self.obstacle_unity_out_var, save=True)
        self._make_note(obs_group, 3, "作用：把 Python 侧的避障规划结果整理成 Unity 友好版 JSON，供障碍物、目标点、蓝/红轨迹回放使用。")
        ttk.Button(obs_group, text="导出避障回放 JSON", command=self.run_obstacle_unity_export).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))

    def _build_figure_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="图表")
        tab.columnconfigure(0, weight=1)

        core_group = ttk.LabelFrame(tab, text="1. 核心图表", padding=8)
        core_group.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        core_group.columnconfigure(0, weight=1)
        ttk.Label(
            core_group,
            text=(
                "作用：汇总当前工程的核心结果图，包括 FK 偏置验证、子空间划分对比、子空间预测误差、"
                "分类器表现、benchmark 总结等。适合论文主体性能章节使用。"
            ),
            foreground="#666666",
            wraplength=860,
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            core_group,
            text="主要输出目录：figure/figures/ 与 figure/data/",
            foreground="#666666",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(core_group, text="生成核心图表", command=self.run_core_figures).grid(row=2, column=0, sticky="ew", pady=(8, 0))

        workspace_group = ttk.LabelFrame(tab, text="2. 工作空间图", padding=8)
        workspace_group.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        workspace_group.columnconfigure(0, weight=1)
        ttk.Label(
            workspace_group,
            text=(
                "作用：读取保存的子空间参考样本，生成三视图投影和三维样本可达空间图。"
                "适合说明 ABB_IRB 的样本覆盖范围与工作空间分布。"
            ),
            foreground="#666666",
            wraplength=860,
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            workspace_group,
            text="依赖：data/subspace_reference_abb_strict_samples512_seed2026/ 下的参考样本。",
            foreground="#666666",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(workspace_group, text="生成工作空间图", command=self.run_workspace_figures).grid(row=2, column=0, sticky="ew", pady=(8, 0))

        obstacle_group = ttk.LabelFrame(tab, text="3. 避障图", padding=8)
        obstacle_group.grid(row=2, column=0, sticky="ew")
        obstacle_group.columnconfigure(0, weight=1)
        ttk.Label(
            obstacle_group,
            text=(
                "作用：基于固定障碍物规划结果绘制候选轨迹、碰撞/无碰撞对比和论文风格避障示意图。"
                "适合展示为什么系统需要进行候选重选。"
            ),
            foreground="#666666",
            wraplength=860,
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            obstacle_group,
            text="依赖：artifacts/obstacle_avoidance/open_space_reselect_demo_plan.json。",
            foreground="#666666",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(obstacle_group, text="生成避障图", command=self.run_obstacle_figures).grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def clear_outputs(self) -> None:
        self.summary.clear()
        self.log.clear()
        self.status_var.set("Ready")

    def append_log(self, text: str) -> None:
        self.log.append(text + "\n")
        self.update_idletasks()

    def set_summary(self, text: str) -> None:
        self.summary.set_text(text)
        self.update_idletasks()

    def _ensure_parent(self, path_text: str) -> None:
        path = Path(path_text)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    def _format_q_deg(self, values: list[float]) -> str:
        return ",".join(f"{float(v):.6f}".rstrip("0").rstrip(".") for v in values)

    def _extract_goal_q_from_json(self, json_path: Path) -> tuple[list[float] | None, str]:
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

    def _sync_q_targets_from_result(self, json_path: Path) -> str:
        q_deg, source = self._extract_goal_q_from_json(json_path)
        if not q_deg:
            return f"未同步 Unity 关节目标：{source}"
        text = self._format_q_deg(q_deg)
        self.traj_q_goal_var.set(text)
        self.fk_q_var.set(text)
        return (
            f"已同步到 Unity 页 q_goal：{text}\n"
            f"已同步到 FK参考导出 q：{text}\n"
            f"{source}"
        )

    def _build_predict_summary(self, json_path: Path) -> str:
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
        lines.append(self._sync_q_targets_from_result(json_path))
        lines.append("")
        lines.append(f"结果文件：{json_path}")
        return "\n".join(lines)

    def _build_obstacle_summary(self, json_path: Path) -> str:
        if not json_path.exists():
            return f"未找到输出文件：{json_path}"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        selected = payload.get("selected_solution", {})
        traj = selected.get("trajectory_summary", {})
        lines = ["任务：plan_collision_free_ik", ""]
        lines.append(f"候选数量：{len(payload.get('evaluated_candidates', []))}")
        lines.append(f"选中子空间：{selected.get('subspace_id', '-')}")
        lines.append(f"NR 收敛：{selected.get('nr_converged', '-')}")
        lines.append(f"NR 迭代次数：{selected.get('nr_iters', '-')}")
        lines.append(f"最终位置误差(mm)：{selected.get('final_pos_err_mm', '-')}")
        lines.append(f"最终姿态误差(rad)：{selected.get('final_ori_err_rad', '-')}")
        lines.append(f"是否碰撞：{traj.get('collision', '-')}")
        lines.append(f"碰撞帧数：{traj.get('collision_frame_count', '-')}")
        lines.append(f"最小净空(mm)：{traj.get('min_clearance_mm', '-')}")
        lines.append(f"轨迹步数：{traj.get('trajectory_steps', '-')}")
        lines.append("")
        lines.append(self._sync_q_targets_from_result(json_path))
        lines.append("")
        lines.append(f"结果文件：{json_path}")
        return "\n".join(lines)

    def _build_fk_summary(self, json_path: Path) -> str:
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

    def _build_traj_summary(self, json_path: Path) -> str:
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

    def _build_obstacle_unity_summary(self, json_path: Path) -> str:
        if not json_path.exists():
            return f"未找到输出文件：{json_path}"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        selected = payload.get("selected_solution", {})
        compare = payload.get("comparison_collision_solution")
        lines = ["任务：export_unity_obstacle_avoidance_demo", ""]
        lines.append(f"demo_name：{payload.get('demo_name', '-')}")
        lines.append(f"scene_name：{payload.get('scene_name', '-')}")
        lines.append(f"障碍物数量：{len(payload.get('obstacles', []))}")
        lines.append(f"无碰撞轨迹子空间：{selected.get('subspace_id', '-')}")
        lines.append(f"无碰撞轨迹步数：{selected.get('trajectory_steps', '-')}")
        if compare:
            lines.append(f"对比碰撞轨迹子空间：{compare.get('subspace_id', '-')}")
            lines.append(f"对比碰撞帧数：{compare.get('collision_frame_count', '-')}")
        lines.append("")
        lines.append(f"结果文件：{json_path}")
        return "\n".join(lines)

    def run_command(self, title: str, cmd: list[str], summary_builder=None, summary_path: Path | None = None) -> None:
        def worker() -> None:
            self.status_var.set(title)
            self.append_log(f"[{title}]")
            self.append_log("CMD: " + " ".join(cmd))
            self.set_summary(f"任务：{title}\n\n运行中...")
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.append_log(line.rstrip())
                code = proc.wait()
                self.append_log(f"[exit code] {code}")
                if code == 0 and summary_builder is not None and summary_path is not None:
                    self.set_summary(summary_builder(summary_path))
                elif code == 0:
                    self.set_summary(f"任务：{title}\n\n执行成功。")
                else:
                    self.set_summary(f"任务：{title}\n\n执行失败，退出码：{code}\n请查看原始日志。")
            except Exception as exc:
                self.append_log(f"[error] {exc}")
                self.set_summary(f"任务：{title}\n\n发生异常：{exc}")
            finally:
                self.status_var.set("Ready")

        threading.Thread(target=worker, daemon=True).start()

    def run_predict(self) -> None:
        out_path = Path(self.predict_out_json_var.get())
        self._ensure_parent(str(out_path))
        cmd = [
            PYTHON, "-X", "utf8", "predict_ik.py",
            "--candidate_mode", "hierarchical",
            f"--pose={self.pose_var.get()}",
            "--pred_meta", self.pred_meta_var.get(),
            "--branch_meta", self.branch_meta_var.get(),
            "--fine_meta", self.fine_meta_var.get(),
            "--topk_shoulder", "2",
            "--topk_elbow", "1",
            "--topk_wrist", "2",
            "--max_branch_candidates", "4",
            "--fine_topk_per_branch", "3",
            "--max_subspace_candidates", "15",
            "--enable_nr",
            "--out_json", str(out_path),
        ]
        self.run_command("predict_ik", cmd, self._build_predict_summary, out_path)

    def run_obstacle(self) -> None:
        out_path = Path(self.obstacle_out_json_var.get())
        self._ensure_parent(str(out_path))
        self.obstacle_plan_json_var.set(str(out_path))
        cmd = [
            PYTHON, "-X", "utf8", "scripts\\plan_collision_free_ik.py",
            f"--pose={self.obstacle_pose_var.get()}",
            f"--q_start={self.q_start_var.get()}",
            "--scene_json", self.scene_var.get(),
            "--pred_meta", self.pred_meta_var.get(),
            "--branch_meta", self.branch_meta_var.get(),
            "--fine_meta", self.fine_meta_var.get(),
            "--topk_shoulder", "2",
            "--topk_elbow", "1",
            "--topk_wrist", "2",
            "--max_branch_candidates", "6",
            "--fine_topk_per_branch", "3",
            "--max_subspace_candidates", "18",
            "--max_evaluated_candidates", "18",
            "--trajectory_steps", "120",
            "--save_selected_frames",
            "--out_json", str(out_path),
        ]
        self.run_command("plan_collision_free_ik", cmd, self._build_obstacle_summary, out_path)

    def run_fk_export(self) -> None:
        out_path = Path(self.fk_out_json_var.get())
        self._ensure_parent(str(out_path))
        cmd = [
            PYTHON,
            "-X",
            "utf8",
            "scripts\\export_unity_fk_reference.py",
            f"--q={self.fk_q_var.get()}",
            "--out_json",
            str(out_path),
        ]
        self.run_command("export_unity_fk_reference", cmd, self._build_fk_summary, out_path)

    def run_trajectory_export(self) -> None:
        out_path = Path(self.traj_out_json_var.get())
        self._ensure_parent(str(out_path))
        cmd = [
            PYTHON, "-X", "utf8", "scripts\\export_unity_trajectory.py",
            f"--q_start={self.traj_q_start_var.get()}",
            f"--q_goal={self.traj_q_goal_var.get()}",
            "--steps", self.traj_steps_var.get(),
            "--duration", self.traj_duration_var.get(),
            "--name", self.traj_name_var.get(),
            "--out_json", str(out_path),
        ]
        self.run_command("export_unity_trajectory", cmd, self._build_traj_summary, out_path)

    def run_obstacle_unity_export(self) -> None:
        plan_path = Path(self.obstacle_plan_json_var.get())
        out_path = Path(self.obstacle_unity_out_var.get())
        self._ensure_parent(str(out_path))
        cmd = [
            PYTHON, "-X", "utf8", "scripts\\export_unity_obstacle_avoidance_demo.py",
            "--plan_json", str(plan_path),
            "--demo_name", self.obstacle_demo_name_var.get(),
            "--out_json", str(out_path),
        ]
        self.run_command("export_unity_obstacle_avoidance_demo", cmd, self._build_obstacle_unity_summary, out_path)

    def run_core_figures(self) -> None:
        self.run_command("generate_core_figures", [PYTHON, "-X", "utf8", "figure\\scripts\\generate_core_figures.py"])

    def run_workspace_figures(self) -> None:
        self.run_command("generate_workspace_figures", [PYTHON, "-X", "utf8", "figure\\scripts\\generate_workspace_figures.py"])

    def run_obstacle_figures(self) -> None:
        self.run_command("generate_obstacle_figures", [PYTHON, "-X", "utf8", "figure\\scripts\\generate_obstacle_candidate_trajectory_figures.py"])

    def open_result_dir(self) -> None:
        subprocess.Popen(["explorer", str(ROOT / "artifacts")])

    def open_unity_dir(self) -> None:
        subprocess.Popen(["explorer", str(UNITY_DIR)])


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
