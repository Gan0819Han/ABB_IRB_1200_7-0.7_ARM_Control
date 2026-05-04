#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_config import JOINT_LIMITS_DEG


PYTHON = sys.executable
UNITY_DIR = Path(r"E:\Software\Unity\Project\ABB_IRB_Demo1")
GUI_DEFAULTS_PATH = ROOT / "gui" / "gui_defaults.json"
DEFAULT_OBSTACLE_NAME = "demo_box_1"
DEFAULT_OBSTACLE_CENTER_MM = [221.7, 274.53, 493.57]
DEFAULT_OBSTACLE_SIZE_MM = [127.62, 90.45, 175.06]
NEW_OBSTACLE_CENTER_MM = [360.0, -180.0, 540.0]
NEW_OBSTACLE_SIZE_MM = [110.0, 90.0, 160.0]


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


class ScrollableForm(ttk.Frame):
    def __init__(self, master: tk.Widget) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.canvas.bind("<Configure>", self._sync_inner_width)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    def _sync_inner_width(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self.winfo_exists() and self.canvas.winfo_exists():
            try:
                widget_under_pointer = self.winfo_containing(event.x_root, event.y_root)
            except Exception:
                widget_under_pointer = None
            if widget_under_pointer is None:
                return
            parent = widget_under_pointer
            while parent is not None:
                if parent == self:
                    delta = -1 if event.delta > 0 else 1
                    self.canvas.yview_scroll(delta, "units")
                    break
                parent = parent.master


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ABB_IRB Control GUI")
        self.geometry("1380x860")
        self.minsize(1200, 780)

        self.path_defaults = self._load_path_defaults()
        self.status_var = tk.StringVar(value="Ready")

        self.pose_var = tk.StringVar(value="100,200,800,0.1,-0.2,0.3")
        self.pred_meta_var = tk.StringVar(value=self._default_path_value("pred_meta", ROOT / "artifacts" / "prediction_system_formal" / "metadata.json"))
        self.branch_meta_var = tk.StringVar(value=self._default_path_value("branch_meta", ROOT / "artifacts" / "branch_classification_system" / "metadata.json"))
        self.fine_meta_var = tk.StringVar(value=self._default_path_value("fine_meta", ROOT / "artifacts" / "fine_classification_system" / "metadata.json"))
        self.predict_out_json_var = tk.StringVar(value=self._default_path_value("predict_out_json", ROOT / "artifacts" / "gui_outputs" / "predict_result.json"))
        self.predict_topk_shoulder_var = tk.StringVar(value="2")
        self.predict_topk_elbow_var = tk.StringVar(value="1")
        self.predict_topk_wrist_var = tk.StringVar(value="2")
        self.predict_max_branch_candidates_var = tk.StringVar(value="4")
        self.predict_fine_topk_per_branch_var = tk.StringVar(value="3")
        self.predict_max_subspace_candidates_var = tk.StringVar(value="15")
        self.predict_enable_nr_var = tk.BooleanVar(value=True)
        self.predict_nr_max_iters_var = tk.StringVar(value="40")
        self.predict_nr_tol_pos_mm_var = tk.StringVar(value="1e-3")
        self.predict_nr_tol_ori_rad_var = tk.StringVar(value="1e-3")
        self.predict_nr_damping_var = tk.StringVar(value="1e-5")
        self.predict_nr_step_scale_var = tk.StringVar(value="1.0")
        self.pose_part_vars = self._create_component_vars(self.pose_var, 6)

        self.obstacle_pose_var = self.pose_var
        self.q_start_var = tk.StringVar(value="0,0,0,0,0,0")
        self.q_start_part_vars = self._create_component_vars(self.q_start_var, 6)
        self.scene_var = tk.StringVar(value=self._default_path_value("scene_json", ROOT / "data" / "obstacles" / "open_space_reselect_demo.json"))
        self.obstacle_out_json_var = tk.StringVar(value=self._default_path_value("obstacle_out_json", ROOT / "artifacts" / "obstacle_avoidance" / "gui_plan.json"))
        self.obstacle_name_var = tk.StringVar(value=DEFAULT_OBSTACLE_NAME)
        self.obstacle_center_var = tk.StringVar(value=self._format_q_deg(DEFAULT_OBSTACLE_CENTER_MM))
        self.obstacle_size_var = tk.StringVar(value=self._format_q_deg(DEFAULT_OBSTACLE_SIZE_MM))
        self.obstacle_center_part_vars = self._create_component_vars(self.obstacle_center_var, 3)
        self.obstacle_size_part_vars = self._create_component_vars(self.obstacle_size_var, 3)
        self.obstacle_selector_var = tk.StringVar(value="")
        self.obstacle_selector_values: list[str] = []
        self.current_obstacle_index = 0
        self.obstacle_topk_shoulder_var = tk.StringVar(value="2")
        self.obstacle_topk_elbow_var = tk.StringVar(value="1")
        self.obstacle_topk_wrist_var = tk.StringVar(value="2")
        self.obstacle_max_branch_candidates_var = tk.StringVar(value="6")
        self.obstacle_fine_topk_per_branch_var = tk.StringVar(value="3")
        self.obstacle_max_subspace_candidates_var = tk.StringVar(value="18")
        self.obstacle_max_evaluated_candidates_var = tk.StringVar(value="18")
        self.obstacle_nr_max_iters_var = tk.StringVar(value="40")
        self.obstacle_nr_tol_pos_mm_var = tk.StringVar(value="1e-3")
        self.obstacle_nr_tol_ori_rad_var = tk.StringVar(value="1e-3")
        self.obstacle_nr_damping_var = tk.StringVar(value="1e-5")
        self.obstacle_nr_step_scale_var = tk.StringVar(value="1.0")
        self.obstacle_trajectory_steps_var = tk.StringVar(value="120")
        self.obstacle_dedupe_tol_deg_var = tk.StringVar(value="0.5")

        self.fk_q_var = tk.StringVar(value="20,30,-40,10,20,0")
        self.fk_out_json_var = tk.StringVar(value=self._default_path_value("fk_out_json", UNITY_DIR / "Assets" / "ReferenceData" / "gui_fk_reference.json"))

        self.traj_q_start_var = tk.StringVar(value="0,0,0,0,0,0")
        self.traj_q_goal_var = tk.StringVar(value="20,30,-40,10,20,0")
        self.traj_steps_var = tk.StringVar(value="120")
        self.traj_duration_var = tk.StringVar(value="3.0")
        self.traj_name_var = tk.StringVar(value="abb_gui_demo_traj")
        self.traj_out_json_var = tk.StringVar(value=self._default_path_value("traj_out_json", UNITY_DIR / "Assets" / "TrajectoryData" / "abb_gui_demo_traj.json"))

        self.obstacle_plan_json_var = tk.StringVar(value=self._default_path_value("obstacle_plan_json", ROOT / "artifacts" / "obstacle_avoidance" / "gui_plan.json"))
        self.obstacle_demo_name_var = tk.StringVar(value="gui_obstacle_demo")
        self.obstacle_unity_out_var = tk.StringVar(value=self._default_path_value("obstacle_unity_out_json", UNITY_DIR / "Assets" / "PlanningData" / "gui_obstacle_demo_unity.json"))

        self.figure_output_dir_var = tk.StringVar(value=self._default_path_value("figure_output_dir", ROOT / "figure" / "figures"))
        self.figure_data_dir_var = tk.StringVar(value=self._default_path_value("figure_data_dir", ROOT / "figure" / "data"))
        self.figure_core_case_json_var = tk.StringVar(value=self._default_path_value("figure_core_case_json", ROOT / "artifacts" / "gui_outputs" / "predict_result.json"))
        self.figure_workspace_ref_dir_var = tk.StringVar(value=self._default_path_value("figure_workspace_ref_dir", ROOT / "data" / "subspace_reference_abb_strict_samples512_seed2026"))
        self.figure_obstacle_plan_json_var = tk.StringVar(value=self._default_path_value("figure_obstacle_plan_json", ROOT / "artifacts" / "obstacle_avoidance" / "gui_plan.json"))
        self.figure_preview_status_var = tk.StringVar(value="尚未生成避障图预览。")
        self.figure_preview_path_var = tk.StringVar(value="")
        self.obstacle_preview_photo: tk.PhotoImage | None = None

        self._build_ui()
        self._try_load_obstacle_editor_from_scene()
        self.refresh_obstacle_figure_preview()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="工程路径").grid(row=0, column=0, sticky="w")
        ttk.Label(top, text=str(ROOT)).grid(row=0, column=1, sticky="w")
        ttk.Label(top, textvariable=self.status_var, foreground="#0B5394").grid(row=0, column=2, sticky="e")

        main_pane = ttk.Panedwindow(self, orient="horizontal")
        main_pane.grid(row=1, column=0, sticky="nsew")

        left = ttk.Frame(main_pane, padding=8)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.notebook = ttk.Notebook(left)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self._build_predict_tab()
        self._build_obstacle_tab()
        self._build_unity_tab()
        self._build_figure_tab()

        right = ttk.Frame(main_pane, padding=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        output_pane = ttk.Panedwindow(right, orient="vertical")
        output_pane.grid(row=0, column=0, sticky="nsew")

        summary_group = ttk.LabelFrame(output_pane, text="结果总结", padding=6)
        summary_group.columnconfigure(0, weight=1)
        summary_group.rowconfigure(0, weight=1)
        self.summary = ScrollText(summary_group, height=12, font=("Consolas", 10))
        self.summary.grid(row=0, column=0, sticky="nsew")

        log_group = ttk.LabelFrame(output_pane, text="原始日志", padding=6)
        log_group.columnconfigure(0, weight=1)
        log_group.rowconfigure(0, weight=1)
        self.log = ScrollText(log_group, height=24, font=("Consolas", 10))
        self.log.grid(row=0, column=0, sticky="nsew")

        output_pane.add(summary_group, weight=1)
        output_pane.add(log_group, weight=2)

        main_pane.add(left, weight=5)
        main_pane.add(right, weight=2)

        bottom = ttk.Frame(self, padding=8)
        bottom.grid(row=2, column=0, sticky="ew")
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

    def _create_component_vars(self, source_var: tk.StringVar, count: int) -> list[tk.StringVar]:
        part_vars = [tk.StringVar() for _ in range(count)]
        state = {"syncing": False}

        def sync_parts_from_source(*_args: object) -> None:
            if state["syncing"]:
                return
            state["syncing"] = True
            try:
                raw = [item.strip() for item in source_var.get().split(",")]
                values = raw[:count] + [""] * max(0, count - len(raw))
                for var, value in zip(part_vars, values):
                    if var.get() != value:
                        var.set(value)
            finally:
                state["syncing"] = False

        def sync_source_from_parts(*_args: object) -> None:
            if state["syncing"]:
                return
            state["syncing"] = True
            try:
                source_var.set(",".join(var.get().strip() for var in part_vars))
            finally:
                state["syncing"] = False

        source_var.trace_add("write", sync_parts_from_source)
        for var in part_vars:
            var.trace_add("write", sync_source_from_parts)
        sync_parts_from_source()
        return part_vars

    def _make_vector_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        part_labels: list[str],
        part_vars: list[tk.StringVar],
        *,
        columns: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="nw", pady=2)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        for col in range(columns):
            frame.columnconfigure(col, weight=1)

        for idx, (part_label, part_var) in enumerate(zip(part_labels, part_vars)):
            item = ttk.Frame(frame)
            item.grid(
                row=idx // columns,
                column=idx % columns,
                sticky="ew",
                padx=(0, 6) if (idx % columns) != columns - 1 else (0, 0),
                pady=(0, 4),
            )
            item.columnconfigure(0, weight=1)
            ttk.Label(item, text=part_label).grid(row=0, column=0, sticky="w")
            ttk.Entry(item, textvariable=part_var, width=12).grid(row=1, column=0, sticky="ew")

    def _make_labeled_path_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        var: tk.StringVar,
        *,
        save: bool,
        default_key: str | None = None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Button(
            parent,
            text="选择...",
            command=lambda: self._browse_json_path(var, save=save),
            width=10,
        ).grid(row=row, column=2, sticky="ew", padx=(6, 0), pady=2)
        if default_key is not None:
            ttk.Button(
                parent,
                text="设为默认",
                command=lambda: self._save_default_path(default_key, var, label),
                width=10,
            ).grid(row=row, column=3, sticky="ew", padx=(6, 0), pady=2)

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

    def _make_labeled_dir_entry(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, *, default_key: str | None = None) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Button(
            parent,
            text="选择目录...",
            command=lambda: self._browse_directory(var),
            width=12,
        ).grid(row=row, column=2, sticky="ew", padx=(6, 0), pady=2)
        if default_key is not None:
            ttk.Button(
                parent,
                text="设为默认",
                command=lambda: self._save_default_path(default_key, var, label),
                width=10,
            ).grid(row=row, column=3, sticky="ew", padx=(6, 0), pady=2)

    def _browse_directory(self, var: tk.StringVar) -> None:
        initial = Path(var.get()) if var.get().strip() else ROOT
        initial_dir = initial if initial.exists() else ROOT
        selected = filedialog.askdirectory(
            title="选择输出目录",
            initialdir=str(initial_dir),
            mustexist=False,
        )
        if selected:
            var.set(selected)

    def _make_note(self, parent: ttk.Frame, row: int, text: str, *, columnspan: int = 3) -> None:
        note = tk.Label(
            parent,
            text=text,
            fg="#666666",
            justify="left",
            anchor="w",
        )
        note.grid(row=row, column=0, columnspan=columnspan, sticky="ew", pady=(4, 0))

        def _refresh_wrap(event: tk.Event) -> None:
            note.configure(wraplength=max(220, event.width - 20))

        parent.bind("<Configure>", _refresh_wrap, add="+")

    def _joint_limits_note_text(self) -> str:
        labels = ["q1", "q2", "q3", "q4", "q5", "q6"]
        ranges = [
            f"{name}: [{limits[0]:.0f}, {limits[1]:.0f}] deg"
            for name, limits in zip(labels, JOINT_LIMITS_DEG)
        ]
        return "范围说明：" + "； ".join(ranges)

    def _load_path_defaults(self) -> dict[str, str]:
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

    def _default_path_value(self, key: str, fallback: Path) -> str:
        return self.path_defaults.get(key, str(fallback))

    def _write_path_defaults(self) -> None:
        GUI_DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {"path_defaults": self.path_defaults}
        GUI_DEFAULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_default_path(self, key: str, var: tk.StringVar, label: str) -> None:
        value = var.get().strip()
        if not value:
            self.set_summary(f"任务：默认路径设置\n\n{label} 当前为空，不能设为默认。")
            self.append_log(f"[default path] {label} 为空，未保存默认值。")
            return
        self.path_defaults[key] = value
        self._write_path_defaults()
        self.set_summary(f"任务：默认路径设置\n\n已将 {label} 设为默认。\n\n默认值：{value}\n\n配置文件：{GUI_DEFAULTS_PATH}")
        self.append_log(f"[default path] {label} -> {value}")

    def _build_predict_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="推理")
        tab.columnconfigure(1, weight=1)
        self._make_vector_entry(tab, 0, "目标位姿 pose6", ["x", "y", "z", "phi", "theta", "psi"], self.pose_part_vars, columns=3)
        self._make_note(tab, 1, "范围说明：x/y/z 单位 mm，需位于机械臂可达空间；phi/theta/psi 单位 rad，通常建议填写在 [-3.1416, 3.1416]。")
        self._make_labeled_path_entry(tab, 2, "prediction metadata", self.pred_meta_var, save=False, default_key="pred_meta")
        self._make_labeled_path_entry(tab, 3, "branch metadata", self.branch_meta_var, save=False, default_key="branch_meta")
        self._make_labeled_path_entry(tab, 4, "fine metadata", self.fine_meta_var, save=False, default_key="fine_meta")
        self._make_labeled_path_entry(tab, 5, "输出 JSON", self.predict_out_json_var, save=True, default_key="predict_out_json")

        hyper_group = ttk.LabelFrame(tab, text="推理超参数", padding=8)
        hyper_group.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        hyper_group.columnconfigure(1, weight=1)
        hyper_group.columnconfigure(3, weight=1)
        self._make_labeled_entry(hyper_group, 0, "topk_shoulder", self.predict_topk_shoulder_var)
        self._make_labeled_entry(hyper_group, 1, "topk_elbow", self.predict_topk_elbow_var)
        self._make_labeled_entry(hyper_group, 2, "topk_wrist", self.predict_topk_wrist_var)
        self._make_labeled_entry(hyper_group, 3, "max_branch_candidates", self.predict_max_branch_candidates_var)
        self._make_labeled_entry(hyper_group, 4, "fine_topk_per_branch", self.predict_fine_topk_per_branch_var)
        self._make_labeled_entry(hyper_group, 5, "max_subspace_candidates", self.predict_max_subspace_candidates_var)
        ttk.Checkbutton(hyper_group, text="enable_nr", variable=self.predict_enable_nr_var).grid(row=6, column=0, sticky="w", pady=(4, 0))
        self._make_labeled_entry(hyper_group, 7, "nr_max_iters", self.predict_nr_max_iters_var)
        self._make_labeled_entry(hyper_group, 8, "nr_tol_pos_mm", self.predict_nr_tol_pos_mm_var)
        self._make_labeled_entry(hyper_group, 9, "nr_tol_ori_rad", self.predict_nr_tol_ori_rad_var)
        self._make_labeled_entry(hyper_group, 10, "nr_damping", self.predict_nr_damping_var)
        self._make_labeled_entry(hyper_group, 11, "nr_step_scale", self.predict_nr_step_scale_var)
        self._make_note(
            hyper_group,
            12,
            "说明：前 6 项控制分层分类候选召回范围；后 5 项控制 Newton-Raphson 校正。若关闭 enable_nr，则只输出网络初值解。",
            columnspan=4,
        )

        self._make_note(tab, 7, "说明：本页只做逆解推理，q_start 不参与 predict_ik。此处 pose6 与“避障”页共用同一输入，任一页面修改都会同步。输出结果会保存为完整 JSON，右侧同时给出摘要。")
        ttk.Button(tab, text="运行 predict_ik", command=self.run_predict).grid(row=8, column=0, columnspan=3, sticky="ew", pady=(12, 0))

    def _build_obstacle_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="避障")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        scroll = ScrollableForm(tab)
        scroll.grid(row=0, column=0, sticky="nsew")
        form = scroll.inner
        form.columnconfigure(1, weight=1)
        self._make_vector_entry(form, 0, "目标位姿 pose6", ["x", "y", "z", "phi", "theta", "psi"], self.pose_part_vars, columns=3)
        self._make_note(form, 1, "范围说明：x/y/z 单位 mm，需位于机械臂可达空间；phi/theta/psi 单位 rad，通常建议填写在 [-3.1416, 3.1416]。")
        self._make_vector_entry(form, 2, "起始关节 q_start", ["q1", "q2", "q3", "q4", "q5", "q6"], self.q_start_part_vars, columns=3)
        self._make_note(form, 3, self._joint_limits_note_text())
        self._make_labeled_path_entry(form, 4, "scene_json", self.scene_var, save=False, default_key="scene_json")
        self._make_labeled_path_entry(form, 5, "prediction metadata", self.pred_meta_var, save=False, default_key="pred_meta")
        self._make_labeled_path_entry(form, 6, "branch metadata", self.branch_meta_var, save=False, default_key="branch_meta")
        self._make_labeled_path_entry(form, 7, "fine metadata", self.fine_meta_var, save=False, default_key="fine_meta")
        self._make_labeled_path_entry(form, 8, "输出 JSON", self.obstacle_out_json_var, save=True, default_key="obstacle_out_json")
        self._make_note(form, 9, "说明：本页会执行候选逆解评估、轨迹碰撞检测与自动换解。此处 pose6 与“推理”页共用同一输入，任一页面修改都会同步。q_start 会参与整条轨迹的碰撞分析。")

        obstacle_hyper_group = ttk.LabelFrame(form, text="避障常用超参数", padding=8)
        obstacle_hyper_group.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        obstacle_hyper_group.columnconfigure(1, weight=1)
        obstacle_hyper_group.columnconfigure(3, weight=1)
        self._make_labeled_entry(obstacle_hyper_group, 0, "topk_shoulder", self.obstacle_topk_shoulder_var)
        self._make_labeled_entry(obstacle_hyper_group, 1, "topk_elbow", self.obstacle_topk_elbow_var)
        self._make_labeled_entry(obstacle_hyper_group, 2, "topk_wrist", self.obstacle_topk_wrist_var)
        self._make_labeled_entry(obstacle_hyper_group, 3, "max_branch_candidates", self.obstacle_max_branch_candidates_var)
        self._make_labeled_entry(obstacle_hyper_group, 4, "fine_topk_per_branch", self.obstacle_fine_topk_per_branch_var)
        self._make_labeled_entry(obstacle_hyper_group, 5, "max_subspace_candidates", self.obstacle_max_subspace_candidates_var)
        self._make_labeled_entry(obstacle_hyper_group, 6, "max_evaluated_candidates", self.obstacle_max_evaluated_candidates_var)
        self._make_labeled_entry(obstacle_hyper_group, 7, "nr_max_iters", self.obstacle_nr_max_iters_var)
        self._make_labeled_entry(obstacle_hyper_group, 8, "nr_tol_pos_mm", self.obstacle_nr_tol_pos_mm_var)
        self._make_labeled_entry(obstacle_hyper_group, 9, "nr_tol_ori_rad", self.obstacle_nr_tol_ori_rad_var)
        self._make_labeled_entry(obstacle_hyper_group, 10, "nr_damping", self.obstacle_nr_damping_var)
        self._make_labeled_entry(obstacle_hyper_group, 11, "nr_step_scale", self.obstacle_nr_step_scale_var)
        self._make_labeled_entry(obstacle_hyper_group, 12, "trajectory_steps", self.obstacle_trajectory_steps_var)
        self._make_labeled_entry(obstacle_hyper_group, 13, "dedupe_tol_deg", self.obstacle_dedupe_tol_deg_var)
        self._make_note(
            obstacle_hyper_group,
            14,
            "说明：前 7 项控制候选召回与实际评估数量；中间 5 项控制 NR 修正；trajectory_steps 控制轨迹离散步数；dedupe_tol_deg 控制候选解去重容差，越小通常保留的不同候选越多。",
            columnspan=4,
        )

        obstacle_editor = ttk.LabelFrame(form, text="障碍物编辑（AABB）", padding=8)
        obstacle_editor.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        obstacle_editor.columnconfigure(1, weight=1)
        ttk.Label(obstacle_editor, text="当前障碍物").grid(row=0, column=0, sticky="w", pady=2)
        self.obstacle_selector = ttk.Combobox(
            obstacle_editor,
            textvariable=self.obstacle_selector_var,
            state="readonly",
            values=self.obstacle_selector_values,
        )
        self.obstacle_selector.grid(row=0, column=1, columnspan=2, sticky="ew", pady=2)
        self.obstacle_selector.bind("<<ComboboxSelected>>", self._on_obstacle_selector_changed)
        self._make_labeled_entry(obstacle_editor, 1, "obstacle name", self.obstacle_name_var)
        self._make_vector_entry(obstacle_editor, 2, "center_mm", ["x", "y", "z"], self.obstacle_center_part_vars, columns=3)
        self._make_vector_entry(obstacle_editor, 3, "size_mm", ["dx", "dy", "dz"], self.obstacle_size_part_vars, columns=3)
        self._make_note(
            obstacle_editor,
            4,
            "范围说明：center_mm 为障碍物中心坐标，单位 mm；size_mm 为长方体三边长度，单位 mm，3 个值都应大于 0。运行避障前会自动写回 scene_json。",
        )
        ttk.Button(obstacle_editor, text="从 scene_json 读取", command=self.load_obstacle_editor_from_scene).grid(row=5, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(obstacle_editor, text="写回 scene_json", command=self.save_obstacle_editor_to_scene).grid(row=5, column=1, sticky="ew", pady=(8, 0), padx=(6, 0))
        ttk.Button(obstacle_editor, text="恢复默认值", command=self.reset_obstacle_editor_to_default).grid(row=5, column=2, sticky="ew", pady=(8, 0), padx=(6, 0))
        ttk.Button(obstacle_editor, text="新增障碍物", command=self.add_obstacle_to_scene).grid(row=6, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(obstacle_editor, text="删除当前障碍物", command=self.delete_current_obstacle_from_scene).grid(row=6, column=1, columnspan=2, sticky="ew", pady=(8, 0), padx=(6, 0))

        ttk.Button(form, text="运行 plan_collision_free_ik", command=self.run_obstacle).grid(row=12, column=0, columnspan=3, sticky="ew", pady=(12, 0))

    def _build_unity_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Unity")
        tab.columnconfigure(1, weight=1)

        fk_group = ttk.LabelFrame(tab, text="FK参考导出", padding=8)
        fk_group.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        fk_group.columnconfigure(1, weight=1)
        self._make_labeled_entry(fk_group, 0, "关节角 q", self.fk_q_var)
        self._make_labeled_path_entry(fk_group, 1, "输出 JSON", self.fk_out_json_var, save=True, default_key="fk_out_json")
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
        self._make_labeled_path_entry(traj_group, 5, "输出 JSON", self.traj_out_json_var, save=True, default_key="traj_out_json")
        self._make_note(traj_group, 6, "作用：将 q_start 到 q_goal 的关节空间插值轨迹导出为 Unity 可直接播放的 JSON。")
        ttk.Button(traj_group, text="导出轨迹 JSON", command=self.run_trajectory_export).grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        obs_group = ttk.LabelFrame(tab, text="避障结果转 Unity 回放", padding=8)
        obs_group.grid(row=2, column=0, columnspan=2, sticky="ew")
        obs_group.columnconfigure(1, weight=1)
        self._make_labeled_path_entry(obs_group, 0, "plan_json", self.obstacle_plan_json_var, save=False, default_key="obstacle_plan_json")
        self._make_labeled_entry(obs_group, 1, "demo_name", self.obstacle_demo_name_var)
        self._make_labeled_path_entry(obs_group, 2, "输出 JSON", self.obstacle_unity_out_var, save=True, default_key="obstacle_unity_out_json")
        self._make_note(obs_group, 3, "作用：把 Python 侧的避障规划结果整理成 Unity 友好版 JSON，供障碍物、目标点、蓝/红轨迹回放使用。")
        ttk.Button(obs_group, text="导出避障回放 JSON", command=self.run_obstacle_unity_export).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))

    def _build_figure_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="图表")
        self.figure_tab = tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        figure_pane = ttk.Panedwindow(tab, orient="horizontal")
        figure_pane.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(figure_pane)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(3, weight=1)

        right = ttk.Frame(figure_pane)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        output_group = ttk.LabelFrame(left, text="输出路径", padding=8)
        output_group.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        output_group.columnconfigure(1, weight=1)
        self._make_labeled_dir_entry(output_group, 0, "图像输出目录", self.figure_output_dir_var, default_key="figure_output_dir")
        self._make_labeled_dir_entry(output_group, 1, "数据输出目录", self.figure_data_dir_var, default_key="figure_data_dir")
        self._make_note(
            output_group,
            2,
            "说明：不修改时仍写入默认的 figure/figures 与 figure/data。避障图主要写图像目录，核心图和工作空间图会同时写图像与数据目录。",
        )

        core_group = ttk.LabelFrame(left, text="1. 核心图表", padding=8)
        core_group.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        core_group.columnconfigure(1, weight=1)
        self._make_labeled_path_entry(core_group, 0, "单案例 IK JSON", self.figure_core_case_json_var, save=False, default_key="figure_core_case_json")
        core_note_1 = tk.Label(
            core_group,
            text=(
                "作用：汇总当前工程的核心结果图，包括 FK 偏置验证、子空间划分对比、子空间预测误差、"
                "分类器表现、benchmark 总结等。适合论文主体性能章节使用。单案例误差/时间图会读取上面的 JSON。"
            ),
            fg="#666666",
            justify="left",
            anchor="w",
        )
        core_note_1.grid(row=1, column=0, columnspan=3, sticky="ew")
        core_note_2 = tk.Label(
            core_group,
            text="主要输出目录：figure/figures/ 与 figure/data/；其余汇总图仍读取 artifacts 下的固定实验结果文件。",
            fg="#666666",
            justify="left",
            anchor="w",
        )
        core_note_2.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(core_group, text="生成核心图表", command=self.run_core_figures).grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        workspace_group = ttk.LabelFrame(left, text="2. 工作空间图", padding=8)
        workspace_group.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        workspace_group.columnconfigure(1, weight=1)
        self._make_labeled_dir_entry(workspace_group, 0, "参考样本目录", self.figure_workspace_ref_dir_var, default_key="figure_workspace_ref_dir")
        workspace_note_1 = tk.Label(
            workspace_group,
            text=(
                "作用：读取保存的子空间参考样本，生成三视图投影和三维样本可达空间图。"
                "适合说明 ABB_IRB 的样本覆盖范围与工作空间分布。"
            ),
            fg="#666666",
            justify="left",
            anchor="w",
        )
        workspace_note_1.grid(row=1, column=0, columnspan=3, sticky="ew")
        workspace_note_2 = tk.Label(
            workspace_group,
            text="依赖：所选目录下的 subspace_*_reference.npz 参考样本文件。",
            fg="#666666",
            justify="left",
            anchor="w",
        )
        workspace_note_2.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(workspace_group, text="生成工作空间图", command=self.run_workspace_figures).grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        obstacle_group = ttk.LabelFrame(left, text="3. 避障图", padding=8)
        obstacle_group.grid(row=3, column=0, sticky="nsew")
        obstacle_group.columnconfigure(1, weight=1)
        self._make_labeled_path_entry(obstacle_group, 0, "避障规划 JSON", self.figure_obstacle_plan_json_var, save=False, default_key="figure_obstacle_plan_json")
        obstacle_note_1 = tk.Label(
            obstacle_group,
            text=(
                "作用：基于所选避障规划结果绘制候选轨迹、碰撞/无碰撞对比和论文风格避障示意图。"
                "适合展示不同起始关节与目标位姿下为什么系统需要进行候选重选。"
            ),
            fg="#666666",
            justify="left",
            anchor="w",
        )
        obstacle_note_1.grid(row=1, column=0, columnspan=3, sticky="ew")
        obstacle_note_2 = tk.Label(
            obstacle_group,
            text="依赖：所选 plan_json 内的 q_start_deg、target_pose6、scene、evaluated_candidates。",
            fg="#666666",
            justify="left",
            anchor="w",
        )
        obstacle_note_2.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(obstacle_group, text="生成避障图", command=self.run_obstacle_figures).grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        preview_group = ttk.LabelFrame(right, text="避障图预览", padding=8)
        preview_group.grid(row=0, column=0, sticky="nsew", padx=(10, 0))
        preview_group.columnconfigure(0, weight=1)
        preview_group.rowconfigure(2, weight=1)
        preview_status = tk.Label(
            preview_group,
            textvariable=self.figure_preview_status_var,
            fg="#0B5394",
            justify="left",
            anchor="w",
        )
        preview_status.grid(row=0, column=0, sticky="ew")
        preview_path = tk.Label(
            preview_group,
            textvariable=self.figure_preview_path_var,
            fg="#666666",
            justify="left",
            anchor="w",
        )
        preview_path.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        self.figure_preview_label = ttk.Label(preview_group, text="暂无预览", anchor="center")
        self.figure_preview_label.grid(row=2, column=0, sticky="nsew")
        ttk.Button(preview_group, text="刷新预览", command=self.refresh_obstacle_figure_preview).grid(row=3, column=0, sticky="ew", pady=(8, 0))

        figure_pane.add(left, weight=3)
        figure_pane.add(right, weight=4)

        for parent, widgets in [
            (core_group, [core_note_1, core_note_2]),
            (workspace_group, [workspace_note_1, workspace_note_2]),
            (obstacle_group, [obstacle_note_1, obstacle_note_2]),
            (preview_group, [preview_status, preview_path]),
        ]:
            def _refresh_wrap(event: tk.Event, items: list[tk.Label] = widgets) -> None:
                wrap = max(220, event.width - 24)
                for item in items:
                    item.configure(wraplength=wrap)

            parent.bind("<Configure>", _refresh_wrap, add="+")

    def _get_obstacle_preview_candidates(self) -> list[Path]:
        figures_dir = Path(self.figure_output_dir_var.get())
        return [
            figures_dir / "obstacle_candidates_overview_thesis.png",
            figures_dir / "obstacle_candidates_overview.png",
            figures_dir / "obstacle_candidates_free_only_thesis.png",
            figures_dir / "obstacle_candidates_colliding_only_thesis.png",
        ]

    def _find_latest_obstacle_preview_path(self) -> Path | None:
        for path in self._get_obstacle_preview_candidates():
            if path.exists():
                return path
        return None

    def _build_obstacle_figure_summary(self, output_dir: Path) -> str:
        preview_path = self._find_latest_obstacle_preview_path()
        lines = ["任务：generate_obstacle_figures", ""]
        lines.append(f"规划结果 JSON：{self.figure_obstacle_plan_json_var.get()}")
        lines.append(f"图像输出目录：{output_dir}")
        if preview_path:
            lines.append(f"预览图：{preview_path}")
        else:
            lines.append("预览图：未找到已生成 PNG")
        return "\n".join(lines)

    def _update_obstacle_preview_widget(self, image_path: Path | None) -> None:
        if image_path is None or not image_path.exists():
            self.obstacle_preview_photo = None
            self.figure_preview_label.configure(image="", text="暂无预览")
            self.figure_preview_status_var.set("尚未生成避障图预览。")
            self.figure_preview_path_var.set("")
            return

        image = tk.PhotoImage(file=str(image_path))
        max_w = 360
        max_h = 420
        factor = max(
            1,
            (image.width() + max_w - 1) // max_w,
            (image.height() + max_h - 1) // max_h,
        )
        if factor > 1:
            image = image.subsample(factor, factor)
        self.obstacle_preview_photo = image
        self.figure_preview_label.configure(image=self.obstacle_preview_photo, text="")
        self.figure_preview_status_var.set("已加载最新避障图预览。")
        self.figure_preview_path_var.set(str(image_path))

    def refresh_obstacle_figure_preview(self) -> None:
        path = self._find_latest_obstacle_preview_path()
        self._update_obstacle_preview_widget(path)

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

    def _parse_vec3_text(self, text: str, *, label: str) -> list[float]:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
        if len(values) != 3:
            raise ValueError(f"{label} 必须为 3 个逗号分隔数值，例如 221.7,274.53,493.57")
        return values

    def _build_obstacle_selector_label(self, index: int, name: str) -> str:
        return f"{index + 1}. {name}"

    def _obstacle_payload_to_center_size(self, obstacle: dict) -> tuple[list[float], list[float]]:
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

    def _set_obstacle_editor_values(self, name: str, center: list[float], size: list[float]) -> None:
        self.obstacle_name_var.set(name)
        self.obstacle_center_var.set(self._format_q_deg(center))
        self.obstacle_size_var.set(self._format_q_deg(size))

    def _refresh_obstacle_selector(self, obstacles: list[dict], selected_index: int = 0) -> None:
        values = [
            self._build_obstacle_selector_label(idx, str(obstacle.get("name", f"obstacle_{idx + 1}")))
            for idx, obstacle in enumerate(obstacles)
        ]
        self.obstacle_selector_values = values
        self.obstacle_selector.configure(values=values)
        if not values:
            self.current_obstacle_index = 0
            self.obstacle_selector_var.set("")
            return
        selected_index = max(0, min(selected_index, len(values) - 1))
        self.current_obstacle_index = selected_index
        self.obstacle_selector_var.set(values[selected_index])

    def _load_selected_obstacle_into_editor(self, obstacles: list[dict], index: int) -> str:
        if not obstacles:
            raise ValueError("scene_json 中没有 obstacles")
        index = max(0, min(index, len(obstacles) - 1))
        obstacle = obstacles[index]
        name = str(obstacle.get("name", f"obstacle_{index + 1}"))
        center, size = self._obstacle_payload_to_center_size(obstacle)
        self._refresh_obstacle_selector(obstacles, selected_index=index)
        self._set_obstacle_editor_values(name, center, size)
        return f"已加载障碍物：{self._build_obstacle_selector_label(index, name)}"

    def _build_obstacle_payload_from_editor(self, *, fallback_name: str) -> dict:
        center = self._parse_vec3_text(self.obstacle_center_var.get(), label="center_mm")
        size = self._parse_vec3_text(self.obstacle_size_var.get(), label="size_mm")
        if any(v <= 0.0 for v in size):
            raise ValueError("size_mm 的三个分量都必须大于 0")
        return {
            "name": self.obstacle_name_var.get().strip() or fallback_name,
            "center_mm": center,
            "size_mm": size,
        }

    def _make_new_obstacle_name(self, obstacles: list[dict]) -> str:
        used = {str(obstacle.get("name", "")).strip() for obstacle in obstacles}
        idx = 1
        while True:
            candidate = f"demo_box_{idx}"
            if candidate not in used:
                return candidate
            idx += 1

    def _load_scene_payload(self) -> tuple[Path, dict]:
        scene_path = Path(self.scene_var.get())
        if not scene_path.exists():
            raise FileNotFoundError(f"未找到 scene_json：{scene_path}")
        payload = json.loads(scene_path.read_text(encoding="utf-8"))
        return scene_path, payload

    def _write_scene_payload(self, scene_path: Path, payload: dict) -> None:
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        scene_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _try_load_obstacle_editor_from_scene(self) -> None:
        try:
            self._load_obstacle_editor_from_scene()
        except Exception:
            self._refresh_obstacle_selector([], selected_index=0)

    def _reset_obstacle_editor_to_default(self) -> str:
        self.obstacle_name_var.set(DEFAULT_OBSTACLE_NAME)
        self.obstacle_center_var.set(self._format_q_deg(DEFAULT_OBSTACLE_CENTER_MM))
        self.obstacle_size_var.set(self._format_q_deg(DEFAULT_OBSTACLE_SIZE_MM))
        return (
            f"已恢复默认障碍物参数：{DEFAULT_OBSTACLE_NAME}\n"
            f"center_mm：{DEFAULT_OBSTACLE_CENTER_MM}\n"
            f"size_mm：{DEFAULT_OBSTACLE_SIZE_MM}"
        )

    def reset_obstacle_editor_to_default(self) -> None:
        msg = self._reset_obstacle_editor_to_default()
        self.set_summary(f"任务：障碍物编辑\n\n{msg}")
        self.append_log(msg)

    def _load_obstacle_editor_from_scene(self) -> str:
        scene_path, payload = self._load_scene_payload()
        obstacles = payload.get("obstacles", [])
        if not obstacles:
            raise ValueError(f"scene_json 中没有 obstacles：{scene_path}")
        load_msg = self._load_selected_obstacle_into_editor(obstacles, self.current_obstacle_index)
        return f"{load_msg}\nscene_json：{scene_path}"

    def load_obstacle_editor_from_scene(self) -> None:
        try:
            msg = self._load_obstacle_editor_from_scene()
            self.set_summary(f"任务：障碍物编辑\n\n{msg}")
            self.append_log(msg)
        except Exception as exc:
            self.set_summary(f"任务：障碍物编辑\n\n读取失败：{exc}")
            self.append_log(f"[obstacle editor] 读取失败：{exc}")

    def _save_obstacle_editor_to_scene(self) -> str:
        scene_path, payload = self._load_scene_payload()
        obstacles = payload.setdefault("obstacles", [])
        obstacle_payload = self._build_obstacle_payload_from_editor(fallback_name=f"obstacle_{self.current_obstacle_index + 1}")
        if obstacles:
            index = max(0, min(self.current_obstacle_index, len(obstacles) - 1))
            obstacles[index] = obstacle_payload
        else:
            obstacles.append(obstacle_payload)
            index = 0
        self._write_scene_payload(scene_path, payload)
        self._refresh_obstacle_selector(obstacles, selected_index=index)
        return (
            f"已写回 scene_json 当前障碍物：{self._build_obstacle_selector_label(index, obstacle_payload['name'])}\n"
            f"center_mm：{obstacle_payload['center_mm']}\n"
            f"size_mm：{obstacle_payload['size_mm']}\n"
            f"scene_json：{scene_path}"
        )

    def save_obstacle_editor_to_scene(self) -> None:
        try:
            msg = self._save_obstacle_editor_to_scene()
            self.set_summary(f"任务：障碍物编辑\n\n{msg}")
            self.append_log(msg)
        except Exception as exc:
            self.set_summary(f"任务：障碍物编辑\n\n写回失败：{exc}")
            self.append_log(f"[obstacle editor] 写回失败：{exc}")

    def _on_obstacle_selector_changed(self, _event: tk.Event | None = None) -> None:
        try:
            scene_path, payload = self._load_scene_payload()
            obstacles = payload.get("obstacles", [])
            if not obstacles:
                return
            try:
                index = self.obstacle_selector_values.index(self.obstacle_selector_var.get())
            except ValueError:
                index = 0
            msg = self._load_selected_obstacle_into_editor(obstacles, index)
            self.set_summary(f"任务：障碍物编辑\n\n{msg}\nscene_json：{scene_path}")
            self.append_log(msg)
        except Exception as exc:
            self.set_summary(f"任务：障碍物编辑\n\n切换障碍物失败：{exc}")
            self.append_log(f"[obstacle editor] 切换失败：{exc}")

    def _add_obstacle_to_scene(self) -> str:
        scene_path, payload = self._load_scene_payload()
        obstacles = payload.setdefault("obstacles", [])
        new_name = self._make_new_obstacle_name(obstacles)
        obstacle_payload = {
            "name": new_name,
            "center_mm": list(NEW_OBSTACLE_CENTER_MM),
            "size_mm": list(NEW_OBSTACLE_SIZE_MM),
        }
        obstacles.append(obstacle_payload)
        new_index = len(obstacles) - 1
        self._write_scene_payload(scene_path, payload)
        self._load_selected_obstacle_into_editor(obstacles, new_index)
        return (
            f"已新增障碍物：{self._build_obstacle_selector_label(new_index, new_name)}\n"
            f"center_mm：{obstacle_payload['center_mm']}\n"
            f"size_mm：{obstacle_payload['size_mm']}\n"
            f"scene_json：{scene_path}"
        )

    def add_obstacle_to_scene(self) -> None:
        try:
            msg = self._add_obstacle_to_scene()
            self.set_summary(f"任务：障碍物编辑\n\n{msg}")
            self.append_log(msg)
        except Exception as exc:
            self.set_summary(f"任务：障碍物编辑\n\n新增失败：{exc}")
            self.append_log(f"[obstacle editor] 新增失败：{exc}")

    def _delete_current_obstacle_from_scene(self) -> str:
        scene_path, payload = self._load_scene_payload()
        obstacles = payload.get("obstacles", [])
        if not obstacles:
            raise ValueError("scene_json 中没有可删除的障碍物")
        if len(obstacles) == 1:
            raise ValueError("当前至少需要保留 1 个障碍物；如需空场景，请手动编辑 scene_json。")
        index = max(0, min(self.current_obstacle_index, len(obstacles) - 1))
        removed = obstacles.pop(index)
        next_index = min(index, len(obstacles) - 1)
        payload["obstacles"] = obstacles
        self._write_scene_payload(scene_path, payload)
        load_msg = self._load_selected_obstacle_into_editor(obstacles, next_index)
        return (
            f"已删除障碍物：{removed.get('name', f'obstacle_{index + 1}')}\n"
            f"{load_msg}\n"
            f"scene_json：{scene_path}"
        )

    def delete_current_obstacle_from_scene(self) -> None:
        try:
            msg = self._delete_current_obstacle_from_scene()
            self.set_summary(f"任务：障碍物编辑\n\n{msg}")
            self.append_log(msg)
        except Exception as exc:
            self.set_summary(f"任务：障碍物编辑\n\n删除失败：{exc}")
            self.append_log(f"[obstacle editor] 删除失败：{exc}")

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

    def _build_figure_env(self) -> dict[str, str]:
        figures_dir = Path(self.figure_output_dir_var.get())
        data_dir = Path(self.figure_data_dir_var.get())
        figures_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["ABB_FIGURE_OUTPUT_DIR"] = str(figures_dir)
        env["ABB_FIGURE_DATA_DIR"] = str(data_dir)
        return env

    def run_command(
        self,
        title: str,
        cmd: list[str],
        summary_builder=None,
        summary_path: Path | None = None,
        env: dict[str, str] | None = None,
        on_success=None,
    ) -> None:
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
                    env=env,
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
                if code == 0 and on_success is not None:
                    self.after(0, on_success)
            except Exception as exc:
                self.append_log(f"[error] {exc}")
                self.set_summary(f"任务：{title}\n\n发生异常：{exc}")
            finally:
                self.status_var.set("Ready")

        threading.Thread(target=worker, daemon=True).start()

    def run_predict(self) -> None:
        out_path = Path(self.predict_out_json_var.get())
        self._ensure_parent(str(out_path))
        self.figure_core_case_json_var.set(str(out_path))
        cmd = [
            PYTHON, "-X", "utf8", "predict_ik.py",
            "--candidate_mode", "hierarchical",
            f"--pose={self.pose_var.get()}",
            "--pred_meta", self.pred_meta_var.get(),
            "--branch_meta", self.branch_meta_var.get(),
            "--fine_meta", self.fine_meta_var.get(),
            "--topk_shoulder", self.predict_topk_shoulder_var.get(),
            "--topk_elbow", self.predict_topk_elbow_var.get(),
            "--topk_wrist", self.predict_topk_wrist_var.get(),
            "--max_branch_candidates", self.predict_max_branch_candidates_var.get(),
            "--fine_topk_per_branch", self.predict_fine_topk_per_branch_var.get(),
            "--max_subspace_candidates", self.predict_max_subspace_candidates_var.get(),
            "--nr_max_iters", self.predict_nr_max_iters_var.get(),
            "--nr_tol_pos_mm", self.predict_nr_tol_pos_mm_var.get(),
            "--nr_tol_ori_rad", self.predict_nr_tol_ori_rad_var.get(),
            "--nr_damping", self.predict_nr_damping_var.get(),
            "--nr_step_scale", self.predict_nr_step_scale_var.get(),
            "--out_json", str(out_path),
        ]
        if self.predict_enable_nr_var.get():
            cmd.append("--enable_nr")
        self.run_command("predict_ik", cmd, self._build_predict_summary, out_path)

    def run_obstacle(self) -> None:
        out_path = Path(self.obstacle_out_json_var.get())
        self._ensure_parent(str(out_path))
        self.obstacle_plan_json_var.set(str(out_path))
        self.figure_obstacle_plan_json_var.set(str(out_path))
        try:
            save_msg = self._save_obstacle_editor_to_scene()
            self.append_log(save_msg)
        except Exception as exc:
            self.set_summary(f"任务：plan_collision_free_ik\n\n障碍物参数写回失败：{exc}")
            self.append_log(f"[plan_collision_free_ik] 障碍物参数写回失败：{exc}")
            return
        cmd = [
            PYTHON, "-X", "utf8", "scripts\\plan_collision_free_ik.py",
            f"--pose={self.obstacle_pose_var.get()}",
            f"--q_start={self.q_start_var.get()}",
            "--scene_json", self.scene_var.get(),
            "--pred_meta", self.pred_meta_var.get(),
            "--branch_meta", self.branch_meta_var.get(),
            "--fine_meta", self.fine_meta_var.get(),
            "--topk_shoulder", self.obstacle_topk_shoulder_var.get(),
            "--topk_elbow", self.obstacle_topk_elbow_var.get(),
            "--topk_wrist", self.obstacle_topk_wrist_var.get(),
            "--max_branch_candidates", self.obstacle_max_branch_candidates_var.get(),
            "--fine_topk_per_branch", self.obstacle_fine_topk_per_branch_var.get(),
            "--max_subspace_candidates", self.obstacle_max_subspace_candidates_var.get(),
            "--max_evaluated_candidates", self.obstacle_max_evaluated_candidates_var.get(),
            "--nr_max_iters", self.obstacle_nr_max_iters_var.get(),
            "--nr_tol_pos_mm", self.obstacle_nr_tol_pos_mm_var.get(),
            "--nr_tol_ori_rad", self.obstacle_nr_tol_ori_rad_var.get(),
            "--nr_damping", self.obstacle_nr_damping_var.get(),
            "--nr_step_scale", self.obstacle_nr_step_scale_var.get(),
            "--trajectory_steps", self.obstacle_trajectory_steps_var.get(),
            "--dedupe_tol_deg", self.obstacle_dedupe_tol_deg_var.get(),
            "--save_selected_frames",
            "--out_json", str(out_path),
        ]
        self.run_command(
            "plan_collision_free_ik",
            cmd,
            self._build_obstacle_summary,
            out_path,
            on_success=self.run_obstacle_figures_after_planning,
        )

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

    def run_obstacle_figures_after_planning(self) -> None:
        self.notebook.select(self.figure_tab)
        self.run_obstacle_figures()

    def run_core_figures(self) -> None:
        self.run_command(
            "generate_core_figures",
            [
                PYTHON, "-X", "utf8", "figure\\scripts\\generate_core_figures.py",
                "--single_case_json", self.figure_core_case_json_var.get(),
            ],
            env=self._build_figure_env(),
        )

    def run_workspace_figures(self) -> None:
        self.run_command(
            "generate_workspace_figures",
            [
                PYTHON, "-X", "utf8", "figure\\scripts\\generate_workspace_figures.py",
                "--reference_dir", self.figure_workspace_ref_dir_var.get(),
            ],
            env=self._build_figure_env(),
        )

    def run_obstacle_figures(self) -> None:
        figures_dir = Path(self.figure_output_dir_var.get())
        self.run_command(
            "generate_obstacle_figures",
            [
                PYTHON, "-X", "utf8", "figure\\scripts\\generate_obstacle_candidate_trajectory_figures.py",
                "--plan_json", self.figure_obstacle_plan_json_var.get(),
            ],
            summary_builder=self._build_obstacle_figure_summary,
            summary_path=figures_dir,
            env=self._build_figure_env(),
            on_success=self.refresh_obstacle_figure_preview,
        )

    def open_result_dir(self) -> None:
        subprocess.Popen(["explorer", str(ROOT / "artifacts")])

    def open_unity_dir(self) -> None:
        subprocess.Popen(["explorer", str(UNITY_DIR)])


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
