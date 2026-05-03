#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abb_nn.optimization import NROptions, evaluate_solution_metrics, newton_raphson_refine
from fk_model import JOINT_LIMITS_DEG, pose6_from_q
from predict_ik import (
    apply_normalizer,
    generate_hierarchical_candidates,
    load_json,
    load_prediction_pair,
    position_l2_norm,
    predict_q_deg,
    safe_torch_load,
)


FIGURE_DIR = ROOT / "figure"
DATA_DIR = FIGURE_DIR / "data"
ARTIFACTS_DIR = ROOT / "artifacts"


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def pose_delta(target_pose6: np.ndarray, pred_pose6: np.ndarray) -> np.ndarray:
    delta = np.asarray(target_pose6, dtype=float).reshape(6) - np.asarray(pred_pose6, dtype=float).reshape(6)
    delta[3:] = wrap_to_pi(delta[3:])
    return delta


def format_float(x: float, decimals: int = 6) -> str:
    return f"{float(x):.{decimals}f}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_prediction_system() -> tuple[dict, np.ndarray, np.ndarray, dict[int, dict], dict[int, dict]]:
    pred_meta_path = ARTIFACTS_DIR / "prediction_system_formal" / "metadata.json"
    pred_meta = load_json(pred_meta_path)
    pred_mean = np.array(pred_meta["normalizer"]["mean"], dtype=np.float32).reshape(1, -1)
    pred_std = np.array(pred_meta["normalizer"]["std"], dtype=np.float32).reshape(1, -1)
    model_index = {int(item["subspace_id"]): item for item in pred_meta["trained_subspaces"]}
    model_cache: dict[int, dict] = {}
    for sid, item in model_index.items():
        ckpt = safe_torch_load(pred_meta_path.parent / "subspace_models" / item["model_file"])
        m15, m6 = load_prediction_pair(ckpt)
        model_cache[int(sid)] = {
            "m15": m15,
            "m6": m6,
            "e_max": float(ckpt.get("e_max", np.inf)),
        }
    return pred_meta, pred_mean, pred_std, model_index, model_cache


def solve_case(
    goal_pose6: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    model_index: dict[int, dict],
    model_cache: dict[int, dict],
) -> dict:
    branch_meta_path = ARTIFACTS_DIR / "branch_classification_system" / "metadata.json"
    fine_meta_path = ARTIFACTS_DIR / "fine_classification_system" / "metadata.json"
    branch_meta = load_json(branch_meta_path)
    fine_meta = load_json(fine_meta_path)

    candidate_labels, _, _ = generate_hierarchical_candidates(
        goal_pose6.astype(np.float32),
        branch_meta_path,
        branch_meta,
        fine_meta_path,
        fine_meta,
        topk_shoulder=2,
        topk_elbow=1,
        topk_wrist=2,
        max_branch_candidates=6,
        fine_topk_per_branch=3,
        max_subspace_candidates=18,
    )

    x_pred = apply_normalizer(goal_pose6.reshape(1, -1).astype(np.float32), pred_mean, pred_std)
    trained_all = sorted(model_index.keys())
    candidate_labels = [int(sid) for sid in candidate_labels if int(sid) in model_index]
    if not candidate_labels:
        candidate_labels = trained_all

    best = None
    for sid in candidate_labels:
        model_item = model_cache[int(sid)]
        q0_deg = predict_q_deg(model_item["m15"], model_item["m6"], x_pred)
        initial_pos_l2 = position_l2_norm(q0_deg, goal_pose6)
        initial_metrics = evaluate_solution_metrics(q0_deg, goal_pose6)
        item = {
            "subspace_id": int(sid),
            "q0_deg": np.asarray(q0_deg, dtype=float),
            "initial_pos_l2_mm": float(initial_pos_l2),
            "initial_pos_err_mm": float(initial_metrics["final_pos_err_mm"]),
            "initial_ori_err_rad": float(initial_metrics["final_ori_err_rad"]),
        }
        if best is None or item["initial_pos_l2_mm"] < best["initial_pos_l2_mm"]:
            best = item

    if best is None:
        raise RuntimeError("No initial candidate generated.")

    q0_deg = np.asarray(best["q0_deg"], dtype=float)
    nr = newton_raphson_refine(
        q0_deg=q0_deg,
        target_pose6=goal_pose6,
        options=NROptions(max_iters=60, tol_pos_mm=1e-4, tol_ori_rad=1e-6, damping=1e-3),
    )
    q_refined_deg = np.asarray(nr["q_deg"], dtype=float)
    refined_metrics = evaluate_solution_metrics(q_refined_deg, goal_pose6)

    return {
        "selected_subspace_id": int(best["subspace_id"]),
        "initial_q_deg": q0_deg,
        "initial_pose6": np.asarray(initial_metrics["final_pose6"], dtype=float),
        "initial_pos_l2_mm": float(best["initial_pos_l2_mm"]),
        "initial_pos_err_mm": float(best["initial_pos_err_mm"]),
        "initial_ori_err_rad": float(best["initial_ori_err_rad"]),
        "optimized_q_deg": np.asarray(q_refined_deg, dtype=float),
        "optimized_pose6": np.asarray(refined_metrics["final_pose6"], dtype=float),
        "optimized_pos_err_mm": float(refined_metrics["final_pos_err_mm"]),
        "optimized_ori_err_rad": float(refined_metrics["final_ori_err_rad"]),
        "nr_iters": int(nr["iters"]),
        "nr_converged": bool(nr["converged"]),
    }


def sample_success_cases(n_cases: int = 5, seed: int = 2026, max_trials: int = 400) -> list[dict]:
    rng = np.random.default_rng(seed)
    _, pred_mean, pred_std, model_index, model_cache = load_prediction_system()

    rows: list[dict] = []
    trials = 0
    while len(rows) < n_cases and trials < max_trials:
        trials += 1
        q_min = np.asarray([lo for lo, _ in JOINT_LIMITS_DEG], dtype=float)
        q_max = np.asarray([hi for _, hi in JOINT_LIMITS_DEG], dtype=float)
        exact_q_deg = rng.uniform(q_min, q_max)
        exact_pose6 = np.asarray(pose6_from_q(exact_q_deg, input_unit="deg"), dtype=float)

        solved = solve_case(exact_pose6, pred_mean, pred_std, model_index, model_cache)
        if not solved["nr_converged"]:
            continue
        if solved["optimized_pos_err_mm"] > 1e-2 or solved["optimized_ori_err_rad"] > 1e-4:
            continue

        start_q_deg = rng.uniform(q_min, q_max)
        start_pose6 = np.asarray(pose6_from_q(start_q_deg, input_unit="deg"), dtype=float)
        initial_delta = pose_delta(exact_pose6, solved["initial_pose6"])
        optimized_delta = pose_delta(exact_pose6, solved["optimized_pose6"])
        exact_delta = pose_delta(exact_pose6, exact_pose6)

        rows.append(
            {
                "sample_id": len(rows) + 1,
                "start_pose6": start_pose6,
                "goal_pose6": exact_pose6,
                "exact_q_deg": np.asarray(exact_q_deg, dtype=float),
                "exact_pose6": np.asarray(exact_pose6, dtype=float),
                "exact_pose_delta": np.asarray(exact_delta, dtype=float),
                "selected_subspace_id": int(solved["selected_subspace_id"]),
                "initial_q_deg": np.asarray(solved["initial_q_deg"], dtype=float),
                "initial_pose6": np.asarray(solved["initial_pose6"], dtype=float),
                "initial_pose_delta": np.asarray(initial_delta, dtype=float),
                "initial_pos_l2_mm": float(solved["initial_pos_l2_mm"]),
                "initial_pos_err_mm": float(solved["initial_pos_err_mm"]),
                "initial_ori_err_rad": float(solved["initial_ori_err_rad"]),
                "optimized_q_deg": np.asarray(solved["optimized_q_deg"], dtype=float),
                "optimized_pose6": np.asarray(solved["optimized_pose6"], dtype=float),
                "optimized_pose_delta": np.asarray(optimized_delta, dtype=float),
                "optimized_pos_err_mm": float(solved["optimized_pos_err_mm"]),
                "optimized_ori_err_rad": float(solved["optimized_ori_err_rad"]),
                "nr_iters": int(solved["nr_iters"]),
                "nr_converged": bool(solved["nr_converged"]),
            }
        )

    if len(rows) < n_cases:
        raise RuntimeError(f"Only collected {len(rows)} successful cases within {max_trials} trials.")
    return rows


def build_section_table(title: str, headers: list[str], data_rows: list[list[str]]) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in data_rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def section_rows(cases: list[dict], section_key: str) -> list[list[str]]:
    if section_key == "start_pose":
        return [[str(case["sample_id"])] + [format_float(v) for v in case["start_pose6"]] for case in cases]
    if section_key == "goal_pose":
        return [[str(case["sample_id"])] + [format_float(v) for v in case["goal_pose6"]] for case in cases]
    if section_key == "initial_q":
        return [[str(case["sample_id"])] + [format_float(v) for v in case["initial_q_deg"]] for case in cases]
    if section_key == "exact_q":
        return [[str(case["sample_id"])] + [format_float(v) for v in case["exact_q_deg"]] for case in cases]
    if section_key == "optimized_q":
        return [[str(case["sample_id"])] + [format_float(v) for v in case["optimized_q_deg"]] for case in cases]
    if section_key == "final_error":
        return [
            [
                str(case["sample_id"]),
                format_float(case["optimized_pos_err_mm"], 8),
                format_float(case["optimized_ori_err_rad"], 10),
            ]
            for case in cases
        ]
    raise ValueError(f"Unsupported section key: {section_key}")


def build_reference_csv_text(cases: list[dict]) -> str:
    blocks = [
        (
            "Start Pose",
            ["No", "x (mm)", "y (mm)", "z (mm)", "phi (rad)", "theta (rad)", "psi (rad)"],
            section_rows(cases, "start_pose"),
        ),
        (
            "Goal Pose",
            ["No", "x (mm)", "y (mm)", "z (mm)", "phi (rad)", "theta (rad)", "psi (rad)"],
            section_rows(cases, "goal_pose"),
        ),
        (
            "Initial Solution",
            ["No", "q1 (deg)", "q2 (deg)", "q3 (deg)", "q4 (deg)", "q5 (deg)", "q6 (deg)"],
            section_rows(cases, "initial_q"),
        ),
        (
            "Exact Solution",
            ["No", "q1 (deg)", "q2 (deg)", "q3 (deg)", "q4 (deg)", "q5 (deg)", "q6 (deg)"],
            section_rows(cases, "exact_q"),
        ),
        (
            "Optimized Solution",
            ["No", "q1 (deg)", "q2 (deg)", "q3 (deg)", "q4 (deg)", "q5 (deg)", "q6 (deg)"],
            section_rows(cases, "optimized_q"),
        ),
        (
            "Final Error",
            ["No", "position error (mm)", "orientation error (rad)"],
            section_rows(cases, "final_error"),
        ),
    ]
    lines: list[str] = []
    for idx, (title, headers, rows) in enumerate(blocks):
        lines.append(title)
        lines.append(",".join(headers))
        for row in rows:
            lines.append(",".join(row))
        if idx != len(blocks) - 1:
            lines.append("")
    return "\n".join(lines) + "\n"


def build_markdown(cases: list[dict]) -> str:
    lines = [
        "# Table 6. Example of Random Data Test",
        "",
        "说明：以下样本按论文表格风格整理。位置单位为 `mm`，姿态单位为 `rad`，关节角单位为 `deg`。",
        "",
    ]
    lines.extend(
        build_section_table(
            "Start Pose",
            ["No", "x (mm)", "y (mm)", "z (mm)", "phi (rad)", "theta (rad)", "psi (rad)"],
            section_rows(cases, "start_pose"),
        )
    )
    lines.extend(
        build_section_table(
            "Goal Pose",
            ["No", "x (mm)", "y (mm)", "z (mm)", "phi (rad)", "theta (rad)", "psi (rad)"],
            section_rows(cases, "goal_pose"),
        )
    )
    lines.extend(
        build_section_table(
            "Initial Solution",
            ["No", "q1 (deg)", "q2 (deg)", "q3 (deg)", "q4 (deg)", "q5 (deg)", "q6 (deg)"],
            section_rows(cases, "initial_q"),
        )
    )
    lines.extend(
        build_section_table(
            "Exact Solution",
            ["No", "q1 (deg)", "q2 (deg)", "q3 (deg)", "q4 (deg)", "q5 (deg)", "q6 (deg)"],
            section_rows(cases, "exact_q"),
        )
    )
    lines.extend(
        build_section_table(
            "Optimized Solution",
            ["No", "q1 (deg)", "q2 (deg)", "q3 (deg)", "q4 (deg)", "q5 (deg)", "q6 (deg)"],
            section_rows(cases, "optimized_q"),
        )
    )
    lines.extend(
        build_section_table(
            "Final Error",
            ["No", "position error (mm)", "orientation error (rad)"],
            section_rows(cases, "final_error"),
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    cases = sample_success_cases(n_cases=5, seed=2026, max_trials=400)
    csv_text = build_reference_csv_text(cases)
    markdown = build_markdown(cases)

    wide_csv_path = DATA_DIR / "thesis_ik_case_samples_wide.csv"
    md_path = DATA_DIR / "thesis_ik_case_samples.md"

    write_text(wide_csv_path, csv_text)
    write_text(md_path, markdown)

    print("Generated thesis IK case samples:")
    print(wide_csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
