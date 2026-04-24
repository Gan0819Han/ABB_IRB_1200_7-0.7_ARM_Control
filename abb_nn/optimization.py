#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

from fk_model import JOINT_LIMITS_DEG, fk_abb_irb, numerical_pose_jacobian_rad, pose6_from_q, zyx_euler_to_rot


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def rotation_geodesic_error_rad(R_target: np.ndarray, R_pred: np.ndarray) -> float:
    cos_theta = (np.trace(R_target.T @ R_pred) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def target_pose_to_rotation(target_pose6: Iterable[float]) -> np.ndarray:
    pose = np.asarray(list(target_pose6), dtype=float).reshape(6)
    return zyx_euler_to_rot(float(pose[3]), float(pose[4]), float(pose[5]))


def evaluate_solution_metrics(
    q_deg: Iterable[float],
    target_pose6: Iterable[float],
) -> Dict[str, object]:
    q = np.asarray(list(q_deg), dtype=float).reshape(6)
    target = np.asarray(list(target_pose6), dtype=float).reshape(6)
    final_pose6 = pose6_from_q(q, input_unit="deg")
    _, p_pred, R_pred = fk_abb_irb(q, input_unit="deg")
    R_target = target_pose_to_rotation(target)
    pos_err = float(np.linalg.norm(p_pred - target[:3]))
    ori_err = rotation_geodesic_error_rad(R_target, R_pred)
    within_limits = bool(np.all((q >= JOINT_LIMITS_DEG[:, 0]) & (q <= JOINT_LIMITS_DEG[:, 1])))
    return {
        "q_deg": q.tolist(),
        "final_pose6": final_pose6.tolist(),
        "final_pos_err_mm": pos_err,
        "final_ori_err_rad": ori_err,
        "within_joint_limits": within_limits,
    }


def weighted_pose_error_vector(
    target_pose6: Iterable[float],
    q_rad: np.ndarray,
    orientation_weight: float = 200.0,
) -> np.ndarray:
    target = np.asarray(list(target_pose6), dtype=float).reshape(6)
    cur = pose6_from_q(q_rad, input_unit="rad")
    err = target - cur
    err[3:] = wrap_to_pi(err[3:])
    err_w = err.copy()
    err_w[3:] *= orientation_weight
    return err_w


def weighted_pose_jacobian(
    q_rad: np.ndarray,
    orientation_weight: float = 200.0,
) -> np.ndarray:
    J = numerical_pose_jacobian_rad(q_rad)
    J_w = J.copy()
    J_w[3:, :] *= orientation_weight
    return J_w


def weighted_pose_cost(
    target_pose6: Iterable[float],
    q_rad: np.ndarray,
    orientation_weight: float = 200.0,
) -> float:
    err_w = weighted_pose_error_vector(
        target_pose6=target_pose6,
        q_rad=q_rad,
        orientation_weight=orientation_weight,
    )
    return float(0.5 * np.dot(err_w, err_w))


@dataclass
class NROptions:
    max_iters: int = 40
    tol_pos_mm: float = 1e-3
    tol_ori_rad: float = 1e-3
    damping: float = 1e-5
    step_scale: float = 1.0
    project_to_joint_limits: bool = True


@dataclass
class DLSOptions:
    max_iters: int = 80
    tol_pos_mm: float = 1.0
    tol_ori_rad: float = 1e-2
    damping: float = 1e-2
    step_scale: float = 1.0
    orientation_weight: float = 200.0
    project_to_joint_limits: bool = True


@dataclass
class LBFGSBOptions:
    max_iters: int = 200
    tol_pos_mm: float = 1.0
    tol_ori_rad: float = 1e-2
    orientation_weight: float = 200.0
    ftol: float = 1e-12
    gtol: float = 1e-8


def _pose_error(target_pose6: np.ndarray, q_rad: np.ndarray) -> np.ndarray:
    cur = pose6_from_q(q_rad, input_unit="rad")
    err = target_pose6 - cur
    err[3:] = wrap_to_pi(err[3:])
    return err


def _limits_rad() -> tuple[np.ndarray, np.ndarray]:
    lower_rad = np.deg2rad(JOINT_LIMITS_DEG[:, 0])
    upper_rad = np.deg2rad(JOINT_LIMITS_DEG[:, 1])
    return lower_rad, upper_rad


def _finalize_iterative_result(
    q_rad: np.ndarray,
    target_pose6: Iterable[float],
    iters: int,
    converged: bool,
    method: str,
    start_q_deg: Sequence[float],
) -> Dict[str, object]:
    q_deg = np.rad2deg(np.asarray(q_rad, dtype=float).reshape(6))
    metrics = evaluate_solution_metrics(q_deg, target_pose6)
    metrics.update(
        {
            "iters": int(iters),
            "converged": bool(converged),
            "method": method,
            "start_q_deg": [float(x) for x in start_q_deg],
            "weighted_cost": weighted_pose_cost(
                target_pose6=target_pose6,
                q_rad=np.deg2rad(q_deg),
            ),
        }
    )
    return metrics


def newton_raphson_refine(
    q0_deg: Iterable[float],
    target_pose6: Iterable[float],
    options: NROptions | None = None,
) -> Dict[str, object]:
    """
    Damped NR / DLS-style local refinement:
        q_{k+1} = q_k + J^T (J J^T + λI)^(-1) e
    """
    opt = options or NROptions()
    q_rad = np.deg2rad(np.asarray(list(q0_deg), dtype=float).reshape(6))
    target = np.asarray(list(target_pose6), dtype=float).reshape(6)
    lower_rad, upper_rad = _limits_rad()

    converged = False
    it_used = 0
    last_err = np.zeros(6, dtype=float)

    for k in range(opt.max_iters):
        e = _pose_error(target, q_rad)
        pos_norm = float(np.linalg.norm(e[:3]))
        ori_norm = float(np.linalg.norm(e[3:]))
        last_err = e
        it_used = k + 1

        if pos_norm <= opt.tol_pos_mm and ori_norm <= opt.tol_ori_rad:
            converged = True
            break

        J = numerical_pose_jacobian_rad(q_rad)
        jt = J.T
        inv = np.linalg.inv(J @ jt + opt.damping * np.eye(6))
        dq = jt @ (inv @ e)
        q_rad = q_rad + opt.step_scale * dq
        if opt.project_to_joint_limits:
            q_rad = np.clip(q_rad, lower_rad, upper_rad)

    q_deg = np.rad2deg(q_rad)
    return {
        "q_deg": q_deg,
        "iters": it_used,
        "converged": converged,
        "final_error_pose6": last_err,
        "final_pos_err_mm": float(np.linalg.norm(last_err[:3])),
        "final_ori_err_rad": float(np.linalg.norm(last_err[3:])),
    }


def dls_refine(
    q0_deg: Iterable[float],
    target_pose6: Iterable[float],
    options: DLSOptions | None = None,
) -> Dict[str, object]:
    opt = options or DLSOptions()
    q0 = np.asarray(list(q0_deg), dtype=float).reshape(6)
    q_rad = np.deg2rad(q0)
    lower_rad, upper_rad = _limits_rad()

    converged = False
    it_used = 0
    for k in range(opt.max_iters):
        metrics = evaluate_solution_metrics(np.rad2deg(q_rad), target_pose6)
        it_used = k + 1
        if (
            metrics["final_pos_err_mm"] <= opt.tol_pos_mm
            and metrics["final_ori_err_rad"] <= opt.tol_ori_rad
        ):
            converged = True
            break

        J_w = weighted_pose_jacobian(q_rad, orientation_weight=opt.orientation_weight)
        e_w = weighted_pose_error_vector(
            target_pose6=target_pose6,
            q_rad=q_rad,
            orientation_weight=opt.orientation_weight,
        )
        jt = J_w.T
        dq = jt @ np.linalg.solve(J_w @ jt + opt.damping * np.eye(6), e_w)
        q_rad = q_rad + opt.step_scale * dq
        if opt.project_to_joint_limits:
            q_rad = np.clip(q_rad, lower_rad, upper_rad)

    return _finalize_iterative_result(
        q_rad=q_rad,
        target_pose6=target_pose6,
        iters=it_used,
        converged=converged,
        method="dls",
        start_q_deg=q0,
    )


def lbfgsb_refine(
    q0_deg: Iterable[float],
    target_pose6: Iterable[float],
    options: LBFGSBOptions | None = None,
) -> Dict[str, object]:
    opt = options or LBFGSBOptions()
    q0 = np.asarray(list(q0_deg), dtype=float).reshape(6)
    q0_rad = np.deg2rad(q0)
    lower_rad, upper_rad = _limits_rad()
    target = np.asarray(list(target_pose6), dtype=float).reshape(6)

    eval_count = 0

    def objective(q_rad: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal eval_count
        q = np.asarray(q_rad, dtype=float).reshape(6)
        e_w = weighted_pose_error_vector(
            target_pose6=target,
            q_rad=q,
            orientation_weight=opt.orientation_weight,
        )
        J_w = weighted_pose_jacobian(q, orientation_weight=opt.orientation_weight)
        val = float(0.5 * np.dot(e_w, e_w))
        grad = -J_w.T @ e_w
        eval_count += 1
        return val, grad

    res = minimize(
        fun=lambda q: objective(q)[0],
        x0=q0_rad,
        jac=lambda q: objective(q)[1],
        bounds=list(zip(lower_rad.tolist(), upper_rad.tolist())),
        method="L-BFGS-B",
        options={
            "maxiter": int(opt.max_iters),
            "ftol": float(opt.ftol),
            "gtol": float(opt.gtol),
            "maxls": 50,
        },
    )

    q_best_rad = np.asarray(res.x, dtype=float).reshape(6)
    metrics = evaluate_solution_metrics(np.rad2deg(q_best_rad), target_pose6)
    converged = bool(
        metrics["final_pos_err_mm"] <= opt.tol_pos_mm
        and metrics["final_ori_err_rad"] <= opt.tol_ori_rad
    )
    metrics.update(
        {
            "iters": int(getattr(res, "nit", 0)),
            "converged": converged,
            "method": "lbfgsb",
            "start_q_deg": q0.tolist(),
            "weighted_cost": weighted_pose_cost(
                target_pose6=target_pose6,
                q_rad=q_best_rad,
                orientation_weight=opt.orientation_weight,
            ),
            "eval_count": int(eval_count),
            "optimizer_success": bool(res.success),
            "optimizer_message": str(res.message),
        }
    )
    return metrics


def sample_multistart_initial_guesses(
    n_starts: int,
    rng: np.random.Generator,
    include_zero: bool = True,
) -> np.ndarray:
    if n_starts <= 0:
        raise ValueError("n_starts must be positive.")
    guesses = []
    if include_zero:
        guesses.append(np.zeros(6, dtype=float))
    remain = n_starts - len(guesses)
    if remain > 0:
        rand = rng.uniform(JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1], size=(remain, 6))
        guesses.extend(rand)
    return np.asarray(guesses[:n_starts], dtype=float)


def _pick_best_multistart_result(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not results:
        raise ValueError("results must not be empty.")

    def rank_key(item: Dict[str, object]) -> tuple[int, float]:
        success = int(
            bool(item.get("within_joint_limits", False))
            and float(item["final_pos_err_mm"]) <= 1.0
            and float(item["final_ori_err_rad"]) <= 1e-2
        )
        cost = float(
            item.get(
                "weighted_cost",
                float(item["final_pos_err_mm"]) + 200.0 * float(item["final_ori_err_rad"]),
            )
        )
        return (-success, cost)

    return min(results, key=rank_key)


def multistart_dls_refine(
    target_pose6: Iterable[float],
    n_starts: int,
    rng: np.random.Generator,
    options: DLSOptions | None = None,
    include_zero: bool = True,
) -> Dict[str, object]:
    guesses = sample_multistart_initial_guesses(n_starts=n_starts, rng=rng, include_zero=include_zero)
    results = [dls_refine(q0_deg=q0, target_pose6=target_pose6, options=options) for q0 in guesses]
    best = dict(_pick_best_multistart_result(results))
    best["method"] = "multistart_dls"
    best["starts_used"] = int(len(results))
    best["all_start_costs"] = [float(x["weighted_cost"]) for x in results]
    return best


def multistart_lbfgsb_refine(
    target_pose6: Iterable[float],
    n_starts: int,
    rng: np.random.Generator,
    options: LBFGSBOptions | None = None,
    include_zero: bool = True,
) -> Dict[str, object]:
    guesses = sample_multistart_initial_guesses(n_starts=n_starts, rng=rng, include_zero=include_zero)
    results = [lbfgsb_refine(q0_deg=q0, target_pose6=target_pose6, options=options) for q0 in guesses]
    best = dict(_pick_best_multistart_result(results))
    best["method"] = "multistart_lbfgsb"
    best["starts_used"] = int(len(results))
    best["all_start_costs"] = [float(x["weighted_cost"]) for x in results]
    return best
