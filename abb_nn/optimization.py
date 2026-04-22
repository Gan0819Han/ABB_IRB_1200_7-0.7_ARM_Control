#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np

from fk_model import JOINT_LIMITS_DEG, numerical_pose_jacobian_rad, pose6_from_q


@dataclass
class NROptions:
    max_iters: int = 40
    tol_pos_mm: float = 1e-3
    tol_ori_rad: float = 1e-3
    damping: float = 1e-5
    step_scale: float = 1.0
    project_to_joint_limits: bool = True


def _pose_error(target_pose6: np.ndarray, q_rad: np.ndarray) -> np.ndarray:
    cur = pose6_from_q(q_rad, input_unit="rad")
    err = target_pose6 - cur
    err[3:] = (err[3:] + np.pi) % (2.0 * np.pi) - np.pi
    return err


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
    lower_rad = np.deg2rad(JOINT_LIMITS_DEG[:, 0])
    upper_rad = np.deg2rad(JOINT_LIMITS_DEG[:, 1])

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
