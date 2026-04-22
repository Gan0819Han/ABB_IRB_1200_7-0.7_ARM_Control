#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Forward kinematics model for ABB_IRB (IRB 1200-7/0.7).

This module follows the same project-facing interfaces used in the previous
xArm workflow so that later migration of training, NR, DLS, and benchmarking
code stays simple.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from robot_config import (
    DH_A_MM,
    DH_ALPHA_DEG,
    DH_D_MM,
    JOINT_LIMITS_DEG,
    ORIENTATION_REPRESENTATION,
    ROBOT_MODEL,
    ROBOT_NAME,
    THETA_OFFSETS_DEG,
)


def _resolve_theta_rad(
    q: Iterable[float],
    input_unit: str = "deg",
    theta_offsets_deg: Sequence[float] | None = None,
) -> np.ndarray:
    q_arr = np.asarray(list(q), dtype=float)
    if q_arr.shape != (6,):
        raise ValueError("Expected 6 joint angles.")
    if input_unit not in ("deg", "rad"):
        raise ValueError("input_unit must be 'deg' or 'rad'.")

    offsets_deg = THETA_OFFSETS_DEG if theta_offsets_deg is None else np.asarray(theta_offsets_deg, dtype=float)
    if offsets_deg.shape != (6,):
        raise ValueError("Expected theta_offsets_deg shape (6,).")

    if input_unit == "deg":
        return np.deg2rad(q_arr + offsets_deg)
    return q_arr + np.deg2rad(offsets_deg)


def _resolve_theta_rad_torch(
    q: torch.Tensor,
    input_unit: str = "deg",
    theta_offsets_deg: Sequence[float] | None = None,
) -> torch.Tensor:
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError("Expected q shape (N, 6).")
    if input_unit not in ("deg", "rad"):
        raise ValueError("input_unit must be 'deg' or 'rad'.")

    offsets_deg = THETA_OFFSETS_DEG if theta_offsets_deg is None else np.asarray(theta_offsets_deg, dtype=float)
    if offsets_deg.shape != (6,):
        raise ValueError("Expected theta_offsets_deg shape (6,).")
    offsets_t = torch.as_tensor(offsets_deg, dtype=q.dtype, device=q.device).view(1, 6)

    if input_unit == "deg":
        return torch.deg2rad(q + offsets_t)
    return q + torch.deg2rad(offsets_t)


def _dh_transform(a_mm: float, alpha_deg: float, d_mm: float, theta_rad: float) -> np.ndarray:
    alpha_rad = math.radians(alpha_deg)
    ct, st = math.cos(theta_rad), math.sin(theta_rad)
    ca, sa = math.cos(alpha_rad), math.sin(alpha_rad)

    return np.array(
        [
            [ct, -st * ca, st * sa, a_mm * ct],
            [st, ct * ca, -ct * sa, a_mm * st],
            [0.0, sa, ca, d_mm],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _dh_transform_torch(
    a_mm: float,
    alpha_deg: float,
    d_mm: float,
    theta_rad: torch.Tensor,
) -> torch.Tensor:
    alpha_rad = math.radians(alpha_deg)
    ct, st = torch.cos(theta_rad), torch.sin(theta_rad)
    ca = torch.tensor(math.cos(alpha_rad), dtype=theta_rad.dtype, device=theta_rad.device)
    sa = torch.tensor(math.sin(alpha_rad), dtype=theta_rad.dtype, device=theta_rad.device)

    n = theta_rad.shape[0]
    T = torch.zeros((n, 4, 4), dtype=theta_rad.dtype, device=theta_rad.device)
    T[:, 0, 0] = ct
    T[:, 0, 1] = -st * ca
    T[:, 0, 2] = st * sa
    T[:, 0, 3] = a_mm * ct
    T[:, 1, 0] = st
    T[:, 1, 1] = ct * ca
    T[:, 1, 2] = -ct * sa
    T[:, 1, 3] = a_mm * st
    T[:, 2, 1] = sa
    T[:, 2, 2] = ca
    T[:, 2, 3] = d_mm
    T[:, 3, 3] = 1.0
    return T


def check_joint_limits_deg(
    q_deg: Iterable[float],
    limits_deg: np.ndarray | None = None,
) -> bool:
    q = np.asarray(list(q_deg), dtype=float)
    if q.shape != (6,):
        raise ValueError("Expected 6 joint angles.")
    limits = JOINT_LIMITS_DEG if limits_deg is None else np.asarray(limits_deg, dtype=float)
    if limits.shape != (6, 2):
        raise ValueError("Expected limits_deg shape (6, 2).")
    return bool(np.all((q >= limits[:, 0]) & (q <= limits[:, 1])))


def fk_abb_irb_all_frames(
    q: Iterable[float],
    input_unit: str = "deg",
    theta_offsets_deg: Sequence[float] | None = None,
    base_transform: np.ndarray | None = None,
) -> List[np.ndarray]:
    """
    Return transforms [T00, T01, ..., T06].
    """
    theta = _resolve_theta_rad(q, input_unit=input_unit, theta_offsets_deg=theta_offsets_deg)
    T = np.eye(4, dtype=float) if base_transform is None else np.asarray(base_transform, dtype=float).copy()
    if T.shape != (4, 4):
        raise ValueError("Expected base_transform shape (4, 4).")

    frames = [T.copy()]
    for i in range(6):
        Ti = _dh_transform(
            a_mm=float(DH_A_MM[i]),
            alpha_deg=float(DH_ALPHA_DEG[i]),
            d_mm=float(DH_D_MM[i]),
            theta_rad=float(theta[i]),
        )
        T = T @ Ti
        frames.append(T.copy())
    return frames


def fk_abb_irb(
    q: Iterable[float],
    input_unit: str = "deg",
    theta_offsets_deg: Sequence[float] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        T06: homogeneous transform, shape (4, 4)
        p: end-effector position, shape (3,)
        R: rotation matrix, shape (3, 3)
    """
    frames = fk_abb_irb_all_frames(q, input_unit=input_unit, theta_offsets_deg=theta_offsets_deg)
    T06 = frames[-1]
    p = T06[:3, 3].copy()
    R = T06[:3, :3].copy()
    return T06, p, R


def fk_abb_irb_joint_points(
    q: Iterable[float],
    input_unit: str = "deg",
    theta_offsets_deg: Sequence[float] | None = None,
    base_pos_xyz_mm: Iterable[float] | None = None,
) -> np.ndarray:
    """
    Return frame origins [base, joint1, ..., joint6/end-effector], shape (7, 3).
    """
    base_transform = np.eye(4, dtype=float)
    if base_pos_xyz_mm is not None:
        base_pos = np.asarray(list(base_pos_xyz_mm), dtype=float)
        if base_pos.shape != (3,):
            raise ValueError("Expected base_pos_xyz_mm shape (3,).")
        base_transform[:3, 3] = base_pos

    frames = fk_abb_irb_all_frames(
        q,
        input_unit=input_unit,
        theta_offsets_deg=theta_offsets_deg,
        base_transform=base_transform,
    )
    return np.asarray([T[:3, 3].copy() for T in frames], dtype=float)


def wrist_center_from_q(
    q: Iterable[float],
    input_unit: str = "deg",
    theta_offsets_deg: Sequence[float] | None = None,
) -> np.ndarray:
    """
    Mechanical wrist center for workspace comparison.
    In the current DH chain this is the origin of frame 4 (after T04).
    """
    frames = fk_abb_irb_all_frames(q, input_unit=input_unit, theta_offsets_deg=theta_offsets_deg)
    return frames[4][:3, 3].copy()


def rot_to_zyx_euler_rad(R: np.ndarray) -> np.ndarray:
    if R.shape != (3, 3):
        raise ValueError("Expected R shape (3, 3).")

    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-9

    if not singular:
        phi = math.atan2(R[1, 0], R[0, 0])
        theta = math.atan2(-R[2, 0], sy)
        psi = math.atan2(R[2, 1], R[2, 2])
    else:
        phi = math.atan2(-R[0, 1], R[1, 1])
        theta = math.atan2(-R[2, 0], sy)
        psi = 0.0
    return np.array([phi, theta, psi], dtype=float)


def rot_to_zyx_euler_rad_torch(R: torch.Tensor) -> torch.Tensor:
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError("Expected R shape (N, 3, 3).")

    sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    singular = sy < 1e-9

    phi = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    theta = torch.atan2(-R[:, 2, 0], sy)
    psi = torch.atan2(R[:, 2, 1], R[:, 2, 2])

    if singular.any():
        phi_s = torch.atan2(-R[singular, 0, 1], R[singular, 1, 1])
        theta_s = torch.atan2(-R[singular, 2, 0], sy[singular])
        phi = phi.clone()
        theta = theta.clone()
        psi = psi.clone()
        phi[singular] = phi_s
        theta[singular] = theta_s
        psi[singular] = 0.0

    return torch.stack([phi, theta, psi], dim=1)


def zyx_euler_to_rot(phi: float, theta: float, psi: float) -> np.ndarray:
    c1, s1 = math.cos(phi), math.sin(phi)
    c2, s2 = math.cos(theta), math.sin(theta)
    c3, s3 = math.cos(psi), math.sin(psi)

    rz = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    ry = np.array([[c2, 0.0, s2], [0.0, 1.0, 0.0], [-s2, 0.0, c2]], dtype=float)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c3, -s3], [0.0, s3, c3]], dtype=float)
    return rz @ ry @ rx


def pose6_from_q(
    q: Iterable[float],
    input_unit: str = "deg",
    theta_offsets_deg: Sequence[float] | None = None,
) -> np.ndarray:
    """
    Pose representation:
    [x_mm, y_mm, z_mm, phi_rad, theta_rad, psi_rad]
    """
    _, p, R = fk_abb_irb(q, input_unit=input_unit, theta_offsets_deg=theta_offsets_deg)
    eul = rot_to_zyx_euler_rad(R)
    return np.concatenate([p, eul], axis=0)


def fk_abb_irb_torch_batch(
    q: torch.Tensor,
    input_unit: str = "deg",
    theta_offsets_deg: Sequence[float] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta = _resolve_theta_rad_torch(q, input_unit=input_unit, theta_offsets_deg=theta_offsets_deg)
    n = theta.shape[0]
    T = torch.eye(4, dtype=theta.dtype, device=theta.device).unsqueeze(0).repeat(n, 1, 1)
    for i in range(6):
        Ti = _dh_transform_torch(
            a_mm=float(DH_A_MM[i]),
            alpha_deg=float(DH_ALPHA_DEG[i]),
            d_mm=float(DH_D_MM[i]),
            theta_rad=theta[:, i],
        )
        T = torch.bmm(T, Ti)
    p = T[:, :3, 3].contiguous()
    R = T[:, :3, :3].contiguous()
    return T, p, R


def pose6_from_q_torch_batch(
    q: torch.Tensor,
    input_unit: str = "deg",
    theta_offsets_deg: Sequence[float] | None = None,
) -> torch.Tensor:
    _, p, R = fk_abb_irb_torch_batch(q, input_unit=input_unit, theta_offsets_deg=theta_offsets_deg)
    eul = rot_to_zyx_euler_rad_torch(R)
    return torch.cat([p, eul], dim=1)


def numerical_pose_jacobian_rad(
    q_rad: np.ndarray,
    h: float = 1e-6,
    theta_offsets_deg: Sequence[float] | None = None,
) -> np.ndarray:
    """
    Numerical Jacobian of pose6 wrt q(rad), shape (6, 6).
    """
    q = np.asarray(q_rad, dtype=float).reshape(6)
    J = np.zeros((6, 6), dtype=float)

    for i in range(6):
        qp = q.copy()
        qm = q.copy()
        qp[i] += h
        qm[i] -= h
        fp = pose6_from_q(qp, input_unit="rad", theta_offsets_deg=theta_offsets_deg)
        fm = pose6_from_q(qm, input_unit="rad", theta_offsets_deg=theta_offsets_deg)
        J[:, i] = (fp - fm) / (2.0 * h)
    return J


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    q_demo = np.zeros(6, dtype=float)
    print(f"robot={ROBOT_NAME}, model={ROBOT_MODEL}, euler={ORIENTATION_REPRESENTATION}")
    print("q_demo (deg):", q_demo)
    print("within limits:", check_joint_limits_deg(q_demo))

    T06, p, R = fk_abb_irb(q_demo, input_unit="deg")
    print("\nT06:\n", T06)
    print("\nposition_mm [x, y, z]:", p)
    print("\nrotation R:\n", R)
    print("\npose6 [x, y, z, phi, theta, psi]:\n", pose6_from_q(q_demo, input_unit="deg"))
