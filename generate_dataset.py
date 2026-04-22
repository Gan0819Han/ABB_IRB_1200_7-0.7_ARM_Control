#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate FK dataset for ABB_IRB.

Default strategy:
1. sample joint angles uniformly within current project joint limits
2. map each sample to end-effector pose by forward kinematics
3. save CSV / NPZ / metadata files under the target directory
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from fk_model import fk_abb_irb_torch_batch, rot_to_zyx_euler_rad_torch
from naming import load_naming_config, make_base_name, make_full_filenames
from robot_config import JOINT_LIMITS_DEG, ROBOT_MODEL, ROBOT_NAME, THETA_OFFSETS_DEG


def resolve_feature_device(mode: str) -> torch.device:
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("feature_device=cuda requested, but CUDA is not available.")
    return torch.device(mode)


def build_dataset(
    n_samples: int,
    seed: int | None = None,
    feature_device: torch.device | None = None,
    feature_batch_size: int = 65536,
) -> Dict[str, np.ndarray]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be positive.")

    rng = np.random.default_rng(seed)
    lower = JOINT_LIMITS_DEG[:, 0]
    upper = JOINT_LIMITS_DEG[:, 1]
    q_deg = rng.uniform(lower, upper, size=(n_samples, 6)).astype(np.float32)

    position_mm = np.zeros((n_samples, 3), dtype=np.float32)
    rotation = np.zeros((n_samples, 3, 3), dtype=np.float32)
    euler_rad = np.zeros((n_samples, 3), dtype=np.float32)
    T06_all = np.zeros((n_samples, 4, 4), dtype=np.float32)

    device = torch.device("cpu") if feature_device is None else feature_device
    with torch.no_grad():
        for i in range(0, n_samples, feature_batch_size):
            j = min(i + feature_batch_size, n_samples)
            qb = torch.from_numpy(q_deg[i:j]).to(
                device,
                dtype=torch.float32,
                non_blocking=(device.type == "cuda"),
            )
            T06, p, R = fk_abb_irb_torch_batch(qb, input_unit="deg")
            eul = rot_to_zyx_euler_rad_torch(R)
            T06_all[i:j] = T06.float().cpu().numpy()
            position_mm[i:j] = p.float().cpu().numpy()
            rotation[i:j] = R.float().cpu().numpy()
            euler_rad[i:j] = eul.float().cpu().numpy()

    return {
        "q_deg": q_deg,
        "position_mm": position_mm,
        "rotation": rotation,
        "euler_rad_zyx": euler_rad,
        "T06": T06_all,
    }


def save_dataset(
    dataset: Dict[str, np.ndarray],
    out_dir: str,
    base_name: str,
    overwrite: bool = False,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    names = make_full_filenames(base_name)
    csv_path = os.path.join(out_dir, names["csv"])
    npz_path = os.path.join(out_dir, names["npz"])
    meta_path = os.path.join(out_dir, names["meta"])

    if not overwrite:
        for p in (csv_path, npz_path, meta_path):
            if os.path.exists(p):
                raise FileExistsError(f"File already exists: {p}. Use --overwrite to replace.")

    q_deg = dataset["q_deg"]
    pos = dataset["position_mm"]
    rot = dataset["rotation"]
    eul = dataset["euler_rad_zyx"]

    header = [
        "q1_deg", "q2_deg", "q3_deg", "q4_deg", "q5_deg", "q6_deg",
        "x_mm", "y_mm", "z_mm",
        "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33",
        "phi_rad", "theta_rad", "psi_rad",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(q_deg.shape[0]):
            row = list(map(float, q_deg[i])) + list(map(float, pos[i])) + list(map(float, rot[i].reshape(-1))) + list(map(float, eul[i]))
            writer.writerow(row)

    np.savez_compressed(
        npz_path,
        q_deg=dataset["q_deg"],
        position_mm=dataset["position_mm"],
        rotation=dataset["rotation"],
        euler_rad_zyx=dataset["euler_rad_zyx"],
        T06=dataset["T06"],
    )

    meta = {
        "robot_name": ROBOT_NAME,
        "robot_model": ROBOT_MODEL,
        "base_name": base_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": int(q_deg.shape[0]),
        "sampling": "uniform random within current ABB_IRB project joint limits",
        "joint_limits_deg": JOINT_LIMITS_DEG.tolist(),
        "theta_offsets_deg": THETA_OFFSETS_DEG.tolist(),
        "pose_orientation": "ZYX euler (yaw-pitch-roll) in radians",
        "units": {"position": "mm", "angles": "deg/rad"},
        "files": {
            "csv": os.path.basename(csv_path),
            "npz": os.path.basename(npz_path),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"csv": csv_path, "npz": npz_path, "meta": meta_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FK dataset for ABB_IRB.")
    parser.add_argument("--n_samples", type=int, default=4800, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out_dir", type=str, default="data", help="Output directory.")
    parser.add_argument(
        "--naming_config",
        type=str,
        default="naming_config.json",
        help="Naming config JSON path.",
    )
    parser.add_argument(
        "--name_base",
        type=str,
        default="",
        help="Optional explicit base name. If empty, generated from naming config.",
    )
    parser.add_argument(
        "--append_timestamp",
        action="store_true",
        help="Append timestamp to base name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files with the same name.",
    )
    parser.add_argument(
        "--feature_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for batched FK generation.",
    )
    parser.add_argument(
        "--feature_batch_size",
        type=int,
        default=65536,
        help="Batch size for batched FK generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_naming_config(args.naming_config)
    base_name = args.name_base.strip() or make_base_name(args.n_samples, args.seed, cfg)
    if args.append_timestamp:
        base_name = f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    feature_device = resolve_feature_device(args.feature_device)
    dataset = build_dataset(
        n_samples=args.n_samples,
        seed=args.seed,
        feature_device=feature_device,
        feature_batch_size=args.feature_batch_size,
    )
    paths = save_dataset(
        dataset,
        out_dir=args.out_dir,
        base_name=base_name,
        overwrite=args.overwrite,
    )

    print(f"Robot: {ROBOT_NAME} ({ROBOT_MODEL})")
    print(f"Base name: {base_name}")
    print(f"Generated samples: {args.n_samples}")
    print(f"Feature device: {feature_device}")
    print(f"CSV:  {paths['csv']}")
    print(f"NPZ:  {paths['npz']}")
    print(f"META: {paths['meta']}")


if __name__ == "__main__":
    main()
