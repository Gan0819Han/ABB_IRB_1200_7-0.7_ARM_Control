#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from abb_nn.data_utils import save_json
from abb_nn.subspace import (
    get_segments,
    get_subspace_count,
    sample_q_in_subspace_deg,
    subspace_bounds_deg,
)
from fk_model import pose6_from_q_torch_batch
from robot_config import ROBOT_MODEL, ROBOT_NAME, THETA_OFFSETS_DEG


def parse_int_list(text: str) -> List[int]:
    if not text.strip():
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def resolve_feature_device(mode: str) -> torch.device:
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("feature_device=cuda requested, but CUDA is not available.")
    return torch.device(mode)


def build_pose_features(
    q_deg: np.ndarray,
    feature_device: torch.device,
    feature_batch_size: int,
) -> np.ndarray:
    q = np.asarray(q_deg, dtype=np.float32)
    x = np.zeros((q.shape[0], 6), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, q.shape[0], feature_batch_size):
            j = min(i + feature_batch_size, q.shape[0])
            qb = torch.from_numpy(q[i:j]).to(
                feature_device,
                dtype=torch.float32,
                non_blocking=(feature_device.type == "cuda"),
            )
            pose = pose6_from_q_torch_batch(qb, input_unit="deg")
            x[i:j] = pose.float().cpu().numpy()
    return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reference FK samples for each ABB subspace without retraining models."
    )
    parser.add_argument(
        "--segment_profile",
        type=str,
        default="abb_strict",
        choices=["abb_simplified", "abb_strict", "simplified", "strict"],
        help="Subspace segmentation profile.",
    )
    parser.add_argument(
        "--samples_per_subspace",
        type=int,
        default=512,
        help="Reference sample count exported for each subspace.",
    )
    parser.add_argument(
        "--subspaces",
        type=str,
        default="",
        help="Optional comma-separated subspace IDs. Empty means all subspaces.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Optional export directory. Empty means auto-generate under data/.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--feature_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for FK feature generation.",
    )
    parser.add_argument(
        "--feature_batch_size",
        type=int,
        default=65536,
        help="Batch size used when converting q to pose6 by FK.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow exporting into a non-empty target directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_subspace <= 0:
        raise ValueError("--samples_per_subspace must be positive.")
    if args.feature_batch_size <= 0:
        raise ValueError("--feature_batch_size must be positive.")

    rng = np.random.default_rng(args.seed)
    profile = args.segment_profile
    segments = get_segments(profile)
    subspace_count = get_subspace_count(profile)
    subspaces = parse_int_list(args.subspaces)
    if not subspaces:
        subspaces = list(range(subspace_count))

    for sid in subspaces:
        if sid < 0 or sid >= subspace_count:
            raise ValueError(f"Invalid subspace id: {sid}")

    if args.out_dir.strip():
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(
            f"data/subspace_reference_{profile}_samples{args.samples_per_subspace}_seed{args.seed}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}. "
            "Use --overwrite or choose another --out_dir."
        )

    feature_device = resolve_feature_device(args.feature_device)
    exported_entries: List[Dict[str, object]] = []

    for sid in subspaces:
        q_deg = sample_q_in_subspace_deg(
            sid,
            args.samples_per_subspace,
            rng,
            profile=profile,
        ).astype(np.float32)
        pose6 = build_pose_features(q_deg, feature_device, args.feature_batch_size).astype(np.float32)
        bounds = subspace_bounds_deg(sid, profile=profile).astype(np.float32)

        file_name = f"subspace_{sid:03d}_reference.npz"
        file_path = out_dir / file_name
        np.savez_compressed(
            file_path,
            q_deg=q_deg,
            pose6=pose6,
            bounds_deg=bounds,
        )

        exported_entries.append(
            {
                "subspace_id": int(sid),
                "file": file_name,
                "samples": int(args.samples_per_subspace),
                "bounds_deg": bounds.tolist(),
            }
        )
        print(
            f"[subspace {sid:03d}] "
            f"samples={args.samples_per_subspace} saved={file_name}"
        )

    metadata = {
        "export_type": "abb_subspace_reference_samples",
        "robot_name": ROBOT_NAME,
        "robot_model": ROBOT_MODEL,
        "segment_profile": profile,
        "subspace_count": int(subspace_count),
        "exported_subspaces": len(exported_entries),
        "samples_per_subspace": int(args.samples_per_subspace),
        "seed": int(args.seed),
        "fields": {
            "q_deg": ["q1_deg", "q2_deg", "q3_deg", "q4_deg", "q5_deg", "q6_deg"],
            "pose6": ["x_mm", "y_mm", "z_mm", "phi_rad", "theta_rad", "psi_rad"],
            "bounds_deg": "shape (6, 2), lower/upper angle bounds for the exported subspace",
        },
        "theta_offsets_deg": THETA_OFFSETS_DEG.tolist(),
        "joint_segments_deg": segments,
        "feature_device": str(feature_device),
        "feature_batch_size": int(args.feature_batch_size),
        "files": exported_entries,
    }
    save_json(out_dir / "metadata.json", metadata)
    print(f"Saved metadata: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
