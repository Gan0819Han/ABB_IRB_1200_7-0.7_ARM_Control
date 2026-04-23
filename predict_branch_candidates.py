#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from abb_nn.branch_models import build_branch_classifier_variant
from abb_nn.branching import (
    BRANCH_HEAD_DIMS,
    branch_label_to_name,
    branch_to_subspace_map,
    encode_branch_index,
)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def safe_torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_pose(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 6:
        raise ValueError("--pose must be x_mm,y_mm,z_mm,phi_rad,theta_rad,psi_rad")
    return np.array(vals, dtype=np.float32).reshape(1, 6)


def apply_normalizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict hierarchical ABB branch candidates.")
    parser.add_argument("--pose", type=str, required=True, help="x_mm,y_mm,z_mm,phi_rad,theta_rad,psi_rad")
    parser.add_argument(
        "--branch_meta",
        type=str,
        default="artifacts/branch_classification_system/metadata.json",
    )
    parser.add_argument("--topk_shoulder", type=int, default=1)
    parser.add_argument("--topk_elbow", type=int, default=1)
    parser.add_argument("--topk_wrist", type=int, default=2)
    parser.add_argument("--max_branch_candidates", type=int, default=8)
    parser.add_argument("--out_json", type=str, default="", help="Optional path to save result JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.topk_shoulder <= 0 or args.topk_elbow <= 0 or args.topk_wrist <= 0:
        raise ValueError("top-k for each head must be >= 1.")
    if args.max_branch_candidates <= 0:
        raise ValueError("--max_branch_candidates must be >= 1.")

    pose = parse_pose(args.pose)
    branch_meta_path = Path(args.branch_meta)
    branch_meta = load_json(branch_meta_path)
    mean = np.array(branch_meta["normalizer"]["mean"], dtype=np.float32).reshape(1, -1)
    std = np.array(branch_meta["normalizer"]["std"], dtype=np.float32).reshape(1, -1)
    x_norm = apply_normalizer(pose.astype(np.float32), mean, std)
    xt = torch.from_numpy(x_norm)

    branch_scores: Dict[int, float] = {}
    per_model_candidates = []
    for item in branch_meta["models"]:
        ckpt = safe_torch_load(branch_meta_path.parent / item["file"])
        model = build_branch_classifier_variant(
            variant=int(item["variant"]),
            input_dim=6,
            head_dims=ckpt["branch_head_dims"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            logits_shoulder, logits_elbow, logits_wrist = model(xt)
            logp_shoulder = torch.log_softmax(logits_shoulder, dim=1)
            logp_elbow = torch.log_softmax(logits_elbow, dim=1)
            logp_wrist = torch.log_softmax(logits_wrist, dim=1)

        k_shoulder = min(args.topk_shoulder, BRANCH_HEAD_DIMS[0])
        k_elbow = min(args.topk_elbow, BRANCH_HEAD_DIMS[1])
        k_wrist = min(args.topk_wrist, BRANCH_HEAD_DIMS[2])

        shoulder_ids = torch.topk(logp_shoulder, k=k_shoulder, dim=1).indices.reshape(-1).tolist()
        elbow_ids = torch.topk(logp_elbow, k=k_elbow, dim=1).indices.reshape(-1).tolist()
        wrist_ids = torch.topk(logp_wrist, k=k_wrist, dim=1).indices.reshape(-1).tolist()

        local_candidates = []
        for shoulder, elbow, wrist in itertools.product(shoulder_ids, elbow_ids, wrist_ids):
            label = encode_branch_index((shoulder, elbow, wrist))
            score = float(
                logp_shoulder[0, shoulder].item()
                + logp_elbow[0, elbow].item()
                + logp_wrist[0, wrist].item()
            )
            branch_scores[label] = max(branch_scores.get(label, float("-inf")), score)
            local_candidates.append(
                {
                    "branch_label": int(label),
                    "branch_name": branch_label_to_name(label),
                    "score": score,
                }
            )
        local_candidates.sort(key=lambda x: x["score"], reverse=True)
        per_model_candidates.append(
            {
                "variant": int(item["variant"]),
                "candidates": local_candidates,
            }
        )

    sorted_labels = sorted(branch_scores.items(), key=lambda kv: kv[1], reverse=True)
    sorted_labels = sorted_labels[:args.max_branch_candidates]
    subspace_mapping = branch_to_subspace_map(branch_meta["segment_profile"])

    candidate_branches = []
    candidate_subspaces_union = set()
    for label, score in sorted_labels:
        compatible_subspaces = subspace_mapping[int(label)]
        candidate_subspaces_union.update(compatible_subspaces)
        candidate_branches.append(
            {
                "branch_label": int(label),
                "branch_name": branch_label_to_name(int(label)),
                "score": float(score),
                "compatible_subspaces": [int(x) for x in compatible_subspaces],
            }
        )

    result = {
        "target_pose6": pose.reshape(-1).tolist(),
        "branch_profile": branch_meta["branch_profile"],
        "segment_profile": branch_meta["segment_profile"],
        "topk": {
            "shoulder": int(args.topk_shoulder),
            "elbow": int(args.topk_elbow),
            "wrist": int(args.topk_wrist),
        },
        "per_model_candidates": per_model_candidates,
        "candidate_branches": candidate_branches,
        "candidate_subspaces_union": sorted(int(x) for x in candidate_subspaces_union),
    }

    if args.out_json.strip():
        save_json(Path(args.out_json), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
