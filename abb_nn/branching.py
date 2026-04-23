#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .subspace import decode_subspace_label, get_joint_bins, get_segments, get_subspace_count


BRANCH_PROFILE_NAME = "coarse12"
BRANCH_HEAD_NAMES = ("shoulder", "elbow", "wrist")
BRANCH_HEAD_DIMS = (2, 2, 3)
BRANCH_COUNT = int(np.prod(BRANCH_HEAD_DIMS))

SHOULDER_CLASS_NAMES = ("shoulder_negative", "shoulder_positive")
ELBOW_CLASS_NAMES = ("elbow_low", "elbow_high")
WRIST_CLASS_NAMES = ("wrist_negative", "wrist_middle", "wrist_positive")

Q1_SPLIT_DEG = 0.0
Q3_SPLIT_DEG = -65.0
Q5_SPLIT_EDGES_DEG = (-45.0, 45.0)
FINE_JOINT_INDICES = (1, 3, 5)
FINE_JOINT_NAMES = ("q2_bin", "q4_bin", "q6_bin")


def assign_branch_head_labels(q_deg: np.ndarray) -> np.ndarray:
    """
    Assign coarse branch labels for each joint vector.
    Output shape: (N, 3) = [shoulder, elbow, wrist]
    """
    q = np.asarray(q_deg, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError("Expected q_deg shape (N, 6).")

    shoulder = (q[:, 0] >= Q1_SPLIT_DEG).astype(np.int64)
    elbow = (q[:, 2] >= Q3_SPLIT_DEG).astype(np.int64)
    wrist = np.searchsorted(
        np.asarray(Q5_SPLIT_EDGES_DEG, dtype=float),
        q[:, 4],
        side="right",
    ).astype(np.int64)
    return np.stack([shoulder, elbow, wrist], axis=1)


def assign_branch_labels(q_deg: np.ndarray) -> np.ndarray:
    heads = assign_branch_head_labels(q_deg)
    labels = heads[:, 0].copy()
    for j in range(1, len(BRANCH_HEAD_DIMS)):
        labels = labels * BRANCH_HEAD_DIMS[j] + heads[:, j]
    return labels.astype(np.int64)


def encode_branch_index(indices: Iterable[int]) -> int:
    ids = list(indices)
    if len(ids) != len(BRANCH_HEAD_DIMS):
        raise ValueError(f"Expected {len(BRANCH_HEAD_DIMS)} indices.")
    for idx, dim in zip(ids, BRANCH_HEAD_DIMS):
        if idx < 0 or idx >= dim:
            raise ValueError(f"Branch index {idx} out of range [0, {dim - 1}].")
    label = ids[0]
    for j in range(1, len(BRANCH_HEAD_DIMS)):
        label = label * BRANCH_HEAD_DIMS[j] + ids[j]
    return int(label)


def decode_branch_label(label: int) -> Tuple[int, int, int]:
    if label < 0 or label >= BRANCH_COUNT:
        raise ValueError(f"Branch label out of range [0, {BRANCH_COUNT - 1}].")
    out = [0] * len(BRANCH_HEAD_DIMS)
    rem = int(label)
    for j in range(len(BRANCH_HEAD_DIMS) - 1, -1, -1):
        out[j] = rem % BRANCH_HEAD_DIMS[j]
        rem //= BRANCH_HEAD_DIMS[j]
    return tuple(out)  # type: ignore[return-value]


def branch_label_to_name(label: int) -> str:
    shoulder, elbow, wrist = decode_branch_label(label)
    return (
        f"{SHOULDER_CLASS_NAMES[shoulder]}|"
        f"{ELBOW_CLASS_NAMES[elbow]}|"
        f"{WRIST_CLASS_NAMES[wrist]}"
    )


def branch_indices_to_name(indices: Sequence[int]) -> str:
    return branch_label_to_name(encode_branch_index(indices))


def subspace_to_branch_label(
    subspace_label: int,
    segment_profile: str = "abb_strict",
) -> int:
    joint_bin_indices = decode_subspace_label(subspace_label, profile=segment_profile)
    return encode_branch_index(
        (
            joint_bin_indices[0],  # q1 -> shoulder
            joint_bin_indices[2],  # q3 -> elbow
            joint_bin_indices[4],  # q5 -> wrist
        )
    )


def branch_to_subspace_map(segment_profile: str = "abb_strict") -> Dict[int, List[int]]:
    mapping: Dict[int, List[int]] = {label: [] for label in range(BRANCH_COUNT)}
    for subspace_label in range(get_subspace_count(segment_profile)):
        mapping[subspace_to_branch_label(subspace_label, segment_profile)].append(subspace_label)
    return mapping


def get_fine_bins(segment_profile: str = "abb_strict") -> List[int]:
    joint_bins = get_joint_bins(segment_profile)
    return [joint_bins[j] for j in FINE_JOINT_INDICES]


def get_fine_class_count(segment_profile: str = "abb_strict") -> int:
    return int(np.prod(get_fine_bins(segment_profile)))


def encode_fine_index(indices: Iterable[int], segment_profile: str = "abb_strict") -> int:
    ids = list(indices)
    bins = get_fine_bins(segment_profile)
    if len(ids) != len(bins):
        raise ValueError(f"Expected {len(bins)} fine indices.")
    for idx, dim in zip(ids, bins):
        if idx < 0 or idx >= dim:
            raise ValueError(f"Fine index {idx} out of range [0, {dim - 1}].")
    label = ids[0]
    for j in range(1, len(bins)):
        label = label * bins[j] + ids[j]
    return int(label)


def decode_fine_label(label: int, segment_profile: str = "abb_strict") -> Tuple[int, int, int]:
    bins = get_fine_bins(segment_profile)
    count = int(np.prod(bins))
    if label < 0 or label >= count:
        raise ValueError(f"Fine label out of range [0, {count - 1}].")
    out = [0] * len(bins)
    rem = int(label)
    for j in range(len(bins) - 1, -1, -1):
        out[j] = rem % bins[j]
        rem //= bins[j]
    return tuple(out)  # type: ignore[return-value]


def assign_fine_labels(q_deg: np.ndarray, segment_profile: str = "abb_strict") -> np.ndarray:
    q = np.asarray(q_deg, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError("Expected q_deg shape (N, 6).")
    segments_all = get_segments(segment_profile)
    bins = get_fine_bins(segment_profile)
    idx_list = []
    for local_idx, joint_idx in enumerate(FINE_JOINT_INDICES):
        segments = segments_all[joint_idx]
        lower, upper = segments[0][0], segments[-1][1]
        values = q[:, joint_idx]
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError(f"Joint values out of segmentation range [{lower}, {upper}].")
        edges = np.array([seg[1] for seg in segments[:-1]], dtype=float)
        idx_list.append(np.searchsorted(edges, values, side="right"))
    idx_arr = np.stack(idx_list, axis=1)
    labels = idx_arr[:, 0].copy()
    for j in range(1, len(bins)):
        labels = labels * bins[j] + idx_arr[:, j]
    return labels.astype(np.int64)


def subspace_to_fine_label(subspace_label: int, segment_profile: str = "abb_strict") -> int:
    joint_bin_indices = decode_subspace_label(subspace_label, profile=segment_profile)
    return encode_fine_index(
        (
            joint_bin_indices[FINE_JOINT_INDICES[0]],
            joint_bin_indices[FINE_JOINT_INDICES[1]],
            joint_bin_indices[FINE_JOINT_INDICES[2]],
        ),
        segment_profile=segment_profile,
    )


def branch_fine_to_subspace_label(
    branch_label: int,
    fine_label: int,
    segment_profile: str = "abb_strict",
) -> int:
    branch_ids = decode_branch_label(branch_label)
    fine_ids = decode_fine_label(fine_label, segment_profile=segment_profile)
    joint_ids = [0] * 6
    joint_ids[0] = branch_ids[0]
    joint_ids[2] = branch_ids[1]
    joint_ids[4] = branch_ids[2]
    joint_ids[FINE_JOINT_INDICES[0]] = fine_ids[0]
    joint_ids[FINE_JOINT_INDICES[1]] = fine_ids[1]
    joint_ids[FINE_JOINT_INDICES[2]] = fine_ids[2]
    joint_bins = get_joint_bins(segment_profile)
    label = joint_ids[0]
    for j in range(1, len(joint_bins)):
        label = label * joint_bins[j] + joint_ids[j]
    return int(label)
