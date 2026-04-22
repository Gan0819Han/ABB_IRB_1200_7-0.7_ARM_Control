#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


# ABB simplified profile discussed in the design notes:
# q1: 2, q2: 2, q3: 2, q4: 2, q5: 3, q6: 2 => 96 subspaces
ABB_SIMPLIFIED_JOINT_SEGMENTS_DEG: List[List[Tuple[float, float]]] = [
    [(-170.0, 0.0), (0.0, 170.0)],  # q1
    [(-100.0, 20.0), (20.0, 135.0)],  # q2
    [(-200.0, -65.0), (-65.0, 70.0)],  # q3
    [(-270.0, 0.0), (0.0, 270.0)],  # q4
    [(-130.0, -45.0), (-45.0, 45.0), (45.0, 130.0)],  # q5
    [(-180.0, 0.0), (0.0, 180.0)],  # q6
]


# ABB strict profile:
# same as simplified except q4 has 4 bins => 192 subspaces
ABB_STRICT_JOINT_SEGMENTS_DEG: List[List[Tuple[float, float]]] = [
    [(-170.0, 0.0), (0.0, 170.0)],  # q1
    [(-100.0, 20.0), (20.0, 135.0)],  # q2
    [(-200.0, -65.0), (-65.0, 70.0)],  # q3
    [(-270.0, -135.0), (-135.0, 0.0), (0.0, 135.0), (135.0, 270.0)],  # q4
    [(-130.0, -45.0), (-45.0, 45.0), (45.0, 130.0)],  # q5
    [(-180.0, 0.0), (0.0, 180.0)],  # q6
]


SEGMENT_PROFILES = {
    "abb_simplified": ABB_SIMPLIFIED_JOINT_SEGMENTS_DEG,
    "abb_strict": ABB_STRICT_JOINT_SEGMENTS_DEG,
    # aliases kept for easier migration from the xArm workflow
    "simplified": ABB_SIMPLIFIED_JOINT_SEGMENTS_DEG,
    "strict": ABB_STRICT_JOINT_SEGMENTS_DEG,
}


DEFAULT_SEGMENT_PROFILE = "abb_strict"
JOINT_SEGMENTS_DEG = ABB_STRICT_JOINT_SEGMENTS_DEG
JOINT_BINS = [len(x) for x in JOINT_SEGMENTS_DEG]
SUBSPACE_COUNT = int(np.prod(JOINT_BINS))


def get_segments(profile: str = DEFAULT_SEGMENT_PROFILE) -> List[List[Tuple[float, float]]]:
    if profile not in SEGMENT_PROFILES:
        raise ValueError(
            f"Unknown segment profile: {profile}. "
            f"choices={list(SEGMENT_PROFILES.keys())}"
        )
    return SEGMENT_PROFILES[profile]


def get_joint_bins(profile: str = DEFAULT_SEGMENT_PROFILE) -> List[int]:
    return [len(x) for x in get_segments(profile)]


def get_subspace_count(profile: str = DEFAULT_SEGMENT_PROFILE) -> int:
    return int(np.prod(get_joint_bins(profile)))


def _joint_bin_indices(values: np.ndarray, segments: Sequence[Tuple[float, float]]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lower, upper = segments[0][0], segments[-1][1]
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError(f"Joint values out of segmentation range [{lower}, {upper}].")
    edges = np.array([seg[1] for seg in segments[:-1]], dtype=float)
    return np.searchsorted(edges, values, side="right")


def assign_subspace_labels(q_deg: np.ndarray, profile: str = DEFAULT_SEGMENT_PROFILE) -> np.ndarray:
    """
    Assign one subspace label per joint vector.
    Input:
        q_deg: shape (N, 6)
    Output:
        labels: shape (N,), in [0, subspace_count - 1]
    """
    q = np.asarray(q_deg, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError("Expected q_deg shape (N, 6).")

    segments_all = get_segments(profile)
    bins = get_joint_bins(profile)

    idx_list = []
    for j in range(6):
        idx_list.append(_joint_bin_indices(q[:, j], segments_all[j]))
    idx_arr = np.stack(idx_list, axis=1)

    labels = idx_arr[:, 0].copy()
    for j in range(1, 6):
        labels = labels * bins[j] + idx_arr[:, j]
    return labels.astype(np.int64)


def encode_subspace_index(indices: Iterable[int], profile: str = DEFAULT_SEGMENT_PROFILE) -> int:
    ids = list(indices)
    if len(ids) != 6:
        raise ValueError("Expected 6 indices.")
    bins = get_joint_bins(profile)
    label = ids[0]
    for j in range(1, 6):
        label = label * bins[j] + ids[j]
    return int(label)


def decode_subspace_label(label: int, profile: str = DEFAULT_SEGMENT_PROFILE) -> Tuple[int, int, int, int, int, int]:
    bins = get_joint_bins(profile)
    count = get_subspace_count(profile)
    if label < 0 or label >= count:
        raise ValueError(f"Label out of range [0, {count - 1}].")
    out = [0] * 6
    rem = int(label)
    for j in range(5, -1, -1):
        out[j] = rem % bins[j]
        rem //= bins[j]
    return tuple(out)  # type: ignore[return-value]


def subspace_bounds_deg(label: int, profile: str = DEFAULT_SEGMENT_PROFILE) -> np.ndarray:
    idx = decode_subspace_label(label, profile=profile)
    segments_all = get_segments(profile)
    bounds = np.zeros((6, 2), dtype=float)
    for j in range(6):
        lo, hi = segments_all[j][idx[j]]
        bounds[j, 0] = lo
        bounds[j, 1] = hi
    return bounds


def sample_q_in_subspace_deg(
    label: int,
    n: int,
    rng: np.random.Generator,
    profile: str = DEFAULT_SEGMENT_PROFILE,
) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    bounds = subspace_bounds_deg(label, profile=profile)
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, 6))
