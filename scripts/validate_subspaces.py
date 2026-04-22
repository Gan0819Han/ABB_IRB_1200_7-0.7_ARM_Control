#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abb_nn.subspace import get_joint_bins, get_segments, get_subspace_count, subspace_bounds_deg


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_profile_summary(profile: str) -> dict:
    segments = get_segments(profile)
    bins = get_joint_bins(profile)
    count = get_subspace_count(profile)
    examples = {
        "subspace_000_bounds_deg": subspace_bounds_deg(0, profile=profile).tolist(),
        "subspace_last_bounds_deg": subspace_bounds_deg(count - 1, profile=profile).tolist(),
    }
    return {
        "profile": profile,
        "joint_bins": bins,
        "subspace_count": count,
        "joint_segments_deg": [[list(seg) for seg in joint] for joint in segments],
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ABB subspace profiles.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/subspace_validation",
        help="Directory for validation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (ROOT / args.out_dir).resolve()

    data = {
        "profiles": [
            build_profile_summary("abb_simplified"),
            build_profile_summary("abb_strict"),
        ]
    }
    out_path = out_dir / "subspace_profiles.json"
    save_json(data, out_path)

    print(f"Saved validation report: {out_path}")
    for item in data["profiles"]:
        bins = " x ".join(str(x) for x in item["joint_bins"])
        print(f"[{item['profile']}] bins={bins} => subspaces={item['subspace_count']}")


if __name__ == "__main__":
    main()
