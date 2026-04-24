#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


LINK_SEGMENTS = [
    ("base_to_joint1", 0, 1),
    ("joint1_to_joint2", 1, 2),
    ("joint2_to_joint3", 2, 3),
    ("joint3_to_joint4", 3, 4),
    ("joint4_to_joint5", 4, 5),
    ("joint5_to_tool0", 5, 6),
]


@dataclass(frozen=True)
class AABBObstacle:
    name: str
    center_mm: np.ndarray
    size_mm: np.ndarray

    @staticmethod
    def from_dict(payload: dict) -> "AABBObstacle":
        name = str(payload.get("name", "unnamed_box"))
        if "center_mm" in payload and "size_mm" in payload:
            center = np.asarray(payload["center_mm"], dtype=float).reshape(3)
            size = np.asarray(payload["size_mm"], dtype=float).reshape(3)
            return AABBObstacle(name=name, center_mm=center, size_mm=size)
        if "min_mm" in payload and "max_mm" in payload:
            box_min = np.asarray(payload["min_mm"], dtype=float).reshape(3)
            box_max = np.asarray(payload["max_mm"], dtype=float).reshape(3)
            center = 0.5 * (box_min + box_max)
            size = box_max - box_min
            return AABBObstacle(name=name, center_mm=center, size_mm=size)
        raise ValueError(f"Obstacle '{name}' must contain either center_mm/size_mm or min_mm/max_mm.")

    @property
    def min_mm(self) -> np.ndarray:
        return self.center_mm - 0.5 * self.size_mm

    @property
    def max_mm(self) -> np.ndarray:
        return self.center_mm + 0.5 * self.size_mm

    def inflated_bounds(self, inflate_mm: float) -> tuple[np.ndarray, np.ndarray]:
        grow = float(max(0.0, inflate_mm))
        return self.min_mm - grow, self.max_mm + grow

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "center_mm": self.center_mm.tolist(),
            "size_mm": self.size_mm.tolist(),
            "min_mm": self.min_mm.tolist(),
            "max_mm": self.max_mm.tolist(),
        }


@dataclass(frozen=True)
class ObstacleScene:
    scene_name: str
    link_radius_mm: float
    safety_margin_mm: float
    clearance_sample_count: int
    obstacles: tuple[AABBObstacle, ...]

    @staticmethod
    def from_dict(payload: dict) -> "ObstacleScene":
        obstacles = tuple(AABBObstacle.from_dict(item) for item in payload.get("obstacles", []))
        return ObstacleScene(
            scene_name=str(payload.get("scene_name", "unnamed_scene")),
            link_radius_mm=float(payload.get("link_radius_mm", 35.0)),
            safety_margin_mm=float(payload.get("safety_margin_mm", 5.0)),
            clearance_sample_count=int(payload.get("clearance_sample_count", 25)),
            obstacles=obstacles,
        )

    @staticmethod
    def from_json(path: Path | str) -> "ObstacleScene":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return ObstacleScene.from_dict(payload)

    def to_dict(self) -> dict:
        return {
            "scene_name": self.scene_name,
            "link_radius_mm": float(self.link_radius_mm),
            "safety_margin_mm": float(self.safety_margin_mm),
            "clearance_sample_count": int(self.clearance_sample_count),
            "obstacles": [item.to_dict() for item in self.obstacles],
        }


def segment_intersects_aabb(
    p0_mm: Iterable[float],
    p1_mm: Iterable[float],
    box_min_mm: Iterable[float],
    box_max_mm: Iterable[float],
    eps: float = 1e-9,
) -> bool:
    p0 = np.asarray(list(p0_mm), dtype=float).reshape(3)
    p1 = np.asarray(list(p1_mm), dtype=float).reshape(3)
    box_min = np.asarray(list(box_min_mm), dtype=float).reshape(3)
    box_max = np.asarray(list(box_max_mm), dtype=float).reshape(3)
    direction = p1 - p0

    t_enter = 0.0
    t_exit = 1.0
    for axis in range(3):
        if abs(direction[axis]) <= eps:
            if p0[axis] < box_min[axis] or p0[axis] > box_max[axis]:
                return False
            continue

        inv_d = 1.0 / direction[axis]
        t1 = (box_min[axis] - p0[axis]) * inv_d
        t2 = (box_max[axis] - p0[axis]) * inv_d
        if t1 > t2:
            t1, t2 = t2, t1
        t_enter = max(t_enter, t1)
        t_exit = min(t_exit, t2)
        if t_enter > t_exit:
            return False
    return True


def point_to_aabb_distance_mm(
    point_mm: Iterable[float],
    box_min_mm: Iterable[float],
    box_max_mm: Iterable[float],
) -> float:
    point = np.asarray(list(point_mm), dtype=float).reshape(3)
    box_min = np.asarray(list(box_min_mm), dtype=float).reshape(3)
    box_max = np.asarray(list(box_max_mm), dtype=float).reshape(3)
    delta = np.maximum(0.0, np.maximum(box_min - point, point - box_max))
    return float(np.linalg.norm(delta))


def sampled_segment_clearance_mm(
    p0_mm: Iterable[float],
    p1_mm: Iterable[float],
    obstacle: AABBObstacle,
    inflate_mm: float,
    sample_count: int,
) -> float:
    p0 = np.asarray(list(p0_mm), dtype=float).reshape(3)
    p1 = np.asarray(list(p1_mm), dtype=float).reshape(3)
    total = max(2, int(sample_count))
    ts = np.linspace(0.0, 1.0, total, dtype=float)
    samples = p0.reshape(1, 3) + ts.reshape(-1, 1) * (p1 - p0).reshape(1, 3)
    dists = [
        point_to_aabb_distance_mm(point_mm=sample, box_min_mm=obstacle.min_mm, box_max_mm=obstacle.max_mm)
        for sample in samples
    ]
    return float(min(dists) - max(0.0, inflate_mm))


def evaluate_link_obstacle_collision(
    p0_mm: Iterable[float],
    p1_mm: Iterable[float],
    obstacle: AABBObstacle,
    inflate_mm: float,
    sample_count: int,
) -> dict:
    box_min, box_max = obstacle.inflated_bounds(inflate_mm=inflate_mm)
    collision = segment_intersects_aabb(p0_mm=p0_mm, p1_mm=p1_mm, box_min_mm=box_min, box_max_mm=box_max)
    clearance = sampled_segment_clearance_mm(
        p0_mm=p0_mm,
        p1_mm=p1_mm,
        obstacle=obstacle,
        inflate_mm=inflate_mm,
        sample_count=sample_count,
    )
    if collision:
        clearance = min(clearance, 0.0)
    return {
        "collision": bool(collision),
        "clearance_mm": float(clearance),
    }


def evaluate_robot_aabb_collision(
    joint_points_mm: Sequence[Sequence[float]],
    scene: ObstacleScene,
) -> dict:
    points = np.asarray(joint_points_mm, dtype=float)
    if points.shape != (7, 3):
        raise ValueError("joint_points_mm must have shape (7, 3).")

    inflate_mm = float(max(0.0, scene.link_radius_mm) + max(0.0, scene.safety_margin_mm))
    min_clearance = float("inf")
    hits: list[dict] = []

    for link_name, idx0, idx1 in LINK_SEGMENTS:
        p0 = points[idx0]
        p1 = points[idx1]
        for obstacle in scene.obstacles:
            result = evaluate_link_obstacle_collision(
                p0_mm=p0,
                p1_mm=p1,
                obstacle=obstacle,
                inflate_mm=inflate_mm,
                sample_count=scene.clearance_sample_count,
            )
            min_clearance = min(min_clearance, float(result["clearance_mm"]))
            if result["collision"]:
                hits.append(
                    {
                        "link_name": link_name,
                        "obstacle_name": obstacle.name,
                        "clearance_mm": float(result["clearance_mm"]),
                    }
                )

    if not np.isfinite(min_clearance):
        min_clearance = float("inf")

    return {
        "collision": bool(len(hits) > 0),
        "min_clearance_mm": float(min_clearance),
        "hits": hits,
        "colliding_links": sorted({item["link_name"] for item in hits}),
        "colliding_obstacles": sorted({item["obstacle_name"] for item in hits}),
    }
