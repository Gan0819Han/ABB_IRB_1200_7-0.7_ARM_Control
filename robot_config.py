#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Central ABB_IRB robot configuration for the current project stage.

This module keeps all robot-specific constants in one place so that FK/IK,
training, benchmarking, and obstacle-avoidance code can reuse the same source
of truth.
"""

from __future__ import annotations

import numpy as np


ROBOT_NAME = "ABB_IRB"
ROBOT_MODEL = "IRB 1200-7/0.7"
KINEMATICS_CONVENTION = "standard_dh"
LENGTH_UNIT = "mm"
JOINT_INPUT_UNIT = "deg"
ORIENTATION_REPRESENTATION = "zyx_euler"


# Standard DH parameters for ABB IRB 1200-7/0.7 used in this project.
# Length unit: mm
# Angle unit: deg
DH_A_MM = np.array([0.0, 350.0, 42.0, 0.0, 0.0, 0.0], dtype=float)
DH_ALPHA_DEG = np.array([-90.0, 0.0, -90.0, 90.0, -90.0, 0.0], dtype=float)
DH_D_MM = np.array([399.1, 0.0, 0.0, 351.0, 0.0, 82.0], dtype=float)

# Mechanical joint angles q are converted to DH theta via:
# theta = q + THETA_OFFSETS_DEG
THETA_OFFSETS_DEG = np.array([0.0, -90.0, 0.0, 0.0, 0.0, 0.0], dtype=float)


# ABB official joint limits for IRB 1200-7/0.7.
OFFICIAL_JOINT_LIMITS_DEG = np.array(
    [
        [-170.0, 170.0],
        [-100.0, 135.0],
        [-200.0, 70.0],
        [-270.0, 270.0],
        [-130.0, 130.0],
        [-400.0, 400.0],
    ],
    dtype=float,
)


# Current project modeling limits. Axis-6 is intentionally reduced to one turn
# to keep dataset generation and NN segmentation manageable.
PROJECT_JOINT_LIMITS_DEG = OFFICIAL_JOINT_LIMITS_DEG.copy()
PROJECT_JOINT_LIMITS_DEG[5] = np.array([-180.0, 180.0], dtype=float)

# Backward-compatible default name for downstream code.
JOINT_LIMITS_DEG = PROJECT_JOINT_LIMITS_DEG.copy()


ZERO_POSE_Q_DEG = np.zeros(6, dtype=float)


# ABB official workspace references for IRB 1200-7/0.7 mechanical wrist center.
# Source: IRB1200产品规格.pdf, page 54.
OFFICIAL_WRIST_CENTER_REFERENCES = (
    {
        "name": "Pos0",
        "q_deg": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "official_wrist_center_x_mm": 351.0,
        "official_wrist_center_z_mm": 791.0,
        "source": "IRB1200产品规格.pdf p54",
    },
    {
        "name": "Pos3",
        "q_deg": [0.0, 90.0, -83.0, 0.0, 0.0, 0.0],
        "official_wrist_center_x_mm": 703.0,
        "official_wrist_center_z_mm": 398.0,
        "source": "IRB1200产品规格.pdf p54",
    },
    {
        "name": "Pos6",
        "q_deg": [0.0, -100.0, 70.0, 0.0, 0.0, 0.0],
        "official_wrist_center_x_mm": -62.0,
        "official_wrist_center_z_mm": 550.0,
        "source": "IRB1200产品规格.pdf p54",
    },
    {
        "name": "Pos7",
        "q_deg": [0.0, -90.0, -83.0, 0.0, 0.0, 0.0],
        "official_wrist_center_x_mm": -703.0,
        "official_wrist_center_z_mm": 400.0,
        "source": "IRB1200产品规格.pdf p54",
    },
    {
        "name": "Pos9",
        "q_deg": [0.0, 135.0, -200.0, 0.0, 0.0, 0.0],
        "official_wrist_center_x_mm": 358.0,
        "official_wrist_center_z_mm": 488.0,
        "source": "IRB1200产品规格.pdf p54",
    },
)
