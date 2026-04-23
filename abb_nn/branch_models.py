#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .models import ResidualBlock


class MultiHeadBranchClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dims: Sequence[int],
        hidden_width: int,
        hidden_layers: int,
        use_residual: bool,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        self.head_dims: Tuple[int, ...] = tuple(int(x) for x in head_dims)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.ReLU(inplace=True),
        )

        blocks: List[nn.Module] = []
        for _ in range(hidden_layers - 1):
            if use_residual:
                blocks.append(ResidualBlock(hidden_width, use_batchnorm=use_batchnorm))
            else:
                unit: List[nn.Module] = [nn.Linear(hidden_width, hidden_width)]
                if use_batchnorm:
                    unit.append(nn.BatchNorm1d(hidden_width))
                unit.append(nn.ReLU(inplace=True))
                blocks.append(nn.Sequential(*unit))
        self.hidden = nn.Sequential(*blocks)
        self.shoulder_head = nn.Linear(hidden_width, self.head_dims[0])
        self.elbow_head = nn.Linear(hidden_width, self.head_dims[1])
        self.wrist_head = nn.Linear(hidden_width, self.head_dims[2])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x = self.hidden(x)
        return (
            self.shoulder_head(x),
            self.elbow_head(x),
            self.wrist_head(x),
        )


def build_branch_classifier_variant(
    variant: int,
    input_dim: int,
    head_dims: Sequence[int],
) -> MultiHeadBranchClassifier:
    if variant == 1:
        return MultiHeadBranchClassifier(
            input_dim=input_dim,
            head_dims=head_dims,
            hidden_width=35,
            hidden_layers=6,
            use_residual=False,
            use_batchnorm=False,
        )
    if variant == 2:
        return MultiHeadBranchClassifier(
            input_dim=input_dim,
            head_dims=head_dims,
            hidden_width=35,
            hidden_layers=20,
            use_residual=True,
            use_batchnorm=False,
        )
    if variant == 3:
        return MultiHeadBranchClassifier(
            input_dim=input_dim,
            head_dims=head_dims,
            hidden_width=35,
            hidden_layers=30,
            use_residual=True,
            use_batchnorm=True,
        )
    raise ValueError("variant must be 1, 2, or 3")
