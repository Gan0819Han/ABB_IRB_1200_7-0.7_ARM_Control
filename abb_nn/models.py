#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Iterable[int]) -> None:
        super().__init__()
        dims: List[int] = [input_dim] + list(hidden_dims) + [output_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, use_batchnorm: bool = False) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(width, width)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(width, width))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(width))
        self.block = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ClassifierMLP(nn.Module):
    """
    Three classifier variants migrated from the previous Ref[22]-style workflow.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_width: int,
        hidden_layers: int,
        use_residual: bool,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

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
        self.head = nn.Linear(hidden_width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.hidden(x)
        return self.head(x)


def build_classifier_variant(variant: int, input_dim: int, num_classes: int) -> ClassifierMLP:
    if variant == 1:
        return ClassifierMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_width=35,
            hidden_layers=6,
            use_residual=False,
            use_batchnorm=False,
        )
    if variant == 2:
        return ClassifierMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_width=35,
            hidden_layers=20,
            use_residual=True,
            use_batchnorm=False,
        )
    if variant == 3:
        return ClassifierMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_width=35,
            hidden_layers=30,
            use_residual=True,
            use_batchnorm=True,
        )
    raise ValueError("variant must be 1, 2, or 3")
