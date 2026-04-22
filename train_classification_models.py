#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from abb_nn.data_utils import apply_normalizer, fit_normalizer, save_json
from abb_nn.models import build_classifier_variant
from abb_nn.subspace import assign_subspace_labels, get_segments, get_subspace_count
from fk_model import JOINT_LIMITS_DEG, pose6_from_q_torch_batch
from robot_config import ROBOT_MODEL, ROBOT_NAME, THETA_OFFSETS_DEG


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


def make_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    tr_ds = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    va_ds = TensorDataset(
        torch.from_numpy(x_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.int64)),
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    tr_loader = DataLoader(tr_ds, shuffle=True, **loader_kwargs)
    va_loader = DataLoader(va_ds, shuffle=False, **loader_kwargs)
    return tr_loader, va_loader


def make_amp_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> Tuple[float, float]:
    ce = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))
            with amp_autocast(use_amp):
                logits = model(xb)
            loss = ce(logits, yb).item()
            total_loss += loss * xb.shape[0]
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())
            total += xb.shape[0]
    return total_loss / max(total, 1), correct / max(total, 1)


def train_classifier(
    model: nn.Module,
    tr_loader: DataLoader,
    va_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    use_amp: bool,
) -> Dict[str, object]:
    model = model.to(device)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = make_amp_grad_scaler(use_amp)
    best = {"val_loss": float("inf"), "val_acc": 0.0, "state": None}

    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))
            opt.zero_grad(set_to_none=True)
            with amp_autocast(use_amp):
                logits = model(xb)
                loss = ce(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        val_loss, val_acc = evaluate(model, va_loader, device, use_amp)
        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["val_acc"] = val_acc
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {
        "model": model.cpu(),
        "best_val_loss": float(best["val_loss"]),
        "best_val_acc": float(best["val_acc"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ABB_IRB classification system.")
    parser.add_argument(
        "--segment_profile",
        type=str,
        default="abb_simplified",
        choices=["abb_simplified", "abb_strict", "simplified", "strict"],
        help="Subspace segmentation profile.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="artifacts/classification_system")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainset_v1", type=int, default=300000)
    parser.add_argument("--trainset_v2", type=int, default=500000)
    parser.add_argument("--trainset_v3", type=int, default=400000)
    parser.add_argument("--val_samples", type=int, default=3000)
    parser.add_argument(
        "--feature_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for FK feature generation.",
    )
    parser.add_argument(
        "--feature_batch_size",
        type=int,
        default=65536,
        help="Batch size for FK feature generation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(8, max((os.cpu_count() or 1) - 1, 0)),
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--disable_amp",
        action="store_true",
        help="Disable mixed precision on CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    subspace_count = get_subspace_count(args.segment_profile)
    joint_segments_deg = get_segments(args.segment_profile)
    if args.feature_batch_size <= 0:
        raise ValueError("--feature_batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0.")

    n_max = max(args.trainset_v1, args.trainset_v2, args.trainset_v3)
    q_train_full = rng.uniform(JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1], size=(n_max, 6))
    q_val = rng.uniform(JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1], size=(args.val_samples, 6))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_device = resolve_feature_device(args.feature_device)
    pin_memory = device.type == "cuda"
    use_amp = (device.type == "cuda") and (not args.disable_amp)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    x_train_full = build_pose_features(q_train_full, feature_device, args.feature_batch_size)
    y_train_full = assign_subspace_labels(q_train_full, profile=args.segment_profile)
    x_val = build_pose_features(q_val, feature_device, args.feature_batch_size)
    y_val = assign_subspace_labels(q_val, profile=args.segment_profile)

    normalizer = fit_normalizer(x_train_full)
    x_train_full_n = apply_normalizer(x_train_full, normalizer).astype(np.float32)
    x_val_n = apply_normalizer(x_val, normalizer).astype(np.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sizes = {1: args.trainset_v1, 2: args.trainset_v2, 3: args.trainset_v3}
    results = []

    for variant in (1, 2, 3):
        n_train = train_sizes[variant]
        x_tr = x_train_full_n[:n_train]
        y_tr = y_train_full[:n_train]
        tr_loader, va_loader = make_loaders(
            x_tr,
            y_tr,
            x_val_n,
            y_val,
            args.batch_size,
            args.num_workers,
            pin_memory,
        )
        model = build_classifier_variant(variant=variant, input_dim=6, num_classes=subspace_count)
        res = train_classifier(model, tr_loader, va_loader, device, args.epochs, args.lr, use_amp)

        file_name = f"classifier_v{variant}.pt"
        torch.save(
            {
                "robot_name": ROBOT_NAME,
                "robot_model": ROBOT_MODEL,
                "theta_offsets_deg": THETA_OFFSETS_DEG.tolist(),
                "variant": variant,
                "input_dim": 6,
                "num_classes": subspace_count,
                "segment_profile": args.segment_profile,
                "state_dict": res["model"].state_dict(),
                "best_val_loss": res["best_val_loss"],
                "best_val_acc": res["best_val_acc"],
                "train_samples": n_train,
                "val_samples": args.val_samples,
            },
            out_dir / file_name,
        )
        results.append(
            {
                "variant": variant,
                "file": file_name,
                "train_samples": n_train,
                "val_samples": args.val_samples,
                "best_val_loss": res["best_val_loss"],
                "best_val_acc": res["best_val_acc"],
            }
        )
        print(
            f"[classifier v{variant}] "
            f"train={n_train} val={args.val_samples} "
            f"val_loss={res['best_val_loss']:.6f} val_acc={res['best_val_acc']:.4f}"
        )

    metadata = {
        "framework": "pytorch",
        "robot_name": ROBOT_NAME,
        "robot_model": ROBOT_MODEL,
        "paper_style": "ABB_IRB classification system (migrated from Ref[22] workflow)",
        "segment_profile": args.segment_profile,
        "input_features": ["x_mm", "y_mm", "z_mm", "phi_rad", "theta_rad", "psi_rad"],
        "num_classes": subspace_count,
        "joint_segments_deg": joint_segments_deg,
        "theta_offsets_deg": THETA_OFFSETS_DEG.tolist(),
        "models": results,
        "normalizer": {
            "mean": normalizer["mean"].reshape(-1).tolist(),
            "std": normalizer["std"].reshape(-1).tolist(),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "trainset_v1": args.trainset_v1,
            "trainset_v2": args.trainset_v2,
            "trainset_v3": args.trainset_v3,
            "val_samples": args.val_samples,
            "feature_device": str(feature_device),
            "feature_batch_size": args.feature_batch_size,
            "num_workers": args.num_workers,
            "amp_enabled": use_amp,
        },
    }
    save_json(out_dir / "metadata.json", metadata)
    print(f"Saved metadata: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
