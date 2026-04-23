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

from abb_nn.branching import (
    BRANCH_COUNT,
    BRANCH_PROFILE_NAME,
    FINE_JOINT_NAMES,
    assign_branch_labels,
    assign_fine_labels,
    branch_fine_to_subspace_label,
    branch_label_to_name,
    get_fine_bins,
    get_fine_class_count,
)
from abb_nn.data_utils import apply_normalizer, fit_normalizer, save_json
from abb_nn.models import build_classifier_variant
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


def build_conditioned_features(
    x_pose_norm: np.ndarray,
    branch_labels: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x_pose_norm, dtype=np.float32)
    b = np.asarray(branch_labels, dtype=np.int64).reshape(-1)
    onehot = np.zeros((x.shape[0], BRANCH_COUNT), dtype=np.float32)
    onehot[np.arange(x.shape[0]), b] = 1.0
    return np.concatenate([x, onehot], axis=1).astype(np.float32)


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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> Dict[str, float]:
    ce = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total = 0
    correct_top1 = 0
    correct_top3 = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))
            with amp_autocast(use_amp):
                logits = model(xb)
                loss = ce(logits, yb)
            total_loss += loss.item() * xb.shape[0]
            pred_top1 = torch.argmax(logits, dim=1)
            correct_top1 += int((pred_top1 == yb).sum().item())
            k_top3 = min(3, logits.shape[1])
            top3 = torch.topk(logits, k=k_top3, dim=1).indices
            correct_top3 += int((top3 == yb.unsqueeze(1)).any(dim=1).sum().item())
            total += xb.shape[0]
    return {
        "val_loss": total_loss / max(total, 1),
        "val_acc_top1": correct_top1 / max(total, 1),
        "val_acc_top3": correct_top3 / max(total, 1),
    }


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
    best = {
        "val_acc_top1": -1.0,
        "val_loss": float("inf"),
        "metrics": None,
        "state": None,
    }

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

        metrics = evaluate(model, va_loader, device, use_amp)
        if (
            metrics["val_acc_top1"] > best["val_acc_top1"]
            or (
                metrics["val_acc_top1"] == best["val_acc_top1"]
                and metrics["val_loss"] < best["val_loss"]
            )
        ):
            best["val_acc_top1"] = metrics["val_acc_top1"]
            best["val_loss"] = metrics["val_loss"]
            best["metrics"] = metrics
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    out = {"model": model.cpu()}
    if isinstance(best["metrics"], dict):
        out.update(best["metrics"])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ABB_IRB fine classification system under coarse branches.")
    parser.add_argument(
        "--segment_profile",
        type=str,
        default="abb_strict",
        choices=["abb_simplified", "abb_strict", "simplified", "strict"],
        help="Subspace profile for fine labels.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="artifacts/fine_classification_system")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainset_v1", type=int, default=250000)
    parser.add_argument("--trainset_v2", type=int, default=400000)
    parser.add_argument("--trainset_v3", type=int, default=320000)
    parser.add_argument("--val_samples", type=int, default=4000)
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
    y_branch_train = assign_branch_labels(q_train_full)
    y_fine_train = assign_fine_labels(q_train_full, segment_profile=args.segment_profile)
    x_val = build_pose_features(q_val, feature_device, args.feature_batch_size)
    y_branch_val = assign_branch_labels(q_val)
    y_fine_val = assign_fine_labels(q_val, segment_profile=args.segment_profile)

    normalizer = fit_normalizer(x_train_full)
    x_train_full_n = apply_normalizer(x_train_full, normalizer).astype(np.float32)
    x_val_n = apply_normalizer(x_val, normalizer).astype(np.float32)
    x_train_cond = build_conditioned_features(x_train_full_n, y_branch_train)
    x_val_cond = build_conditioned_features(x_val_n, y_branch_val)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fine_count = get_fine_class_count(args.segment_profile)
    train_sizes = {1: args.trainset_v1, 2: args.trainset_v2, 3: args.trainset_v3}
    results = []

    for variant in (1, 2, 3):
        n_train = train_sizes[variant]
        x_tr = x_train_cond[:n_train]
        y_tr = y_fine_train[:n_train]
        tr_loader, va_loader = make_loaders(
            x_tr,
            y_tr,
            x_val_cond,
            y_fine_val,
            args.batch_size,
            args.num_workers,
            pin_memory,
        )
        model = build_classifier_variant(
            variant=variant,
            input_dim=x_train_cond.shape[1],
            num_classes=fine_count,
        )
        res = train_classifier(model, tr_loader, va_loader, device, args.epochs, args.lr, use_amp)

        file_name = f"fine_classifier_v{variant}.pt"
        torch.save(
            {
                "robot_name": ROBOT_NAME,
                "robot_model": ROBOT_MODEL,
                "theta_offsets_deg": THETA_OFFSETS_DEG.tolist(),
                "variant": variant,
                "input_dim": int(x_train_cond.shape[1]),
                "num_classes": fine_count,
                "branch_profile": BRANCH_PROFILE_NAME,
                "segment_profile": args.segment_profile,
                "fine_joint_names": list(FINE_JOINT_NAMES),
                "fine_bins": get_fine_bins(args.segment_profile),
                "state_dict": res["model"].state_dict(),
                "best_val_loss": res["val_loss"],
                "best_val_acc_top1": res["val_acc_top1"],
                "best_val_acc_top3": res["val_acc_top3"],
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
                "best_val_loss": res["val_loss"],
                "best_val_acc_top1": res["val_acc_top1"],
                "best_val_acc_top3": res["val_acc_top3"],
            }
        )
        print(
            f"[fine classifier v{variant}] "
            f"train={n_train} val={args.val_samples} "
            f"val_loss={res['val_loss']:.6f} "
            f"top1={res['val_acc_top1']:.4f} top3={res['val_acc_top3']:.4f}"
        )

    branch_fine_map = []
    for branch_label in range(BRANCH_COUNT):
        local_items = []
        for fine_label in range(fine_count):
            subspace_id = branch_fine_to_subspace_label(
                branch_label,
                fine_label,
                segment_profile=args.segment_profile,
            )
            local_items.append(
                {
                    "fine_label": int(fine_label),
                    "subspace_id": int(subspace_id),
                }
            )
        branch_fine_map.append(
            {
                "branch_label": int(branch_label),
                "branch_name": branch_label_to_name(branch_label),
                "fine_to_subspace": local_items,
            }
        )

    metadata = {
        "framework": "pytorch",
        "robot_name": ROBOT_NAME,
        "robot_model": ROBOT_MODEL,
        "paper_style": "ABB_IRB fine classification system under coarse branches",
        "segment_profile": args.segment_profile,
        "branch_profile": BRANCH_PROFILE_NAME,
        "input_features": [
            "x_mm",
            "y_mm",
            "z_mm",
            "phi_rad",
            "theta_rad",
            "psi_rad",
            "branch_onehot_12",
        ],
        "fine_joint_names": list(FINE_JOINT_NAMES),
        "fine_bins": get_fine_bins(args.segment_profile),
        "num_fine_classes": fine_count,
        "theta_offsets_deg": THETA_OFFSETS_DEG.tolist(),
        "models": results,
        "branch_fine_to_subspace": branch_fine_map,
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
