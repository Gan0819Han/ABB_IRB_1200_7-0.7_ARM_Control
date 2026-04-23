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

from abb_nn.branch_models import MultiHeadBranchClassifier, build_branch_classifier_variant
from abb_nn.branching import (
    BRANCH_COUNT,
    BRANCH_HEAD_DIMS,
    BRANCH_HEAD_NAMES,
    BRANCH_PROFILE_NAME,
    ELBOW_CLASS_NAMES,
    SHOULDER_CLASS_NAMES,
    WRIST_CLASS_NAMES,
    assign_branch_head_labels,
    branch_label_to_name,
    branch_to_subspace_map,
)
from abb_nn.data_utils import apply_normalizer, fit_normalizer, save_json
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
        torch.from_numpy(y_train[:, 0].astype(np.int64)),
        torch.from_numpy(y_train[:, 1].astype(np.int64)),
        torch.from_numpy(y_train[:, 2].astype(np.int64)),
    )
    va_ds = TensorDataset(
        torch.from_numpy(x_val.astype(np.float32)),
        torch.from_numpy(y_val[:, 0].astype(np.int64)),
        torch.from_numpy(y_val[:, 1].astype(np.int64)),
        torch.from_numpy(y_val[:, 2].astype(np.int64)),
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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    ce = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total = 0
    correct_shoulder = 0
    correct_elbow = 0
    correct_wrist = 0
    correct_joint = 0
    with torch.no_grad():
        for xb, y_shoulder, y_elbow, y_wrist in loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            y_shoulder = y_shoulder.to(device, non_blocking=(device.type == "cuda"))
            y_elbow = y_elbow.to(device, non_blocking=(device.type == "cuda"))
            y_wrist = y_wrist.to(device, non_blocking=(device.type == "cuda"))
            with amp_autocast(use_amp):
                logits_shoulder, logits_elbow, logits_wrist = model(xb)
                loss = (
                    ce(logits_shoulder, y_shoulder)
                    + ce(logits_elbow, y_elbow)
                    + ce(logits_wrist, y_wrist)
                )
            total_loss += loss.item() * xb.shape[0]
            pred_shoulder = torch.argmax(logits_shoulder, dim=1)
            pred_elbow = torch.argmax(logits_elbow, dim=1)
            pred_wrist = torch.argmax(logits_wrist, dim=1)
            correct_shoulder += int((pred_shoulder == y_shoulder).sum().item())
            correct_elbow += int((pred_elbow == y_elbow).sum().item())
            correct_wrist += int((pred_wrist == y_wrist).sum().item())
            joint_ok = (
                (pred_shoulder == y_shoulder)
                & (pred_elbow == y_elbow)
                & (pred_wrist == y_wrist)
            )
            correct_joint += int(joint_ok.sum().item())
            total += xb.shape[0]
    return {
        "val_loss": total_loss / max(total, 1),
        "val_acc_shoulder": correct_shoulder / max(total, 1),
        "val_acc_elbow": correct_elbow / max(total, 1),
        "val_acc_wrist": correct_wrist / max(total, 1),
        "val_acc_joint": correct_joint / max(total, 1),
    }


def train_classifier(
    model: MultiHeadBranchClassifier,
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
        "val_acc_joint": -1.0,
        "val_loss": float("inf"),
        "metrics": None,
        "state": None,
    }

    for _ in range(epochs):
        model.train()
        for xb, y_shoulder, y_elbow, y_wrist in tr_loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            y_shoulder = y_shoulder.to(device, non_blocking=(device.type == "cuda"))
            y_elbow = y_elbow.to(device, non_blocking=(device.type == "cuda"))
            y_wrist = y_wrist.to(device, non_blocking=(device.type == "cuda"))
            opt.zero_grad(set_to_none=True)
            with amp_autocast(use_amp):
                logits_shoulder, logits_elbow, logits_wrist = model(xb)
                loss = (
                    ce(logits_shoulder, y_shoulder)
                    + ce(logits_elbow, y_elbow)
                    + ce(logits_wrist, y_wrist)
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        metrics = evaluate(model, va_loader, device, use_amp)
        if (
            metrics["val_acc_joint"] > best["val_acc_joint"]
            or (
                metrics["val_acc_joint"] == best["val_acc_joint"]
                and metrics["val_loss"] < best["val_loss"]
            )
        ):
            best["val_acc_joint"] = metrics["val_acc_joint"]
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
    parser = argparse.ArgumentParser(description="Train ABB_IRB hierarchical branch classification system.")
    parser.add_argument(
        "--segment_profile",
        type=str,
        default="abb_strict",
        choices=["abb_simplified", "abb_strict", "simplified", "strict"],
        help="Subspace profile used for branch-to-subspace compatibility metadata.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="artifacts/branch_classification_system")
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
    y_train_full = assign_branch_head_labels(q_train_full)
    x_val = build_pose_features(q_val, feature_device, args.feature_batch_size)
    y_val = assign_branch_head_labels(q_val)

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
        model = build_branch_classifier_variant(
            variant=variant,
            input_dim=6,
            head_dims=BRANCH_HEAD_DIMS,
        )
        res = train_classifier(model, tr_loader, va_loader, device, args.epochs, args.lr, use_amp)

        file_name = f"branch_classifier_v{variant}.pt"
        torch.save(
            {
                "robot_name": ROBOT_NAME,
                "robot_model": ROBOT_MODEL,
                "theta_offsets_deg": THETA_OFFSETS_DEG.tolist(),
                "variant": variant,
                "input_dim": 6,
                "branch_profile": BRANCH_PROFILE_NAME,
                "branch_head_dims": list(BRANCH_HEAD_DIMS),
                "segment_profile": args.segment_profile,
                "state_dict": res["model"].state_dict(),
                "best_val_loss": res["val_loss"],
                "best_val_acc_joint": res["val_acc_joint"],
                "best_val_acc_shoulder": res["val_acc_shoulder"],
                "best_val_acc_elbow": res["val_acc_elbow"],
                "best_val_acc_wrist": res["val_acc_wrist"],
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
                "best_val_acc_joint": res["val_acc_joint"],
                "best_val_acc_shoulder": res["val_acc_shoulder"],
                "best_val_acc_elbow": res["val_acc_elbow"],
                "best_val_acc_wrist": res["val_acc_wrist"],
            }
        )
        print(
            f"[branch classifier v{variant}] "
            f"train={n_train} val={args.val_samples} "
            f"val_loss={res['val_loss']:.6f} "
            f"joint_acc={res['val_acc_joint']:.4f} "
            f"shoulder_acc={res['val_acc_shoulder']:.4f} "
            f"elbow_acc={res['val_acc_elbow']:.4f} "
            f"wrist_acc={res['val_acc_wrist']:.4f}"
        )

    branch_to_subspaces = branch_to_subspace_map(args.segment_profile)
    metadata = {
        "framework": "pytorch",
        "robot_name": ROBOT_NAME,
        "robot_model": ROBOT_MODEL,
        "paper_style": "ABB_IRB hierarchical branch classification system",
        "segment_profile": args.segment_profile,
        "branch_profile": BRANCH_PROFILE_NAME,
        "input_features": ["x_mm", "y_mm", "z_mm", "phi_rad", "theta_rad", "psi_rad"],
        "branch_head_names": list(BRANCH_HEAD_NAMES),
        "branch_head_dims": list(BRANCH_HEAD_DIMS),
        "branch_class_names": {
            "shoulder": list(SHOULDER_CLASS_NAMES),
            "elbow": list(ELBOW_CLASS_NAMES),
            "wrist": list(WRIST_CLASS_NAMES),
        },
        "num_combined_branches": BRANCH_COUNT,
        "branch_to_subspaces": [
            {
                "branch_label": int(branch_label),
                "branch_name": branch_label_to_name(branch_label),
                "compatible_subspaces": [int(x) for x in subspaces],
            }
            for branch_label, subspaces in sorted(branch_to_subspaces.items())
        ],
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
