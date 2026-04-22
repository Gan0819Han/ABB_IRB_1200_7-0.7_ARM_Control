#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from abb_nn.data_utils import apply_normalizer, fit_normalizer, save_json
from abb_nn.models import MLPRegressor
from abb_nn.subspace import get_segments, get_subspace_count, sample_q_in_subspace_deg
from fk_model import JOINT_LIMITS_DEG, pose6_from_q_torch_batch
from robot_config import ROBOT_MODEL, ROBOT_NAME, THETA_OFFSETS_DEG


def parse_int_list(text: str) -> List[int]:
    if not text.strip():
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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


def split_dataset(
    q_deg: np.ndarray,
    x: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    n = q_deg.shape[0]
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    tr = idx[:n_train]
    va = idx[n_train:n_train + n_val]
    te = idx[n_train + n_val:]
    return {
        "q_train": q_deg[tr],
        "x_train": x[tr],
        "q_val": q_deg[va],
        "x_val": x[va],
        "q_test": q_deg[te],
        "x_test": x[te],
    }


def make_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    tr_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    va_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
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


def train_model(
    model: nn.Module,
    tr_loader: DataLoader,
    va_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    use_amp: bool,
) -> Tuple[nn.Module, float]:
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = make_amp_grad_scaler(use_amp)
    best_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(use_amp):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device, non_blocking=(device.type == "cuda"))
                yb = yb.to(device, non_blocking=(device.type == "cuda"))
                with amp_autocast(use_amp):
                    loss = criterion(model(xb), yb).item()
                total += loss * xb.shape[0]
                count += xb.shape[0]
        val_loss = total / max(count, 1)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), float(best_loss)


def batch_predict(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    use_amp: bool,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    model = model.to(device)
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).to(
                device,
                non_blocking=(device.type == "cuda"),
            )
            with amp_autocast(use_amp):
                y = model(xb)
            out.append(y.float().cpu().numpy())
    model = model.cpu()
    return np.concatenate(out, axis=0)


def calc_position_l2_norms(
    q_pred_deg: np.ndarray,
    target_x: np.ndarray,
    feature_device: torch.device,
    feature_batch_size: int,
) -> np.ndarray:
    pose_pred = build_pose_features(q_pred_deg, feature_device, feature_batch_size)
    delta = pose_pred[:, :3] - target_x[:, :3].astype(np.float32)
    return np.linalg.norm(delta, axis=1).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ABB_IRB segmented prediction system.")
    parser.add_argument(
        "--segment_profile",
        type=str,
        default="abb_simplified",
        choices=["abb_simplified", "abb_strict", "simplified", "strict"],
        help="Subspace segmentation profile.",
    )
    parser.add_argument("--samples_per_subspace", type=int, default=80000)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--neurons_per_layer", type=int, default=20)
    parser.add_argument(
        "--subspaces",
        type=str,
        default="",
        help="Comma-separated subspace IDs. Empty means all subspaces in the chosen profile.",
    )
    parser.add_argument("--out_dir", type=str, default="artifacts/prediction_system")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalizer_samples",
        type=int,
        default=200000,
        help="Samples from global joint limits to fit feature normalizer.",
    )
    parser.add_argument(
        "--feature_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for FK feature generation and L2 evaluation.",
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

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    subspaces = parse_int_list(args.subspaces)
    if not subspaces:
        subspaces = list(range(subspace_count))
    for sid in subspaces:
        if sid < 0 or sid >= subspace_count:
            raise ValueError(f"Invalid subspace id: {sid}")

    out_dir = Path(args.out_dir)
    models_dir = out_dir / "subspace_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_batch_size <= 0:
        raise ValueError("--feature_batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0.")

    q_norm = rng.uniform(JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1], size=(args.normalizer_samples, 6))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_device = resolve_feature_device(args.feature_device)
    pin_memory = device.type == "cuda"
    use_amp = (device.type == "cuda") and (not args.disable_amp)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    x_norm = build_pose_features(q_norm, feature_device, args.feature_batch_size)
    normalizer = fit_normalizer(x_norm)

    hidden_dims = [args.neurons_per_layer] * args.hidden_layers
    trained_entries: List[Dict[str, object]] = []

    for sid in subspaces:
        q_all = sample_q_in_subspace_deg(sid, args.samples_per_subspace, rng, profile=args.segment_profile)
        x_all = build_pose_features(q_all, feature_device, args.feature_batch_size)
        split = split_dataset(q_all, x_all, args.train_ratio, args.val_ratio, rng)

        x_train = apply_normalizer(split["x_train"], normalizer).astype(np.float32)
        x_val = apply_normalizer(split["x_val"], normalizer).astype(np.float32)
        x_test = apply_normalizer(split["x_test"], normalizer).astype(np.float32)

        y_train = split["q_train"].astype(np.float32)
        y_val = split["q_val"].astype(np.float32)
        y_test = split["q_test"].astype(np.float32)

        tr15, va15 = make_loaders(
            x_train,
            y_train[:, :5],
            x_val,
            y_val[:, :5],
            args.batch_size,
            args.num_workers,
            pin_memory,
        )
        tr6, va6 = make_loaders(
            x_train,
            y_train[:, 5:6],
            x_val,
            y_val[:, 5:6],
            args.batch_size,
            args.num_workers,
            pin_memory,
        )

        m15 = MLPRegressor(input_dim=6, output_dim=5, hidden_dims=hidden_dims)
        m6 = MLPRegressor(input_dim=6, output_dim=1, hidden_dims=hidden_dims)

        m15, val_loss_15 = train_model(m15, tr15, va15, device, args.epochs, args.lr, use_amp)
        m6, val_loss_6 = train_model(m6, tr6, va6, device, args.epochs, args.lr, use_amp)

        q15_val = batch_predict(m15, x_val, device, use_amp)
        q6_val = batch_predict(m6, x_val, device, use_amp)
        q_val_pred = np.concatenate([q15_val, q6_val], axis=1).astype(np.float32)
        e_max = float(
            np.max(
                calc_position_l2_norms(
                    q_val_pred,
                    split["x_val"],
                    feature_device,
                    args.feature_batch_size,
                )
            )
        )

        q15_test = batch_predict(m15, x_test, device, use_amp)
        q6_test = batch_predict(m6, x_test, device, use_amp)
        q_test_pred = np.concatenate([q15_test, q6_test], axis=1).astype(np.float32)
        test_pos_l2_mean = float(
            np.mean(
                calc_position_l2_norms(
                    q_test_pred,
                    split["x_test"],
                    feature_device,
                    args.feature_batch_size,
                )
            )
        )

        ckpt_name = f"subspace_{sid:03d}.pt"
        torch.save(
            {
                "robot_name": ROBOT_NAME,
                "robot_model": ROBOT_MODEL,
                "theta_offsets_deg": THETA_OFFSETS_DEG.tolist(),
                "subspace_id": sid,
                "input_dim": 6,
                "hidden_dims_q15": hidden_dims,
                "hidden_dims_q6": hidden_dims,
                "state_q15": m15.state_dict(),
                "state_q6": m6.state_dict(),
                "val_loss_q15": val_loss_15,
                "val_loss_q6": val_loss_6,
                "e_max": e_max,
                "train_samples": int(split["q_train"].shape[0]),
                "val_samples": int(split["q_val"].shape[0]),
                "test_samples": int(split["q_test"].shape[0]),
            },
            models_dir / ckpt_name,
        )

        trained_entries.append(
            {
                "subspace_id": sid,
                "model_file": ckpt_name,
                "train_samples": int(split["q_train"].shape[0]),
                "val_samples": int(split["q_val"].shape[0]),
                "test_samples": int(split["q_test"].shape[0]),
                "val_loss_q15": val_loss_15,
                "val_loss_q6": val_loss_6,
                "e_max": e_max,
                "test_pos_l2_mean_mm": test_pos_l2_mean,
            }
        )
        print(
            f"[subspace {sid:03d}] "
            f"train={split['q_train'].shape[0]} val={split['q_val'].shape[0]} "
            f"mse(q1-5)={val_loss_15:.4f} mse(q6)={val_loss_6:.4f} e_max={e_max:.6f}"
        )

    metadata = {
        "framework": "pytorch",
        "robot_name": ROBOT_NAME,
        "robot_model": ROBOT_MODEL,
        "paper_style": "ABB_IRB segmented prediction system (migrated from Ref[22] workflow)",
        "segment_profile": args.segment_profile,
        "input_features": ["x_mm", "y_mm", "z_mm", "phi_rad", "theta_rad", "psi_rad"],
        "output_targets": ["q1_deg", "q2_deg", "q3_deg", "q4_deg", "q5_deg", "q6_deg"],
        "subspace_count": subspace_count,
        "joint_segments_deg": joint_segments_deg,
        "theta_offsets_deg": THETA_OFFSETS_DEG.tolist(),
        "trained_subspaces": trained_entries,
        "normalizer": {
            "mean": normalizer["mean"].reshape(-1).tolist(),
            "std": normalizer["std"].reshape(-1).tolist(),
        },
        "hyperparameters": {
            "samples_per_subspace": args.samples_per_subspace,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_layers": args.hidden_layers,
            "neurons_per_layer": args.neurons_per_layer,
            "seed": args.seed,
            "normalizer_samples": args.normalizer_samples,
            "feature_device": str(feature_device),
            "feature_batch_size": args.feature_batch_size,
            "num_workers": args.num_workers,
            "amp_enabled": use_amp,
        },
    }
    save_json(out_dir / "metadata.json", metadata)
    print(f"Saved metadata: {out_dir / 'metadata.json'}")
    print(f"Trained subspaces: {len(trained_entries)}")


if __name__ == "__main__":
    main()
