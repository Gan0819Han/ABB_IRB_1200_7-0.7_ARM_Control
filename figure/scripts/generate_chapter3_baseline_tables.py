#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "figure"
DATA_DIR = FIGURE_DIR / "data"
ARTIFACTS_DIR = ROOT / "artifacts"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metric_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def percent_text(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_flat_192_table() -> tuple[list[dict], str]:
    metadata = load_json(ARTIFACTS_DIR / "classification_system_formal" / "metadata.json")
    rows = []
    for model in metadata["models"]:
        train_samples = int(model["train_samples"])
        val_samples = int(model["val_samples"])
        best_val_loss = float(model["best_val_loss"])
        best_val_acc = float(model["best_val_acc"])
        rows.append(
            {
                "variant": int(model["variant"]),
                "train_samples": train_samples,
                "val_samples": val_samples,
                "best_val_loss": f"{best_val_loss:.4f}",
                "top1_acc": f"{best_val_acc:.4f}",
                "top1_acc_percent": percent_text(best_val_acc),
            }
        )

    mean_acc = sum(float(row["top1_acc"]) for row in rows) / len(rows)
    max_acc = max(float(row["top1_acc"]) for row in rows)
    min_acc = min(float(row["top1_acc"]) for row in rows)

    md_lines = [
        "# 第3章 3.2.1 单层192类分类基线统计表",
        "",
        "## 表1 单层192类分类基线验证结果",
        "",
        "| 变体编号 | 训练样本数 | 验证样本数 | 最优验证损失 | Top-1准确率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['variant']} | {row['train_samples']} | {row['val_samples']} | "
            f"{row['best_val_loss']} | {row['top1_acc_percent']} |"
        )
    md_lines.extend(
        [
            "",
            "注：表中三个分类器变体采用相同的输入与输出定义，输入均为目标位姿"
            " $[x,y,z,\\phi,\\theta,\\psi]^T$，输出均为192个子空间类别的概率分布，"
            "其差异主要体现在网络深度与结构增强方式上。",
            "变体1采用6层隐藏层的普通前馈多层感知机结构，不使用残差连接与批量归一化；"
            "变体2在相同隐藏宽度下将网络加深至20层，并引入残差连接，以缓解深层网络训练中的梯度退化问题；"
            "变体3进一步将网络加深至30层，同时在残差结构基础上加入批量归一化，以增强训练稳定性与特征分布一致性。",
            "",
            f"统计说明：三组单层192类分类模型的 Top-1 准确率均值为 {percent_text(mean_acc)}，"
            f"最高值为 {percent_text(max_acc)}，最低值为 {percent_text(min_acc)}。",
            "在训练样本规模达到40万至60万的条件下，验证准确率仍维持在较低水平，"
            "说明直接在192个子空间上进行单阶段判别存在较明显的类别混淆问题。",
            "",
        ]
    )
    return rows, "\n".join(md_lines)


def build_hierarchical_comparison_table() -> tuple[list[dict], str]:
    metrics = load_metric_rows(DATA_DIR / "classification_metrics_summary.csv")
    branch_map: dict[int, dict] = {}
    fine_map: dict[int, dict] = {}

    for row in metrics:
        family = row["family"]
        variant = int(row["variant"])
        metric = row["metric"]
        value = float(row["value"])
        if family == "branch":
            branch_map.setdefault(variant, {})[metric] = value
        elif family == "fine_16":
            fine_map.setdefault(variant, {})[metric] = value

    rows = []
    for variant in sorted(set(branch_map) | set(fine_map)):
        branch_item = branch_map.get(variant, {})
        fine_item = fine_map.get(variant, {})
        rows.append(
            {
                "variant": variant,
                "branch_joint_acc": percent_text(branch_item.get("joint_acc", 0.0)),
                "shoulder_acc": percent_text(branch_item.get("shoulder_acc", 0.0)),
                "elbow_acc": percent_text(branch_item.get("elbow_acc", 0.0)),
                "wrist_acc": percent_text(branch_item.get("wrist_acc", 0.0)),
                "fine_top1_acc": percent_text(fine_item.get("top1_acc", 0.0)),
                "fine_top3_acc": percent_text(fine_item.get("top3_acc", 0.0)),
            }
        )

    md_lines = [
        "# 第3章 分层分类相关统计表",
        "",
        "## 表2 分层分类结构验证结果",
        "",
        "| 变体编号 | 粗分支联合准确率 | 肩部分支准确率 | 肘部分支准确率 | 腕部分支准确率 | 分支内16类Top-1准确率 | 分支内16类Top-3准确率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['variant']} | {row['branch_joint_acc']} | {row['shoulder_acc']} | "
            f"{row['elbow_acc']} | {row['wrist_acc']} | {row['fine_top1_acc']} | {row['fine_top3_acc']} |"
        )
    md_lines.append("")
    return rows, "\n".join(md_lines)


def main() -> None:
    flat_rows, flat_md = build_flat_192_table()
    hier_rows, hier_md = build_hierarchical_comparison_table()

    write_csv(
        DATA_DIR / "chapter3_flat192_baseline_summary.csv",
        fieldnames=[
            "variant",
            "train_samples",
            "val_samples",
            "best_val_loss",
            "top1_acc",
            "top1_acc_percent",
        ],
        rows=flat_rows,
    )
    write_markdown(DATA_DIR / "chapter3_flat192_baseline_summary.md", flat_md)

    write_csv(
        DATA_DIR / "chapter3_hierarchical_classification_summary.csv",
        fieldnames=[
            "variant",
            "branch_joint_acc",
            "shoulder_acc",
            "elbow_acc",
            "wrist_acc",
            "fine_top1_acc",
            "fine_top3_acc",
        ],
        rows=hier_rows,
    )
    write_markdown(DATA_DIR / "chapter3_hierarchical_classification_summary.md", hier_md)

    print("Generated chapter 3 baseline tables:")
    print(DATA_DIR / "chapter3_flat192_baseline_summary.csv")
    print(DATA_DIR / "chapter3_flat192_baseline_summary.md")
    print(DATA_DIR / "chapter3_hierarchical_classification_summary.csv")
    print(DATA_DIR / "chapter3_hierarchical_classification_summary.md")


if __name__ == "__main__":
    main()
