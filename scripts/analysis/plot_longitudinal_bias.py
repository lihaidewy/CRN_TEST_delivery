#!/usr/bin/env python3
"""Plot the signed longitudinal bias of CRN predictions.

Expected input columns in ``tracking_preds.csv``:
    frame_id, score, x, y, gt_x, gt_y

Only rows with a matched ground-truth coordinate (non-empty ``gt_x`` and
``gt_y``) are used.  The plotted signed longitudinal error is:

    e_x = x_pred - x_gt

Positive values mean that the predicted object is farther along the road than
its matched ground truth; negative values mean that it is predicted nearer.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 使用前只修改这一处配置；不需要在运行命令中传入任何参数。
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 当前绘制：512x1920 几何修正后的高分辨率模型，S2M（中度），score=0.35。
INPUT_CSV = PROJECT_ROOT / "TEST/results/current/highres_nodepth/raw/S2M/tracking_preds.csv"
OUT_PREFIX = PROJECT_ROOT / "TEST/results/current/highres_nodepth/analysis/longitudinal_bias/S2M/longitudinal_bias_scatter"
# 如需绘制 S2S（重度），替换为以下两行：
# INPUT_CSV = PROJECT_ROOT / "TEST/results/current/highres_nodepth/raw/S2S/tracking_preds.csv"
# OUT_PREFIX = PROJECT_ROOT / "TEST/results/current/highres_nodepth/analysis/longitudinal_bias/S2S/longitudinal_bias_scatter"

BIN_WIDTH_M = 10.0        # 用于绘制分箱均值的距离宽度（m）
MIN_SCORE: float | None = None  # 一般保持 None；CSV 已按正式评估阈值筛选


def read_tp_rows(path: Path, min_score: float | None) -> list[dict[str, float | str]]:
    """Read matched prediction rows and calculate signed planar errors."""
    required = {"x", "y", "gt_x", "gt_y"}
    records: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns: {', '.join(sorted(required))}; "
                f"found: {reader.fieldnames}"
            )
        for row in reader:
            if not row.get("gt_x", "").strip() or not row.get("gt_y", "").strip():
                continue
            score = float(row.get("score", "nan"))
            if min_score is not None and (not np.isfinite(score) or score < min_score):
                continue
            pred_x, pred_y = float(row["x"]), float(row["y"])
            gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
            records.append({
                "frame_id": row.get("frame_id", row.get("frame", "")),
                "gt_token": row.get("gt_token", row.get("gt_id", "")),
                "score": score,
                "gt_x": gt_x,
                "gt_y": gt_y,
                "pred_x": pred_x,
                "pred_y": pred_y,
                "signed_dx": pred_x - gt_x,
                "signed_dy": pred_y - gt_y,
                "ate_2d": float(np.hypot(pred_x - gt_x, pred_y - gt_y)),
            })
    return records


def save_points(path: Path, records: list[dict[str, float | str]]) -> None:
    fields = [
        "frame_id", "gt_token", "score", "gt_x", "gt_y", "pred_x", "pred_y",
        "signed_dx", "signed_dy", "ate_2d",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def plot_bias(records: list[dict[str, float | str]], out_prefix: Path, bin_width: float,
              title: str = "CRN longitudinal prediction bias") -> tuple[float, float]:
    """Draw scatter, 10/50 m bin means, and an ordinary least-squares trend."""
    gt_x = np.asarray([float(row["gt_x"]) for row in records])
    signed_dx = np.asarray([float(row["signed_dx"]) for row in records])
    slope, intercept = np.polyfit(gt_x, signed_dx, 1)
    corr = float(np.corrcoef(gt_x, signed_dx)[0, 1]) if len(records) > 1 else float("nan")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(10.4, 6.2), constrained_layout=True)
    ax.scatter(gt_x, signed_dx, s=13, alpha=0.16, color="#3B6EA5", linewidths=0,
               label=f"Matched predictions (n={len(records)})", rasterized=True)

    x_line = np.linspace(float(gt_x.min()), float(gt_x.max()), 300)
    ax.plot(x_line, slope * x_line + intercept, color="#C75B39", linewidth=2.5,
            label=rf"Linear fit: $e_x={intercept:+.3f}{slope:+.4f}x_{{gt}}$")

    edges = np.arange(np.floor(gt_x.min() / bin_width) * bin_width,
                      np.ceil(gt_x.max() / bin_width) * bin_width + bin_width,
                      bin_width)
    centers, means = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        values = signed_dx[(gt_x >= low) & (gt_x < high)]
        if len(values) >= 5:
            centers.append((low + high) / 2.0)
            means.append(float(values.mean()))
    if centers:
        ax.plot(centers, means, "o-", color="#1B4D7A", linewidth=1.5, markersize=5,
                label=f"{bin_width:g} m bin mean")

    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xlabel(r"Ground-truth longitudinal position $x_{gt}$ (m)")
    ax.set_ylabel(r"Signed longitudinal error $e_x=x_{pred}-x_{gt}$ (m)")
    ax.set_title(title)
    ax.grid(axis="y", color="#D9DDE3", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.99, 0.02,
        f"mean = {signed_dx.mean():+.3f} m\nmedian = {np.median(signed_dx):+.3f} m\n"
        f"P(eₓ > 0) = {(signed_dx > 0).mean():.1%}\nPearson r = {corr:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BFC7D5", "alpha": 0.92},
    )

    for suffix in ("png", "pdf"):
        fig.savefig(out_prefix.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return slope, intercept


def main() -> None:
    if BIN_WIDTH_M <= 0:
        raise ValueError("BIN_WIDTH_M must be positive")
    records = read_tp_rows(INPUT_CSV, MIN_SCORE)
    if not records:
        raise RuntimeError(f"No matched rows found in {INPUT_CSV}. Check that gt_x and gt_y are populated.")
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    points_path = OUT_PREFIX.with_name(f"{OUT_PREFIX.name}_points.csv")
    save_points(points_path, records)
    slope, intercept = plot_bias(records, OUT_PREFIX, BIN_WIDTH_M)
    signed_dx = np.asarray([float(row["signed_dx"]) for row in records])
    print(f"[OK] Input file: {INPUT_CSV}")
    print(f"[OK] Input TP rows: {len(records)}")
    print(f"[OK] Mean signed longitudinal error: {signed_dx.mean():+.4f} m")
    print(f"[OK] Linear trend: e_x = {intercept:+.6f}{slope:+.6f} * gt_x")
    print(f"[OK] Figure: {OUT_PREFIX.with_suffix('.png')}")
    print(f"[OK] Point-wise errors: {points_path}")


if __name__ == "__main__":
    main()
