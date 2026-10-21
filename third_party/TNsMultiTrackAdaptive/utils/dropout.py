"""量测丢失注入工具（用于验证抗临时丢失能力）。

背景：``meas.csv`` 没有目标 id（只有 ``frame, x, y``），因此"每个目标每帧以概率 p
丢失量测"必须先做**量测 ↔ 真值目标**的归属，再按 ``(帧, 目标)`` 独立抛硬币。

用法（见 ``examples/make_dropout.py``）::

    python TNsMultiTrack/examples/make_dropout.py --p 0.2 --seed 0 \
        --out meas_dropout20.csv --mask dropout20_mask.csv

产物：
- 处理后的量测文件（同 ``frame, x, y`` 结构，``correct_x/correct_y`` 一并保留）；
- **dropout 掩码** ``frame, gt_id``：被注入丢失的 (帧, 目标) 对，供评估统计
  "丢失期间是否仍被航迹覆盖"与"恢复后是否还是同一条航迹"。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np

from ..typedef.meas import Measurement
from .io import load_measurements, load_truth, min_frame

Position = np.ndarray


@dataclass
class DropoutStats:
    """注入结果统计。"""

    probability: float = 0.0
    seed: int = 0
    gate: float = float("nan")
    total_measurements: int = 0        # 原始量测条数
    kept_measurements: int = 0         # 处理后保留条数
    target_slots: int = 0              # (帧, 目标) 槽位总数（真值里出现的目标）
    already_missing: int = 0           # 真实数据本身就缺量测的槽位
    injected: int = 0                  # 本次注入丢失的槽位
    clutter: int = 0                   # 未归属到任何目标的量测（离群点）
    injected_clutter: int = 0          # 本次注入的近距杂波点数
    run_lengths: List[int] = field(default_factory=list)   # 连续丢失run长度

    @property
    def injected_rate(self) -> float:
        """在"原本有量测的槽位"上实际注入的比例（应接近 probability）。"""
        available = self.target_slots - self.already_missing
        return self.injected / available if available else 0.0

    def summary(self) -> str:
        longest = max(self.run_lengths) if self.run_lengths else 0
        mean = float(np.mean(self.run_lengths)) if self.run_lengths else 0.0
        return "\n".join(
            [
                f"注入概率            : {self.probability}（seed={self.seed}）",
                f"归属门限            : {self.gate} m",
                f"量测条数            : {self.total_measurements} → {self.kept_measurements}",
                f"(帧,目标) 槽位      : {self.target_slots}",
                f"其中真实数据已缺测  : {self.already_missing}",
                f"本次注入丢失        : {self.injected}（实际比例 {self.injected_rate:.4f}）",
                f"未归属量测(离群/杂波): {self.clutter}（全部保留）",
                f"注入近距杂波        : {self.injected_clutter}",
                f"连续丢失 run        : {len(self.run_lengths)} 段，"
                f"最长 {longest} 帧，均值 {mean:.2f} 帧",
            ]
        )


def match_measurements(
    detections: Sequence[Measurement],
    targets: Sequence[Tuple[int, Position]],
    gate: float,
) -> Tuple[Dict[int, int], List[int]]:
    """把一帧量测唯一地分配给真值目标。

    Returns
    -------
    ``(assignment, clutter)``：``{gt_id: 量测下标}`` 与未归属的量测下标。
    门限外的量测不分配（视为杂波，保留原样）。
    """
    if not detections or not targets:
        return {}, list(range(len(detections)))

    cost = np.zeros((len(targets), len(detections)))
    for i, (_, position) in enumerate(targets):
        for j, detection in enumerate(detections):
            cost[i, j] = float(np.linalg.norm(position - detection.value))

    pairs: List[Tuple[int, int]] = []
    if cost.size:
        try:
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(cost)
            pairs = [(int(r), int(c)) for r, c in zip(rows, cols)]
        except ImportError:  # pragma: no cover - 环境相关
            used_rows: Set[int] = set()
            used_cols: Set[int] = set()
            for flat in np.argsort(cost, axis=None):
                row, col = divmod(int(flat), cost.shape[1])
                if row in used_rows or col in used_cols:
                    continue
                used_rows.add(row)
                used_cols.add(col)
                pairs.append((row, col))

    assignment: Dict[int, int] = {}
    used: Set[int] = set()
    for row, col in pairs:
        if cost[row, col] > gate:
            continue
        assignment[targets[row][0]] = col
        used.add(col)
    clutter = [j for j in range(len(detections)) if j not in used]
    return assignment, clutter


def inject_dropout(
    frames: Sequence[Tuple[float, List[Measurement]]],
    truth: Mapping[float, Sequence[Tuple[int, Position]]],
    probability: float = 0.2,
    seed: int = 0,
    gate: float = 2.0,
    clutter_per_frame: int = 0,
    clutter_range: Tuple[float, float] = (0.5, 3.0),
) -> Tuple[List[Tuple[float, Position]], Dict[float, Set[int]], DropoutStats]:
    """按 ``(帧, 目标)`` 独立伯努利注入量测丢失（可选叠加近距杂波）。

    Parameters
    ----------
    clutter_per_frame:
        每帧额外注入的**近距杂波**点数（默认 0）。杂波点放在随机目标周围
        ``clutter_range`` 米内，用来检验"门限放宽后会不会抓错点"。
    clutter_range:
        杂波到目标距离的区间（米）。

    Returns
    -------
    ``(kept, dropout_map, stats)``：
    - ``kept``：保留的量测 ``[(timestamp, position), ...]``；
    - ``dropout_map``：``{timestamp: {gt_id, ...}}``，本次注入丢失的槽位。
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability 必须在 [0, 1] 内")
    rng = np.random.default_rng(seed)

    stats = DropoutStats(probability=probability, seed=seed, gate=gate)
    stats.total_measurements = sum(len(detections) for _, detections in frames)
    kept: List[Tuple[float, Position]] = []
    dropout_map: Dict[float, Set[int]] = {}
    lost_slots: Dict[int, List[bool]] = {}          # gt_id -> 逐帧是否丢失（用于 run 统计）

    for timestamp, detections in frames:
        targets = list(truth.get(timestamp, []))
        stats.target_slots += len(targets)
        assignment, clutter = match_measurements(detections, targets, gate)
        stats.clutter += len(clutter)

        dropped_targets: Set[int] = set()
        for gt_id, _ in targets:
            if rng.random() < probability:
                dropped_targets.add(gt_id)

        # 掩码语义：记录**所有"该目标本帧没有量测"的槽位**，并区分来源：
        #   - 原本就缺测（真实数据里的漏检）→ already_missing
        #   - 本次注入删除            → injected
        # 评估侧需要用它来量化"丢失期间航迹是否仍在覆盖该目标"。
        for gt_id, _ in targets:
            if gt_id not in assignment:
                stats.already_missing += 1
                dropout_map.setdefault(timestamp, set()).add(gt_id)
            elif gt_id in dropped_targets:
                stats.injected += 1
                dropout_map.setdefault(timestamp, set()).add(gt_id)
            else:
                kept.append((timestamp, detections[assignment[gt_id]].value))
        for index in clutter:
            kept.append((timestamp, detections[index].value))   # 未归属量测原样保留

        # 可选：注入近距杂波（放在随机目标周围 clutter_range 米内）
        for _ in range(max(0, int(clutter_per_frame))):
            if not targets:
                break
            _, anchor = targets[int(rng.integers(len(targets)))]
            radius = float(rng.uniform(*clutter_range))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            kept.append((timestamp, anchor + offset))
            stats.injected_clutter += 1

        for gt_id, _ in targets:
            lost_slots.setdefault(gt_id, []).append(gt_id in dropout_map.get(timestamp, ()))

    stats.kept_measurements = len(kept)
    # 把"真实数据本身缺测"的槽位也记进掩码，方便评估区分两类丢失
    for gt_id, flags in lost_slots.items():
        run = 0
        for flag in flags:
            if flag:
                run += 1
            elif run:
                stats.run_lengths.append(run)
                run = 0
        if run:
            stats.run_lengths.append(run)
    return kept, dropout_map, stats


def write_measurements(
    path: Path,
    kept: Sequence[Tuple[float, Position]],
    ts: float,
    frame_origin: int,
) -> None:
    """写出处理后的量测 CSV（``frame, x, y``）。"""
    rows = [(int(round(timestamp / ts)) + frame_origin, position) for timestamp, position in kept]
    rows.sort(key=lambda item: item[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x", "y"])
        for frame, position in rows:
            writer.writerow([frame, f"{position[0]:.6f}", f"{position[1]:.6f}"])


def write_dropout_mask(
    path: Path,
    dropout_map: Mapping[float, Set[int]],
    ts: float,
    frame_origin: int,
) -> None:
    """写出 dropout 掩码（``frame, gt_id``）。"""
    rows = [
        (int(round(timestamp / ts)) + frame_origin, gt_id)
        for timestamp, ids in dropout_map.items()
        for gt_id in sorted(ids)
    ]
    rows.sort()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "gt_id"])
        writer.writerows(rows)


def load_dropout_mask(path: Path, ts: float, frame_origin: int) -> Dict[float, Set[int]]:
    """读取 dropout 掩码，返回 ``{timestamp: {gt_id, ...}}``。

    ``frame_origin`` **必须**与读取量测/真值时的原点一致（都是二者的最小帧）。
    掩码文件里的最小帧通常大于数据原点（前几帧没发生丢失），若隐式用掩码自身的最小帧，
    整张掩码会静默错位——这是实测踩过的坑，故此处不提供默认值。
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return {}
    mask: Dict[float, Set[int]] = {}
    for row in rows:
        timestamp = (int(float(row["frame"])) - frame_origin) * ts
        mask.setdefault(timestamp, set()).add(int(float(row["gt_id"])))
    return mask


def build_dataset(
    meas_path: Path,
    true_path: Path,
    probability: float = 0.2,
    seed: int = 0,
    ts: float = 0.1,
    gate: float = 2.0,
    out_path: Path | None = None,
    mask_path: Path | None = None,
    clutter_per_frame: int = 0,
    clutter_range: Tuple[float, float] = (0.5, 3.0),
) -> DropoutStats:
    """一步到位：读原始量测/真值 → 注入丢失（可叠加近距杂波）→ 写出量测与掩码。"""
    origin = min(min_frame(meas_path), min_frame(true_path))
    _, frames = load_measurements(meas_path, ts=ts, frame_origin=origin)
    truth = load_truth(true_path, ts=ts, frame_origin=origin)

    kept, dropout_map, stats = inject_dropout(
        frames, truth, probability, seed, gate,
        clutter_per_frame=clutter_per_frame, clutter_range=clutter_range,
    )
    if out_path is not None:
        write_measurements(out_path, kept, ts, origin)
    if mask_path is not None:
        write_dropout_mask(mask_path, dropout_map, ts, origin)
    return stats


__all__ = [
    "DropoutStats",
    "build_dataset",
    "inject_dropout",
    "load_dropout_mask",
    "match_measurements",
    "write_dropout_mask",
    "write_measurements",
]
