"""跟踪评估指标（行为化验收，见设计报告 §11）。

不要求与 MATLAB 数值等价，所以这里评的是"跟踪得好不好"：
逐帧用匈牙利把估计与真值配对（带门限），再统计
TP/FP/FN、ID 切换、碎片化、位置 RMSE 与 MOTA。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

Position = np.ndarray


@dataclass
class PerIdStats:
    """单个真值目标的统计。"""

    gt_id: int
    frames: int = 0
    matched: int = 0
    first_frame: float = float("nan")
    last_frame: float = float("nan")
    errors: List[float] = field(default_factory=list)
    track_ids: List[int] = field(default_factory=list)
    switches: int = 0
    fragments: int = 0

    @property
    def coverage(self) -> float:
        return self.matched / self.frames if self.frames else 0.0

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.errors)))) if self.errors else float("nan")

    def percentile_error(self, percentile: float) -> float:
        """位置误差的任意分位点（米）。"""
        if not self.errors:
            return float("nan")
        return float(np.percentile(self.errors, percentile))

    @property
    def p90(self) -> float:
        """位置误差 P90（米）——对长尾比 RMSE 更稳健的精度指标。"""
        return self.percentile_error(90.0)

    @property
    def p95(self) -> float:
        """位置误差 P95（米）。"""
        return self.percentile_error(95.0)

    @property
    def distinct_tracks(self) -> int:
        return len(set(self.track_ids))


@dataclass
class EvaluationReport:
    """整段序列的评估结果。"""

    frames: int = 0
    gt_points: int = 0
    matched: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    id_switches: int = 0
    fragments: int = 0
    errors: List[float] = field(default_factory=list)
    frame_errors: Dict[float, float] = field(default_factory=dict)
    per_id: Dict[int, PerIdStats] = field(default_factory=dict)
    track_count: int = 0
    gate: float = float("nan")

    # ── 抗临时丢失（dropout）统计：仅在传入 dropout 掩码时有效 ───────────────
    dropout_points: int = 0            # 目标无实测量测的 (帧,目标) 槽位数
    dropout_covered: int = 0           # 其中仍被某条航迹覆盖（滑行预测）的槽位数
    dropout_events: int = 0            # 连续丢失事件段数
    dropout_recovered_same_id: int = 0 # 恢复时仍由**同一条航迹**接上的事件数
    dropout_recovered_any: int = 0     # 恢复时被任意航迹接上的事件数

    @property
    def dropout_coverage(self) -> float:
        """丢失期间仍被覆盖的比例——衡量"滑行外推能不能撑住"。"""
        return self.dropout_covered / self.dropout_points if self.dropout_points else float("nan")

    @property
    def dropout_same_id_rate(self) -> float:
        """恢复后仍是同一条航迹的比例——衡量"身份有没有断"。"""
        if not self.dropout_events:
            return float("nan")
        return self.dropout_recovered_same_id / self.dropout_events

    @property
    def precision(self) -> float:
        denominator = self.matched + self.false_positives
        return self.matched / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.gt_points if self.gt_points else 0.0

    @property
    def mota(self) -> float:
        if not self.gt_points:
            return float("nan")
        return 1.0 - (self.false_negatives + self.false_positives + self.id_switches) / self.gt_points

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.errors)))) if self.errors else float("nan")

    @property
    def mean_error(self) -> float:
        return float(np.mean(self.errors)) if self.errors else float("nan")

    def percentile_error(self, percentile: float) -> float:
        """位置误差的任意分位点（米）。"""
        if not self.errors:
            return float("nan")
        return float(np.percentile(self.errors, percentile))

    @property
    def p90_error(self) -> float:
        """位置误差 P90（米）——对长尾比 RMSE 更稳健的精度指标。"""
        return self.percentile_error(90.0)

    @property
    def p95_error(self) -> float:
        return self.percentile_error(95.0)

    def summary(self) -> str:
        lines = [
            f"评估帧数            : {self.frames}",
            f"真值点总数          : {self.gt_points}",
            f"匹配成功 (TP)       : {self.matched}",
            f"虚警 (FP)           : {self.false_positives}",
            f"漏检 (FN)           : {self.false_negatives}",
            f"ID 切换             : {self.id_switches}",
            f"航迹碎片            : {self.fragments}",
            f"航迹总数            : {self.track_count}",
            f"准确率 / 召回率     : {self.precision:.4f} / {self.recall:.4f}",
            f"MOTA                : {self.mota:.4f}",
            f"位置 RMSE           : {self.rmse:.4f} m",
            f"位置平均误差        : {self.mean_error:.4f} m",
            f"位置误差 P90        : {self.p90_error:.4f} m",
            f"位置误差 P95        : {self.p95_error:.4f} m",
        ]
        if self.dropout_points:
            lines += [
                f"丢失槽位(无实测量测) : {self.dropout_points}",
                f"  丢失期间仍被覆盖   : {self.dropout_covered}（覆盖率 "
                f"{self.dropout_coverage:.4f}）← 滑行外推的贡献",
                f"  丢失事件 / 同 ID 恢复: {self.dropout_events} / "
                f"{self.dropout_recovered_same_id}（同 ID 率 {self.dropout_same_id_rate:.4f}）；"
                f"任意航迹恢复 {self.dropout_recovered_any}",
            ]
        return "\n".join(lines)


def _assign(cost: np.ndarray) -> List[Tuple[int, int]]:
    """最小代价一对一指派（优先 scipy，缺失时退化为贪心）。"""
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]
    except ImportError:  # pragma: no cover - 环境相关
        pairs: List[Tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for flat in np.argsort(cost, axis=None):
            row, col = divmod(int(flat), cost.shape[1])
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            pairs.append((row, col))
        return pairs


def evaluate(
    estimates: Mapping[float, Mapping[int, Position]],
    truth: Mapping[float, Sequence[Tuple[int, Position]]],
    gate: float = 2.0,
    dropout: Mapping[float, set] | None = None,
    recovery_window: int = 1,
) -> EvaluationReport:
    """逐帧配对并统计指标。

    Parameters
    ----------
    estimates:
        ``{timestamp: {track_id: position}}``。
    truth:
        ``{timestamp: [(gt_id, position), ...]}``。
    gate:
        配对门限（米）；超过门限的配对不算命中。
    dropout:
        可选，``{timestamp: {gt_id, ...}}``——该目标在该帧**没有实测量测**
        （由 ``utils/dropout.py`` 生成）。传入后额外统计抗临时丢失指标：
        丢失期间是否仍被覆盖、恢复后是否还是同一条航迹。
    recovery_window:
        丢失结束后允许几帧内接上就算"恢复"。
    """
    report = EvaluationReport(gate=gate)
    previous_track: Dict[int, int] = {}
    last_matched_frame: Dict[int, float] = {}
    track_ids_seen: set[int] = set()
    matched_track: Dict[Tuple[float, int], int] = {}          # (帧, gt) -> 航迹 id
    truth_frames: Dict[int, List[float]] = {}                 # gt -> 出现的帧（升序）

    for timestamp in sorted(truth):
        gt = list(truth[timestamp])
        if not gt:
            continue
        report.frames += 1
        report.gt_points += len(gt)

        frame_estimates = estimates.get(timestamp, {})
        estimate_ids = sorted(frame_estimates)
        track_ids_seen.update(estimate_ids)

        for gt_id, _ in gt:
            stats = report.per_id.setdefault(gt_id, PerIdStats(gt_id=gt_id))
            stats.frames += 1
            if np.isnan(stats.first_frame):
                stats.first_frame = timestamp
            stats.last_frame = timestamp
            truth_frames.setdefault(gt_id, []).append(timestamp)

        if not estimate_ids:
            report.false_negatives += len(gt)
            for gt_id, _ in gt:
                last_matched_frame.pop(gt_id, None)
            continue

        cost = np.zeros((len(estimate_ids), len(gt)))
        for i, track_id in enumerate(estimate_ids):
            position = frame_estimates[track_id]
            for j, (_, gt_position) in enumerate(gt):
                cost[i, j] = float(np.linalg.norm(position - gt_position))

        matched_estimates: set[int] = set()
        matched_gt: set[int] = set()
        frame_errors: List[float] = []
        for i, j in _assign(cost):
            if cost[i, j] > gate:
                continue
            track_id = estimate_ids[i]
            gt_id = gt[j][0]
            error = float(cost[i, j])
            matched_estimates.add(i)
            matched_gt.add(j)
            matched_track[(timestamp, gt_id)] = track_id
            report.matched += 1
            report.errors.append(error)
            frame_errors.append(error)

            stats = report.per_id[gt_id]
            stats.matched += 1
            stats.errors.append(error)
            stats.track_ids.append(track_id)

            previous = previous_track.get(gt_id)
            if previous is not None and previous != track_id:
                stats.switches += 1
                report.id_switches += 1
            previous_track[gt_id] = track_id

            if gt_id in last_matched_frame and timestamp > last_matched_frame[gt_id]:
                gap = round((timestamp - last_matched_frame[gt_id]) / 0.1)
                if gap > 1:
                    stats.fragments += 1
                    report.fragments += 1
            last_matched_frame[gt_id] = timestamp

        report.false_positives += len(estimate_ids) - len(matched_estimates)
        report.false_negatives += len(gt) - len(matched_gt)
        if frame_errors:
            report.frame_errors[timestamp] = float(np.mean(frame_errors))

    report.track_count = len(track_ids_seen)
    if dropout is not None:
        _analyse_dropout(report, dropout, matched_track, truth_frames, recovery_window)
    return report


def _analyse_dropout(
    report: EvaluationReport,
    dropout: Mapping[float, set],
    matched_track: Mapping[Tuple[float, int], int],
    truth_frames: Mapping[int, List[float]],
    recovery_window: int,
) -> None:
    """统计抗临时丢失指标（原地写入 ``report``）。

    - **丢失期间覆盖率**：目标本帧没有实测量测时，是否仍有航迹（滑行预测）落在评估门限内；
    - **同 ID 恢复率**：一段连续丢失结束后，接上的是不是丢失前那条航迹。
    """
    missing = {(timestamp, gt_id) for timestamp, ids in dropout.items() for gt_id in ids}
    report.dropout_points = len(missing)
    report.dropout_covered = sum(1 for key in missing if key in matched_track)

    for gt_id, timestamps in truth_frames.items():
        ordered = sorted(timestamps)
        index = 0
        while index < len(ordered):
            if (ordered[index], gt_id) not in missing:
                index += 1
                continue
            start = index
            while index < len(ordered) and (ordered[index], gt_id) in missing:
                index += 1
            end = index                                            # 丢失段 [start, end)
            report.dropout_events += 1

            before = None
            for position in range(start - 1, -1, -1):
                before = matched_track.get((ordered[position], gt_id))
                if before is not None:
                    break
            after = None
            for position in range(end, min(end + recovery_window, len(ordered))):
                after = matched_track.get((ordered[position], gt_id))
                if after is not None:
                    break
            if after is not None:
                report.dropout_recovered_any += 1
                if before is not None and after == before:
                    report.dropout_recovered_same_id += 1


__all__ = ["EvaluationReport", "PerIdStats", "evaluate"]