"""集中式航迹起始模块（M/N 逻辑法）。

设计依据：``ALGORITHM_DESIGN.md`` §5.4。

沿用 MATLAB ``Track_Init`` 的"未关联量测点池 + N 帧滑窗 + M/N 确认"思路，
并修掉它的三处工程缺陷：

1. **墓碑代替物理删除**——量测下标永远有效；
2. **双上限剪枝**（``max_branch`` / ``max_candidates``）——消灭指数级递归；
3. **候选冲突消解**——按质量降序贪心接受，两条新航迹不共享量测。

评审意见 ⑤：起批判据仍在滑窗上做，但候选**末点必须位于当前帧**，
新航迹由 :meth:`Track._born` 从确认帧开始记录，**不回填历史**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from ..typedef.meas import Measurement
from ..typedef.motion import MotionModel
from ..typedef.track import Track

_TIME_TOLERANCE = 1e-9


@dataclass
class FrameRecord:
    """滑窗中的一帧：只保存**未被关联**的量测。"""

    frame: int
    timestamp: float
    measurements: tuple[Measurement, ...]
    tombstone: list[bool]


@dataclass
class _Branch:
    """一条部分航迹假设。"""

    seed: tuple[int, int]
    points: list[Measurement] = field(default_factory=list)
    indices: list[tuple[int, int]] = field(default_factory=list)

    def extended(self, point: Measurement, index: tuple[int, int]) -> "_Branch":
        return _Branch(self.seed, self.points + [point], self.indices + [index])


@dataclass
class Candidate:
    """满足 M/N 的起始候选（证据链）。"""

    points: list[Measurement]
    indices: list[tuple[int, int]]
    quality: float
    span: int
    hit_ratio: float


class MNInitializer:
    """集中式 M/N 起始器。

    Parameters
    ----------
    model:
        默认模型实例（模型是无状态的，可被所有新航迹共享，M5）。
    model_for:
        可选：按候选返回模型的回调，用于将来"异构/升模"；默认返回 ``model``。
    window_size / confirm:
        即 M/N 逻辑中的 N 与 M。
    frame_interval:
        标称帧间隔（秒），仅用于质量评分里的命中率；实际门限按真实时间差缩放。
    """

    def __init__(
        self,
        model: MotionModel,
        *,
        window_size: int = 4,
        confirm: int = 3,
        patience: int = 7,
        v_min: float = 0.0,
        v_max: float = 12.0,
        a_max: float = 200.0,
        angle_max: float = 45.0,
        max_branch: int = 8,
        max_candidates: int = 64,
        weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        frame_interval: float = 0.1,
        model_for: Optional[Callable[[Candidate], MotionModel]] = None,
        seed_measurements: bool = True,
    ) -> None:
        if confirm < 2:
            raise ValueError("confirm (M) 必须 ≥ 2")
        if window_size <= confirm:
            raise ValueError("window_size (N) 必须 > confirm (M)，与 MATLAB 的 N>M>2 一致")
        if not 0.0 <= v_min < v_max:
            raise ValueError("需要 0 <= v_min < v_max")
        self.model = model
        self.model_for = model_for
        self.seed_measurements = bool(seed_measurements)
        self.window_size = int(window_size)
        self.confirm = int(confirm)
        self.patience = int(patience)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.angle_max = float(angle_max)
        self.max_branch = int(max_branch)
        self.max_candidates = int(max_candidates)
        self.weights = weights
        self.frame_interval = float(frame_interval)

        self.window: list[FrameRecord] = []
        self.tombstoned: set[tuple[int, int]] = set()
        self.last_candidate_count = 0

    # ── 主入口 ──────────────────────────────────────────────────────────────
    def update(
        self, unmatched: Sequence[Measurement], timeStamp: float, frame: int
    ) -> list[Track]:
        """收下本帧未被关联的量测，返回本帧新生成的航迹（可为空）。"""
        record = FrameRecord(int(frame), float(timeStamp), tuple(unmatched), [False] * len(unmatched))
        self.window.append(record)
        if len(self.window) > self.window_size:
            dropped = self.window.pop(0)
            self._purge_tombstones(dropped)

        if len(self.window) < self.window_size:
            self.last_candidate_count = 0
            return []

        candidates = self._enumerate(timeStamp)
        self.last_candidate_count = len(candidates)
        return self._materialize(candidates, timeStamp)

    # ── 枚举候选 ────────────────────────────────────────────────────────────
    def _enumerate(self, timeStamp: float) -> list[Candidate]:
        records = self.window
        branches = self._seed(records[0], records[1])
        for index in range(2, len(records)):
            if not branches:
                break
            branches = self._extend(branches, records[index])
        return self._collect(branches, timeStamp)

    def _seed(self, first: FrameRecord, second: FrameRecord) -> list[_Branch]:
        gap = second.timestamp - first.timestamp
        if gap <= 0.0:
            return []
        branches: list[_Branch] = []
        for i, point_0 in enumerate(first.measurements):
            if first.tombstone[i] or (first.frame, i) in self.tombstoned:
                continue
            for j, point_1 in enumerate(second.measurements):
                if second.tombstone[j] or (second.frame, j) in self.tombstoned:
                    continue
                speed = float(np.linalg.norm(point_1.value - point_0.value)) / gap
                if not self.v_min <= speed <= self.v_max:
                    continue
                branches.append(
                    _Branch(
                        seed=(i, j),
                        points=[point_0, point_1],
                        indices=[(first.frame, i), (second.frame, j)],
                    )
                )
        return self._prune(branches)

    def _extend(self, branches: list[_Branch], record: FrameRecord) -> list[_Branch]:
        children: list[_Branch] = []
        for branch in branches:
            last = branch.points[-1]
            gap = record.timestamp - last.timestamp
            if gap <= 0.0:
                children.append(branch)
                continue
            predicted = self._predict(branch.points, record.timestamp)
            matched = 0
            for index, measurement in enumerate(record.measurements):
                if record.tombstone[index] or (record.frame, index) in self.tombstoned:
                    continue
                if float(np.linalg.norm(measurement.value - predicted)) > self.v_max * gap:
                    continue
                if not self._kinematic_ok(branch.points, measurement):
                    continue
                children.append(branch.extended(measurement, (record.frame, index)))
                matched += 1
            if matched == 0:
                children.append(branch)          # 允许漏检：分支原样前进一帧
        return self._prune(children)

    def _collect(self, branches: list[_Branch], timeStamp: float) -> list[Candidate]:
        candidates: list[Candidate] = []
        for branch in branches:
            if len(branch.points) < self.confirm:
                continue
            if abs(branch.points[-1].timestamp - timeStamp) > _TIME_TOLERANCE:
                continue                          # 评审意见 ⑤：末点必须在当前帧
            quality, hit_ratio = self._quality(branch)
            candidates.append(
                Candidate(
                    points=list(branch.points),
                    indices=list(branch.indices),
                    quality=quality,
                    span=branch.indices[-1][0] - branch.indices[0][0] + 1,
                    hit_ratio=hit_ratio,
                )
            )
        return self._dedupe(candidates)

    # ── 生成航迹 ────────────────────────────────────────────────────────────
    def _materialize(self, candidates: list[Candidate], timeStamp: float) -> list[Track]:
        tracks: list[Track] = []
        for candidate in sorted(candidates, key=lambda c: -c.quality):
            if any(index in self.tombstoned for index in candidate.indices):
                continue                          # 冲突消解：点已被更高分候选占用
            model = self.model_for(candidate) if self.model_for else self.model
            state0 = model.initial_state(candidate.points, timeStamp)
            tracks.append(
                Track._born(
                    state0,
                    model,
                    self.patience,
                    first_evidence_time=candidate.points[0].timestamp,
                    evidence=tuple(candidate.points),
                    quality=candidate.quality,
                    span=candidate.span,
                    seed_measurements=self.seed_measurements,
                )
            )
            self.tombstoned.update(candidate.indices)
        return tracks

    # ── 几何 / 运动学 ───────────────────────────────────────────────────────
    @staticmethod
    def _predict(points: list[Measurement], timeStamp: float) -> np.ndarray:
        """由最后两点线性外推到 ``timeStamp``。"""
        last, previous = points[-1], points[-2]
        gap = last.timestamp - previous.timestamp
        if gap <= 0.0:
            return last.value.copy()
        velocity = (last.value - previous.value) / gap
        return last.value + velocity * (timeStamp - last.timestamp)

    def _kinematic_ok(self, points: list[Measurement], candidate: Measurement) -> bool:
        if len(points) < 2:
            return True
        last, previous = points[-1], points[-2]
        gap = last.timestamp - previous.timestamp
        next_gap = candidate.timestamp - last.timestamp
        if gap <= 0.0 or next_gap <= 0.0:
            return True

        previous_vector = last.value - previous.value
        vector = candidate.value - last.value
        norm_previous = float(np.linalg.norm(previous_vector))
        norm_vector = float(np.linalg.norm(vector))
        if norm_previous > 0.0 and norm_vector > 0.0:
            cosine = float(
                np.dot(previous_vector, vector) / (norm_previous * norm_vector)
            )
            angle = float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))
            if angle > self.angle_max:
                return False

        speed_previous = norm_previous / gap
        speed = norm_vector / next_gap
        if abs(speed - speed_previous) / next_gap > self.a_max:
            return False
        return True

    # ── 质量评分（§5.4.4）────────────────────────────────────────────────────
    def _quality(self, branch: _Branch) -> tuple[float, float]:
        points = branch.points
        if len(points) < 2:
            return 0.0, 0.0
        speeds: list[float] = []
        accelerations: list[float] = []
        cosines: list[float] = []
        for index in range(1, len(points)):
            gap = points[index].timestamp - points[index - 1].timestamp
            if gap <= 0.0:
                continue
            distance = float(np.linalg.norm(points[index].value - points[index - 1].value))
            speeds.append(distance / gap)
            if index >= 2:
                previous_gap = points[index - 1].timestamp - points[index - 2].timestamp
                if previous_gap <= 0.0:
                    continue
                previous_distance = float(
                    np.linalg.norm(points[index - 1].value - points[index - 2].value)
                )
                accelerations.append(abs(distance / gap - previous_distance / previous_gap) / gap)
                first = points[index - 1].value - points[index - 2].value
                second = points[index].value - points[index - 1].value
                norm_first, norm_second = float(np.linalg.norm(first)), float(np.linalg.norm(second))
                if norm_first > 0.0 and norm_second > 0.0:
                    cosines.append(
                        float(np.dot(first, second) / (norm_first * norm_second))
                    )

        center = (self.v_min + self.v_max) / 2.0
        speed_score = 1.0 - min(1.0, abs(float(np.mean(speeds)) - center) / center)
        acceleration_score = 1.0 - min(1.0, float(np.mean(accelerations)) / self.a_max) if accelerations else 1.0
        straightness = float(np.mean([max(0.0, c) for c in cosines])) if cosines else 1.0
        span = branch.indices[-1][0] - branch.indices[0][0] + 1
        hit_ratio = len(points) / max(span, 1)

        w_v, w_a, w_c, w_h = self.weights
        quality = (
            w_v * max(0.0, speed_score)
            + w_a * max(0.0, acceleration_score)
            + w_c * straightness
            + w_h * hit_ratio
        )
        return float(quality), float(hit_ratio)

    # ── 剪枝 / 去重 / 墓碑 ──────────────────────────────────────────────────
    def _prune(self, branches: list[_Branch]) -> list[_Branch]:
        if not branches:
            return []
        grouped: dict[tuple[int, int], list[_Branch]] = {}
        for branch in branches:
            grouped.setdefault(branch.seed, []).append(branch)
        kept: list[_Branch] = []
        for group in grouped.values():
            if len(group) > self.max_branch:
                group = sorted(group, key=lambda b: -self._quality(b)[0])[: self.max_branch]
            kept.extend(group)
        if len(kept) > self.max_candidates:
            kept = sorted(kept, key=lambda b: -self._quality(b)[0])[: self.max_candidates]
        return kept

    @staticmethod
    def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
        best: dict[tuple[tuple[int, int], ...], Candidate] = {}
        for candidate in candidates:
            key = tuple(candidate.indices)
            current = best.get(key)
            if current is None or candidate.quality > current.quality:
                best[key] = candidate
        return list(best.values())

    def _purge_tombstones(self, dropped: FrameRecord) -> None:
        self.tombstoned = {index for index in self.tombstoned if index[0] != dropped.frame}


__all__ = ["Candidate", "FrameRecord", "MNInitializer"]
