"""数据关联结果（契约层）。

设计依据：``ALGORITHM_DESIGN.md`` §4.6 / §5.3.7。

统一使用 0-based 索引；不使用 MATLAB 那种"布尔分配矩阵 + 下标集合反查"的表示。
构造时校验索引分区守恒，避免悄悄丢航迹或丢量测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AssociationResult:
    """一帧的关联结果。

    Parameters
    ----------
    matches:
        ``((track_index, detection_index), ...)``，一对一匹配。
    unmatched_tracks:
        未匹配的航迹下标（0-based）。
    unmatched_detections:
        未匹配的量测下标（0-based），供起始模块使用。
    cost:
        代价矩阵 ``(n_tracks, n_detections)``；可为空矩阵（调试 / 评估用）。
    """

    matches: tuple[tuple[int, int], ...] = ()
    unmatched_tracks: tuple[int, ...] = ()
    unmatched_detections: tuple[int, ...] = ()
    cost: Optional[np.ndarray] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "matches", tuple(tuple(m) for m in self.matches))
        object.__setattr__(self, "unmatched_tracks", tuple(self.unmatched_tracks))
        object.__setattr__(self, "unmatched_detections", tuple(self.unmatched_detections))
        if self.cost is not None:
            object.__setattr__(self, "cost", np.asarray(self.cost, dtype=float))
        self._validate()

    # ── 便捷访问 ────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.matches)

    @property
    def track_to_detection(self) -> dict[int, int]:
        """``{track_index: detection_index}``，供 Tracker 分发结果。"""
        return {t: d for t, d in self.matches}

    @property
    def detection_to_track(self) -> dict[int, int]:
        """``{detection_index: track_index}``。"""
        return {d: t for t, d in self.matches}

    @classmethod
    def all_unmatched(
        cls, num_tracks: int, num_detections: int
    ) -> "AssociationResult":
        """没有任何匹配的结果（``n == 0`` / ``m == 0`` / 全门限外时使用）。"""
        return cls(
            matches=(),
            unmatched_tracks=tuple(range(num_tracks)),
            unmatched_detections=tuple(range(num_detections)),
            cost=np.zeros((num_tracks, num_detections)),
        )

    # ── 校验 ────────────────────────────────────────────────────────────────
    def _validate(self) -> None:
        track_indices = [t for t, _ in self.matches]
        detection_indices = [d for _, d in self.matches]
        if len(set(track_indices)) != len(track_indices):
            raise ValueError("AssociationResult.matches 中航迹下标重复")
        if len(set(detection_indices)) != len(detection_indices):
            raise ValueError("AssociationResult.matches 中量测下标重复")
        for index in track_indices:
            if index < 0:
                raise ValueError("AssociationResult 的航迹下标必须为非负整数")
        for index in detection_indices:
            if index < 0:
                raise ValueError("AssociationResult 的量测下标必须为非负整数")
        overlap = set(track_indices) & set(self.unmatched_tracks)
        if overlap:
            raise ValueError(f"航迹下标同时出现在 matches 与 unmatched_tracks: {sorted(overlap)}")
        overlap = set(detection_indices) & set(self.unmatched_detections)
        if overlap:
            raise ValueError(
                f"量测下标同时出现在 matches 与 unmatched_detections: {sorted(overlap)}"
            )
