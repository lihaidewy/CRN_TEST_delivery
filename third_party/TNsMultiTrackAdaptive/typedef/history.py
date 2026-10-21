"""航迹历史视图（契约层）。

设计依据：``ALGORITHM_DESIGN.md`` §4.3。

为什么需要这个对象：模型需要两类历史，但**不允许持有 ``Track``**。
- KF 类模型只用 ``states``（最新状态 + 协方差容器）；
- TFOT 类模型必须用 ``measurements``——在**原始量测**上做 L0 选阶，
  若把已平滑的状态喂回去会导致选阶偏低并自我强化。

``TrackHistory`` 是冻结值对象：模型拿到它也无法反向修改航迹。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .meas import Measurement
from .state import State


@dataclass(frozen=True)
class TrackHistory:
    """一条航迹在某个时刻的历史快照。

    Parameters
    ----------
    states:
        每帧一个状态（含 ``PRIOR`` 滑行帧），按时间**严格递增**。
    measurements:
        仅真实命中帧的量测，按时间严格递增；长度可以小于 ``states``。
    """

    states: tuple[State, ...]
    measurements: tuple[Measurement, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "measurements", tuple(self.measurements))
        if not self.states:
            raise ValueError("TrackHistory 至少需要一个状态（states 不得为空）")
        _assert_strictly_increasing(self.states, "states")
        _assert_strictly_increasing(self.measurements, "measurements")

    def __len__(self) -> int:
        return len(self.states)

    def __bool__(self) -> bool:
        return True  # 构造时已保证至少一个状态

    @property
    def fresh(self) -> State:
        """最新状态。"""
        return self.states[-1]

    @property
    def measurement_count(self) -> int:
        """真实命中帧数。"""
        return len(self.measurements)

    def state_values(self) -> np.ndarray:
        """状态矩阵 ``(state_dim, n)``，按列存一帧。空输入返回 ``(0, 0)``。"""
        if not self.states:
            return np.empty((0, 0))
        return np.column_stack([s.value for s in self.states])

    def measurement_values(self) -> np.ndarray:
        """量测矩阵 ``(measurement_dim, k)``，按列存一帧。空输入返回 ``(0, 0)``。"""
        if not self.measurements:
            return np.empty((0, 0))
        return np.column_stack([m.value for m in self.measurements])


def _assert_strictly_increasing(items: tuple, name: str) -> None:
    previous = None
    for item in items:
        if previous is not None and not (item.timestamp > previous):
            raise ValueError(f"TrackHistory.{name} 的时间戳必须严格递增")
        previous = item.timestamp
