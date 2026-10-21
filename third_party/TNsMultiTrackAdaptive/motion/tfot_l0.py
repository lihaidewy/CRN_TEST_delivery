"""TFOT（无模型 / L0 多项式）滤波。

设计依据：``ALGORITHM_DESIGN.md`` §5.5.3；数值部分由 MATLAB
``tfot_L0fit_predict.m`` / ``tfot_L0fit_estimate.m`` 移植。

特点：
- 状态就是位置 ``[x, y]``（``state_dim = 2``），**不维护协方差**（``State.extra = None``）；
- 逐维在滑动窗口上做 **L0 准则选阶**，再用多项式外推/平滑；
- 输入是**原始量测历史**（``TrackHistory.measurements``），不是已平滑的状态——
  这正是 :class:`~TNsMultiTrack.typedef.history.TrackHistory` 区分两类历史的原因。

与可选的 ``MahalanobisCost`` 互斥：TFOT 不填 ``extra``，马氏代价会直接拒绝（见设计报告 §5.3.3）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..typedef.history import TrackHistory
from ..typedef.meas import Measurement, measClass
from ..typedef.motion import MotionModel
from ..typedef.state import State, stateClass

STATE_DIM = 2
MEASUREMENT_DIM = 2

# MATLAB 默认：lamda=(x, y)、band=(x, y)、fit_order=(x, y)
DEFAULT_LAMBDA = (1.0, 1.0)
DEFAULT_BAND = (2, 2)
DEFAULT_FIT_ORDER = (1, 1)


def l0_cost(window: np.ndarray, order: int) -> float:
    """MATLAB ``run_FunL0`` 的逐字移植：单维窗口的 L0 型代价。

    数学上等于 ``(bᵀu)² / P``，其中 ``u = (F − I)E``、``F = A(GAᵀ)``、``G = (AᵀA)⁻¹``，
    ``P = C − EᵀAGAᵀE``（``A`` 为范德蒙德矩阵、``E = St^(m+1)``、``C = Σ St^(2m+2)``）。
    """
    window = np.asarray(window, dtype=float).reshape(-1)
    count = window.size
    if count <= order:
        raise ValueError("窗口长度必须大于拟合阶数")
    st = np.arange(1, count + 1, dtype=float)

    design = np.ones((count, order + 1))
    for column in range(2, order + 2):
        design[:, column - 1] = st ** (column - 1)
    offset = (st ** (order + 1)).reshape(-1, 1)

    gram = np.linalg.pinv(design.T @ design)
    projection = design @ gram @ design.T
    denominator = float(np.sum(st ** (2 * order + 2)) - offset.T @ projection @ offset)
    if denominator == 0.0:
        denominator = 0.001
    residual = (projection - np.eye(count)) @ offset
    coefficients = window.reshape(-1, 1)
    return float((coefficients.T @ residual) ** 2) / denominator


def select_order(
    window: np.ndarray, lamda: float, max_order: int
) -> int:
    """自 1 阶起搜索，取**第一个**使 ``l0_cost < lamda`` 的阶数（与 MATLAB 一致）。

    若没有任何阶数达标，返回 1（MATLAB 中 ``orders`` 的初值也是 1）。
    """
    order = 1
    for candidate in range(1, max_order + 1):
        if l0_cost(window, candidate) < lamda:
            order = candidate
            break
    return order


class TFOTModel(MotionModel):
    """无模型 TFOT/L0 多项式估计器。"""

    name = "TFOT"
    state_dim = STATE_DIM
    measurement_dim = MEASUREMENT_DIM

    def __init__(
        self,
        lamda: Sequence[float] = DEFAULT_LAMBDA,
        band: Sequence[int] = DEFAULT_BAND,
        fit_order: Sequence[int] = DEFAULT_FIT_ORDER,
        frame_interval: float = 0.1,
    ) -> None:
        if len(lamda) != STATE_DIM or len(band) != STATE_DIM or len(fit_order) != STATE_DIM:
            raise ValueError("lamda / band / fit_order 必须各给两个分量（x, y）")
        if any(value <= 0 for value in lamda):
            raise ValueError("lamda 必须为正")
        if any(value < 2 for value in band):
            raise ValueError("band 必须 ≥ 2（至少两点才能拟合）")
        if any(value < 1 for value in fit_order):
            raise ValueError("fit_order 必须 ≥ 1")
        self.lamda = tuple(float(value) for value in lamda)
        self.band = tuple(int(value) for value in band)
        self.fit_order = tuple(int(value) for value in fit_order)
        self.frame_interval = float(frame_interval)

    @classmethod
    def from_config(cls, config) -> "TFOTModel":
        tfot = config.tfot
        return cls(lamda=tfot.lamda, band=tfot.band, fit_order=tfot.fit_order)

    # ── M8：确认帧时刻的初始状态 ────────────────────────────────────────────
    def initial_state(self, measurements: Sequence[Measurement], timeStamp: float) -> State:
        if not measurements:
            raise ValueError("initial_state 需要至少一个量测")
        # 候选末点必然位于确认帧（§5.4.2 步骤 6），所以末点即 t_now 的位置
        return State(
            timeStamp,
            np.asarray(measurements[-1].value, dtype=float),
            extra=None,
            category=stateClass.POSTERIOR,
        )

    # ── M2/M7：预测 ─────────────────────────────────────────────────────────
    def predict_state(self, history: TrackHistory, timeStamp: float) -> State:
        points = history.measurements
        if not points:
            return State(timeStamp, history.fresh.value.copy(), category=stateClass.PRIOR)
        if len(points) < 2:
            # 只有一点：L0 选阶无法进行，退化为"保持位置"
            return State(timeStamp, points[-1].value.copy(), category=stateClass.PRIOR)
        position = self._evaluate(points, timeStamp, steps=self._steps(history, timeStamp))
        return State(timeStamp, position, extra=None, category=stateClass.PRIOR)

    def predict_measurement(self, history: TrackHistory, timeStamp: float) -> Measurement:
        predicted = self.predict_state(history, timeStamp)
        return Measurement(timeStamp, predicted.value.copy(), extra=None, category=measClass.PRED)

    # ── M1：更新（必须返回 State）────────────────────────────────────────────
    def update_state(
        self,
        history: TrackHistory,
        measurement: Optional[Measurement],
        timeStamp: float,
    ) -> State:
        if measurement is None:
            return self.predict_state(history, timeStamp)          # 滑行 → PRIOR
        points = list(history.measurements) + [measurement]
        if len(points) < 2:
            return State(timeStamp, measurement.value.copy(), category=stateClass.POSTERIOR)
        position = self._evaluate(points, timeStamp, steps=0)        # 平滑当前帧
        return State(timeStamp, position, extra=None, category=stateClass.POSTERIOR)

    # ── 内部 ───────────────────────────────────────────────────────────────
    def _steps(self, history: TrackHistory, timeStamp: float) -> int:
        """从最后一个量测时刻到 ``timeStamp`` 相隔几个采样周期（MATLAB 的 ``n+1``）。"""
        points = history.measurements
        if len(points) >= 2:
            gaps = np.diff([point.timestamp for point in points])
            interval = float(np.median(gaps))
        else:
            interval = self.frame_interval
        if interval <= 0.0:
            interval = self.frame_interval
        steps = int(round((float(timeStamp) - points[-1].timestamp) / interval))
        return max(1, steps)

    def _evaluate(
        self, points: Sequence[Measurement], timeStamp: float, steps: int
    ) -> np.ndarray:
        """逐维选阶 → 多项式拟合 → 求值。``steps=0`` 表示平滑，``steps>=1`` 表示外推。"""
        values = np.column_stack([np.asarray(point.value, dtype=float) for point in points])
        result = np.empty(STATE_DIM, dtype=float)

        for dim in range(STATE_DIM):
            window = values[dim, -min(values.shape[1], self.band[dim]):]
            length = window.size
            max_order = min(self.fit_order[dim], length - 1)
            if max_order < 1:
                result[dim] = window[-1]
                continue
            order = min(select_order(window, self.lamda[dim], max_order), length - 1)
            try:
                coefficients = np.polyfit(np.arange(1, length + 1, dtype=float), window, order)
                result[dim] = float(np.polyval(coefficients, length + steps))
            except (np.linalg.LinAlgError, ValueError):
                result[dim] = window[-1]
        return result

    def __repr__(self) -> str:
        return (
            f"<TFOTModel lamda={self.lamda} band={self.band} "
            f"fit_order={self.fit_order}>"
        )


__all__ = [
    "DEFAULT_BAND",
    "DEFAULT_FIT_ORDER",
    "DEFAULT_LAMBDA",
    "TFOTModel",
    "l0_cost",
    "select_order",
]
