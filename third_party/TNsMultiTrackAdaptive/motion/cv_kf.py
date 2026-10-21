"""CV（匀速）运动模型：线性卡尔曼滤波。

设计依据：``ALGORITHM_DESIGN.md`` §5.5.1；内核见 :mod:`TNsMultiTrack.motion.kernels`。

- 状态 ``[x, y, vx, vy]``（4 维），量测 ``[x, y]``（直接观测位置，``H`` 线性）；
- 与 CTRV 的关系：CV 是 CTRV 在 ``ψ̇ ≡ 0`` 且「速度矢量自由」时的线性退化，
  小机动/近直线场景更稳、更快，但**无法表达转弯**（机动时会出现系统性滞后）；
- 协方差同样通过 ``State.extra`` 不透明容器流转（M5），模型不持有航迹状态。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..typedef.history import TrackHistory
from ..typedef.meas import Measurement, measClass
from ..typedef.motion import MotionModel
from ..typedef.state import State, stateClass
from .kernels import CV_STATE_DIM, cv_predict, cv_update

MEASUREMENT_DIM = 2
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

# 默认参数：沿用 MATLAB D2Tracks.m 构造函数的 CV 分支
DEFAULT_Q = np.diag([0.0, 0.0, 1e-4, 1e-4])
DEFAULT_R = np.diag([0.01, 0.01])
DEFAULT_P0 = np.diag([4.0, 4.0, 0.1, 0.1])


class CVModel(MotionModel):
    """匀速（Constant Velocity）模型，线性 KF 实现。"""

    name = "CV"
    state_dim = CV_STATE_DIM
    measurement_dim = MEASUREMENT_DIM

    def __init__(
        self,
        q: Optional[np.ndarray] = None,
        r: Optional[np.ndarray] = None,
        p0: Optional[np.ndarray] = None,
    ) -> None:
        self.q = np.asarray(DEFAULT_Q if q is None else q, dtype=float)
        self.r = np.asarray(DEFAULT_R if r is None else r, dtype=float)
        self.p0 = np.asarray(DEFAULT_P0 if p0 is None else p0, dtype=float)
        self.h = H

    @classmethod
    def from_config(cls, config) -> "CVModel":
        """由 :class:`~TNsMultiTrack.config.ModelConfig` 构造（注册表工厂用）。"""
        return cls(
            q=config.build_q(CV_STATE_DIM),
            r=config.r,
            p0=config.build_p0(CV_STATE_DIM),
        )

    # ── M8：确认帧时刻的初始状态（最小二乘，不回填）──────────────────────────
    def initial_state(self, measurements: Sequence[Measurement], timeStamp: float) -> State:
        if not measurements:
            raise ValueError("initial_state 需要至少一个量测")
        if len(measurements) == 1:
            value = np.array(
                [measurements[0].value[0], measurements[0].value[1], 0.0, 0.0]
            )
            return State(timeStamp, value, extra=self.p0.copy(), category=stateClass.POSTERIOR)

        times = np.array([m.timestamp for m in measurements], dtype=float)
        values = np.column_stack([m.value for m in measurements])          # (2, k)
        centered = times - times[0]
        design = np.column_stack([np.ones_like(centered), centered])       # (k, 2)
        coeffs, *_ = np.linalg.lstsq(design, values.T, rcond=None)         # (2, 2)
        position = coeffs[0] + coeffs[1] * (timeStamp - times[0])          # t_now 处取值
        velocity = coeffs[1]
        value = np.array([position[0], position[1], velocity[0], velocity[1]])
        return State(timeStamp, value, extra=self.p0.copy(), category=stateClass.POSTERIOR)

    # ── M2/M7：预测 ─────────────────────────────────────────────────────────
    def predict_state(self, history: TrackHistory, timeStamp: float) -> State:
        fresh = history.fresh
        dt = float(timeStamp) - fresh.timestamp
        covariance = fresh.extra if isinstance(fresh.extra, np.ndarray) else self.p0
        if dt <= 0.0:
            return State(
                timeStamp, fresh.value.copy(), extra=np.asarray(covariance),
                category=stateClass.PRIOR,
            )
        x_pred, p_pred = cv_predict(fresh.value, covariance, self.q, dt)
        return State(timeStamp, x_pred, extra=p_pred, category=stateClass.PRIOR)

    def predict_measurement(self, history: TrackHistory, timeStamp: float) -> Measurement:
        predicted = self.predict_state(history, timeStamp)
        s = self.h @ predicted.extra @ self.h.T          # 投影协方差，不含 R（§5.3.3）
        return Measurement(
            timeStamp, self.h @ predicted.value, extra=s, category=measClass.PRED
        )

    # ── M1：更新（必须返回 State）────────────────────────────────────────────
    def update_state(
        self,
        history: TrackHistory,
        measurement: Optional[Measurement],
        timeStamp: float,
    ) -> State:
        predicted = self.predict_state(history, timeStamp)
        if measurement is None:
            return predicted                                     # 未分配 → PRIOR
        try:
            x_upd, p_upd = cv_update(
                predicted.value, predicted.extra, measurement.value, self.h, self.r
            )
        except np.linalg.LinAlgError:
            return predicted                                     # 无法融合 → PRIOR（M1）
        return State(timeStamp, x_upd, extra=p_upd, category=stateClass.POSTERIOR)

    def __repr__(self) -> str:
        return f"<CVModel q_v={self.q[2, 2]:g} p0_v={self.p0[2, 2]:g}>"


__all__ = ["CVModel"]
