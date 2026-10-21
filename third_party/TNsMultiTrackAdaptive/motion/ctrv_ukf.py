"""CTRV-UKF 运动模型。

设计依据：``ALGORITHM_DESIGN.md`` §5.5.2；内核见 :mod:`TNsMultiTrack.motion.kernels`。

协方差通过 **``State.extra`` 不透明容器**在航迹里流转（模型不持有航迹状态，M5）：
- ``initial_state`` 写入 ``P0``；
- ``predict_state`` 写入 ``P_pred``；
- ``update_state`` 写入 ``P_upd``；
- ``predict_measurement`` 写入 ``S = H P_pred Hᵀ``（**不含 R**），供可选的
  ``MahalanobisCost`` 使用（约定见设计报告 §5.3.3）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..typedef.history import TrackHistory
from ..typedef.meas import Measurement, measClass
from ..typedef.motion import MotionModel
from ..typedef.state import State, stateClass
from .kernels import STATE_DIM, ctrv_predict, ctrv_transition, ctrv_update, wrap_to_pi

MEASUREMENT_DIM = 2
H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])

# 默认参数：Q/R/P0 沿用 MATLAB D2Tracks.m:38-45 的 CTRV 取值
DEFAULT_Q = np.diag([0.0, 0.0, 1e-4, 1e-4, 1e-4])
DEFAULT_R = np.diag([1e-4, 1e-4])
DEFAULT_P0 = np.diag([4.0, 4.0, 0.1, 0.1, 0.1])


class CTRVModel(MotionModel):
    """恒定转率与速度（CTRV）模型，UKF 实现。"""

    name = "CTRV"
    state_dim = STATE_DIM
    measurement_dim = MEASUREMENT_DIM

    def __init__(
        self,
        q: Optional[np.ndarray] = None,
        r: Optional[np.ndarray] = None,
        p0: Optional[np.ndarray] = None,
        psi_dot_from_evidence: bool = True,
    ) -> None:
        self.q = np.asarray(DEFAULT_Q if q is None else q, dtype=float)
        self.r = np.asarray(DEFAULT_R if r is None else r, dtype=float)
        self.p0 = np.asarray(DEFAULT_P0 if p0 is None else p0, dtype=float)
        self.h = H
        self.psi_dot_from_evidence = psi_dot_from_evidence

    # ── M8：确认帧时刻的初始状态（最小二乘，不回填）──────────────────────────
    @classmethod
    def from_config(cls, config) -> "CTRVModel":
        """由 :class:`~TNsMultiTrack.config.ModelConfig` 构造（注册表工厂用）。"""
        return cls(
            q=config.build_q(STATE_DIM),
            r=config.r,
            p0=config.build_p0(STATE_DIM),
        )

    def initial_state(self, measurements: Sequence[Measurement], timeStamp: float) -> State:
        if not measurements:
            raise ValueError("initial_state 需要至少一个量测")
        if len(measurements) == 1:
            # 数据不足：位置取该点、速度置 0（M8 仍要求时间戳为 timeStamp）
            value = np.array(
                [measurements[0].value[0], measurements[0].value[1], 0.0, 0.0, 0.0]
            )
            return State(timeStamp, value, extra=self.p0.copy(), category=stateClass.POSTERIOR)

        times = np.array([m.timestamp for m in measurements], dtype=float)
        values = np.column_stack([m.value for m in measurements])          # (2, k)
        centered = times - times[0]
        design = np.column_stack([np.ones_like(centered), centered])       # (k, 2)
        coeffs, *_ = np.linalg.lstsq(design, values.T, rcond=None)         # (2, 2)
        position = coeffs[0] + coeffs[1] * (timeStamp - times[0])          # t_now 处取值
        velocity = coeffs[1]

        speed = float(np.hypot(velocity[0], velocity[1]))
        heading = float(np.arctan2(velocity[1], velocity[0])) if speed > 1e-9 else 0.0
        yaw_rate = self._estimate_yaw_rate(times, values, coeffs) if self.psi_dot_from_evidence else 0.0

        value = np.array([position[0], position[1], speed, heading, yaw_rate])
        p0 = self.p0.copy()
        # 位置初值来自拟合，标准差约等于量测噪声；速度不确定性交给 P0
        return State(timeStamp, value, extra=p0, category=stateClass.POSTERIOR)

    @staticmethod
    def _estimate_yaw_rate(
        times: np.ndarray, values: np.ndarray, coeffs: np.ndarray
    ) -> float:
        """用候选段前后半段的航向差除以时间间隔粗估转率。"""
        span = times[-1] - times[0]
        if span <= 0.0 or values.shape[1] < 4:
            return 0.0
        half = values.shape[1] // 2
        first = coeffs[0] + coeffs[1] * (times[half - 1] - times[0])
        second = coeffs[0] + coeffs[1] * (times[-1] - times[0])
        heading_1 = float(np.arctan2(first[1] - coeffs[0][1], first[0] - coeffs[0][0]))
        heading_2 = float(np.arctan2(second[1] - first[1], second[0] - first[0]))
        dt = times[-1] - times[half - 1]
        if dt <= 0.0:
            return 0.0
        return float(wrap_to_pi(heading_2 - heading_1) / dt)

    # ── M2/M7：预测 ─────────────────────────────────────────────────────────
    def predict_state(self, history: TrackHistory, timeStamp: float) -> State:
        fresh = history.fresh
        dt = float(timeStamp) - fresh.timestamp
        covariance = fresh.extra if isinstance(fresh.extra, np.ndarray) else self.p0
        if dt <= 0.0:
            return State(timeStamp, fresh.value.copy(), extra=np.asarray(covariance), category=stateClass.PRIOR)
        x_pred, p_pred = ctrv_predict(fresh.value, covariance, self.q, dt)
        return State(timeStamp, x_pred, extra=p_pred, category=stateClass.PRIOR)

    def predict_measurement(self, history: TrackHistory, timeStamp: float) -> Measurement:
        predicted = self.predict_state(history, timeStamp)
        covariance = predicted.extra
        s = self.h @ covariance @ self.h.T          # 投影协方差，不含 R（§5.3.3）
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
            x_upd, p_upd = ctrv_update(
                predicted.value, predicted.extra, measurement.value, self.h, self.r
            )
        except np.linalg.LinAlgError:
            return predicted                                     # 无法融合 → PRIOR（M1）
        return State(timeStamp, x_upd, extra=p_upd, category=stateClass.POSTERIOR)

    # ── 调试辅助 ────────────────────────────────────────────────────────────
    def transition(self, value: np.ndarray, dt: float) -> np.ndarray:
        """暴露状态转移，便于单测。"""
        return ctrv_transition(np.asarray(value, dtype=float), dt)

    def __repr__(self) -> str:
        return f"<CTRVModel q_v={self.q[2, 2]:g} p0_v={self.p0[2, 2]:g}>"
