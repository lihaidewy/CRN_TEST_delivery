"""航迹对象（契约层）——本设计的核心。

设计依据：``ALGORITHM_DESIGN.md`` §4.5 / §5.2 / §5.4.5。

一条航迹自己管理：身份、历史、生命周期、以及**自己那份滤波器**（``MotionModel`` 实例）。
它对外的数据接口只有四个：``predict`` / ``update`` / ``alive`` / ``snapshot``。

契约
----
C1  ``predict(ts)`` 对同一 ``ts`` 幂等：重复调用返回同一对象，且不改变持久状态。
C2  ``predict`` 不写历史，``update`` 才写：``predict`` 只缓存 ``_pending``。
C3  同一帧必须先 ``predict`` 再 ``update``，且活跃航迹每帧恰好 ``update`` 一次。
C4  航迹**绝不持有 ``Tracker`` 引用**；只回答 ``alive``。
C5  **丢失即冻结**：``alive == False`` 的航迹不再预测、不再更新，``predict()``
    返回 ``None``、``update()`` 抛 ``RuntimeError``，历史不再增长。
"""

from __future__ import annotations

import bisect
import itertools
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

import numpy as np

from .history import TrackHistory
from .meas import Measurement, measClass
from .motion import MotionModel
from .state import State, stateClass

_TIME_TOLERANCE = 1e-9


class trackState(str, Enum):
    """航迹生命周期状态。

    - ``TENTATIVE``：未确认（默认路径不产生，保留给"单点快速起批"扩展模式）
    - ``CONFIRMED``：已确认且本帧有实测融合
    - ``COASTED``：连续丢失但未超限；**仍然预测**，参与关联以等待目标重现
    - ``LOST``：判定丢失（连续丢失 ≥ ``patience``）→ **冻结**
    - ``DELETED``：已退休（移出活跃列表）
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTED = "coasted"
    LOST = "lost"
    DELETED = "deleted"


_FROZEN_STATES = (trackState.LOST, trackState.DELETED)


@dataclass(frozen=True)
class _Pending:
    """``predict`` 与 ``update`` 之间的暂存（只存预测量测）。"""

    timestamp: float
    measurement: Measurement


class Track:
    """自治航迹：自己的历史、生命周期与滤波器。"""

    _counter = itertools.count(1)

    def __init__(
        self, state: State, model: MotionModel, patience: int = 7
    ) -> None:
        if not isinstance(state, State):
            raise TypeError("state 必须是 State")
        if not isinstance(model, MotionModel):
            raise TypeError("model 必须是 MotionModel")
        if state.dim != model.state_dim:
            raise ValueError(
                f"状态维数 {state.dim} 与模型 {model.name}.state_dim={model.state_dim} 不一致"
            )
        if not isinstance(patience, int) or patience < 1:
            raise ValueError("patience 必须是 ≥ 1 的整数")

        self.id: int = next(Track._counter)
        self.model: MotionModel = model
        self.model_name: str = model.name

        # 历史（时间戳严格递增；history 与 _timestamps 一一对应，供二分查询）
        self.history: list[State] = [state]
        self._timestamps: list[float] = [state.timestamp]
        self.meas_history: list[Measurement] = []

        self.birth: float = state.timestamp
        self.last_update_time: float = state.timestamp
        self.last_hit_time: float = state.timestamp

        # 起始证据（只读元数据，不参与滤波；见 §5.4.5）
        self.first_evidence_time: float = state.timestamp
        self.evidence: tuple[Measurement, ...] = ()
        self.init_quality: float = 0.0
        self.init_span: int = 1

        self.lifecycle: trackState = trackState.CONFIRMED
        self.patience: int = patience
        self.hits: int = 0
        self.misses: int = 0
        self._lose: int = 0                     # 连续丢失帧数
        self._pending: Optional[_Pending] = None
        #: 新息（预测↔实测距离）的指数滑动平均，供自适应门限判断"这条航迹平时准不准"
        self.innovation_ewma: float = 0.0
        self.innovation_alpha: float = 0.3      # EWMA 平滑系数

    # ── 对外接口（只有这四个）──────────────────────────────────────────────
    def predict(self, timeStamp: float) -> Optional[Measurement]:
        """计算本帧的预测量测；冻结航迹返回 ``None``（C1/C2/C5）。"""
        if not self.alive:
            return None                              # C5：不预测、不改任何状态
        timeStamp = float(timeStamp)
        if self._pending is not None:
            if self._pending.timestamp == timeStamp:
                return self._pending.measurement     # C1：幂等，直接返回缓存
            raise RuntimeError(
                "上一帧的 update 未被调用（TimeStamp 从 "
                f"{self._pending.timestamp} 变为 {timeStamp}）"
            )

        predicted = self.model.predict_measurement(self.history_view(), timeStamp)
        if not isinstance(predicted, Measurement):
            raise TypeError(f"{self.model_name}.predict_measurement 必须返回 Measurement")
        if predicted.category is not measClass.PRED:
            raise ValueError(f"{self.model_name}.predict_measurement 必须返回 PRED 量测（M2）")
        if predicted.dim != self.model.measurement_dim:
            raise ValueError(
                f"预测量测维数 {predicted.dim} != {self.model_name}.measurement_dim"
                f"={self.model.measurement_dim}（M3）"
            )
        self._pending = _Pending(timeStamp, predicted)
        return predicted

    def update(self, measurement: Optional[Measurement]) -> None:
        """消费本帧结果并追加一帧历史；``None`` 表示本帧未分配量测（C3/C5/M1）。"""
        if not self.alive:
            raise RuntimeError("冻结航迹不得更新（C5）")
        if self._pending is None:
            raise RuntimeError("必须先 predict 再 update（C3）")
        pending, self._pending = self._pending, None

        if measurement is not None:
            if not isinstance(measurement, Measurement):
                raise TypeError("measurement 必须是 Measurement 或 None")
            if measurement.category is not measClass.REAL:
                raise ValueError("只能用量测历史收纳 REAL 量测（PRED 禁止入库）")
            if abs(measurement.timestamp - pending.timestamp) > _TIME_TOLERANCE:
                raise ValueError(
                    f"量测时刻 {measurement.timestamp} 与本帧 {pending.timestamp} 不一致"
                )

        # M1：模型必须返回 State（无分配时返回 PRIOR）
        state = self.model.update_state(self.history_view(), measurement, pending.timestamp)
        if not isinstance(state, State):
            raise TypeError(f"{self.model_name}.update_state 必须返回 State（M1），实际 {type(state)!r}")
        if state.dim != self.model.state_dim:
            raise ValueError(f"更新后状态维数 {state.dim} != {self.model.state_dim}")

        hit = state.category is stateClass.POSTERIOR and measurement is not None
        if not hit:
            # 滑行：归一化为 PRIOR（防御模型误标 POSTERIOR 却没有量测的情况）
            if state.category is stateClass.POSTERIOR:
                state = replace(state, category=stateClass.PRIOR)
            self.misses += 1
            self._lose += 1
            self.lifecycle = (
                trackState.LOST if self._lose >= self.patience else trackState.COASTED
            )
        else:
            self.meas_history.append(measurement)
            self.hits += 1
            self._lose = 0
            self.last_hit_time = measurement.timestamp
            self.lifecycle = trackState.CONFIRMED
            # 记录新息（预测↔实测距离）的 EWMA：门限策略据此判断"平时准不准"。
            # 关键点：只在**命中帧**更新，因此旁边有杂波但自己的量测还在时，
            # 新息依然很小 → 门限不会被杂波"骗宽"。
            innovation = float(
                np.linalg.norm(measurement.value - pending.measurement.value)
            )
            alpha = self.innovation_alpha
            self.innovation_ewma = (1.0 - alpha) * self.innovation_ewma + alpha * innovation

        self._append_state(self._pin_timestamp(state, pending.timestamp))

    @property
    def alive(self) -> bool:
        """是否仍参与跟踪（预测 + 更新 + 关联）。"""
        return self.lifecycle not in _FROZEN_STATES

    @property
    def fresh(self) -> State:
        """最新状态。"""
        return self.history[-1]

    @property
    def miss_streak(self) -> int:
        """当前**连续丢失帧数**（本副本新增：供自适应门限策略使用）。

        与 ``_lose`` 同源；命中一帧即归零。``Track`` 自己就是"丢失有多深"的权威来源，
        因此门限策略只需读这一个标量，不需要理解协方差或生命周期细节。
        """
        return int(self._lose)

    def snapshot(self, timeStamp: float) -> Optional[State]:
        """返回**恰好**位于 ``timeStamp`` 的状态；不存在则 ``None``（二分查询）。"""
        index = bisect.bisect_left(self._timestamps, float(timeStamp))
        if index < len(self._timestamps) and abs(self._timestamps[index] - timeStamp) <= _TIME_TOLERANCE:
            return self.history[index]
        return None

    def history_view(self) -> TrackHistory:
        """给模型的冻结历史视图（模型拿不到 ``Track`` 本身）。"""
        return TrackHistory(tuple(self.history), tuple(self.meas_history))

    # ── 包内工厂：仅供起始模块调用（§5.4.5）────────────────────────────────
    @classmethod
    def _born(
        cls,
        state0: State,
        model: MotionModel,
        patience: int = 7,
        *,
        first_evidence_time: float,
        evidence: tuple[Measurement, ...],
        quality: float = 0.0,
        span: int = 1,
        seed_measurements: bool = True,
    ) -> "Track":
        """由起始模块创建新航迹：**状态从确认帧开始记录，不回填历史**。

        ``state0`` 必须是确认帧（``timeStamp``）时刻的状态（M8），且 ``evidence``
        的末点必须落在同一时刻——候选必须"命中当前帧"（§5.4.2 步骤 6）。

        ``seed_measurements``：是否把候选段的**真实量测**放进 ``meas_history``。
        量测历史是滤波器的输入窗口，不是状态历史：
        - KF 类模型（CTRV/CV）只读 ``history.fresh``，用不到它；
        - 量测驱动模型（TFOT）必须靠它做多项式拟合，否则新航迹只有 1 个点、
          前几帧只能"保持位置"。
        因此默认开启，且**不与评审意见 ⑤ 冲突**：``states`` 仍然从确认帧开始，
        确认之前的帧不会凭空多出状态。
        """
        evidence = tuple(evidence)
        if not evidence:
            raise ValueError("新航迹必须带起始证据（evidence 不得为空）")
        if abs(evidence[-1].timestamp - state0.timestamp) > _TIME_TOLERANCE:
            raise ValueError(
                "起始证据的末点必须位于确认帧："
                f"evidence[-1].timestamp={evidence[-1].timestamp} != state0.timestamp={state0.timestamp}"
            )

        track = cls(state0, model, patience)
        track.history = [state0]                       # 不回填状态：长度恰为 1
        track._timestamps = [state0.timestamp]
        track.meas_history = list(evidence) if seed_measurements else [evidence[-1]]
        track.birth = state0.timestamp
        track.last_update_time = state0.timestamp
        track.last_hit_time = state0.timestamp
        track.hits = 1                                 # 确认帧本身就是一次命中
        track.misses = 0
        track._lose = 0
        track.lifecycle = trackState.CONFIRMED
        track.first_evidence_time = float(first_evidence_time)
        track.evidence = evidence
        track.init_quality = float(quality)
        track.init_span = int(span)
        return track

    @classmethod
    def reset_ids(cls) -> None:
        """重置 ID 计数器（仅供测试）。"""
        cls._counter = itertools.count(1)

    # ── 内部 ───────────────────────────────────────────────────────────────
    def _pin_timestamp(self, state: State, timeStamp: float) -> State:
        """把状态时刻钉到本帧时钟上。

        帧时钟归 ``Tracker`` 所有；模型内部可能出现 ``prev + (ts - prev)`` 这类
        浮点漂移，因此在容差内直接改写为权威时刻，超出容差则视为模型错误。
        """
        if abs(state.timestamp - timeStamp) <= 1e-6:
            if state.timestamp != timeStamp:
                return replace(state, timestamp=timeStamp)
            return state
        raise ValueError(
            f"{self.model_name}.update_state 返回的状态时刻 {state.timestamp} "
            f"与本帧 {timeStamp} 不一致"
        )

    def _append_state(self, state: State) -> None:
        if state.timestamp <= self._timestamps[-1]:
            raise ValueError(
                f"状态时间戳必须严格递增：{state.timestamp} <= {self._timestamps[-1]}"
            )
        self.history.append(state)
        self._timestamps.append(state.timestamp)
        self.last_update_time = state.timestamp

    def __repr__(self) -> str:
        return (
            f"<Track id={self.id} model={self.model_name} life={self.lifecycle.value} "
            f"birth={self.birth:.3f} frames={len(self.history)} hits={self.hits} "
            f"misses={self.misses} lose={self._lose}>"
        )
