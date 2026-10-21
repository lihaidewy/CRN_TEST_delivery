"""运动模型接口（契约层）。

设计依据：``ALGORITHM_DESIGN.md`` §4.4。

分界原则：**数值一律归模型，记账一律归航迹**。
模型是纯数学对象——不持有航迹身份、历史与生命周期，只按传入的
:class:`~TNsMultiTrack.typedef.history.TrackHistory` 计算并返回 ``State`` / ``Measurement``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Sequence

from .history import TrackHistory
from .meas import Measurement, measClass
from .state import State, stateClass


class MotionModel(ABC):
    """所有运动模型必须实现的四个接口。

    子类必须定义三个类属性：``name``、``state_dim``、``measurement_dim``
    （由 :meth:`__init_subclass__` 在类定义时校验）。

    契约（编号与设计报告一致）
    --------------------------
    M1  ``update_state`` **必须返回 ``State``，不得返回 ``None``**：
        ``measurement is None``（本帧未分配）→ 返回 ``category=PRIOR`` 的预测状态；
        ``measurement`` 非空 → 返回 ``category=POSTERIOR`` 的融合状态；
        模型自认无法融合（窗口不足等）→ **同样返回 ``PRIOR``**。
    M2  ``predict_state`` 必返回 ``PRIOR``；``predict_measurement`` 必返回 ``PRED``。
    M3  ``predict_measurement(...).value`` 形状为 ``(measurement_dim,)``。
    M4  四个接口都是带参方法，**禁止使用 ``@property``**。
    M5  模型不得持有航迹状态（不存 ``Track`` 引用）。
    M6  所有模型的 ``measurement_dim`` 必须一致（本设计为 2，世界坐标 x-y），
        这是数据关联模块可实现的前提。
    M7  ``update_state(history, None, t)`` 与 ``predict_state(history, t)`` 表达同一件事
        （预测），返回值必须一致；实现可共享内部函数避免重复计算。
    M8  ``initial_state(...)`` 返回的 ``State`` 必须恰好代表传入 ``timeStamp`` 时刻的状态
        （时间戳即 ``timeStamp``）——新航迹从确认帧开始记录、不回填历史。
    """

    name: ClassVar[str]
    state_dim: ClassVar[int]
    measurement_dim: ClassVar[int]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", frozenset()):
            # 仍有未实现的抽象方法：允许部分实现（例如中间基类）
            return
        missing = [
            attr
            for attr in ("name", "state_dim", "measurement_dim")
            if not hasattr(cls, attr)
        ]
        if missing:
            raise TypeError(
                f"{cls.__name__} 必须定义类属性: {', '.join(missing)}（见 MotionModel 契约 M3/M6）"
            )
        for attr in ("state_dim", "measurement_dim"):
            value = getattr(cls, attr)
            if not isinstance(value, int) or value <= 0:
                raise TypeError(f"{cls.__name__}.{attr} 必须是正整数")
        if not isinstance(cls.name, str) or not cls.name:
            raise TypeError(f"{cls.__name__}.name 必须是非空字符串")

    # ── 四个契约接口 ────────────────────────────────────────────────────────
    @abstractmethod
    def initial_state(
        self, measurements: Sequence[Measurement], timeStamp: float
    ) -> State:
        """由起始模块的候选点序列构造**确认帧**的状态（M8）。

        ``measurements`` 是候选段的证据点，其时间跨度允许早于 ``timeStamp``；
        返回值的 ``timestamp`` 必须等于 ``timeStamp``。
        """

    @abstractmethod
    def predict_state(self, history: TrackHistory, timeStamp: float) -> State:
        """预测 ``timeStamp`` 时刻的状态，返回值必须是 ``PRIOR``（M2）。"""

    @abstractmethod
    def predict_measurement(
        self, history: TrackHistory, timeStamp: float
    ) -> Measurement:
        """预测 ``timeStamp`` 时刻的量测，返回值必须是 ``PRED``（M2/M3）。"""

    @abstractmethod
    def update_state(
        self,
        history: TrackHistory,
        measurement: Optional[Measurement],
        timeStamp: float,
    ) -> State:
        """返回本帧的状态，**必为 ``State``**（M1）。

        ``measurement is None`` 表示本帧未分配量测，此时必须返回 ``PRIOR`` 状态。
        ``timeStamp`` 是本帧时刻：``measurement`` 为空时模型只能靠它确定预测目标时刻，
        因此该参数是必需的（M1/M7）。
        """

    # ── 供实现使用的小工具（非契约）────────────────────────────────────────
    @staticmethod
    def _as_prior(state: State) -> State:
        """把状态强制标为 ``PRIOR``（模型返回前的自检辅助）。"""
        if state.category is stateClass.PRIOR:
            return state
        from dataclasses import replace

        return replace(state, category=stateClass.PRIOR)

    @staticmethod
    def _as_pred(measurement: Measurement) -> Measurement:
        """把量测强制标为 ``PRED``（``predict_measurement`` 自检辅助）。"""
        if measurement.category is measClass.PRED:
            return measurement
        from dataclasses import replace

        return replace(measurement, category=measClass.PRED)
