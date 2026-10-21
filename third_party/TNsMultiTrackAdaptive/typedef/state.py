"""目标状态值对象（契约层）。

设计依据：``ALGORITHM_DESIGN.md`` §4.1。

本模块只做两件事：**规范化形状**、**保存不透明容器**；它不对任何字段做统计解释。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np


class stateClass(str, Enum):
    """状态来源分类。

    - ``PRIOR``：由预测产生（本帧没有实测参与融合）
    - ``POSTERIOR``：由实测融合产生

    由**产生该状态的模型**盖章，``Track`` 只读取它做命中/丢失记账。
    """

    PRIOR = "prior"
    POSTERIOR = "posterior"


@dataclass(frozen=True)
class State:
    """单个目标在单个时刻的状态。

    Parameters
    ----------
    timestamp:
        该状态对应的时刻（秒）。
    value:
        状态向量，形状 ``(state_dim,)``；构造时规范为一维 ``ndarray``。
    extra:
        **不透明容器**（评审意见 ①）。框架层（``Track`` / ``Tracker`` / 关联模块 /
        起始模块）一律不得读取或解释它；装什么、怎么解释由需要它的滤波器自己定义。
        例如 KF 类模型把**状态协方差 P** 装进来，TFOT 放 ``None``。
        类型不限于矩阵，故标注为 ``Any``。
    category:
        状态来源，见 :class:`stateClass`。
    """

    timestamp: float
    value: np.ndarray
    extra: Optional[Any] = None
    category: stateClass = stateClass.PRIOR

    def __post_init__(self) -> None:
        value = np.asarray(self.value, dtype=float)
        if value.ndim != 1:
            value = value.reshape(-1)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "timestamp", float(self.timestamp))
        if not isinstance(self.category, stateClass):
            object.__setattr__(self, "category", stateClass(self.category))

    @property
    def dim(self) -> int:
        """状态维数。"""
        return int(self.value.size)
