"""量测值对象（契约层）。

设计依据：``ALGORITHM_DESIGN.md`` §4.2。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np


class measClass(str, Enum):
    """量测来源分类（只作边界断言，不含统计语义）。

    - ``REAL``：来自上游（传感器 / 标定结果），可写入航迹的量测历史
    - ``PRED``：由模型综合出来的预测量测，只用于关联，**禁止写入任何历史**
    """

    REAL = "real"
    PRED = "predict"


@dataclass(frozen=True)
class Measurement:
    """一帧中的一个量测点。

    Parameters
    ----------
    timestamp:
        该量测的时刻（秒）。
    value:
        量测向量，形状 ``(measurement_dim,)``；本设计统一为世界坐标 ``[x, y]``。
    extra:
        **不透明容器**：上游可装量测噪声 ``R``，模型可装自用数据。关联模块的核心
        不解释它，只有被显式选中的代价函数才按自身约定读取
        （见 ``ALGORITHM_DESIGN.md`` §5.3.3 与附录 D）。
    confidence:
        上游给出的置信度（可选）。
    category:
        量测来源，见 :class:`measClass`。
    """

    timestamp: float
    value: np.ndarray
    extra: Optional[Any] = None
    confidence: Optional[float] = None
    category: measClass = measClass.REAL

    def __post_init__(self) -> None:
        value = np.asarray(self.value, dtype=float)
        if value.ndim != 1:
            value = value.reshape(-1)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "timestamp", float(self.timestamp))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", float(self.confidence))
        if not isinstance(self.category, measClass):
            object.__setattr__(self, "category", measClass(self.category))

    @property
    def dim(self) -> int:
        """量测维数。"""
        return int(self.value.size)
