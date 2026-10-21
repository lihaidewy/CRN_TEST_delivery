"""门限策略（本副本新增）：把"门限怎么定"从代价函数里独立出来。

为什么需要单独的抽象
--------------------
代价函数只能看到 ``(预测, 实测)`` 两个 :class:`Measurement`，**看不到航迹状态**
（连续丢失了几帧、自己的协方差多大）。而"丢失越久、预测漂移越大"这个规律恰恰是
航迹状态——所以门限策略必须拿到 ``Track``。

实测依据（``EVALUATION_DROPOUT.md``）：丢失期间预测漂移会累积到 0.6–1.2 m，
固定门限 1.5 m 会让滑行中的航迹接不回自己的量测 → 航迹数 6→16、ID 切换 0→10。

两条实现路线
------------
- :class:`AdaptiveGate`（**推荐**）：按**连续丢失帧数**放大门限，与模型无关
  （TFOT 没有协方差也能用）；
- :class:`CovarianceGate`（备选）：按预测协方差放大，只适用于 CV/CTRV；
  注意当过程噪声 ``Q`` 调大时 ``P`` 会偏悲观（本仓库数据里滤波器自称 0.8 m 不确定、
  实测误差只有 0.075 m），门限容易顶到上限、退化成固定门限。
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np

from ..typedef.track import Track


@runtime_checkable
class GatePolicy(Protocol):
    """逐航迹门限策略：返回值与代价函数同量纲（本设计为米）。"""

    name: str

    def gate(self, track: Track) -> float:
        ...


class FixedGate:
    """固定门限（与原库行为一致）。"""

    name = "fixed"

    def __init__(self, base: float = 1.5) -> None:
        if base <= 0:
            raise ValueError("base 必须为正")
        self.base = float(base)

    def gate(self, track: Track) -> float:
        return self.base

    def __repr__(self) -> str:
        return f"<FixedGate base={self.base:g}>"


class AdaptiveGate:
    """**按连续丢失帧数 + 本航迹新息水平**放大的门限（推荐实现）。

    公式::

        gate = clip(base + growth·sqrt(streak) + innovation_weight·ewma, base, max_gate)

    三项各自的含义：

    - ``base``：健康跟踪时的门限（量测噪声尺度）；
    - ``growth·sqrt(streak)``：``streak`` = 该航迹当前连续丢失帧数
      （``Track.miss_streak``）。用 ``sqrt`` 是因为预测误差近似随机游走式累积；
    - ``innovation_weight·ewma``：``ewma`` = 该航迹**命中帧**新息（预测↔实测距离）的
      指数滑动平均（``Track.innovation_ewma``）。它让门限跟着"这条航迹平时准不准"走：
      机动段新息变大 → 门限变宽；直线段新息很小 → 门限收紧。

    为什么需要新息项（实测教训）
    ---------------------------
    只按丢失帧数放大会有一个反向副作用：**丢失帧恰恰是"自己的量测不见了"的时刻**，
    此时放宽门限，门内最近的点很可能是杂波或邻车。在"20% 丢失 + 每帧 1 个 1.8–3.0 m
    杂波"的数据上，固定门限 2.5 m 的虚警从 37 涨到 148（门限抓杂波），
    而 1.5 m 又会在机动+丢失时接不回自己的量测。
    新息项把两者分开：杂波不会让新息变大（命中的仍是真量测），机动才会。

    ``innovation_weight=0`` 时退化为纯丢失帧数驱动。
    """

    name = "adaptive"

    def __init__(
        self,
        base: float = 1.5,
        growth: float = 0.5,
        max_gate: float = 3.5,
        innovation_weight: float = 0.0,
    ) -> None:
        if base <= 0:
            raise ValueError("base 必须为正")
        if growth < 0 or innovation_weight < 0:
            raise ValueError("growth / innovation_weight 不能为负")
        if max_gate < base:
            raise ValueError("max_gate 必须 ≥ base")
        self.base = float(base)
        self.growth = float(growth)
        self.innovation_weight = float(innovation_weight)
        self.max_gate = float(max_gate)

    def gate(self, track: Track) -> float:
        streak = max(0, int(getattr(track, "miss_streak", 0)))
        ewma = max(0.0, float(getattr(track, "innovation_ewma", 0.0)))
        gate = self.base + self.growth * math.sqrt(streak) + self.innovation_weight * ewma
        return float(min(self.max_gate, max(self.base, gate)))

    def __repr__(self) -> str:
        return (
            f"<AdaptiveGate base={self.base:g} growth={self.growth:g} "
            f"innov_w={self.innovation_weight:g} max={self.max_gate:g}>"
        )


class CovarianceGate:
    """按**预测协方差**放大的门限（备选）。

    ``gate = clip(base + k·sqrt(trace(S)), base, max_gate)``，`S` 取自预测的协方差容器
    （由 KF 类模型给出）。模型不提供 `S` 时退化为 ``base``。

    ⚠️ 使用前提：``P`` 必须是**可信的不确定性**。若为了跟上机动把过程噪声 ``Q`` 调得很大，
    ``P`` 会显著偏悲观，门限将长期顶到 ``max_gate``、失去自适应性（本仓库数据实测如此）。
    因此默认推荐 :class:`AdaptiveGate`。
    """

    name = "covariance"

    def __init__(
        self,
        base: float = 1.0,
        k: float = 3.0,
        max_gate: float = 3.5,
    ) -> None:
        if base <= 0:
            raise ValueError("base 必须为正")
        if k < 0:
            raise ValueError("k 不能为负")
        if max_gate < base:
            raise ValueError("max_gate 必须 ≥ base")
        self.base = float(base)
        self.k = float(k)
        self.max_gate = float(max_gate)

    def gate(self, track: Track) -> float:
        sigma = sigma_of_prediction(track)
        return float(min(self.max_gate, max(self.base, self.base + self.k * sigma)))

    def __repr__(self) -> str:
        return f"<CovarianceGate base={self.base:g} k={self.k:g} max={self.max_gate:g}>"


def sigma_of_prediction(track: Track) -> float:
    """航迹最新状态的 1σ 位置尺度 ``sqrt(trace(P_pos))``；无协方差信息时返回 0。

    这里读的是 ``State.extra``（KF 类模型放协方差）。**这是本副本里唯一读取该容器的地方**，
    且只读迹（trace），不解释其内部结构。
    """
    state = track.history[-1] if track.history else None
    extra = getattr(state, "extra", None)
    if not isinstance(extra, np.ndarray) or extra.ndim != 2 or extra.shape[0] < 2:
        return 0.0
    trace = float(extra[0, 0] + extra[1, 1])
    if not np.isfinite(trace) or trace <= 0.0:
        return 0.0
    return float(np.sqrt(trace))


def build_gate_policy(
    mode: str,
    base: float,
    growth: float,
    max_gate: float,
    innovation_weight: float = 0.0,
) -> GatePolicy:
    """按名字构造门限策略。"""
    kind = (mode or "fixed").lower()
    if kind == "fixed":
        return FixedGate(base)
    if kind in ("adaptive", "streak"):
        return AdaptiveGate(
            base=base, growth=growth, max_gate=max_gate,
            innovation_weight=innovation_weight,
        )
    if kind in ("covariance", "cov"):
        return CovarianceGate(base=base, k=growth, max_gate=max_gate)
    raise ValueError(f"未知门限模式 {mode!r}（可选 fixed / adaptive / covariance）")


__all__ = [
    "AdaptiveGate",
    "CovarianceGate",
    "FixedGate",
    "GatePolicy",
    "build_gate_policy",
    "sigma_of_prediction",
]
