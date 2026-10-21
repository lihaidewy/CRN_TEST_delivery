"""代价函数：决定"预测与量测像不像"。

设计依据：``ALGORITHM_DESIGN.md`` §5.3.3。

关键约定：``extra`` 容器本身**不含语义**，只有具体的代价函数才按自身文档解释它。
``EuclideanCost`` 是默认实现，不读容器，任何模型都能用。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..typedef.meas import Measurement

#: 门限外标记。用有限大值而非 ``np.inf``，规避 solver 对 inf 的支持差异。
FORBIDDEN = 1e12


@runtime_checkable
class CostFunction(Protocol):
    """代价函数协议：``gate`` 必须与 ``__call__`` 的返回值同量纲。

    可选扩展：实现 ``gate_for(prediction) -> float`` 即可提供**逐预测**的门限
    （见 :class:`AdaptiveCost`）；``GNN`` 会自动优先使用它，未实现时回退到标量 ``gate``。
    """

    name: str
    gate: float

    def __call__(self, prediction: Measurement, detection: Measurement) -> float:
        ...


class EuclideanCost:
    """欧氏距离（米）。默认代价函数，不读 ``extra``。"""

    name = "euclidean"

    def __init__(self, gate: float = 1.5) -> None:
        if gate <= 0:
            raise ValueError("gate 必须为正")
        self.gate = float(gate)

    def __call__(self, prediction: Measurement, detection: Measurement) -> float:
        return float(np.linalg.norm(prediction.value - detection.value))


class MahalanobisCost:
    """马氏距离平方，门限为卡方分位点（2 自由度）。

    **本代价函数的私有约定**（调用方选用本类即表示接受）::

        prediction.extra = S = H P⁻ Hᵀ     # 投影后的量测协方差，不含 R
        detection.extra  = R               # 量测噪声协方差
        d² = νᵀ (S + R)⁻¹ ν                # 两者相加即信息融合，无重复计噪

    ``S`` 缺失或形状不对时抛 ``ValueError``（不静默退化成欧氏）。
    """

    name = "mahalanobis"

    def __init__(self, gate: float = 9.2103) -> None:      # chi2.ppf(0.99, df=2)
        if gate <= 0:
            raise ValueError("gate 必须为正")
        self.gate = float(gate)

    def __call__(self, prediction: Measurement, detection: Measurement) -> float:
        s = prediction.extra
        r = detection.extra
        dim = prediction.value.size
        if not isinstance(s, np.ndarray) or s.shape != (dim, dim):
            raise ValueError(
                f"MahalanobisCost 需要 prediction.extra 为 ({dim},{dim}) 的协方差 S，"
                f"实际 {type(s).__name__}"
            )
        if not isinstance(r, np.ndarray) or r.shape != (dim, dim):
            raise ValueError(
                f"MahalanobisCost 需要 detection.extra 为 ({dim},{dim}) 的协方差 R，"
                f"实际 {type(r).__name__}"
            )
        innovation = detection.value - prediction.value
        covariance = s + r
        try:
            return float(innovation @ np.linalg.solve(covariance, innovation))
        except np.linalg.LinAlgError as error:                # pragma: no cover
            raise ValueError("S + R 不可解，检查协方差是否为正定") from error


class AdaptiveCost:
    """欧氏代价 + **随预测不确定性自适应的门限**（本副本新增）。

    动机：固定门限是"量测噪声"尺度的（本仓库数据默认 1.5 m），但**丢失期间只能靠预测**，
    预测漂移会累积到 0.6–1.2 m，于是滑行中的航迹接不回自己的量测 → 该量测被送进起始模块
    → 攒够 N 帧后起一条重复航迹（实测航迹数 6→16、ID 切换 0→10）。

    门限公式::

        σ(pred)  = sqrt(trace(S))            # S = prediction.extra = H P⁻ Hᵀ
        gate(pred) = clip(base + k·σ, base, max_gate)

    - ``S`` 由 KF 类模型（CV/CTRV）通过不透明容器给出，**滑行时 P 每帧累积 Q 而增大，
      因此门限自动随丢失帧数膨胀**，重新命中后 P 收缩、门限自动收紧——无需手工调参；
    - 模型不提供 ``S`` 时（如 TFOT）退化为 ``base``，行为与固定门限一致；
    - ``base`` 是下限，覆盖量测噪声与模型误差；``max_gate`` 是上限，应低于**目标最小间距**
      （本仓库数据 3.5 m），否则会出现交叉误关联。

    与 :class:`MahalanobisCost` 的区别：马氏用 ``νᵀ(S+R)⁻¹ν`` 做**归一化**判决，
    要求量测侧提供 ``R``；本类仍用欧氏距离（对容器无要求），只用协方差**调整门限尺度**，
    因此即插即用，也不会因模型误差未被 P 覆盖而把门限收得过紧。
    """

    name = "adaptive"

    def __init__(
        self,
        base: float = 1.0,
        k: float = 3.0,
        max_gate: float = 3.5,
        min_sigma: float = 0.0,
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
        self.min_sigma = float(min_sigma)
        #: 兼容 CostFunction 协议的标量门限（保守值：无协方差信息时等同固定门限）
        self.gate = float(base)

    # ── 逐预测门限 ──────────────────────────────────────────────────────────
    def gate_for(self, prediction: Measurement) -> float:
        """由该预测的协方差容器算出本帧门限。"""
        sigma = self.sigma_of(prediction)
        return float(min(self.max_gate, max(self.base, self.base + self.k * sigma)))

    def sigma_of(self, prediction: Measurement) -> float:
        """预测的 1σ 位置尺度 ``sqrt(trace(S))``；无协方差信息时返回 0。"""
        s = prediction.extra
        if not isinstance(s, np.ndarray) or s.ndim != 2 or s.shape[0] != s.shape[1]:
            return self.min_sigma
        trace = float(np.trace(s))
        if not np.isfinite(trace) or trace <= 0.0:
            return self.min_sigma
        return max(self.min_sigma, float(np.sqrt(trace)))

    def __call__(self, prediction: Measurement, detection: Measurement) -> float:
        return float(np.linalg.norm(prediction.value - detection.value))

    def __repr__(self) -> str:
        return f"<AdaptiveCost base={self.base:g} k={self.k:g} max={self.max_gate:g}>"


__all__ = ["FORBIDDEN", "AdaptiveCost", "CostFunction", "EuclideanCost", "MahalanobisCost"]
