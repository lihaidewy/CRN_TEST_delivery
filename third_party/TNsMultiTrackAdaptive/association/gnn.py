"""GNN 关联器：组合代价函数 + 求解器 + 未匹配代价策略。

设计依据：``ALGORITHM_DESIGN.md`` §5.3。

输入只有 ``Measurement``（预测侧 ``PRED``、实测侧 ``REAL``）：不接触 ``Track``、
不认识模型、不解释 ``extra``（解释权在具体代价函数手里）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..typedef.association import AssociationResult
from ..typedef.meas import Measurement, measClass
from .cost import FORBIDDEN, CostFunction, EuclideanCost
from .solver import Solver, default_solver


class GNN:
    """最近邻关联 + 全局最优指派。"""

    def __init__(
        self,
        cost: Optional[CostFunction] = None,
        solver: Optional[Solver] = None,
        unmatched_cost_factor: float = 2.0,
    ) -> None:
        self.cost = EuclideanCost() if cost is None else cost
        self.solver = default_solver() if solver is None else solver
        self.unmatched_cost_factor = float(unmatched_cost_factor)

    # ── 代价矩阵 ────────────────────────────────────────────────────────────
    def gate_of(self, prediction: Measurement) -> float:
        """该预测对应的门限。

        代价函数若实现了 ``gate_for(prediction)``（如 :class:`AdaptiveCost`），
        则用**逐预测门限**；否则回退到标量 ``cost.gate``。
        """
        resolver = getattr(self.cost, "gate_for", None)
        if callable(resolver):
            return float(resolver(prediction))
        return float(self.cost.gate)

    def cost_matrix(
        self,
        predictions: Sequence[Measurement],
        detections: Sequence[Measurement],
        gates: Sequence[float] | None = None,
    ) -> np.ndarray:
        """构造代价矩阵。

        ``gates`` 给出**逐预测门限**（由 :mod:`~TNsMultiTrackAdaptive.association.gate`
        的策略按航迹状态算出）；为 ``None`` 时回退到代价函数自带的门限
        （``cost.gate_for(prediction)`` 或标量 ``cost.gate``）。
        """
        if gates is not None and len(gates) != len(predictions):
            raise ValueError("gates 长度必须与 predictions 一致")
        matrix = np.full((len(predictions), len(detections)), FORBIDDEN, dtype=float)
        for i, prediction in enumerate(predictions):
            gate = float(gates[i]) if gates is not None else self.gate_of(prediction)
            for j, detection in enumerate(detections):
                value = self.cost(prediction, detection)
                if value <= gate:
                    matrix[i, j] = value
        return matrix

    def unmatched_cost(self, cost: np.ndarray) -> float:
        """``factor × max(有限代价)``；全被门限滤除时退化为 ``1e6``（沿用 MATLAB）。"""
        finite = cost[cost < FORBIDDEN]
        if finite.size == 0:
            return 1e6
        return float(self.unmatched_cost_factor * finite.max())

    # ── 主入口 ──────────────────────────────────────────────────────────────
    def associate(
        self,
        predictions: Sequence[Measurement],
        detections: Sequence[Measurement],
        gates: Sequence[float] | None = None,
    ) -> AssociationResult:
        for prediction in predictions:
            if prediction.category is not measClass.PRED:
                raise ValueError("GNN 的预测侧必须是 PRED 量测")
        for detection in detections:
            if detection.category is not measClass.REAL:
                raise ValueError("GNN 的实测侧必须是 REAL 量测")

        num_tracks, num_detections = len(predictions), len(detections)
        if num_tracks == 0 or num_detections == 0:
            return AssociationResult.all_unmatched(num_tracks, num_detections)

        matrix = self.cost_matrix(predictions, detections, gates)
        matches, unmatched_tracks, unmatched_detections = self.solver.solve(
            matrix, FORBIDDEN, self.unmatched_cost(matrix)
        )
        result = AssociationResult(
            matches=tuple(matches),
            unmatched_tracks=tuple(unmatched_tracks),
            unmatched_detections=tuple(unmatched_detections),
            cost=matrix,
        )
        if len(result.matches) + len(result.unmatched_tracks) != num_tracks:
            raise AssertionError("关联结果未覆盖全部航迹")
        if len(result.matches) + len(result.unmatched_detections) != num_detections:
            raise AssertionError("关联结果未覆盖全部量测")
        return result


__all__ = ["GNN"]
