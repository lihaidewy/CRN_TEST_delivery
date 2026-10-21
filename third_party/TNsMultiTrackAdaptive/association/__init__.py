"""数据关联模块（可插拔：代价函数 ⊗ 门限策略 ⊗ 求解器 → AssociationResult）。

本副本额外提供：
- :class:`AdaptiveCost`：协方差驱动的自适应门限（代价层）；
- :mod:`~TNsMultiTrackAdaptive.association.gate`：**丢失帧数驱动**的逐航迹门限策略（推荐）。
"""

from .cost import FORBIDDEN, AdaptiveCost, CostFunction, EuclideanCost, MahalanobisCost
from .gate import AdaptiveGate, CovarianceGate, FixedGate, GatePolicy, build_gate_policy
from .gnn import GNN
from .solver import GreedySolver, HungarianSolver, Solver, default_solver

__all__ = [
    "FORBIDDEN",
    "AdaptiveCost",
    "AdaptiveGate",
    "CostFunction",
    "CovarianceGate",
    "EuclideanCost",
    "FixedGate",
    "GNN",
    "GatePolicy",
    "GreedySolver",
    "HungarianSolver",
    "MahalanobisCost",
    "Solver",
    "build_gate_policy",
    "default_solver",
]
