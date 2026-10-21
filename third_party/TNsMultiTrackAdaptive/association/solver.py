"""指派求解器：决定"怎么配对"。

设计依据：``ALGORITHM_DESIGN.md`` §5.3.5。

- :class:`HungarianSolver`：``scipy.optimize.linear_sum_assignment`` + 增广矩阵，全局最优
  （等价于 MATLAB ``matchpairs``）。
- :class:`GreedySolver`：代价升序贪心，作为无 scipy 环境的兜底与快速路径
  （MATLAB 里那个 ``hungarianAlgorithm`` 兜底其实是未实现的，这里补齐）。
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Solver(Protocol):
    """求解器协议。"""

    name: str

    def solve(
        self, cost: np.ndarray, forbidden: float, unmatched_cost: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """返回 ``(matches, unmatched_tracks, unmatched_detections)``。"""


class HungarianSolver:
    """全局最优指派（增广矩阵构造见设计报告附录 B）。"""

    name = "hungarian"

    def __init__(self, forbidden_scale: float = 10.0) -> None:
        self.forbidden_scale = float(forbidden_scale)

    def solve(
        self, cost: np.ndarray, forbidden: float, unmatched_cost: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        from scipy.optimize import linear_sum_assignment

        num_tracks, num_detections = cost.shape
        size = num_tracks + num_detections
        big = unmatched_cost * (min(num_tracks, num_detections) + 1) * self.forbidden_scale

        augmented = np.full((size, size), big, dtype=float)
        augmented[:num_tracks, :num_detections] = cost
        for index in range(num_tracks):
            augmented[index, num_detections + index] = unmatched_cost
        for index in range(num_detections):
            augmented[num_tracks + index, index] = unmatched_cost
        augmented[num_tracks:, num_detections:] = 0.0

        rows, cols = linear_sum_assignment(augmented)
        matches: list[tuple[int, int]] = []
        for row, col in zip(rows, cols):
            if row < num_tracks and col < num_detections and augmented[row, col] < forbidden:
                matches.append((int(row), int(col)))

        matched_tracks = {row for row, _ in matches}
        matched_detections = {col for _, col in matches}
        unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]
        unmatched_detections = [j for j in range(num_detections) if j not in matched_detections]
        return matches, unmatched_tracks, unmatched_detections


class GreedySolver:
    """代价升序贪心（行/列互斥）。"""

    name = "greedy"

    def solve(
        self, cost: np.ndarray, forbidden: float, unmatched_cost: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        num_tracks, num_detections = cost.shape
        order = np.argsort(cost, axis=None)
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        matches: list[tuple[int, int]] = []
        for flat in order:
            row, col = divmod(int(flat), num_detections)
            if cost[row, col] >= forbidden:
                break
            if cost[row, col] >= unmatched_cost:
                break
            if row in used_tracks or col in used_detections:
                continue
            used_tracks.add(row)
            used_detections.add(col)
            matches.append((row, col))
        unmatched_tracks = [i for i in range(num_tracks) if i not in used_tracks]
        unmatched_detections = [j for j in range(num_detections) if j not in used_detections]
        return matches, unmatched_tracks, unmatched_detections


def default_solver() -> Solver:
    """优先返回全局最优求解器；无 scipy 时退化为贪心。"""
    try:
        import scipy.optimize  # noqa: F401
    except ImportError:  # pragma: no cover - 环境相关
        return GreedySolver()
    return HungarianSolver()


__all__ = ["GreedySolver", "HungarianSolver", "Solver", "default_solver"]
