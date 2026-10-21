"""配置对象。

设计依据：``ALGORITHM_DESIGN.md`` §7。

- 门限**不在** ``TrackerConfig`` 里：它属于代价函数（保证与代价同量纲）。
- 默认值面向本仓库的 ``meas.csv`` / ``true.csv``（Ts=0.1 s、量测噪声≈0.1 m、
  目标间距≥3.5 m、0.55 m/帧），因此在 MATLAB 默认值上收紧了运动学门限。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


def _diag(values: Tuple[float, ...]) -> np.ndarray:
    return np.diag(list(values))


@dataclass(frozen=True)
class TfotConfig:
    """TFOT/L0 参数：逐维的选阶门限、窗口长度、最大阶数（与 MATLAB 默认一致）。"""

    lamda: Tuple[float, float] = (1.0, 1.0)
    band: Tuple[int, int] = (2, 2)
    fit_order: Tuple[int, int] = (1, 1)


@dataclass(frozen=True)
class ModelConfig:
    """运动模型的噪声与初值参数（与状态维数**解耦**：各模型按自身维数展开）。

    - ``q_velocity``：速度分量的过程噪声方差，位置分量固定为 0
      （MATLAB ``D2Tracks.m`` 的 ``Q`` 就是这个结构）；
    - ``p0_position`` / ``p0_velocity``：初始协方差对角元；
    - ``r``：量测噪声协方差（σ 由调用方给出，例如评估脚本的 ``--sigma``）；
    - ``tfot``：TFOT 专属参数。
    """

    q_velocity: float = 1e-4
    p0_position: float = 4.0
    p0_velocity: float = 0.1
    r: np.ndarray = field(default_factory=lambda: _diag((0.01, 0.01)))       # σ≈0.1 m
    tfot: TfotConfig = field(default_factory=TfotConfig)

    def build_q(self, state_dim: int) -> np.ndarray:
        """按状态维数展开过程噪声：前两维（位置）为 0，其余（速度/转率）取 ``q_velocity``。"""
        return np.diag([0.0] * 2 + [self.q_velocity] * (state_dim - 2))

    def build_p0(self, state_dim: int) -> np.ndarray:
        """按状态维数展开初始协方差：前两维（位置）用 ``p0_position``，其余用 ``p0_velocity``。"""
        return np.diag([self.p0_position] * 2 + [self.p0_velocity] * (state_dim - 2))


@dataclass(frozen=True)
class InitConfig:
    """集中式起始模块参数。"""

    window_size: int = 4          # N
    confirm: int = 3              # M
    v_min: float = 0.0
    v_max: float = 12.0           # m/s；距离门限 = v_max × 帧间隔
    a_max: float = 200.0          # m/s²
    angle_max: float = 45.0       # deg
    max_branch: int = 8
    max_candidates: int = 64
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)


@dataclass(frozen=True)
class TrackerConfig:
    """编排层配置。"""

    ts: float = 0.1
    patience: int = 7
    lost_retention_frames: int = 0
    unmatched_cost_factor: float = 2.0
    model_name: str = "CTRV"      # 注册名：CV / CTRV / TFOT
    gate_mode: str = "fixed"      # 门限策略：fixed / adaptive（按丢失帧数）/ covariance
    gate: float = 1.5             # 门限（米）：fixed 时为常数；adaptive 时为下限 base
    gate_growth: float = 0.5      # adaptive：gate = clip(base + growth·sqrt(连续丢失帧数) + w·新息EWMA, base, max_gate)
    gate_innovation_weight: float = 2.0   # adaptive：新息 EWMA 权重（推荐 2.0：实测能把"机动"与"杂波"分开）
    max_gate: float = 3.5         # adaptive：门限上限（应低于目标最小间距）
    seed_measurements: bool = True  # 新航迹是否把候选段量测作为滤波器预热窗口（§5.4.5）
    model: ModelConfig = field(default_factory=ModelConfig)
    init: InitConfig = field(default_factory=InitConfig)


__all__ = ["InitConfig", "ModelConfig", "TfotConfig", "TrackerConfig"]
