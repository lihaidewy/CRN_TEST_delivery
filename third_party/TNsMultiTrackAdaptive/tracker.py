"""编排层：``Tracker``。

设计依据：``ALGORITHM_DESIGN.md`` §5.6 / §6。

它只做五件事：**预测收集 → 关联 → 分发结果 → 退休失效航迹 → 收编新航迹**。
不认识 CV/CTRV/TFOT，不做任何数值计算（G3）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .association import GNN, AdaptiveCost, EuclideanCost, MahalanobisCost
from .association.gate import GatePolicy, build_gate_policy
from .config import TrackerConfig
from .initialization import MNInitializer
from .motion import make_model
from .typedef.association import AssociationResult
from .typedef.meas import Measurement, measClass
from .typedef.motion import MotionModel
from .typedef.state import State, stateClass
from .typedef.track import Track


class Tracker:
    """多目标跟踪编排器。"""

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        *,
        model: Optional[MotionModel] = None,
        associator: Optional[GNN] = None,
        initializer: Optional[MNInitializer] = None,
        cost: Optional[str] = None,
        gate_policy: Optional[GatePolicy] = None,
    ) -> None:
        self.config = config or TrackerConfig()
        cfg = self.config
        self.model = model or make_model(cfg.model_name, cfg.model)
        self.associator = associator or GNN(
            cost=self._build_cost(cost), unmatched_cost_factor=cfg.unmatched_cost_factor
        )
        self.gate_policy: GatePolicy = gate_policy or build_gate_policy(
            cfg.gate_mode, cfg.gate, cfg.gate_growth, cfg.max_gate,
            cfg.gate_innovation_weight,
        )
        self.initializer = initializer or MNInitializer(
            self.model,
            window_size=cfg.init.window_size,
            confirm=cfg.init.confirm,
            patience=cfg.patience,
            v_min=cfg.init.v_min,
            v_max=cfg.init.v_max,
            a_max=cfg.init.a_max,
            angle_max=cfg.init.angle_max,
            max_branch=cfg.init.max_branch,
            max_candidates=cfg.init.max_candidates,
            weights=cfg.init.weights,
            frame_interval=cfg.ts,
            seed_measurements=cfg.seed_measurements,
        )
        self.tracks: list[Track] = []
        self.dead_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self.frame: int = 0
        self.time: Optional[float] = None
        #: 最近一帧逐航迹使用的门限（诊断用）
        self.last_gates: list[float] = []

    def _build_cost(self, cost: Optional[str]):
        kind = (cost or "euclidean").lower()
        if kind == "euclidean":
            return EuclideanCost(self.config.gate)
        if kind == "mahalanobis":
            return MahalanobisCost()
        if kind == "adaptive":
            return AdaptiveCost(
                base=self.config.gate,
                k=self.config.gate_k,
                max_gate=self.config.max_gate,
            )
        raise ValueError(
            f"未知代价类型 {cost!r}（可选 euclidean / mahalanobis / adaptive）"
        )

    # ── 主循环 ──────────────────────────────────────────────────────────────
    def update(
        self, detections: Sequence[Measurement], timeStamp: Optional[float] = None
    ) -> None:
        for detection in detections:
            if detection.category is not measClass.REAL:
                raise ValueError("Tracker 只接受 REAL 量测")

        self.frame += 1
        if timeStamp is None:
            timeStamp = self.frame * self.config.ts
        timeStamp = float(timeStamp)
        self.time = timeStamp

        # 0) 冻结航迹（保留期）绝不进入预测/关联
        active = [track for track in self.tracks if track.alive]
        frozen = [track for track in self.tracks if not track.alive]

        # 1) 预测：航迹自己算，管理器只收集返回值
        predictions = [track.predict(timeStamp) for track in active]
        if any(prediction is None for prediction in predictions):   # C5 保证
            raise AssertionError("活跃航迹必须给出预测量测")

        # 2) 关联（门限策略逐条给出：fixed 时是常数，adaptive 时随连续丢失帧数放大）
        if active:
            gates = [self.gate_policy.gate(track) for track in active]
            self.last_gates = gates
            result = self.associator.associate(predictions, detections, gates)
        else:
            self.last_gates = []
            result = AssociationResult.all_unmatched(0, len(detections))

        # 3) 分发：命中的更新，未命中的滑行
        hit_of = result.track_to_detection
        for index, track in enumerate(active):
            detection = detections[hit_of[index]] if index in hit_of else None
            track.update(detection)

        # 4) 退休：本帧判定 LOST 的航迹移出活跃列表
        retired = [track for track in active if not track.alive]
        self.tracks = [track for track in active if track.alive]
        if self.config.lost_retention_frames > 0:
            self.lost_tracks.extend(retired)
        else:
            self.dead_tracks.extend(frozen + retired)

        # 5) 起始：未被关联的量测交给集中式起始模块
        unmatched = [detections[index] for index in result.unmatched_detections]
        self.tracks.extend(self.initializer.update(unmatched, timeStamp, self.frame))

    # ── 输出辅助 ────────────────────────────────────────────────────────────
    def all_tracks(self) -> list[Track]:
        """全部航迹（死亡 + 活跃 + 保留期）。"""
        return [*self.dead_tracks, *self.lost_tracks, *self.tracks]

    def snapshot_all(self, timeStamp: float) -> dict[int, State]:
        """某一时刻所有航迹的状态：``{track_id: State}``（缺失的航迹不出现）。"""
        snapshot: dict[int, State] = {}
        for track in self.all_tracks():
            state = track.snapshot(timeStamp)
            if state is not None:
                snapshot[track.id] = state
        return snapshot

    def estimates(
        self, only_posterior: bool = False
    ) -> dict[float, dict[int, np.ndarray]]:
        """把所有航迹的历史摊平成 ``{时刻: {航迹id: 位置}}``，供评估使用。

        ``only_posterior=True`` 时只输出**命中帧**（``POSTERIOR``）的状态，
        跳过滑行预测帧——用于把"精度"与"滑行外推"的贡献分开看
        （滑行帧是虚警的主要来源，见评估示例）。
        """
        by_time: dict[float, dict[int, np.ndarray]] = {}
        for track in self.all_tracks():
            for state in track.history:
                if only_posterior and state.category is not stateClass.POSTERIOR:
                    continue
                by_time.setdefault(state.timestamp, {})[track.id] = state.value[:2].copy()
        return by_time

    def __repr__(self) -> str:
        return (
            f"<Tracker frame={self.frame} active={len(self.tracks)} "
            f"dead={len(self.dead_tracks)} model={self.model.name}>"
        )


__all__ = ["Tracker"]
