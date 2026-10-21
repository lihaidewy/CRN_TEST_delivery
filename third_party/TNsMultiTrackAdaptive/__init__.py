"""TNsMultiTrackAdaptive：``TNsMultiTrack`` 的**副本**，新增自适应关联门限。

与原库的关系
------------
- 原库 ``TNsMultiTrack/`` **保持不变**（本副本是独立目录，两者可并存、可对照）；
- 本副本唯一的算法改动：``association/cost.py`` 新增 :class:`AdaptiveCost`
  （门限随预测协方差自动膨胀/收缩），以及 ``association/gnn.py`` 支持逐预测门限。

设计蓝图仍见 ``TNsMultiTrack/ALGORITHM_DESIGN.md``；本副本的实验结论见
``EVALUATION_ADAPTIVE_GATE.md``。
"""

from .association import (
    GNN,
    AdaptiveCost,
    AdaptiveGate,
    CovarianceGate,
    EuclideanCost,
    FixedGate,
    GreedySolver,
    HungarianSolver,
    MahalanobisCost,
    build_gate_policy,
)
from .config import InitConfig, ModelConfig, TfotConfig, TrackerConfig
from .initialization import MNInitializer
from .motion import CTRVModel, CVModel, TFOTModel, make_model
from .tracker import Tracker
from .typedef import (
    AssociationResult,
    Measurement,
    MotionModel,
    State,
    Track,
    TrackHistory,
    measClass,
    stateClass,
    trackState,
)
from .utils import EvaluationReport, evaluate, load_measurements, load_truth, min_frame

__all__ = [
    "AdaptiveCost",
    "AdaptiveGate",
    "AssociationResult",
    "CTRVModel",
    "CVModel",
    "CovarianceGate",
    "EvaluationReport",
    "EuclideanCost",
    "FixedGate",
    "GNN",
    "GreedySolver",
    "HungarianSolver",
    "InitConfig",
    "MNInitializer",
    "MahalanobisCost",
    "Measurement",
    "ModelConfig",
    "MotionModel",
    "State",
    "TFOTModel",
    "TfotConfig",
    "Track",
    "TrackHistory",
    "Tracker",
    "TrackerConfig",
    "build_gate_policy",
    "evaluate",
    "load_measurements",
    "load_truth",
    "make_model",
    "measClass",
    "min_frame",
    "stateClass",
    "trackState",
]

__version__ = "0.4.1-adaptive"

