"""契约层：值对象、接口与关联结果。

设计依据：``ALGORITHM_DESIGN.md`` 第 4 节。
"""

from .association import AssociationResult
from .history import TrackHistory
from .meas import Measurement, measClass
from .motion import MotionModel
from .state import State, stateClass
from .track import Track, trackState

__all__ = [
    "AssociationResult",
    "Measurement",
    "MotionModel",
    "State",
    "Track",
    "TrackHistory",
    "measClass",
    "stateClass",
    "trackState",
]

__version__ = "0.1.0"
