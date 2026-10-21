"""运动模型实现层：注册表 + 工厂。

设计依据：``ALGORITHM_DESIGN.md`` §4.4。

已注册：
- ``CV``：匀速线性 KF（``cv_kf.py``）；
- ``CTRV``：CTRV-UKF（``ctrv_ukf.py``）；
- ``TFOT``：无模型 L0 多项式（``tfot_l0.py``）。

新增模型只需实现 ``MotionModel`` 四个接口与 ``from_config`` 并在此登记。
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional

from ..typedef.motion import MotionModel
from .ctrv_ukf import CTRVModel
from .cv_kf import CVModel
from .tfot_l0 import TFOTModel

MODELS: Mapping[str, Callable[..., MotionModel]] = {
    "CV": CVModel,
    "CTRV": CTRVModel,
    "TFOT": TFOTModel,
}


def make_model(name: str, config: Optional[object] = None, **kwargs: object) -> MotionModel:
    """按注册名构造模型。

    - 给了 ``**kwargs`` → 直接透传给构造函数（便于手动指定参数）；
    - 给了 ``config``（``ModelConfig``）→ 走类方法 ``from_config``；
    - 都没有 → 用模型自己的默认值。
    """
    try:
        factory = MODELS[name]
    except KeyError:  # pragma: no cover - 配置错误路径
        raise KeyError(f"未注册的运动模型 {name!r}；已注册: {sorted(MODELS)}") from None
    if kwargs:
        return factory(**kwargs)
    if config is not None:
        return factory.from_config(config)
    return factory()


__all__ = ["CTRVModel", "CVModel", "MODELS", "TFOTModel", "make_model"]
