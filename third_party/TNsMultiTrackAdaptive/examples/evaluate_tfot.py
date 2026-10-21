"""TFOT（无模型 / L0 多项式）跟踪评估示例入口。

对应 `evaluate.py --model TFOT` 的薄封装::

    python TNsMultiTrackAdaptive/examples/evaluate_tfot.py                 # 默认参数
    python TNsMultiTrackAdaptive/examples/evaluate_tfot.py --sweep         # 起始参数扫描
    python TNsMultiTrackAdaptive/examples/evaluate_tfot.py --compare       # 与 CTRV 对照
    python TNsMultiTrackAdaptive/examples/evaluate_tfot.py --tfot-band 3 3 --tfot-order 2 2

TFOT 与 CTRV 的区别（同一套 Track/关联/起始，只有滤波器不同）：
- 状态就是位置 ``[x, y]``（2 维），**不维护协方差**，因此
  `--cost mahalanobis` 会被拒绝（马氏代价要求 `extra` 里放协方差）；
- 输入是**原始量测历史**（`TrackHistory.measurements`），在滑动窗口上逐维做
  L0 准则选阶后多项式拟合，`update` 取当前帧平滑值、`predict` 外推一帧；
- 参数：`--tfot-lambda`（选阶门限）、`--tfot-band`（窗口长度）、
  `--tfot-order`（最大阶数），默认与 MATLAB 一致（1 1 / 2 2 / 1 1）；
- 新航迹默认把 M/N 候选段的量测作为**预热窗口**（`--no-seed` 可关），
  否则量测驱动模型只有 1 个点、前几帧只能"保持位置"。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["--model", "TFOT", *sys.argv[1:]]))
