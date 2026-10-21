"""CTRV 全模型跟踪评估入口（`evaluate.py --model CTRV` 的薄封装）。

CTRV 亦是 `evaluate.py` 的默认模型，因此下面两条命令等价::

    python TNsMultiTrackAdaptive/examples/evaluate_ctrv.py
    python TNsMultiTrackAdaptive/examples/evaluate.py --model CTRV

CTRV 用 UKF 维护 `[x, y, v, ψ, ψ̇]` 与协方差，适合机动目标；
与无模型的 TFOT 对照请用 `--compare`。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["--model", "CTRV", *sys.argv[1:]]))
