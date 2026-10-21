"""最小可用示例：源码方式使用 TNsMultiTrackAdaptive（不需要安装、不需要 pip）。

场景：合成 2 个目标（一个匀速直线、一个匀速转弯），逐帧喂给 Tracker，
最后打印航迹摘要并做几条自检断言。

运行::

    python TNsMultiTrackAdaptive/examples/minimal.py

把这个文件复制到你自己的工程里时，只要保证 `TNsMultiTrackAdaptive/` 目录在你的
`sys.path` 上（最简单：与你的脚本同级），其余依赖只有 ``numpy``。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── 1) 让 TNsMultiTrackAdaptive 可导入（源码方式：把它放在你的项目里，这里指到它）──
PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from TNsMultiTrackAdaptive import (  # noqa: E402
    InitConfig,
    Measurement,
    ModelConfig,
    Tracker,
    TrackerConfig,
)

TS = 0.1            # 帧间隔 100 ms
FRAMES = 60         # 帧数
NOISE = 0.05        # 量测噪声 σ（米）


def simulate() -> list[list[np.ndarray]]:
    """返回逐帧量测：``frames[k] = [[x, y], ...]``。"""
    rng = np.random.default_rng(0)
    frames: list[list[np.ndarray]] = []
    for k in range(FRAMES):
        t = k * TS
        # 目标 A：从 (0,0) 出发，以 5 m/s 沿 +x 匀速直线
        a = np.array([5.0 * t, 0.0])
        # 目标 B：从 (0,20) 出发，以 5 m/s 做半径 10 m 的匀速圆周（ω = v/r）
        angle = (5.0 / 10.0) * t
        b = np.array([10.0 * np.sin(angle), 20.0 - 10.0 * (1.0 - np.cos(angle))])
        detections = [a + rng.normal(0.0, NOISE, size=2), b + rng.normal(0.0, NOISE, size=2)]
        frames.append(detections)
    return frames


def main() -> int:
    # ── 2) 配置：帧间隔、模型、门限、起始参数 ──────────────────────────────
    config = TrackerConfig(
        ts=TS,
        model_name="CTRV",              # 可换 "CV" / "TFOT"
        gate=1.5,                       # 关联门限（米）
        patience=7,                     # 连续丢失 7 帧后判死
        # 合成数据与 CTRV 假设完全一致（匀速直线 / 匀速转弯），故用默认 Q 即可。
        # 真实数据通常要调 q_velocity，见 EVALUATION_MODELS.md（本仓库数据用 0.5）。
        model=ModelConfig(),
        init=InitConfig(window_size=4, confirm=3, v_max=12.0),
    )
    tracker = Tracker(config)           # 也可 Tracker(config, cost="mahalanobis")

    # ── 3) 逐帧喂数据（时间戳必须严格递增）────────────────────────────────
    for index, detections in enumerate(simulate()):
        timestamp = index * TS
        measurements = [Measurement(timestamp, position) for position in detections]
        tracker.update(measurements, timestamp)

    # ── 4) 读取结果 ────────────────────────────────────────────────────────
    print(f"活跃航迹 {len(tracker.tracks)} 条，已终结 {len(tracker.dead_tracks)} 条")
    print(f"{'id':>3} {'状态':>10} {'帧数':>5} {'命中':>5} {'丢失':>5} {'起始(s)':>8} {'末位置':>18}")
    for track in sorted(tracker.all_tracks(), key=lambda t: t.id):
        last = track.fresh.value[:2]
        print(
            f"{track.id:>3} {track.lifecycle.value:>10} {len(track.history):>5} "
            f"{track.hits:>5} {track.misses:>5} {track.birth:>8.2f} "
            f"({last[0]:>7.3f},{last[1]:>7.3f})"
        )

    # 某一时刻的全部状态（{track_id: State}）
    snapshot = tracker.snapshot_all(timestamp)
    print(f"\nt={timestamp:.1f}s 时刻快照: {sorted(snapshot)}")

    # 摊平成 {时刻: {track_id: 位置}}，便于评估/落盘
    estimates = tracker.estimates()
    print(f"逐帧估计共 {len(estimates)} 个时刻；是否只取命中帧: "
          f"{len(tracker.estimates(only_posterior=True))} 个时刻")

    # ── 5) 自检（合成数据下应稳定跟踪 2 个目标）─────────────────────────────
    assert len(tracker.all_tracks()) == 2, "应当恰好起出 2 条航迹"
    assert all(track.alive for track in tracker.tracks), "两条航迹都应存活"
    assert all(track.lifecycle.value == "confirmed" for track in tracker.tracks)
    # 起批延迟 = N-1 帧，之后每帧一列状态
    for track in tracker.tracks:
        assert len(track.history) == FRAMES - (config.init.window_size - 1), len(track.history)
        # 两个目标的真实速率都是 5 m/s；CTRV 状态里的速度应估准（容差 0.2 m/s）
        assert abs(track.fresh.value[2] - 5.0) < 0.2, track.fresh.value
    print("\n自检通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
