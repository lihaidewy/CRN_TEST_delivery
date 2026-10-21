"""多目标跟踪评估示例（模型可选：CTRV / TFOT）。

- **模型**：`--model CTRV`（CTRV-UKF）或 `--model TFOT`（无模型 L0 多项式），
  所有航迹统一使用所选模型；`--compare` 可一次跑两种模型做对照。
- **数据**：`meas.csv`（量测，逐帧多点、无 id）与 `true.csv`（真值，含 id 与速度）。
- **帧间隔**：100 ms（`--ts 0.1`）。
- **评估**：逐帧匈牙利配对（门限 `--eval-gate`，默认 2.0 m），统计
  TP/FP/FN、ID 切换、碎片化、位置 RMSE 与 MOTA，并给出每个真值目标的明细。

用法::

    python TNsMultiTrackAdaptive/examples/evaluate.py --model TFOT
    python TNsMultiTrackAdaptive/examples/evaluate.py --model CTRV --plot eval_ctrv.png
    python TNsMultiTrackAdaptive/examples/evaluate.py --compare
    python TNsMultiTrackAdaptive/examples/evaluate.py --model TFOT --sweep

两个专用入口 `evaluate_ctrv.py` / `evaluate_tfot.py` 只是本脚本的薄封装。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TNsMultiTrackAdaptive import (  # noqa: E402
    InitConfig,
    ModelConfig,
    TfotConfig,
    Tracker,
    TrackerConfig,
    evaluate,
    load_measurements,
    load_truth,
    min_frame,
)

MODEL_CHOICES = ("CV", "CTRV", "TFOT")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TNsMultiTrackAdaptive 多目标跟踪评估")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="CTRV",
                        help="所有航迹统一使用的运动模型（默认 CTRV）")
    parser.add_argument("--compare", action="store_true",
                        help="同时跑 CTRV 与 TFOT 并打印对照表")
    parser.add_argument("--meas", type=Path, default=REPO_ROOT / "meas.csv",
                        help="量测文件（默认仓库根目录 meas.csv）")
    parser.add_argument("--true", type=Path, default=REPO_ROOT.parent / "true.csv",
                        help="真值文件（默认 ../true.csv）")
    parser.add_argument("--meas-x", default="x", help="量测 x 列名（也可用 correct_x）")
    parser.add_argument("--meas-y", default="y", help="量测 y 列名（也可用 correct_y）")
    parser.add_argument("--ts", type=float, default=0.1, help="帧间隔（秒），默认 0.1")

    parser.add_argument("--cost", choices=("euclidean", "mahalanobis", "adaptive"),
                        default="euclidean",
                        help="距离度量；adaptive = 协方差驱动的门限（代价层，备选）")
    parser.add_argument("--gate-mode", choices=("fixed", "adaptive", "covariance"),
                        default="fixed",
                        help="门限策略：fixed 固定；adaptive 随连续丢失帧数放大（推荐）；"
                             "covariance 随预测协方差放大")
    parser.add_argument("--gate", type=float, default=1.5,
                        help="关联门限（米）；adaptive/covariance 模式下作为门限下限 base")
    parser.add_argument("--gate-growth", type=float, default=0.5,
                        help="adaptive：门限随连续丢失帧数的增长系数")
    parser.add_argument("--gate-innov", type=float, default=2.0,
                        help="adaptive：门限随本航迹新息 EWMA 的增长系数（推荐 2.0；0 = 只按丢失帧数）")
    parser.add_argument("--max-gate", type=float, default=3.5,
                        help="adaptive/covariance：门限上限（米），应低于目标最小间距")
    parser.add_argument("--eval-gate", type=float, default=2.0, help="评估配对门限（米）")
    parser.add_argument("--sigma", type=float, default=0.1, help="量测噪声标准差（米）")
    parser.add_argument("--q-v", type=float, default=1e-4,
                        help="CV/CTRV 速度过程噪声方差（MATLAB 默认 1e-4；本数据集调到 0.1 精度最好）")
    parser.add_argument("--max-range", type=float, default=1000.0,
                        help="位置自检上限（米），超过则视为数值发散")

    parser.add_argument("-N", "--window", type=int, default=4, help="起始滑窗帧数")
    parser.add_argument("-M", "--confirm", type=int, default=3, help="起始确认点数")
    parser.add_argument("--patience", type=int, default=7, help="连续丢失帧上限")
    parser.add_argument("--v-max", type=float, default=12.0, help="起始速度门限（m/s）")
    parser.add_argument("--no-seed", action="store_true",
                        help="新航迹不把候选段量测作为滤波器预热窗口（默认会预热）")

    parser.add_argument("--tfot-lambda", type=float, nargs=2, default=(1.0, 1.0),
                        metavar=("X", "Y"), help="TFOT 逐维选阶门限（默认 1 1）")
    parser.add_argument("--tfot-band", type=int, nargs=2, default=(2, 2),
                        metavar=("X", "Y"), help="TFOT 逐维窗口长度（默认 2 2）")
    parser.add_argument("--tfot-order", type=int, nargs=2, default=(1, 1),
                        metavar=("X", "Y"), help="TFOT 逐维最大拟合阶数（默认 1 1）")

    parser.add_argument("--plot", type=Path, default=None, help="保存轨迹对比图到该路径")
    parser.add_argument("--csv", type=Path, default=None, help="把逐帧估计导出为 CSV")
    parser.add_argument("--dropout-mask", type=Path, default=None,
                        help="量测丢失掩码（由 examples/make_dropout.py 生成）；"
                             "给出后额外统计抗临时丢失指标")
    parser.add_argument("--recovery-window", type=int, default=1,
                        help="丢失结束后允许几帧内接上算恢复（默认 1）")
    parser.add_argument("--posterior-only", action="store_true",
                        help="只评估命中帧（跳过滑行预测帧）")
    parser.add_argument("--baseline", action="store_true",
                        help="附带『原始量测（无跟踪）』基线：同口径打分，用于判断跟踪器带来了什么")
    parser.add_argument("--sweep", action="store_true",
                        help="额外扫描 (M,N) × patience 参数组合并打印对照表")

    args = parser.parse_args(argv)
    if args.cost == "mahalanobis" and args.model == "TFOT":
        parser.error("TFOT 不维护协方差（extra=None），无法使用 mahalanobis 代价；"
                     "请改用 --cost euclidean，或 --gate-mode adaptive（与模型无关）")
    if args.cost == "adaptive" and args.model == "TFOT":
        parser.error("--cost adaptive 依赖预测协方差，TFOT 不提供；"
                     "请改用 --gate-mode adaptive（该模式按丢失帧数，与模型无关）")
    return args


def build_config(args: argparse.Namespace, model_name: str | None = None) -> TrackerConfig:
    sigma2 = args.sigma**2
    return TrackerConfig(
        ts=args.ts,
        patience=args.patience,
        model_name=model_name or args.model,
        gate_mode=args.gate_mode,
        gate=args.gate,
        gate_growth=args.gate_growth,
        gate_innovation_weight=args.gate_innov,
        max_gate=args.max_gate,
        seed_measurements=not args.no_seed,
        model=ModelConfig(
            q_velocity=args.q_v,               # CV/CTRV 各自按状态维数展开 Q
            r=np.diag([sigma2, sigma2]),
            tfot=TfotConfig(
                lamda=tuple(args.tfot_lambda),
                band=tuple(args.tfot_band),
                fit_order=tuple(args.tfot_order),
            ),
        ),
        init=InitConfig(window_size=args.window, confirm=args.confirm, v_max=args.v_max),
    )


def load_data(args: argparse.Namespace):
    origin = min(min_frame(args.meas), min_frame(args.true))
    _, frames = load_measurements(
        args.meas, x_col=args.meas_x, y_col=args.meas_y, ts=args.ts, frame_origin=origin
    )
    truth = load_truth(args.true, ts=args.ts, frame_origin=origin)
    return frames, truth


def load_dropout(args: argparse.Namespace, origin: int):
    """读取 dropout 掩码（若有）。"""
    if args.dropout_mask is None:
        return None
    from TNsMultiTrackAdaptive.utils.dropout import load_dropout_mask

    return load_dropout_mask(args.dropout_mask, ts=args.ts, frame_origin=origin)


def build_tracker(
    args: argparse.Namespace, frames, model_name: str | None = None
) -> Tracker:
    config = build_config(args, model_name)
    tracker = Tracker(config, cost=args.cost)
    for timestamp, detections in frames:
        if args.cost == "mahalanobis":
            detections = [replace(m, extra=config.model.r) for m in detections]
        tracker.update(detections, timestamp)
    return tracker


def run(args: argparse.Namespace) -> tuple[Tracker, dict]:
    frames, truth = load_data(args)
    return build_tracker(args, frames), truth


def raw_measurement_estimates(frames) -> dict:
    """基线：把每帧每条量测当作一个"匿名估计"（无关联、无身份）。"""
    baseline: dict = {}
    for timestamp, detections in frames:
        baseline[timestamp] = {index: m.value for index, m in enumerate(detections)}
    return baseline


def compare_models(args: argparse.Namespace) -> None:
    """CTRV 与 TFOT 对照：同一份数据、同一套起始与关联参数。"""
    frames, truth = load_data(args)
    print("\n── 模型对照（其余参数完全一致）" + "─" * 40)
    print(f"{'模型':>6} {'TP':>5} {'FP':>4} {'FN':>4} {'ID切换':>6} {'碎片':>5} "
          f"{'航迹数':>6} {'MOTA':>7} {'RMSE(m)':>8} {'P90(m)':>7} {'P95(m)':>7}")
    if args.baseline:
        report = evaluate(raw_measurement_estimates(frames), truth, gate=args.eval_gate)
        print(
            f"{'原始量测':>6} {report.matched:>5} {report.false_positives:>4} "
            f"{report.false_negatives:>4} {'—':>6} {'—':>5} {'—':>6} "
            f"{'—':>7} {report.rmse:>8.4f} {report.p90_error:>7.4f} {report.p95_error:>7.4f}"
        )
    for name in MODEL_CHOICES:
        tracker = build_tracker(args, frames, model_name=name)
        report = evaluate(tracker.estimates(), truth, gate=args.eval_gate)
        print(
            f"{name:>6} {report.matched:>5} {report.false_positives:>4} "
            f"{report.false_negatives:>4} {report.id_switches:>6} {report.fragments:>5} "
            f"{report.track_count:>6} {report.mota:>7.4f} {report.rmse:>8.4f} "
            f"{report.p90_error:>7.4f} {report.p95_error:>7.4f}"
        )
    if args.baseline:
        print("  注：原始量测没有身份概念（每条量测每帧都是新『身份』），"
              "故 ID 切换 / 碎片 / 航迹数 / MOTA 不适用（显示为 —）；")
        print("      它的 RMSE 是『直接用量测当估计』的口径，用于判断跟踪器是否真的提升了精度。")


def sweep(args: argparse.Namespace) -> None:
    """扫描起始窗口与 patience：两者分别决定确认延迟与滑行虚警。"""
    frames, truth = load_data(args)
    print(f"\n── 参数扫描（模型={args.model}，代价={args.cost}）" + "─" * 35)
    print(f"{'M/N':>6} {'patience':>9} {'TP':>5} {'FP':>4} {'FN':>4} {'ID切换':>6} "
          f"{'航迹数':>6} {'MOTA':>7} {'RMSE(m)':>8} {'P90(m)':>7}")
    for window, confirm in ((3, 2), (4, 2), (4, 3)):
        for patience in (5, 7, 10):
            local = argparse.Namespace(**vars(args))
            local.window, local.confirm, local.patience = window, confirm, patience
            tracker = build_tracker(local, frames)
            report = evaluate(tracker.estimates(), truth, gate=args.eval_gate)
            print(
                f"{confirm}/{window:>4} {patience:>9} {report.matched:>5} "
                f"{report.false_positives:>4} {report.false_negatives:>4} "
                f"{report.id_switches:>6} {report.track_count:>6} "
                f"{report.mota:>7.4f} {report.rmse:>8.4f} {report.p90_error:>7.4f}"
            )


def check_divergence(tracker: Tracker, limit: float) -> list:
    """检查是否有航迹状态跑到数据范围之外（数值发散的自检）。"""
    diverged = []
    for track in tracker.all_tracks():
        for state in track.history:
            if float(max(abs(state.value[0]), abs(state.value[1]))) > limit:
                diverged.append(track)
                break
    return diverged


def report_tracks(tracker: Tracker, ts: float) -> None:
    tracks = tracker.all_tracks()
    print(f"\n── 航迹明细（共 {len(tracks)} 条；活跃 {len(tracker.tracks)}）" + "─" * 30)
    print(f"{'id':>4} {'帧数':>5} {'命中':>5} {'丢失':>5} {'生命周期':>10} "
          f"{'证据(帧)':>8} {'确认延迟':>8} {'质量':>6} {'模型':>6}")
    for track in sorted(tracks, key=lambda t: t.id):
        evidence_frame = round(track.first_evidence_time / ts)
        delay = (track.birth - track.first_evidence_time) / ts
        print(
            f"{track.id:>4} {len(track.history):>5} {track.hits:>5} {track.misses:>5} "
            f"{track.lifecycle.value:>10} {evidence_frame:>8} {delay:>8.0f} "
            f"{track.init_quality:>6.3f} {track.model_name:>6}"
        )


def plot_result(tracker: Tracker, truth: dict, report, path: Path, label: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - 环境相关
        print("\n[跳过绘图] 未安装 matplotlib")
        return

    # 图内标签统一用英文：默认字体（DejaVu Sans）缺中文字形，会渲染成方框
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 8))

    series = _truth_series(truth)
    for index, (_gt_id, points) in enumerate(series.items()):
        left.plot(
            points[:, 0],
            points[:, 1],
            "k--",
            linewidth=1.0,
            alpha=0.55,
            label="ground truth" if index == 0 else None,
        )
    for track in tracker.all_tracks():
        positions = np.array([state.value[:2] for state in track.history])
        left.plot(positions[:, 0], positions[:, 1], linewidth=1.2, label=f"track {track.id}")
        left.plot(positions[0, 0], positions[0, 1], "o", markersize=3.5)
        left.plot(positions[-1, 0], positions[-1, 1], "x", markersize=3.5)
    left.set_xlabel("x (m)")
    left.set_ylabel("y (m)")
    left.set_title(f"trajectories ({label}): tracks (solid) vs truth (dashed)")
    left.grid(True, alpha=0.3)
    left.legend(fontsize=7, ncol=2)

    if report.frame_errors:
        frames = sorted(report.frame_errors)
        values = [report.frame_errors[frame] for frame in frames]
        right.plot(frames, values, linewidth=0.9, color="tab:blue")
        right.axhline(report.rmse, color="tab:red", linestyle="--", linewidth=1.0,
                      label=f"RMSE={report.rmse:.3f} m")
        right.axhline(report.p90_error, color="tab:orange", linestyle="-.", linewidth=1.0,
                      label=f"P90={report.p90_error:.3f} m")
        right.axhline(report.p95_error, color="tab:green", linestyle="-.", linewidth=0.8,
                      label=f"P95={report.p95_error:.3f} m")
        right.axhline(report.gate, color="0.6", linestyle=":", linewidth=1.0,
                      label=f"eval gate={report.gate:.1f} m")
        right.set_ylim(0, min(report.gate, max(values) * 1.15 + 1e-6))
        right.set_xlabel("time (s)")
        right.set_ylabel("mean position error (m)")
        right.set_title(f"per-frame error ({label})   MOTA={report.mota:.4f}   "
                        f"FP={report.false_positives}  FN={report.false_negatives}  "
                        f"IDSW={report.id_switches}")
        right.grid(True, alpha=0.3)
        right.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    print(f"\n轨迹图已保存: {path}")


def _truth_series(truth: dict) -> dict[int, np.ndarray]:
    series: dict[int, list] = {}
    for timestamp in sorted(truth):
        for gt_id, position in truth[timestamp]:
            series.setdefault(gt_id, []).append(position)
    return {gt_id: np.array(points) for gt_id, points in series.items()}


def export_csv(tracker: Tracker, path: Path) -> None:
    rows = ["time_s,track_id,x,y,category"]
    for track in tracker.all_tracks():
        for state in track.history:
            rows.append(
                f"{state.timestamp:.3f},{track.id},{state.value[0]:.6f},"
                f"{state.value[1]:.6f},{state.category.value}"
            )
    path.write_text("\n".join(rows), encoding="utf-8")
    print(f"逐帧估计已导出: {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=" * 78)
    print(f"TNsMultiTrackAdaptive · {args.model} 全模型跟踪评估")
    print("=" * 78)
    print(f"量测文件 : {args.meas}")
    print(f"真值文件 : {args.true}")
    print(f"帧间隔   : {args.ts * 1000:.0f} ms    关联代价: {args.cost}    门限: {args.gate} m")
    print(f"起始参数 : N={args.window} M={args.confirm} patience={args.patience} "
          f"v_max={args.v_max} m/s    预热窗口: {'开' if not args.no_seed else '关'}")
    if args.model == "TFOT":
        print(f"TFOT 参数: lamda={tuple(args.tfot_lambda)} band={tuple(args.tfot_band)} "
              f"fit_order={tuple(args.tfot_order)}")
    else:
        print(f"{args.model} 参数: R=σ²I (σ={args.sigma})    Q_v={args.q_v}（MATLAB 默认 1e-4）")

    frames, truth = load_data(args)
    tracker = build_tracker(args, frames)
    dropout = load_dropout(args, min(min_frame(args.meas), min_frame(args.true)))
    if dropout is not None:
        print(f"丢失掩码 : {args.dropout_mask}（{sum(len(v) for v in dropout.values())} 个槽位）")

    diverged = check_divergence(tracker, args.max_range)
    if diverged:
        print(f"\n[警告] {len(diverged)} 条航迹的位置超出 ±{args.max_range:.0f} m（疑似数值发散）: "
              f"{[(track.id, round(float(max(abs(s.value[0]), abs(s.value[1])) for s in track.history))) for track in diverged]}")

    report = evaluate(
        tracker.estimates(only_posterior=args.posterior_only), truth,
        gate=args.eval_gate, dropout=dropout, recovery_window=args.recovery_window,
    )

    print("\n── 总体指标" + "─" * 62)
    print(report.summary())

    if args.baseline:
        baseline = evaluate(raw_measurement_estimates(frames), truth, gate=args.eval_gate)
        print(
            f"\n  基线·原始量测(无跟踪): TP={baseline.matched} FP={baseline.false_positives} "
            f"FN={baseline.false_negatives} RMSE={baseline.rmse:.4f} "
            f"P90={baseline.p90_error:.4f} P95={baseline.p95_error:.4f} "
            f"覆盖率={baseline.recall:.4f}"
        )
        print("  → 它没有身份概念（ID 指标不适用）；用于判断跟踪器是否真的提升了精度")

    if not args.posterior_only:
        updated_only = evaluate(tracker.estimates(only_posterior=True), truth,
                                gate=args.eval_gate)
        print(
            f"\n  对照·仅命中帧(跳过滑行预测): FP={updated_only.false_positives} "
            f"FN={updated_only.false_negatives} ID切换={updated_only.id_switches} "
            f"MOTA={updated_only.mota:.4f} RMSE={updated_only.rmse:.4f} "
            f"P90={updated_only.p90_error:.4f} P95={updated_only.p95_error:.4f}"
        )
        print("  → 滑行预测帧是虚警的主要来源（设计取舍：用 patience 换抗遮挡能力）")

    print("\n── 每个真值目标" + "─" * 56)
    print(f"{'gt_id':>6} {'帧数':>5} {'命中':>5} {'覆盖率':>7} {'RMSE(m)':>8} {'P90(m)':>7} "
          f"{'轨迹数':>6} {'切换':>5} {'碎片':>5}")
    for gt_id in sorted(report.per_id):
        stats = report.per_id[gt_id]
        print(
            f"{gt_id:>6} {stats.frames:>5} {stats.matched:>5} {stats.coverage:>7.3f} "
            f"{stats.rmse:>8.4f} {stats.p90:>7.4f} {stats.distinct_tracks:>6} "
            f"{stats.switches:>5} {stats.fragments:>5}"
        )

    report_tracks(tracker, args.ts)

    if args.compare:
        compare_models(args)
    if args.sweep:
        sweep(args)
    if args.plot is not None:
        plot_result(tracker, truth, report, args.plot, args.model)
    if args.csv is not None:
        export_csv(tracker, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
