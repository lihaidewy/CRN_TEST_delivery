"""生成"带量测丢失"的数据集，用于验证抗临时丢失能力。

对 ``meas.csv`` 的每帧、每个真值目标，以概率 ``--p`` 独立删除其量测
（量测↔目标归属见 ``utils/dropout.py``；未归属的离群量测保留）。

用法::

    # 每帧每目标 20% 概率丢失（默认）
    python TNsMultiTrackAdaptive/examples/make_dropout.py

    # 指定概率/随机种子/输出文件
    python TNsMultiTrackAdaptive/examples/make_dropout.py --p 0.3 --seed 7 \
        --out meas_dropout30.csv --mask dropout30_mask.csv

产物：处理后的量测 CSV 与 dropout 掩码（供 ``evaluate.py --dropout-mask`` 使用）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TNsMultiTrackAdaptive.utils.dropout import build_dataset  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="注入量测丢失（每帧每目标独立概率）")
    parser.add_argument("--meas", type=Path, default=REPO_ROOT / "meas.csv")
    parser.add_argument("--true", type=Path, default=REPO_ROOT.parent / "true.csv")
    parser.add_argument("--p", type=float, default=0.2, help="每帧每个目标的丢失概率")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（保证可复现）")
    parser.add_argument("--ts", type=float, default=0.1, help="帧间隔（秒）")
    parser.add_argument("--gate", type=float, default=2.0, help="量测↔真值归属门限（米）")
    parser.add_argument("--clutter", type=int, default=0,
                        help="每帧注入的近距杂波点数（放在随机目标周围，见 --clutter-range）")
    parser.add_argument("--clutter-range", type=float, nargs=2, default=(0.5, 3.0),
                        metavar=("MIN", "MAX"),
                        help="杂波到目标的距离区间（米）。放在 1.8–3.0 可专门检验"
                             "『宽门限会不会抓错点』")
    parser.add_argument("--tag", default=None,
                        help="输出文件名标签（默认按概率生成，例如 02 / 04）")
    parser.add_argument("--out", type=Path, default=None,
                        help="输出量测文件（默认 meas_dropout<tag>.csv）")
    parser.add_argument("--mask", type=Path, default=None,
                        help="输出 dropout 掩码（默认 dropout<tag>_mask.csv）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tag = args.tag or f"{args.p:.2g}".replace(".", "")
    out_path = args.out or (REPO_ROOT / f"meas_dropout{tag}.csv")
    mask_path = args.mask or (REPO_ROOT / f"dropout{tag}_mask.csv")

    print("=" * 74)
    print(f"注入量测丢失：每帧每目标 p={args.p}")
    print("=" * 74)
    stats = build_dataset(
        meas_path=args.meas,
        true_path=args.true,
        probability=args.p,
        seed=args.seed,
        ts=args.ts,
        gate=args.gate,
        out_path=out_path,
        mask_path=mask_path,
        clutter_per_frame=args.clutter,
        clutter_range=tuple(args.clutter_range),
    )
    print(stats.summary())
    print(f"\n量测文件: {out_path}")
    print(f"丢失掩码: {mask_path}")
    print(
        "\n下一步（评估抗丢失能力）:\n"
        f"  python TNsMultiTrackAdaptive/examples/evaluate.py --meas {out_path.name} "
        f"--dropout-mask {mask_path.name} --model CTRV --q-v 0.5"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
