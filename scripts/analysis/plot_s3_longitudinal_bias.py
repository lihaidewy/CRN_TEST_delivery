#!/usr/bin/env python3
"""Create longitudinal-bias scatter/trend figures for S3M and S3S evaluation outputs.

Inputs are the matched-prediction CSVs produced by the S3 evaluator at epoch 23.
The script writes a PNG, PDF and point-wise bias CSV for each congestion level.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
# ``SCRIPT_DIR`` is <project>/scripts/analysis.  Its direct parent is
# <project>/scripts, and the next parent is the portable project root.
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_longitudinal_bias import plot_bias, read_tp_rows, save_points


# Only edit these two entries if evaluating a different checkpoint/output folder.
JOBS = (
    (
        "S3M moderate congestion",
        PROJECT_ROOT / "outputs/evaluation/S3M_moderate/tracking_preds_tp.csv",
        PROJECT_ROOT / "outputs/evaluation/S3M_moderate/analysis/longitudinal_bias_scatter",
    ),
    (
        "S3S heavy congestion",
        PROJECT_ROOT / "outputs/evaluation/S3S_heavy/tracking_preds_tp.csv",
        PROJECT_ROOT / "outputs/evaluation/S3S_heavy/analysis/longitudinal_bias_scatter",
    ),
)
BIN_WIDTH_M = 10.0


def main() -> None:
    completed = 0
    for name, input_csv, out_prefix in JOBS:
        if not input_csv.is_file():
            print(f"[SKIP] {name}: evaluation TP CSV not found: {input_csv}")
            continue
        records = read_tp_rows(input_csv, min_score=None)
        if not records:
            raise RuntimeError(f"No matched predictions in {input_csv}")
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        save_points(out_prefix.with_name(f"{out_prefix.name}_points.csv"), records)
        slope, intercept = plot_bias(records, out_prefix, BIN_WIDTH_M, title=f"{name}: longitudinal prediction bias")
        signed_error = sum(float(row["signed_dx"]) for row in records) / len(records)
        print(f"[OK] {name}: TP={len(records)}, mean signed x error={signed_error:+.4f} m")
        print(f"     trend: e_x = {intercept:+.6f}{slope:+.6f} * x_gt")
        print(f"     figure: {out_prefix.with_suffix('.png')}")
        completed += 1
    if completed == 0:
        raise FileNotFoundError("No S3 evaluation TP CSV was found. Run scripts/run_evaluate_s3.sh first.")


if __name__ == "__main__":
    main()
