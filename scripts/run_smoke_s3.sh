#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/third_party/mmdetection3d:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# MultiStepLR in the S3 experiment requires at least three epochs. Running one
# batch per epoch verifies data loading, forward/backward, optimizer and saving.
python -u exps/det/CRN_s3_portable.py -b 1 --seed 0 --max_epochs 3 \
  --limit_train_batches 1 --limit_val_batches 0 --disable_early_stopping \
  --checkpoint_on_train_epoch_end --rolling_terminal_log \
  --default_root_dir outputs/smoke
