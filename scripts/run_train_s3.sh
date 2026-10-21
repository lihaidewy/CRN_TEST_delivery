#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${1:-100}"
if [[ $# -gt 0 ]]; then
  shift
fi
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/third_party/mmdetection3d:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -u exps/det/CRN_s3_portable.py -b 1 --seed 0 --max_epochs "$EPOCHS" \
  --disable_early_stopping --limit_val_batches 0 --checkpoint_on_train_epoch_end \
  --rolling_terminal_log --default_root_dir outputs/det/CRN_s3_portable "$@"
