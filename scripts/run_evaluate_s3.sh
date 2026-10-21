#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_evaluate_s3.sh <checkpoint.ckpt> <S3M|S3S> [extra eval args]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$1"
SUBSET="$2"
shift 2
case "$SUBSET" in
  S3M) OUT="outputs/evaluation/S3M_moderate" ;;
  S3S) OUT="outputs/evaluation/S3S_heavy" ;;
  *) echo "Subset must be S3M or S3S." >&2; exit 2 ;;
esac
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/third_party/mmdetection3d:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -u scripts/evaluation/eval_s3.py --full_s3 --subset "$SUBSET" --ckpt "$CKPT" \
  --out_dir "$OUT" --score_thr "${CRN_SCORE_THR:-0.35}" --match_dist 2.0 "$@"
