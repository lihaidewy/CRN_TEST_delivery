#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/third_party/mmdetection3d:${PYTHONPATH:-}"
python -u scripts/data/build_s3_full_data.py
