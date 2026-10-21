#!/usr/bin/env bash
set -Eeuo pipefail

# Run this on a Linux/WSL machine with Internet access.
# Example: bash prepare_offline_bundle.sh /path/to/CRN_TEST /path/to/offline_bundle

PROJECT_ROOT="${1:-$(pwd)}"
OUT="${2:-$PROJECT_ROOT/offline_bundle}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu117}"
TORCH_VERSION="${TORCH_VERSION:-2.0.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.15.2}"
MMCV_WHEEL_INDEX="${MMCV_WHEEL_INDEX:-https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html}"

[[ -f "$PROJECT_ROOT/requirements.txt" ]] || { echo "找不到 $PROJECT_ROOT/requirements.txt" >&2; exit 1; }
command -v python >/dev/null || { echo "需要 Python 3.10 环境" >&2; exit 1; }
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f'需要 Python 3.10，当前为 {sys.version_info[:2]}')
PY

mkdir -p "$OUT/wheels" "$OUT/conda" "$OUT/logs"
cp "$PROJECT_ROOT/requirements.txt" "$OUT/requirements.txt"
cp "$PROJECT_ROOT/environment.yml" "$OUT/environment.yml"
cp "$PROJECT_ROOT/setup.py" "$OUT/setup.py"

echo '[0/4] 下载 Conda 基础包'
if command -v conda >/dev/null 2>&1; then
  mkdir -p "$OUT/conda_pkgs"
  CONDA_PKGS_DIRS="$OUT/conda_pkgs" conda create --download-only -n "crn_offline_seed_$$" \
    python=3.10 pip cmake ninja -y 2>&1 | tee "$OUT/logs/conda_download.log"
else
  echo '未找到 conda，跳过 Conda 包下载；目标机需要自行准备 Python/Conda。' >&2
fi

echo '[1/4] 下载 PyTorch 与 torchvision'
python -m pip download --dest "$OUT/wheels" --only-binary=:all: \
  --index-url "$TORCH_INDEX_URL" \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  2>&1 | tee "$OUT/logs/torch_download.log"

# Lightning 1.6.0 uses an old dependency specifier and needs pip<24.1.
python -m pip download --dest "$OUT/wheels" --only-binary=:all: \
  --index-url https://pypi.org/simple pip==24.0
python -m pip download --dest "$OUT/wheels" --only-binary=:all: \
  --index-url https://pypi.org/simple setuptools==60.2.0

echo '[2/4] 下载项目 Python 依赖'
# requirements.txt 中的 torch 由上一步提供；约束文件防止 Lightning 重新选择其他 torch。
CONSTRAINTS="$(mktemp)"
printf 'torch==%s\ntorchvision==%s\n' "$TORCH_VERSION" "$TORCHVISION_VERSION" > "$CONSTRAINTS"
python -m pip download --dest "$OUT/wheels" -r "$PROJECT_ROOT/requirements.txt" \
  --constraint "$CONSTRAINTS" \
  --index-url https://pypi.org/simple --extra-index-url "$TORCH_INDEX_URL" \
  --extra-index-url "$TORCH_INDEX_URL" \
  --find-links "$OUT/wheels" \
  --find-links "$MMCV_WHEEL_INDEX" \
  2>&1 | tee "$OUT/logs/python_download.log" || {
  rm -f "$CONSTRAINTS"
  echo '依赖下载失败。若失败项是 mmcv-full，请保留源码并在目标机编译。' >&2
  exit 1
}
rm -f "$CONSTRAINTS"

# 避免 PyPI 下载的 CPU/nightly torch 覆盖指定 CUDA wheel。
find "$OUT/wheels" -maxdepth 1 -type f \
  \( -iname 'torch-*.whl' -o -iname 'torchvision-*.whl' \) \
  ! -name "torch-${TORCH_VERSION}+*.whl" \
  ! -name "torchvision-${TORCHVISION_VERSION}+*.whl" -delete

echo '[3/4] 记录目标环境信息'
cat > "$OUT/MANIFEST.txt" <<EOF
CRN_TEST offline bundle
created=$(date -Iseconds)
python=3.10
torch=$TORCH_VERSION
torchvision=$TORCHVISION_VERSION
torch_index=$TORCH_INDEX_URL
注意：mmcv-full 和 CRN 自定义 CUDA 算子可能需要在目标机重新编译。
EOF
(cd "$OUT" && find wheels -maxdepth 1 -type f -printf '%f %s bytes\n' | sort) >> "$OUT/MANIFEST.txt"

echo '[4/4] 完成'
echo "离线包目录：$OUT"
echo '请将该目录和 CRN_TEST 项目一起拷贝到目标电脑。'
