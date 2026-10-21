#!/usr/bin/env bash
set -Eeuo pipefail

# Run this on the offline target machine from the CRN_TEST project root.
BUNDLE="${1:-$(pwd)/offline_bundle}"
ENV_NAME="${ENV_NAME:-CRN_50}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -d "$BUNDLE/wheels" ]] || { echo "找不到离线 wheel 目录: $BUNDLE/wheels" >&2; exit 1; }
[[ -d "$BUNDLE/conda_pkgs" ]] || { echo "找不到离线 Conda 包目录: $BUNDLE/conda_pkgs" >&2; exit 1; }
command -v conda >/dev/null || { echo '未找到 conda' >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  CONDA_PKGS_DIRS="$BUNDLE/conda_pkgs" conda create --offline -y -n "$ENV_NAME" python=3.10 pip cmake ninja
fi
conda activate "$ENV_NAME"
python -m pip install --no-index --find-links="$BUNDLE/wheels" pip==24.0
python -m pip install --no-index --find-links="$BUNDLE/wheels" setuptools==60.2.0

echo '[1/3] 安装 PyTorch wheel'
python -m pip install --no-index --find-links="$BUNDLE/wheels" torch torchvision

echo '[2/3] 安装其余 Python 依赖'
# 先固定 NumPy，避免 mmcv 编译阶段误使用 torchvision 拉入的 NumPy 2.x。
python -m pip install --no-index --find-links="$BUNDLE/wheels" \
  numpy==1.26.4 Pillow==12.1.1
# Build MMCV against the already-installed PyTorch/CUDA environment. Disabling
# build isolation is important offline: an isolated PEP 517 build may try to
# fetch build requirements even though the project wheelhouse is complete.
python -m pip install --no-index --find-links="$BUNDLE/wheels" \
  --no-build-isolation mmcv-full==1.7.0
python -m pip install --no-index --find-links="$BUNDLE/wheels" -r "$PROJECT_ROOT/requirements.txt"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/mmdetection3d:${PYTHONPATH:-}"
echo '[3/3] 检查项目依赖'
if ! python -c "from ops.average_voxel_pooling_v2 import average_voxel_pooling_ext; from ops.voxel_pooling_v2 import voxel_pooling_ext"; then
  echo '自定义算子不可用，开始在目标机编译。'
  python "$PROJECT_ROOT/setup.py" build_ext --inplace
fi
python "$PROJECT_ROOT/scripts/check_installation.py"
echo '离线环境配置完成。'
