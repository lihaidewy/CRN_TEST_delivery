#!/usr/bin/env bash
set -Eeuo pipefail

# CRN_TEST portable environment bootstrapper.
# Usage: bash setup_environment.sh [--env-name=NAME] [--cpu] [--skip-pytorch]

ENV_NAME="CRN_50"
CPU_ONLY=0
SKIP_TORCH=0
for arg in "$@"; do
  case "$arg" in
    --env-name=*) ENV_NAME="${arg#*=}" ;;
    --cpu) CPU_ONLY=1 ;;
    --skip-pytorch) SKIP_TORCH=1 ;;
    -h|--help) sed -n '1,8p' "$0"; exit 0 ;;
    *) echo "未知参数: $arg"; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$ROOT/requirements.txt" ]]; then
  echo "请从 CRN_TEST 项目目录运行此脚本。当前目录: $ROOT" >&2; exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "未找到 conda。请先安装 Miniconda/Anaconda，并在 WSL 终端中运行。" >&2; exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[1/6] 检查系统"
uname -a
python3 --version 2>/dev/null || true
if [[ $CPU_ONLY -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  echo "检测到 NVIDIA GPU："
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  [[ $CPU_ONLY -eq 1 ]] && echo "已选择 CPU 模式（仅适合预处理/导入检查）。" || echo "未检测到 nvidia-smi，将按 CPU 模式处理。"
  CPU_ONLY=1
fi

echo "[2/6] 创建或激活 Conda 环境: $ENV_NAME"
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10 pip cmake ninja
fi
conda activate "$ENV_NAME"
python -m pip install -U "pip<24.1" setuptools

echo "[3/6] 安装 PyTorch（torch 与 torchvision 由同一索引自动配套）"
if [[ $SKIP_TORCH -eq 0 ]]; then
  if [[ $CPU_ONLY -eq 1 ]]; then
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  else
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    echo "使用 PyTorch 索引: $TORCH_INDEX_URL"
    python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
  fi
else
  echo "跳过 PyTorch 安装，使用环境中已有版本。"
fi

echo "[4/6] 安装 CRN 其余依赖"
python -m pip install -r "$ROOT/requirements.txt"

echo "[5/6] 设置当前 shell 的源码路径"
export PYTHONPATH="$ROOT:$ROOT/third_party/mmdetection3d:${PYTHONPATH:-}"
echo "如需永久设置，请将以下内容加入 ~/.bashrc："
echo "export PYTHONPATH=\"$ROOT:$ROOT/third_party/mmdetection3d:\${PYTHONPATH:-}\""

echo "[6/6] 运行安装检查"
if [[ $CPU_ONLY -eq 1 ]]; then
  python - <<'PY'
import torch
print(f'torch={torch.__version__}, cuda_available={torch.cuda.is_available()}')
print('CPU dependency check: PASS')
PY
else
  python "$ROOT/scripts/check_installation.py"
fi
echo "配置完成。后续每次运行前请执行: conda activate $ENV_NAME"
