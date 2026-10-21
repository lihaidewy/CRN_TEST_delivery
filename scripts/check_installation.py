#!/usr/bin/env python3
"""Fail early with actionable messages when the portable runtime is incomplete."""

from __future__ import annotations

import importlib
import sys


def require(module: str) -> None:
    try:
        importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - diagnostic utility
        raise RuntimeError(f'Cannot import {module}: {exc}') from exc


def main() -> None:
    require('torch')
    import torch
    print(f'torch={torch.__version__}, cuda_available={torch.cuda.is_available()}')
    if not torch.cuda.is_available():
        raise RuntimeError('CRN training/evaluation requires CUDA-enabled PyTorch and an available NVIDIA GPU.')
    require('mmcv')
    require('mmdet')
    require('mmdet3d')
    require('ops.average_voxel_pooling_v2.average_voxel_pooling_ext')
    require('ops.voxel_pooling_v2.voxel_pooling_ext')
    print('CRN S3 portable installation: PASS')


if __name__ == '__main__':
    main()
