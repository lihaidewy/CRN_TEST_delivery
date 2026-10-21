"""Centralised paths for the portable S3 workflow.

All defaults are relative to the portable project root.  Set the environment
variables when data, checkpoints or outputs should reside on another disk.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
S3_DATA_ROOT = Path(os.environ.get('CRN_S3_DATA_ROOT', PROJECT_ROOT / 'data' / 's3_myformatted')).resolve()
S3_INPUT_ROOT = Path(os.environ.get('CRN_S3_INPUT_ROOT', PROJECT_ROOT / 'data')).resolve()
S3_OUTPUT_ROOT = Path(os.environ.get('CRN_S3_OUTPUT_ROOT', PROJECT_ROOT / 'outputs')).resolve()
S3_INIT_CKPT = Path(os.environ.get('CRN_S3_INIT_CKPT', PROJECT_ROOT / 'pretrained' / 's2_init.ckpt')).resolve()

S3_FULL_PKL = S3_DATA_ROOT / 'nuscenes_infos_s3_full_v1_all.pkl'
