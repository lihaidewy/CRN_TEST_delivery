#!/usr/bin/env python3
"""Verify prepared scene ingestion and optionally one CUDA model forward."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scene_config import load_scene_config
from exps.det.CRN_scene_dynamic import CRNDynamicScene, SCENE_CONFIG_ENV


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--forward', action='store_true', help='Also run one CUDA joint forward.')
    args = parser.parse_args()

    scene = load_scene_config(args.config)
    scene.validate_prepared()
    os.environ[SCENE_CONFIG_ENV] = str(scene.path)
    model = CRNDynamicScene(batch_size_per_device=1)
    batch = next(iter(model.train_dataloader()))
    sweep_imgs, mats, _, gt_boxes, gt_labels, _, depth_labels, pts_pv = batch
    print('[PASS] configured dataloader')
    print('  images:', tuple(sweep_imgs.shape))
    print('  radar PV:', tuple(pts_pv.shape))
    print('  depth labels:', tuple(depth_labels.shape))
    print('  GT boxes:', tuple(gt_boxes[0].shape), 'labels:', tuple(gt_labels[0].shape))
    print('  matrix keys:', sorted(mats))
    if not args.forward:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('--forward requires an available CUDA GPU.')

    model.model.cuda().eval()
    sweep_imgs = sweep_imgs.cuda().half()
    pts_pv = pts_pv.cuda().float()
    mats = {key: value.cuda().float() for key, value in mats.items()}
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        predictions, depth_logits = model.model(
            sweep_imgs, mats, sweep_ptss=pts_pv, is_train=True)
    print('[PASS] configured camera-radar joint forward')
    print('  depth logits:', tuple(depth_logits.shape))
    for task_index, task_levels in enumerate(predictions):
        for level_index, heads in enumerate(task_levels):
            shapes = ', '.join(f'{name}={tuple(value.shape)}' for name, value in heads.items())
            print(f'  task {task_index}, level {level_index}: {shapes}')


if __name__ == '__main__':
    main()
