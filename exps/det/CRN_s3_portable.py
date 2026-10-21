"""Portable S3 CRN training configuration.

This is self-contained with respect to S3 initialisation: it does not inherit
the historical S1/S2 fine-tuning classes that require legacy checkpoints.
An optional compatible S2 checkpoint can be supplied through
``CRN_S3_INIT_CKPT``; when absent the model trains from scratch.
"""

from __future__ import annotations

import os
import sys

import torch
from mmcv.runner import build_optimizer
from torch.optim.lr_scheduler import MultiStepLR

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.paths import S3_DATA_ROOT, S3_FULL_PKL, S3_INIT_CKPT, S3_OUTPUT_ROOT
from exps.base_cli import run_cli
from exps.det.CRN_r50_256x704_128x128_4key_oblique_480 import CRNObliqueModel_480
from models.camera_radar_net_det import CameraRadarNetDet


class CRNS3Portable(CRNObliqueModel_480):
    """Single-camera, camera-radar CRN for the S3 roadside L-shaped scene."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.data_root = str(S3_DATA_ROOT)
        self.train_info_paths = [str(S3_FULL_PKL)]
        self.val_info_paths = str(S3_FULL_PKL)
        self.predict_info_paths = str(S3_FULL_PKL)
        self.default_root_dir = str(S3_OUTPUT_ROOT / 'det' / 'CRN_s3_portable')
        self.batch_size_per_device = 1

        # S3 BEV: x=forward, y=left.  H=200, W=480, 0.4 m per cell.
        x_bound = [0.0, 192.0, 0.4]
        y_bound = [-40.0, 40.0, 0.4]
        z_bound = [-5.0, 3.0, 8.0]
        voxel_size = [0.4, 0.4, 8.0]
        pc_range = [0.0, -40.0, -5.0, 192.0, 40.0, 3.0]

        self.backbone_img_conf.update(x_bound=x_bound, y_bound=y_bound, z_bound=z_bound)
        self.fuser_conf['bev_shape'] = (200, 480)
        self.head_conf['bbox_coder'].update(
            post_center_range=[-10.0, -50.0, -10.0, 202.0, 50.0, 10.0],
            voxel_size=voxel_size,
            pc_range=pc_range,
        )
        self.head_conf['train_cfg'].update(
            point_cloud_range=pc_range, grid_size=[480, 200, 1], voxel_size=voxel_size,
        )
        self.head_conf['test_cfg'].update(
            post_center_limit_range=[-10.0, -50.0, -10.0, 202.0, 50.0, 10.0],
            voxel_size=voxel_size,
        )

        # Keep S3 labels inside the BEV range and retain the distant road in
        # the 512-pixel-tall image crop.
        self.bda_aug_conf.update(rot_ratio=1.0, rot_lim=(-1.0, 1.0), scale_lim=(0.95, 1.03),
                                 flip_dx_ratio=0.0)
        self.ida_aug_conf.update(final_dim=(512, 1408), bot_pct_lim=(0.145, 0.155))
        self.backbone_img_conf['final_dim'] = (512, 1408)

        self.model = CameraRadarNetDet(
            self.backbone_img_conf, self.backbone_pts_conf, self.fuser_conf, self.head_conf,
        )
        self._load_optional_init_checkpoint(S3_INIT_CKPT)
        print('[S3 portable] BEV x=[0,192] m, y=[-40,40] m, 0.4 m cells; input=512x1408.')
        print('[S3 portable] Explicit sparse radar-depth loss is disabled.')

    def _load_optional_init_checkpoint(self, checkpoint_path) -> None:
        """Load compatible non-spatial weights, or intentionally start fresh."""
        if not checkpoint_path.exists():
            print(f'[S3 portable] No transfer checkpoint at {checkpoint_path}; training from scratch.')
            return
        source_state = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        target_state = self.state_dict()
        excluded = {
            'model.backbone_img.voxel_size', 'model.backbone_img.voxel_coord',
            'model.backbone_img.voxel_num', 'model.backbone_img.frustum',
            'model.fuser.ref_2d', 'model.fuser.positional_encoding.row_embed.weight',
            'model.fuser.positional_encoding.col_embed.weight',
        }
        compatible = {
            key: value for key, value in source_state.items()
            if key in target_state and key not in excluded and value.shape == target_state[key].shape
        }
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        unexpected_missing = set(missing) - excluded
        if unexpected or unexpected_missing:
            raise RuntimeError(f'Unexpected transfer mismatch: missing={sorted(unexpected_missing)}, '
                               f'unexpected={unexpected}')
        print(f'[S3 portable] Loaded {len(compatible)} compatible tensors from {checkpoint_path}.')

    def training_step(self, batch, batch_idx=0):
        if self.global_rank == 0:
            for group in self.trainer.optimizers[0].param_groups:
                self.log('learning_rate', group['lr'])
        sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, _, pts_pv = batch
        if torch.cuda.is_available():
            sweep_imgs = sweep_imgs.cuda()
            mats = {key: value.cuda() for key, value in mats.items()}
            pts_pv = pts_pv.cuda()
            gt_boxes_3d = [box.cuda() for box in gt_boxes_3d]
            gt_labels_3d = [label.cuda() for label in gt_labels_3d]
        predictions, _ = self(sweep_imgs, mats, pts_pv=pts_pv, is_train=True)
        targets = self.model.get_targets(gt_boxes_3d, gt_labels_3d)
        loss_detection, loss_heatmap, loss_bbox = self.model.loss(targets, predictions)
        zero = loss_detection.new_zeros(())
        self.log('train/total', loss_detection)
        self.log('train/detection', loss_detection)
        self.log('train/heatmap', loss_heatmap)
        self.log('train/bbox', loss_bbox)
        self.log('train/depth', zero)
        return loss_detection

    def validation_step(self, batch, batch_idx):
        """Validation counterpart of the no-explicit-depth-loss objective."""
        sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, _, pts_pv = batch
        if torch.cuda.is_available():
            sweep_imgs = sweep_imgs.cuda()
            mats = {key: value.cuda() for key, value in mats.items()}
            pts_pv = pts_pv.cuda()
            gt_boxes_3d = [box.cuda() for box in gt_boxes_3d]
            gt_labels_3d = [label.cuda() for label in gt_labels_3d]
        with torch.no_grad():
            predictions, _ = self(sweep_imgs, mats, pts_pv=pts_pv, is_train=True)
            targets = self.model.get_targets(gt_boxes_3d, gt_labels_3d)
            loss_detection, loss_heatmap, loss_bbox = self.model.loss(targets, predictions)
        return loss_detection, loss_heatmap, loss_bbox, loss_detection.new_zeros(())

    def configure_optimizers(self):
        max_epochs = int(getattr(self.trainer, 'max_epochs', 50))
        if max_epochs < 3:
            raise ValueError('max_epochs must be at least 3.')
        first, second = max(1, round(max_epochs * 0.30)), round(max_epochs * 0.70)
        second = min(max(first + 1, second), max_epochs - 1)
        optimizer = build_optimizer(self.model, dict(type='AdamW', lr=5e-5, weight_decay=1e-4))
        print(f'[S3 portable] AdamW lr=5e-5; milestones=[{first}, {second}] for {max_epochs} epochs.')
        return [optimizer], [MultiStepLR(optimizer, milestones=[first, second], gamma=0.1)]


if __name__ == '__main__':
    run_cli(CRNS3Portable, 'det/CRN_s3_portable')
