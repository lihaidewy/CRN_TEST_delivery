"""Configuration-driven CRN experiment for a static roadside scene.

Set ``CRN_SCENE_CONFIG`` to a validated scene JSON before invoking this file.
The higher-level ``scripts/crn_scene.py`` command does that automatically.
"""

from __future__ import annotations

import os
import math
import sys
import csv
from pathlib import Path

import torch
from mmcv.runner import build_optimizer
from torch.optim.lr_scheduler import MultiStepLR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# CRNS3Portable is used only as a known-good architecture template.  Its own
# optional S3 checkpoint must not be loaded before dynamic geometry is applied.
os.environ['CRN_S3_INIT_CKPT'] = str(PROJECT_ROOT / 'pretrained' / '__dynamic_deferred__.ckpt')

from config.scene_config import load_scene_config
from exps.base_cli import run_cli
from exps.det.CRN_s3_portable import CRNS3Portable
from models.camera_radar_net_det import CameraRadarNetDet
from layers.refiner.minimal_far_range_bev_refiner import MinimalFarRangeBEVRefiner


SCENE_CONFIG_ENV = 'CRN_SCENE_CONFIG'


class CRNDynamicScene(CRNS3Portable):
    """CRN whose spatial geometry and paths come from one scene config."""

    def __init__(self, *args, **kwargs) -> None:
        config_path = os.environ.get(SCENE_CONFIG_ENV)
        if not config_path:
            raise RuntimeError(
                f'{SCENE_CONFIG_ENV} is not set. Use `python scripts/crn_scene.py train --config ...`.')
        self.scene = load_scene_config(config_path)
        self.scene.validate_prepared()
        requested_batch = kwargs.get('batch_size_per_device')

        # This creates the known-good CRN architecture first.  We then replace
        # every scene-dependent geometry value and rebuild exactly once.
        super().__init__(*args, **kwargs)
        cfg, derived = self.scene.raw, self.scene.derived
        training = cfg['training']

        self.data_root = str(self.scene.data_root)
        self.train_info_paths = [str(self.scene.info_path_for('train'))]
        self.val_info_paths = [str(self.scene.info_path_for('val'))]
        self.predict_info_paths = [str(self.scene.info_path_for('test'))]
        self.default_root_dir = str(self.scene.output_root / 'training')
        self.batch_size_per_device = int(requested_batch or training['batch_size'])
        self.num_workers = int(training['num_workers'])
        self.max_radar_points_pv = int(cfg['model']['max_radar_points_pv'])
        radar_signal = self.scene.radar_signal_parameters(require_auto=True)
        self.radar_signal_type = str(radar_signal['type'])
        self.radar_signal_normalization = str(radar_signal['normalization'])
        self.radar_signal_mean = float(radar_signal['mean'])
        self.radar_signal_std = float(radar_signal['std'])
        self.radar_signal_count = radar_signal['count']

        camera = cfg['calibration']['camera']
        final_h, final_w = derived['final_dim']
        ida = cfg['model']['image_augmentation']
        bda = cfg['model']['bev_augmentation']
        self.ida_aug_conf.update(
            resize_lim=tuple(float(v) for v in ida['resize_lim']),
            final_dim=(final_h, final_w),
            rot_lim=tuple(float(v) for v in ida['rotation_deg']),
            H=int(camera['image_height']),
            W=int(camera['image_width']),
            rand_flip=bool(ida['random_flip']),
            bot_pct_lim=tuple(float(v) for v in ida['bottom_crop_fraction']),
            cams=['CAM_FRONT'],
            Ncams=1,
        )
        self.bda_aug_conf.update(
            rot_ratio=float(bda['rotation_probability']),
            rot_lim=tuple(float(v) for v in bda['rotation_deg']),
            scale_lim=tuple(float(v) for v in bda['scale']),
            flip_dx_ratio=float(bda['flip_x_probability']),
            flip_dy_ratio=float(bda['flip_y_probability']),
        )

        self.backbone_img_conf.update(
            x_bound=derived['x_bound'],
            y_bound=derived['y_bound'],
            z_bound=derived['z_bound'],
            d_bound=derived['d_bound'],
            final_dim=(final_h, final_w),
        )

        pv_voxel_size = derived['pv_voxel_size']
        pv_range = derived['pv_point_cloud_range']
        self.backbone_pts_conf['pts_voxel_layer'].update(
            voxel_size=pv_voxel_size,
            point_cloud_range=pv_range,
        )
        self.backbone_pts_conf['pts_voxel_encoder'].update(
            voxel_size=pv_voxel_size,
            point_cloud_range=pv_range,
        )
        self.backbone_pts_conf['pts_middle_encoder']['output_shape'] = tuple(
            derived['pv_output_shape'])
        self.fuser_conf['bev_shape'] = tuple(derived['bev_shape'])

        margin = float(cfg['model'].get('post_center_margin_m', 10.0))
        x_min, y_min, z_min, x_max, y_max, z_max = derived['pc_range']
        post_range = [
            x_min - margin, y_min - margin, z_min - margin,
            x_max + margin, y_max + margin, z_max + margin,
        ]
        self.head_conf['bbox_coder'].update(
            post_center_range=post_range,
            voxel_size=derived['voxel_size'],
            pc_range=derived['pc_range'],
        )
        self.head_conf['train_cfg'].update(
            point_cloud_range=derived['pc_range'],
            grid_size=derived['grid_size'],
            voxel_size=derived['voxel_size'],
        )
        self.head_conf['test_cfg'].update(
            post_center_limit_range=post_range,
            voxel_size=derived['voxel_size'],
        )

        self.dbound = self.backbone_img_conf['d_bound']
        self.depth_channels = derived['depth_bins']
        self.model = CameraRadarNetDet(
            self.backbone_img_conf,
            self.backbone_pts_conf,
            self.fuser_conf,
            self.head_conf,
        )

        init_checkpoint = self.scene.init_checkpoint
        step8b_init = os.environ.get('CRN_STEP8B_INIT_CKPT')
        if step8b_init:
            init_checkpoint = Path(step8b_init).expanduser().resolve()
            print(f'[Step8B] initialization checkpoint override: {init_checkpoint}')
        if os.environ.get('CRN_SKIP_SCENE_INIT', '0') == '1':
            print('[Dynamic scene] Initialization checkpoint loading is deferred to Lightning resume/evaluation.')
        elif init_checkpoint is None:
            print(f'[Dynamic scene] {self.scene.scene_id}: training from scratch.')
        else:
            if not init_checkpoint.is_file():
                raise FileNotFoundError(
                    f'Configured initialization checkpoint does not exist: {init_checkpoint}')
            self._load_optional_init_checkpoint(init_checkpoint)

        self.step8b_enabled = os.environ.get('CRN_STEP8B', '0') == '1'
        if self.step8b_enabled:
            cap = float(os.environ.get('CRN_STEP8B_ALPHA_CAP', '0.1'))
            self.model.bev_refiner = MinimalFarRangeBEVRefiner(
                channels=128, x_min=derived['x_bound'][0], x_max=derived['x_bound'][1],
                x_threshold=100.0, alpha_cap=cap)
            self.apply_step8b_freeze_contract()
            print(f'[Step8B] enabled: x>=100m, alpha_cap={cap:g}; original CRN frozen.')

        self.dynamic_learning_rate = (1e-3 if os.environ.get('CRN_STEP8B', '0') == '1'
                                      else float(training['learning_rate']))
        self.dynamic_weight_decay = float(training['weight_decay'])
        self.dynamic_milestone_ratios = tuple(float(v) for v in training['lr_milestone_ratios'])
        self.dynamic_lr_gamma = float(training['lr_gamma'])
        self.step8b_audit_enabled = os.environ.get('CRN_STEP8B_AUDIT', '0') == '1'
        self._step8b_audit_file = None
        print(
            f'[Dynamic scene] scene={self.scene.scene_id}; image={final_h}x{final_w}; '
            f'BEV={derived["bev_shape"][0]}x{derived["bev_shape"][1]}; '
            f'depth_bins={derived["depth_bins"]}; '
            f'radar_signal={self.radar_signal_type} '
            f'(mode={self.radar_signal_normalization}, mean={self.radar_signal_mean:.6g}, '
            f'std={self.radar_signal_std:.6g}, count={self.radar_signal_count}).')

    def apply_step8b_freeze_contract(self):
        """Make the Step8B single-variable training contract explicit.

        The original CRN is frozen both at the parameter level and at the
        module-mode level.  This method is deliberately idempotent because
        Lightning/PyTorch may call ``train()`` again at epoch boundaries.
        """
        if not hasattr(self.model, 'bev_refiner'):
            raise RuntimeError('Step8B freeze contract requires model.bev_refiner.')
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.bev_refiner.parameters():
            p.requires_grad = True
        # Keep all original CRN submodules in inference mode.  This freezes
        # BatchNorm running statistics and stochastic layers as well as grads.
        for name, module in self.model.named_modules():
            if name == 'bev_refiner' or name.startswith('bev_refiner.'):
                continue
            module.eval()
        self.model.bev_refiner.train(True)
        return [p for p in self.model.parameters() if p.requires_grad]

    def train(self, mode: bool = True):
        """Persist Step8B frozen-module eval policy after recursive train()."""
        result = super().train(mode)
        if mode and getattr(self, 'step8b_enabled', False):
            self.apply_step8b_freeze_contract()
        return result

    def on_train_start(self):
        super().on_train_start()
        if not self.step8b_audit_enabled or not hasattr(self.model, 'bev_refiner'):
            return
        audit_root = os.environ.get('CRN_STEP8B_AUDIT_OUTPUT', self.default_root_dir)
        path = Path(audit_root).expanduser().resolve() / 'step8b_training_dynamics.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        self._step8b_audit_file = path.open('w', newline='', encoding='utf-8')
        writer = csv.writer(self._step8b_audit_file)
        writer.writerow([
            'epoch', 'global_step', 'alpha_raw', 'alpha_eff',
            'alpha_grad_norm', 'conv1_grad_norm', 'norm_grad_norm', 'conv2_grad_norm',
            'conv1_param_norm', 'norm_param_norm', 'conv2_param_norm'])
        self._step8b_audit_file.flush()

    def on_after_backward(self):
        super().on_after_backward()
        if self._step8b_audit_file is None or not hasattr(self.model, 'bev_refiner'):
            return
        refiner = self.model.bev_refiner
        def grad_norm(parameter):
            return float(parameter.grad.detach().norm().cpu()) if parameter.grad is not None else 0.0
        alpha_raw = float(refiner.alpha_raw.detach().cpu())
        alpha_eff = float(refiner.alpha_cap * math.tanh(alpha_raw))
        writer = csv.writer(self._step8b_audit_file)
        writer.writerow([
            int(self.current_epoch), int(self.global_step), alpha_raw, alpha_eff,
            grad_norm(refiner.alpha_raw), grad_norm(refiner.conv1.weight),
            grad_norm(refiner.norm.weight), grad_norm(refiner.conv2.weight),
            float(refiner.conv1.weight.detach().norm().cpu()),
            float(refiner.norm.weight.detach().norm().cpu()),
            float(refiner.conv2.weight.detach().norm().cpu()),
        ])
        self._step8b_audit_file.flush()

    def on_train_end(self):
        if self._step8b_audit_file is not None:
            self._step8b_audit_file.close()
            self._step8b_audit_file = None
        super().on_train_end()

    def _radar_signal_checkpoint_metadata(self) -> dict:
        return {
            'type': self.radar_signal_type,
            'normalization': self.radar_signal_normalization,
            'mean': self.radar_signal_mean,
            'std': self.radar_signal_std,
        }

    def validate_checkpoint_radar_signal(self, checkpoint: dict, purpose: str) -> None:
        saved = checkpoint.get('crn_radar_signal')
        if saved is None:
            print(
                f'[Dynamic scene] WARNING: {purpose} checkpoint predates radar-signal metadata; '
                'compatibility cannot be verified. This is allowed only for legacy/model initialization.')
            return
        current = self._radar_signal_checkpoint_metadata()
        if saved.get('type') != current['type']:
            raise ValueError(
                f'{purpose} checkpoint radar feature={saved.get("type")!r}, '
                f'but scene feature={current["type"]!r}.')
        for key in ('mean', 'std'):
            if not math.isclose(float(saved[key]), float(current[key]), rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f'{purpose} checkpoint radar {key}={saved[key]}, but scene uses '
                    f'{current[key]}. Use the training statistics stored with that checkpoint.')

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint['crn_radar_signal'] = self._radar_signal_checkpoint_metadata()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.validate_checkpoint_radar_signal(checkpoint, purpose='resume')

    def configure_optimizers(self):
        max_epochs = int(getattr(self.trainer, 'max_epochs', self.scene.raw['training']['epochs']))
        if max_epochs < 3:
            raise ValueError('max_epochs must be at least 3.')
        first_ratio, second_ratio = self.dynamic_milestone_ratios
        first = max(1, round(max_epochs * first_ratio))
        second = min(max(first + 1, round(max_epochs * second_ratio)), max_epochs - 1)
        if getattr(self, 'step8b_enabled', False):
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable or any(not p.requires_grad for p in trainable):
                raise RuntimeError('Step8B optimizer received an invalid trainable parameter list.')
            optimizer = torch.optim.AdamW(
                trainable, lr=self.dynamic_learning_rate,
                weight_decay=self.dynamic_weight_decay)
            print(f'[Step8B] optimizer tensors={len(trainable)} (refiner-only).')
        else:
            optimizer = build_optimizer(self.model, {
                'type': 'AdamW',
                'lr': self.dynamic_learning_rate,
                'weight_decay': self.dynamic_weight_decay,
            })
        print(
            f'[Dynamic scene] AdamW lr={self.dynamic_learning_rate:g}, '
            f'weight_decay={self.dynamic_weight_decay:g}; milestones=[{first}, {second}], '
            f'gamma={self.dynamic_lr_gamma:g}.')
        scheduler = MultiStepLR(
            optimizer, milestones=[first, second], gamma=self.dynamic_lr_gamma)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    run_cli(CRNDynamicScene, 'det/CRN_scene_dynamic')
