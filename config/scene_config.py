"""Validated, configuration-driven scene description for CRN.

One JSON file is the single source of truth for preprocessing, model geometry,
training and evaluation.  Paths in the JSON are resolved relative to the JSON
file, not to the caller's current working directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


AXIS_FORWARD_LEFT_UP_TO_RIGHT_DOWN_FORWARD = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

SUPPORTED_CAMERA_CONVENTIONS = {
    'world_forward_left_up__camera_forward_left_up',
    'world_forward_left_up__camera_right_down_forward',
}


def radar_signal_config_fingerprint(config: dict) -> str:
    """Fingerprint inputs that determine which radar signals reach PV."""
    calibration = copy.deepcopy(config['calibration'])
    signal = calibration['radar'].get('signal_feature', {})
    calibration['radar']['signal_feature'] = {
        'type': signal.get('type', 'rcs'),
        'normalization': signal.get('normalization', 'manual'),
    }
    payload = {
        'scene_id': config['scene_id'],
        'inputs': config['inputs'],
        'calibration': calibration,
        'dataset_split': config.get('dataset_split', {'strategy': 'none'}),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _required(mapping: dict, key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f'Missing `{context}.{key}` in scene config.')
    return mapping[key]


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f'{name} must be a finite positive number, got {value!r}.')
    return number


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'{name} must contain exactly two numbers.')
    return float(value[0]), float(value[1])


def _range(value: Any, name: str, length: int = 3) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{name} must contain exactly {length} numbers.')
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f'{name} contains a non-finite number.')
    if result[1] <= result[0]:
        raise ValueError(f'{name} maximum must be greater than its minimum.')
    if length == 3 and result[2] <= 0:
        raise ValueError(f'{name} step must be positive.')
    return result


def _matrix4(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f'{name} must be a 4x4 numeric matrix, got {matrix.shape}.')
    if not np.isfinite(matrix).all():
        raise ValueError(f'{name} contains a non-finite value.')
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f'{name} final row must be [0, 0, 0, 1].')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f'{name} rotation block is not orthonormal.')
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise ValueError(f'{name} rotation determinant must be +1.')
    return matrix


def _integer_cells(span: float, cell: float, name: str) -> int:
    raw = span / cell
    cells = int(round(raw))
    if not math.isclose(raw, cells, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f'{name} span {span:g} m is not an integer multiple of cell size {cell:g} m.')
    return cells


class SceneConfig:
    """Loaded and validated scene configuration."""

    def __init__(self, raw: dict, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.base_dir = self.path.parent
        self.raw = copy.deepcopy(raw)
        self._apply_defaults()
        self._validate_static()

    @classmethod
    def load(cls, path: str | Path) -> 'SceneConfig':
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f'Scene config does not exist: {resolved}')
        with resolved.open(encoding='utf-8') as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError('Scene config root must be a JSON object.')
        return cls(raw, resolved)

    def _apply_defaults(self) -> None:
        cfg = self.raw
        cfg.setdefault('schema_version', 2)
        cfg.setdefault('model', {})
        cfg.setdefault('training', {})
        cfg.setdefault('evaluation', {})
        cfg.setdefault('annotations', {})
        split = cfg.setdefault('dataset_split', {})
        split.setdefault('strategy', 'none')
        if split['strategy'] == 'chronological':
            split.setdefault('train_ratio', 0.7)
            split.setdefault('val_ratio', 0.2)
            split.setdefault('test_ratio', 0.1)

        inputs = cfg.setdefault('inputs', {})
        inputs.setdefault('radar_coordinate_mode', 'without_xy')
        inputs.setdefault('radar_xy_y_positive', 'left')
        inputs.setdefault('image_filename_template', '{frame}.png')
        inputs.setdefault('annotation_filename_template', '{frame}.json')
        inputs.setdefault('frame_id_offset', 0)
        inputs.setdefault('radar_columns', {
            'frame': 'Frame', 'range': 'Range', 'azimuth': 'Angle',
            'radial_speed': 'Speed', 'snr': 'SNR',
        })
        inputs['radar_columns'].setdefault('x', 'X')
        inputs['radar_columns'].setdefault('y', 'Y')

        output = cfg.setdefault('output', {})
        output.setdefault('info_pkl_name', f'nuscenes_infos_{cfg.get("scene_id", "scene")}_all.pkl')
        output.setdefault('output_root', f'../outputs/{cfg.get("scene_id", "scene")}')
        stem = Path(output['info_pkl_name']).stem
        prefix = stem[:-4] if stem.endswith('_all') else stem
        output.setdefault('info_pkl_names', {
            name: f'{prefix}_{name}.pkl' for name in ('train', 'val', 'test')
        })

        radar = cfg.setdefault('calibration', {}).setdefault('radar', {})
        radar.setdefault('range_type', 'slant')
        radar.setdefault('angle_type', 'horizontal_azimuth')
        radar.setdefault('angle_unit', 'degree')
        radar.setdefault('angle_positive', 'left')
        radar.setdefault('speed_type', 'radial')
        radar.setdefault('return_height_above_ground_m', 0.0)
        signal_feature = radar.setdefault('signal_feature', {})
        # Legacy configs used fixed RCS statistics. New SNR scenes should use
        # auto_train, whose statistics are computed from prepared training PV
        # points and then frozen for training/evaluation.
        signal_feature.setdefault('type', 'rcs')
        signal_feature.setdefault('normalization', 'manual')
        signal_feature.setdefault('stats_filename', 'radar_signal_stats.json')
        if signal_feature['normalization'] == 'manual':
            signal_feature.setdefault('mean', 4.783)
            signal_feature.setdefault('std', 7.576)

        cfg['annotations'].setdefault('category_name', 'vehicle.car')
        cfg['annotations'].setdefault('size_order', 'length_width_height')
        cfg['annotations'].setdefault('translation_is_ground_contact', True)

        model = cfg['model']
        model.setdefault('input_size_hw', [512, 1408])
        model.setdefault('depth_bound_m', [2.0, 242.0, 2.0])
        model.setdefault('z_bound_m', [-5.0, 3.0])
        model.setdefault('max_radar_points_pv', 1536)
        model.setdefault('image_augmentation', {})
        model.setdefault('bev_augmentation', {})
        ida = model['image_augmentation']
        ida.setdefault('rotation_deg', [0.0, 0.0])
        ida.setdefault('bottom_crop_fraction', [0.145, 0.155])
        ida.setdefault('random_flip', True)
        bda = model['bev_augmentation']
        bda.setdefault('rotation_probability', 1.0)
        bda.setdefault('rotation_deg', [-1.0, 1.0])
        bda.setdefault('scale', [0.95, 1.03])
        bda.setdefault('flip_x_probability', 0.0)
        bda.setdefault('flip_y_probability', 0.0)

        training = cfg['training']
        training.setdefault('epochs', 100)
        training.setdefault('batch_size', 1)
        training.setdefault('seed', 0)
        training.setdefault('learning_rate', 5e-5)
        training.setdefault('weight_decay', 1e-4)
        training.setdefault('lr_milestone_ratios', [0.30, 0.70])
        training.setdefault('lr_gamma', 0.1)
        training.setdefault('num_workers', 0)
        training.setdefault('checkpoint_every_n_epochs', 8)
        training.setdefault('init_checkpoint', None)
        training.setdefault('validate_during_training', False)
        training.setdefault('validation_every_n_epochs', 5)

        evaluation = cfg['evaluation']
        evaluation.setdefault('score_threshold', 0.35)
        evaluation.setdefault('match_distance_m', 2.0)
        evaluation.setdefault('max_frames', -1)
        evaluation.setdefault('visualizations', True)
        evaluation.setdefault('distance_bins_m', [0.0, 50.0, 100.0, 150.0, 200.0, 240.0])

    @property
    def scene_id(self) -> str:
        return str(self.raw['scene_id'])

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.base_dir / path).resolve()

    @property
    def data_root(self) -> Path:
        return self.resolve(self.raw['output']['data_root'])

    @property
    def info_path(self) -> Path:
        return self.info_path_for('train')

    @property
    def has_dataset_split(self) -> bool:
        return self.raw['dataset_split']['strategy'] != 'none'

    def info_path_for(self, split: str) -> Path:
        if split not in ('train', 'val', 'test', 'all'):
            raise ValueError(f'Unknown dataset split {split!r}.')
        output = self.raw['output']
        filename = (output['info_pkl_name'] if split == 'all' else
                    (output['info_pkl_names'][split]
                     if self.has_dataset_split else output['info_pkl_name']))
        return self.data_root / filename

    @property
    def radar_signal_stats_path(self) -> Path:
        filename = self.raw['calibration']['radar']['signal_feature']['stats_filename']
        return self.data_root / filename

    def radar_signal_parameters(self, require_auto: bool = False) -> dict[str, Any]:
        """Return the effective, immutable signal normalization parameters."""
        signal = self.raw['calibration']['radar']['signal_feature']
        mode = signal['normalization']
        if mode == 'manual':
            return {
                'type': signal['type'], 'normalization': mode,
                'mean': float(signal['mean']), 'std': float(signal['std']),
                'count': None, 'source': 'scene_config',
            }
        path = self.radar_signal_stats_path
        if not path.is_file():
            if not require_auto:
                return {
                    'type': signal['type'], 'normalization': mode,
                    'mean': None, 'std': None, 'count': None,
                    'source': str(path),
                }
            raise FileNotFoundError(
                f'Automatic radar-signal statistics do not exist: {path}. '
                f'Run `python scripts/crn_scene.py prepare --config {self.path}` first.')
        with path.open(encoding='utf-8') as handle:
            stats = json.load(handle)
        expected_fingerprint = radar_signal_config_fingerprint(self.raw)
        checks = {
            'scene_id': self.scene_id,
            'feature_type': signal['type'],
            'config_fingerprint': expected_fingerprint,
        }
        for key, expected in checks.items():
            if stats.get(key) != expected:
                raise ValueError(
                    f'Stale/incompatible radar statistics in {path}: {key}='
                    f'{stats.get(key)!r}, expected {expected!r}. Re-run prepare.')
        count = int(stats.get('count', 0))
        mean = float(stats.get('mean', float('nan')))
        std = float(stats.get('std', float('nan')))
        if count < 2 or not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f'Invalid automatic radar statistics in {path}. Re-run prepare.')
        return {
            'type': signal['type'], 'normalization': mode,
            'mean': mean, 'std': std, 'count': count,
            'source': str(path),
        }

    @property
    def output_root(self) -> Path:
        return self.resolve(self.raw['output']['output_root'])

    @property
    def init_checkpoint(self) -> Path | None:
        value = self.raw['training'].get('init_checkpoint')
        return self.resolve(value) if value else None

    @property
    def camera_matrix_raw(self) -> np.ndarray:
        return _matrix4(
            self.raw['calibration']['camera']['world_to_camera_4x4'],
            'calibration.camera.world_to_camera_4x4',
        )

    @property
    def world_to_standard_camera(self) -> np.ndarray:
        calibration = self.raw['calibration']
        normalized_to_raw = np.eye(4, dtype=np.float64)
        normalized_to_raw[2, 3] = float(calibration['ground_z_world_m'])
        transform = self.camera_matrix_raw @ normalized_to_raw
        if calibration['coordinate_convention'].endswith('camera_forward_left_up'):
            transform = AXIS_FORWARD_LEFT_UP_TO_RIGHT_DOWN_FORWARD @ transform
        return transform

    @property
    def cell_size_xy(self) -> tuple[float, float]:
        value = self.raw['bev']['cell_size_m']
        if isinstance(value, (list, tuple)):
            return _positive(value[0], 'bev.cell_size_m[0]'), _positive(value[1], 'bev.cell_size_m[1]')
        cell = _positive(value, 'bev.cell_size_m')
        return cell, cell

    @property
    def derived(self) -> dict[str, Any]:
        bev = self.raw['bev']
        x_min, x_max = _pair(bev['x_bound_m'], 'bev.x_bound_m')
        y_min, y_max = _pair(bev['y_bound_m'], 'bev.y_bound_m')
        cell_x, cell_y = self.cell_size_xy
        nx = _integer_cells(x_max - x_min, cell_x, 'BEV x')
        ny = _integer_cells(y_max - y_min, cell_y, 'BEV y')
        z_min, z_max = _pair(self.raw['model']['z_bound_m'], 'model.z_bound_m')
        d_min, d_max, d_step = _range(self.raw['model']['depth_bound_m'], 'model.depth_bound_m')
        depth_bins = _integer_cells(d_max - d_min, d_step, 'Depth')
        final_h, final_w = (int(item) for item in self.raw['model']['input_size_hw'])
        return {
            'x_bound': [x_min, x_max, cell_x],
            'y_bound': [y_min, y_max, cell_y],
            'z_bound': [z_min, z_max, z_max - z_min],
            'd_bound': [d_min, d_max, d_step],
            'pc_range': [x_min, y_min, z_min, x_max, y_max, z_max],
            'voxel_size': [cell_x, cell_y, z_max - z_min],
            'grid_size': [nx, ny, 1],
            'bev_shape': [ny, nx],
            'depth_bins': depth_bins,
            'final_dim': [final_h, final_w],
            # PointPillars operates on the PV plane before its fixed 2x
            # reduction, hence half-depth steps and 8 source pixels.
            'pv_voxel_size': [8.0, d_step / 2.0, 2.0],
            'pv_point_cloud_range': [0.0, d_min, 0.0, float(final_w), d_max, 2.0],
            'pv_output_shape': [depth_bins * 2, final_w // 8],
        }

    def _validate_static(self) -> None:
        cfg = self.raw
        if int(cfg['schema_version']) not in (1, 2):
            raise ValueError('Supported scene schema versions are 1 and 2.')
        scene_id = str(_required(cfg, 'scene_id', 'root'))
        if not re.fullmatch(r'[A-Za-z0-9_]+', scene_id):
            raise ValueError('scene_id may contain only letters, digits and underscores.')

        inputs = _required(cfg, 'inputs', 'root')
        for key in ('radar_csv', 'image_dir', 'annotation_dir'):
            _required(inputs, key, 'inputs')
        radar_coordinate_mode = inputs.get('radar_coordinate_mode', 'without_xy')
        if radar_coordinate_mode not in ('with_xy', 'without_xy'):
            raise ValueError('inputs.radar_coordinate_mode must be `with_xy` or `without_xy`.')
        columns = inputs['radar_columns']
        column_keys = ['frame', 'radial_speed', 'snr']
        column_keys += ['x', 'y'] if radar_coordinate_mode == 'with_xy' else ['range', 'azimuth']
        for key in column_keys:
            if not str(_required(columns, key, 'inputs.radar_columns')):
                raise ValueError(f'inputs.radar_columns.{key} may not be empty.')
        if radar_coordinate_mode == 'with_xy' and inputs.get('radar_xy_y_positive', 'left') not in ('left', 'right'):
            raise ValueError('inputs.radar_xy_y_positive must be `left` or `right`.')

        output = _required(cfg, 'output', 'root')
        _required(output, 'data_root', 'output')
        if not str(output['info_pkl_name']).endswith('.pkl'):
            raise ValueError('output.info_pkl_name must end in .pkl.')
        for name in ('train', 'val', 'test'):
            filename = _required(output['info_pkl_names'], name, 'output.info_pkl_names')
            if not str(filename).endswith('.pkl') or Path(str(filename)).name != str(filename):
                raise ValueError(f'output.info_pkl_names.{name} must be a plain .pkl filename.')

        split = cfg['dataset_split']
        if split['strategy'] not in ('none', 'chronological'):
            raise ValueError('dataset_split.strategy must be `none` or `chronological`.')
        if split['strategy'] == 'chronological':
            ratios = [float(split[f'{name}_ratio']) for name in ('train', 'val', 'test')]
            if any(value <= 0 or value >= 1 for value in ratios):
                raise ValueError('All dataset split ratios must be between 0 and 1.')
            if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError('Train/val/test ratios must sum to 1.0.')

        calibration = _required(cfg, 'calibration', 'root')
        convention = _required(calibration, 'coordinate_convention', 'calibration')
        if convention not in SUPPORTED_CAMERA_CONVENTIONS:
            raise ValueError(f'Unsupported calibration.coordinate_convention={convention!r}.')
        float(_required(calibration, 'ground_z_world_m', 'calibration'))
        camera = _required(calibration, 'camera', 'calibration')
        intrinsics = _required(camera, 'intrinsics', 'calibration.camera')
        _positive(_required(intrinsics, 'fx', 'calibration.camera.intrinsics'), 'camera fx')
        _positive(_required(intrinsics, 'fy', 'calibration.camera.intrinsics'), 'camera fy')
        width = int(_positive(_required(camera, 'image_width', 'calibration.camera'), 'camera image_width'))
        height = int(_positive(_required(camera, 'image_height', 'calibration.camera'), 'camera image_height'))
        cx, cy = float(_required(intrinsics, 'cx', 'calibration.camera.intrinsics')), float(
            _required(intrinsics, 'cy', 'calibration.camera.intrinsics'))
        if not (0.0 <= cx < width and 0.0 <= cy < height):
            raise ValueError('Camera principal point (cx, cy) must lie inside the declared source image.')
        _ = self.camera_matrix_raw

        radar = _required(calibration, 'radar', 'calibration')
        _pair(_required(radar, 'position_world_xy_m', 'calibration.radar'),
              'calibration.radar.position_world_xy_m')
        height_radar = _positive(_required(radar, 'height_above_ground_m', 'calibration.radar'),
                                 'radar height_above_ground_m')
        return_height = float(radar['return_height_above_ground_m'])
        if return_height < 0 or return_height >= height_radar:
            raise ValueError('Radar return height must satisfy 0 <= return height < radar height.')
        if radar_coordinate_mode == 'without_xy':
            if radar['range_type'] not in ('slant', 'horizontal'):
                raise ValueError('calibration.radar.range_type must be `slant` or `horizontal`.')
            if radar['angle_type'] != 'horizontal_azimuth':
                raise ValueError('This CRN interface currently accepts horizontal azimuth only.')
            if radar['angle_unit'] not in ('degree', 'radian'):
                raise ValueError('calibration.radar.angle_unit must be `degree` or `radian`.')
            if radar['angle_positive'] not in ('left', 'right'):
                raise ValueError('calibration.radar.angle_positive must be `left` or `right`.')
        if radar['speed_type'] != 'radial':
            raise ValueError('This CRN input expects radial Doppler speed, not vehicle world velocity.')
        float(_required(radar, 'yaw_deg', 'calibration.radar'))
        signal_feature = radar['signal_feature']
        if signal_feature['type'] not in ('rcs', 'snr'):
            raise ValueError('calibration.radar.signal_feature.type must be `rcs` or `snr`.')
        if signal_feature['normalization'] not in ('manual', 'auto_train'):
            raise ValueError(
                'calibration.radar.signal_feature.normalization must be `manual` or `auto_train`.')
        stats_filename = str(signal_feature['stats_filename'])
        if not stats_filename or Path(stats_filename).name != stats_filename:
            raise ValueError('Radar signal stats_filename must be a plain filename.')
        if signal_feature['normalization'] == 'manual':
            signal_mean = float(_required(
                signal_feature, 'mean', 'calibration.radar.signal_feature'))
            signal_std = float(_required(
                signal_feature, 'std', 'calibration.radar.signal_feature'))
            if not math.isfinite(signal_mean):
                raise ValueError('Radar signal-feature mean must be finite.')
            if not math.isfinite(signal_std) or signal_std <= 0:
                raise ValueError('Radar signal-feature std must be finite and positive.')

        annotations = cfg['annotations']
        if annotations['size_order'] != 'length_width_height':
            raise ValueError('Only annotation size_order=`length_width_height` is currently supported.')
        if not annotations['translation_is_ground_contact']:
            raise ValueError('Current preprocessing expects ground-contact object translations.')

        derived = self.derived
        ny, nx = derived['bev_shape']
        if nx % 8 or ny % 8:
            raise ValueError(
                f'BEV grid is {nx}x{ny}; both cell counts must be divisible by 8 for the current FPN. '
                'Adjust bounds or cell size.')
        final_h, final_w = derived['final_dim']
        if final_h % 32 or final_w % 64:
            raise ValueError(
                f'model.input_size_hw={final_h}x{final_w}; height must be divisible by 32 and width by 64.')
        if derived['depth_bins'] % 4:
            raise ValueError('The number of depth bins must be divisible by 4 for the PV PointPillars FPN.')
        if int(cfg['model']['max_radar_points_pv']) < 1:
            raise ValueError('model.max_radar_points_pv must be positive.')

        ida = cfg['model']['image_augmentation']
        if 'resize_lim' not in ida:
            minimum = max(final_h / height, final_w / width)
            ida['resize_lim'] = [minimum, minimum + 0.02]
        resize_min, resize_max = _pair(ida['resize_lim'], 'model.image_augmentation.resize_lim')
        if resize_min <= 0 or resize_max < resize_min:
            raise ValueError('Image resize limits must be positive and ordered.')
        if resize_min * width + 1e-6 < final_w or resize_min * height + 1e-6 < final_h:
            raise ValueError('The minimum image resize is too small to cover model.input_size_hw without padding.')
        crop_low, crop_high = _pair(ida['bottom_crop_fraction'],
                                    'model.image_augmentation.bottom_crop_fraction')
        if not (0.0 <= crop_low <= crop_high <= 1.0):
            raise ValueError('Bottom crop fractions must satisfy 0 <= low <= high <= 1.')

        bda = cfg['model']['bev_augmentation']
        for name in ('rotation_probability', 'flip_x_probability', 'flip_y_probability'):
            value = float(bda[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'model.bev_augmentation.{name} must be in [0, 1].')
        _pair(bda['rotation_deg'], 'model.bev_augmentation.rotation_deg')
        scale_min, scale_max = _pair(bda['scale'], 'model.bev_augmentation.scale')
        if scale_min <= 0 or scale_max < scale_min:
            raise ValueError('BEV augmentation scale limits must be positive and ordered.')

        training = cfg['training']
        if int(training['epochs']) < 3:
            raise ValueError('training.epochs must be at least 3.')
        if int(training['batch_size']) < 1:
            raise ValueError('training.batch_size must be positive.')
        if int(training['num_workers']) < 0:
            raise ValueError('training.num_workers may not be negative.')
        if int(training['checkpoint_every_n_epochs']) < 1:
            raise ValueError('training.checkpoint_every_n_epochs must be positive.')
        if not isinstance(training['validate_during_training'], bool):
            raise ValueError('training.validate_during_training must be true or false.')
        if int(training['validation_every_n_epochs']) < 1:
            raise ValueError('training.validation_every_n_epochs must be positive.')
        _positive(training['learning_rate'], 'training.learning_rate')
        if float(training['weight_decay']) < 0:
            raise ValueError('training.weight_decay may not be negative.')
        ratios = _pair(training['lr_milestone_ratios'], 'training.lr_milestone_ratios')
        if not (0 < ratios[0] < ratios[1] < 1):
            raise ValueError('LR milestone ratios must satisfy 0 < first < second < 1.')
        gamma = float(training['lr_gamma'])
        if not 0 < gamma <= 1:
            raise ValueError('training.lr_gamma must be in (0, 1].')

        evaluation = cfg['evaluation']
        if not 0 <= float(evaluation['score_threshold']) <= 1:
            raise ValueError('evaluation.score_threshold must be in [0, 1].')
        _positive(evaluation['match_distance_m'], 'evaluation.match_distance_m')
        distance_bins = [float(value) for value in evaluation['distance_bins_m']]
        if len(distance_bins) < 2 or any(
                right <= left for left, right in zip(distance_bins[:-1], distance_bins[1:])):
            raise ValueError('evaluation.distance_bins_m must contain at least two increasing values.')

        # Prevent a silent far-range crop.  Check all BEV corners in the
        # normalized ground plane against the configured camera-depth bins.
        x_min, _, _ = derived['x_bound']
        _, x_max, _ = derived['x_bound']
        y_min, _, _ = derived['y_bound']
        _, y_max, _ = derived['y_bound']
        corners = np.array([[x, y, 0.0, 1.0] for x in (x_min, x_max) for y in (y_min, y_max)])
        camera = (self.world_to_standard_camera @ corners.T).T
        max_forward = float(camera[:, 2].max())
        if max_forward > derived['d_bound'][1] + 1e-6:
            raise ValueError(
                f'model.depth_bound_m ends at {derived["d_bound"][1]:g} m, but a BEV corner reaches '
                f'{max_forward:.2f} m in camera-forward depth. Increase the depth maximum.')
        farthest_radius = max(math.hypot(x, y) for x in (x_min, x_max) for y in (y_min, y_max))
        if distance_bins[-1] + 1e-6 < farthest_radius:
            raise ValueError(
                f'evaluation.distance_bins_m ends at {distance_bins[-1]:g} m, but the BEV reaches '
                f'{farthest_radius:.2f} m. Extend the final evaluation bin.')

    def validate_raw_inputs(self) -> None:
        for key in ('radar_csv', 'image_dir', 'annotation_dir'):
            path = self.resolve(self.raw['inputs'][key])
            if not path.exists():
                raise FileNotFoundError(f'inputs.{key} does not exist: {path}')
        if not self.resolve(self.raw['inputs']['radar_csv']).is_file():
            raise ValueError('inputs.radar_csv must be a file.')

    def validate_prepared(self) -> None:
        if not self.data_root.is_dir():
            raise FileNotFoundError(f'Prepared data root does not exist: {self.data_root}')
        paths = ([self.info_path_for(name) for name in ('train', 'val', 'test')]
                 if self.has_dataset_split else [self.info_path])
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f'Prepared info PKL does not exist: {path}')
        self.radar_signal_parameters(require_auto=True)

    def validate_training_assets(self, require_init_checkpoint: bool = True) -> None:
        self.validate_prepared()
        checkpoint = self.init_checkpoint
        if require_init_checkpoint and checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(
                f'training.init_checkpoint was specified but does not exist: {checkpoint}. '
                'Set it to null to intentionally train from scratch.')

    def summary_lines(self) -> Iterable[str]:
        d = self.derived
        camera = self.raw['calibration']['camera']
        radar = self.raw['calibration']['radar']
        yield f'scene_id: {self.scene_id}'
        yield f'config: {self.path}'
        yield f'prepared data: {self.data_root}'
        if self.has_dataset_split:
            yield ('info PKLs: ' + ', '.join(
                f'{name}={self.info_path_for(name)}' for name in ('train', 'val', 'test')))
        else:
            yield f'info PKL: {self.info_path}'
        yield f'source image: {camera["image_height"]}x{camera["image_width"]}'
        yield f'model image: {d["final_dim"][0]}x{d["final_dim"][1]}'
        yield f'BEV x/y: {d["x_bound"]} / {d["y_bound"]}'
        yield f'BEV grid HxW: {d["bev_shape"][0]}x{d["bev_shape"][1]}'
        yield f'depth bins: {d["d_bound"]} -> {d["depth_bins"]}'
        if self.raw['inputs'].get('radar_coordinate_mode', 'without_xy') == 'with_xy':
            yield (f'radar: input=with_xy, y_positive='
                   f'{self.raw["inputs"].get("radar_xy_y_positive", "left")}, '
                   f'height={radar["height_above_ground_m"]} m, yaw={radar["yaw_deg"]} deg')
        else:
            yield (f'radar: input=without_xy, range={radar["range_type"]}, '
                   f'angle={radar["angle_type"]}, height={radar["height_above_ground_m"]} m, '
                   f'yaw={radar["yaw_deg"]} deg')
        signal = self.radar_signal_parameters(require_auto=False)
        if signal['mean'] is None:
            yield (f'radar signal: type={signal["type"]}, normalization=auto_train, '
                   f'stats=pending ({signal["source"]})')
        else:
            yield (f'radar signal: type={signal["type"]}, normalization={signal["normalization"]}, '
                   f'mean={signal["mean"]}, std={signal["std"]}, count={signal["count"]}')
        yield (f'train: epochs={self.raw["training"]["epochs"]}, batch={self.raw["training"]["batch_size"]}, '
               f'lr={self.raw["training"]["learning_rate"]}')
        yield (f'train validation: enabled={self.raw["training"]["validate_during_training"]}, '
               f'every_n_epochs={self.raw["training"]["validation_every_n_epochs"]}')


def load_scene_config(path: str | Path) -> SceneConfig:
    return SceneConfig.load(path)
