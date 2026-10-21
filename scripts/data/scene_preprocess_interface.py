#!/usr/bin/env python3
"""Config-driven preparation of one static roadside camera-radar scene.

The interface deliberately separates *scene calibration* from model code:

  1. ``init --interactive`` collects the camera/radar installation parameters
     into one JSON scene manifest;
  2. ``validate`` checks geometry, paths and calibration conventions without
     writing data; and
  3. ``prepare`` creates the CRN radar BEV, PV, sparse-depth and info-pkl
     files from that manifest.

It is intended for tabular radar data and one S3-style JSON annotation per
frame. The scene JSON selects either polar range/azimuth input or Cartesian
radar-local x/y input. CSV columns and radar installation parameters are
declared in the scene JSON; speed must be radial Doppler speed.

Examples:
  python scripts/data/scene_preprocess_interface.py init \
      --interactive --output configs/my_real_scene.json
  python scripts/data/scene_preprocess_interface.py validate \
      --config configs/my_real_scene.json
  python scripts/data/scene_preprocess_interface.py prepare \
      --config configs/my_real_scene.json --image-mode symlink
  python scripts/data/scene_preprocess_interface.py run \
      --config configs/my_real_scene.json --image-mode symlink

``prepare`` is independent from the legacy S1/S2/S3 scripts and never edits
their files.  The dynamic training experiment reads the same JSON, so its
spatial geometry cannot silently diverge from preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from pyquaternion import Quaternion


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scene_config import (
    AXIS_FORWARD_LEFT_UP_TO_RIGHT_DOWN_FORWARD,
    load_scene_config,
    radar_signal_config_fingerprint,
)


def load_config(path: Path) -> tuple[dict, Path]:
    scene = load_scene_config(path)
    scene.validate_raw_inputs()
    return scene.raw, scene.base_dir


def _config_path(config_dir: Path, text: str) -> Path:
    path = Path(text).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def _frame_name(template: str, frame: int) -> str:
    try:
        return template.format(frame=frame)
    except Exception as error:
        raise ValueError(f'Invalid filename template {template!r}; use e.g. {{frame}}.png.') from error


def chronological_split(raw_frames: list[int], config: dict) -> tuple[dict[int, str], dict[str, int]]:
    """Split sorted valid frames into contiguous train/val/test blocks."""
    split = config['dataset_split']
    if split['strategy'] == 'none':
        return {frame: 'train' for frame in raw_frames}, {
            'train': len(raw_frames), 'val': 0, 'test': 0,
        }
    total = len(raw_frames)
    if total < 3:
        raise ValueError('Chronological train/val/test splitting requires at least three valid frames.')
    n_train = round(total * float(split['train_ratio']))
    n_test = round(total * float(split['test_ratio']))
    n_val = total - n_train - n_test
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            f'Split produced train/val/test={n_train}/{n_val}/{n_test}; provide more frames.')
    names = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test
    return dict(zip(raw_frames, names)), {
        'train': n_train, 'val': n_val, 'test': n_test,
    }


def load_radar_csv(
        path: Path, columns: dict[str, str], coordinate_mode: str = 'without_xy'
) -> dict[int, list[dict[str, float]]]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    if coordinate_mode not in ('with_xy', 'without_xy'):
        raise ValueError('radar_coordinate_mode must be `with_xy` or `without_xy`.')
    coordinate_keys = ('x', 'y') if coordinate_mode == 'with_xy' else ('range', 'azimuth')
    required_keys = ('frame', 'radial_speed', 'snr', *coordinate_keys)
    missing_keys = [key for key in required_keys if key not in columns or not str(columns[key])]
    if missing_keys:
        raise ValueError(f'radar_columns missing required keys for {coordinate_mode}: {missing_keys}.')
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {columns[key] for key in required_keys}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f'{path} must contain CSV columns {sorted(required)}; got {reader.fieldnames}.')
        for point_id, row in enumerate(reader):
            values = {'radial_speed_mps': float(row[columns['radial_speed']]),
                      'snr': float(row[columns['snr']])}
            if coordinate_mode == 'with_xy':
                values.update(x_m=float(row[columns['x']]), y_m=float(row[columns['y']]))
            else:
                values.update(range_m=float(row[columns['range']]),
                              angle=float(row[columns['azimuth']]))
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(f'{path}: row {point_id + 2} contains NaN or infinity.')
            grouped[int(row[columns['frame']])].append({
                **values,
                'point_id': point_id,
            })
    return grouped


def normalized_world_to_standard_camera(config: dict) -> np.ndarray:
    """Return normalized-BEV-world -> CRN-standard-camera transform."""
    calibration = config['calibration']
    raw_world_to_camera = np.asarray(
        calibration['camera']['world_to_camera_4x4'], dtype=np.float64)
    normalized_to_raw = np.eye(4, dtype=np.float64)
    normalized_to_raw[2, 3] = float(calibration['ground_z_world_m'])
    if calibration['coordinate_convention'] == 'world_forward_left_up__camera_forward_left_up':
        # S3-style JSON camera coordinate is [forward, left, up].
        return AXIS_FORWARD_LEFT_UP_TO_RIGHT_DOWN_FORWARD @ raw_world_to_camera @ normalized_to_raw
    # S1/S2 effective calibration is already CRN-standard [right, down,
    # forward], so an extra axis conversion here would be a coordinate error.
    return raw_world_to_camera @ normalized_to_raw


def radar_point_to_bev(point: dict[str, float], config: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Convert configured polar or radar-local Cartesian coordinates to BEV.

    Polar slant range uses the configured return-plane height as an elevation
    approximation. Cartesian x/y are already horizontal coordinates; both
    paths then apply the configured radar yaw and world-plane position.
    """
    radar = config['calibration']['radar']
    inputs = config['inputs']
    coordinate_mode = inputs.get('radar_coordinate_mode', 'without_xy')
    if coordinate_mode == 'with_xy':
        local_x, local_y = point['x_m'], point['y_m']
        if inputs.get('radar_xy_y_positive', 'left') == 'right':
            local_y = -local_y
        angle = np.arctan2(local_y, local_x)
    else:
        vertical_delta = (float(radar['height_above_ground_m'])
                          - float(radar.get('return_height_above_ground_m', 0.0)))
        measured_range = point['range_m']
        if radar['range_type'] == 'slant':
            horizontal_sq = measured_range ** 2 - vertical_delta ** 2
            if horizontal_sq <= 0.1:
                return None
            horizontal_range = np.sqrt(horizontal_sq)
        else:
            horizontal_range = measured_range
        angle = point['angle']
        if radar['angle_unit'] == 'degree':
            angle = np.deg2rad(angle)
        if radar['angle_positive'] == 'right':
            angle = -angle
        local_x = horizontal_range * np.cos(angle)
        local_y = horizontal_range * np.sin(angle)
    yaw = np.deg2rad(float(radar['yaw_deg']))
    x = local_x * np.cos(yaw) - local_y * np.sin(yaw) + float(radar['position_world_xy_m'][0])
    y = local_x * np.sin(yaw) + local_y * np.cos(yaw) + float(radar['position_world_xy_m'][1])
    # Doppler is radial speed, not the physical vehicle vx/vy.  These remain
    # directional radar features for model compatibility.
    radial_vx = point['radial_speed_mps'] * np.cos(angle + yaw)
    radial_vy = point['radial_speed_mps'] * np.sin(angle + yaw)
    return (np.array([x, y, float(radar.get('return_height_above_ground_m', 0.0))], dtype=np.float32),
            np.array([radial_vx, radial_vy], dtype=np.float32))


def _link_or_copy(source: Path, destination: Path, mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        if mode == 'copy':
            shutil.copy2(source, destination)
            return
        if mode == 'none' and destination.is_file():
            return
        raise FileExistsError(f'Output image already exists and differs: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'symlink':
        destination.symlink_to(source.resolve())
    elif mode == 'copy':
        shutil.copy2(source, destination)
    elif mode == 'none':
        raise FileNotFoundError(f'--image-mode none requires existing image: {destination}')
    else:
        raise ValueError(f'Unknown image mode {mode!r}.')


def _validate_annotation(annotation: dict, config: dict, frame: int) -> None:
    objects = annotation.get('objects', [])
    if not isinstance(objects, list):
        raise ValueError(f'Frame {frame}: annotation objects must be a list.')
    for object_index, obj in enumerate(objects):
        context = f'Frame {frame}, object {object_index}'
        for key, length in (('translation', 3), ('size', 3)):
            value = obj.get(key)
            if not isinstance(value, (list, tuple)) or len(value) != length:
                raise ValueError(f'{context}: {key} must contain {length} numbers.')
            if not np.isfinite(np.asarray(value, dtype=np.float64)).all():
                raise ValueError(f'{context}: {key} contains a non-finite number.')
        if any(float(value) <= 0 for value in obj['size']):
            raise ValueError(f'{context}: length, width and height must all be positive.')
        if not np.isfinite(float(obj.get('rotation', np.nan))):
            raise ValueError(f'{context}: rotation must be a finite yaw in radians.')

    camera = config['calibration']['camera']
    if 'intrinsics' in annotation:
        for key in ('fx', 'fy', 'cx', 'cy'):
            if not np.isclose(
                    float(annotation['intrinsics'][key]), float(camera['intrinsics'][key]),
                    rtol=0.0, atol=1e-5):
                raise ValueError(
                    f'Frame {frame}: annotation intrinsics.{key} differs from the scene config. '
                    'Do not mix images from a different crop/zoom/calibration.')
    if 'extrinsics_matrix_4x4' in annotation:
        if not np.allclose(
                np.asarray(annotation['extrinsics_matrix_4x4'], dtype=np.float64),
                np.asarray(camera['world_to_camera_4x4'], dtype=np.float64), atol=1e-6):
            raise ValueError(
                f'Frame {frame}: annotation extrinsics differ from the static scene config.')


def _annotation_to_infos(annotation: dict, config: dict, frame: int) -> list[dict]:
    scene_id = config['scene_id']
    bev = config.get('bev')
    annotation_config = config['annotations']
    infos: list[dict] = []
    for obj in annotation.get('objects', []):
        length, width, height = map(float, obj['size'])
        x, y, _ = map(float, obj['translation'])
        yaw = float(obj['rotation'])
        if bev is not None:
            x_min, x_max = map(float, bev['x_bound_m'])
            y_min, y_max = map(float, bev['y_bound_m'])
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                raise ValueError(f'{scene_id} frame {frame}: GT {(x, y)} outside configured BEV bounds.')
        velocity = list(map(float, obj.get('velocity', [0.0, 0.0, 0.0])))
        infos.append({
            'token': f'{scene_id}_{obj.get("token", "car")}',
            'category_name': annotation_config['category_name'],
            # Input S3-style labels use ground-contact XY; CRN expects box center Z.
            'translation': [x, y, height / 2.0],
            'size': [width, length, height],  # nuScenes: [width, length, height]
            'rotation': [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))],
            'velocity': (velocity + [0.0, 0.0, 0.0])[:3],
            'num_lidar_pts': 0,
            'num_radar_pts': 1,
        })
    return infos


def prepare(config: dict, config_dir: Path, image_mode: str, dry_run: bool, max_frames: int | None) -> None:
    inputs, output, calibration = config['inputs'], config['output'], config['calibration']
    scene_id = config['scene_id']
    data_root = _config_path(config_dir, output['data_root'])
    radar_csv = _config_path(config_dir, inputs['radar_csv'])
    image_dir = _config_path(config_dir, inputs['image_dir'])
    annotation_dir = _config_path(config_dir, inputs['annotation_dir'])
    image_template = inputs.get('image_filename_template', '{frame}.png')
    annotation_template = inputs.get('annotation_filename_template', '{frame}.json')
    timestamp_start = int(config.get('timestamp_start_us', 1700000000000000))
    timestamp_step = int(config.get('timestamp_step_us', 100000))
    key_prefix = scene_id
    target_image_dir = data_root / 'samples' / 'CAM_FRONT'
    derived_dirs = {
        'raw': data_root / 'samples' / 'RADAR_FRONT',
        'bev': data_root / 'radar_bev_filter',
        'pv': data_root / 'radar_pv_filter',
        'depth': data_root / 'depth_gt',
    }
    world_to_camera = normalized_world_to_standard_camera(config)
    camera_to_world = np.linalg.inv(world_to_camera)
    intrinsics = calibration['camera']['intrinsics']
    k = np.array([[float(intrinsics['fx']), 0.0, float(intrinsics['cx'])],
                  [0.0, float(intrinsics['fy']), float(intrinsics['cy'])],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    image_w, image_h = int(calibration['camera']['image_width']), int(calibration['camera']['image_height'])
    radar_coordinate_mode = inputs.get('radar_coordinate_mode', 'without_xy')
    points_by_frame = load_radar_csv(
        radar_csv, inputs['radar_columns'], radar_coordinate_mode)
    raw_frames = sorted(points_by_frame)
    if max_frames is not None:
        raw_frames = raw_frames[:max_frames]
    frame_offset = int(inputs.get('frame_id_offset', 0))

    print(f'[scene={scene_id}] input radar frames={len(points_by_frame)}; selected={len(raw_frames)}; '
          f'frame offset={frame_offset}')
    if radar_coordinate_mode == 'with_xy':
        print('[geometry] Radar input=with_xy; '
              f"local y positive={inputs.get('radar_xy_y_positive', 'left')}; ")
    else:
        print('[geometry] Radar input=without_xy; '
              f"Range={calibration['radar']['range_type']}; Angle=horizontal azimuth; ")
    print(f"radar H={calibration['radar']['height_above_ground_m']} m; "
          f"return plane={calibration['radar'].get('return_height_above_ground_m', 0.0)} m; "
          f"yaw={calibration['radar']['yaw_deg']} deg")
    if dry_run:
        for raw_frame in raw_frames:
            frame = raw_frame + frame_offset
            image_path = image_dir / _frame_name(image_template, frame)
            annotation_path = annotation_dir / _frame_name(annotation_template, frame)
            if not image_path.is_file():
                raise FileNotFoundError(f'Missing image for frame {frame}.')
            if not annotation_path.is_file():
                raise FileNotFoundError(f'Missing annotation for frame {frame}.')
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f'Cannot read image for frame {frame}: {image_path}')
            if image.shape[1] != image_w or image.shape[0] != image_h:
                raise ValueError(
                    f'Frame {frame}: image is {image.shape[1]}x{image.shape[0]}, but camera calibration declares '
                    f'{image_w}x{image_h}.  Do not reuse calibration after crop/resize/zoom changes.')
            annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
            _validate_annotation(annotation, config, frame)
        print('[DRY RUN PASS] All selected radar/image/annotation frames and calibration are valid; no files were written.')
        return

    valid_raw_frames = []
    for raw_frame in raw_frames:
        frame = raw_frame + frame_offset
        image_path = image_dir / _frame_name(image_template, frame)
        annotation_path = annotation_dir / _frame_name(annotation_template, frame)
        if image_path.is_file() and annotation_path.is_file():
            valid_raw_frames.append(raw_frame)
        else:
            print(f'[SKIP] frame={frame}: missing image or annotation')
    split_by_frame, split_counts = chronological_split(valid_raw_frames, config)
    print(
        f'[dataset split] strategy={config["dataset_split"]["strategy"]}; '
        f'train/val/test={split_counts["train"]}/{split_counts["val"]}/{split_counts["test"]}')

    target_image_dir.mkdir(parents=True, exist_ok=True)
    for directory in derived_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    infos, summary = [], []
    infos_by_split = {name: [] for name in ('train', 'val', 'test')}
    train_pv_signal_values = []
    for raw_frame in valid_raw_frames:
        frame = raw_frame + frame_offset
        source_image = image_dir / _frame_name(image_template, frame)
        source_annotation = annotation_dir / _frame_name(annotation_template, frame)
        split_name = split_by_frame[raw_frame]
        key = f'{key_prefix}_{frame:06d}'
        target_image = target_image_dir / f'{key}{source_image.suffix.lower()}'
        _link_or_copy(source_image, target_image, image_mode)
        annotation = json.loads(source_annotation.read_text(encoding='utf-8'))
        _validate_annotation(annotation, config, frame)
        raw_rows: list[np.ndarray] = []
        bev_rows: list[list[float]] = []
        pv_rows: list[list[float]] = []
        depth_rows: list[list[float]] = []
        for point in points_by_frame[raw_frame]:
            converted = radar_point_to_bev(point, config)
            if converted is None:
                continue
            xyz, radial_vxy = converted
            camera_xyz = world_to_camera @ np.append(xyz, 1.0)
            right, down, forward = camera_xyz[:3]
            raw = np.zeros(18, dtype=np.float32)
            raw[:3], raw[4], raw[5], raw[6:8], raw[8:10] = xyz, point['point_id'], point['snr'], radial_vxy, radial_vxy
            raw_rows.append(raw)
            bev_rows.append([xyz[0], xyz[1], xyz[2], point['snr'], radial_vxy[0], radial_vxy[1], 0.0])
            if forward > 0.1:
                u, v = float(k[0, 2] + k[0, 0] * right / forward), float(k[1, 2] + k[1, 1] * down / forward)
                if 0 <= u < image_w and 0 <= v < image_h:
                    pv_rows.append([u, v, forward, point['snr'], radial_vxy[0], radial_vxy[1], 0.0])
                    depth_rows.append([u, v, forward])
                    if split_name == 'train':
                        train_pv_signal_values.append(point['snr'])
        np.asarray(raw_rows, dtype=np.float32).reshape(-1, 18).tofile(derived_dirs['raw'] / f'radar_{key}.pcd.bin')
        np.asarray(bev_rows, dtype=np.float32).reshape(-1, 7).tofile(derived_dirs['bev'] / f'radar_{key}.pcd.bin')
        np.asarray(pv_rows, dtype=np.float32).reshape(-1, 7).tofile(derived_dirs['pv'] / f'{key}.png.bin')
        np.asarray(depth_rows, dtype=np.float32).reshape(-1, 3).tofile(derived_dirs['depth'] / f'{key}.png.bin')
        annotations = _annotation_to_infos(annotation, config, frame)
        timestamp = timestamp_start + frame * timestamp_step
        camera = {
            'sample_token': key, 'timestamp': timestamp,
            'filename': f'samples/CAM_FRONT/{target_image.name}', 'height': image_h, 'width': image_w,
            'is_key_frame': True,
            'ego_pose': {'translation': [0.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0]},
            'calibrated_sensor': {
                'translation': camera_to_world[:3, 3].astype(float).tolist(),
                'rotation': [float(v) for v in Quaternion(matrix=camera_to_world[:3, :3]).elements],
                'camera_intrinsic': k.astype(float).tolist(),
            },
            'radar_pv_filename': f'{key}.png', 'depth_filename': f'{key}.png',
        }
        info = {
            'scene_name': f'roadside_{scene_id}', 'scene_token': f'scene_token_{scene_id}',
            'sample_token': key, 'timestamp': timestamp, 'frame_id': frame,
            'cam_infos': {'CAM_FRONT': camera},
            'lidar_infos': {'LIDAR_TOP': {
                'sample_token': key, 'timestamp': timestamp,
                'filename': f'radar_bev_filter/radar_{key}.pcd.bin',
                'ego_pose': {'translation': [0.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0]},
                'calibrated_sensor': {'translation': [0.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0]},
            }},
            'ann_infos': annotations, 'lidar_sweeps': [], 'cam_sweeps': [],
        }
        infos.append(info)
        infos_by_split[split_name].append(info)
        summary.append({'raw_frame': raw_frame, 'frame': frame, 'split': split_name,
                        'gt': len(annotations), 'radar_raw': len(points_by_frame[raw_frame]),
                        'radar_bev': len(bev_rows), 'radar_pv': len(pv_rows)})
        print(f'[OK] split={split_name} raw/frame={raw_frame}/{frame}: '
              f'GT={len(annotations)}, radar BEV/PV={len(bev_rows)}/{len(pv_rows)}')

    observed = np.asarray(train_pv_signal_values, dtype=np.float64)
    if observed.size < 2:
        raise ValueError('Training split has fewer than two radar PV points; cannot normalize the signal channel.')
    observed_mean = float(observed.mean())
    observed_std = float(observed.std(ddof=0))
    if not np.isfinite(observed_std) or observed_std <= 0:
        raise ValueError('Prepared radar PV signal has zero/invalid standard deviation.')
    signal_config = calibration['radar']['signal_feature']
    if signal_config['normalization'] == 'auto_train':
        effective_mean, effective_std = observed_mean, observed_std
    else:
        effective_mean = float(signal_config['mean'])
        effective_std = float(signal_config['std'])
    signal_stats = {
        'schema_version': 1,
        'scene_id': scene_id,
        'feature_type': signal_config['type'],
        'normalization': signal_config['normalization'],
        'population_std_ddof': 0,
        'source_points': 'prepared_train_radar_pv_points',
        'count': int(observed.size),
        'mean': effective_mean,
        'std': effective_std,
        'observed_mean': observed_mean,
        'observed_std': observed_std,
        'observed_min': float(observed.min()),
        'observed_max': float(observed.max()),
        'prepared_frame_count': len(infos),
        'prepared_split_counts': {name: len(values) for name, values in infos_by_split.items()},
        'config_fingerprint': radar_signal_config_fingerprint(config),
    }
    stats_path = data_root / signal_config['stats_filename']
    with stats_path.open('w', encoding='utf-8') as handle:
        json.dump(signal_stats, handle, ensure_ascii=False, indent=2)
    print(
        f'[RADAR SIGNAL] type={signal_config["type"]}; '
        f'mode={signal_config["normalization"]}; train PV n={observed.size}; '
        f'mean={effective_mean:.9g}; std={effective_std:.9g}; saved={stats_path}')

    pkl_path = data_root / output['info_pkl_name']
    with pkl_path.open('wb') as handle:
        pickle.dump(infos, handle)
    split_pkl_paths = {}
    if config['dataset_split']['strategy'] != 'none':
        for name, split_infos in infos_by_split.items():
            split_path = data_root / output['info_pkl_names'][name]
            with split_path.open('wb') as handle:
                pickle.dump(split_infos, handle)
            split_pkl_paths[name] = str(split_path)
    manifest_path = data_root / f'{scene_id}_prepare_manifest.json'
    with manifest_path.open('w', encoding='utf-8') as handle:
        json.dump({'scene_id': scene_id, 'config': config, 'prepared_frames': summary,
                   'radar_signal_statistics': signal_stats,
                   'radar_coordinate_mode': radar_coordinate_mode,
                   'geometry': (
                       '[x,y]=R_yaw*[x_radar,y_radar]+radar_xy'
                       if radar_coordinate_mode == 'with_xy' else
                       'r_xy=sqrt(R^2-(H-z_return)^2); '
                       '[x,y]=R_yaw*[r_xy*cos(angle),r_xy*sin(angle)]+radar_xy')},
                  handle, ensure_ascii=False, indent=2)
    print(f'[DONE] frames={len(infos)}; all_pkl={pkl_path}; split_pkls={split_pkl_paths}; '
          f'manifest={manifest_path}')


def _prompt(prompt: str, default: str | None = None) -> str:
    suffix = f' [{default}]' if default is not None else ''
    answer = input(f'{prompt}{suffix}: ').strip()
    return answer if answer else (default if default is not None else '')


def _prompt_float(prompt: str, default: float | None = None) -> float:
    while True:
        try:
            return float(_prompt(prompt, None if default is None else str(default)))
        except ValueError:
            print('Please enter a numeric value.')


def init_interactive(output_path: Path) -> None:
    """Collect one static-scene calibration manifest without touching raw data."""
    print('Static roadside scene setup. Coordinate convention is fixed to:')
    print('  world=[forward, left, up]; Angle positive=left.')
    scene_id = _prompt('scene_id', 'new_scene')
    radar_csv = _prompt('Radar CSV path (Frame,Range,Angle,Speed,SNR)')
    image_dir = _prompt('Image directory')
    annotation_dir = _prompt('Per-frame annotation JSON directory')
    output_path = output_path.expanduser().resolve()
    data_root_default = os.path.relpath(
        PROJECT_ROOT / 'data' / 'prepared' / scene_id, output_path.parent)
    output_root_default = os.path.relpath(
        PROJECT_ROOT / 'outputs' / 'scenes' / scene_id, output_path.parent)
    data_root = _prompt('Output data_root (relative paths use the config directory)', data_root_default)
    fx, fy = _prompt_float('Camera fx (pixels)'), _prompt_float('Camera fy (pixels)')
    cx, cy = _prompt_float('Camera cx (pixels)'), _prompt_float('Camera cy (pixels)')
    image_width, image_height = int(_prompt_float('Image width')), int(_prompt_float('Image height'))
    print('Enter the static world_to_camera 4x4 matrix, one comma-separated row at a time.')
    extrinsics = []
    for index in range(4):
        while True:
            try:
                row = [float(value.strip()) for value in _prompt(f'row {index}').split(',')]
                if len(row) != 4:
                    raise ValueError
                extrinsics.append(row)
                break
            except ValueError:
                print('Enter exactly four comma-separated numbers.')
    coordinate_convention = _prompt(
        'Camera coordinate convention (1=forward,left,up / 2=right,down,forward)', '1')
    if coordinate_convention == '1':
        coordinate_convention = 'world_forward_left_up__camera_forward_left_up'
    elif coordinate_convention == '2':
        coordinate_convention = 'world_forward_left_up__camera_right_down_forward'
    else:
        raise ValueError('Choose 1 or 2 for the camera coordinate convention.')
    ground_z = _prompt_float('Ground z in raw world coordinates (m)', -30.0)
    radar_x, radar_y = _prompt_float('Radar world x (m)', 0.0), _prompt_float('Radar world y (m)', 0.0)
    radar_height = _prompt_float('Radar height above ground (m)')
    radar_yaw = _prompt_float('Radar yaw from world +x, CCW/left positive (deg)', 0.0)
    return_height = _prompt_float('Assumed radar return height above ground (m)', 0.0)
    config = {
        'schema_version': 2, 'scene_id': scene_id,
        'inputs': {'radar_csv': radar_csv, 'image_dir': image_dir, 'annotation_dir': annotation_dir,
                   'image_filename_template': '{frame}.png', 'annotation_filename_template': '{frame}.json'},
        'output': {'data_root': data_root, 'info_pkl_name': f'nuscenes_infos_{scene_id}_all.pkl',
                   'output_root': output_root_default},
        'calibration': {
            'coordinate_convention': coordinate_convention,
            'ground_z_world_m': ground_z,
            'camera': {'intrinsics': {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy},
                       'image_width': image_width, 'image_height': image_height,
                       'world_to_camera_4x4': extrinsics},
            'radar': {'position_world_xy_m': [radar_x, radar_y], 'height_above_ground_m': radar_height,
                      'yaw_deg': radar_yaw, 'return_height_above_ground_m': return_height,
                      'range_type': 'slant', 'angle_type': 'horizontal_azimuth',
                      'angle_unit': 'degree', 'angle_positive': 'left', 'speed_type': 'radial',
                      'signal_feature': {'type': 'snr', 'normalization': 'auto_train'}},
        },
        'bev': {'x_bound_m': [0.0, 192.0], 'y_bound_m': [-40.0, 40.0], 'cell_size_m': 0.4},
        'annotations': {'category_name': 'vehicle.car', 'size_order': 'length_width_height',
                        'translation_is_ground_contact': True},
        'model': {'input_size_hw': [512, 1408], 'depth_bound_m': [2.0, 242.0, 2.0],
                  'z_bound_m': [-5.0, 3.0]},
        'training': {'epochs': 100, 'batch_size': 1, 'learning_rate': 5e-5,
                     'weight_decay': 1e-4, 'init_checkpoint': None},
        'evaluation': {'score_threshold': 0.35, 'match_distance_m': 2.0,
                       'visualizations': True},
        'timestamp_start_us': 1700000000000000, 'timestamp_step_us': 100000,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    print(f'[OK] Wrote {output_path}. Run validate before prepare.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    init_parser = subparsers.add_parser('init', help='Interactively collect calibration into a JSON manifest.')
    init_parser.add_argument('--interactive', action='store_true', help='Required safety flag for terminal prompts.')
    init_parser.add_argument('--output', type=Path, required=True)
    validate_parser = subparsers.add_parser('validate', help='Validate paths and geometry without writes.')
    validate_parser.add_argument('--config', type=Path, required=True)
    prepare_parser = subparsers.add_parser('prepare', help='Create CRN input files from a validated manifest.')
    prepare_parser.add_argument('--config', type=Path, required=True)
    prepare_parser.add_argument('--image-mode', choices=('symlink', 'copy', 'none'), default='symlink',
                                help='How output data_root obtains camera images (default: symlink).')
    prepare_parser.add_argument('--dry-run', action='store_true', help='Validate all selected frames without writes.')
    prepare_parser.add_argument('--max-frames', type=int, default=None, help='Prepare only first N radar frames.')
    run_parser = subparsers.add_parser(
        'run', help='One command: read-only preflight, then prepare the requested scene data.')
    run_parser.add_argument('--config', type=Path, required=True)
    run_parser.add_argument('--image-mode', choices=('symlink', 'copy', 'none'), default='symlink')
    run_parser.add_argument('--preflight-frames', type=int, default=2,
                            help='How many initial frames to validate before writes (default: 2).')
    run_parser.add_argument('--max-frames', type=int, default=None,
                            help='Prepare only first N frames after preflight; default prepares all frames.')
    args = parser.parse_args()
    if args.command == 'init':
        if not args.interactive:
            parser.error('init requires --interactive because it prompts for calibration values.')
        init_interactive(args.output)
    else:
        config, config_dir = load_config(args.config)
        if args.command == 'validate':
            print(f'[PASS] scene={config["scene_id"]}; manifest, input paths and static calibration are valid.')
        elif args.command == 'prepare':
            prepare(config, config_dir, args.image_mode, args.dry_run, args.max_frames)
        else:  # run
            if args.preflight_frames <= 0:
                raise ValueError('--preflight-frames must be positive.')
            print(f'[RUN] preflight first {args.preflight_frames} frame(s); no files will be written during preflight.')
            prepare(config, config_dir, args.image_mode, dry_run=True, max_frames=args.preflight_frames)
            print('[RUN] preflight passed; starting requested preprocessing.')
            prepare(config, config_dir, args.image_mode, dry_run=False, max_frames=args.max_frames)


if __name__ == '__main__':
    main()
