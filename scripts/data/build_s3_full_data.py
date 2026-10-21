#!/usr/bin/env python3
"""Build a merged all-frame S3M+S3S training pkl from the complete radar CSVs.

S3M and S3S have overlapping frame numbers (194--323).  Their image and
annotation directories represent distinct traffic-density sequences, so all
derived radar/depth filenames are prefixed with the scene name.  This prevents
one sequence silently overwriting the other during merged training.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Defaults make the package self-contained.  These variables allow large raw
# data to remain on another disk without editing source code.
INPUT_ROOT = Path(os.environ.get('CRN_S3_INPUT_ROOT', PROJECT_ROOT / 'data')).resolve()
DATA_ROOT = Path(os.environ.get('CRN_S3_DATA_ROOT', PROJECT_ROOT / 'data' / 's3_myformatted')).resolve()
RADAR_HEIGHT_M = 30.0
OUTPUT_NAME = 's3_full_v1'

SCENES = {
    'S3M': {
        'csv': INPUT_ROOT / 'S3_M.csv',
        'json_dir': DATA_ROOT / 'M3dbox',
        'image_dir': DATA_ROOT / 'samples' / 'S3M',
    },
    'S3S': {
        'csv': INPUT_ROOT / 'S3_S.csv',
        'json_dir': DATA_ROOT / 'S3dbox',
        'image_dir': DATA_ROOT / 'samples' / 'S3S',
    },
}

# JSON camera: [forward, left, up]; CRN pinhole: [right, down, forward].
AXIS_JSON_TO_STANDARD = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def load_csv(path: Path) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    with path.open(newline='', encoding='utf-8-sig') as file:
        for point_id, row in enumerate(csv.DictReader(file)):
            grouped[int(row['Frame'])].append({
                'range_m': float(row['Range']),
                'angle_deg': float(row['Angle']),
                'radial_speed_mps': float(row['Speed']),
                'snr': float(row['SNR']),
                'point_id': point_id,
            })
    return grouped


def world_to_standard_camera(json_data: dict) -> np.ndarray:
    raw_world_to_camera = np.asarray(json_data['extrinsics_matrix_4x4'], dtype=np.float64)
    normalized_to_raw = np.eye(4, dtype=np.float64)
    normalized_to_raw[2, 3] = -RADAR_HEIGHT_M
    return AXIS_JSON_TO_STANDARD @ raw_world_to_camera @ normalized_to_raw


def radar_point(point: dict) -> tuple[np.ndarray, np.ndarray] | None:
    angle = np.deg2rad(point['angle_deg'])
    slant_range = point['range_m']
    # Range is slant range and Angle is horizontal azimuth.  Keep the existing
    # ground-return approximation, but remove height before azimuth resolution.
    horizontal_sq = slant_range ** 2 - RADAR_HEIGHT_M ** 2
    if horizontal_sq <= 0.1:
        return None
    horizontal_range = np.sqrt(horizontal_sq)
    xyz = np.array([
        horizontal_range * np.cos(angle),
        horizontal_range * np.sin(angle),
        0.0,
    ], dtype=np.float32)
    velocity = np.array([
        point['radial_speed_mps'] * np.cos(angle),
        point['radial_speed_mps'] * np.sin(angle),
    ], dtype=np.float32)
    return xyz, velocity


def write_derivatives(scene: str, frame: int, points: list[dict], json_data: dict) -> dict:
    for name in ('samples/RADAR_FRONT', 'radar_bev_filter', 'radar_pv_filter', 'depth_gt'):
        (DATA_ROOT / name).mkdir(parents=True, exist_ok=True)
    key = f'{scene}_{frame:06d}'
    world_to_camera = world_to_standard_camera(json_data)
    intrinsics = json_data['intrinsics']
    raw_rows, bev_rows, pv_rows, depth_rows = [], [], [], []
    for point in points:
        converted = radar_point(point)
        if converted is None:
            continue
        xyz, radial_vxy = converted
        camera_xyz = world_to_camera @ np.append(xyz, 1.0)
        right, down, forward = camera_xyz[:3]
        if forward <= 0.1:
            continue
        u = float(intrinsics['cx']) + float(intrinsics['fx']) * right / forward
        v = float(intrinsics['cy']) + float(intrinsics['fy']) * down / forward
        raw = np.zeros(18, dtype=np.float32)
        raw[:3], raw[4], raw[5], raw[6:8], raw[8:10] = xyz, point['point_id'], point['snr'], radial_vxy, radial_vxy
        raw_rows.append(raw)
        bev_rows.append([xyz[0], xyz[1], xyz[2], point['snr'], radial_vxy[0], radial_vxy[1], 0.0])
        if 0 <= u < 3840 and 0 <= v < 2160:
            pv_rows.append([u, v, forward, point['snr'], radial_vxy[0], radial_vxy[1], 0.0])
            depth_rows.append([u, v, forward])

    np.asarray(raw_rows, dtype=np.float32).reshape(-1, 18).tofile(DATA_ROOT / 'samples/RADAR_FRONT' / f'radar_{key}.pcd.bin')
    np.asarray(bev_rows, dtype=np.float32).reshape(-1, 7).tofile(DATA_ROOT / 'radar_bev_filter' / f'radar_{key}.pcd.bin')
    np.asarray(pv_rows, dtype=np.float32).reshape(-1, 7).tofile(DATA_ROOT / 'radar_pv_filter' / f'{key}.png.bin')
    np.asarray(depth_rows, dtype=np.float32).reshape(-1, 3).tofile(DATA_ROOT / 'depth_gt' / f'{key}.png.bin')
    return {'radar_raw': len(points), 'radar_bev': len(bev_rows), 'radar_pv': len(pv_rows)}


def make_info(scene: str, frame: int, json_data: dict) -> dict:
    key = f'{scene}_{frame:06d}'
    world_to_camera = world_to_standard_camera(json_data)
    camera_to_world = np.linalg.inv(world_to_camera)
    intrinsics = json_data['intrinsics']
    annotations = []
    for obj in json_data.get('objects', []):
        length, width, height = map(float, obj['size'])
        x, y, _ = map(float, obj['translation'])
        yaw = float(obj['rotation'])
        velocity = list(map(float, obj.get('velocity', [0.0, 0.0, 0.0])))
        annotations.append({
            'token': f'{scene}_{obj.get("token", "car")}',
            'category_name': 'vehicle.car',
            'translation': [x, y, height / 2.0],
            # nuScenes Box convention: [width, length, height].
            'size': [width, length, height],
            'rotation': [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))],
            'velocity': (velocity + [0.0, 0.0, 0.0])[:3],
            'num_lidar_pts': 0,
            'num_radar_pts': 1,
        })
    for annotation in annotations:
        x, y, _ = annotation['translation']
        if not (0.0 <= x <= 192.0 and -40.0 <= y <= 40.0):
            raise ValueError(f'{scene} frame {frame}: GT outside S3 BEV: {(x, y)}')
    camera = {
        'sample_token': key,
        'timestamp': 1700000000000000 + frame * 100000,
        'filename': f'samples/{scene}/{frame}.png',
        # Derived files are scene-prefixed because frame IDs overlap.
        'radar_pv_filename': f'{key}.png',
        'depth_filename': f'{key}.png',
        'height': 2160,
        'width': 3840,
        'is_key_frame': True,
        'ego_pose': {'translation': [0.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0]},
        'calibrated_sensor': {
            'translation': camera_to_world[:3, 3].astype(float).tolist(),
            'rotation': [float(value) for value in Quaternion(matrix=camera_to_world[:3, :3]).elements],
            'camera_intrinsic': [[float(intrinsics['fx']), 0.0, float(intrinsics['cx'])],
                                 [0.0, float(intrinsics['fy']), float(intrinsics['cy'])],
                                 [0.0, 0.0, 1.0]],
        },
    }
    return {
        'scene_name': f'roadside_{scene.lower()}', 'scene_token': f'scene_token_{scene.lower()}',
        'sample_token': key, 'timestamp': camera['timestamp'], 'frame_id': frame,
        'cam_infos': {'CAM_FRONT': camera},
        'lidar_infos': {'LIDAR_TOP': {
            'sample_token': key, 'timestamp': camera['timestamp'],
            'filename': f'radar_bev_filter/radar_{key}.pcd.bin',
            'ego_pose': {'translation': [0.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0]},
            'calibrated_sensor': {'translation': [0.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0]},
        }},
        'ann_infos': annotations, 'lidar_sweeps': [], 'cam_sweeps': [],
    }


def main() -> None:
    infos, manifest = [], {'scenes': {}, 'radar_height_m': RADAR_HEIGHT_M,
                           'annotation_size': '[width, length, height]',
                           'note': 'all frames are intended for training; no held-out split'}
    for scene, paths in SCENES.items():
        if not all(path.exists() for path in paths.values()):
            missing = [str(path) for path in paths.values() if not path.exists()]
            raise FileNotFoundError(f'{scene} missing: {missing}')
        grouped = load_csv(paths['csv'])
        scene_infos, summary = [], []
        for frame in sorted(grouped):
            json_path, image_path = paths['json_dir'] / f'{frame}.json', paths['image_dir'] / f'{frame}.png'
            if not (json_path.exists() and image_path.exists()):
                continue
            with json_path.open(encoding='utf-8') as file:
                json_data = json.load(file)
            stats = write_derivatives(scene, frame, grouped[frame], json_data)
            info = make_info(scene, frame, json_data)
            scene_infos.append(info)
            summary.append({'frame': frame, 'gt': len(info['ann_infos']), **stats})
        infos.extend(scene_infos)
        manifest['scenes'][scene] = {'csv': str(paths['csv']), 'frames': len(scene_infos), 'per_frame': summary}
        print(f'[OK] {scene}: {len(scene_infos)} complete radar/image/json frames')
    output_path = DATA_ROOT / f'nuscenes_infos_{OUTPUT_NAME}_all.pkl'
    with output_path.open('wb') as file:
        pickle.dump(infos, file)
    with (DATA_ROOT / f'{OUTPUT_NAME}_manifest.json').open('w', encoding='utf-8') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f'[OK] merged frames={len(infos)}; pkl={output_path.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
