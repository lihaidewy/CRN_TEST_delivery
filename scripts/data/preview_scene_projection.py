#!/usr/bin/env python3
"""Draw configured GT boxes and radar projections before model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scene_config import load_scene_config
from scripts.data.scene_preprocess_interface import (
    _frame_name,
    _validate_annotation,
    load_radar_csv,
    radar_point_to_bev,
)


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def make_box_corners(obj: dict) -> np.ndarray:
    length, width, height = map(float, obj['size'])
    x, y, _ = map(float, obj['translation'])
    yaw = float(obj['rotation'])
    local = np.array([
        [length / 2, width / 2, 0.0], [length / 2, -width / 2, 0.0],
        [-length / 2, -width / 2, 0.0], [-length / 2, width / 2, 0.0],
        [length / 2, width / 2, height], [length / 2, -width / 2, height],
        [-length / 2, -width / 2, height], [-length / 2, width / 2, height],
    ], dtype=np.float64)
    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return local @ rotation.T + np.array([x, y, 0.0])


def project(points_world: np.ndarray, world_to_camera: np.ndarray, intrinsic: np.ndarray):
    homogeneous = np.concatenate((points_world, np.ones((len(points_world), 1))), axis=1)
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    forward = camera[:, 2]
    valid = forward > 0.1
    pixels = np.full((len(points_world), 2), np.nan, dtype=np.float64)
    pixels[valid, 0] = intrinsic[0, 0] * camera[valid, 0] / forward[valid] + intrinsic[0, 2]
    pixels[valid, 1] = intrinsic[1, 1] * camera[valid, 1] / forward[valid] + intrinsic[1, 2]
    return pixels, valid


def draw_box(image: np.ndarray, pixels: np.ndarray, valid: np.ndarray) -> None:
    for start, end in BOX_EDGES:
        if valid[start] and valid[end] and np.isfinite(pixels[[start, end]]).all():
            cv2.line(
                image,
                tuple(np.rint(pixels[start]).astype(int)),
                tuple(np.rint(pixels[end]).astype(int)),
                (0, 0, 255), 3, cv2.LINE_AA,
            )


def parse_frames(text: str | None, available_final: list[int], limit: int) -> list[int]:
    if text:
        requested = [int(value.strip()) for value in text.split(',') if value.strip()]
        missing = sorted(set(requested) - set(available_final))
        if missing:
            raise ValueError(f'Requested frames are absent from radar CSV: {missing}')
        return requested
    return available_final[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--frames', help='Comma-separated final frame IDs, e.g. 112,120,150.')
    parser.add_argument('--max-frames', type=int, default=5)
    parser.add_argument('--out-dir', type=Path, default=None)
    args = parser.parse_args()
    if args.max_frames < 1:
        raise ValueError('--max-frames must be positive.')

    scene = load_scene_config(args.config)
    scene.validate_raw_inputs()
    cfg = scene.raw
    inputs = cfg['inputs']
    radar_by_raw_frame = load_radar_csv(
        scene.resolve(inputs['radar_csv']), inputs['radar_columns'],
        inputs.get('radar_coordinate_mode', 'without_xy'))
    frame_offset = int(inputs['frame_id_offset'])
    final_to_raw = {raw + frame_offset: raw for raw in radar_by_raw_frame}
    frames = parse_frames(args.frames, sorted(final_to_raw), args.max_frames)
    image_dir = scene.resolve(inputs['image_dir'])
    annotation_dir = scene.resolve(inputs['annotation_dir'])
    output_dir = (args.out_dir.expanduser().resolve() if args.out_dir is not None
                  else scene.output_root / 'preprocess_preview')
    output_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = cfg['calibration']['camera']['intrinsics']
    intrinsic = np.array([
        [float(intrinsics['fx']), 0.0, float(intrinsics['cx'])],
        [0.0, float(intrinsics['fy']), float(intrinsics['cy'])],
        [0.0, 0.0, 1.0],
    ])
    image_template = inputs['image_filename_template']
    annotation_template = inputs['annotation_filename_template']
    image_width = int(cfg['calibration']['camera']['image_width'])
    image_height = int(cfg['calibration']['camera']['image_height'])

    for frame in frames:
        raw_frame = final_to_raw[frame]
        image_path = image_dir / _frame_name(image_template, frame)
        annotation_path = annotation_dir / _frame_name(annotation_template, frame)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'Cannot read image: {image_path}')
        if image.shape[:2] != (image_height, image_width):
            raise ValueError(
                f'Frame {frame}: actual image is {image.shape[1]}x{image.shape[0]}, '
                f'config declares {image_width}x{image_height}.')
        annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
        _validate_annotation(annotation, cfg, frame)

        visible_radar = 0
        for point_data in radar_by_raw_frame[raw_frame]:
            converted = radar_point_to_bev(point_data, cfg)
            if converted is None:
                continue
            xyz, _ = converted
            pixels, valid = project(xyz[None], scene.world_to_standard_camera, intrinsic)
            if valid[0] and 0 <= pixels[0, 0] < image_width and 0 <= pixels[0, 1] < image_height:
                cv2.circle(image, tuple(np.rint(pixels[0]).astype(int)), 8, (0, 255, 255), -1, cv2.LINE_AA)
                visible_radar += 1
        for obj in annotation.get('objects', []):
            pixels, valid = project(make_box_corners(obj), scene.world_to_standard_camera, intrinsic)
            draw_box(image, pixels, valid)

        cv2.rectangle(image, (20, 20), (950, 95), (20, 20, 20), -1)
        cv2.putText(
            image, f'{scene.scene_id} frame {frame}: GT red / radar yellow',
            (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            image, f'GT={len(annotation.get("objects", []))}, visible radar={visible_radar}',
            (35, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (230, 230, 230), 2, cv2.LINE_AA)
        output_path = output_dir / f'frame_{frame:06d}.jpg'
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f'[OK] frame={frame}; GT={len(annotation.get("objects", []))}; '
              f'visible radar={visible_radar}; {output_path}')

    print(f'[DONE] projection previews: {output_dir}')


if __name__ == '__main__':
    main()
