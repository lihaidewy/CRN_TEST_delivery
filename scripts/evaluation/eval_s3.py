#!/usr/bin/env python3
"""Evaluate and visualize S3 checkpoints on a selected training-data subset.

This is an S3-specific adaptation of ``eval_oblique_480.py``.  It keeps the
same score filtering and 2 m one-to-one BEV matching convention, but uses the
S3 camera calibration serialized in the smoke pkl.  It writes one original
image-size overlay per frame:

* red: ground-truth 3D box;
* green: prediction matched to a ground truth (TP);
* magenta: unmatched prediction (FP).

By default this evaluates the 124-frame S3S smoke dataset.  ``--full_s3``
uses the merged S3M/S3S pkl (558 frames), optionally restricted to moderate
S3M or heavy S3S.  All modes are training-set fit checks, not generalization
results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pyquaternion import Quaternion
from torch.utils.data import DataLoader

# Portable layout: <root>/scripts/evaluation/eval_s3.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.nusc_det_dataset import NuscDatasetRadarDet, collate_fn
from exps.det.CRN_s3_portable import CRNS3Portable
from mmdet3d.core import LiDARInstance3DBoxes


DISTANCE_BINS = (0.0, 50.0, 100.0, 150.0, 200.0, 240.0)
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='S3 training-set evaluation with per-frame visualizations.')
    parser.add_argument('--ckpt', required=True, help='S3 smoke checkpoint, e.g. .../last.ckpt')
    parser.add_argument('--out_dir', required=True, help='Directory for metrics, CSVs and visualizations.')
    parser.add_argument('--score_thr', type=float, default=0.35, help='Keep predictions with score >= threshold.')
    parser.add_argument('--match_dist', type=float, default=2.0, help='One-to-one BEV matching distance in metres.')
    parser.add_argument('--max_frames', type=int, default=-1, help='Use -1 for every frame in the selected dataset/subset.')
    parser.add_argument('--full_s3', action='store_true', help='Evaluate merged full data: S3M=211 + S3S=347 frames.')
    parser.add_argument('--s3_512h', action='store_true',
                        help='Retained only for command compatibility; the portable S3 model always uses 512x1408.')
    parser.add_argument('--subset', choices=['all', 'S3M', 'S3S'], default='all',
                        help='With --full_s3: all, S3M (moderate), or S3S (heavy).')
    parser.add_argument('--no_visualizations', action='store_true', help='Skip frame overlays but keep metrics and CSVs.')
    return parser.parse_args()


def make_corners(length: float, width: float, height: float, center: np.ndarray, yaw: float) -> np.ndarray:
    """Return 8 S3-world corners for a box whose center is ``center``."""
    local = np.array([
        [ length / 2,  width / 2, -height / 2],
        [ length / 2, -width / 2, -height / 2],
        [-length / 2, -width / 2, -height / 2],
        [-length / 2,  width / 2, -height / 2],
        [ length / 2,  width / 2,  height / 2],
        [ length / 2, -width / 2,  height / 2],
        [-length / 2, -width / 2,  height / 2],
        [-length / 2,  width / 2,  height / 2],
    ], dtype=np.float64)
    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw),  np.cos(yaw), 0.0],
        [0.0,          0.0,         1.0],
    ], dtype=np.float64)
    return local @ rotation.T + center[None, :]


def gt_box(annotation: dict) -> dict:
    center = np.asarray(annotation['translation'], dtype=np.float64)
    orientation = Quaternion(annotation['rotation'])
    forward = orientation.rotation_matrix @ np.array([1.0, 0.0, 0.0])
    # pkl annotation follows nuScenes Box convention [width, length, height].
    # Rendering is clearer in physical [length, width, height] order.
    stored_width, stored_length, height = map(float, annotation['size'])
    return {
        'token': annotation.get('token', ''),
        'center': center,
        'size': np.array([stored_length, stored_width, height], dtype=np.float64),
        'yaw': float(np.arctan2(forward[1], forward[0])),
    }


def prediction_box(box: np.ndarray, corners: np.ndarray, score: float, label: int,
                   class_names: list[str]) -> dict:
    """Read a decoded LiDAR box using its native X/Y/Z size and yaw convention.

    ``CenterPointBBoxCoder`` produces [x, y, z, x_size, y_size, z_size, yaw].
    ``CenterHead.get_bboxes`` converts z from the gravity centre to bottom
    centre before constructing ``LiDARInstance3DBoxes``.  The latter's
    ``corners`` property is therefore the authoritative geometry for drawing;
    no legacy yaw-minus-90-degree correction is applied to S3.
    """
    x, y, bottom_z, length, width, height, yaw = map(float, box[:7])
    return {
        'center': np.array([x, y, bottom_z + height / 2.0], dtype=np.float64),
        'size': np.array([length, width, height], dtype=np.float64),
        'corners': np.asarray(corners, dtype=np.float64),
        'yaw': yaw,
        'raw_yaw': yaw,
        'score': float(score),
        'label': int(label),
        'class_name': class_names[int(label)] if 0 <= int(label) < len(class_names) else f'class_{int(label)}',
        'velocity': [float(box[7]) if len(box) > 7 else 0.0, float(box[8]) if len(box) > 8 else 0.0],
    }


def camera_world_to_standard(cam_info: dict) -> np.ndarray:
    """Read standard-camera calibration stored by build_s3s_smoke_data.py."""
    sensor = cam_info['calibrated_sensor']
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = Quaternion(sensor['rotation']).rotation_matrix
    camera_to_world[:3, 3] = np.asarray(sensor['translation'], dtype=np.float64)
    return np.linalg.inv(camera_to_world)


def project_corners(corners_world: np.ndarray, world_to_camera: np.ndarray, intrinsic: np.ndarray):
    """Project corners in CRN's standard pinhole frame: [right, down, forward]."""
    homogeneous = np.concatenate((corners_world, np.ones((len(corners_world), 1))), axis=1)
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    forward = camera[:, 2]
    valid = forward > 0.1
    pixels = np.full((len(corners_world), 2), np.nan, dtype=np.float64)
    if valid.any():
        pixels[valid, 0] = intrinsic[0, 0] * camera[valid, 0] / forward[valid] + intrinsic[0, 2]
        pixels[valid, 1] = intrinsic[1, 1] * camera[valid, 1] / forward[valid] + intrinsic[1, 2]
    return pixels, valid


def draw_box(image: np.ndarray, pixels: np.ndarray, valid: np.ndarray, color: tuple[int, int, int], label: str) -> bool:
    """Draw only visible, in-front cuboid segments; return whether anything was drawn."""
    image_height, image_width = image.shape[:2]
    drawn = False
    for start, end in BOX_EDGES:
        if not (valid[start] and valid[end]):
            continue
        p0, p1 = pixels[start], pixels[end]
        if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            continue
        line = np.vstack((p0, p1))
        if line[:, 0].max() < 0 or line[:, 0].min() >= image_width or line[:, 1].max() < 0 or line[:, 1].min() >= image_height:
            continue
        cv2.line(image, tuple(np.rint(p0).astype(int)), tuple(np.rint(p1).astype(int)), color, 3, cv2.LINE_AA)
        drawn = True
    if drawn:
        anchor_candidates = pixels[valid & np.isfinite(pixels).all(axis=1)]
        anchor = np.rint(anchor_candidates[np.argmin(anchor_candidates[:, 1])]).astype(int)
        anchor[0] = int(np.clip(anchor[0], 0, image_width - 1))
        anchor[1] = int(np.clip(anchor[1] - 6, 20, image_height - 1))
        cv2.putText(image, label, tuple(anchor), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    return drawn


def match_bev(gt_boxes: list[dict], pred_boxes: list[dict], threshold: float):
    """Score-independent greedy one-to-one match by ascending BEV distance."""
    pairs = []
    for gi, gt in enumerate(gt_boxes):
        for pi, prediction in enumerate(pred_boxes):
            distance = float(np.linalg.norm(gt['center'][:2] - prediction['center'][:2]))
            if distance <= threshold:
                pairs.append((distance, gi, pi))
    pairs.sort(key=lambda item: item[0])
    used_gt, used_pred, matches = set(), set(), []
    for distance, gi, pi in pairs:
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            matches.append((gi, pi, distance))
    return matches, [i for i in range(len(gt_boxes)) if i not in used_gt], [i for i in range(len(pred_boxes)) if i not in used_pred]


def distance_bin(distance: float) -> str | None:
    for left, right in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
        if left <= distance < right:
            return f'{int(left)}-{int(right)}m'
    return None


def empty_stats() -> dict:
    names = [f'{int(left)}-{int(right)}m' for left, right in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:])]
    return {
        'frames': 0, 'gt': 0, 'predictions': 0, 'tp': 0,
        'ate_2d': [], 'lon_abs': [], 'lat_abs': [], 'aoe_rad': [],
        'bins': {name: {'gt': 0, 'tp': 0, 'fp': 0, 'ate_2d': []} for name in names},
    }


def update_stats(stats: dict, gt_boxes: list[dict], pred_boxes: list[dict], matches, unmatched_pred: list[int]) -> None:
    stats['frames'] += 1
    stats['gt'] += len(gt_boxes)
    stats['predictions'] += len(pred_boxes)
    stats['tp'] += len(matches)
    for gt in gt_boxes:
        name = distance_bin(float(np.linalg.norm(gt['center'][:2])))
        if name:
            stats['bins'][name]['gt'] += 1
    for pi in unmatched_pred:
        name = distance_bin(float(np.linalg.norm(pred_boxes[pi]['center'][:2])))
        if name:
            stats['bins'][name]['fp'] += 1
    for gi, pi, _ in matches:
        gt, prediction = gt_boxes[gi], pred_boxes[pi]
        delta = prediction['center'][:2] - gt['center'][:2]
        ate = float(np.linalg.norm(delta))
        aoe = abs((prediction['yaw'] - gt['yaw'] + np.pi) % (2 * np.pi) - np.pi)
        stats['ate_2d'].append(ate)
        stats['lon_abs'].append(abs(float(delta[0])))
        stats['lat_abs'].append(abs(float(delta[1])))
        stats['aoe_rad'].append(float(aoe))
        name = distance_bin(float(np.linalg.norm(gt['center'][:2])))
        if name:
            stats['bins'][name]['tp'] += 1
            stats['bins'][name]['ate_2d'].append(ate)


def summarize(stats: dict, score_threshold: float, match_distance: float, split: str) -> dict:
    tp, gt, predictions = stats['tp'], stats['gt'], stats['predictions']
    fn, fp = gt - tp, predictions - tp
    values = lambda key: np.asarray(stats[key], dtype=np.float64)
    result = {
        'split': split,
        'score_threshold': score_threshold,
        'match_distance_m': match_distance,
        'frames': stats['frames'], 'gt': gt, 'predictions': predictions, 'tp': tp, 'fn': fn, 'fp': fp,
        'precision': float(tp / predictions) if predictions else 0.0,
        'recall': float(tp / gt) if gt else 0.0,
        'mota': float(1.0 - (fn + fp) / gt) if gt else 0.0,
        'mean_2d_ate_m': float(values('ate_2d').mean()) if tp else None,
        'p90_2d_ate_m': float(np.percentile(values('ate_2d'), 90)) if tp else None,
        'mean_longitudinal_abs_error_m': float(values('lon_abs').mean()) if tp else None,
        'mean_lateral_abs_error_m': float(values('lat_abs').mean()) if tp else None,
        'mean_aoe_deg': float(np.degrees(values('aoe_rad').mean())) if tp else None,
        'distance_bins': {},
    }
    for name, values_in_bin in stats['bins'].items():
        errors = np.asarray(values_in_bin['ate_2d'], dtype=np.float64)
        result['distance_bins'][name] = {
            'gt': values_in_bin['gt'], 'tp': values_in_bin['tp'], 'fp': values_in_bin['fp'],
            'mean_2d_ate_m': float(errors.mean()) if len(errors) else None,
            'p90_2d_ate_m': float(np.percentile(errors, 90)) if len(errors) else None,
        }
    return result


def write_prediction(writer: csv.writer, frame: int, status: str, box: dict, gt: dict | None = None) -> None:
    row = [frame, status, box['class_name'], f"{box['score']:.6f}", *[f'{v:.6f}' for v in box['center']],
           *[f'{v:.6f}' for v in box['size']], f"{box['raw_yaw']:.6f}", *[f'{v:.6f}' for v in box['velocity']]]
    if gt is None:
        row += ['', '', '']
    else:
        row += [gt['token'], f"{gt['center'][0]:.6f}", f"{gt['center'][1]:.6f}"]
    writer.writerow(row)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This evaluator requires a CUDA-enabled PyTorch runtime.')
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    output_dir = Path(args.out_dir).expanduser().resolve()
    visual_dir = output_dir / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_visualizations:
        visual_dir.mkdir(parents=True, exist_ok=True)

    if not args.full_s3:
        raise ValueError('The portable evaluator supports the complete S3 pkl only; pass --full_s3.')
    model = CRNS3Portable()
    split_label = {
        'all': 'S3 all-frame training set (S3M moderate + S3S heavy; 558 frames)',
        'S3M': 'S3M moderate all-frame training set (211 frames)',
        'S3S': 'S3S heavy all-frame training set (347 frames)',
    }[args.subset] + '; 512x1408 input'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.cuda().eval()
    print(f'[S3 evaluation] checkpoint: {checkpoint_path}')
    print(f'[S3 evaluation] split: {split_label}')

    dataset = NuscDatasetRadarDet(
        ida_aug_conf=model.ida_aug_conf, bda_aug_conf=model.bda_aug_conf,
        rda_aug_conf=model.rda_aug_conf, img_backbone_conf=model.backbone_img_conf,
        classes=model.class_names, data_root=model.data_root, info_paths=model.train_info_paths,
        is_train=False, img_conf=model.img_conf, load_interval=1, num_sweeps=1,
        sweep_idxes=[], key_idxes=[], return_image=True, return_depth=model.return_depth,
        return_radar_pv=model.return_radar_pv, remove_z_axis=model.remove_z_axis,
        radar_pv_path=model.radar_pv_path, radar_pv_num_features=model.radar_pv_num_features,
        radar_signal_mean=getattr(model, 'radar_signal_mean', 4.783),
        radar_signal_std=getattr(model, 'radar_signal_std', 7.576),
    )
    if args.full_s3 and args.subset != 'all':
        required_scene = {'S3M': 'roadside_s3m', 'S3S': 'roadside_s3s'}[args.subset]
        dataset.infos = [info for info in dataset.infos if info['scene_name'] == required_scene]
        if not dataset.infos:
            raise RuntimeError(f'No frames found for requested subset {args.subset}.')
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, is_return_image=True, is_return_depth=model.return_depth,
                                             is_return_radar_pv=model.return_radar_pv),
    )
    limit = len(dataset) if args.max_frames < 0 else min(args.max_frames, len(dataset))
    stats = empty_stats()
    csv_path = output_dir / 'tracking_preds.csv'
    csv_tp_path = output_dir / 'tracking_preds_tp.csv'
    header = ['frame', 'status', 'class_name', 'score', 'x', 'y', 'z', 'l', 'w', 'h', 'yaw_raw', 'vx', 'vy', 'gt_id', 'gt_x', 'gt_y']

    with csv_path.open('w', newline='', encoding='utf-8') as all_file, csv_tp_path.open('w', newline='', encoding='utf-8') as tp_file:
        all_writer, tp_writer = csv.writer(all_file), csv.writer(tp_file)
        all_writer.writerow(header)
        tp_writer.writerow(header)
        for index, batch in enumerate(loader):
            if index >= limit:
                break
            sweep_imgs, mats, _, _, _, _, _, pts_pv = batch
            sweep_imgs = sweep_imgs.cuda(non_blocking=True).half()
            pts_pv = pts_pv.cuda(non_blocking=True).float()
            mats = {key: value.cuda(non_blocking=True).float() for key, value in mats.items()}
            info = dataset.infos[index]
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                predictions, _ = model.model(sweep_imgs, mats, sweep_ptss=pts_pv, is_train=True)
            metadata = info.copy()
            metadata['box_type_3d'] = LiDARInstance3DBoxes
            boxes3d, scores, labels = model.model.get_bboxes(predictions, [metadata])[0]
            raw_boxes = boxes3d.tensor.detach().cpu().numpy()
            decoded_corners = boxes3d.corners.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            predictions_kept = [
                prediction_box(box, corners, score, label, model.class_names)
                for box, corners, score, label in zip(raw_boxes, decoded_corners, scores, labels)
                if float(score) >= args.score_thr
            ]
            gt_boxes = [gt_box(annotation) for annotation in info['ann_infos']]
            matches, unmatched_gt, unmatched_pred = match_bev(gt_boxes, predictions_kept, args.match_dist)
            match_for_prediction = {pi: gi for gi, pi, _ in matches}
            update_stats(stats, gt_boxes, predictions_kept, matches, unmatched_pred)

            frame = int(str(info['sample_token']).rsplit('_', 1)[-1])
            for pi, prediction in enumerate(predictions_kept):
                gt = gt_boxes[match_for_prediction[pi]] if pi in match_for_prediction else None
                status = 'TP' if gt is not None else 'FP'
                write_prediction(all_writer, frame, status, prediction, gt)
                if gt is not None:
                    write_prediction(tp_writer, frame, status, prediction, gt)

            if not args.no_visualizations:
                image_path = Path(model.data_root) / info['cam_infos']['CAM_FRONT']['filename']
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f'Cannot read S3 image: {image_path}')
                camera = info['cam_infos']['CAM_FRONT']
                world_to_camera = camera_world_to_standard(camera)
                intrinsic = np.asarray(camera['calibrated_sensor']['camera_intrinsic'], dtype=np.float64)
                for gt in gt_boxes:
                    corners = make_corners(*gt['size'], gt['center'], gt['yaw'])
                    pixels, valid = project_corners(corners, world_to_camera, intrinsic)
                    draw_box(image, pixels, valid, (0, 0, 255), 'GT')
                for pi, prediction in enumerate(predictions_kept):
                    pixels, valid = project_corners(prediction['corners'], world_to_camera, intrinsic)
                    color = (0, 210, 0) if pi in match_for_prediction else (255, 0, 255)
                    status = 'TP' if pi in match_for_prediction else 'FP'
                    draw_box(image, pixels, valid, color, f'{status} {prediction["score"]:.2f}')
                cv2.rectangle(image, (20, 20), (770, 98), (20, 20, 20), thickness=-1)
                scene_tag = info.get('scene_name', 'S3').replace('roadside_', '').upper()
                cv2.putText(image, f'{scene_tag} train | frame {frame} | GT red / TP green / FP magenta', (35, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'GT={len(gt_boxes)}  TP={len(matches)}  FN={len(unmatched_gt)}  FP={len(unmatched_pred)}',
                            (35, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (230, 230, 230), 2, cv2.LINE_AA)
                cv2.imwrite(str(visual_dir / f'frame_{frame:06d}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 92])

            if (index + 1) % 10 == 0 or index + 1 == limit:
                print(f'[S3 evaluation] processed {index + 1}/{limit} frames')

    metrics = summarize(stats, args.score_thr, args.match_dist, split_label)
    metrics_path = output_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n[S3 evaluation] complete')
    print(f"frames={metrics['frames']} | GT={metrics['gt']} | Pred={metrics['predictions']} | TP={metrics['tp']} | FN={metrics['fn']} | FP={metrics['fp']}")
    if metrics['mean_2d_ate_m'] is None:
        print(f"MOTA={metrics['mota'] * 100:.2f}% | Mean 2D ATE=N/A (no TP)")
    else:
        print(f"MOTA={metrics['mota'] * 100:.2f}% | Mean 2D ATE={metrics['mean_2d_ate_m']:.3f} m | P90={metrics['p90_2d_ate_m']:.3f} m")
    print(f'[S3 evaluation] outputs: {output_dir}')


if __name__ == '__main__':
    main()
