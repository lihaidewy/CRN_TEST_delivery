#!/usr/bin/env python3
"""Evaluate one configuration-driven CRN scene with metrics and overlays."""

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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TRACKER_ROOT = PROJECT_ROOT / 'third_party'
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from config.scene_config import load_scene_config
from datasets.nusc_det_dataset import NuscDatasetRadarDet, collate_fn
from exps.det.CRN_scene_dynamic import CRNDynamicScene, SCENE_CONFIG_ENV
from mmdet3d.core import LiDARInstance3DBoxes
from scripts.evaluation import eval_s3 as metrics_utils
from TNsMultiTrackAdaptive import (
    Measurement, Tracker, TrackerConfig, evaluate as evaluate_tracking,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--score-threshold', type=float, default=None)
    parser.add_argument('--match-distance', type=float, default=None)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--split', choices=('train', 'val', 'test', 'all'), default='test')
    parser.add_argument('--no-visualizations', action='store_true')
    return parser.parse_args()


def tracking_metrics_dict(report) -> dict:
    """Convert the tracker evaluation report into JSON-safe CRN metrics."""
    def number(value: float) -> float | None:
        value = float(value)
        return value if np.isfinite(value) else None

    return {
        'frames': int(report.frames),
        'gt': int(report.gt_points),
        'tp': int(report.matched),
        'fp': int(report.false_positives),
        'fn': int(report.false_negatives),
        'id_switches': int(report.id_switches),
        'fragments': int(report.fragments),
        'track_count': int(report.track_count),
        'precision': number(report.precision),
        'recall': number(report.recall),
        'mota': number(report.mota),
        'rmse_2d_m': number(report.rmse),
        'mean_2d_ate_m': number(report.mean_error),
        'p90_2d_ate_m': number(report.p90_error),
        'p95_2d_ate_m': number(report.p95_error),
        'match_distance_m': number(report.gate),
        'includes_prior_states': True,
        'note': ('Tracking metrics use tracker [x,y] states after M/N confirmation; '
                 'prior states during CRN misses are included.'),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Scene evaluation requires a CUDA-enabled PyTorch runtime.')

    scene = load_scene_config(args.config)
    scene.validate_prepared()
    checkpoint_path = args.ckpt.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint_path}')
    evaluation = scene.raw['evaluation']
    score_threshold = float(
        evaluation['score_threshold'] if args.score_threshold is None else args.score_threshold)
    match_distance = float(
        evaluation['match_distance_m'] if args.match_distance is None else args.match_distance)
    max_frames = int(evaluation['max_frames'] if args.max_frames is None else args.max_frames)
    visualize = bool(evaluation['visualizations']) and not args.no_visualizations
    output_dir = (args.out_dir.expanduser().resolve() if args.out_dir is not None
                  else scene.output_root / 'evaluation' / args.split)
    visual_dir = output_dir / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    if visualize:
        visual_dir.mkdir(parents=True, exist_ok=True)

    os.environ[SCENE_CONFIG_ENV] = str(scene.path)
    os.environ['CRN_SKIP_SCENE_INIT'] = '1'
    model = CRNDynamicScene(batch_size_per_device=1)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.validate_checkpoint_radar_signal(checkpoint, purpose='evaluation')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.cuda().eval()

    dataset = NuscDatasetRadarDet(
        ida_aug_conf=model.ida_aug_conf,
        bda_aug_conf=model.bda_aug_conf,
        rda_aug_conf=model.rda_aug_conf,
        img_backbone_conf=model.backbone_img_conf,
        classes=model.class_names,
        data_root=model.data_root,
        info_paths=[str(scene.info_path_for(args.split))],
        is_train=False,
        img_conf=model.img_conf,
        load_interval=1,
        num_sweeps=model.num_sweeps,
        sweep_idxes=model.sweep_idxes,
        key_idxes=model.key_idxes,
        return_image=True,
        return_depth=model.return_depth,
        return_radar_pv=model.return_radar_pv,
        remove_z_axis=model.remove_z_axis,
        radar_pv_path=model.radar_pv_path,
        radar_pv_num_features=model.radar_pv_num_features,
        max_radar_points_pv=model.max_radar_points_pv,
        radar_signal_mean=model.radar_signal_mean,
        radar_signal_std=model.radar_signal_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(
            batch,
            is_return_image=True,
            is_return_depth=model.return_depth,
            is_return_radar_pv=model.return_radar_pv,
        ),
    )
    limit = len(dataset) if max_frames < 0 else min(max_frames, len(dataset))
    metrics_utils.DISTANCE_BINS = tuple(float(v) for v in evaluation['distance_bins_m'])
    stats = metrics_utils.empty_stats()
    split_label = f'{scene.scene_id} {args.split} split ({limit}/{len(dataset)} frames)'
    csv_path = output_dir / 'tracking_preds.csv'
    csv_tp_path = output_dir / 'tracking_preds_tp.csv'
    tracking_csv_path = output_dir / 'tracking_states.csv'
    header = [
        'frame', 'status', 'class_name', 'score', 'x', 'y', 'z', 'l', 'w', 'h',
        'yaw_raw', 'vx', 'vy', 'gt_id', 'gt_x', 'gt_y',
    ]
    tracking_header = ['frame', 'timestamp_s', 'track_id', 'x', 'y', 'state_type']
    tracking_ts = float(scene.raw.get('timestamp_step_us', 100000)) / 1_000_000.0
    tracker = Tracker(TrackerConfig(
        ts=tracking_ts,
        model_name='CTRV',
        gate_mode='adaptive',
        gate=1.5,
        gate_growth=0.5,
        gate_innovation_weight=2.0,
        max_gate=3.5,
    ))
    first_timestamp_us: int | None = None
    tracking_estimates: dict[float, dict[int, np.ndarray]] = {}
    tracking_truth: dict[float, list[tuple[str, np.ndarray]]] = {}

    print(f'[Scene evaluation] config: {scene.path}')
    print(f'[Scene evaluation] checkpoint: {checkpoint_path}')
    print(f'[Scene evaluation] frames: {limit}; score={score_threshold}; match={match_distance} m')
    with csv_path.open('w', newline='', encoding='utf-8') as all_file, csv_tp_path.open(
            'w', newline='', encoding='utf-8') as tp_file, tracking_csv_path.open(
            'w', newline='', encoding='utf-8') as tracking_file:
        all_writer, tp_writer, tracking_writer = csv.writer(all_file), csv.writer(tp_file), csv.writer(tracking_file)
        all_writer.writerow(header)
        tp_writer.writerow(header)
        tracking_writer.writerow(tracking_header)
        for index, batch in enumerate(loader):
            if index >= limit:
                break
            sweep_imgs, mats, _, _, _, _, _, pts_pv = batch
            sweep_imgs = sweep_imgs.cuda(non_blocking=True).half()
            pts_pv = pts_pv.cuda(non_blocking=True).float()
            mats = {key: value.cuda(non_blocking=True).float() for key, value in mats.items()}
            info = dataset.infos[index]
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                predictions, _ = model.model(
                    sweep_imgs, mats, sweep_ptss=pts_pv, is_train=True)
            metadata = info.copy()
            metadata['box_type_3d'] = LiDARInstance3DBoxes
            boxes3d, scores, labels = model.model.get_bboxes(predictions, [metadata])[0]
            raw_boxes = boxes3d.tensor.detach().cpu().numpy()
            decoded_corners = boxes3d.corners.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            predictions_kept = [
                metrics_utils.prediction_box(box, corners, score, label, model.class_names)
                for box, corners, score, label in zip(raw_boxes, decoded_corners, scores, labels)
                if float(score) >= score_threshold
            ]
            frame = int(info.get('frame_id', index))
            timestamp_us = int(info['timestamp'])
            if first_timestamp_us is None:
                first_timestamp_us = timestamp_us
            timestamp_s = (timestamp_us - first_timestamp_us) / 1_000_000.0
            measurements = [
                Measurement(timestamp_s, prediction['center'][:2], confidence=prediction['score'])
                for prediction in predictions_kept
            ]
            tracker.update(measurements, timestamp_s)
            tracking_states = tracker.snapshot_all(timestamp_s)
            tracking_estimates[timestamp_s] = {
                track_id: state.value[:2].copy()
                for track_id, state in tracking_states.items()
            }
            for track_id, state in sorted(tracking_states.items()):
                tracking_writer.writerow([
                    frame, f'{timestamp_s:.6f}', track_id,
                    f'{state.value[0]:.6f}', f'{state.value[1]:.6f}', state.category.value,
                ])
            gt_boxes = [metrics_utils.gt_box(annotation) for annotation in info['ann_infos']]
            tracking_truth[timestamp_s] = [
                (str(gt['token']), gt['center'][:2].copy()) for gt in gt_boxes
            ]
            matches, unmatched_gt, unmatched_pred = metrics_utils.match_bev(
                gt_boxes, predictions_kept, match_distance)
            match_for_prediction = {prediction_index: gt_index
                                    for gt_index, prediction_index, _ in matches}
            metrics_utils.update_stats(stats, gt_boxes, predictions_kept, matches, unmatched_pred)

            for prediction_index, prediction in enumerate(predictions_kept):
                gt = (gt_boxes[match_for_prediction[prediction_index]]
                      if prediction_index in match_for_prediction else None)
                status = 'TP' if gt is not None else 'FP'
                metrics_utils.write_prediction(all_writer, frame, status, prediction, gt)
                if gt is not None:
                    metrics_utils.write_prediction(tp_writer, frame, status, prediction, gt)

            if visualize:
                camera = info['cam_infos']['CAM_FRONT']
                image_path = Path(model.data_root) / camera['filename']
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f'Cannot read image: {image_path}')
                world_to_camera = metrics_utils.camera_world_to_standard(camera)
                intrinsic = np.asarray(camera['calibrated_sensor']['camera_intrinsic'], dtype=np.float64)
                for gt in gt_boxes:
                    corners = metrics_utils.make_corners(*gt['size'], gt['center'], gt['yaw'])
                    pixels, valid = metrics_utils.project_corners(corners, world_to_camera, intrinsic)
                    metrics_utils.draw_box(image, pixels, valid, (0, 0, 255), 'GT')
                for prediction_index, prediction in enumerate(predictions_kept):
                    pixels, valid = metrics_utils.project_corners(
                        prediction['corners'], world_to_camera, intrinsic)
                    matched = prediction_index in match_for_prediction
                    color, status = ((0, 210, 0), 'TP') if matched else ((255, 0, 255), 'FP')
                    metrics_utils.draw_box(
                        image, pixels, valid, color, f'{status} {prediction["score"]:.2f}')
                cv2.rectangle(image, (20, 20), (900, 98), (20, 20, 20), thickness=-1)
                cv2.putText(
                    image,
                    f'{scene.scene_id} | frame {frame} | GT red / TP green / FP magenta',
                    (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    image,
                    f'GT={len(gt_boxes)}  TP={len(matches)}  FN={len(unmatched_gt)}  FP={len(unmatched_pred)}',
                    (35, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (230, 230, 230), 2, cv2.LINE_AA)
                cv2.imwrite(
                    str(visual_dir / f'frame_{frame:06d}.jpg'), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

            if (index + 1) % 10 == 0 or index + 1 == limit:
                print(f'[Scene evaluation] processed {index + 1}/{limit}')

    tracking_report = evaluate_tracking(
        tracking_estimates, tracking_truth, gate=match_distance)
    tracking_metrics = tracking_metrics_dict(tracking_report)
    metrics = metrics_utils.summarize(stats, score_threshold, match_distance, split_label)
    metrics['scene_config'] = str(scene.path)
    metrics['checkpoint'] = str(checkpoint_path)
    metrics_path = output_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    tracking_metrics_path = output_dir / 'tracking_metrics.json'
    tracking_metrics_path.write_text(
        json.dumps(tracking_metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n[Scene evaluation] complete')
    print(
        f"frames={metrics['frames']} | GT={metrics['gt']} | Pred={metrics['predictions']} | "
        f"TP={metrics['tp']} | FN={metrics['fn']} | FP={metrics['fp']}")
    if metrics['mean_2d_ate_m'] is None:
        print(f"MOTA={metrics['mota'] * 100:.2f}% | Mean 2D ATE=N/A")
    else:
        print(
            f"MOTA={metrics['mota'] * 100:.2f}% | Mean 2D ATE={metrics['mean_2d_ate_m']:.3f} m | "
            f"P90={metrics['p90_2d_ate_m']:.3f} m")
    print(f'[Scene evaluation] tracking states: {tracking_csv_path}')
    print(
        f"[Scene evaluation] tracking metrics: MOTA={tracking_metrics['mota'] * 100:.2f}% | "
        f"P90={tracking_metrics['p90_2d_ate_m']:.3f} m | "
        f"ID switches={tracking_metrics['id_switches']} | {tracking_metrics_path}")
    print(f'[Scene evaluation] outputs: {output_dir}')


if __name__ == '__main__':
    main()
