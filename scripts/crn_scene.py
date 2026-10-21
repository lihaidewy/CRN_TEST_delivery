#!/usr/bin/env python3
"""Unified interface for CRN scene setup, preprocessing, training and evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scene_config import load_scene_config


def _base_environment(config_path: Path | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    local_paths = [str(PROJECT_ROOT), str(PROJECT_ROOT / 'third_party' / 'mmdetection3d')]
    current = environment.get('PYTHONPATH')
    if current:
        local_paths.append(current)
    environment['PYTHONPATH'] = os.pathsep.join(local_paths)
    if config_path is not None:
        environment['CRN_SCENE_CONFIG'] = str(config_path)
    return environment


def _run(command: list[str], config_path: Path | None = None) -> None:
    printable = ' '.join(str(part) for part in command)
    print(f'[RUN] {printable}', flush=True)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_base_environment(config_path),
        check=True,
    )


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--config', type=Path, required=True, help='Scene JSON configuration file.')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    init_parser = subparsers.add_parser('init', help='Interactively create a scene JSON template.')
    init_parser.add_argument('--output', type=Path, required=True)

    for name, help_text in (
        ('validate', 'Validate raw paths, calibration and model geometry.'),
        ('inspect', 'Print all important derived model dimensions.'),
    ):
        child = subparsers.add_parser(name, help=help_text)
        _add_config(child)

    prepare_parser = subparsers.add_parser('prepare', help='Preflight and prepare all configured data.')
    _add_config(prepare_parser)
    prepare_parser.add_argument('--image-mode', choices=('symlink', 'copy', 'none'), default='symlink')
    prepare_parser.add_argument('--max-frames', type=int, default=None)
    prepare_parser.add_argument('--dry-run', action='store_true')

    preview_parser = subparsers.add_parser(
        'preview', help='Draw GT boxes and projected radar before training.')
    _add_config(preview_parser)
    preview_parser.add_argument('--frames', default=None, help='Comma-separated final frame IDs.')
    preview_parser.add_argument('--max-frames', type=int, default=5)
    preview_parser.add_argument('--out-dir', type=Path, default=None)

    smoke_parser = subparsers.add_parser('smoke', help='Verify dataloader and optionally one CUDA forward.')
    _add_config(smoke_parser)
    smoke_parser.add_argument('--forward', action='store_true')

    train_parser = subparsers.add_parser('train', help='Train using the scene JSON as the single source of truth.')
    _add_config(train_parser)
    epoch_group = train_parser.add_mutually_exclusive_group()
    epoch_group.add_argument('--epochs', type=int, default=None,
                             help='Total max epochs; defaults to JSON value.')
    epoch_group.add_argument('--additional-epochs', type=int, default=None,
                             help='With --resume, train this many more epochs.')
    train_parser.add_argument('--batch-size', type=int, default=None, help='Overrides JSON batch size.')
    train_parser.add_argument('--resume', type=Path, default=None, help='Resume complete Lightning state.')
    train_parser.add_argument('--output-dir', type=Path, default=None)
    train_parser.add_argument('--limit-train-batches', type=int, default=None)
    train_parser.add_argument('--no-rolling-log', action='store_true')

    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a checkpoint and optionally draw every frame.')
    _add_config(eval_parser)
    eval_parser.add_argument('--ckpt', type=Path, required=True)
    eval_parser.add_argument('--out-dir', type=Path, default=None)
    eval_parser.add_argument('--score-threshold', type=float, default=None)
    eval_parser.add_argument('--match-distance', type=float, default=None)
    eval_parser.add_argument('--max-frames', type=int, default=None)
    eval_parser.add_argument('--split', choices=('train', 'val', 'test', 'all'), default='test',
                             help='Dataset split to evaluate; all uses every prepared frame (default: test).')
    eval_parser.add_argument('--no-visualizations', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == 'init':
        _run([
            sys.executable, '-u', 'scripts/data/scene_preprocess_interface.py',
            'init', '--interactive', '--output', str(args.output),
        ])
        return

    scene = load_scene_config(args.config)
    if args.command in ('validate', 'prepare'):
        scene.validate_raw_inputs()
    if args.command == 'validate':
        print('[PASS] raw paths, calibration and model geometry are valid.')
        for line in scene.summary_lines():
            print(f'  {line}')
        return
    if args.command == 'inspect':
        for line in scene.summary_lines():
            print(line)
        print('derived model parameters:')
        for key, value in scene.derived.items():
            print(f'  {key}: {value}')
        return
    if args.command == 'prepare':
        command = [
            sys.executable, '-u', 'scripts/data/scene_preprocess_interface.py',
            'prepare' if args.dry_run else 'run', '--config', str(scene.path),
            '--image-mode', args.image_mode,
        ]
        if args.dry_run:
            command.append('--dry-run')
        if args.max_frames is not None:
            command.extend(['--max-frames', str(args.max_frames)])
        _run(command, scene.path)
        return
    if args.command == 'preview':
        scene.validate_raw_inputs()
        command = [
            sys.executable, '-u', 'scripts/data/preview_scene_projection.py',
            '--config', str(scene.path), '--max-frames', str(args.max_frames),
        ]
        if args.frames is not None:
            command.extend(['--frames', args.frames])
        if args.out_dir is not None:
            command.extend(['--out-dir', str(args.out_dir)])
        _run(command, scene.path)
        return
    if args.command == 'smoke':
        command = [sys.executable, '-u', 'scripts/smoke/smoke_scene.py', '--config', str(scene.path)]
        if args.forward:
            command.append('--forward')
        _run(command, scene.path)
        return
    if args.command == 'train':
        scene.validate_training_assets(require_init_checkpoint=args.resume is None)
        training = scene.raw['training']
        if args.additional_epochs is not None:
            if args.additional_epochs < 1:
                raise ValueError('--additional-epochs must be positive.')
            if args.resume is None:
                raise ValueError('--additional-epochs requires --resume.')
            import torch
            resume_preview = args.resume.expanduser().resolve()
            if not resume_preview.is_file():
                raise FileNotFoundError(f'Resume checkpoint does not exist: {resume_preview}')
            checkpoint = torch.load(resume_preview, map_location='cpu')
            completed_epochs = int(checkpoint.get('epoch', -1)) + 1
            epochs = completed_epochs + args.additional_epochs
            print(
                f'[Resume] checkpoint completed epochs={completed_epochs}; '
                f'target max_epochs={epochs}.')
        else:
            epochs = int(training['epochs'] if args.epochs is None else args.epochs)
        batch_size = int(training['batch_size'] if args.batch_size is None else args.batch_size)
        output_dir = (args.output_dir.expanduser().resolve() if args.output_dir is not None
                      else scene.output_root / 'training')
        command = [
            sys.executable, '-u', 'exps/det/CRN_scene_dynamic.py',
            '-b', str(batch_size), '--seed', str(training['seed']),
            '--max_epochs', str(epochs), '--disable_early_stopping',
            '--checkpoint_on_train_epoch_end',
            '--checkpoint_every_n_epochs', str(training['checkpoint_every_n_epochs']),
            '--default_root_dir', str(output_dir),
        ]
        if training['validate_during_training']:
            command.extend([
                '--limit_val_batches', '1.0',
                '--check_val_every_n_epoch', str(training['validation_every_n_epochs']),
            ])
            print(
                f'[Validation] enabled; full val split every '
                f'{training["validation_every_n_epochs"]} epoch(s).')
        else:
            command.extend(['--limit_val_batches', '0'])
            print('[Validation] disabled by scene config.')
        if not args.no_rolling_log:
            command.append('--rolling_terminal_log')
        if args.resume is not None:
            resume = args.resume.expanduser().resolve()
            if not resume.is_file():
                raise FileNotFoundError(f'Resume checkpoint does not exist: {resume}')
            command.extend(['--ckpt_path', str(resume)])
            os.environ['CRN_SKIP_SCENE_INIT'] = '1'
        if args.limit_train_batches is not None:
            command.extend(['--limit_train_batches', str(args.limit_train_batches)])
        _run(command, scene.path)
        return

    scene.validate_prepared()
    command = [
        sys.executable, '-u', 'scripts/evaluation/eval_scene.py',
        '--config', str(scene.path), '--ckpt', str(args.ckpt), '--split', args.split,
    ]
    if args.out_dir is not None:
        command.extend(['--out-dir', str(args.out_dir)])
    if args.score_threshold is not None:
        command.extend(['--score-threshold', str(args.score_threshold)])
    if args.match_distance is not None:
        command.extend(['--match-distance', str(args.match_distance)])
    if args.max_frames is not None:
        command.extend(['--max-frames', str(args.max_frames)])
    if args.no_visualizations:
        command.append('--no-visualizations')
    _run(command, scene.path)


if __name__ == '__main__':
    main()
