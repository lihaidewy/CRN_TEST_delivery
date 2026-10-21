"""Persistent epoch summaries for terminals and redirected training logs.

PyTorch Lightning's default tqdm progress bar rewrites its current terminal
line. It is convenient interactively, but makes epoch history hard to review
in tmux, VS Code terminals and ``tee`` logs. This callback keeps the live
progress bar and emits persistent end-of-epoch summaries. It does not alter
optimization, validation, or checkpointing.
"""

from __future__ import annotations

from typing import Iterable

import torch
from pytorch_lightning.callbacks import Callback
from tqdm.auto import tqdm


class RollingEpochLogger(Callback):
    """Print one persistent training line and, when run, one validation line."""

    TRAIN_KEYS: tuple[str, ...] = (
        'learning_rate', 'train/total', 'train/detection', 'train/heatmap',
        'train/bbox', 'train/neighbor_reg', 'train/center_rank', 'train/depth',
    )
    VAL_KEYS: tuple[str, ...] = (
        'val/total', 'val/detection', 'val/heatmap', 'val/bbox', 'val/depth',
    )

    @staticmethod
    def _is_global_zero(trainer) -> bool:
        return bool(getattr(trainer, 'is_global_zero', getattr(trainer, 'global_rank', 0) == 0))

    @staticmethod
    def _format_metrics(metrics: dict, keys: Iterable[str]) -> str:
        values = []
        for key in keys:
            if key not in metrics:
                continue
            value = metrics[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().float().cpu().item()
            try:
                values.append(f'{key}={float(value):.5g}')
            except (TypeError, ValueError):
                values.append(f'{key}={value}')
        return ' | '.join(values) if values else 'no metrics reported'

    def _print(self, trainer, stage: str, keys: Iterable[str]) -> None:
        if not self._is_global_zero(trainer) or getattr(trainer, 'sanity_checking', False):
            return
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        metrics = trainer.callback_metrics
        text = self._format_metrics(metrics, keys)
        # Print above the active bar, then let tqdm redraw it. The current
        # epoch remains visible while completed epochs stay in scrollback.
        tqdm.write(
            f'[Epoch {epoch:03d}/{max_epochs:03d} {stage}] '
            f'global_step={trainer.global_step} | {text}'
        )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._print(trainer, 'train', self.TRAIN_KEYS)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._print(trainer, 'val', self.VAL_KEYS)
