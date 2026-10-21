"""CSV 读取工具（只依赖标准库 + numpy）。

数据约定（本仓库的 ``meas.csv`` / ``true.csv``）：

- ``meas.csv``：``frame, x, y, ...``——同一帧多行，每行一个量测，无目标 id；
- ``true.csv``：``frame, id, x, y, vx, vy``——真值，每帧多行，每行一个目标。

时间戳统一为 ``(frame - frame_origin) * ts``，量测与真值必须用**同一个 origin**，
否则评估时会整体错位。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..typedef.meas import Measurement

PathLike = Union[str, Path]


def _read_rows(path: PathLike) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} 没有表头")
        return [name.strip() for name in reader.fieldnames], [
            {(key or "").strip(): (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def min_frame(path: PathLike, frame_col: str = "frame") -> int:
    """返回文件中的最小帧号（用于对齐量测与真值的时间原点）。"""
    _, rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path} 没有数据行")
    return min(int(float(row[frame_col])) for row in rows)


def load_measurements(
    path: PathLike,
    *,
    x_col: str = "x",
    y_col: str = "y",
    frame_col: str = "frame",
    ts: float = 0.1,
    frame_origin: Optional[int] = None,
) -> tuple[int, list[tuple[float, list[Measurement]]]]:
    """读取量测文件。

    Returns
    -------
    ``(frame_origin, frames)``，其中 ``frames`` 为按帧升序的
    ``[(timestamp, [Measurement, ...]), ...]``，**每个出现的帧都会出现一次**
    （没有量测的帧不会出现在列表里，与 MATLAB 逐帧遍历略有差别，
    但 Tracker 只关心调用时刻，不影响语义）。
    """
    fieldnames, rows = _read_rows(path)
    for column in (frame_col, x_col, y_col):
        if column not in fieldnames:
            raise ValueError(f"{path} 缺少列 {column!r}；实际列: {fieldnames}")

    grouped: dict[int, list[Measurement]] = {}
    for row in rows:
        frame = int(float(row[frame_col]))
        grouped.setdefault(frame, []).append(
            Measurement(0.0, np.array([float(row[x_col]), float(row[y_col])]))
        )

    origin = min(grouped) if frame_origin is None else int(frame_origin)
    frames: list[tuple[float, list[Measurement]]] = []
    for frame in sorted(grouped):
        timestamp = (frame - origin) * ts
        frames.append(
            (
                timestamp,
                [Measurement(timestamp, m.value, extra=m.extra) for m in grouped[frame]],
            )
        )
    return origin, frames


def load_truth(
    path: PathLike,
    *,
    frame_col: str = "frame",
    id_col: str = "id",
    x_col: str = "x",
    y_col: str = "y",
    ts: float = 0.1,
    frame_origin: Optional[int] = None,
) -> dict[float, list[tuple[int, np.ndarray]]]:
    """读取真值文件，返回 ``{timestamp: [(gt_id, position), ...]}``。"""
    fieldnames, rows = _read_rows(path)
    for column in (frame_col, id_col, x_col, y_col):
        if column not in fieldnames:
            raise ValueError(f"{path} 缺少列 {column!r}；实际列: {fieldnames}")

    grouped: dict[int, list[tuple[int, np.ndarray]]] = {}
    for row in rows:
        frame = int(float(row[frame_col]))
        grouped.setdefault(frame, []).append(
            (int(float(row[id_col])), np.array([float(row[x_col]), float(row[y_col])]))
        )

    origin = min(grouped) if frame_origin is None else int(frame_origin)
    return {
        (frame - origin) * ts: sorted(grouped[frame], key=lambda item: item[0])
        for frame in sorted(grouped)
    }


__all__ = ["load_measurements", "load_truth", "min_frame"]
