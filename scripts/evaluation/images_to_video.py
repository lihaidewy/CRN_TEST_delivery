#!/usr/bin/env python3
"""Convert ordered prediction visualisations into MP4 videos without altering images."""

from __future__ import annotations

import re
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# User-editable settings
# ---------------------------------------------------------------------------
# Add or edit a pair here for another evaluation run.
JOBS = (
    (
        Path("outputs/evaluation/S3M_moderate/visualizations"),
        Path("outputs/evaluation/S3M_moderate/S3M_moderate_1080p.mp4"),
    ),
    (
        Path("outputs/evaluation/S3S_heavy/visualizations"),
        Path("outputs/evaluation/S3S_heavy/S3S_heavy_1080p.mp4"),
    ),
)
FPS = 10
# 保留所有帧，缩放到 1080p，以压缩体积但不缩短视频时长。
FRAME_STRIDE = 1
OUTPUT_SIZE = (1920, 1080)  # (width, height); use None to retain source resolution.
OVERWRITE = False
# ---------------------------------------------------------------------------

FRAME_PATTERN = re.compile(r"^(?:predict_)?frame_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def collect_frames(input_dir: Path) -> list[tuple[int, Path]]:
    frames = []
    for path in input_dir.iterdir():
        match = FRAME_PATTERN.match(path.name)
        if match:
            frames.append((int(match.group(1)), path))
    return sorted(frames, key=lambda item: item[0])


def convert(input_dir: Path, output_video: Path) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if output_video.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output already exists: {output_video}\n"
            "Set OVERWRITE = True in this script to replace it."
        )

    frames = collect_frames(input_dir)
    if not frames:
        raise RuntimeError(f"No frame_<number>.jpg/png images found in {input_dir}")
    total_input_frames = len(frames)
    frames = frames[::FRAME_STRIDE]

    first = cv2.imread(str(frames[0][1]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read the first image: {frames[0][1]}")
    source_height, source_width = first.shape[:2]
    width, height = OUTPUT_SIZE if OUTPUT_SIZE is not None else (source_width, source_height)
    # mp4v is widely available through OpenCV and produces an MP4 readable by
    # PowerPoint and ordinary desktop players without requiring an ffmpeg CLI.
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot open the MP4 writer. Check OpenCV codec support.")

    try:
        expected_index = frames[0][0]
        for position, (frame_index, image_path) in enumerate(frames, start=1):
            if frame_index != expected_index:
                print(f"[warning] Frame index jump: expected {expected_index}, got {frame_index}")
            expected_index = frame_index + FRAME_STRIDE
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Cannot read image: {image_path}")
            if image.shape[:2] != (source_height, source_width):
                raise ValueError(
                    f"Image size mismatch at {image_path}: {image.shape[1]}x{image.shape[0]}, "
                    f"expected {source_width}x{source_height}."
                )
            if (source_width, source_height) != (width, height):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
            if position % 50 == 0 or position == len(frames):
                print(f"written {position}/{len(frames)} frames")
    finally:
        writer.release()

    duration = len(frames) / FPS
    print(f"[OK] video: {output_video}")
    print(f"input_frames={total_input_frames}, written_frames={len(frames)}, stride={FRAME_STRIDE}")
    print(f"fps={FPS}, size={width}x{height}, duration={duration:.1f}s")


def main() -> None:
    if FPS <= 0:
        raise ValueError("FPS must be positive.")
    if FRAME_STRIDE <= 0:
        raise ValueError("FRAME_STRIDE must be positive.")
    for input_dir, output_video in JOBS:
        print(f"\n[video] {input_dir} -> {output_video}")
        convert(input_dir, output_video)


if __name__ == "__main__":
    main()
