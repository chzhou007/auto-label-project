from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .constants import IMAGE_SUFFIXES, VIDEO_SUFFIXES
from .utils import get_image_size, slugify, write_csv


MANIFEST_FIELDS = [
    "sample_id",
    "image_id",
    "image_uri",
    "source_type",
    "task_mode",
    "task_key",
    "object_type",
    "anomaly_type",
    "site",
    "building",
    "floor",
    "room_name",
    "room_type",
    "task_group",
    "inspection_content",
    "collection_batch",
    "camera_id",
    "capture_time",
    "width",
    "height",
]


def _iter_files(root: str | Path, suffixes: set[str]) -> list[Path]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def _copy_image_to_sequence(image_path: Path, output_dir: Path, copy_images: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / image_path.name
    if copy_images:
        if image_path.resolve() != target.resolve():
            shutil.copy2(image_path, target)
        return target
    return image_path


def _extract_video_frames(
    video_path: Path,
    output_dir: Path,
    frame_stride: int,
    max_frames: int | None,
) -> list[Path]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required to extract video frames.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    written: list[Path] = []
    frame_idx = 0
    saved_idx = 0
    stem = slugify(video_path.stem)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % frame_stride == 0:
                target = output_dir / f"{stem}_frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(target), frame)
                written.append(target)
                saved_idx += 1
                if max_frames is not None and saved_idx >= max_frames:
                    break
            frame_idx += 1
    finally:
        capture.release()
    return written


def preprocess_raw_assets(
    raw_images: str | Path,
    raw_videos: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    copy_images: bool = True,
    video_frame_stride: int = 30,
    video_max_frames: int | None = None,
    default_source_type: str = "manual_upload",
) -> list[dict[str, Any]]:
    output = Path(output_dir)
    rows: list[dict[str, Any]] = []

    for image_path in _iter_files(raw_images, IMAGE_SUFFIXES):
        sequence_path = _copy_image_to_sequence(image_path, output, copy_images)
        width, height = get_image_size(sequence_path)
        image_id = slugify(sequence_path.stem)
        rows.append(
            {
                "sample_id": f"sample_{image_id}",
                "image_id": image_id,
                "image_uri": str(sequence_path),
                "source_type": default_source_type,
                "task_mode": "direct",
                "task_key": "",
                "object_type": "",
                "anomaly_type": "",
                "width": width,
                "height": height,
            }
        )

    for video_path in _iter_files(raw_videos, VIDEO_SUFFIXES):
        frame_dir = output / slugify(video_path.stem)
        for frame_path in _extract_video_frames(video_path, frame_dir, video_frame_stride, video_max_frames):
            width, height = get_image_size(frame_path)
            image_id = slugify(frame_path.stem)
            rows.append(
                {
                    "sample_id": f"sample_{image_id}",
                    "image_id": image_id,
                    "image_uri": str(frame_path),
                    "source_type": "cctv",
                    "task_mode": "direct",
                    "task_key": "",
                    "object_type": "",
                    "anomaly_type": "",
                    "width": width,
                    "height": height,
                }
            )

    write_csv(manifest_path, rows, MANIFEST_FIELDS)
    return rows
