from __future__ import annotations

import shutil
import subprocess
import tempfile
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


def estimate_extracted_frame_count(
    total_frames: int,
    frame_stride: int,
    max_frames: int | None = None,
) -> int:
    if total_frames <= 0:
        return 0
    if frame_stride <= 0:
        raise ValueError("video_frame_stride must be greater than 0")
    count = ((total_frames - 1) // frame_stride) + 1
    if max_frames is not None:
        count = min(count, max_frames)
    return count


def _extract_video_frames_cpu(
    video_path: Path,
    output_dir: Path,
    frame_stride: int,
    max_frames: int | None,
) -> list[Path]:
    if frame_stride <= 0:
        raise ValueError("video_frame_stride must be greater than 0")

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


def _extract_video_frames_gpu_ffmpeg(
    video_path: Path,
    output_dir: Path,
    frame_stride: int,
    max_frames: int | None,
    ffmpeg_path: str = "ffmpeg",
    hwaccel: str = "cuda",
) -> list[Path]:
    if frame_stride <= 0:
        raise ValueError("video_frame_stride must be greater than 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(video_path.stem)
    with tempfile.TemporaryDirectory(prefix="autolabel_ffmpeg_frames_") as tmpdir:
        tmp_path = Path(tmpdir)
        pattern = tmp_path / "frame_%06d.jpg"
        command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-hwaccel",
            hwaccel,
            "-i",
            str(video_path),
            "-vf",
            f"select=not(mod(n\\,{frame_stride}))",
            "-vsync",
            "vfr",
            "-q:v",
            "2",
        ]
        if max_frames is not None:
            command += ["-frames:v", str(max_frames)]
        command.append(str(pattern))

        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"FFmpeg GPU frame extraction failed: {detail}")

        written: list[Path] = []
        for saved_idx, frame_path in enumerate(sorted(tmp_path.glob("frame_*.jpg"))):
            frame_idx = saved_idx * frame_stride
            target = output_dir / f"{stem}_frame_{frame_idx:06d}.jpg"
            shutil.move(str(frame_path), target)
            written.append(target)
        return written


def _extract_video_frames(
    video_path: Path,
    output_dir: Path,
    frame_stride: int,
    max_frames: int | None,
    decode_mode: str = "cpu",
    ffmpeg_path: str = "ffmpeg",
    gpu_hwaccel: str = "cuda",
    gpu_fallback_to_cpu: bool = True,
) -> list[Path]:
    mode = (decode_mode or "cpu").lower()
    if mode not in {"cpu", "gpu", "auto"}:
        raise ValueError("video_decode_mode must be one of: cpu, gpu, auto")
    if mode in {"gpu", "auto"}:
        try:
            return _extract_video_frames_gpu_ffmpeg(
                video_path=video_path,
                output_dir=output_dir,
                frame_stride=frame_stride,
                max_frames=max_frames,
                ffmpeg_path=ffmpeg_path,
                hwaccel=gpu_hwaccel,
            )
        except Exception as exc:
            if mode == "gpu" and not gpu_fallback_to_cpu:
                raise
            print(f"  !! GPU 视频解析不可用，回退 CPU: {exc}")
    return _extract_video_frames_cpu(video_path, output_dir, frame_stride, max_frames)


def preprocess_raw_assets(
    raw_images: str | Path,
    raw_videos: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    copy_images: bool = True,
    video_frame_stride: int = 30,
    video_max_frames: int | None = None,
    video_decode_mode: str = "cpu",
    ffmpeg_path: str = "ffmpeg",
    video_gpu_hwaccel: str = "cuda",
    video_gpu_fallback_to_cpu: bool = True,
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
        for frame_path in _extract_video_frames(
            video_path=video_path,
            output_dir=frame_dir,
            frame_stride=video_frame_stride,
            max_frames=video_max_frames,
            decode_mode=video_decode_mode,
            ffmpeg_path=ffmpeg_path,
            gpu_hwaccel=video_gpu_hwaccel,
            gpu_fallback_to_cpu=video_gpu_fallback_to_cpu,
        ):
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
