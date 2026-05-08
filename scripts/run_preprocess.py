from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.config_loader import load_config
from autolabel.preprocess import preprocess_raw_assets


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess raw images/videos into an image sequence manifest.")
    parser.add_argument("--config", default="configs/autolabel.yaml")
    parser.add_argument("--raw-images", default=None)
    parser.add_argument("--raw-videos", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--video-frame-stride", type=int, default=None)
    parser.add_argument("--video-max-frames", type=int, default=None)
    parser.add_argument("--video-decode-mode", choices=["cpu", "gpu", "auto"], default=None)
    parser.add_argument("--ffmpeg-path", default=None)
    parser.add_argument("--video-gpu-hwaccel", default=None)
    parser.add_argument("--no-video-gpu-fallback", action="store_true")
    parser.add_argument("--no-copy-images", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})
    preprocess = config.get("preprocess", {})

    rows = preprocess_raw_assets(
        raw_images=args.raw_images or paths.get("raw_images_dir", "data/raw/images"),
        raw_videos=args.raw_videos or paths.get("raw_videos_dir", "data/raw/videos"),
        output_dir=args.output or paths.get("image_sequence_dir", "data/staging/image_sequence"),
        manifest_path=args.manifest or paths.get("image_sequence_manifest", "data/staging/image_sequence/manifest.csv"),
        copy_images=False if args.no_copy_images else bool(preprocess.get("copy_images", True)),
        video_frame_stride=args.video_frame_stride or int(preprocess.get("video_frame_stride", 30)),
        video_max_frames=args.video_max_frames if args.video_max_frames is not None else preprocess.get("video_max_frames"),
        video_decode_mode=args.video_decode_mode or preprocess.get("video_decode_mode", "cpu"),
        ffmpeg_path=args.ffmpeg_path or preprocess.get("ffmpeg_path", "ffmpeg"),
        video_gpu_hwaccel=args.video_gpu_hwaccel or preprocess.get("video_gpu_hwaccel", "cuda"),
        video_gpu_fallback_to_cpu=(
            False if args.no_video_gpu_fallback else bool(preprocess.get("video_gpu_fallback_to_cpu", True))
        ),
        default_source_type=preprocess.get("default_source_type", "manual_upload"),
    )
    manifest = args.manifest or paths.get("image_sequence_manifest", "data/staging/image_sequence/manifest.csv")
    print(f"Wrote {len(rows)} image sequence rows to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
