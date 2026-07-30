from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I2I_SRC = ROOT / "external" / "I2I" / "src"
sys.path.insert(0, str(I2I_SRC))

from cabinet_door_batch import CabinetDoorBatchConfig, run_cabinet_door_batch
from utils import load_dotenv_if_available


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate cabinet-door-open images with one Qwen bbox call and at most one Seedream call per image."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--vlm-model", default="qwen3.6-27b")
    parser.add_argument("--image-model", default="doubao-seedream-5-0-pro-260628")
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--preview-max-side", type=int, default=1280)
    parser.add_argument("--preview-jpeg-quality", type=int, default=82)
    parser.add_argument("--max-outside-mean-abs-diff", type=float, default=30.0)
    parser.add_argument("--max-outside-structure-change-ratio", type=float, default=0.12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv_if_available()
    config = CabinetDoorBatchConfig(
        image_dir=Path(args.image_dir),
        output_root=Path(args.output_root),
        manifest=Path(args.manifest) if args.manifest else None,
        vlm_model=args.vlm_model,
        image_model=args.image_model,
        min_confidence=args.min_confidence,
        preview_max_side=args.preview_max_side,
        preview_jpeg_quality=args.preview_jpeg_quality,
        max_outside_mean_abs_diff=args.max_outside_mean_abs_diff,
        max_outside_structure_change_ratio=args.max_outside_structure_change_ratio,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        limit=args.limit,
        workers=args.workers,
    )
    summary = run_cabinet_door_batch(config)
    print(
        "Cabinet-door generation summary: "
        f"total={summary['total']} accepted={summary['accepted']} rejected={summary['rejected']} "
        f"skipped={summary['skipped']} failed={summary['failed']} "
        f"qwen_calls={summary['qwen_call_count']} seedream_calls={summary['seedream_call_count']}"
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
