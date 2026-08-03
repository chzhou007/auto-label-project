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
        description=(
            "Generate one-to-one closed/open electrical-cabinet pairs from segmentation candidates "
            "with at most one Seedream call per selected image."
        )
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--priority-image-dir",
        default=None,
        help="Use filenames from a review/annotation directory to prioritize matching clean source images.",
    )
    parser.add_argument(
        "--priority-only",
        action="store_true",
        help="Process only clean source images whose filenames appear in --priority-image-dir.",
    )
    parser.add_argument(
        "--reviewed-all-close",
        action="store_true",
        help=(
            "Treat reviewed false-positive OPEN predictions as manual CLOSE targets. "
            "Prioritize fused_open boxes and bypass room/type/open-state appearance filters."
        ),
    )
    parser.add_argument(
        "--selector-backend",
        choices=["segmentation_candidates", "vlm"],
        default="segmentation_candidates",
        help="Default reads existing door segmentation/classification CSVs and makes no VLM calls.",
    )
    parser.add_argument("--candidate-predictions-csv", default=None)
    parser.add_argument("--image-predictions-csv", default=None)
    parser.add_argument(
        "--allowed-room-type-regex",
        default=(
            r"高压配电室|低压配电室|发电机配电房|电池室|精密空调间|冷冻站|"
            r"水设备间|蓄水泵房"
        ),
        help="Eligible equipment rooms; rack-centric module/OAR/ELV rooms remain excluded.",
    )
    parser.add_argument("--minimum-segmentation-confidence", type=float, default=0.50)
    parser.add_argument("--maximum-closed-probability", type=float, default=0.20)
    parser.add_argument("--maximum-not-door-probability", type=float, default=0.05)
    parser.add_argument("--minimum-bbox-area-ratio", type=float, default=0.002)
    parser.add_argument(
        "--maximum-bbox-area-ratio",
        type=float,
        default=0.10,
        help="Reject merged cabinet-bank masks; Seedream must receive one compact door target.",
    )
    parser.add_argument("--minimum-bbox-short-side", type=int, default=50)
    parser.add_argument(
        "--maximum-bbox-edge-density",
        type=float,
        default=0.12,
        help="Reject open shelves, grilles, and highly textured rack-like false positives.",
    )
    parser.add_argument(
        "--maximum-dark-neutral-ratio",
        type=float,
        default=0.45,
        help="Reject dark, low-saturation perforated machine/server cabinet candidates.",
    )
    parser.add_argument(
        "--maximum-existing-open-overlap-ratio",
        type=float,
        default=0.05,
        help="Reject a close crop when it would include an already-open door candidate.",
    )
    parser.add_argument(
        "--vlm-provider",
        choices=["volcengine_ark_responses", "qwen_openai_compatible"],
        default="volcengine_ark_responses",
    )
    parser.add_argument("--vlm-model", default="doubao-seed-2-1-pro-260628")
    parser.add_argument("--vlm-endpoint", default=None)
    parser.add_argument("--vlm-api-key-env", default="ARK_API_KEY")
    parser.add_argument("--vlm-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--vlm-max-retries",
        type=int,
        default=0,
        help="Retries after transport errors. Keep 0 to avoid duplicate billing after an ambiguous timeout.",
    )
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
        priority_image_dir=Path(args.priority_image_dir) if args.priority_image_dir else None,
        priority_only=args.priority_only,
        reviewed_all_close=args.reviewed_all_close,
        selector_backend=args.selector_backend,
        candidate_predictions_csv=(
            Path(args.candidate_predictions_csv) if args.candidate_predictions_csv else None
        ),
        image_predictions_csv=(
            Path(args.image_predictions_csv) if args.image_predictions_csv else None
        ),
        allowed_room_type_regex=args.allowed_room_type_regex,
        minimum_segmentation_confidence=args.minimum_segmentation_confidence,
        maximum_closed_probability=args.maximum_closed_probability,
        maximum_not_door_probability=args.maximum_not_door_probability,
        minimum_bbox_area_ratio=args.minimum_bbox_area_ratio,
        maximum_bbox_area_ratio=args.maximum_bbox_area_ratio,
        minimum_bbox_short_side=args.minimum_bbox_short_side,
        maximum_bbox_edge_density=args.maximum_bbox_edge_density,
        maximum_dark_neutral_ratio=args.maximum_dark_neutral_ratio,
        maximum_existing_open_overlap_ratio=args.maximum_existing_open_overlap_ratio,
        vlm_provider=args.vlm_provider,
        vlm_model=args.vlm_model,
        vlm_endpoint=args.vlm_endpoint,
        vlm_api_key_env=args.vlm_api_key_env,
        vlm_timeout_seconds=args.vlm_timeout_seconds,
        vlm_max_retries=args.vlm_max_retries,
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
        f"selector_backend={summary['selector_backend']} "
        f"selector_calls={summary['selector_call_count']} seedream_calls={summary['seedream_call_count']}"
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
