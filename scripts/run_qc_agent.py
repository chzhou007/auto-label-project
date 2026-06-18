from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.config_loader import load_config
from autolabel.qc_agent import run_qc_agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run post-labeling QC agent for AutoLabelSample metadata.")
    parser.add_argument("--config", default="configs/autolabel.yaml", help="Pipeline YAML/JSON config path.")
    parser.add_argument("--metadata-dir", default=None, help="Directory containing AutoLabelSample JSON files.")
    parser.add_argument("--sample", action="append", default=None, help="Single metadata JSON path. Can be repeated.")
    parser.add_argument("--output-dir", default=None, help="Directory for QC report and manual review queue.")
    parser.add_argument(
        "--asset-base-dir",
        action="append",
        default=None,
        help="Extra base directory used to resolve relative image/crop/mask paths. Can be repeated.",
    )
    parser.add_argument("--sampling-ratio", type=float, default=None, help="Passed-sample manual sampling ratio.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of metadata files for smoke tests.")
    parser.add_argument("--enable-vlm", action="store_true", help="Enable VLM semantic QC for annotation boxes.")
    parser.add_argument("--disable-vlm", action="store_true", help="Force-disable VLM semantic QC.")
    args = parser.parse_args()

    if not args.metadata_dir and not args.sample:
        config = load_config(args.config)
        args.metadata_dir = config.get("paths", {}).get("metadata_dir", "data/processed/metadata")
    else:
        config = load_config(args.config)

    enable_vlm = None
    if args.enable_vlm and args.disable_vlm:
        raise SystemExit("--enable-vlm and --disable-vlm cannot be used together")
    if args.enable_vlm:
        enable_vlm = True
    if args.disable_vlm:
        enable_vlm = False

    report = run_qc_agent(
        pipeline_config=config,
        metadata_dir=args.metadata_dir,
        sample_paths=args.sample,
        output_dir=args.output_dir,
        asset_base_dirs=args.asset_base_dir,
        sampling_ratio=args.sampling_ratio,
        enable_vlm=enable_vlm,
        limit=args.limit,
    )
    summary = report["summary"]
    print(
        "QC summary: "
        f"samples={summary['total_samples']}, "
        f"objects={summary['total_objects']}, "
        f"passed={summary['passed_samples']}, "
        f"failed={summary['failed_samples']}, "
        f"needs_human_review={summary['needs_human_review_samples']}, "
        f"manual_queue={summary['manual_review_queue_size']}"
    )
    print(f"Report: {report['report_path']}")
    print(f"Manual review queue: {report['manual_review_queue_path']}")
    return 1 if summary["failed_samples"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
