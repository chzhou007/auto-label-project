from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.asset_qc import run_asset_qc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run image/crop asset QC before or outside AutoLabelSample metadata.")
    parser.add_argument("--image-dir", action="append", default=None, help="Image directory or image file. Can repeat.")
    parser.add_argument("--crop-dir", action="append", default=None, help="Crop directory or crop file. Can repeat.")
    parser.add_argument("--manifest", default=None, help="Optional manifest CSV used to map assets back to samples.")
    parser.add_argument("--metadata-dir", default=None, help="Optional AutoLabelSample metadata directory used to map assets.")
    parser.add_argument("--output-dir", default="data/qc", help="Directory for asset QC report and CSV.")
    args = parser.parse_args()

    if not args.image_dir and not args.crop_dir:
        raise SystemExit("Provide --image-dir or --crop-dir")

    report = run_asset_qc(
        image_paths=args.image_dir,
        crop_paths=args.crop_dir,
        manifest_csv=args.manifest,
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
    )
    summary = report["summary"]
    print(
        "Asset QC summary: "
        f"assets={summary['total_assets']}, "
        f"kinds={summary['asset_kind_counts']}, "
        f"passed={summary['passed_assets']}, "
        f"failed={summary['failed_assets']}, "
        f"needs_human_review={summary['needs_human_review_assets']}, "
        f"issues={summary['issue_counts']}, "
        f"inventory_issues={summary.get('inventory_issue_counts', {})}"
    )
    print(f"Report: {report['report_path']}")
    print(f"Asset CSV: {report['asset_csv_path']}")
    return 1 if summary["failed_assets"] or summary.get("failed_inventory_checks", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
