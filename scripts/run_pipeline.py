from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.config_loader import load_config
from autolabel.orchestrator import run_direct_branch, run_generation_branch, run_labelstudio_export


def parse_branches(value: str) -> list[str]:
    branches = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"generation", "direct", "export"}
    invalid = sorted(set(branches) - allowed)
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported branches: {invalid}")
    return branches


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AutoLabel project DAG through configured modules.")
    parser.add_argument("--config", default="configs/autolabel.yaml")
    parser.add_argument("--branches", type=parse_branches, default=["generation", "direct"])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--generation-output-root", default=None)
    parser.add_argument("--detector-config", default=None)
    parser.add_argument("--dry-run-generation", action="store_true")
    parser.add_argument("--skip-existing-generation", action="store_true")
    parser.add_argument("--generation-limit", type=int, default=None)
    parser.add_argument("--no-ingest-generated", action="store_true")
    parser.add_argument("--no-classify", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of manifest rows submitted per direct batch.")
    parser.add_argument("--workers", type=int, default=None, help="Concurrent workers for direct detection/crop/classification.")
    parser.add_argument("--dry-run-geometry", action="store_true")
    parser.add_argument("--dry-run-classification", action="store_true")
    parser.add_argument("--dry-run-models", action="store_true")
    parser.add_argument("--export-output", default=None)
    parser.add_argument("--update-export-status", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if "generation" in args.branches and config.get("generation", {}).get("enabled", True):
        code = run_generation_branch(
            config,
            tasks_csv=args.manifest,
            image_root=args.image_root,
            output_root=args.generation_output_root,
            dry_run=args.dry_run_generation,
            skip_existing=args.skip_existing_generation,
            limit=args.generation_limit,
            ingest_metadata=not args.no_ingest_generated,
        )
        if code != 0:
            return code

    if "direct" in args.branches and config.get("direct_annotation", {}).get("enabled", True):
        written = run_direct_branch(
            config,
            manifest_csv=args.manifest,
            output_root=args.processed_root,
            detector_config_path=args.detector_config,
            classify=not args.no_classify,
            dry_run_geometry=args.dry_run_geometry or args.dry_run_models,
            dry_run_classification=args.dry_run_classification or args.dry_run_models,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        print(f"Wrote {len(written)} direct AutoLabelSample files.")
        for path in written:
            print(path)

    if "export" in args.branches:
        return run_labelstudio_export(
            config,
            output_path=args.export_output,
            update_samples=args.update_export_status,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
