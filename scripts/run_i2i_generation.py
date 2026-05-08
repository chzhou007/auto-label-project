from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.modules.generation import build_generation_module
from autolabel.pipeline import ingest_generated_metadata, load_pipeline_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the external I2I generation branch.")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--pipeline-config", default="configs/autolabel.yaml")
    parser.add_argument("--i2i-project", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ingest-metadata-dir", default=None)
    args = parser.parse_args()

    config = load_pipeline_config(args.pipeline_config)
    paths = config.get("paths", {})
    generation = config.get("generation", {})
    if args.i2i_project:
        config.setdefault("modules", {}).setdefault("generation", {}).setdefault("backends", {}).setdefault(
            "i2i_external", {}
        )["project_dir"] = args.i2i_project
    tasks = args.tasks or paths.get("image_sequence_manifest") or "data/staging/image_sequence/manifest.csv"
    image_root = args.image_root or paths.get("image_sequence_dir") or "data/staging/image_sequence"
    output_root = args.output_root or paths.get("i2i_output_dir") or "data/processed/i2i_outputs"
    module = build_generation_module(config)
    result = module.run(
        tasks_csv=tasks,
        image_root=image_root,
        output_root=output_root,
        dry_run=args.dry_run or bool(generation.get("dry_run", False)),
        skip_existing=args.skip_existing or bool(generation.get("skip_existing", False)),
        limit=args.limit,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        return result.returncode

    if args.ingest_metadata_dir:
        written = ingest_generated_metadata(output_root, args.ingest_metadata_dir)
        print(f"Ingested {len(written)} generated metadata files into {args.ingest_metadata_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
