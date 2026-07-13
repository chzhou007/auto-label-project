from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.modules.generation import build_generation_module
from autolabel.modules.generation.preflight import run_generation_preflight
from autolabel.orchestrator import apply_generation_run_overrides, configure_processed_root
from autolabel.pipeline import ingest_generated_metadata, load_pipeline_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the external I2I generation branch.")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--generation-vlm-model-key", default=None)
    parser.add_argument("--generation-image-model-key", default=None)
    parser.add_argument("--generation-workers", type=int, default=None)
    parser.add_argument("--pipeline-config", default="configs/autolabel.yaml")
    parser.add_argument("--i2i-project", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ingest-metadata-dir", default=None)
    args = parser.parse_args()

    config = load_pipeline_config(args.pipeline_config)
    configure_processed_root(config, args.processed_root)
    apply_generation_run_overrides(
        config,
        vlm_model_key=args.generation_vlm_model_key,
        image_model_key=args.generation_image_model_key,
        workers=args.generation_workers,
    )
    paths = config.get("paths", {})
    generation = config.get("generation", {})
    if args.i2i_project:
        config.setdefault("modules", {}).setdefault("generation", {}).setdefault("backends", {}).setdefault(
            "i2i_external", {}
        )["project_dir"] = args.i2i_project
    tasks = args.tasks or paths.get("image_sequence_manifest") or "data/staging/image_sequence/manifest.csv"
    image_root = args.image_root or paths.get("image_sequence_dir") or "data/staging/image_sequence"
    output_root = args.output_root or paths.get("i2i_output_dir") or "data/processed/i2i_outputs"
    preflight = run_generation_preflight(
        config,
        tasks_csv=tasks,
        image_root=image_root,
        output_root=output_root,
        limit=args.limit,
        require_credentials=not (args.dry_run or bool(generation.get("dry_run", False))),
    )
    if preflight.get("skipped"):
        print(f"Generation preflight: no generation rows found in {tasks}")
    else:
        model_pairs = [
            f"{anomaly}:{runtime.get('vlm_model_name')}->{runtime.get('image_model_name')}"
            for anomaly, runtime in sorted((preflight.get("runtime_by_anomaly") or {}).items())
        ]
        print(
            "Generation preflight: "
            f"generation_rows={preflight.get('generation_rows')}, "
            f"effective_rows={preflight.get('effective_generation_rows')}, "
            f"anomaly_types={','.join(preflight.get('anomaly_types', []))}, "
            f"models={';'.join(model_pairs)}"
        )
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

    ingest_metadata_dir = args.ingest_metadata_dir or (paths.get("metadata_dir") if args.processed_root else None)
    if ingest_metadata_dir:
        written = ingest_generated_metadata(
            output_root,
            ingest_metadata_dir,
            pipeline_config=config,
            tasks_csv=tasks,
        )
        print(f"Ingested {len(written)} generated metadata files into {ingest_metadata_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
