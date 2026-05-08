from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .adapters.detector_service import load_detector_config
from .exporters.labelstudio import export_metadata_dir
from .model_config import build_detector_runtime_config
from .modules.generation import build_generation_module
from .pipeline import ingest_generated_metadata, run_direct_pipeline


def default_manifest(config: dict[str, Any]) -> str:
    return config.get("paths", {}).get("image_sequence_manifest") or "data/staging/image_sequence/manifest.csv"


def default_image_root(config: dict[str, Any]) -> str:
    return config.get("paths", {}).get("image_sequence_dir") or "data/staging/image_sequence"


def default_processed_root(config: dict[str, Any]) -> str:
    return str(Path(config.get("paths", {}).get("metadata_dir", "data/processed/metadata")).parent)


def run_generation_branch(
    config: dict[str, Any],
    tasks_csv: str | Path | None = None,
    image_root: str | Path | None = None,
    output_root: str | Path | None = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    limit: int | None = None,
    ingest_metadata: bool = True,
) -> int:
    paths = config.get("paths", {})
    generation_cfg = config.get("generation", {})
    module = build_generation_module(config)
    output_root = output_root or paths.get("i2i_output_dir") or "data/processed/i2i_outputs"
    result = module.run(
        tasks_csv=tasks_csv or default_manifest(config),
        image_root=image_root or default_image_root(config),
        output_root=output_root,
        dry_run=dry_run or bool(generation_cfg.get("dry_run", False)),
        skip_existing=skip_existing or bool(generation_cfg.get("skip_existing", False)),
        limit=limit,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        return result.returncode
    if ingest_metadata:
        metadata_dir = paths.get("metadata_dir", "data/processed/metadata")
        written = ingest_generated_metadata(output_root, metadata_dir)
        print(f"Ingested {len(written)} generated AutoLabelSample files into {metadata_dir}")
    return 0


def run_direct_branch(
    config: dict[str, Any],
    manifest_csv: str | Path | None = None,
    output_root: str | Path | None = None,
    detector_config_path: str | Path | None = None,
    classify: bool = True,
    dry_run_geometry: bool = False,
    dry_run_classification: bool = False,
) -> list[Path]:
    if dry_run_classification:
        config = dict(config)
        modules = dict(config.get("modules", {}))
        classification_module = dict(modules.get("classification", {}))
        classification_module["backend"] = "dry_run_rule"
        modules["classification"] = classification_module
        config["modules"] = modules

    detector_config = (
        load_detector_config(detector_config_path)
        if detector_config_path
        else build_detector_runtime_config(config)
    )
    if dry_run_geometry:
        for service in detector_config.get("services", {}).values():
            service["dry_run"] = True
    return run_direct_pipeline(
        manifest_csv=manifest_csv or default_manifest(config),
        pipeline_config=config,
        detector_config=detector_config,
        output_root=output_root or default_processed_root(config),
        classify=classify,
    )


def run_labelstudio_export(
    config: dict[str, Any],
    metadata_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    update_samples: bool = False,
) -> int:
    paths = config.get("paths", {})
    metadata_dir = metadata_dir or paths.get("metadata_dir", "data/processed/metadata")
    output_path = output_path or paths.get("labelstudio_export", "data/exports/labelstudio/import.json")
    tasks = export_metadata_dir(metadata_dir, output_path, update_samples=update_samples)
    print(f"Wrote {len(tasks)} Label Studio tasks to {output_path}")
    return 0
