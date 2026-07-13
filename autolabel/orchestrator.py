from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .adapters.i2i_generator import iter_generated_metadata
from .adapters.detector_service import load_detector_config
from .exporters.labelstudio import export_metadata_dir
from .model_config import build_detector_runtime_config
from .modules.generation import build_generation_module
from .modules.generation.preflight import run_generation_preflight
from .pipeline import ingest_generated_metadata, run_direct_pipeline


def configure_processed_root(config: dict[str, Any], processed_root: str | Path | None) -> dict[str, Any]:
    if processed_root is None:
        return config
    root = Path(processed_root)
    paths = config.setdefault("paths", {})
    paths["metadata_dir"] = str(root / "metadata")
    paths["crop_dir"] = str(root / "crops")
    paths["i2i_output_dir"] = str(root / "i2i_outputs")
    paths["export_dir"] = str(root / "exports")
    paths["labelstudio_export"] = str(root / "exports" / "labelstudio" / "import.json")
    export_cfg = config.setdefault("export", {})
    quality_gate_cfg = export_cfg.get("generated_quality_gate")
    if isinstance(quality_gate_cfg, dict):
        quality_gate_cfg["rejected_report"] = str(root / "exports" / "labelstudio" / "rejected_generated_quality.json")
    return config


def apply_generation_run_overrides(
    config: dict[str, Any],
    *,
    vlm_model_key: str | None = None,
    image_model_key: str | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    generation_cfg = config.setdefault("generation", {})
    if vlm_model_key:
        generation_cfg["vlm_model_key"] = vlm_model_key
    if image_model_key:
        generation_cfg["image_model_key"] = image_model_key
    if workers is not None:
        generation_cfg["workers"] = int(workers)
    return config


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
    preflight: bool = False,
) -> int:
    paths = config.get("paths", {})
    generation_cfg = config.get("generation", {})
    module = build_generation_module(config)
    output_root = output_root or paths.get("i2i_output_dir") or "data/processed/i2i_outputs"
    tasks_csv = tasks_csv or default_manifest(config)
    image_root = image_root or default_image_root(config)
    if preflight:
        report = run_generation_preflight(
            config,
            tasks_csv=tasks_csv,
            image_root=image_root,
            output_root=output_root,
            limit=limit,
            require_credentials=not (dry_run or bool(generation_cfg.get("dry_run", False))),
        )
        if report.get("skipped"):
            print(
                f"Generation preflight: no generation rows found in {tasks_csv}; "
                f"manifest_rows={report.get('manifest_rows', 0)}"
            )
        else:
            model_pairs = [
                f"{anomaly}:{runtime.get('vlm_model_name')}->{runtime.get('image_model_name')}"
                for anomaly, runtime in sorted((report.get("runtime_by_anomaly") or {}).items())
            ]
            print(
                "Generation preflight: "
                f"generation_rows={report.get('generation_rows')}, "
                f"effective_rows={report.get('effective_generation_rows')}, "
                f"anomaly_types={','.join(report.get('anomaly_types', []))}, "
                f"models={';'.join(model_pairs)}, "
                f"i2i_entrypoint={report.get('i2i_entrypoint')}"
            )
    result = module.run(
        tasks_csv=tasks_csv,
        image_root=image_root,
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
    if getattr(result, "skipped", False):
        return 0
    generated_count = len(iter_generated_metadata(output_root))
    print(f"Generated metadata files found: {generated_count}")
    if ingest_metadata:
        metadata_dir = paths.get("metadata_dir", "data/processed/metadata")
        written = ingest_generated_metadata(
            output_root,
            metadata_dir,
            pipeline_config=config,
            tasks_csv=tasks_csv,
        )
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
    batch_size: int | None = None,
    workers: int | None = None,
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
        output_root=output_root,
        classify=classify,
        batch_size=batch_size,
        workers=workers,
    )


def run_labelstudio_export(
    config: dict[str, Any],
    metadata_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    update_samples: bool = False,
) -> int:
    paths = config.get("paths", {})
    export_cfg = config.get("export", {}) if isinstance(config.get("export"), dict) else {}
    quality_gate_cfg = (
        export_cfg.get("generated_quality_gate", {})
        if isinstance(export_cfg.get("generated_quality_gate"), dict)
        else {}
    )
    generated_quality_gate = bool(quality_gate_cfg.get("enabled", False))
    rejected_report_path = quality_gate_cfg.get("rejected_report")
    metadata_dir = metadata_dir or paths.get("metadata_dir", "data/processed/metadata")
    output_path = output_path or paths.get("labelstudio_export", "data/exports/labelstudio/import.json")
    tasks = export_metadata_dir(
        metadata_dir,
        output_path,
        update_samples=update_samples,
        generated_quality_gate=generated_quality_gate,
        rejected_report_path=rejected_report_path,
    )
    print(f"Wrote {len(tasks)} Label Studio tasks to {output_path}")
    if generated_quality_gate:
        report_path = rejected_report_path or Path(output_path).with_name("rejected_generated_quality.json")
        try:
            from .utils import read_json

            report = read_json(report_path)
            print(
                "Export quality gate: "
                f"generated={report.get('generated_samples', 0)}, "
                f"exportable={report.get('exported_generated_samples', 0)}, "
                f"rejected={report.get('rejected_samples', 0)}"
            )
        except Exception:
            print(f"Export quality gate report path: {report_path}")
    return 0
