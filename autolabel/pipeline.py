from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters.detector_service import DetectorServiceClient
from .adapters.i2i_generator import load_generated_samples
from .config_loader import load_config
from .contract_normalizer import normalize_autolabel_sample
from .cropper import attach_crops
from .modules.classification import build_classification_module
from .sample_factory import make_sample, touch_workflow
from .utils import get_image_size, read_csv, read_json, write_json
from .validators import validate_sample_contract


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    return load_config(path)


def _required_row_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not value:
        raise ValueError(f"manifest row is missing required field: {key}")
    return value


def _image_dimensions(row: dict[str, Any]) -> tuple[int, int]:
    if row.get("width") and row.get("height"):
        return int(row["width"]), int(row["height"])
    return get_image_size(_required_row_value(row, "image_uri"))


def run_direct_pipeline(
    manifest_csv: str | Path,
    pipeline_config: dict[str, Any],
    detector_config: dict[str, Any],
    output_root: str | Path,
    classify: bool = True,
) -> list[Path]:
    rows = [row for row in read_csv(manifest_csv) if (row.get("task_mode") or "direct") == "direct"]
    direct_cfg = pipeline_config.get("direct_annotation", {})
    classification_cfg = pipeline_config.get("classification", {})
    output = Path(output_root)
    metadata_dir = output / "metadata"
    crop_dir = output / "crops"
    detector = DetectorServiceClient(detector_config)
    classifier = None
    if classify:
        classifier = build_classification_module(pipeline_config)

    written: list[Path] = []
    for row in rows:
        _required_row_value(row, "sample_id")
        _required_row_value(row, "image_id")
        _required_row_value(row, "image_uri")
        _required_row_value(row, "source_type")
        width, height = _image_dimensions(row)

        sample = make_sample(
            row,
            width=width,
            height=height,
            pipeline_id=pipeline_config.get("pipeline_id", "autolabel_dag_v1"),
            pipeline_version=pipeline_config.get("pipeline_version", "0.1.0"),
            qc_policy=pipeline_config.get("qc_policy"),
            export_config=pipeline_config.get("export"),
            workflow_status="raw",
        )
        task_key = row.get("task_key") or direct_cfg.get("default_task_key") or detector_config.get("default_task_key")
        sample["objects"] = detector.detect(row["image_uri"], task_key)
        touch_workflow(sample, "boxed")

        attach_crops(sample, crop_dir, float(direct_cfg.get("crop_expand_ratio", 0.0)))
        touch_workflow(sample, "cropped")

        if classifier is not None:
            skip_source_types = set(classification_cfg.get("skip_source_types", ["generated"]))
            classifier.classify_sample(sample, skip_source_types=skip_source_types)
        else:
            touch_workflow(sample, "classified")

        sample = normalize_autolabel_sample(
            sample,
            pipeline_id=pipeline_config.get("pipeline_id"),
            pipeline_version=pipeline_config.get("pipeline_version"),
            qc_policy=pipeline_config.get("qc_policy"),
            export_config=pipeline_config.get("export"),
        )
        validate_sample_contract(sample)
        output_path = metadata_dir / f"{sample['sample_id']}.json"
        write_json(output_path, sample)
        written.append(output_path)
    return written


def ingest_generated_metadata(i2i_output_root: str | Path, metadata_dir: str | Path) -> list[Path]:
    target_dir = Path(metadata_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for sample in load_generated_samples(i2i_output_root, validate=False):
        if sample["image_asset"]["source_type"] != "generated":
            raise ValueError(f"Expected generated source_type: {sample['sample_id']}")
        sample = normalize_autolabel_sample(sample)
        validate_sample_contract(sample)
        output_path = target_dir / f"{sample['sample_id']}.json"
        write_json(output_path, sample)
        written.append(output_path)
    return written
