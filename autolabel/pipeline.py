from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .adapters.classification_script import ClassificationJsonParseError
from .adapters.crop_reviewer import VLMCropReviewer, build_crop_review_config
from .adapters.detector_service import DetectorServiceClient
from .adapters.vlm_labelstudio_detector import VLMJsonParseError
from .adapters.i2i_generator import load_generated_samples
from .config_loader import load_config
from .contract_normalizer import normalize_autolabel_sample
from .cropper import attach_crops, cleanup_sample_crops
from .modules.classification import build_classification_module
from .sample_factory import make_sample, touch_workflow
from .utils import get_image_size, read_csv, write_json
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


def _positive_int(value: Any, default: int = 1) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"Expected a positive integer, got: {value!r}")
    return parsed


def _nonnegative_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"Expected a nonnegative number, got: {value!r}")
    return parsed


def _bool_config(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _box_width_height(obj: dict[str, Any]) -> tuple[int, int]:
    box = obj.get("box", {})
    return int(box.get("x2", 0)) - int(box.get("x1", 0)), int(box.get("y2", 0)) - int(box.get("y1", 0))


def _object_confidence(obj: dict[str, Any]) -> float | None:
    geometry_model = obj.get("geometry_model") or {}
    confidence = geometry_model.get("confidence")
    if confidence is None:
        return None
    return float(confidence)


def filter_detected_objects(objects: list[dict[str, Any]], direct_cfg: dict[str, Any], sample_id: str) -> list[dict[str, Any]]:
    min_confidence = _nonnegative_float(direct_cfg.get("min_box_confidence"), 0.0)
    min_box_width = _nonnegative_float(direct_cfg.get("min_box_width"), 1.0)
    min_box_height = _nonnegative_float(direct_cfg.get("min_box_height"), 1.0)
    min_box_area = _nonnegative_float(direct_cfg.get("min_box_area"), 1.0)
    max_box_aspect_ratio = _nonnegative_float(direct_cfg.get("max_box_aspect_ratio"), 0.0)

    kept: list[dict[str, Any]] = []
    dropped = 0
    for obj in objects:
        confidence = _object_confidence(obj)
        if confidence is not None and confidence < min_confidence:
            dropped += 1
            continue
        width, height = _box_width_height(obj)
        if width < min_box_width or height < min_box_height or width * height < min_box_area:
            dropped += 1
            continue
        if max_box_aspect_ratio and max(width / height, height / width) > max_box_aspect_ratio:
            dropped += 1
            continue
        kept.append(obj)

    if dropped:
        print(f"  !! {sample_id} 已过滤 {dropped} 个过小或低置信度检测框")
    return kept


MODEL_JSON_PARSE_ERRORS = (VLMJsonParseError, ClassificationJsonParseError)


def run_direct_pipeline(
    manifest_csv: str | Path,
    pipeline_config: dict[str, Any],
    detector_config: dict[str, Any],
    output_root: str | Path,
    classify: bool = True,
    batch_size: int | None = None,
    workers: int | None = None,
) -> list[Path]:
    rows = [row for row in read_csv(manifest_csv) if (row.get("task_mode") or "direct") == "direct"]
    direct_cfg = pipeline_config.get("direct_annotation", {})
    classification_cfg = pipeline_config.get("classification", {})
    crop_review_cfg = build_crop_review_config(pipeline_config, detector_config)
    batch_size = _positive_int(batch_size if batch_size is not None else direct_cfg.get("batch_size"), 1)
    workers = _positive_int(workers if workers is not None else direct_cfg.get("workers"), 1)
    json_retry_attempts = _positive_int(direct_cfg.get("json_retry_attempts"), 3)
    output = Path(output_root)
    metadata_dir = output / "metadata"
    crop_dir = output / "crops"
    thread_state = threading.local()

    def detector() -> DetectorServiceClient:
        client = getattr(thread_state, "detector", None)
        if client is None:
            client = DetectorServiceClient(detector_config)
            thread_state.detector = client
        return client

    def classifier() -> Any:
        if not classify:
            return None
        module = getattr(thread_state, "classifier", None)
        if module is None:
            module = build_classification_module(pipeline_config)
            thread_state.classifier = module
        return module

    def crop_reviewer() -> VLMCropReviewer | None:
        if not crop_review_cfg.get("enabled"):
            return None
        module = getattr(thread_state, "crop_reviewer", None)
        if module is None:
            module = VLMCropReviewer(crop_review_cfg)
            thread_state.crop_reviewer = module
        return module

    def make_base_sample(row: dict[str, Any], workflow_status: str = "raw") -> dict[str, Any]:
        _required_row_value(row, "sample_id")
        _required_row_value(row, "image_id")
        _required_row_value(row, "image_uri")
        _required_row_value(row, "source_type")
        width, height = _image_dimensions(row)
        return make_sample(
            row,
            width=width,
            height=height,
            pipeline_id=pipeline_config.get("pipeline_id", "autolabel_dag_v1"),
            pipeline_version=pipeline_config.get("pipeline_version", "0.1.0"),
            qc_policy=pipeline_config.get("qc_policy"),
            export_config=pipeline_config.get("export"),
            workflow_status=workflow_status,
        )

    def process_row(row: dict[str, Any]) -> Path:
        sample = make_base_sample(row, workflow_status="raw")
        task_key = row.get("task_key") or direct_cfg.get("default_task_key") or detector_config.get("default_task_key")
        sample["objects"] = filter_detected_objects(detector().detect(row["image_uri"], task_key), direct_cfg, sample["sample_id"])
        touch_workflow(sample, "boxed")

        if _bool_config(direct_cfg.get("cleanup_existing_crops"), default=True):
            cleanup_sample_crops(crop_dir, sample["sample_id"])
        attach_crops(sample, crop_dir, float(direct_cfg.get("crop_expand_ratio", 0.0)))
        touch_workflow(sample, "cropped")

        reviewer = crop_reviewer()
        if reviewer is not None:
            reviewer.review_sample(sample)

        classifier_module = classifier()
        if classifier_module is not None:
            skip_source_types = set(classification_cfg.get("skip_source_types", ["generated"]))
            classifier_module.classify_sample(sample, skip_source_types=skip_source_types)
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
        return output_path

    def handle_row(row: dict[str, Any], attempt: int) -> tuple[str, Path | None, dict[str, Any], int, Exception | None]:
        try:
            return "ok", process_row(row), row, attempt, None
        except MODEL_JSON_PARSE_ERRORS as exc:
            return "json_error", None, row, attempt, exc

    def remove_stale_retry_outputs(row: dict[str, Any]) -> None:
        sample_id = row.get("sample_id")
        if not sample_id:
            return
        for path in (
            metadata_dir / f"{sample_id}.json",
            output / "retry_failures" / f"{sample_id}.json",
        ):
            if path.exists():
                path.unlink()

    def handle_result(
        status: str,
        path: Path | None,
        row: dict[str, Any],
        attempt: int,
        exc: Exception | None,
        pending: list[tuple[dict[str, Any], int]],
    ) -> None:
        if status == "ok" and path is not None:
            written.append(path)
            return
        if status == "json_error" and exc is not None:
            sample_id = row.get("sample_id", "<missing_sample_id>")
            if attempt < json_retry_attempts:
                print(f"  !! {sample_id} JSON 不合法，进入重试队列 {attempt + 1}/{json_retry_attempts}: {exc}")
                pending.append((row, attempt + 1))
                return
            print(f"  !! {sample_id} 连续 {json_retry_attempts} 次 JSON 不合法，未写入最终 metadata: {exc}")
            remove_stale_retry_outputs(row)
            return
        raise RuntimeError(f"Unexpected direct pipeline row status: {status}")

    written: list[Path] = []
    pending: list[tuple[dict[str, Any], int]] = [(row, 1) for row in rows]
    if workers == 1:
        while pending:
            row_batch, pending = pending[:batch_size], pending[batch_size:]
            for row, attempt in row_batch:
                handle_result(*handle_row(row, attempt), pending=pending)
        return written

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while pending:
            row_batch, pending = pending[:batch_size], pending[batch_size:]
            futures = [executor.submit(handle_row, row, attempt) for row, attempt in row_batch]
            for future in as_completed(futures):
                handle_result(*future.result(), pending=pending)
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
