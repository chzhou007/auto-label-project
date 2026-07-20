from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from ...cropper import attach_crops, cleanup_sample_crops
from ...sample_factory import make_box, touch_workflow
from ...utils import ensure_dir, relpath_if_possible, resolve_path
from ...model_config import resolve_localizer_config
from .benchmark import compute_localizer_metrics
from .localizers import create_localizer
from .quality import build_quality_failure, evaluate_localization_quality


def _as_box_list(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return [int(float(item)) for item in value]
        except (TypeError, ValueError):
            return None
    return None


def _deep_get(mapping: dict[str, Any], path: Iterable[str]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def resolve_prompt_box(obj: dict[str, Any]) -> list[int] | None:
    generation_params = ((obj.get("geometry_detail") or {}).get("generation_params") or {})
    candidate_paths = [
        ("localizer", "prompt_box"),
        ("localizer", "grid_bbox"),
        ("prompt_box",),
        ("grid_bbox",),
        ("expanded_edit_bbox",),
        ("edit_bbox",),
        ("generation", "prompt_box"),
        ("generation", "grid_bbox"),
    ]
    for path in candidate_paths:
        box = _as_box_list(_deep_get(generation_params, path))
        if box is not None:
            return box

    wan_boxes = _deep_get(generation_params, ("wan", "bbox_list"))
    if isinstance(wan_boxes, list) and wan_boxes:
        box = _as_box_list(wan_boxes[0])
        if box is not None:
            return box

    return _as_box_list([
        obj["box"]["x1"],
        obj["box"]["y1"],
        obj["box"]["x2"],
        obj["box"]["y2"],
    ])


def resolve_original_image_uri(sample: dict[str, Any], obj: dict[str, Any], source_row: dict[str, Any] | None) -> str | None:
    if source_row and source_row.get("image_uri"):
        return str(source_row["image_uri"])

    generation_params = ((obj.get("geometry_detail") or {}).get("generation_params") or {})
    candidate_paths = [
        ("original_image_uri",),
        ("original_image_path",),
        ("source_image_uri",),
        ("source_image_path",),
        ("input_image_uri",),
        ("input_image_path",),
        ("clean_image_uri",),
        ("clean_image_path",),
        ("generation", "original_image_uri"),
        ("generation", "original_image_path"),
        ("wan", "source_image_uri"),
        ("wan", "source_image_path"),
    ]
    for path in candidate_paths:
        value = _deep_get(generation_params, path)
        if isinstance(value, str) and value:
            return value
    return None


def resolve_anomaly_type(source_row: dict[str, Any] | None, obj: dict[str, Any]) -> str:
    if source_row and source_row.get("anomaly_type"):
        return str(source_row["anomaly_type"])
    for label in (obj.get("classification") or {}).get("multi_labels", []):
        if label.get("label_key") == "anomaly_type" and label.get("label_value"):
            return str(label["label_value"])
    return "unknown"


def build_localizer_kwargs(localizer_name: str, localizer_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        # Benchmark runs also need debug artifacts for post-hoc review.
        "debug": bool(localizer_cfg.get("debug", False) or localizer_cfg.get("benchmark", False)),
    }
    pgcd_cfg = localizer_cfg.get("pgcd", {}) if isinstance(localizer_cfg.get("pgcd"), dict) else {}
    if localizer_name in {"pgcd_lpips", "pgcd_lpips_sam2"}:
        if "min_component_area" in pgcd_cfg:
            kwargs["min_component_area"] = int(pgcd_cfg["min_component_area"])
        if "threshold" in pgcd_cfg:
            kwargs["threshold_mode"] = pgcd_cfg["threshold"]
        if "lpips_backbone" in pgcd_cfg:
            kwargs["lpips_backbone"] = str(pgcd_cfg["lpips_backbone"])
        if "lpips_input_max_side" in pgcd_cfg:
            kwargs["lpips_input_max_side"] = int(pgcd_cfg["lpips_input_max_side"])
        if "heatmap_normalization" in pgcd_cfg:
            kwargs["heatmap_normalization"] = str(pgcd_cfg["heatmap_normalization"])
        if "heatmap_percentile_low" in pgcd_cfg:
            kwargs["heatmap_percentile_low"] = float(pgcd_cfg["heatmap_percentile_low"])
        if "heatmap_percentile_high" in pgcd_cfg:
            kwargs["heatmap_percentile_high"] = float(pgcd_cfg["heatmap_percentile_high"])
        if isinstance(pgcd_cfg.get("prompt_prior_weights"), dict):
            kwargs["prompt_prior_weights"] = deepcopy(pgcd_cfg["prompt_prior_weights"])
    if localizer_name == "rgb_diff":
        if "min_component_area" in pgcd_cfg:
            kwargs["min_component_area"] = int(pgcd_cfg["min_component_area"])
        if "max_global_change_ratio" in pgcd_cfg:
            kwargs["max_global_change_ratio"] = float(pgcd_cfg["max_global_change_ratio"])
    if localizer_name == "pgcd_lpips_sam2":
        sam2_cfg = localizer_cfg.get("sam2", {}) if isinstance(localizer_cfg.get("sam2"), dict) else {}
        kwargs["sam2_required"] = bool(sam2_cfg.get("required", False))
        if sam2_cfg.get("model"):
            kwargs["sam2_model"] = sam2_cfg["model"]
    return kwargs


def _normalize_localizer_names(value: Any) -> list[str]:
    if value in ("", None, False):
        return []
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized.lower() in {"false", "none", "null"}:
            return []
        return [item.strip() for item in normalized.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        names: list[str] = []
        for item in value:
            for name in _normalize_localizer_names(item):
                if name not in names:
                    names.append(name)
        return names
    return [str(value)]


def _ensure_localizer_block(obj: dict[str, Any]) -> dict[str, Any]:
    geometry_detail = obj.setdefault("geometry_detail", {})
    generation_params = geometry_detail.get("generation_params")
    if not isinstance(generation_params, dict):
        generation_params = {}
        geometry_detail["generation_params"] = generation_params
    localizer_block = generation_params.get("localizer")
    if not isinstance(localizer_block, dict):
        localizer_block = {}
        generation_params["localizer"] = localizer_block
    return localizer_block


def _resolve_image_size(sample: dict[str, Any]) -> tuple[int, int] | None:
    image_asset = sample.get("image_asset", {})
    width = image_asset.get("width")
    height = image_asset.get("height")
    if width in (None, "") or height in (None, ""):
        return None
    return (int(width), int(height))


def _resolve_reference_bbox(obj: dict[str, Any], prompt_box: list[int] | None) -> list[int] | None:
    generation_params = ((obj.get("geometry_detail") or {}).get("generation_params") or {})
    candidate_paths = [
        ("benchmark_bbox",),
        ("ground_truth_bbox",),
        ("reference_bbox",),
        ("localizer", "benchmark_bbox"),
        ("localizer", "ground_truth_bbox"),
        ("localizer", "reference_bbox"),
    ]
    for path in candidate_paths:
        box = _as_box_list(_deep_get(generation_params, path))
        if box is not None:
            return box

    box = obj.get("box")
    if isinstance(box, dict):
        xyxy = _as_box_list([box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")])
        if xyxy is not None:
            return xyxy
    return prompt_box


def _resolve_reference_mask_uri(obj: dict[str, Any]) -> str | None:
    geometry_detail = obj.get("geometry_detail") or {}
    generation_params = geometry_detail.get("generation_params") or {}
    candidate_paths = [
        ("benchmark_mask_uri",),
        ("ground_truth_mask_uri",),
        ("reference_mask_uri",),
        ("localizer", "benchmark_mask_uri"),
        ("localizer", "ground_truth_mask_uri"),
        ("localizer", "reference_mask_uri"),
    ]
    for path in candidate_paths:
        value = _deep_get(generation_params, path)
        if isinstance(value, str) and value:
            return value
    mask_uri = geometry_detail.get("mask_uri")
    return str(mask_uri) if isinstance(mask_uri, str) and mask_uri else None


def _relative_path_string(value: str, base_dir: str | Path | None = None) -> str:
    if base_dir is None or not value:
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return relpath_if_possible(candidate, base_dir)
    return value


def _serialize_metadata_value(value: Any, relative_to: str | Path | None = None) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize_metadata_value(item, relative_to=relative_to) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_metadata_value(item, relative_to=relative_to) for item in value]
    if isinstance(value, Path):
        return _relative_path_string(str(value), relative_to)
    if isinstance(value, str):
        return _relative_path_string(value, relative_to)
    return value


def _serialize_mapping(mapping: dict[str, Any] | None, relative_to: str | Path | None = None) -> dict[str, Any]:
    return {
        str(key): _serialize_metadata_value(value, relative_to=relative_to)
        for key, value in (mapping or {}).items()
    }


def _trusted_i2i_quality(generation_params: dict[str, Any], anomaly_type: str) -> dict[str, Any] | None:
    if bool(generation_params.get("coarse_bbox_requires_postprocess")):
        return None
    final_bbox_source = str(generation_params.get("final_bbox_source") or "")
    if final_bbox_source.startswith("coarse_"):
        return None
    seedream_gate = generation_params.get("seedream_quality_gate")
    if not isinstance(seedream_gate, dict) or not bool(seedream_gate.get("passes_quality")):
        return None

    outside_change = float(seedream_gate.get("outside_change_ratio") or 0.0)
    return {
        "background_preservation_score": max(0.0, min(1.0, 1.0 - outside_change)),
        "anomaly_visibility_score": 1.0,
        "passes_quality": True,
        "quality_reason": None,
        "quality_threshold_profile": anomaly_type,
        "seedream_quality_gate": deepcopy(seedream_gate),
    }


def _copy_existing_mask_to_processed(
    obj: dict[str, Any],
    sample_id: str,
    mask_dir: Path,
    processed_root: str | Path,
) -> str | None:
    object_id = str(obj.get("object_id") or "obj")
    geometry_detail = obj.setdefault("geometry_detail", {})
    generation_params = geometry_detail.get("generation_params") if isinstance(geometry_detail.get("generation_params"), dict) else {}
    mask_uri = geometry_detail.get("mask_uri") or generation_params.get("mask_uri")
    if not isinstance(mask_uri, str) or not mask_uri:
        return None

    source = resolve_path(mask_uri)
    if not source.exists() or not source.is_file():
        return None

    target = mask_dir / f"{sample_id}_{object_id}_mask.png"
    try:
        if source.resolve() != target.resolve():
            ensure_dir(target.parent)
            shutil.copy2(source, target)
        else:
            target = source
    except FileNotFoundError:
        return None
    return _relative_path_string(str(target), processed_root)


def _persist_wan_response_artifact(
    processed_root: str | Path,
    sample_id: str,
    object_id: str,
    generation_params: dict[str, Any],
) -> str | None:
    wan_block = generation_params.get("wan")
    if not isinstance(wan_block, dict):
        return None

    source_path = wan_block.get("wan_response_path")
    if not isinstance(source_path, str) or not source_path:
        return None

    source = Path(source_path)
    if not source.exists() or not source.is_file():
        return None

    target_dir = ensure_dir(Path(processed_root) / "metadata" / "api_responses")
    suffix = source.suffix or ".json"
    target = target_dir / f"{sample_id}_{object_id}_wan_response{suffix}"

    try:
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        else:
            target = source
    except FileNotFoundError:
        return None

    relative_target = _relative_path_string(str(target), processed_root)
    wan_block["wan_response_path"] = relative_target
    return relative_target


def _run_localizer(
    localizer_name: str,
    localizer_cfg: dict[str, Any],
    original_image_path: str,
    generated_image_path: str,
    prompt_box: list[int],
    anomaly_type: str,
    mask_output_path: Path,
    debug_dir: Path,
    sample_id: str | None = None,
    object_id: str | None = None,
    attempt_role: str | None = None,
) -> tuple[Any, float]:
    localizer = create_localizer(localizer_name, **build_localizer_kwargs(localizer_name, localizer_cfg))
    start = perf_counter()
    result = localizer.localize(
        original_image_path=original_image_path,
        generated_image_path=generated_image_path,
        prompt_box=prompt_box,
        anomaly_type=anomaly_type,
        mask_output_path=str(mask_output_path),
        debug_dir=str(debug_dir),
        sample_id=sample_id,
        object_id=object_id,
        attempt_role=attempt_role,
    )
    elapsed_ms = round((perf_counter() - start) * 1000.0, 3)
    return result, elapsed_ms


def _effective_reason(result: Any, quality: dict[str, Any]) -> str | None:
    if bool(getattr(result, "success", False)) and not bool(quality.get("passes_quality")):
        quality_reason = quality.get("quality_reason") or "quality_threshold_not_met"
        return f"quality_failed:{quality_reason}"
    return getattr(result, "reason", None)


def _postprocess_status(result: Any, quality: dict[str, Any]) -> str:
    if bool(getattr(result, "success", False)) and bool(quality.get("passes_quality")):
        return "success"
    if bool(getattr(result, "success", False)):
        return "quality_failed"
    return "failed"


def _evaluate_attempt(
    *,
    sample_id: str,
    object_id: str,
    anomaly_type: str,
    localizer_name: str,
    attempt_role: str,
    result: Any,
    elapsed_ms: float,
    original_image_path: str,
    generated_image_path: str,
    prompt_box: list[int],
    quality_config: dict[str, Any] | None,
    image_size: tuple[int, int] | None,
    reference_bbox: list[int] | None,
    reference_mask_path: str | None,
    task_id: str,
    original_image_uri: str,
    generated_image_uri: str,
    processed_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], bool, str | None, dict[str, Any], dict[str, Any]]:
    result_success = bool(getattr(result, "success", False))
    if result_success and getattr(result, "final_bbox", None) is not None and getattr(result, "mask_path", None):
        quality = evaluate_localization_quality(
            original_image_path=original_image_path,
            generated_image_path=generated_image_path,
            final_bbox=getattr(result, "final_bbox", None),
            mask_path=getattr(result, "mask_path", None),
            anomaly_type=anomaly_type,
            prompt_box=prompt_box,
            quality_config=quality_config,
        )
    else:
        quality = build_quality_failure("localizer_failed", anomaly_type, quality_config=quality_config)

    benchmark = compute_localizer_metrics(
        final_bbox=getattr(result, "final_bbox", None),
        mask_path=getattr(result, "mask_path", None),
        image_size=image_size,
        reference_bbox=reference_bbox,
        reference_mask_path=reference_mask_path,
    )
    effective_success = bool(result_success and quality.get("passes_quality"))
    reason = _effective_reason(result, quality)

    row = {
        "task_id": task_id,
        "sample_id": sample_id,
        "object_id": object_id,
        "anomaly_type": anomaly_type,
        "localizer": getattr(result, "method", localizer_name),
        "localizer_used": getattr(result, "method", localizer_name),
        "attempt_role": attempt_role,
        "success": effective_success,
        "localize_success": result_success,
        "reason": reason,
        "failure_reason": reason,
        "fallback_used": attempt_role == "fallback" or bool(getattr(result, "fallback_used", False)),
        "elapsed_ms": elapsed_ms,
        "bbox_iou": float(benchmark.get("bbox_iou", 0.0)),
        "precision": float(benchmark.get("precision", 0.0)),
        "recall": float(benchmark.get("recall", 0.0)),
        "mask_iou": float(benchmark.get("mask_iou", 0.0)),
        "metric_reference_source": benchmark.get("metric_reference_source", "none"),
        "passes_quality": bool(quality.get("passes_quality")),
        "quality_reason": quality.get("quality_reason"),
        "quality_threshold_profile": quality.get("quality_threshold_profile"),
        "background_preservation_score": float(quality.get("background_preservation_score", 0.0)),
        "anomaly_visibility_score": float(quality.get("anomaly_visibility_score", 0.0)),
        "pgcd_component_score": getattr(result, "metrics", {}).get("pgcd_component_score"),
        "image_uri": str(original_image_uri),
        "generated_image_uri": str(generated_image_uri),
        "final_bbox": deepcopy(getattr(result, "final_bbox", None)),
        "mask_uri": (
            _relative_path_string(str(getattr(result, "mask_path")), processed_root)
            if getattr(result, "mask_path", None) is not None
            else None
        ),
        "crop_uri": None,
        "manual_accept": None,
        "metrics": _serialize_mapping(getattr(result, "metrics", {}) or {}, relative_to=processed_root),
        "quality": deepcopy(quality),
    }

    attempt_record = {
        "name": localizer_name,
        "used": getattr(result, "method", localizer_name),
        "attempt_role": attempt_role,
        "success": effective_success,
        "localize_success": result_success,
        "reason": reason,
        "fallback_used": row["fallback_used"],
        "elapsed_ms": elapsed_ms,
        "metrics": _serialize_mapping(getattr(result, "metrics", {}) or {}, relative_to=processed_root),
        "quality": deepcopy(quality),
        "benchmark": deepcopy(benchmark),
        "debug_artifacts": _serialize_mapping(getattr(result, "debug_artifacts", {}) or {}, relative_to=processed_root),
    }
    return quality, benchmark, effective_success, reason, row, attempt_record


def apply_localizer_postprocess(
    sample: dict[str, Any],
    pipeline_config: dict[str, Any],
    source_row: dict[str, Any] | None,
    processed_root: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    processed = deepcopy(sample)
    localizer_rows: list[dict[str, Any]] = []
    mask_dir = ensure_dir(Path(processed_root) / "masks")
    crop_dir = ensure_dir(Path(processed_root) / "crops")
    debug_root = ensure_dir(Path(processed_root) / "metadata" / "debug" / processed["sample_id"])
    generation_cfg = pipeline_config.get("generation", {})
    crop_expand_ratio = float(generation_cfg.get("crop_expand_ratio", 0.03))
    any_box_updated = False

    for obj in processed.get("objects", []):
        object_row_start = len(localizer_rows)
        anomaly_type = resolve_anomaly_type(source_row, obj)
        localizer_cfg = resolve_localizer_config(pipeline_config, anomaly_type=anomaly_type)
        localizer_block = _ensure_localizer_block(obj)
        generation_params = obj["geometry_detail"]["generation_params"]
        localizer_strategy = localizer_cfg.get("primary")
        localizer_fallback = localizer_cfg.get("fallback")
        localizer_debug = bool(localizer_cfg.get("debug", False))
        allow_quality_fallback = bool(localizer_cfg.get("allow_quality_fallback", False))
        quality_cfg = localizer_cfg.get("quality") if isinstance(localizer_cfg.get("quality"), dict) else None
        sidecar_names = _normalize_localizer_names(localizer_cfg.get("sidecar_eval"))
        generation_params["localizer_strategy"] = localizer_strategy
        generation_params["localizer_fallback"] = localizer_fallback
        generation_params["localizer_debug"] = localizer_debug
        localizer_block["strategy"] = localizer_strategy
        localizer_block["primary"] = localizer_strategy
        localizer_block["fallback"] = localizer_fallback
        localizer_block["debug"] = localizer_debug
        localizer_block["allow_quality_fallback"] = allow_quality_fallback
        localizer_block["sidecar_eval"] = sidecar_names
        localizer_block["anomaly_type"] = anomaly_type
        _persist_wan_response_artifact(
            processed_root=processed_root,
            sample_id=processed["sample_id"],
            object_id=obj["object_id"],
            generation_params=generation_params,
        )

        original_image_uri = resolve_original_image_uri(processed, obj, source_row)
        generated_image_uri = processed.get("image_asset", {}).get("image_uri")
        task_id = (
            str(source_row.get("task_id"))
            if source_row and source_row.get("task_id")
            else str(processed["sample_id"])
        )
        prompt_box = resolve_prompt_box(obj)
        if not original_image_uri or not generated_image_uri or prompt_box is None:
            localizer_block["postprocess_status"] = "inputs_unavailable"
            continue

        primary_name = localizer_cfg.get("primary")
        if not primary_name:
            localizer_block["postprocess_status"] = "localizer_not_configured"
            continue

        original_image_path = str(resolve_path(original_image_uri))
        generated_image_path = str(resolve_path(generated_image_uri))
        image_size = _resolve_image_size(processed)
        reference_bbox = _resolve_reference_bbox(obj, prompt_box)
        reference_mask_uri = _resolve_reference_mask_uri(obj)
        reference_mask_path = str(resolve_path(reference_mask_uri)) if reference_mask_uri else None

        trusted_quality = _trusted_i2i_quality(generation_params, anomaly_type)
        existing_bbox = _as_box_list([
            obj.get("box", {}).get("x1"),
            obj.get("box", {}).get("y1"),
            obj.get("box", {}).get("x2"),
            obj.get("box", {}).get("y2"),
        ])
        if trusted_quality is not None and existing_bbox is not None:
            copied_mask_uri = _copy_existing_mask_to_processed(
                obj,
                str(processed["sample_id"]),
                mask_dir,
                processed_root,
            )
            copied_mask_path = str(Path(processed_root) / copied_mask_uri) if copied_mask_uri else reference_mask_path
            trusted_benchmark = compute_localizer_metrics(
                final_bbox=existing_bbox,
                mask_path=copied_mask_path,
                image_size=image_size,
                reference_bbox=reference_bbox,
                reference_mask_path=reference_mask_path,
            )
            trusted_metrics = {
                "localizer_method": "i2i_diff",
                "final_bbox_source": generation_params.get("final_bbox_source"),
                "selected_grid": generation_params.get("selected_grid"),
                "red_box_bbox": generation_params.get("red_box_bbox"),
                "prompt_box": generation_params.get("prompt_box"),
            }
            localizer_block["used"] = "i2i_diff"
            localizer_block["fallback_used"] = False
            localizer_block["fallback_trigger"] = None
            localizer_block["reason"] = None
            localizer_block["metrics"] = deepcopy(trusted_metrics)
            localizer_block["benchmark"] = deepcopy(trusted_benchmark)
            localizer_block["quality"] = deepcopy(trusted_quality)
            localizer_block["debug_artifacts"] = {}
            localizer_block["attempts"] = [
                {
                    "name": "i2i_diff",
                    "used": "i2i_diff",
                    "attempt_role": "i2i",
                    "success": True,
                    "localize_success": True,
                    "reason": None,
                    "fallback_used": False,
                    "elapsed_ms": 0.0,
                    "metrics": deepcopy(trusted_metrics),
                    "quality": deepcopy(trusted_quality),
                    "benchmark": deepcopy(trusted_benchmark),
                    "debug_artifacts": {},
                }
            ]
            localizer_block["postprocess_status"] = "success"
            generation_params["quality"] = deepcopy(trusted_quality)
            if copied_mask_uri:
                obj.setdefault("geometry_detail", {})["mask_uri"] = copied_mask_uri
                obj["geometry_detail"]["mask_format"] = "png"
            x1, y1, x2, y2 = existing_bbox
            obj["box"] = make_box(x1, y1, x2, y2)
            localizer_rows.append(
                {
                    "task_id": task_id,
                    "sample_id": processed["sample_id"],
                    "object_id": obj["object_id"],
                    "anomaly_type": anomaly_type,
                    "localizer": "i2i_diff",
                    "localizer_used": "i2i_diff",
                    "attempt_role": "i2i",
                    "success": True,
                    "localize_success": True,
                    "reason": None,
                    "failure_reason": None,
                    "fallback_used": False,
                    "elapsed_ms": 0.0,
                    "bbox_iou": float(trusted_benchmark.get("bbox_iou", 0.0)),
                    "precision": float(trusted_benchmark.get("precision", 0.0)),
                    "recall": float(trusted_benchmark.get("recall", 0.0)),
                    "mask_iou": float(trusted_benchmark.get("mask_iou", 0.0)),
                    "metric_reference_source": trusted_benchmark.get("metric_reference_source", "none"),
                    "passes_quality": True,
                    "quality_reason": None,
                    "quality_threshold_profile": trusted_quality.get("quality_threshold_profile"),
                    "background_preservation_score": float(trusted_quality.get("background_preservation_score", 0.0)),
                    "anomaly_visibility_score": float(trusted_quality.get("anomaly_visibility_score", 0.0)),
                    "pgcd_component_score": None,
                    "image_uri": str(original_image_uri),
                    "generated_image_uri": str(generated_image_uri),
                    "final_bbox": existing_bbox,
                    "mask_uri": copied_mask_uri,
                    "crop_uri": None,
                    "manual_accept": None,
                    "metrics": deepcopy(trusted_metrics),
                    "quality": deepcopy(trusted_quality),
                }
            )
            any_box_updated = True
            continue

        attempts: list[dict[str, Any]] = []

        primary_mask_path = mask_dir / f"{processed['sample_id']}_{obj['object_id']}_mask.png"
        primary_result, primary_elapsed_ms = _run_localizer(
            primary_name,
            localizer_cfg,
            original_image_path=original_image_path,
            generated_image_path=generated_image_path,
            prompt_box=prompt_box,
            anomaly_type=anomaly_type,
            mask_output_path=primary_mask_path,
            debug_dir=debug_root,
            sample_id=processed["sample_id"],
            object_id=obj["object_id"],
            attempt_role="primary",
        )
        primary_quality, primary_benchmark, primary_effective_success, primary_reason, primary_row, primary_attempt = _evaluate_attempt(
            sample_id=processed["sample_id"],
            object_id=obj["object_id"],
            anomaly_type=anomaly_type,
            localizer_name=primary_name,
            attempt_role="primary",
            result=primary_result,
            elapsed_ms=primary_elapsed_ms,
            original_image_path=original_image_path,
            generated_image_path=generated_image_path,
            prompt_box=prompt_box,
            quality_config=quality_cfg,
            image_size=image_size,
            reference_bbox=reference_bbox,
            reference_mask_path=reference_mask_path,
            task_id=task_id,
            original_image_uri=str(original_image_uri),
            generated_image_uri=str(generated_image_uri),
            processed_root=processed_root,
        )
        localizer_rows.append(primary_row)
        attempts.append(primary_attempt)

        selected_result = primary_result
        selected_quality = primary_quality
        selected_benchmark = primary_benchmark
        selected_effective_success = primary_effective_success
        selected_reason = primary_reason
        fallback_used = False
        fallback_trigger: str | None = None
        fallback_name = (
            str(localizer_fallback)
            if isinstance(localizer_fallback, str) and localizer_fallback and str(localizer_fallback).lower() != "none"
            else None
        )
        should_try_fallback = bool(
            fallback_name
            and fallback_name != primary_name
            and (
                not bool(getattr(primary_result, "success", False))
                or (bool(getattr(primary_result, "success", False)) and not primary_quality.get("passes_quality") and allow_quality_fallback)
            )
        )

        if should_try_fallback and fallback_name is not None:
            fallback_trigger = "primary_failed" if not bool(getattr(primary_result, "success", False)) else "quality_failed"
            fallback_mask_path = mask_dir / f"{processed['sample_id']}_{obj['object_id']}_{fallback_name}_mask.png"
            fallback_result, fallback_elapsed_ms = _run_localizer(
                fallback_name,
                localizer_cfg,
                original_image_path=original_image_path,
                generated_image_path=generated_image_path,
                prompt_box=prompt_box,
                anomaly_type=anomaly_type,
                mask_output_path=fallback_mask_path,
                debug_dir=debug_root,
                sample_id=processed["sample_id"],
                object_id=obj["object_id"],
                attempt_role="fallback",
            )
            fallback_quality, fallback_benchmark, fallback_effective_success, fallback_reason, fallback_row, fallback_attempt = _evaluate_attempt(
                sample_id=processed["sample_id"],
                object_id=obj["object_id"],
                anomaly_type=anomaly_type,
                localizer_name=fallback_name,
                attempt_role="fallback",
                result=fallback_result,
                elapsed_ms=fallback_elapsed_ms,
                original_image_path=original_image_path,
                generated_image_path=generated_image_path,
                prompt_box=prompt_box,
                quality_config=quality_cfg,
                image_size=image_size,
                reference_bbox=reference_bbox,
                reference_mask_path=reference_mask_path,
                task_id=task_id,
                original_image_uri=str(original_image_uri),
                generated_image_uri=str(generated_image_uri),
                processed_root=processed_root,
            )
            localizer_rows.append(fallback_row)
            attempts.append(fallback_attempt)
            fallback_used = True
            if fallback_effective_success or not primary_effective_success:
                selected_result = fallback_result
                selected_quality = fallback_quality
                selected_benchmark = fallback_benchmark
                selected_effective_success = fallback_effective_success
                selected_reason = fallback_reason

        localizer_block["used"] = getattr(selected_result, "method", primary_name)
        localizer_block["fallback_used"] = fallback_used
        localizer_block["fallback_trigger"] = fallback_trigger if fallback_used else None
        localizer_block["reason"] = selected_reason
        localizer_block["metrics"] = _serialize_mapping(getattr(selected_result, "metrics", {}) or {}, relative_to=processed_root)
        localizer_block["benchmark"] = deepcopy(selected_benchmark)
        localizer_block["quality"] = deepcopy(selected_quality)
        localizer_block["debug_artifacts"] = _serialize_mapping(
            getattr(selected_result, "debug_artifacts", {}) or {},
            relative_to=processed_root,
        )
        localizer_block["attempts"] = attempts
        localizer_block["postprocess_status"] = _postprocess_status(selected_result, selected_quality)
        generation_params["quality"] = deepcopy(selected_quality)

        for row in localizer_rows[object_row_start:]:
            row["task_id"] = task_id
            row["localizer_used"] = localizer_block["used"]

        if selected_effective_success and getattr(selected_result, "final_bbox", None):
            x1, y1, x2, y2 = [int(value) for value in getattr(selected_result, "final_bbox")]
            obj["box"] = make_box(x1, y1, x2, y2)
            mask_uri = getattr(selected_result, "mask_path", None)
            obj.setdefault("geometry_detail", {})["mask_uri"] = (
                _relative_path_string(str(mask_uri), processed_root)
                if mask_uri is not None
                else None
            )
            obj["geometry_detail"]["mask_format"] = "png"
            any_box_updated = True

        sidecar_records: list[dict[str, Any]] = []
        for sidecar_name in sidecar_names:
            if sidecar_name == primary_name:
                continue
            sidecar_mask_path = mask_dir / f"{processed['sample_id']}_{obj['object_id']}_{sidecar_name}_mask.png"
            sidecar_result, sidecar_elapsed_ms = _run_localizer(
                sidecar_name,
                localizer_cfg,
                original_image_path=original_image_path,
                generated_image_path=generated_image_path,
                prompt_box=prompt_box,
                anomaly_type=anomaly_type,
                mask_output_path=sidecar_mask_path,
                debug_dir=debug_root,
                sample_id=processed["sample_id"],
                object_id=obj["object_id"],
                attempt_role="sidecar",
            )
            sidecar_quality, sidecar_benchmark, sidecar_effective_success, sidecar_reason, sidecar_row, _sidecar_attempt = _evaluate_attempt(
                sample_id=processed["sample_id"],
                object_id=obj["object_id"],
                anomaly_type=anomaly_type,
                localizer_name=sidecar_name,
                attempt_role="sidecar",
                result=sidecar_result,
                elapsed_ms=sidecar_elapsed_ms,
                original_image_path=original_image_path,
                generated_image_path=generated_image_path,
                prompt_box=prompt_box,
                quality_config=quality_cfg,
                image_size=image_size,
                reference_bbox=reference_bbox,
                reference_mask_path=reference_mask_path,
                task_id=task_id,
                original_image_uri=str(original_image_uri),
                generated_image_uri=str(generated_image_uri),
                processed_root=processed_root,
            )
            localizer_rows.append(sidecar_row)
            sidecar_records.append(
                {
                    "name": sidecar_name,
                    "used": getattr(sidecar_result, "method", sidecar_name),
                    "success": sidecar_effective_success,
                    "localize_success": bool(getattr(sidecar_result, "success", False)),
                    "reason": sidecar_reason,
                    "metrics": _serialize_mapping(
                        getattr(sidecar_result, "metrics", {}) or {},
                        relative_to=processed_root,
                    ),
                    "quality": deepcopy(sidecar_quality),
                    "benchmark": deepcopy(sidecar_benchmark),
                    "elapsed_ms": sidecar_elapsed_ms,
                    "debug_artifacts": _serialize_mapping(
                        getattr(sidecar_result, "debug_artifacts", {}) or {},
                        relative_to=processed_root,
                    ),
                }
            )
        if sidecar_records:
            localizer_block["sidecars"] = sidecar_records
            localizer_block["sidecar"] = deepcopy(sidecar_records[0])

    if any_box_updated:
        cleanup_sample_crops(crop_dir, processed["sample_id"])
        attach_crops(processed, crop_dir, crop_expand_ratio)
        for obj in processed.get("objects", []):
            crop = obj.get("crop")
            if isinstance(crop, dict) and crop.get("crop_uri") not in (None, "", "pending"):
                crop["crop_uri"] = _relative_path_string(str(crop["crop_uri"]), processed_root)
        touch_workflow(processed)
    crop_uri_by_object = {
        str(obj.get("object_id")): obj.get("crop", {}).get("crop_uri")
        for obj in processed.get("objects", [])
        if isinstance(obj.get("crop"), dict)
    }
    for row in localizer_rows:
        crop_uri = crop_uri_by_object.get(str(row.get("object_id")))
        if crop_uri not in (None, "", "pending"):
            row["crop_uri"] = crop_uri
    return processed, localizer_rows
