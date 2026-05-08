from __future__ import annotations

from typing import Any

from .utils import now_iso_shanghai


def _none_if_blank(value: Any) -> Any:
    return None if value in ("", None) else value


def source_context_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation_prompt": _none_if_blank(row.get("generation_prompt")),
        "generation_model": _none_if_blank(row.get("generation_model")),
        "collection_batch": _none_if_blank(row.get("collection_batch")),
        "camera_id": _none_if_blank(row.get("camera_id")),
        "capture_time": _none_if_blank(row.get("capture_time")),
    }


def scene_context_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "site": _none_if_blank(row.get("site")),
        "building": _none_if_blank(row.get("building")),
        "floor": _none_if_blank(row.get("floor")),
        "room_name": _none_if_blank(row.get("room_name")),
        "room_type": _none_if_blank(row.get("room_type")),
        "task_group": _none_if_blank(row.get("task_group")),
        "inspection_content": _none_if_blank(row.get("inspection_content")),
    }


def empty_classification(
    classifier_type: str = "vlm",
    classifier_name: str | None = None,
    classifier_version: str | None = None,
    prompt_version: str | None = None,
) -> dict[str, Any]:
    return {
        "multi_labels": [],
        "classifier_type": classifier_type,
        "classifier_name": classifier_name,
        "classifier_version": classifier_version,
        "prompt_version": prompt_version,
        "raw_response": None,
    }


def make_box(x1: int, y1: int, x2: int, y2: int) -> dict[str, int | str]:
    return {
        "format": "xyxy",
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
    }


def make_object(
    object_id: str,
    object_type: str,
    box: dict[str, Any],
    geometry_source: str,
    geometry_model: dict[str, Any] | None = None,
    geometry_detail: dict[str, Any] | None = None,
    crop: dict[str, Any] | None = None,
    classification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "object_type": object_type,
        "box": box,
        "geometry_source": geometry_source,
        "geometry_model": geometry_model,
        "geometry_detail": geometry_detail
        or {
            "polygon": None,
            "mask_uri": None,
            "mask_format": None,
            "generation_params": None,
        },
        "crop": crop
        or {
            "crop_id": f"{object_id}_crop",
            "crop_uri": "pending",
            "crop_box": None,
            "crop_expand_ratio": None,
            "is_valid_crop": False,
        },
        "classification": classification or empty_classification(),
        "quality_check": None,
    }


def make_sample(
    row: dict[str, Any],
    width: int,
    height: int,
    pipeline_id: str,
    pipeline_version: str,
    qc_policy: dict[str, Any] | None = None,
    export_config: dict[str, Any] | None = None,
    workflow_status: str = "raw",
) -> dict[str, Any]:
    now = now_iso_shanghai()
    export_config = export_config or {}
    return {
        "sample_id": row["sample_id"],
        "image_asset": {
            "image_id": row["image_id"],
            "image_uri": row["image_uri"],
            "width": int(width),
            "height": int(height),
            "source_type": row["source_type"],
            "source_context": source_context_from_row(row),
            "scene_context": scene_context_from_row(row),
        },
        "objects": [],
        "qc_policy": qc_policy,
        "workflow": {
            "workflow_status": workflow_status,
            "pipeline_id": pipeline_id,
            "pipeline_version": pipeline_version,
            "created_time": now,
            "updated_time": now,
        },
        "export": {
            "export_format": export_config.get("export_format", "labelstudio"),
            "export_status": export_config.get("export_status", "not_exported"),
            "export_uri": export_config.get("export_uri"),
            "labelstudio_mapping": export_config.get("labelstudio_mapping"),
        },
    }


def touch_workflow(sample: dict[str, Any], status: str | None = None) -> None:
    if status is not None:
        sample["workflow"]["workflow_status"] = status
    sample["workflow"]["updated_time"] = now_iso_shanghai()
