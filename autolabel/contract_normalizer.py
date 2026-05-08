from __future__ import annotations

from copy import deepcopy
from typing import Any

from .utils import now_iso_shanghai


SOURCE_CONTEXT_DEFAULTS = {
    "generation_prompt": None,
    "generation_model": None,
    "collection_batch": None,
    "camera_id": None,
    "capture_time": None,
}

SCENE_CONTEXT_DEFAULTS = {
    "site": None,
    "building": None,
    "floor": None,
    "room_name": None,
    "room_type": None,
    "task_group": None,
    "inspection_content": None,
}

GEOMETRY_DETAIL_DEFAULTS = {
    "polygon": None,
    "mask_uri": None,
    "mask_format": None,
    "generation_params": None,
}

CLASSIFICATION_DEFAULTS = {
    "multi_labels": [],
    "classifier_type": "vlm",
    "classifier_name": None,
    "classifier_version": None,
    "prompt_version": None,
    "raw_response": None,
}

QC_POLICY_DEFAULTS = {
    "qc_mode": "sampling",
    "sampling_ratio": None,
    "sampling_method": "random",
    "fail_policy": "batch_rework",
    "qc_batch_id": None,
}

EXPORT_DEFAULTS = {
    "export_format": "labelstudio",
    "export_status": "not_exported",
    "export_uri": None,
    "labelstudio_mapping": None,
}


def _merge_defaults(defaults: dict[str, Any], value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return deepcopy(defaults)
    merged = deepcopy(defaults)
    merged.update(value)
    return merged


def normalize_autolabel_sample(
    sample: dict[str, Any],
    pipeline_id: str | None = None,
    pipeline_version: str | None = None,
    qc_policy: dict[str, Any] | None = None,
    export_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = deepcopy(sample)
    now = now_iso_shanghai()

    image_asset = normalized.setdefault("image_asset", {})
    image_asset["source_context"] = _merge_defaults(
        SOURCE_CONTEXT_DEFAULTS,
        image_asset.get("source_context", SOURCE_CONTEXT_DEFAULTS),
    )
    image_asset["scene_context"] = _merge_defaults(
        SCENE_CONTEXT_DEFAULTS,
        image_asset.get("scene_context", SCENE_CONTEXT_DEFAULTS),
    )

    for obj in normalized.setdefault("objects", []):
        if obj.get("geometry_model") is not None:
            geometry_model = obj.setdefault("geometry_model", {})
            geometry_model.setdefault("model_name", None)
            geometry_model.setdefault("model_version", None)
            geometry_model.setdefault("confidence", None)

        obj["geometry_detail"] = _merge_defaults(
            GEOMETRY_DETAIL_DEFAULTS,
            obj.get("geometry_detail", GEOMETRY_DETAIL_DEFAULTS),
        )

        crop = obj.setdefault("crop", {})
        crop.setdefault("crop_id", f"{obj.get('object_id', 'object')}_crop")
        crop.setdefault("crop_uri", "")
        crop.setdefault("crop_box", None)
        crop.setdefault("crop_expand_ratio", None)
        crop.setdefault("is_valid_crop", True)

        classification = obj.setdefault("classification", {})
        for key, default_value in CLASSIFICATION_DEFAULTS.items():
            classification.setdefault(key, deepcopy(default_value))

        obj.setdefault("quality_check", None)

    if qc_policy is not None:
        normalized["qc_policy"] = _merge_defaults(QC_POLICY_DEFAULTS, qc_policy)
    elif "qc_policy" not in normalized:
        normalized["qc_policy"] = None

    workflow = normalized.setdefault("workflow", {})
    workflow.setdefault("workflow_status", "raw")
    workflow.setdefault("pipeline_id", pipeline_id)
    workflow.setdefault("pipeline_version", pipeline_version)
    workflow.setdefault("created_time", now)
    workflow.setdefault("updated_time", now)

    export_defaults = deepcopy(EXPORT_DEFAULTS)
    if export_config:
        export_defaults.update(export_config)
    export = normalized.setdefault("export", {})
    for key, default_value in export_defaults.items():
        export.setdefault(key, default_value)

    return normalized
