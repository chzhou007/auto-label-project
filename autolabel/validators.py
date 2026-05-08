from __future__ import annotations

from typing import Any

from .constants import (
    CLASSIFIER_TYPES,
    EXPORT_FORMATS,
    EXPORT_STATUSES,
    GEOMETRY_SOURCES,
    SOURCE_TYPES,
    WORKFLOW_STATUSES,
)


class ValidationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _require_string(data: dict[str, Any], key: str, prefix: str) -> str:
    value = data.get(key)
    _require(isinstance(value, str) and bool(value), f"{prefix}.{key} is required")
    return value


def validate_box(box: dict[str, Any], width: int, height: int, prefix: str = "box") -> None:
    _require(isinstance(box, dict), f"{prefix} must be an object")
    for key in ("format", "x1", "y1", "x2", "y2"):
        _require(key in box, f"{prefix}.{key} is required")
    _require(box["format"] == "xyxy", f"{prefix}.format must be xyxy")
    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    _require(0 <= x1 < x2 <= width, f"{prefix} has invalid x coordinates: {box}")
    _require(0 <= y1 < y2 <= height, f"{prefix} has invalid y coordinates: {box}")


def validate_label(label: dict[str, Any], prefix: str) -> None:
    _require(isinstance(label, dict), f"{prefix} must be an object")
    _require_string(label, "label_key", prefix)
    _require_string(label, "label_value", prefix)
    confidence = label.get("confidence")
    if confidence is not None:
        confidence = float(confidence)
        _require(0 <= confidence <= 1, f"{prefix}.confidence must be between 0 and 1")


def validate_classification(classification: dict[str, Any], prefix: str) -> None:
    _require(isinstance(classification, dict), f"{prefix} must be an object")
    labels = classification.get("multi_labels")
    _require(isinstance(labels, list), f"{prefix}.multi_labels must be a list")
    for idx, label in enumerate(labels):
        validate_label(label, f"{prefix}.multi_labels[{idx}]")
    classifier_type = classification.get("classifier_type", "vlm")
    _require(classifier_type in CLASSIFIER_TYPES, f"{prefix}.classifier_type is invalid: {classifier_type}")


def validate_sample_contract(sample: dict[str, Any]) -> None:
    _require(isinstance(sample, dict), "sample must be an object")
    _require_string(sample, "sample_id", "sample")

    image_asset = sample.get("image_asset")
    _require(isinstance(image_asset, dict), "image_asset is required")
    _require_string(image_asset, "image_id", "image_asset")
    _require_string(image_asset, "image_uri", "image_asset")
    width = int(image_asset.get("width", 0))
    height = int(image_asset.get("height", 0))
    _require(width > 0, "image_asset.width must be positive")
    _require(height > 0, "image_asset.height must be positive")
    source_type = _require_string(image_asset, "source_type", "image_asset")
    _require(source_type in SOURCE_TYPES, f"image_asset.source_type is invalid: {source_type}")

    objects = sample.get("objects")
    _require(isinstance(objects, list), "objects must be a list")
    for idx, obj in enumerate(objects):
        prefix = f"objects[{idx}]"
        _require(isinstance(obj, dict), f"{prefix} must be an object")
        _require_string(obj, "object_id", prefix)
        _require_string(obj, "object_type", prefix)
        validate_box(obj.get("box"), width, height, f"{prefix}.box")
        geometry_source = _require_string(obj, "geometry_source", prefix)
        _require(geometry_source in GEOMETRY_SOURCES, f"{prefix}.geometry_source is invalid: {geometry_source}")

        crop = obj.get("crop")
        _require(isinstance(crop, dict), f"{prefix}.crop is required")
        _require_string(crop, "crop_id", f"{prefix}.crop")
        _require_string(crop, "crop_uri", f"{prefix}.crop")
        if crop.get("crop_box") is not None:
            validate_box(crop["crop_box"], width, height, f"{prefix}.crop.crop_box")

        validate_classification(obj.get("classification"), f"{prefix}.classification")

    workflow = sample.get("workflow")
    _require(isinstance(workflow, dict), "workflow is required")
    workflow_status = _require_string(workflow, "workflow_status", "workflow")
    _require(workflow_status in WORKFLOW_STATUSES, f"workflow.workflow_status is invalid: {workflow_status}")

    export = sample.get("export")
    _require(isinstance(export, dict), "export is required")
    export_format = _require_string(export, "export_format", "export")
    export_status = _require_string(export, "export_status", "export")
    _require(export_format in EXPORT_FORMATS, f"export.export_format is invalid: {export_format}")
    _require(export_status in EXPORT_STATUSES, f"export.export_status is invalid: {export_status}")
