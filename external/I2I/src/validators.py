from __future__ import annotations

from config import SUPPORTED_ANOMALY_TYPES


class ValidationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_bbox(box: dict, width: int, height: int) -> None:
    _require(isinstance(box, dict), "box must be dict")
    for key in ("format", "x1", "y1", "x2", "y2"):
        _require(key in box, f"box.{key} is required")
    _require(box["format"] == "xyxy", "box.format must be xyxy")
    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    _require(0 <= x1 < x2 <= width, f"invalid x coordinates: {box}")
    _require(0 <= y1 < y2 <= height, f"invalid y coordinates: {box}")


def validate_task(task: dict[str, str]) -> None:
    for key in ("sample_id", "image_id", "image_uri", "anomaly_type", "source_type"):
        _require(bool(task.get(key)), f"task.{key} is required")
    _require(task["anomaly_type"] in SUPPORTED_ANOMALY_TYPES, f"unsupported anomaly_type: {task['anomaly_type']}")


def validate_required_fields(sample: dict) -> None:
    _require(isinstance(sample.get("sample_id"), str) and sample["sample_id"], "sample_id is required")

    image_asset = sample.get("image_asset")
    _require(isinstance(image_asset, dict), "image_asset is required")
    for key in ("image_id", "image_uri", "width", "height", "source_type"):
        _require(image_asset.get(key) not in (None, ""), f"image_asset.{key} is required")

    width = int(image_asset["width"])
    height = int(image_asset["height"])
    objects = sample.get("objects")
    _require(isinstance(objects, list) and objects, "objects must be a non-empty list")
    for idx, obj in enumerate(objects):
        prefix = f"objects[{idx}]"
        _require(obj.get("object_id") not in (None, ""), f"{prefix}.object_id is required")
        _require(obj.get("object_type") not in (None, ""), f"{prefix}.object_type is required")
        validate_bbox(obj.get("box"), width, height)
        _require(obj.get("geometry_source") not in (None, ""), f"{prefix}.geometry_source is required")
        crop = obj.get("crop")
        _require(isinstance(crop, dict), f"{prefix}.crop is required")
        _require(crop.get("crop_id") not in (None, ""), f"{prefix}.crop.crop_id is required")
        _require(crop.get("crop_uri") not in (None, ""), f"{prefix}.crop.crop_uri is required")
        classification = obj.get("classification")
        _require(isinstance(classification, dict), f"{prefix}.classification is required")
        labels = classification.get("multi_labels")
        _require(isinstance(labels, list) and labels, f"{prefix}.classification.multi_labels is required")
        for label_idx, item in enumerate(labels):
            label_prefix = f"{prefix}.classification.multi_labels[{label_idx}]"
            _require(item.get("label_key") not in (None, ""), f"{label_prefix}.label_key is required")
            _require(item.get("label_value") not in (None, ""), f"{label_prefix}.label_value is required")

    workflow = sample.get("workflow")
    _require(isinstance(workflow, dict), "workflow is required")
    _require(workflow.get("workflow_status") not in (None, ""), "workflow.workflow_status is required")
    export = sample.get("export")
    _require(isinstance(export, dict), "export is required")
    _require(export.get("export_format") not in (None, ""), "export.export_format is required")
    _require(export.get("export_status") not in (None, ""), "export.export_status is required")
