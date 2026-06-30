from __future__ import annotations

from pathlib import Path
from typing import Any

from autolabel.validators import ValidationError, validate_sample_contract


def validate_autolabel_sample(sample: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        validate_sample_contract(sample)
    except (ValidationError, ValueError, TypeError) as exc:
        errors.append(str(exc))

    image_asset = sample.get("image_asset") if isinstance(sample, dict) else None
    width = int(image_asset.get("width", 0)) if isinstance(image_asset, dict) else 0
    height = int(image_asset.get("height", 0)) if isinstance(image_asset, dict) else 0
    if isinstance(image_asset, dict):
        image_uri = image_asset.get("image_uri")
        if not image_uri or not Path(str(image_uri)).exists():
            errors.append(f"image_asset.image_uri does not exist: {image_uri}")

    objects = sample.get("objects") if isinstance(sample, dict) else None
    if not isinstance(objects, list) or not objects:
        errors.append("objects must be a non-empty list")
        return False, errors

    for idx, obj in enumerate(objects):
        prefix = f"objects[{idx}]"
        if not isinstance(obj, dict):
            errors.append(f"{prefix} must be an object")
            continue
        box = obj.get("box")
        if isinstance(box, dict):
            for key in ("x1", "y1", "x2", "y2"):
                if not isinstance(box.get(key), int) or isinstance(box.get(key), bool):
                    errors.append(f"{prefix}.box.{key} must be an integer")
            if box.get("format") != "xyxy":
                errors.append(f"{prefix}.box.format must be xyxy")
            if width and height:
                x1, y1, x2, y2 = int(box.get("x1", -1)), int(box.get("y1", -1)), int(box.get("x2", -1)), int(box.get("y2", -1))
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    errors.append(f"{prefix}.box is outside image boundary")
        else:
            errors.append(f"{prefix}.box is required")

        crop = obj.get("crop")
        if isinstance(crop, dict):
            crop_uri = crop.get("crop_uri")
            if not crop_uri or not Path(str(crop_uri)).exists():
                errors.append(f"{prefix}.crop.crop_uri does not exist: {crop_uri}")
        else:
            errors.append(f"{prefix}.crop is required")

        classification = obj.get("classification")
        labels = classification.get("multi_labels") if isinstance(classification, dict) else None
        if not isinstance(labels, list) or not labels:
            errors.append(f"{prefix}.classification.multi_labels must be non-empty")
        else:
            for label_idx, label in enumerate(labels):
                if not isinstance(label, dict):
                    errors.append(f"{prefix}.classification.multi_labels[{label_idx}] must be an object")
                    continue
                if not label.get("label_key"):
                    errors.append(f"{prefix}.classification.multi_labels[{label_idx}].label_key is required")
                if not label.get("label_value"):
                    errors.append(f"{prefix}.classification.multi_labels[{label_idx}].label_value is required")

        geometry_detail = obj.get("geometry_detail")
        if isinstance(geometry_detail, dict):
            mask_uri = geometry_detail.get("mask_uri")
            if not mask_uri or not Path(str(mask_uri)).exists():
                errors.append(f"{prefix}.geometry_detail.mask_uri does not exist: {mask_uri}")
            if geometry_detail.get("mask_format") != "png":
                errors.append(f"{prefix}.geometry_detail.mask_format must be png")
        else:
            errors.append(f"{prefix}.geometry_detail is required")

    workflow = sample.get("workflow") if isinstance(sample, dict) else None
    if not isinstance(workflow, dict) or not workflow.get("workflow_status"):
        errors.append("workflow.workflow_status is required")
    export = sample.get("export") if isinstance(sample, dict) else None
    if not isinstance(export, dict) or not export.get("export_format") or not export.get("export_status"):
        errors.append("export.export_format and export.export_status are required")
    return not errors, errors
