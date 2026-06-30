from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autolabel.utils import read_json

from .validators import validate_autolabel_sample


def xyxy_to_labelstudio(box: dict[str, Any], image_width: int, image_height: int) -> dict[str, float | int]:
    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    return {
        "x": x1 / image_width * 100.0,
        "y": y1 / image_height * 100.0,
        "width": (x2 - x1) / image_width * 100.0,
        "height": (y2 - y1) / image_height * 100.0,
        "rotation": 0,
    }


def anomaly_label_for_object(obj: dict[str, Any]) -> str:
    labels = obj.get("classification", {}).get("multi_labels", [])
    for label in labels:
        if label.get("label_key") == "anomaly_type" and label.get("label_value"):
            return str(label["label_value"])
    if labels and labels[0].get("label_value"):
        return str(labels[0]["label_value"])
    return str(obj.get("object_type") or "leakage_area")


def sample_to_labelstudio_task(sample: dict[str, Any]) -> dict[str, Any]:
    image = sample["image_asset"]
    width, height = int(image["width"]), int(image["height"])
    results = []
    for obj in sample.get("objects", []):
        value = xyxy_to_labelstudio(obj["box"], width, height)
        value["rectanglelabels"] = [anomaly_label_for_object(obj)]
        results.append(
            {
                "id": obj["object_id"],
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": value,
                "meta": {
                    "sample_id": sample["sample_id"],
                    "object_id": obj["object_id"],
                    "box_xyxy": obj["box"],
                    "mask_uri": obj.get("geometry_detail", {}).get("mask_uri"),
                    "crop": obj.get("crop"),
                    "classification": obj.get("classification"),
                },
            }
        )
    return {
        "data": {"image": str(image["image_uri"]).replace("\\", "/"), "sample_id": sample["sample_id"]},
        "predictions": [{"model_version": sample.get("workflow", {}).get("pipeline_version", "v1"), "result": results}],
    }


def export_metadata_dir(metadata_dir: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    tasks = []
    for path in sorted(Path(metadata_dir).glob("*.json")):
        sample = read_json(path)
        ok, errors = validate_autolabel_sample(sample)
        if not ok:
            raise ValueError(f"Cannot export invalid metadata {path}: {errors}")
        tasks.append(sample_to_labelstudio_task(sample))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    return tasks
