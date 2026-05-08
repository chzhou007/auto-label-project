from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from ..sample_factory import touch_workflow
from ..utils import read_json, write_json


OBJECT_TYPE_TO_LABEL = {
    "person": "Person",
    "equipment": "Equipment",
    "fire_extinguisher": "Fire Extinguisher",
    "ground_area": "Ground Area",
    "yellow_black_line": "Yellow Black Line",
    "pipe": "Pipe",
    "support": "Support",
    "cable": "Cable",
    "leakage_area": "Leakage Area",
}

DEFAULT_CLASSIFICATION_CHOICES = {
    "helmet": ["unknown", "wearing_helmet", "no_helmet"],
    "reflective_vest": ["unknown", "wearing_reflective_vest", "no_reflective_vest"],
    "safety_belt": ["unknown", "wearing_safety_belt", "no_safety_belt"],
    "safety_shoes": ["unknown", "wearing_safety_shoes", "no_safety_shoes"],
    "smoking": ["unknown", "smoking", "not_smoking"],
    "sleeping": ["unknown", "sleeping", "not_sleeping"],
    "falling": ["unknown", "falling", "not_falling"],
    "phone_usage": ["unknown", "using_phone", "not_using_phone"],
    "climbing_over_railing": ["unknown", "climbing_over_railing", "not_climbing_over_railing"],
    "touching_equipment": ["unknown", "touching_equipment", "not_touching_equipment"],
    "fighting": ["unknown", "fighting", "not_fighting"],
    "safety_goggles": ["unknown", "wearing_safety_goggles", "no_safety_goggles"],
    "anomaly_type": ["unknown", "diesel_leak", "oil_leak", "coolant_leak"],
}


def normalize_labelstudio_image_uri(image_uri: str) -> str:
    return image_uri.replace("\\", "/")


def box_to_labelstudio_value(box: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    return {
        "x": max(0.0, min(100.0, x1 / width * 100.0)),
        "y": max(0.0, min(100.0, y1 / height * 100.0)),
        "width": max(0.0, min(100.0, (x2 - x1) / width * 100.0)),
        "height": max(0.0, min(100.0, (y2 - y1) / height * 100.0)),
        "rotation": 0,
    }


def rectangle_labels_for_object(obj: dict[str, Any]) -> list[str]:
    object_type = obj.get("object_type", "object")
    return [OBJECT_TYPE_TO_LABEL.get(object_type, object_type)]


def classification_control_name(label_key: str) -> str:
    return f"cls_{label_key}"


def classification_results_for_object(obj: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for label in obj.get("classification", {}).get("multi_labels", []):
        label_key = label.get("label_key")
        label_value = label.get("label_value")
        if not label_key or not label_value:
            continue
        results.append(
            {
                "id": obj["object_id"],
                "type": "choices",
                "from_name": classification_control_name(label_key),
                "to_name": "image",
                "value": {
                    "choices": [label_value],
                },
                "meta": {
                    "sample_id": obj.get("sample_id"),
                    "object_id": obj["object_id"],
                    "object_type": obj["object_type"],
                    "label_key": label_key,
                    "confidence": label.get("confidence"),
                    "evidence": label.get("evidence"),
                },
            }
        )
    return results


def sample_to_labelstudio_task(sample: dict[str, Any]) -> dict[str, Any]:
    image_asset = sample["image_asset"]
    width = int(image_asset["width"])
    height = int(image_asset["height"])
    results = []
    for obj in sample.get("objects", []):
        value = box_to_labelstudio_value(obj["box"], width, height)
        value["rectanglelabels"] = rectangle_labels_for_object(obj)
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
                    "object_type": obj["object_type"],
                    "geometry_source": obj["geometry_source"],
                    "box_xyxy": obj["box"],
                    "crop": obj.get("crop"),
                    "classification": obj.get("classification"),
                    "quality_check": obj.get("quality_check"),
                },
            }
        )
        enriched_obj = dict(obj)
        enriched_obj["sample_id"] = sample["sample_id"]
        results.extend(classification_results_for_object(enriched_obj))

    return {
        "data": {
            "image": normalize_labelstudio_image_uri(image_asset["image_uri"]),
            "sample_id": sample["sample_id"],
        },
        "predictions": [
            {
                "model_version": sample.get("workflow", {}).get("pipeline_version") or "autolabel",
                "result": results,
            }
        ],
    }


def export_samples(samples: list[dict[str, Any]], output_path: str | Path) -> list[dict[str, Any]]:
    tasks = [sample_to_labelstudio_task(sample) for sample in samples]
    write_json(output_path, tasks)
    write_labelstudio_config(default_config_path(output_path), samples)
    return tasks


def export_metadata_dir(metadata_dir: str | Path, output_path: str | Path, update_samples: bool = False) -> list[dict[str, Any]]:
    root = Path(metadata_dir)
    samples = []
    for path in sorted(root.glob("*.json")):
        sample = read_json(path)
        samples.append(sample)
        if update_samples:
            sample["export"]["export_format"] = "labelstudio"
            sample["export"]["export_status"] = "exported"
            sample["export"]["export_uri"] = str(output_path)
            touch_workflow(sample, "exported")
            write_json(path, sample)
    return export_samples(samples, output_path)


def default_config_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_name("labelstudio_config.xml")


def collect_classification_choices(samples: list[dict[str, Any]]) -> dict[str, list[str]]:
    choices = {key: list(values) for key, values in DEFAULT_CLASSIFICATION_CHOICES.items()}
    for sample in samples:
        for obj in sample.get("objects", []):
            for label in obj.get("classification", {}).get("multi_labels", []):
                label_key = label.get("label_key")
                label_value = label.get("label_value")
                if not label_key or not label_value:
                    continue
                choices.setdefault(label_key, ["unknown"])
                if label_value not in choices[label_key]:
                    choices[label_key].append(label_value)
    return choices


def collect_rectangle_labels(samples: list[dict[str, Any]]) -> list[str]:
    labels = set()
    for sample in samples:
        for obj in sample.get("objects", []):
            labels.update(rectangle_labels_for_object(obj))
    return sorted(labels) or ["Person"]


def build_labelstudio_config(samples: list[dict[str, Any]]) -> str:
    rectangle_labels = collect_rectangle_labels(samples)
    classification_choices = collect_classification_choices(samples)
    lines = [
        '<View>',
        '  <Image name="image" value="$image"/>',
        '  <RectangleLabels name="label" toName="image">',
    ]
    for label in rectangle_labels:
        lines.append(f'    <Label value="{escape(label)}"/>')
    lines.extend(
        [
            "  </RectangleLabels>",
            '  <Header value="区域分类标签"/>',
        ]
    )
    for label_key, values in classification_choices.items():
        control_name = classification_control_name(label_key)
        lines.append(
            f'  <Choices name="{escape(control_name)}" toName="image" perRegion="true" choice="single" showInline="true">'
        )
        for value in values:
            lines.append(f'    <Choice value="{escape(value)}"/>')
        lines.append("  </Choices>")
    lines.append("</View>")
    return "\n".join(lines) + "\n"


def write_labelstudio_config(path: str | Path, samples: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_labelstudio_config(samples), encoding="utf-8")
