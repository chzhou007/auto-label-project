from __future__ import annotations

from typing import Any

from utils import now_iso_shanghai
from validators import validate_required_fields


FLUID_TYPE = {
    "water_leak": "water",
    "diesel_leak": "diesel",
    "oil_leak": "engine_oil",
    "coolant_leak": "coolant",
}


def build_classification_labels(anomaly_type: str) -> list[dict[str, Any]]:
    return [
        {
            "label_key": "anomaly_type",
            "label_value": anomaly_type,
            "confidence": 1.0,
            "evidence": "synthetic generation task label",
        },
        {
            "label_key": "fluid_type",
            "label_value": FLUID_TYPE[anomaly_type],
            "confidence": 1.0,
            "evidence": "synthetic generation task label",
        },
        {
            "label_key": "severity",
            "label_value": "early_stage_low_pressure_leakage",
            "confidence": 1.0,
            "evidence": "prompt constraint",
        },
    ]


def build_autolabel_sample(
    task: dict,
    generated_image_uri: str,
    width: int,
    height: int,
    object_box: dict,
    crop_info: dict,
    classification_labels: list[dict],
    generation_params: dict,
    generation_prompt: str,
    image_model: str,
    vlm_model: str,
    selector_model: str | None = None,
    selector_backend: str = "qwen_grid_selector",
) -> dict:
    now = now_iso_shanghai()
    anomaly_type = task["anomaly_type"]
    sample = {
        "sample_id": task["sample_id"],
        "image_asset": {
            "image_id": task["image_id"],
            "image_uri": generated_image_uri,
            "width": width,
            "height": height,
            "source_type": "generated",
            "source_context": {
                "generation_prompt": generation_prompt,
                "generation_model": image_model,
                "collection_batch": task.get("collection_batch"),
                "camera_id": task.get("camera_id"),
                "capture_time": task.get("capture_time"),
            },
            "scene_context": {
                "site": task.get("site") or None,
                "building": task.get("building") or None,
                "floor": task.get("floor") or None,
                "room_name": task.get("room_name") or None,
                "room_type": task.get("room_type") or "diesel_generator_room",
                "task_group": "equipment_environment_anomaly",
                "inspection_content": anomaly_type,
            },
        },
        "objects": [
            {
                "object_id": "obj_000001",
                "object_type": "leakage_area",
                "box": object_box,
                "geometry_source": "synthetic_generator",
                "geometry_model": {
                    "model_name": (
                        f"{selector_model or vlm_model}_selection + "
                        f"{image_model} + image_diff_connected_components"
                    ),
                    "model_version": "v1.0",
                    "confidence": None,
                },
                "geometry_detail": {
                    "polygon": None,
                    "mask_uri": generation_params.get("mask_uri"),
                    "mask_format": "png",
                    "generation_params": generation_params,
                },
                "crop": crop_info,
                "classification": {
                    "multi_labels": classification_labels,
                    "classifier_type": "rule",
                    "classifier_name": "synthetic_label_rule",
                    "classifier_version": "v1.0",
                    "prompt_version": "industrial_leak_prompt_v1.0",
                    "raw_response": None,
                },
                "quality_check": None,
            }
        ],
        "qc_policy": None,
        "workflow": {
            "workflow_status": "classified",
            "pipeline_id": (
                "mmseg_floor_image_edit_autolabel_pipeline"
                if selector_backend == "mmseg_floor_selector"
                else "vlm_grid_image_edit_autolabel_pipeline"
            ),
            "pipeline_version": "v1.0",
            "created_time": now,
            "updated_time": now,
        },
        "export": {
            "export_format": "labelstudio",
            "export_status": "not_exported",
            "export_uri": None,
            "labelstudio_mapping": None,
        },
    }
    validate_required_fields(sample)
    return sample
