from __future__ import annotations

from pathlib import Path
from typing import Any

from autolabel.sample_factory import make_box, make_object, make_sample

from .grid import BBox


def build_autolabel_sample(
    task: dict[str, Any],
    generated_image_path: str | Path,
    original_image_path: str | Path,
    image_width: int,
    image_height: int,
    final_bbox: BBox,
    crop_info: dict[str, Any],
    mask_path: str | Path,
    generation_params: dict[str, Any],
) -> dict[str, Any]:
    sample_id = str(task.get("sample_id") or f"sample_{task['task_id']}")
    row = dict(task)
    row["sample_id"] = sample_id
    row["image_id"] = str(task.get("image_id") or f"generated_{task['task_id']}")
    row["image_uri"] = str(generated_image_path)
    row["source_type"] = "generated"
    row.setdefault("room_type", "generator_room")
    row["generation_prompt"] = generation_params.get("wan_prompt")
    row["generation_model"] = generation_params.get("image_generation_model")

    sample = make_sample(
        row,
        width=image_width,
        height=image_height,
        pipeline_id="vlm_grid_wan_diff_autolabel",
        pipeline_version="v1.0",
        qc_policy=None,
        export_config={"export_format": "labelstudio", "export_status": "not_exported"},
        workflow_status="classified",
    )
    generation_params = dict(generation_params)
    generation_params.setdefault("clean_original_image_uri", str(original_image_path))
    generation_params.setdefault("final_bbox_source", "image_difference_connected_components")

    obj = make_object(
        object_id=f"{sample_id}_obj_000001",
        object_type="leakage_area",
        box=make_box(*final_bbox),
        geometry_source="synthetic_generator",
        geometry_model={
            "model_name": "diff_localizer",
            "model_version": "v1.0",
            "confidence": float(generation_params.get("anomaly_visibility_score", 0.0)),
        },
        geometry_detail={
            "polygon": None,
            "mask_uri": str(mask_path),
            "mask_format": "png",
            "generation_params": generation_params,
        },
        crop={
            "crop_id": f"crop_{sample_id}",
            **crop_info,
        },
        classification={
            "multi_labels": [
                {
                    "label_key": "anomaly_type",
                    "label_value": task["anomaly_type"],
                    "confidence": 1.0,
                    "evidence": "Label inherited from generation task configuration.",
                }
            ],
            "classifier_type": "rule",
            "classifier_name": "task_config_labeler",
            "classifier_version": "v1",
            "prompt_version": None,
            "raw_response": None,
        },
    )
    sample["objects"] = [obj]
    return sample
