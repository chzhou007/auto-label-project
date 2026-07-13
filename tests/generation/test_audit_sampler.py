from __future__ import annotations

import csv
from pathlib import Path

from autolabel.modules.generation.audit_sampler import write_audit_sample_csv


def test_write_audit_sample_csv_uses_design_fields(tmp_path) -> None:
    output_path = tmp_path / "audit_sample_list_test.csv"
    written = write_audit_sample_csv(
        [
            {
                "task_id": "task_001",
                "sample_id": "sample_001",
                "object_id": "obj_001",
                "image_uri": "data/raw/original.jpg",
                "generated_image_uri": "metadata/images/generated.jpg",
                "localizer_used": "rgb_diff",
                "localizer": "pgcd_lpips",
                "attempt_role": "sidecar",
                "success": True,
                "passes_quality": True,
                "quality_reason": None,
                "reason": "quality_failed:prompt_iou_below_threshold",
                "failure_reason": "quality_failed:prompt_iou_below_threshold",
                "fallback_used": False,
                "final_bbox": [10, 12, 40, 44],
                "mask_uri": "masks/sample_001_obj_001_mask.png",
                "crop_uri": "crops/sample_001_obj_001.jpg",
                "elapsed_ms": 15.2,
                "anomaly_type": "water_leak",
                "background_preservation_score": 0.91,
                "anomaly_visibility_score": 0.67,
                "pgcd_component_score": 0.72,
                "manual_accept": True,
                "manual_comment": "looks good",
            }
        ],
        output_path,
        sampling_ratio=1.0,
        min_samples=1,
    )

    assert written == output_path
    with output_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows and len(rows) == 1
    row = rows[0]
    assert row["task_id"] == "task_001"
    assert row["image_uri"] == "data/raw/original.jpg"
    assert row["generated_image_uri"] == "metadata/images/generated.jpg"
    assert row["localizer_used"] == "rgb_diff"
    assert row["fallback_used"] == "False"
    assert row["final_bbox"] == "[10, 12, 40, 44]"
    assert row["mask_uri"] == "masks/sample_001_obj_001_mask.png"
    assert row["crop_uri"] == "crops/sample_001_obj_001.jpg"
    assert row["failure_reason"] == "quality_failed:prompt_iou_below_threshold"
    assert row["manual_label"] == "accepted"
    assert row["manual_accept"] == "True"
    assert row["manual_comment"] == "looks good"
