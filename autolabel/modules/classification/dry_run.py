from __future__ import annotations

from typing import Any

from ...sample_factory import touch_workflow


class DryRunClassificationModule:
    """Deterministic classifier for validating DAG shape without model keys."""

    def __init__(self, pipeline_config: dict[str, Any], module_config: dict[str, Any] | None = None) -> None:
        self.pipeline_config = pipeline_config
        self.module_config = module_config or {}
        self.labels = self.module_config.get(
            "labels",
            [
                {"label_key": "helmet", "label_value": "unknown"},
                {"label_key": "reflective_vest", "label_value": "unknown"},
                {"label_key": "phone_usage", "label_value": "unknown"},
            ],
        )

    def classify_sample(
        self,
        sample: dict[str, Any],
        skip_source_types: set[str] | None = None,
    ) -> dict[str, Any]:
        skip_source_types = skip_source_types or {"generated"}
        if sample["image_asset"]["source_type"] in skip_source_types:
            return sample

        for obj in sample.get("objects", []):
            obj["classification"] = {
                "multi_labels": [
                    {
                        "label_key": item["label_key"],
                        "label_value": item["label_value"],
                        "confidence": item.get("confidence"),
                        "evidence": item.get("evidence", "dry-run classification placeholder"),
                    }
                    for item in self.labels
                ],
                "classifier_type": "rule",
                "classifier_name": "dry_run_rule_classifier",
                "classifier_version": "v1",
                "prompt_version": "dry_run",
                "raw_response": {
                    "dry_run": True,
                    "reason": "model key is not configured; validating DAG and data contract only",
                },
            }
        touch_workflow(sample, "classified")
        return sample
