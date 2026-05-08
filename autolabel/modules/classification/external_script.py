from __future__ import annotations

from pathlib import Path
from typing import Any

from ...adapters.classification_script import ClassificationScriptAdapter
from ...model_config import resolve_classification_runtime


class ExternalScriptClassificationModule:
    """Classification module backed by an existing classification.py script."""

    def __init__(self, pipeline_config: dict[str, Any], module_config: dict[str, Any] | None = None) -> None:
        self.pipeline_config = pipeline_config
        self.module_config = module_config or {}
        script_path = (
            self.module_config.get("script_path")
            or pipeline_config.get("paths", {}).get("classification_script")
            or "classification.py"
        )
        self.adapter = ClassificationScriptAdapter.from_config(
            Path(script_path),
            resolve_classification_runtime(pipeline_config),
        )

    def classify_sample(
        self,
        sample: dict[str, Any],
        skip_source_types: set[str] | None = None,
    ) -> dict[str, Any]:
        return self.adapter.classify_sample(sample, skip_source_types=skip_source_types)
