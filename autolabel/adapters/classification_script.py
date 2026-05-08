from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from ..sample_factory import touch_workflow


BOOLEAN_LABEL_MAP = {
    "safety_harness": ("safety_belt", "wearing_safety_belt", "no_safety_belt"),
    "hard_hat": ("helmet", "wearing_helmet", "no_helmet"),
    "reflective_vest": ("reflective_vest", "wearing_reflective_vest", "no_reflective_vest"),
    "safety_shoes": ("safety_shoes", "wearing_safety_shoes", "no_safety_shoes"),
    "smoking": ("smoking", "smoking", "not_smoking"),
    "falling_down": ("falling", "falling", "not_falling"),
    "sleeping_on_duty": ("sleeping", "sleeping", "not_sleeping"),
    "climbing_over_railing": ("climbing_over_railing", "climbing_over_railing", "not_climbing_over_railing"),
    "touching_equipment": ("touching_equipment", "touching_equipment", "not_touching_equipment"),
    "fighting": ("fighting", "fighting", "not_fighting"),
    "using_phone": ("phone_usage", "using_phone", "not_using_phone"),
    "safety_goggles": ("safety_goggles", "wearing_safety_goggles", "no_safety_goggles"),
}


def load_classification_module(script_path: str | Path) -> ModuleType:
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"classification.py not found: {path}")
    spec = importlib.util.spec_from_file_location("external_classification_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load classification script: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit as exc:
        raise RuntimeError(
            "classification.py exited during import. Check that its dependencies are installed."
        ) from exc
    return module


class ClassificationScriptAdapter:
    def __init__(
        self,
        script_path: str | Path,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        classifier_type: str = "vlm",
        classifier_name: str | None = None,
        classifier_version: str | None = None,
        prompt_version: str | None = None,
        delay_seconds: float = 0.5,
    ) -> None:
        self.script_path = Path(script_path)
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.classifier_type = classifier_type
        self.classifier_name = classifier_name or model or "classification.py"
        self.classifier_version = classifier_version
        self.prompt_version = prompt_version
        self.delay_seconds = delay_seconds
        self._module: ModuleType | None = None
        self._client: Any | None = None

    @classmethod
    def from_config(cls, script_path: str | Path, config: dict[str, Any]) -> "ClassificationScriptAdapter":
        return cls(
            script_path=script_path,
            api_url=config.get("api_url") or os.getenv(config.get("api_url_env", "")) or None,
            api_key=config.get("api_key") or os.getenv(config.get("api_key_env", "")) or None,
            model=config.get("model") or os.getenv(config.get("model_env", "")) or None,
            classifier_type=config.get("classifier_type", "vlm"),
            classifier_name=config.get("classifier_name"),
            classifier_version=config.get("classifier_version") or config.get("model_version"),
            prompt_version=config.get("prompt_version"),
            delay_seconds=float(config.get("delay_seconds", 0.5)),
        )

    @property
    def module(self) -> ModuleType:
        if self._module is None:
            self._module = load_classification_module(self.script_path)
            if self.api_url:
                setattr(self._module, "DEFAULT_API_URL", self.api_url)
            if self.api_key:
                setattr(self._module, "DEFAULT_API_KEY", self.api_key)
            if self.model:
                setattr(self._module, "DEFAULT_MODEL", self.model)
        return self._module

    @property
    def client(self) -> Any:
        if self._client is None:
            module = self.module
            api_key = getattr(module, "DEFAULT_API_KEY", None)
            api_url = getattr(module, "DEFAULT_API_URL", None)
            if not api_key:
                raise RuntimeError("Classification API key is not configured.")
            self._client = module.OpenAI(api_key=api_key, base_url=api_url)
        return self._client

    def classify_crop(self, crop_uri: str | Path) -> dict[str, Any]:
        raw_response = self.module.process_image(str(crop_uri), self.client)
        labels = labels_from_boolean_response(raw_response)
        return {
            "multi_labels": labels,
            "classifier_type": self.classifier_type,
            "classifier_name": self.classifier_name,
            "classifier_version": self.classifier_version,
            "prompt_version": self.prompt_version,
            "raw_response": raw_response if isinstance(raw_response, dict) else {"raw": raw_response},
        }

    def classify_sample(
        self,
        sample: dict[str, Any],
        skip_source_types: set[str] | None = None,
    ) -> dict[str, Any]:
        skip_source_types = skip_source_types or {"generated"}
        source_type = sample["image_asset"]["source_type"]
        if source_type in skip_source_types:
            return sample

        sample_classifier = getattr(self.module, "classify_sample", None)
        if callable(sample_classifier):
            result = sample_classifier(
                sample,
                self.client,
                skip_source_types=skip_source_types,
                classifier_config={
                    "classifier_type": self.classifier_type,
                    "classifier_name": self.classifier_name,
                    "classifier_version": self.classifier_version,
                    "prompt_version": self.prompt_version,
                    "model": self.model,
                    "model_version": self.classifier_version,
                    "delay_seconds": self.delay_seconds,
                },
                delay_seconds=self.delay_seconds,
            )
            if isinstance(result, dict):
                return result
            touch_workflow(sample, "classified")
            return sample

        for obj in sample.get("objects", []):
            crop_uri = obj.get("crop", {}).get("crop_uri")
            if not crop_uri:
                continue
            obj["classification"] = self.classify_crop(crop_uri)
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)
        touch_workflow(sample, "classified")
        return sample


def labels_from_boolean_response(raw_response: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(raw_response, dict) or raw_response.get("error"):
        return []
    notes = raw_response.get("notes") if isinstance(raw_response.get("notes"), dict) else {}
    labels = []
    for source_key, mapping in BOOLEAN_LABEL_MAP.items():
        if source_key not in raw_response:
            continue
        label_key, true_value, false_value = mapping
        value = raw_response[source_key]
        if not isinstance(value, bool):
            continue
        labels.append(
            {
                "label_key": label_key,
                "label_value": true_value if value else false_value,
                "confidence": None,
                "evidence": notes.get(source_key) or "vlm classification",
            }
        )
    return labels
