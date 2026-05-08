from __future__ import annotations

import json
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..config_loader import load_config
from ..model_config import deep_merge
from ..sample_factory import make_box, make_object
from ..utils import read_json
from .vlm_labelstudio_detector import VLMLabelStudioDetector


def load_detector_config(path: str | Path) -> dict[str, Any]:
    config = load_config(path)
    return config.get("detector_services", config)


class DetectorServiceClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_profiles = config.get("model_profiles", {})
        self.services = {
            key: self._resolve_service(service)
            for key, service in config.get("services", {}).items()
        }

    def _resolve_service(self, service: dict[str, Any]) -> dict[str, Any]:
        model_ref = service.get("model_ref")
        if not model_ref:
            return service
        profile = self.model_profiles.get(model_ref)
        if not isinstance(profile, dict):
            raise KeyError(f"Geometry model profile not found: {model_ref}")
        merged = deep_merge(profile, service)
        merged["model_ref"] = model_ref
        return merged

    def detect(self, image_uri: str, task_key: str | None = None) -> list[dict[str, Any]]:
        key = task_key or self.config.get("default_task_key")
        if not key:
            raise ValueError("task_key is required when detector config has no default_task_key")
        service = self.services.get(key)
        if service is None:
            raise KeyError(f"Detector service is not configured for task_key={key}")

        if service.get("backend") == "vlm_labelstudio_detector" or service.get("service_type") == "vlm_detector":
            return VLMLabelStudioDetector(service).detect(image_uri)
        if service.get("command"):
            payload = self._run_local_command(service, image_uri)
        else:
            payload = self._post_http(service, image_uri, key)
        return self._normalize_objects(payload, service)

    def _post_http(self, service: dict[str, Any], image_uri: str, task_key: str) -> dict[str, Any]:
        endpoint = service.get("endpoint")
        if not endpoint:
            raise ValueError("Detector service endpoint is required")
        request_cfg = service.get("request", {})
        image_field = request_cfg.get("image_field", "image_uri")
        payload = dict(request_cfg.get("extra_payload", {}))
        payload[image_field] = image_uri
        payload.setdefault("task_key", task_key)

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method=service.get("method", "POST"),
        )
        timeout = float(service.get("timeout_seconds", 60))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Detector service call failed: {endpoint}") from exc

    def _run_local_command(self, service: dict[str, Any], image_uri: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="autolabel_detector_") as tmpdir:
            output_json = Path(tmpdir) / "detector_output.json"
            cmd = [
                part.format(image_uri=image_uri, output_json=str(output_json))
                for part in service["command"]
            ]
            completed = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=False,
                timeout=float(service.get("timeout_seconds", 120)),
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Local detector command failed "
                    f"with code {completed.returncode}: {completed.stderr.strip()}"
                )
            if output_json.exists():
                return read_json(output_json)
            try:
                return json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Local detector did not write JSON output.") from exc

    def _normalize_objects(self, payload: dict[str, Any], service: dict[str, Any]) -> list[dict[str, Any]]:
        response_cfg = service.get("response", {})
        response_format = response_cfg.get("format", "boxes")
        if response_format == "objects":
            items = payload.get(response_cfg.get("objects_field", "objects"), [])
        else:
            items = payload.get(response_cfg.get("boxes_field", "boxes"), [])

        normalized = []
        for idx, item in enumerate(items, start=1):
            normalized.append(self._normalize_item(item, idx, service, response_cfg))
        return normalized

    def _normalize_item(
        self,
        item: dict[str, Any],
        idx: int,
        service: dict[str, Any],
        response_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        label_field = response_cfg.get("label_field", "label")
        bbox_field = response_cfg.get("bbox_field", "bbox")
        confidence_field = response_cfg.get("confidence_field", "score")
        object_type_map = service.get("object_type_map", {})

        object_type = item.get("object_type") or object_type_map.get(item.get(label_field), item.get(label_field))
        if not object_type:
            object_type = service.get("default_object_type", "object")

        raw_box = item.get("box") or item.get(bbox_field)
        box = normalize_box(raw_box)
        confidence = item.get("confidence", item.get(confidence_field))
        if confidence is not None:
            confidence = float(confidence)

        geometry_detail = {
            "polygon": item.get("polygon"),
            "mask_uri": item.get("mask_uri"),
            "mask_format": item.get("mask_format"),
            "generation_params": None,
        }
        geometry_model = {
            "model_name": service.get("model_name"),
            "model_version": service.get("model_version"),
            "confidence": confidence,
        }
        return make_object(
            object_id=item.get("object_id") or f"obj_{idx:06d}",
            object_type=object_type,
            box=box,
            geometry_source=service.get("geometry_source", "detector"),
            geometry_model=geometry_model,
            geometry_detail=geometry_detail,
        )


def normalize_box(raw_box: Any) -> dict[str, int | str]:
    if isinstance(raw_box, dict):
        return make_box(raw_box["x1"], raw_box["y1"], raw_box["x2"], raw_box["y2"])
    if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
        return make_box(raw_box[0], raw_box[1], raw_box[2], raw_box[3])
    raise ValueError(f"Unsupported bbox shape: {raw_box!r}")
