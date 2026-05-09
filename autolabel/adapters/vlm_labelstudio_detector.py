from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..sample_factory import make_box, make_object
from ..utils import get_image_size, image_to_data_url


PLACEHOLDER_IDS = {
    "",
    "自动生成的唯一ID",
    "unique_id",
    "auto_id",
    "id",
}

JSON_RETRY_PROMPT = """上一次输出不是合法 JSON。请重新分析同一张图，只输出下面这种 JSON 数组，不要解释、不要 markdown、不要自然语言：
[
  {
    "data": {"image": "image.jpg"},
    "predictions": [
      {
        "model_version": "vlm-pre-annotation-v1",
        "score": 0.95,
        "result": []
      }
    ]
  }
]
如果没有检测到人员，必须输出 result 为空数组。"""


class VLMJsonParseError(ValueError):
    """Raised when the VLM detector does not return parseable Label Studio JSON."""


def preview_text(text: str, limit: int = 300) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    if len(value) > limit:
        return value[: limit - 3].rstrip() + "..."
    return value


def strip_json_text(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(value):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(value[idx:])
        except json.JSONDecodeError:
            continue
        return value[idx : idx + end].strip()
    raise ValueError("VLM detector output does not contain JSON.")


def parse_json_output(text: str) -> Any:
    return json.loads(strip_json_text(text))


def parse_detector_payload(
    text: str,
    *,
    fail_on_parse_error: bool = False,
    log_error: bool = True,
) -> Any | None:
    try:
        return parse_json_output(text)
    except (ValueError, json.JSONDecodeError) as exc:
        if fail_on_parse_error:
            raise VLMJsonParseError(
                "VLM detector output does not contain parseable Label Studio JSON. "
                f"Preview: {preview_text(text)}"
            ) from exc
        if log_error:
            print(f"  !! VLM 检测 JSON 解析失败: {exc}")
            print(f"  !! VLM 检测原始返回前300字符: {preview_text(text)}")
        return None


def percent_box_to_xyxy(value: dict[str, Any], width: int, height: int) -> dict[str, int | str]:
    x_pct = float(value["x"])
    y_pct = float(value["y"])
    w_pct = float(value["width"])
    h_pct = float(value["height"])
    x1 = round(width * x_pct / 100.0)
    y1 = round(height * y_pct / 100.0)
    x2 = round(width * (x_pct + w_pct) / 100.0)
    y2 = round(height * (y_pct + h_pct) / 100.0)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return make_box(x1, y1, x2, y2)


def stable_object_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:06d}"


def labelstudio_payload_to_objects(
    payload: Any,
    image_uri: str,
    width: int,
    height: int,
    service: dict[str, Any],
    raw_response: Any | None = None,
) -> list[dict[str, Any]]:
    tasks = payload if isinstance(payload, list) else [payload]
    object_type_map = service.get("object_type_map", {})
    default_object_type = service.get("default_object_type", "person")
    object_id_prefix = service.get("object_id_prefix", "obj")
    geometry_model_name = service.get("model_name")
    geometry_model_version = service.get("model_version")
    objects: list[dict[str, Any]] = []

    for task in tasks:
        predictions = task.get("predictions", []) if isinstance(task, dict) else []
        for prediction in predictions:
            prediction_score = prediction.get("score")
            for result in prediction.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue
                value = result.get("value", {})
                labels = value.get("rectanglelabels") or []
                label = labels[0] if labels else service.get("target_label", "Person")
                object_type = object_type_map.get(label, default_object_type)
                box = percent_box_to_xyxy(value, width, height)
                confidence = result.get("score", prediction_score)
                if confidence is not None:
                    confidence = float(confidence)
                object_idx = len(objects) + 1
                source_result_id = result.get("id")
                object_id = (
                    source_result_id
                    if isinstance(source_result_id, str) and source_result_id.strip() not in PLACEHOLDER_IDS
                    else stable_object_id(object_id_prefix, object_idx)
                )
                objects.append(
                    make_object(
                        object_id=object_id,
                        object_type=object_type,
                        box=box,
                        geometry_source=service.get("geometry_source", "detector"),
                        geometry_model={
                            "model_name": geometry_model_name,
                            "model_version": geometry_model_version,
                            "confidence": confidence,
                        },
                        geometry_detail={
                            "polygon": None,
                            "mask_uri": None,
                            "mask_format": None,
                            "generation_params": {
                                "detector_backend": "vlm_labelstudio_detector",
                                "output_contract": "AutoLabelSample.objects[]",
                                "source_format": "labelstudio_percent_rectangle",
                                "prompt_version": service.get("prompt_version"),
                                "target_label": label,
                                "source_result_id": source_result_id,
                                "labelstudio_percent_box": value,
                                "converted_xyxy_box": box,
                                "image_width": width,
                                "image_height": height,
                                "source_image_uri": image_uri,
                                "raw_response": raw_response,
                            },
                        },
                    )
                )
    return objects


class VLMLabelStudioDetector:
    def __init__(self, service: dict[str, Any]) -> None:
        self.service = service

    def detect(self, image_uri: str) -> list[dict[str, Any]]:
        if self.service.get("dry_run"):
            return self._detect_dry_run(image_uri)

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is required for vlm_labelstudio_detector.") from exc

        api_key = self.service.get("api_key")
        base_url = self.service.get("base_url") or self.service.get("api_url")
        model_name = self.service.get("model_name")
        if not api_key:
            raise RuntimeError("VLM detector api_key is not configured.")
        if not model_name:
            raise RuntimeError("VLM detector model_name is not configured.")

        width, height = get_image_size(image_uri)
        image_url = self._image_url(image_uri)
        prompt = self.service.get("prompt")
        if not prompt:
            raise RuntimeError("VLM detector prompt is not configured.")

        client = OpenAI(api_key=api_key, base_url=base_url)
        raw_text = self._request_json_text(client, model_name, image_url, prompt)
        payload = parse_detector_payload(raw_text, log_error=False)
        if payload is None:
            retry_count = int(self.service.get("parse_retry_count", 1))
            retry_prompt = self.service.get("json_retry_prompt") or JSON_RETRY_PROMPT
            for attempt in range(1, retry_count + 1):
                print(f"  !! VLM 检测输出非 JSON，正在重试 {attempt}/{retry_count}")
                raw_text = self._request_json_text(client, model_name, image_url, retry_prompt)
                payload = parse_detector_payload(raw_text, log_error=False)
                if payload is not None:
                    break

        if payload is None:
            if self.service.get("fail_on_parse_error", False):
                parse_detector_payload(raw_text, fail_on_parse_error=True)
            print("  !! VLM 检测 JSON 解析失败，已返回空框并继续处理后续样本")
            print(f"  !! VLM 检测原始返回前300字符: {preview_text(raw_text)}")
            return []

        return labelstudio_payload_to_objects(
            payload=payload,
            image_uri=image_uri,
            width=width,
            height=height,
            service=self.service,
            raw_response=payload if self.service.get("store_raw_response", False) else None,
        )

    def _image_url(self, image_uri: str) -> str:
        if image_uri.startswith("http://") or image_uri.startswith("https://") or image_uri.startswith("data:"):
            return image_uri
        return image_to_data_url(Path(image_uri))

    def _request_json_text(self, client: Any, model_name: str, image_url: str, prompt: str) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": self.service.get(
                        "system_message",
                        "You are a precise computer vision annotation engine. Return valid JSON only. "
                        "Do not output prose, markdown, or explanations.",
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                },
            ],
            temperature=float(self.service.get("temperature", 0.0)),
            max_tokens=int(self.service.get("max_tokens", 2000)),
        )
        return response.choices[0].message.content or ""

    def _detect_dry_run(self, image_uri: str) -> list[dict[str, Any]]:
        width, height = get_image_size(image_uri)
        payload = [
            {
                "data": {
                    "image": Path(image_uri).name,
                },
                "predictions": [
                    {
                        "model_version": self.service.get("model_version") or "vlm-pre-annotation-v1",
                        "score": float(self.service.get("dry_run_score", 0.5)),
                        "result": [
                            {
                                "id": "自动生成的唯一ID",
                                "type": "rectanglelabels",
                                "from_name": "label",
                                "to_name": "image",
                                "image_rotation": 0,
                                "value": {
                                    "rotation": 0,
                                    "x": float(self.service.get("dry_run_x", 20.0)),
                                    "y": float(self.service.get("dry_run_y", 10.0)),
                                    "width": float(self.service.get("dry_run_width", 45.0)),
                                    "height": float(self.service.get("dry_run_height", 80.0)),
                                    "rectanglelabels": [
                                        self.service.get("target_label", "Person"),
                                    ],
                                },
                            }
                        ],
                    }
                ],
            }
        ]
        return labelstudio_payload_to_objects(
            payload=payload,
            image_uri=image_uri,
            width=width,
            height=height,
            service=self.service,
            raw_response=payload if self.service.get("store_raw_response", False) else None,
        )
