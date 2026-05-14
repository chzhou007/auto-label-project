from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..sample_factory import make_box, make_object
from ..utils import get_image_size, image_to_data_url_with_dimensions


PLACEHOLDER_IDS = {
    "",
    "自动生成的唯一ID",
    "unique_id",
    "auto_id",
    "id",
}

JSON_RETRY_PROMPT = """上一次输出不是合法 JSON。请重新分析同一张图，只输出下面这种 JSON 对象，不要解释、不要 markdown、不要自然语言：
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
如果没有检测到人员，必须输出 result 为空数组。"""


class VLMJsonParseError(ValueError):
    """Raised when the VLM detector does not return parseable Label Studio JSON."""


def preview_text(text: str, limit: int = 300) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    if len(value) > limit:
        return value[: limit - 3].rstrip() + "..."
    return value


def bool_config(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


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


def scale_xyxy_box(
    box: dict[str, int | str],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> dict[str, int | str]:
    if source_width == target_width and source_height == target_height:
        return box
    x_scale = target_width / float(source_width)
    y_scale = target_height / float(source_height)
    x1 = max(0, min(target_width - 1, round(int(box["x1"]) * x_scale)))
    y1 = max(0, min(target_height - 1, round(int(box["y1"]) * y_scale)))
    x2 = max(x1 + 1, min(target_width, round(int(box["x2"]) * x_scale)))
    y2 = max(y1 + 1, min(target_height, round(int(box["y2"]) * y_scale)))
    return make_box(x1, y1, x2, y2)


def pixel_rect_to_xyxy(
    value: dict[str, Any],
    source_width: int,
    source_height: int,
    target_width: int | None = None,
    target_height: int | None = None,
) -> dict[str, int | str]:
    target_width = target_width or source_width
    target_height = target_height or source_height
    x = float(value["x"])
    y = float(value["y"])
    w = float(value["width"])
    h = float(value["height"])
    x1 = max(0, min(source_width - 1, round(x)))
    y1 = max(0, min(source_height - 1, round(y)))
    x2 = max(x1 + 1, min(source_width, round(x + w)))
    y2 = max(y1 + 1, min(source_height, round(y + h)))
    return scale_xyxy_box(make_box(x1, y1, x2, y2), source_width, source_height, target_width, target_height)


def corner_box_to_xyxy(
    value: dict[str, Any],
    width: int,
    height: int,
    request_width: int | None = None,
    request_height: int | None = None,
) -> dict[str, int | str]:
    raw_x1 = float(value["x1"])
    raw_y1 = float(value["y1"])
    raw_x2 = float(value["x2"])
    raw_y2 = float(value["y2"])
    looks_percent = max(abs(raw_x1), abs(raw_y1), abs(raw_x2), abs(raw_y2)) <= 100.0
    if looks_percent:
        raw_x1 = width * raw_x1 / 100.0
        raw_x2 = width * raw_x2 / 100.0
        raw_y1 = height * raw_y1 / 100.0
        raw_y2 = height * raw_y2 / 100.0
        x1 = max(0, min(width - 1, round(min(raw_x1, raw_x2))))
        y1 = max(0, min(height - 1, round(min(raw_y1, raw_y2))))
        x2 = max(x1 + 1, min(width, round(max(raw_x1, raw_x2))))
        y2 = max(y1 + 1, min(height, round(max(raw_y1, raw_y2))))
        return make_box(x1, y1, x2, y2)

    source_width = request_width or width
    source_height = request_height or height
    x1 = max(0, min(source_width - 1, round(min(raw_x1, raw_x2))))
    y1 = max(0, min(source_height - 1, round(min(raw_y1, raw_y2))))
    x2 = max(x1 + 1, min(source_width, round(max(raw_x1, raw_x2))))
    y2 = max(y1 + 1, min(source_height, round(max(raw_y1, raw_y2))))
    return scale_xyxy_box(make_box(x1, y1, x2, y2), source_width, source_height, width, height)


def labelstudio_value_to_xyxy(
    value: dict[str, Any],
    width: int,
    height: int,
    *,
    coordinate_units: str = "auto",
    auto_detect_coordinate_units: bool = True,
    request_width: int | None = None,
    request_height: int | None = None,
) -> tuple[dict[str, int | str], str]:
    if {"x1", "y1", "x2", "y2"}.issubset(value):
        box = corner_box_to_xyxy(value, width, height, request_width=request_width, request_height=request_height)
        unit = "percent_xyxy" if max(abs(float(value[key])) for key in ("x1", "y1", "x2", "y2")) <= 100.0 else "pixel_xyxy"
        return box, unit

    if not {"x", "y", "width", "height"}.issubset(value):
        raise ValueError(f"Missing rectangle fields in VLM output: {value!r}")

    x = float(value["x"])
    y = float(value["y"])
    w = float(value["width"])
    h = float(value["height"])
    coordinate_units = (coordinate_units or "auto").lower()
    source_width = request_width or width
    source_height = request_height or height
    if coordinate_units in {"pixel", "pixels", "xywh_pixel"}:
        return pixel_rect_to_xyxy(value, source_width, source_height, width, height), "pixel_xywh_configured"
    if coordinate_units in {"percent", "percentage", "labelstudio"}:
        return percent_box_to_xyxy(value, width, height), "labelstudio_percent_xywh"

    looks_pixel = auto_detect_coordinate_units and (x > 100.0 or y > 100.0 or w > 100.0 or h > 100.0)
    if looks_pixel:
        return pixel_rect_to_xyxy(value, source_width, source_height, width, height), "pixel_xywh_auto_detected"
    return percent_box_to_xyxy(value, width, height), "labelstudio_percent_xywh"


def stable_object_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:06d}"


def resized_dimensions(width: int, height: int, max_side: int | None) -> tuple[int, int]:
    if max_side is None or max_side <= 0 or max(width, height) <= max_side:
        return width, height
    scale = max_side / float(max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def labelstudio_payload_to_objects(
    payload: Any,
    image_uri: str,
    width: int,
    height: int,
    service: dict[str, Any],
    raw_response: Any | None = None,
    request_width: int | None = None,
    request_height: int | None = None,
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
                box, coordinate_unit = labelstudio_value_to_xyxy(
                    value,
                    width,
                    height,
                    coordinate_units=str(service.get("coordinate_units", "auto")),
                    auto_detect_coordinate_units=bool_config(
                        service.get("auto_detect_coordinate_units"),
                        default=True,
                    ),
                    request_width=request_width,
                    request_height=request_height,
                )
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
                                "source_format": "labelstudio_rectangle",
                                "coordinate_unit": coordinate_unit,
                                "prompt_version": service.get("prompt_version"),
                                "target_label": label,
                                "source_result_id": source_result_id,
                                "labelstudio_percent_box": value,
                                "converted_xyxy_box": box,
                                "image_width": width,
                                "image_height": height,
                                "request_image_width": request_width or width,
                                "request_image_height": request_height or height,
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

        image_url, width, height, request_width, request_height = self._image_payload(image_uri)
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
            request_width=request_width,
            request_height=request_height,
        )

    def _image_payload(self, image_uri: str) -> tuple[str, int, int, int, int]:
        if image_uri.startswith("http://") or image_uri.startswith("https://") or image_uri.startswith("data:"):
            width, height = get_image_size(image_uri)
            return image_uri, width, height, width, height
        max_side = self.service.get("request_image_max_side")
        max_side = int(max_side) if max_side not in (None, "") else None
        image_url, width, height, request_width, request_height = image_to_data_url_with_dimensions(
            Path(image_uri),
            max_side=max_side,
        )
        return image_url, width, height, request_width, request_height

    def _request_json_text(self, client: Any, model_name: str, image_url: str, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": self.service.get(
                    "system_message",
                    "You are a strict JSON API for computer-vision annotation. Return valid JSON only. "
                    "Do not output prose, markdown, explanations, analysis, or thinking process.",
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
        ]

        def create(with_response_format: bool) -> Any:
            kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": float(self.service.get("temperature", 0.0)),
                "max_tokens": max(1, int(self.service.get("max_tokens", 2000))),
            }
            if with_response_format:
                kwargs["response_format"] = {
                    "type": self.service.get("response_format_type", "json_object"),
                }
            return client.chat.completions.create(**kwargs)

        if bool_config(self.service.get("use_response_format"), default=False):
            try:
                response = create(True)
            except Exception as exc:
                message = str(exc).lower()
                if "response_format" not in message and "json_object" not in message and "guided" not in message:
                    raise
                response = create(False)
        else:
            response = create(False)
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
