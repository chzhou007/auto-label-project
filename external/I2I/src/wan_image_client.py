from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from config import DashScopeConfig, I2IServiceConfig, ModelServiceConfig
from prompts import NEGATIVE_PROMPT
from utils import cv2_imread, cv2_imwrite, image_to_data_url, redact_headers, write_json

logger = logging.getLogger(__name__)


def _save_base64_image(data: str, output_path: str) -> None:
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    image_bytes = base64.b64decode(data)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(image_bytes)


def _extract_image_url_or_base64(response: dict[str, Any]) -> str:
    def walk(value: Any) -> str | None:
        if isinstance(value, dict):
            for key in ("image", "image_url", "url", "image_base64", "b64_json"):
                item = value.get(key)
                if isinstance(item, str) and item:
                    return item
            for item in value.values():
                found = walk(item)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = walk(item)
                if found:
                    return found
        elif isinstance(value, str) and (
            value.startswith("http://")
            or value.startswith("https://")
            or value.startswith("data:image/")
        ):
            return value
        return None

    found = walk(response)
    if found:
        return found
    output = response.get("output", {})
    for key in ("image_url", "url", "image", "image_base64", "b64_json"):
        value = output.get(key)
        if isinstance(value, str) and value:
            return value
    results = output.get("results") or output.get("images")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            for key in ("url", "image_url", "image", "image_base64", "b64_json"):
                value = first.get(key)
                if isinstance(value, str) and value:
                    return value
    raise ValueError(f"unable to extract generated image from response: {response}")


def _download_or_decode_image(value: str, output_path: str) -> None:
    if value.startswith("http://") or value.startswith("https://"):
        resp = requests.get(value, timeout=180)
        resp.raise_for_status()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(resp.content)
    else:
        _save_base64_image(value, output_path)


def _dry_run_edit(image_path: str, bbox: tuple[int, int, int, int], anomaly_type: str, output_path: str) -> None:
    image = cv2_imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    center = (x1 + int(w * 0.52), y1 + int(h * 0.62))
    axes = (max(12, int(w * 0.12)), max(8, int(h * 0.06)))
    overlay = image.copy()
    if anomaly_type == "oil_leak":
        color = (20, 18, 16)
        alpha = 0.62
    elif anomaly_type == "water_leak":
        color = (210, 235, 245)
        alpha = 0.32
    elif anomaly_type == "coolant_leak":
        color = (150, 210, 120)
        alpha = 0.45
    else:
        color = (80, 190, 225)
        alpha = 0.38
    cv2.ellipse(overlay, center, axes, -12, 0, 360, color, -1)
    if anomaly_type != "oil_leak":
        cv2.line(
            overlay,
            (center[0] - axes[0] // 2, max(y1, center[1] - axes[1] * 4)),
            center,
            color,
            max(2, min(w, h) // 80),
        )
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2_imwrite(output_path, image)


def _service_endpoint(config: ModelServiceConfig | DashScopeConfig) -> str:
    endpoint = getattr(config, "endpoint", None) or getattr(config, "wan_endpoint", None)
    if not endpoint:
        raise ValueError("image generation endpoint is required")
    return str(endpoint)


def _is_seedream_provider(config: ModelServiceConfig | DashScopeConfig, endpoint: str, model: str) -> bool:
    provider = str(getattr(config, "provider", "") or "").lower()
    if provider in {"volcengine_ark", "ark", "seedream"}:
        return True
    marker = f"{endpoint} {model}".lower()
    return "seedream" in marker or "ark.cn-beijing.volces.com" in marker or "/images/generations" in marker


def _seedream_prompt(prompt: str, bbox: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    return (
        f"{prompt}\n\n"
        "Edit region constraint: use the clean input image as the source image and edit only inside "
        f"pixel bbox [x1={x1}, y1={y1}, x2={x2}, y2={y2}]. Keep the camera, background, equipment, "
        "lighting, timestamp, and all content outside this box unchanged. The final anomaly must be a "
        "small realistic early-stage leak located inside this box."
    )


def _with_seedream_image_input(payload: dict[str, Any], image_data_url: str) -> dict[str, Any]:
    field = os.getenv("SEEDREAM_IMAGE_FIELD", "image_urls").strip() or "image_urls"
    if field == "image":
        payload["image"] = image_data_url
    elif field == "image_url":
        payload["image_url"] = image_data_url
    else:
        payload["image_urls"] = [image_data_url]
    return payload


class WanImageClient:
    def __init__(self, model: str, dashscope_config: ModelServiceConfig | DashScopeConfig, dry_run: bool = False):
        self.model = model
        self.config = dashscope_config
        self.dry_run = dry_run

    def _build_request_payloads(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str,
        bbox: tuple[int, int, int, int],
    ) -> tuple[str, dict[str, Any], dict[str, Any], bool]:
        endpoint = _service_endpoint(self.config)
        full_prompt = prompt
        if negative_prompt:
            full_prompt = f"{prompt}\n\nNegative prompt: {negative_prompt}"

        if _is_seedream_provider(self.config, endpoint, self.model):
            seedream_prompt = _seedream_prompt(full_prompt, bbox)
            request_payload: dict[str, Any] = {
                "model": self.model,
                "prompt": seedream_prompt,
                "n": int(os.getenv("SEEDREAM_N", "1")),
                "watermark": False,
            }
            response_format = os.getenv("SEEDREAM_RESPONSE_FORMAT")
            if response_format:
                request_payload["response_format"] = response_format
            size = os.getenv("SEEDREAM_SIZE")
            if size:
                request_payload["size"] = size
            request_payload = _with_seedream_image_input(request_payload, image_to_data_url(image_path))

            request_log_payload = dict(request_payload)
            for image_key in ("image", "image_url"):
                if image_key in request_log_payload:
                    request_log_payload[image_key] = image_path
            if "image_urls" in request_log_payload:
                request_log_payload["image_urls"] = [image_path]
            request_log_payload["edit_bbox"] = list(bbox)
            return endpoint, request_payload, request_log_payload, True

        request_payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": image_to_data_url(image_path)},
                            {"text": full_prompt},
                        ],
                    }
                ]
            },
            "parameters": {
                "bbox_list": [[list(bbox)]],
                "n": 1,
                "watermark": False,
                "thinking_mode": True,
            },
        }
        request_log_payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image_path": image_path},
                            {"text": full_prompt},
                        ],
                    }
                ]
            },
            "parameters": {
                "bbox_list": [[list(bbox)]],
                "n": 1,
                "watermark": False,
                "thinking_mode": True,
            },
        }
        return endpoint, request_payload, request_log_payload, False

    def edit_image_with_wan(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str,
        bbox: tuple[int, int, int, int],
        output_path: str,
        anomaly_type: str,
        request_log_path: str | None = None,
        response_log_path: str | None = None,
    ) -> dict[str, Any]:
        endpoint, request_payload, request_log_payload, is_seedream = self._build_request_payloads(
            image_path, prompt, negative_prompt, bbox
        )
        if request_log_path:
            write_json(
                request_log_path,
                {"endpoint": endpoint, "provider": self.config.provider, "body": request_log_payload},
            )

        if self.dry_run:
            _dry_run_edit(image_path, bbox, anomaly_type, output_path)
            response = {"dry_run": True, "output_path": output_path, "bbox_list": [[list(bbox)]]}
            if response_log_path:
                write_json(response_log_path, response)
            return response

        if not self.config.api_key:
            raise RuntimeError("image generation API key is required unless --dry-run is used; set ARK_API_KEY")

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if request_log_path:
            write_json(
                request_log_path,
                {
                    "endpoint": endpoint,
                    "provider": self.config.provider,
                    "headers": redact_headers(headers),
                    "body": request_log_payload,
                },
            )
        response = requests.post(endpoint, headers=headers, json=request_payload, timeout=180)
        raw: dict[str, Any]
        try:
            raw = response.json()
        except Exception:
            raw = {"status_code": response.status_code, "text": response.text}
        if not response.ok:
            if response_log_path:
                write_json(response_log_path, raw)
            raise RuntimeError(f"image generation request failed: HTTP {response.status_code}")

        final_response = self._resolve_async_if_needed(raw, headers, is_seedream)
        image_value = _extract_image_url_or_base64(final_response)
        _download_or_decode_image(image_value, output_path)
        if response_log_path:
            write_json(response_log_path, final_response)
        return final_response

    def _resolve_async_if_needed(
        self,
        raw: dict[str, Any],
        headers: dict[str, str],
        is_seedream: bool = False,
    ) -> dict[str, Any]:
        task_id = raw.get("output", {}).get("task_id") or raw.get("task_id")
        if not task_id:
            return raw
        if is_seedream:
            template = os.getenv("SEEDREAM_TASK_URL_TEMPLATE")
            if not template:
                return raw
            task_url = template.format(task_id=task_id)
            for _ in range(60):
                time.sleep(5)
                poll = requests.get(task_url, headers=headers, timeout=60)
                data = poll.json()
                status = data.get("status") or data.get("output", {}).get("task_status") or data.get("task_status")
                if status in {"SUCCEEDED", "succeeded", "SUCCESS", "success", "completed"}:
                    return data
                if status in {"FAILED", "failed", "CANCELED", "canceled", "cancelled"}:
                    raise RuntimeError(f"Seedream async task failed: {data}")
            raise TimeoutError(f"Seedream async task timed out: {task_id}")

        endpoint = _service_endpoint(self.config)
        base = endpoint.split("/api/v1/", 1)[0]
        task_url = f"{base}/api/v1/tasks/{task_id}"
        for _ in range(60):
            time.sleep(5)
            poll = requests.get(task_url, headers=headers, timeout=60)
            data = poll.json()
            status = data.get("output", {}).get("task_status") or data.get("task_status")
            if status in {"SUCCEEDED", "succeeded", "SUCCESS"}:
                return data
            if status in {"FAILED", "failed", "CANCELED", "canceled"}:
                raise RuntimeError(f"image generation async task failed: {data}")
        raise TimeoutError(f"image generation async task timed out: {task_id}")


def edit_image_with_wan(
    image_path: str,
    prompt: str,
    negative_prompt: str,
    bbox: tuple[int, int, int, int],
    output_path: str,
) -> dict[str, Any]:
    client = WanImageClient("doubao-seedream-5.0-lite", I2IServiceConfig.from_env().image)
    return client.edit_image_with_wan(image_path, prompt, negative_prompt or NEGATIVE_PROMPT, bbox, output_path, "")
