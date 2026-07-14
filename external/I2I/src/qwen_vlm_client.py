from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import requests

from config import DashScopeConfig, I2IServiceConfig, ModelServiceConfig, VALID_GRIDS
from grid import normalize_grid_id
from prompts import build_vlm_prompt
from utils import image_to_data_url, redact_headers, write_json

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _extract_text_from_response(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        if isinstance(content, str):
            return content

    output = response.get("output", {})
    choices = output.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        if isinstance(content, str):
            return content
    text = output.get("text")
    if isinstance(text, str):
        return text
    raise ValueError(f"unable to extract VLM text from response keys: {list(response.keys())}")


def _service_endpoint(config: ModelServiceConfig | DashScopeConfig) -> str:
    endpoint = getattr(config, "endpoint", None) or getattr(config, "vlm_endpoint", None)
    if not endpoint:
        raise ValueError("VLM endpoint is required")
    return str(endpoint)


def _is_openai_compatible(config: ModelServiceConfig | DashScopeConfig, endpoint: str) -> bool:
    provider = str(getattr(config, "provider", "") or "").lower()
    if provider in {"openai_compatible", "openai-compatible", "qwen397b", "qwen"}:
        return True
    normalized = endpoint.rstrip("/")
    return normalized.endswith("/v1") or normalized.endswith("/chat/completions")


def _chat_completions_endpoint(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _use_response_format() -> bool:
    return os.getenv("QWEN397B_USE_RESPONSE_FORMAT", "1").lower() not in {"0", "false", "no"}


def validate_vlm_selection(payload: dict[str, Any]) -> dict[str, Any]:
    selected = normalize_grid_id(str(payload.get("selected_grid", "")))
    candidates = payload.get("top_candidates") or []
    clean_candidates = []
    if isinstance(candidates, list):
        for item in candidates[:3]:
            if not isinstance(item, dict):
                continue
            grid = normalize_grid_id(str(item.get("grid", "")))
            if grid in VALID_GRIDS:
                try:
                    score = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                clean_candidates.append({"grid": grid, "score": max(0.0, min(1.0, score))})

    if selected not in VALID_GRIDS and clean_candidates:
        selected = clean_candidates[0]["grid"]
    if selected not in VALID_GRIDS:
        raise ValueError("VLM response has no valid selected_grid")
    if not clean_candidates:
        clean_candidates = [{"grid": selected, "score": float(payload.get("confidence", 0.0) or 0.0)}]

    try:
        confidence = float(payload.get("confidence", clean_candidates[0]["score"]))
    except (TypeError, ValueError):
        confidence = clean_candidates[0]["score"]

    return {
        "selected_grid": selected,
        "confidence": max(0.0, min(1.0, confidence)),
        "top_candidates": clean_candidates[:3],
        "reason": str(payload.get("reason", "")),
        "edit_region_hint": str(payload.get("edit_region_hint", "")),
    }


class QwenVLMClient:
    def __init__(self, model: str, dashscope_config: ModelServiceConfig | DashScopeConfig, dry_run: bool = False):
        self.model = model
        self.config = dashscope_config
        self.dry_run = dry_run

    def _build_request_payloads(self, grid_image_path: str, prompt: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
        endpoint = _service_endpoint(self.config)
        if _is_openai_compatible(self.config, endpoint):
            request_endpoint = _chat_completions_endpoint(endpoint)
            body: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_to_data_url(grid_image_path)}},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": int(os.getenv("QWEN397B_MAX_TOKENS", "1000")),
            }
            if _use_response_format():
                body["response_format"] = {"type": "json_object"}
            log_body = {
                **body,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": grid_image_path}},
                        ],
                    }
                ],
            }
            return request_endpoint, body, log_body

        body = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": image_to_data_url(grid_image_path)},
                            {"text": prompt},
                        ],
                    }
                ]
            },
            "parameters": {"result_format": "message"},
        }
        log_body = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image_path": grid_image_path},
                            {"text": prompt},
                        ],
                    }
                ]
            },
            "parameters": {"result_format": "message"},
        }
        return endpoint, body, log_body

    def select_grid_with_qwen(
        self,
        grid_image_path: str,
        anomaly_type: str,
        log_path: str | None = None,
    ) -> dict[str, Any]:
        if self.dry_run:
            result = {
                "selected_grid": "C2",
                "confidence": 0.80,
                "top_candidates": [
                    {"grid": "C2", "score": 0.80},
                    {"grid": "C3", "score": 0.65},
                    {"grid": "B2", "score": 0.55},
                ],
                "reason": "dry-run 默认选择机组中下部区域",
                "edit_region_hint": "在设备中下部连接件或地面交界处生成泄漏",
                "raw_response": {"dry_run": True},
            }
            if log_path:
                write_json(log_path, result)
            return result

        if not self.config.api_key:
            raise RuntimeError("VLM API key is required unless --dry-run is used; set QWEN397B_API_KEY")

        prompt = build_vlm_prompt(anomaly_type)
        endpoint, body, log_body = self._build_request_payloads(grid_image_path, prompt)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        request_log = {
            "endpoint": endpoint,
            "provider": self.config.provider,
            "headers": redact_headers(headers),
            "body": log_body,
        }
        response = requests.post(endpoint, headers=headers, json=body, timeout=120)
        raw: dict[str, Any]
        try:
            raw = response.json()
        except Exception:
            raw = {"status_code": response.status_code, "text": response.text}
        if not response.ok:
            if log_path:
                write_json(log_path, {"request": request_log, "response": raw})
            raise RuntimeError(f"Qwen request failed: HTTP {response.status_code}")

        text = _extract_text_from_response(raw)
        parsed = validate_vlm_selection(_extract_json(text))
        parsed["raw_response"] = raw
        if log_path:
            write_json(log_path, {"request": request_log, "parsed": parsed, "response": raw})
        return parsed


def select_grid_with_qwen(grid_image_path: str, anomaly_type: str) -> dict[str, Any]:
    client = QwenVLMClient("qwen3.6-27b", I2IServiceConfig.from_env().vlm)
    return client.select_grid_with_qwen(grid_image_path, anomaly_type)
