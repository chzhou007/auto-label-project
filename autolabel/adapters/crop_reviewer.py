from __future__ import annotations

import json
from typing import Any

from ..model_config import deep_merge
from ..utils import image_to_data_url, now_iso_shanghai
from .vlm_labelstudio_detector import VLMJsonParseError, bool_config, preview_text


DEFAULT_CROP_REVIEW_PROMPT = """你是一个严格的工业安全数据质检员。请检查这张人体检测框裁剪图（crop）是否包含可见人体，以及这个 crop 是否完整覆盖了该人体在原场景中的可见范围。

判定规则：
1. 如果 crop 只包含背景、柜体、门板、设备、地面、黑边、墙面、管线或纹理，没有任何人体部位，contains_person 必须为 false。
2. 如果 crop 中能看到头、躯干、四肢、手、脚、衣服轮廓等任意明确人体部位，contains_person 为 true。
3. 如果 crop 边界把头部、脚部、躯干或主要肢体额外截断，is_complete_visible_person 为 false。
4. 如果人体只是被原场景中的设备、栏杆或其他物体遮挡，但 crop 边界没有额外截断人体，is_complete_visible_person 可为 true。
5. 不要把柜门缝、地砖边缘、机柜黑边、阴影、地面纹理误判成人。

只输出一个合法 JSON object，不要 markdown，不要解释文字：
{
  "contains_person": true,
  "is_complete_visible_person": true,
  "missing_parts": [],
  "reason": ""
}
"""


DEFAULT_SYSTEM_MESSAGE = (
    "You are a strict JSON API for crop quality review. Return only one valid JSON object. "
    "Do not output prose, markdown, explanations, analysis, or thinking process."
)


def parse_review_payload(text: str) -> dict[str, Any]:
    value = (text or "").strip()
    decoder = json.JSONDecoder()
    for idx, char in enumerate(value):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise VLMJsonParseError(f"Crop review output is not parseable JSON. Preview: {preview_text(text)}")


def build_crop_review_config(
    pipeline_config: dict[str, Any],
    detector_config: dict[str, Any],
) -> dict[str, Any]:
    direct_cfg = pipeline_config.get("direct_annotation", {})
    review_cfg = dict(direct_cfg.get("crop_review") or {})
    if not bool_config(review_cfg.get("enabled"), default=False):
        return {"enabled": False}

    model_ref = review_cfg.get("model_ref")
    profile = {}
    if model_ref:
        profiles = detector_config.get("model_profiles", {})
        profile = profiles.get(model_ref)
        if not isinstance(profile, dict):
            raise KeyError(f"Crop review model profile not found: {model_ref}")

    merged = deep_merge(profile, review_cfg)
    merged["enabled"] = True
    merged["model_ref"] = model_ref
    if model_ref:
        for service in detector_config.get("services", {}).values():
            if service.get("model_ref") == model_ref and bool_config(service.get("dry_run")):
                merged["dry_run"] = True
                break
    if not merged.get("prompt"):
        merged["prompt"] = DEFAULT_CROP_REVIEW_PROMPT
    if "system_message" not in review_cfg:
        merged["system_message"] = DEFAULT_SYSTEM_MESSAGE
    merged.setdefault("target_object_type", "person")
    merged.setdefault("reviewer", "vlm_crop_reviewer")
    merged.setdefault("prompt_version", "crop_full_person_review_v1")
    merged.setdefault("failed_issue_flag", "incomplete_person_crop")
    merged.setdefault("record_passed", False)
    merged.setdefault("drop_failed", False)
    merged.setdefault("drop_incomplete_person", False)
    merged.setdefault("use_response_format", True)
    merged.setdefault("response_format_type", "json_object")
    if "max_tokens" not in review_cfg:
        merged["max_tokens"] = 600
    return merged


def apply_crop_review_result(
    obj: dict[str, Any],
    result: dict[str, Any],
    review_cfg: dict[str, Any],
) -> None:
    passed = bool(result.get("contains_person")) and bool(result.get("is_complete_visible_person"))
    record_passed = bool_config(review_cfg.get("record_passed"), default=False)
    if passed and not record_passed:
        return

    missing_parts = result.get("missing_parts")
    if not isinstance(missing_parts, list):
        missing_parts = []
    issue_flags = []
    if not passed:
        issue_flags.append(str(review_cfg.get("failed_issue_flag", "incomplete_person_crop")))
        for part in missing_parts:
            issue_flags.append(f"missing_{part}")

    obj["quality_check"] = {
        "qc_sampled": True,
        "qc_status": "passed" if passed else "failed",
        "reviewed_labels": None,
        "issue_flags": issue_flags,
        "reviewer": str(review_cfg.get("reviewer", "vlm_crop_reviewer")),
        "review_time": now_iso_shanghai(),
        "comment": review_comment(result, review_cfg),
    }


def should_drop_reviewed_object(result: dict[str, Any], review_cfg: dict[str, Any]) -> bool:
    if not bool_config(review_cfg.get("drop_failed"), default=False):
        return False
    contains_person = bool(result.get("contains_person"))
    if not contains_person:
        return True
    if bool_config(review_cfg.get("drop_incomplete_person"), default=False):
        return not bool(result.get("is_complete_visible_person"))
    return False


def review_comment(result: dict[str, Any], review_cfg: dict[str, Any]) -> str:
    reason = str(result.get("reason") or "").strip()
    prompt_version = review_cfg.get("prompt_version", "crop_full_person_review_v1")
    if reason:
        return f"{prompt_version}: {reason}"
    return f"{prompt_version}: crop review passed"


class VLMCropReviewer:
    def __init__(self, review_cfg: dict[str, Any]) -> None:
        self.review_cfg = review_cfg
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("openai is required for VLM crop review.") from exc
            api_key = self.review_cfg.get("api_key")
            base_url = self.review_cfg.get("base_url") or self.review_cfg.get("api_url")
            if not api_key:
                raise RuntimeError("Crop review api_key is not configured.")
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def review_sample(self, sample: dict[str, Any]) -> None:
        target_object_type = self.review_cfg.get("target_object_type", "person")
        kept_objects = []
        dropped = 0
        for obj in sample.get("objects", []):
            if target_object_type and obj.get("object_type") != target_object_type:
                kept_objects.append(obj)
                continue
            crop_uri = obj.get("crop", {}).get("crop_uri")
            if not crop_uri:
                kept_objects.append(obj)
                continue
            result = self.review_crop(crop_uri)
            apply_crop_review_result(obj, result, self.review_cfg)
            if should_drop_reviewed_object(result, self.review_cfg):
                dropped += 1
                self._remove_crop_file(crop_uri)
                continue
            kept_objects.append(obj)
        if dropped:
            print(f"  !! {sample.get('sample_id')} crop 复核删除 {dropped} 个非人体对象")
        sample["objects"] = kept_objects

    def _remove_crop_file(self, crop_uri: str) -> None:
        from pathlib import Path

        path = Path(crop_uri)
        if path.exists() and path.is_file():
            path.unlink()

    def review_crop(self, crop_uri: str) -> dict[str, Any]:
        if self.review_cfg.get("dry_run"):
            return {
                "contains_person": True,
                "is_complete_visible_person": True,
                "missing_parts": [],
                "reason": "dry_run",
            }
        max_side = self.review_cfg.get("request_image_max_side")
        max_side = int(max_side) if max_side not in (None, "") else None
        image_url = image_to_data_url(crop_uri, max_side=max_side)
        raw_text = self._request_json_text(image_url)
        return parse_review_payload(raw_text)

    def _request_json_text(self, image_url: str) -> str:
        model_name = self.review_cfg.get("model_name")
        if not model_name:
            raise RuntimeError("Crop review model_name is not configured.")

        messages = [
            {
                "role": "system",
                "content": self.review_cfg.get("system_message") or DEFAULT_SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": self.review_cfg.get("prompt") or DEFAULT_CROP_REVIEW_PROMPT},
                ],
            },
        ]

        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": float(self.review_cfg.get("temperature", 0.0)),
            "max_tokens": max(1, int(self.review_cfg.get("max_tokens", 600))),
        }
        if bool_config(self.review_cfg.get("use_response_format"), default=True):
            kwargs["response_format"] = {
                "type": self.review_cfg.get("response_format_type", "json_object"),
            }
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if "response_format" not in message and "json_object" not in message and "guided" not in message:
                raise
            kwargs.pop("response_format", None)
            response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
