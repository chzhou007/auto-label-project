#!/usr/bin/env python3
"""Safety-label classifier for AutoLabelSample objects.

The pipeline imports this file as a module and calls ``classify_sample`` with a
full AutoLabelSample. The script can also be used from the command line to
classify one sample JSON file.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


DEFAULT_API_URL = os.getenv("QWEN397B_API_URL", "https://deepseek.gds-services.com/vllm-qwen35b/v1")
DEFAULT_API_KEY = os.getenv("QWEN397B_API_KEY", "")
DEFAULT_MODEL = os.getenv("QWEN397B_MODEL", "aios-smart-eye-vlm")

SYSTEM_PROMPT = """你是一个严格的工业安全视觉识别专家，专门从人体照片中检测安全穿戴及不安全行为。
请仔细观察提供的照片，依据下列定义识别这些标签，并只返回合法 JSON 对象。

待识别标签：
1. safety_harness: 佩戴安全带。
2. hard_hat: 佩戴安全帽。
3. reflective_vest: 穿反光衣。
4. safety_shoes: 穿戴防砸鞋。
5. smoking: 吸烟行为。
6. falling_down: 跌倒。
7. sleeping_on_duty: 睡岗。
8. climbing_over_railing: 翻越栏杆。
9. touching_equipment: 触摸设备。
10. fighting: 打架。
11. using_phone: 玩手机。
12. safety_goggles: 穿戴护目镜。

输出必须是纯 JSON，不要 markdown，不要解释文字：
{
  "safety_harness": true,
  "hard_hat": false,
  "reflective_vest": false,
  "safety_shoes": false,
  "smoking": false,
  "falling_down": false,
  "sleeping_on_duty": false,
  "climbing_over_railing": false,
  "touching_equipment": false,
  "fighting": false,
  "using_phone": false,
  "safety_goggles": false,
  "notes": {}
}

无法判断时设为 false，并在 notes 中简要说明原因。
"""

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


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def encode_image(image_path: str | Path) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def get_image_mime_type(image_path: str | Path) -> str:
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/jpeg")


def parse_json_output(text: str, expected_type: type | tuple[type, ...] | None = dict) -> Any:
    value = (text or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    decoder = json.JSONDecoder()
    first_value = None
    first_error = None
    for idx, char in enumerate(value):
        if char not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[idx:])
        except json.JSONDecodeError as exc:
            first_error = first_error or exc
            continue
        if first_value is None:
            first_value = parsed
        if expected_type is None or isinstance(parsed, expected_type):
            return parsed
    if first_value is not None:
        return first_value
    if first_error is not None:
        raise first_error
    raise json.JSONDecodeError("No JSON object or array found", value, 0)


def process_image(image_path: str | Path, client: OpenAI) -> dict[str, Any]:
    print(f"  识别: {image_path}")
    raw_output = ""
    try:
        image_url = f"data:{get_image_mime_type(image_path)};base64,{encode_image(image_path)}"
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "请按照系统提示分析这张图片，仅输出规定的 JSON。"},
                    ],
                },
            ],
            max_tokens=800,
            temperature=0.0,
        )
        raw_output = response.choices[0].message.content or ""
        result = parse_json_output(raw_output, expected_type=dict)
        if "notes" not in result or not isinstance(result.get("notes"), dict):
            result["notes"] = {}
        return result
    except json.JSONDecodeError as exc:
        print(f"  !! JSON 解析失败: {exc}")
        print(f"  !! 原始返回前300字符: {raw_output.replace(chr(10), ' ')[:300]}")
        return {"error": "JSON decode error", "raw": raw_output}
    except Exception as exc:
        print(f"  !! 调用 API 出错: {exc}")
        return {"error": str(exc), "raw": raw_output}


def labels_from_boolean_response(raw_response: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(raw_response, dict) or raw_response.get("error"):
        return []
    notes = raw_response.get("notes") if isinstance(raw_response.get("notes"), dict) else {}
    labels = []
    for source_key, mapping in BOOLEAN_LABEL_MAP.items():
        if source_key not in raw_response:
            continue
        value = raw_response[source_key]
        if not isinstance(value, bool):
            continue
        label_key, true_value, false_value = mapping
        labels.append(
            {
                "label_key": label_key,
                "label_value": true_value if value else false_value,
                "confidence": None,
                "evidence": notes.get(source_key) or "vlm classification",
            }
        )
    return labels


def build_autolabel_classification(
    raw_response: dict[str, Any],
    classifier_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    classifier_config = classifier_config or {}
    return {
        "multi_labels": labels_from_boolean_response(raw_response),
        "classifier_type": classifier_config.get("classifier_type", "vlm"),
        "classifier_name": classifier_config.get("classifier_name") or classifier_config.get("model") or DEFAULT_MODEL,
        "classifier_version": classifier_config.get("classifier_version") or classifier_config.get("model_version"),
        "prompt_version": classifier_config.get("prompt_version") or "classification_script_default",
        "raw_response": raw_response,
    }


def classify_object(
    obj: dict[str, Any],
    client: OpenAI,
    classifier_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    crop_uri = obj.get("crop", {}).get("crop_uri")
    if not crop_uri:
        raw_response = {"error": "missing crop.crop_uri"}
    else:
        raw_response = process_image(crop_uri, client)
    obj["classification"] = build_autolabel_classification(raw_response, classifier_config)
    return obj


def classify_sample(
    sample: dict[str, Any],
    client: OpenAI,
    skip_source_types: set[str] | list[str] | tuple[str, ...] | None = None,
    classifier_config: dict[str, Any] | None = None,
    delay_seconds: float | None = None,
) -> dict[str, Any]:
    skip_source_types = set(skip_source_types or {"generated"})
    classifier_config = classifier_config or {}
    delay_seconds = float(delay_seconds if delay_seconds is not None else classifier_config.get("delay_seconds", 0.5))

    if sample.get("image_asset", {}).get("source_type") in skip_source_types:
        return sample

    for obj in sample.get("objects", []):
        classify_object(obj, client, classifier_config)
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    workflow = sample.setdefault("workflow", {})
    workflow["workflow_status"] = "classified"
    workflow["updated_time"] = utc_now()
    return sample


def main() -> int:
    global DEFAULT_MODEL
    parser = argparse.ArgumentParser(description="Classify AutoLabelSample crops and write classifications back.")
    parser.add_argument("--sample", required=True, help="AutoLabelSample JSON file")
    parser.add_argument("--output", default=None, help="Output JSON file. Defaults to overwriting --sample.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    if not args.api_key:
        print("错误：未提供 API 密钥，请设置 QWEN397B_API_KEY 或 --api-key")
        return 1
    if OpenAI is None:
        print("错误：需要安装 openai 库，请执行：pip install openai")
        return 1

    DEFAULT_MODEL = args.model
    client = OpenAI(api_key=args.api_key, base_url=args.api_url)

    sample_path = Path(args.sample)
    with open(sample_path, encoding="utf-8") as f:
        sample = json.load(f)

    sample = classify_sample(sample, client, delay_seconds=args.delay)
    output_path = Path(args.output) if args.output else sample_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"输出 AutoLabelSample: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
