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
PROMPT_VERSION = "crop_industrial_safety_v2"

SYSTEM_PROMPT = """你是一个严格的工业安全视觉识别专家，专门针对【人体检测框裁剪图（Crop图）】执行安全穿戴及不安全行为的自动化标注任务。
请仔细观察提供的Crop图，由于画面视野受限、缺乏全局背景且极易存在身体部位截断，请严格遵循以下基于局部特征的推断逻辑识别标签，并只返回合法 JSON 对象。

待识别标签及Crop图专项判定逻辑：

safety_harness (佩戴安全带): 重点观察躯干部分是否有明显的安全带束带（如肩带、胸带）。若躯干被严重遮挡设为 false。

hard_hat (佩戴安全帽): 观察头部。若画面顶部截断导致头部不可见，设为 false。

reflective_vest (穿反光衣): 观察上身是否有荧光色大面积色块或反光条。

safety_shoes (穿戴防砸鞋): 观察脚部。若画面下部截断导致脚部不可见，必须设为 false，并在 notes 中注明截断。

smoking (吸烟行为): 重点寻找手部与面部的交互动作，以及烟支、烟雾等局部视觉特征。

falling_down (跌倒): 缺乏地面参照时，仅根据人体异常姿态（如躯干完全横向、极度倾斜的倒地姿态）进行判断。

sleeping_on_duty (睡岗): 缺乏完整工位参照时，依靠头部低垂、闭眼、身体松垮倚靠支撑物等局部疲劳特征判断。

climbing_over_railing (翻越栏杆): 画面内必须可见栏杆局部特征，且人体有明显的跨越/攀爬动作。若Crop图全图无栏杆特征，一律设为 false。

touching_equipment (触摸设备): 画面内必须可见设备局部，且手部或肢体与之有直接接触。若纯背景或无设备特征，一律设为 false。

fighting (打架): 若Crop图仅截取了单人，且无明显受到攻击或攻击画外目标的剧烈肢体动作，一律设为 false。

using_phone (玩手机): 重点观察双手是否持握矩形发光/暗色物体，且视线向下聚焦于手部。

safety_goggles (穿戴护目镜): 观察眼部。若面部背对镜头、被遮挡或分辨率过低无法分辨，设为 false。

输出规则：

无法判断的情况（包含且不限于：部位截断、背影遮挡、缺乏背景参照物、画面模糊）一律设为 false。

当因为“截断”或“无背景”等客观原因导致设为 false 时，必须在 notes 中使用简短的英文键和中文值说明原因，例如 "notes": {"safety_shoes": "脚部截断无法判断", "touching_equipment": "无设备上下文"}。若所有标签均能清晰判定，notes 留空 {}。

输出必须是纯 JSON，不要 markdown，不要解释文字。你的输出第一个字符必须是 {，最后一个字符必须是 }：
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

CLASSIFICATION_KEYS = tuple(BOOLEAN_LABEL_MAP.keys())


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


def _clean_note(value: str, limit: int = 180) -> str:
    value = re.sub(r"[*_`#>-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" :;,.")
    if len(value) > limit:
        return value[: limit - 3].rstrip() + "..."
    return value


def parse_boolean_text_output(text: str) -> dict[str, Any] | None:
    """Fallback parser for VLMs that return prose instead of strict JSON."""
    if not text:
        return None
    matches: list[tuple[int, str]] = []
    lower_text = text.lower()
    for key in CLASSIFICATION_KEYS:
        pattern = rf"(?<![a-z0-9_]){re.escape(key)}(?![a-z0-9_])"
        for match in re.finditer(pattern, lower_text):
            matches.append((match.start(), key))
    if not matches:
        return None

    matches.sort(key=lambda item: item[0])
    parsed: dict[str, Any] = {"notes": {"_parser": "fallback_text_boolean_parser"}}
    seen: set[str] = set()
    for index, (start, key) in enumerate(matches):
        if key in seen:
            continue
        end = matches[index + 1][0] if index + 1 < len(matches) else len(text)
        segment = text[start:end]
        bool_values = re.findall(r"\b(true|false)\b", segment, flags=re.I)
        if not bool_values:
            continue
        parsed[key] = bool_values[-1].lower() == "true"
        note = _clean_note(segment)
        if note:
            parsed["notes"][key] = note
        seen.add(key)

    if not seen:
        return None

    for key in CLASSIFICATION_KEYS:
        if key not in parsed:
            parsed[key] = False
            parsed["notes"][key] = "fallback parser did not find an explicit value"
    parsed["_parse_mode"] = "text_fallback"
    parsed["_raw_text_preview"] = _clean_note(text, limit=500)
    return parsed


def normalize_boolean_response(raw_response: dict[str, Any]) -> dict[str, Any]:
    """Ensure all classification labels are present for stable Label Studio editing."""
    if not isinstance(raw_response, dict):
        return {"error": "classification response is not a JSON object", "raw": raw_response}
    if raw_response.get("error"):
        return raw_response

    notes = raw_response.get("notes") if isinstance(raw_response.get("notes"), dict) else {}
    normalized: dict[str, Any] = {}
    for key in CLASSIFICATION_KEYS:
        value = raw_response.get(key, False)
        normalized[key] = value if isinstance(value, bool) else False
        if key not in raw_response:
            notes.setdefault(key, "模型未返回该字段，按规则置为 false")
        elif not isinstance(value, bool):
            notes.setdefault(key, "字段不是布尔值，按规则置为 false")
    normalized["notes"] = notes
    for key, value in raw_response.items():
        if key not in normalized:
            normalized[key] = value
    return normalized


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
                        {"type": "text", "text": "请按照系统提示分析这张人体检测框裁剪图，仅输出规定的纯 JSON 对象。"},
                    ],
                },
            ],
            max_tokens=800,
            temperature=0.0,
        )
        raw_output = response.choices[0].message.content or ""
        result = parse_json_output(raw_output, expected_type=dict)
        return normalize_boolean_response(result)
    except json.JSONDecodeError as exc:
        fallback = parse_boolean_text_output(raw_output)
        if fallback is not None:
            print(f"  !! JSON 解析失败，已使用文本兜底解析: {exc}")
            return normalize_boolean_response(fallback)
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
        "prompt_version": classifier_config.get("prompt_version") or PROMPT_VERSION,
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
