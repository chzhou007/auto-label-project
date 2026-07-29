from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.adapters.vlm_labelstudio_detector import (
    VLMLabelStudioDetector,
    labelstudio_payload_to_objects,
    parse_json_output,
    preview_text,
    resized_dimensions,
)
from autolabel.config_loader import load_config
from autolabel.model_config import build_detector_runtime_config
from autolabel.utils import get_image_size, image_to_data_url
from scripts.evaluate_person_detection_benchmark import evaluate_benchmark, write_json


COMPACT_PERSON_BBOX_PROMPT = """Detect all visible people in the image.
Return ONLY valid JSON, with no markdown, no comments, and no explanatory text.
Use pixel coordinates in the image you receive.
Schema:
{"boxes":[{"label":"person","bbox_2d":[0,0,0,0],"confidence":0.0}]}
Rules:
- Include the complete visible body.
- If feet or shoes are visible, the box must include them.
- Prefer a slightly larger box over cutting off head, hands, legs, or feet.
- If no person is visible, return {"boxes":[]}.
"""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:180]


def fallback_image_url(url: str, primary_host: str, fallback_host: str) -> str | None:
    if not primary_host or not fallback_host:
        return None
    parsed = urlsplit(url)
    if parsed.hostname != primary_host:
        return None
    port = f":{parsed.port}" if parsed.port else ""
    username = f"{parsed.username}@" if parsed.username else ""
    netloc = f"{username}{fallback_host}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def download_image(
    url: str,
    output_path: Path,
    *,
    primary_host: str = "172.25.9.142",
    fallback_host: str = "100.106.99.32",
) -> Path:
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fallback = fallback_image_url(url, primary_host, fallback_host)
    urls = [fallback, url] if fallback else [url]
    last_exc: Exception | None = None
    for candidate in urls:
        try:
            response = requests.get(candidate, timeout=60)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            return output_path
        except Exception as exc:
            last_exc = exc
            if candidate != urls[-1]:
                print(f"  !! image download failed, trying fallback URL: {exc}")
    assert last_exc is not None
    raise last_exc


def find_existing_cached_image(cache_root: Path, sample_id: str, suffix: str) -> Path | None:
    safe_sample = safe_name(sample_id)
    candidates = sorted(cache_root.glob(f"ollama_eval_*/images/*_{safe_sample}{suffix}"))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def selected_samples(samples: list[dict[str, Any]], limit: int, seed: int, projects: list[str]) -> list[dict[str, Any]]:
    candidates = [sample for sample in samples if sample.get("source") == "annotation" and sample.get("image")]
    if projects:
        wanted = set(projects)
        candidates = [sample for sample in candidates if sample.get("project_name") in wanted]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:limit] if limit > 0 else candidates


def service_from_config(config_path: Path, model: str, base_url: str, no_tiles: bool) -> dict[str, Any]:
    config = load_config(config_path)
    detector_config = build_detector_runtime_config(config)
    service = detector_config["services"]["ppe_person"]
    if service.get("model_ref"):
        service = detector_config["model_profiles"][service["model_ref"]] | service
    service["api_key"] = service.get("api_key") or "ollama"
    service["base_url"] = base_url.rstrip("/")
    service["model_name"] = model
    service["dry_run"] = False
    service["fail_on_parse_error"] = False
    service["parse_retry_count"] = int(service.get("parse_retry_count", 1))
    service["request_retry_attempts"] = int(service.get("request_retry_attempts", 1))
    service["request_image_max_side"] = int(service.get("request_image_max_side", 1280))
    service["use_response_format"] = False
    if no_tiles:
        tile_cfg = dict(service.get("tile_detection") or {})
        tile_cfg["enabled"] = False
        service["tile_detection"] = tile_cfg
    return service


def object_to_box(obj: dict[str, Any]) -> dict[str, Any]:
    model = obj.get("geometry_model") or {}
    return {
        "id": obj.get("object_id"),
        "label": obj.get("object_type", "person"),
        "bbox_xyxy": [
            int(obj["box"]["x1"]),
            int(obj["box"]["y1"]),
            int(obj["box"]["x2"]),
            int(obj["box"]["y2"]),
        ],
        "confidence": model.get("confidence"),
    }


def strip_json_comments(text: str) -> str:
    return re.sub(r"(?m)\s*//.*$", "", text)


def regex_box_payload(text: str) -> dict[str, Any] | None:
    boxes = []
    pattern = re.compile(
        r'"?(?:bbox_2d|bbox|box|xyxy)"?\s*:?\s*\[\s*'
        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        flags=re.I,
    )
    for match in pattern.finditer(text):
        boxes.append(
            {
                "label": "person",
                "bbox_2d": [float(match.group(index)) for index in range(1, 5)],
                "confidence": None,
            }
        )
    if boxes:
        return {"boxes": boxes}
    keyed_values = {}
    for key in ("x1", "y1", "x2", "y2"):
        match = re.search(rf'"?{key}\s*=?\s*"?\s*:?\s*"?(-?\d+(?:\.\d+)?)', text, flags=re.I)
        if match:
            keyed_values[key] = float(match.group(1))
    if {"x1", "y1", "x2", "y2"}.issubset(keyed_values):
        return {
            "boxes": [
                {
                    "label": "person",
                    "bbox_2d": [keyed_values["x1"], keyed_values["y1"], keyed_values["x2"], keyed_values["y2"]],
                    "confidence": None,
                }
            ]
        }
    return None


def parse_vlm_json(text: str) -> Any:
    try:
        return parse_json_output(text)
    except Exception:
        cleaned = strip_json_comments(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I))
    try:
        return json.loads(cleaned)
    except Exception:
        payload = regex_box_payload(cleaned)
        if payload is not None:
            return payload
        raise


def box_area_xyxy(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def clamp_and_scale_xyxy(
    xyxy: list[Any],
    request_width: int,
    request_height: int,
    image_width: int,
    image_height: int,
) -> list[int] | None:
    if len(xyxy) != 4:
        return None
    coords = [float(value) for value in xyxy]
    if max(abs(value) for value in coords) <= 1.5:
        coords = [coords[0] * request_width, coords[1] * request_height, coords[2] * request_width, coords[3] * request_height]
    elif max(abs(value) for value in coords) <= 100.0:
        coords = [
            coords[0] * request_width / 100.0,
            coords[1] * request_height / 100.0,
            coords[2] * request_width / 100.0,
            coords[3] * request_height / 100.0,
        ]
    x1, y1, x2, y2 = coords
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if request_width != image_width or request_height != image_height:
        x_scale = image_width / float(request_width)
        y_scale = image_height / float(request_height)
        x1, x2 = x1 * x_scale, x2 * x_scale
        y1, y2 = y1 * y_scale, y2 * y_scale
    x1 = max(0, min(image_width - 1, round(x1)))
    y1 = max(0, min(image_height - 1, round(y1)))
    x2 = max(x1 + 1, min(image_width, round(x2)))
    y2 = max(y1 + 1, min(image_height, round(y2)))
    if box_area_xyxy([x1, y1, x2, y2]) <= 0:
        return None
    return [x1, y1, x2, y2]


def candidate_boxes_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("boxes", "detections", "objects", "persons", "people"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def payload_to_prediction_boxes(
    payload: Any,
    image_path: Path,
    image_width: int,
    image_height: int,
    request_width: int,
    request_height: int,
    service: dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(payload, (dict, list)):
        objects = labelstudio_payload_to_objects(
            payload=payload,
            image_uri=str(image_path),
            width=image_width,
            height=image_height,
            service=service,
            raw_response=None,
            request_width=request_width,
            request_height=request_height,
        )
        if objects:
            return [object_to_box(obj) for obj in objects]

    boxes = []
    for idx, item in enumerate(candidate_boxes_from_payload(payload), start=1):
        if not isinstance(item, dict):
            continue
        xyxy = item.get("bbox_2d") or item.get("bbox") or item.get("box") or item.get("xyxy")
        if isinstance(xyxy, dict):
            xyxy = [xyxy.get("x1"), xyxy.get("y1"), xyxy.get("x2"), xyxy.get("y2")]
        if xyxy is None and {"x1", "y1", "x2", "y2"}.issubset(item):
            xyxy = [item["x1"], item["y1"], item["x2"], item["y2"]]
        if xyxy is None and {"x", "y", "width", "height"}.issubset(item):
            x = float(item["x"])
            y = float(item["y"])
            xyxy = [x, y, x + float(item["width"]), y + float(item["height"])]
        if not isinstance(xyxy, list):
            continue
        bbox = clamp_and_scale_xyxy(xyxy, request_width, request_height, image_width, image_height)
        if bbox is None:
            continue
        boxes.append(
            {
                "id": f"ollama_{idx:06d}",
                "label": str(item.get("label") or item.get("class") or "person"),
                "bbox_xyxy": bbox,
                "confidence": item.get("confidence") or item.get("score"),
            }
        )
    return boxes


def detect_with_compact_ollama(
    image_path: Path,
    model: str,
    base_url: str,
    service: dict[str, Any],
    request_image_max_side: int,
) -> tuple[list[dict[str, Any]], str]:
    raw_text = ""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is required for local Ollama VLM benchmark.") from exc

    try:
        image_width, image_height = get_image_size(str(image_path))
        request_width, request_height = resized_dimensions(image_width, image_height, request_image_max_side)
        image_url = image_to_data_url(image_path, max_side=request_image_max_side)
        client = OpenAI(api_key="ollama", base_url=base_url.rstrip("/"))
        messages = [
            {
                "role": "system",
                "content": "You are a computer-vision detection API. Return JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": COMPACT_PERSON_BBOX_PROMPT},
                ],
            },
        ]
        attempts = max(1, int(service.get("request_retry_attempts") or 1))
        delay = max(0.0, float(service.get("request_retry_delay_seconds") or 0.0))
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=800,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts:
                    raise
                print(f"  !! Ollama request failed, retrying {attempt + 1}/{attempts}: {exc}")
                if delay > 0:
                    time.sleep(delay)
        if last_exc is not None and "response" not in locals():
            raise last_exc
        raw_text = response.choices[0].message.content or ""
        payload = parse_vlm_json(raw_text)
        boxes = payload_to_prediction_boxes(
            payload,
            image_path,
            image_width,
            image_height,
            request_width,
            request_height,
            service,
        )
        return boxes, raw_text
    except Exception as exc:
        setattr(exc, "raw_text", raw_text)
        raise


def run_benchmark(
    benchmark_path: Path,
    output_dir: Path,
    config_path: Path,
    model: str,
    base_url: str,
    limit: int,
    seed: int,
    no_tiles: bool,
    projects: list[str],
    mode: str,
    request_image_max_side: int,
    image_cache_dir: Path | None,
    image_primary_host: str,
    image_fallback_host: str,
    resume: bool,
    checkpoint_every: int,
) -> dict[str, Any]:
    benchmark = read_json(benchmark_path)
    samples = selected_samples(benchmark.get("samples") or [], limit, seed, projects)
    service = service_from_config(config_path, model, base_url, no_tiles)
    detector = VLMLabelStudioDetector(service) if mode == "labelstudio" else None
    benchmark_root = benchmark_path.resolve().parent
    cache_dir = image_cache_dir or (benchmark_root / "image_cache")
    checkpoint_path = output_dir / "progress_checkpoint.json"
    run_signature = {
        "model": model,
        "mode": mode,
        "request_image_max_side": request_image_max_side,
        "seed": seed,
        "projects": projects,
        "selected_sample_ids": [str(sample["sample_id"]) for sample in samples],
    }
    result_samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    if resume and checkpoint_path.exists():
        checkpoint = read_json(checkpoint_path)
        if checkpoint.get("run_signature") != run_signature:
            raise RuntimeError(
                f"Checkpoint configuration does not match this run: {checkpoint_path}. "
                "Use a different output directory or pass --no-resume."
            )
        result_samples = list(checkpoint.get("result_samples") or [])
        failures = list(checkpoint.get("failures") or [])
        raw_responses = list(checkpoint.get("raw_responses") or [])
        failed_ids = {str(item.get("sample_id")) for item in failures}
        if failed_ids:
            result_samples = [item for item in result_samples if str(item.get("sample_id")) not in failed_ids]
            raw_responses = [item for item in raw_responses if str(item.get("sample_id")) not in failed_ids]
            failures = []
    completed_ids = {str(item["sample_id"]) for item in result_samples}
    resumed_sample_count = len(completed_ids)
    started_at = time.monotonic()

    def save_checkpoint() -> None:
        write_json(
            checkpoint_path,
            {
                "run_signature": run_signature,
                "result_samples": result_samples,
                "failures": failures,
                "raw_responses": raw_responses,
            },
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(samples, start=1):
        if str(sample["sample_id"]) in completed_ids:
            print(f"[{idx}/{len(samples)}] resumed {sample.get('project_name')}", flush=True)
            continue
        image_url = str(sample["image"])
        suffix = Path(image_url.split("?", 1)[0]).suffix or ".jpg"
        image_path = cache_dir / f"{safe_name(sample['sample_id'])}{suffix}"
        try:
            cached = find_existing_cached_image(benchmark_root, str(sample["sample_id"]), suffix)
            if cached is not None and not image_path.exists():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(cached.read_bytes())
            download_image(
                image_url,
                image_path,
                primary_host=image_primary_host,
                fallback_host=image_fallback_host,
            )
            if mode == "labelstudio":
                assert detector is not None
                objects = detector.detect(str(image_path))
                pred_boxes = [object_to_box(obj) for obj in objects]
            else:
                pred_boxes, raw_text = detect_with_compact_ollama(
                    image_path,
                    model=model,
                    base_url=base_url,
                    service=service,
                    request_image_max_side=request_image_max_side,
                )
                raw_responses.append(
                    {
                        "sample_id": sample.get("sample_id"),
                        "project_name": sample.get("project_name"),
                        "raw_response_preview": preview_text(raw_text, limit=2000),
                    }
                )
        except Exception as exc:
            pred_boxes = []
            raw_text = getattr(exc, "raw_text", "")
            if raw_text:
                raw_responses.append(
                    {
                        "sample_id": sample.get("sample_id"),
                        "project_name": sample.get("project_name"),
                        "error": f"{exc.__class__.__name__}: {exc}",
                        "raw_response_preview": preview_text(raw_text, limit=2000),
                    }
                )
            failures.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "project_name": sample.get("project_name"),
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
        result_samples.append(
            {
                "sample_id": sample["sample_id"],
                "project_name": sample.get("project_name"),
                "source": "annotation",
                "image": str(image_path),
                "original_image": image_url,
                "relative_path": sample.get("relative_path"),
                "width": sample.get("width"),
                "height": sample.get("height"),
                "boxes": sample.get("boxes") or [],
                "prediction_boxes": pred_boxes,
            }
        )
        completed_ids.add(str(sample["sample_id"]))
        if checkpoint_every > 0 and len(result_samples) % checkpoint_every == 0:
            save_checkpoint()
        print(
            f"[{idx}/{len(samples)}] {sample.get('project_name')} "
            f"gt={len(sample.get('boxes') or [])} pred={len(pred_boxes)}",
            flush=True,
        )

    save_checkpoint()
    elapsed_seconds = time.monotonic() - started_at

    prediction_benchmark = output_dir / "ollama_predictions_benchmark.json"
    write_json(prediction_benchmark, {"samples": result_samples})
    if raw_responses:
        write_json(output_dir / "raw_responses.json", raw_responses)
    evaluation = evaluate_benchmark(prediction_benchmark, [0.5, 0.75], include_pseudo=False)
    evaluation.update(
        {
            "model": model,
            "base_url": base_url,
            "mode": mode,
            "source_benchmark": str(benchmark_path),
            "selected_sample_count": len(samples),
            "failure_count": len(failures),
            "failures": failures,
            "no_tiles": no_tiles,
            "request_image_max_side": request_image_max_side,
            "seed": seed,
            "elapsed_seconds_this_invocation": elapsed_seconds,
            "average_seconds_per_new_sample": elapsed_seconds / max(1, len(result_samples) - resumed_sample_count),
        }
    )
    write_json(output_dir / "evaluation.json", evaluation)
    write_json(
        output_dir / "run_summary.json",
        {
            "model": model,
            "base_url": base_url,
            "source_benchmark": str(benchmark_path),
            "selected_sample_count": len(samples),
            "failure_count": len(failures),
            "output_benchmark": str(prediction_benchmark),
            "evaluation": str(output_dir / "evaluation.json"),
            "raw_responses": str(output_dir / "raw_responses.json") if raw_responses else None,
            "no_tiles": no_tiles,
            "mode": mode,
            "request_image_max_side": request_image_max_side,
            "image_cache_dir": str(cache_dir),
            "image_primary_host": image_primary_host,
            "image_fallback_host": image_fallback_host,
            "projects": projects,
            "elapsed_seconds_this_invocation": elapsed_seconds,
        },
    )
    return evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Ollama VLM detector on a person benchmark subset.")
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmarks/person_labelstudio_172_25_9_142_20260706/person_detection_benchmark.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmarks/person_labelstudio_172_25_9_142_20260706/ollama_eval_qwen25vl7b_sample"))
    parser.add_argument("--config", type=Path, default=Path("configs/autolabel.yaml"))
    parser.add_argument("--model", default="qwen2.5vl:7b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--no-tiles", action="store_true", help="Disable tiled detection for a faster first benchmark pass.")
    parser.add_argument("--projects", default="", help="Comma-separated project names to include.")
    parser.add_argument("--mode", choices=["compact", "labelstudio"], default="compact", help="compact uses an Ollama-friendly bbox schema; labelstudio uses the project detector.")
    parser.add_argument("--request-image-max-side", type=int, default=1280)
    parser.add_argument("--image-cache-dir", type=Path, default=None, help="Persistent local image cache. Defaults to <benchmark_dir>/image_cache.")
    parser.add_argument("--image-primary-host", default="172.25.9.142")
    parser.add_argument("--image-fallback-host", default="100.106.99.32")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    projects = [item.strip() for item in args.projects.split(",") if item.strip()]
    evaluation = run_benchmark(
        args.benchmark,
        args.output_dir,
        args.config,
        args.model,
        args.base_url,
        args.limit,
        args.seed,
        args.no_tiles,
        projects,
        args.mode,
        args.request_image_max_side,
        args.image_cache_dir,
        args.image_primary_host,
        args.image_fallback_host,
        args.resume,
        args.checkpoint_every,
    )
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
