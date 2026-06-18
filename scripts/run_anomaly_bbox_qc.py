from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.qc_agent import (  # noqa: E402
    build_overlay_data_url,
    parse_vlm_qc_payload,
    resolve_local_uri,
)
from autolabel.utils import read_json  # noqa: E402


def box_numbers(box: Any) -> tuple[int, int, int, int] | None:
    try:
        if isinstance(box, dict):
            return int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        if isinstance(box, (list, tuple)) and len(box) == 4:
            return int(box[0]), int(box[1]), int(box[2]), int(box[3])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def box_iou(left: Any, right: Any) -> float:
    left_values = box_numbers(left)
    right_values = box_numbers(right)
    if left_values is None or right_values is None:
        return 0.0
    ax1, ay1, ax2, ay2 = left_values
    bx1, by1, bx2, by2 = right_values
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    left_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    right_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def box_contains(outer: Any, inner: Any, tolerance: int = 0) -> bool:
    outer_values = box_numbers(outer)
    inner_values = box_numbers(inner)
    if outer_values is None or inner_values is None:
        return False
    ox1, oy1, ox2, oy2 = outer_values
    ix1, iy1, ix2, iy2 = inner_values
    return ox1 - tolerance <= ix1 and oy1 - tolerance <= iy1 and ox2 + tolerance >= ix2 and oy2 + tolerance >= iy2


def prompt_for(sample: dict[str, Any], obj: dict[str, Any]) -> str:
    labels = obj.get("classification", {}).get("multi_labels") or []
    label_text = ", ".join(
        f"{item.get('label_key')}={item.get('label_value')}"
        for item in labels
        if isinstance(item, dict) and item.get("label_key")
    ) or "no classification labels"
    scene = sample.get("image_asset", {}).get("scene_context") or {}
    return f"""你是工业异常检测标注质检员。图片上的红色矩形是待复核的异常标注框。

只质检“异常区域框”本身。红框必须框住图中真实可见的异常痕迹，例如漏油、漏水、湿痕、液膜、积液、污渍、反光水迹、异常颜色或异常形态。阀门、螺栓、管道、接头、设备边缘、地面空白、阴影和正常反光都不能当成异常目标。

样本信息：
- sample_id: {sample.get("sample_id")}
- object_id: {obj.get("object_id")}
- object_type: {obj.get("object_type")}
- labels: {label_text}
- inspection_content: {scene.get("inspection_content")}
- task_group: {scene.get("task_group")}

判定要求：
1. visible_target 只表示红框内是否能看到与 labels / inspection_content 对应的可见异常痕迹；只看到设备部件时必须为 false。
2. label_match 表示红框内的异常痕迹是否和标签一致。比如 diesel_leak 应该有柴油/油渍/湿润油膜特征，coolant_leak 应该有冷却液或对应液体痕迹。
3. box_quality 只能从 good、over_inclusive、under_inclusive、wrong_target、uncertain 中选一个。
4. good：红框基本覆盖主要异常区域，允许少量边缘留白。
5. over_inclusive：红框包含异常，但背景、设备或正常区域明显过多。
6. under_inclusive：红框只框住异常的一小部分，或者图中主异常明显延伸到红框外。
7. wrong_target：红框框到了设备、螺栓、管道、阴影、文字、正常反光、空白区域，或框内没有可见异常。
8. uncertain：画质差、异常太弱、遮挡严重，无法稳定判断。
9. 如果图中有明显异常但红框没有覆盖主要异常，按 under_inclusive 或 wrong_target 处理，并设置 needs_human_review=true。
10. wrong_target 直接 needs_human_review=true；under_inclusive、over_inclusive、uncertain 也需要复核。
11. 只输出合法 JSON object，不要 markdown，不要解释文字。

输出格式：
{{
  "visible_target": true,
  "box_quality": "good",
  "label_match": true,
  "needs_human_review": false,
  "issue_flags": [],
  "reason": ""
}}
"""


def object_context(obj: dict[str, Any]) -> dict[str, Any]:
    detail = obj.get("geometry_detail") if isinstance(obj.get("geometry_detail"), dict) else {}
    params = detail.get("generation_params") if isinstance(detail.get("generation_params"), dict) else {}
    return {
        "box": obj.get("box"),
        "selected_grid": params.get("selected_grid"),
        "grid_bbox": params.get("grid_bbox"),
        "expanded_edit_bbox": params.get("expanded_edit_bbox"),
        "final_bbox_source": params.get("final_bbox_source"),
        "localization_pipeline": params.get("localization_pipeline"),
    }


def rule_flags(obj: dict[str, Any], *, min_grid_iou: float, tolerance_px: int) -> dict[str, Any]:
    ctx = object_context(obj)
    flags: list[str] = []
    evidence: dict[str, Any] = {}
    box = ctx["box"]
    grid_bbox = ctx["grid_bbox"]
    expanded_edit_bbox = ctx["expanded_edit_bbox"]

    if box_numbers(box) is None:
        flags.append("invalid_box")
    if expanded_edit_bbox is not None and not box_contains(expanded_edit_bbox, box, tolerance=tolerance_px):
        flags.append("box_outside_expanded_edit_region")
        evidence["expanded_edit_bbox"] = expanded_edit_bbox
    if grid_bbox is not None:
        grid_iou = box_iou(box, grid_bbox)
        evidence["grid_iou"] = round(grid_iou, 4)
        if grid_iou < min_grid_iou:
            flags.append("box_disconnected_from_selected_grid")
            evidence["grid_bbox"] = grid_bbox
            evidence["selected_grid"] = ctx["selected_grid"]
    return {"rule_flags": flags, "rule_evidence": evidence}


def save_overlay(image_path: Path, box: Any, output_path: Path) -> None:
    from PIL import Image, ImageDraw

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    values = box_numbers(box)
    if values is not None:
        x1, y1, x2, y2 = values
        line_width = max(3, round(min(image.size) / 180))
        for offset in range(line_width):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=(255, 0, 0))
    image.save(output_path, quality=92)


def pack_record(record: dict[str, Any], pack_dir: Path) -> None:
    status = record["decision"]
    quality = record.get("box_quality") or "unknown"
    if status == "passed":
        return
    if status == "failed":
        bucket = f"failed_{quality}"
    elif record.get("rule_flags"):
        bucket = "review_rule_suspect"
    else:
        bucket = f"review_{quality}"
    sample_id = record["sample_id"]
    object_id = record["object_id"]
    target_dir = pack_dir / bucket
    target_dir.mkdir(parents=True, exist_ok=True)
    overlay_src = Path(record["overlay_path"])
    image_src = Path(record["image_path"])
    shutil.copy2(overlay_src, target_dir / f"{sample_id}_{object_id}_overlay.jpg")
    if image_src.exists():
        shutil.copy2(image_src, target_dir / f"{sample_id}_{object_id}{image_src.suffix.lower()}")


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record.get("sample_id"),
        "object_id": record.get("object_id"),
        "decision": record.get("decision"),
        "box_quality": record.get("box_quality"),
        "visible_target": record.get("visible_target"),
        "label_match": record.get("label_match"),
        "needs_human_review": record.get("needs_human_review"),
        "issue_flags": ";".join(record.get("issue_flags") or []),
        "rule_flags": ";".join(record.get("rule_flags") or []),
        "reason": record.get("reason"),
        "inspection_content": record.get("inspection_content"),
        "labels": record.get("labels"),
        "selected_grid": record.get("selected_grid"),
        "grid_iou": record.get("rule_evidence", {}).get("grid_iou"),
        "box": json.dumps(record.get("box"), ensure_ascii=False),
        "grid_bbox": json.dumps(record.get("grid_bbox"), ensure_ascii=False),
        "expanded_edit_bbox": json.dumps(record.get("expanded_edit_bbox"), ensure_ascii=False),
        "image_path": record.get("image_path"),
        "metadata_path": record.get("metadata_path"),
        "overlay_path": record.get("overlay_path"),
        "error": record.get("error"),
        "latency_seconds": record.get("latency_seconds"),
    }


def load_done(results_path: Path) -> set[str]:
    done: set[str] = set()
    if not results_path.exists():
        return done
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(f"{record.get('sample_id')}::{record.get('object_id')}")
    return done


def call_vlm(client: Any, *, model: str, image_url: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict JSON API for industrial anomaly bounding-box quality control. Return JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or ""
    parsed = parse_vlm_qc_payload(text)
    parsed["raw_response_text"] = text
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resumable VLM QC for anomaly bounding boxes.")
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--asset-base-dir", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=os.getenv("QWEN_GEOMETRY_MODEL") or os.getenv("SJTU_API_DEFAULT_MODEL") or "qwen")
    parser.add_argument("--base-url", default=os.getenv("QWEN_GEOMETRY_API_URL") or os.getenv("SJTU_API_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("QWEN397B_API_KEY") or os.getenv("SJTU_API_KEY"))
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--request-image-max-side", type=int, default=1280)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--interval-seconds", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-grid-iou", type=float, default=0.01)
    parser.add_argument("--region-tolerance-px", type=int, default=2)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set QWEN397B_API_KEY or SJTU_API_KEY.")
    if not args.base_url:
        raise SystemExit("Missing base URL. Set QWEN_GEOMETRY_API_URL or SJTU_API_BASE_URL.")

    from openai import OpenAI

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "overlays"
    pack_dir = output_dir / "problem_pack"
    results_path = output_dir / "bbox_qc_results.jsonl"
    csv_path = output_dir / "bbox_qc_results.csv"
    summary_path = output_dir / "summary.md"

    metadata_paths = sorted(Path(args.metadata_dir).glob("*.json"))
    if args.limit is not None:
        metadata_paths = metadata_paths[: max(0, args.limit)]

    done = set() if args.overwrite else load_done(results_path)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout_seconds)
    asset_base_dirs = [Path(path) for path in args.asset_base_dir]

    new_records = 0
    with results_path.open("a", encoding="utf-8") as results_handle:
        for metadata_path in metadata_paths:
            sample = read_json(metadata_path)
            image_uri = (sample.get("image_asset") or {}).get("image_uri")
            image_path, image_exists, _ = resolve_local_uri(image_uri, metadata_path, asset_base_dirs)
            objects = sample.get("objects") if isinstance(sample.get("objects"), list) else []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                sample_id = sample.get("sample_id") or metadata_path.stem
                object_id = obj.get("object_id") or "object"
                key = f"{sample_id}::{object_id}"
                if key in done:
                    continue

                labels = obj.get("classification", {}).get("multi_labels") or []
                label_text = ";".join(
                    f"{item.get('label_key')}={item.get('label_value')}"
                    for item in labels
                    if isinstance(item, dict) and item.get("label_key")
                )
                scene = sample.get("image_asset", {}).get("scene_context") or {}
                ctx = object_context(obj)
                rules = rule_flags(obj, min_grid_iou=args.min_grid_iou, tolerance_px=args.region_tolerance_px)
                overlay_path = overlays_dir / f"{sample_id}_{object_id}_overlay.jpg"

                base_record = {
                    "sample_id": sample_id,
                    "object_id": object_id,
                    "metadata_path": str(metadata_path),
                    "image_path": str(image_path) if image_path else "",
                    "overlay_path": str(overlay_path),
                    "inspection_content": scene.get("inspection_content"),
                    "labels": label_text,
                    "box": ctx["box"],
                    "selected_grid": ctx["selected_grid"],
                    "grid_bbox": ctx["grid_bbox"],
                    "expanded_edit_bbox": ctx["expanded_edit_bbox"],
                    "final_bbox_source": ctx["final_bbox_source"],
                    "localization_pipeline": ctx["localization_pipeline"],
                    **rules,
                }

                started = time.perf_counter()
                try:
                    if image_path is None or not image_exists:
                        raise FileNotFoundError(f"image not found: {image_uri}")
                    save_overlay(image_path, ctx["box"], overlay_path)
                    image_url = build_overlay_data_url(image_path, ctx["box"], max_side=args.request_image_max_side)
                    parsed = call_vlm(
                        client,
                        model=args.model,
                        image_url=image_url,
                        prompt=prompt_for(sample, obj),
                        max_tokens=args.max_tokens,
                    )
                    quality = parsed.get("box_quality")
                    visible = bool(parsed.get("visible_target"))
                    label_match = bool(parsed.get("label_match"))
                    needs_review = bool(parsed.get("needs_human_review"))
                    if not visible or not label_match or quality == "wrong_target":
                        decision = "failed"
                    elif needs_review or quality in {"over_inclusive", "under_inclusive", "uncertain"} or rules["rule_flags"]:
                        decision = "needs_human_review"
                    else:
                        decision = "passed"
                    record = {
                        **base_record,
                        "decision": decision,
                        "visible_target": visible,
                        "box_quality": quality,
                        "label_match": label_match,
                        "needs_human_review": needs_review,
                        "issue_flags": parsed.get("issue_flags", []),
                        "reason": parsed.get("reason"),
                        "raw_response_text": parsed.get("raw_response_text"),
                    }
                except Exception as exc:
                    record = {
                        **base_record,
                        "decision": "needs_human_review",
                        "visible_target": None,
                        "box_quality": "uncertain",
                        "label_match": None,
                        "needs_human_review": True,
                        "issue_flags": ["api_or_runtime_error"],
                        "reason": "QC call failed; route to manual review",
                        "error": repr(exc),
                    }
                record["latency_seconds"] = round(time.perf_counter() - started, 3)
                results_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                results_handle.flush()
                pack_record(record, pack_dir)
                new_records += 1
                print(
                    f"{new_records:04d} {record['sample_id']} {record['object_id']} "
                    f"{record['decision']} {record.get('box_quality')} {record.get('latency_seconds')}s",
                    flush=True,
                )
                if args.interval_seconds > 0:
                    time.sleep(args.interval_seconds)

    records = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(flatten_record(records[0]).keys()) if records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(flatten_record(record))

    counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    for record in records:
        counts[record["decision"]] = counts.get(record["decision"], 0) + 1
        quality = record.get("box_quality") or "unknown"
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    summary_path.write_text(
        "\n".join(
            [
                "# Anomaly BBox QC Summary",
                "",
                f"- total_records: {len(records)}",
                f"- decision_counts: {json.dumps(counts, ensure_ascii=False)}",
                f"- box_quality_counts: {json.dumps(quality_counts, ensure_ascii=False)}",
                f"- results_jsonl: {results_path}",
                f"- results_csv: {csv_path}",
                f"- problem_pack: {pack_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
