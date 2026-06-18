from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

PRESETS: dict[str, dict[str, str]] = {
    "first_wave": {
        "images_clear_water": "weak_positive_clear_water",
        "images_wall_water": "weak_negative_wall_water",
        "images_others": "weak_negative_other",
    },
    "second_wave": {
        "images_floor_clear_water": "weak_positive_clear_water",
        "images_wall_water": "weak_negative_wall_water",
        "images_others": "weak_negative_other",
    },
}


PROMPTS: dict[str, str] = {
    "v1_zero_shot": """
You are selecting usable clear-water training images from a mixed industrial image pool.

Business goal:
Return ACCEPT only for images that can be used as clear_water positive samples.
The source folder is hidden from you in deployment, so judge from the image itself.

Clear-water positive:
- usable industrial/data-center/equipment-room image
- visible floor-level transparent or nearly transparent water
- acceptable forms: puddle, wet patch, thin water film, reflective wet boundary, water pooling near equipment or pipes
- subtle water is acceptable when there are real visual cues

Reject:
- wall-only leakage or vertical damp stains
- dry floor, ordinary glossy floor reflection, lamp reflection, shadows, floor texture, painted floor markings
- colored coolant/antifreeze/chemical liquid that does not look like clear water
- unreadable, heavily corrupted, nearly black, unrealistic image

Return strict JSON only with keys:
can_see_image(boolean),
usable(boolean),
image_quality(good/fair/poor),
floor_clear_water_visible(boolean),
evidence_strength(strong/medium/weak/none),
liquid_location(floor/wall/both/none/unclear),
hard_negative_type(wall_water_only/reflection_only/dry_floor/colored_liquid/unreadable/not_industrial/none/unclear),
decision(accept_clear_water/reject/review),
confidence(number),
evidence_summary(string).
""".strip(),
    "v2_hard_negatives": """
Role:
You are a conservative VLM quality-control agent for generated industrial inspection images.

Task:
From a mixed image pool, select only usable clear_water positive samples. The output list must be a subset of clear_water.
Losing some usable images is acceptable. Letting wall_water, other anomalies, or dry images pass is not acceptable.

Positive class definition:
ACCEPT when the image is usable and shows floor-level clear or nearly clear water.
Count these as clear_water:
- irregular transparent puddle on the floor
- wet patch or thin water film on the floor
- local wet reflection different from surrounding floor
- water spreading from equipment base, pipe, threshold, drain, or cable trench
- subtle floor water if at least two cues are present: boundary shape, wet shine, pooling area, contact with floor, local contrast

Hard negative examples:
Example A: The wall has stains or dripping marks, but the floor does not show clear water.
Output: reject, hard_negative_type=wall_water_only.
Example B: The floor is glossy and reflects lamps or cabinets evenly, with no irregular puddle boundary.
Output: reject, hard_negative_type=reflection_only.
Example C: The room is clean and dry, or only has shadows, floor texture, grid lines, labels, or painted marks.
Output: reject, hard_negative_type=dry_floor.
Example D: The visible liquid is green, red, pink, blue, yellow, milky, or looks like coolant/antifreeze/chemical fluid.
Output: reject, hard_negative_type=colored_liquid.
Example E: The image is nearly black, corrupted by large color blocks, or too blurry to inspect.
Output: reject, hard_negative_type=unreadable.

Decision policy:
- accept_clear_water: usable image plus credible floor clear-water evidence.
- review: image is usable but the floor evidence is too ambiguous.
- reject: dry image, wall-only water, colored liquid, reflection-only, unreadable, or non-industrial image.

Output strict JSON only, no markdown, with exactly these keys:
can_see_image(boolean),
usable(boolean),
image_quality(good/fair/poor),
floor_clear_water_visible(boolean),
evidence_strength(strong/medium/weak/none),
liquid_location(floor/wall/both/none/unclear),
hard_negative_type(wall_water_only/reflection_only/dry_floor/colored_liquid/unreadable/not_industrial/none/unclear),
decision(accept_clear_water/reject/review),
confidence(number),
evidence_summary(string).
""".strip(),
    "v3_recall_balanced": """
Role:
You are the automatic second screener for a generated-image dataset. A human has already tried to collect clear_water
positive samples, but the final pool is mixed. Your job is to recover usable clear_water images while keeping obvious
wall_water and other negatives out.

Business target:
- Final accepted images should be clear_water positive samples.
- Some loss is acceptable, but do not be overly strict on subtle transparent floor water.
- A good operating point is around 65 percent usable output on this batch if the evidence supports it.

Acceptable clear_water evidence:
1. floor puddle, thin water film, transparent wet patch, or local wet reflection
2. irregular edge or spreading shape on the floor
3. wet shine different from normal glossy floor
4. pooling near equipment base, pipe, cable trench, door threshold, drain, or wall-floor junction
5. subtle evidence is enough when two or more cues point to floor water

Reject hard negatives:
- wall_water_only: wet wall, vertical seepage, wall stain, pipe stain, or ceiling/vertical leakage without floor water
- reflection_only: clean polished floor, cabinet reflection, lamp glare, regular shiny epoxy floor, no irregular boundary
- dry_floor: normal floor texture, shadows, painted lines, labels, dust, stains that do not look wet
- colored_liquid: green/red/pink/blue/yellow/milky coolant or chemical liquid rather than clear water
- unreadable: black frame, large color block corruption, severe blur, unreadable generated artifact

Few-shot text calibration:
Positive example:
The floor has a faint transparent wet area with an uneven boundary and a local reflection near equipment.
decision=accept_clear_water, evidence_strength=medium.

Negative example:
The wall has a damp vertical stain, while the floor surface is dry or only glossy.
decision=reject, hard_negative_type=wall_water_only.

Negative example:
The floor reflects lights evenly, but there is no puddle edge, wet patch shape, or pooling region.
decision=reject, hard_negative_type=reflection_only.

Negative example:
The image contains a green or pink liquid patch. It may be coolant, but it is not clear_water.
decision=reject, hard_negative_type=colored_liquid.

Output strict JSON only, no markdown, with exactly these keys:
can_see_image(boolean),
usable(boolean),
image_quality(good/fair/poor),
floor_clear_water_visible(boolean),
evidence_strength(strong/medium/weak/none),
liquid_location(floor/wall/both/none/unclear),
hard_negative_type(wall_water_only/reflection_only/dry_floor/colored_liquid/unreadable/not_industrial/none/unclear),
decision(accept_clear_water/reject/review),
confidence(number),
evidence_summary(string).
""".strip(),
}


def now_id() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def get_api_config(args: argparse.Namespace) -> tuple[str, str]:
    base_url = (
        args.api_base_url
        or os.environ.get("QC_API_BASE_URL")
        or os.environ.get("QWEN397B_API_URL")
        or os.environ.get("QWEN_GEOMETRY_API_URL")
        or os.environ.get("SJTU_API_BASE_URL")
        or "https://deepseek.gds-services.com/v1"
    )
    api_key = (
        args.api_key
        or os.environ.get("QC_API_KEY")
        or os.environ.get("QWEN397B_API_KEY")
        or os.environ.get("SJTU_API_KEY")
    )
    if not base_url:
        raise SystemExit(
            "missing API base url: set --api-base-url or QC_API_BASE_URL/QWEN397B_API_URL/"
            "QWEN_GEOMETRY_API_URL/SJTU_API_BASE_URL"
        )
    if not api_key:
        raise SystemExit("missing API key: set --api-key or QC_API_KEY/QWEN397B_API_KEY/SJTU_API_KEY")
    return base_url.rstrip("/"), api_key


def clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    return cleaned


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def encode_image(path: Path, max_side: int, quality: int) -> tuple[str, tuple[int, int], tuple[int, int], int]:
    img = Image.open(path).convert("RGB")
    native_size = img.size
    img.thumbnail((max_side, max_side))
    resized_size = img.size
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    payload_bytes = buf.getvalue()
    return base64.b64encode(payload_bytes).decode("ascii"), native_size, resized_size, len(payload_bytes)


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: Path,
    max_side: int,
    jpeg_quality: int,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    image_b64, native_size, resized_size, payload_bytes = encode_image(image_path, max_side, jpeg_quality)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return strict JSON only. No markdown."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    t0 = time.time()
    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
    latency_s = round(time.time() - t0, 3)
    try:
        data = response.json()
    except ValueError:
        data = {"_raw_text": response.text[:1000]}
    return {
        "status_code": response.status_code,
        "latency_s": latency_s,
        "response": data,
        "native_size": native_size,
        "resized_size": resized_size,
        "payload_bytes": payload_bytes,
    }


def collect_images(root: Path, preset: str) -> list[dict[str, Any]]:
    label_map = PRESETS[preset]
    rows: list[dict[str, Any]] = []
    for folder, weak_label in label_map.items():
        folder_path = root / folder
        if not folder_path.exists():
            continue
        for path in sorted(folder_path.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rows.append(
                    {
                        "file": str(path.resolve()),
                        "file_name": path.name,
                        "folder": folder,
                        "weak_label": weak_label,
                    }
                )
    return sorted(rows, key=lambda row: (row["folder"], row["file_name"]))


def sample_rows(rows: list[dict[str, Any]], sample_per_folder: int | None, seed: int) -> list[dict[str, Any]]:
    if not sample_per_folder:
        return rows
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["folder"]].append(row)
    sampled: list[dict[str, Any]] = []
    for folder in sorted(grouped):
        items = grouped[folder]
        if len(items) <= sample_per_folder:
            sampled.extend(items)
        else:
            sampled.extend(sorted(rng.sample(items, sample_per_folder), key=lambda row: row["file_name"]))
    return sorted(sampled, key=lambda row: (row["folder"], row["file_name"]))


def load_jsonl_success(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok") is True:
                existing[row["file"]] = row
    return existing


def call_with_retries(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: Path,
    max_side: int,
    jpeg_quality: int,
    max_tokens: int,
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            result = post_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                image_path=image_path,
                max_side=max_side,
                jpeg_quality=jpeg_quality,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            response = result["response"]
            if "choices" not in response:
                body = json.dumps(response, ensure_ascii=False)[:1000]
                raise RuntimeError(f"missing choices in API response: status={result['status_code']} body={body}")
            content = response["choices"][0]["message"]["content"]
            parsed = json.loads(clean_json_text(content))
            usage = response.get("usage", {})
            return {
                "ok": True,
                "attempt": attempt,
                "raw_content": content,
                "parsed": parsed,
                "resolved_model": response.get("model"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                **{k: result[k] for k in ("status_code", "latency_s", "native_size", "resized_size", "payload_bytes")},
            }
        except Exception as exc:  # noqa: BLE001 - keep batch moving and record the exact failure.
            last_error = f"{type(exc).__name__}: {exc}"
            if "429" in last_error or "Rate limit exceeded" in last_error:
                time.sleep(65)
            else:
                time.sleep(min(2 * attempt, 10))
    return {"ok": False, "error": last_error, "attempt": max_retries}


def selected_by_rule(parsed: dict[str, Any], min_confidence: float, allow_weak: bool) -> tuple[bool, str]:
    decision = str(parsed.get("decision", "")).strip().lower()
    confidence = parse_float(parsed.get("confidence"), 0.0)
    hard_negative = str(parsed.get("hard_negative_type", "")).strip().lower()
    location = str(parsed.get("liquid_location", "")).strip().lower()
    strength = str(parsed.get("evidence_strength", "")).strip().lower()

    if not parse_bool(parsed.get("can_see_image")):
        return False, "cannot_see_image"
    if not parse_bool(parsed.get("usable")):
        return False, "unusable"
    if hard_negative in {
        "wall_water_only",
        "reflection_only",
        "dry_floor",
        "colored_liquid",
        "unreadable",
        "not_industrial",
    }:
        return False, f"hard_negative:{hard_negative}"
    if location not in {"floor", "both"}:
        return False, f"not_floor_liquid:{location}"
    if not parse_bool(parsed.get("floor_clear_water_visible")):
        return False, "no_floor_clear_water"
    if confidence < min_confidence:
        return False, f"low_confidence:{confidence:.2f}"
    if strength in {"strong", "medium"} and decision == "accept_clear_water":
        return True, "model_accept_strong_or_medium"
    if allow_weak and strength == "weak" and decision == "accept_clear_water":
        return True, "model_accept_weak_allowed"
    return False, f"decision_or_strength:{decision}/{strength}"


def flatten_result(
    source: dict[str, Any],
    result: dict[str, Any],
    *,
    run_id: str,
    prompt_version: str,
    model: str,
    min_confidence: float,
    allow_weak: bool,
) -> dict[str, Any]:
    parsed = result.get("parsed") or {}
    selected, rule_reason = selected_by_rule(parsed, min_confidence, allow_weak) if result.get("ok") else (False, "api_error")
    native = result.get("native_size") or ("", "")
    resized = result.get("resized_size") or ("", "")
    row = {
        "run_id": run_id,
        "prompt_version": prompt_version,
        "model": model,
        "resolved_model": result.get("resolved_model"),
        "ok": result.get("ok"),
        "file": source["file"],
        "file_name": source["file_name"],
        "folder": source["folder"],
        "weak_label": source["weak_label"],
        "selected_clear_water": selected,
        "selection_reason": rule_reason,
        "status_code": result.get("status_code"),
        "latency_s": result.get("latency_s"),
        "attempt": result.get("attempt"),
        "native_width": native[0],
        "native_height": native[1],
        "resized_width": resized[0],
        "resized_height": resized[1],
        "payload_bytes": result.get("payload_bytes"),
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
        "total_tokens": result.get("total_tokens"),
        "error": result.get("error"),
        "raw_content": result.get("raw_content"),
    }
    for key in (
        "can_see_image",
        "usable",
        "image_quality",
        "floor_clear_water_visible",
        "evidence_strength",
        "liquid_location",
        "hard_negative_type",
        "decision",
        "confidence",
        "evidence_summary",
    ):
        row[key] = parsed.get(key)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    selected = [row for row in rows if parse_bool(row.get("selected_clear_water"))]
    positives = [row for row in rows if row.get("weak_label") == "weak_positive_clear_water"]
    negative_selected = [row for row in selected if row.get("weak_label") != "weak_positive_clear_water"]
    positive_selected = [row for row in selected if row.get("weak_label") == "weak_positive_clear_water"]
    by_folder: dict[str, dict[str, int]] = {}
    for row in rows:
        folder = str(row.get("folder"))
        item = by_folder.setdefault(folder, {"total": 0, "selected": 0})
        item["total"] += 1
        if parse_bool(row.get("selected_clear_water")):
            item["selected"] += 1
    return {
        "total": total,
        "api_success": sum(1 for row in rows if parse_bool(row.get("ok"))),
        "selected": len(selected),
        "selected_rate": len(selected) / total if total else 0.0,
        "weak_positive_total": len(positives),
        "weak_positive_selected": len(positive_selected),
        "weak_positive_recall": len(positive_selected) / len(positives) if positives else 0.0,
        "weak_negative_selected": len(negative_selected),
        "weak_precision": len(positive_selected) / len(selected) if selected else 0.0,
        "by_folder": by_folder,
        "decision_counts": Counter(str(row.get("decision")) for row in rows),
        "hard_negative_counts": Counter(str(row.get("hard_negative_type")) for row in rows),
        "selection_reason_counts": Counter(str(row.get("selection_reason")) for row in rows),
        "latencies": [parse_float(row.get("latency_s"), 0.0) for row in rows if row.get("latency_s") not in (None, "")],
        "token_total": sum(int(float(row.get("total_tokens") or 0)) for row in rows),
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[idx]


def write_summary(path: Path, rows: list[dict[str, Any]], *, title: str, target_rate: float) -> None:
    summary = summarize_rows(rows)
    latencies = summary["latencies"]
    lines = [
        f"# {title}",
        "",
        f"total_images: {summary['total']}",
        f"api_success: {summary['api_success']}/{summary['total']}",
        f"selected_clear_water: {summary['selected']} ({summary['selected_rate']:.2%})",
        f"target_selected_rate: {target_rate:.2%}",
        f"weak_precision_by_folder_proxy: {summary['weak_precision']:.2%}",
        f"weak_recall_on_clear_water_folder: {summary['weak_positive_recall']:.2%}",
        f"weak_negative_selected: {summary['weak_negative_selected']}",
        f"token_total: {summary['token_total']}",
        f"latency_p50: {percentile(latencies, 0.5):.3f}s",
        f"latency_p95: {percentile(latencies, 0.95):.3f}s",
        "",
        "## By folder",
        "| folder | selected | total | rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for folder, item in sorted(summary["by_folder"].items()):
        total = item["total"]
        selected = item["selected"]
        lines.append(f"| {folder} | {selected} | {total} | {selected / total if total else 0:.2%} |")
    lines.extend(["", "## Decision counts"])
    for key, count in summary["decision_counts"].most_common():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Hard negative counts"])
    for key, count in summary["hard_negative_counts"].most_common():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Selection reason counts"])
    for key, count in summary["selection_reason_counts"].most_common():
        lines.append(f"- {key}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_filter(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "_qc_probe_clear_water_filter"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = now_id()
    prefix = args.prefix or f"{args.preset}_{args.prompt_version}"
    prompt = PROMPTS[args.prompt_version]
    base_url, api_key = get_api_config(args)

    all_rows = collect_images(root, args.preset)
    rows = sample_rows(all_rows, args.sample_per_folder, args.seed)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no images found under {root}")

    jsonl_path = out_dir / f"{prefix}_{run_id}.jsonl"
    partial_path = out_dir / f"{prefix}_{run_id}_partial.jsonl"
    resume_path = Path(args.resume_jsonl).resolve() if args.resume_jsonl else partial_path
    existing = load_jsonl_success(resume_path) if args.resume else {}

    flat_rows: list[dict[str, Any]] = []
    with partial_path.open("a", encoding="utf-8") as jf:
        for idx, source in enumerate(rows, start=1):
            image_path = Path(source["file"])
            if source["file"] in existing:
                result = existing[source["file"]]
            else:
                result = call_with_retries(
                    base_url=base_url,
                    api_key=api_key,
                    model=args.model,
                    prompt=prompt,
                    image_path=image_path,
                    max_side=args.max_side,
                    jpeg_quality=args.jpeg_quality,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )
                record = {**source, **result}
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                jf.flush()
                if idx < len(rows):
                    time.sleep(args.interval_seconds)
            flat = flatten_result(
                source,
                result,
                run_id=run_id,
                prompt_version=args.prompt_version,
                model=args.model,
                min_confidence=args.min_confidence,
                allow_weak=args.allow_weak,
            )
            flat_rows.append(flat)
            print(
                f"[{idx}/{len(rows)}] selected={flat['selected_clear_water']} "
                f"folder={source['folder']} file={source['file_name']} reason={flat['selection_reason']}",
                flush=True,
            )

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in flat_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = out_dir / f"{prefix}_{run_id}.csv"
    selected_csv = out_dir / f"{prefix}_{run_id}_selected.csv"
    rejected_csv = out_dir / f"{prefix}_{run_id}_rejected.csv"
    summary_path = out_dir / f"{prefix}_{run_id}_summary.md"
    write_csv(csv_path, flat_rows)
    write_csv(selected_csv, [row for row in flat_rows if parse_bool(row.get("selected_clear_water"))])
    write_csv(rejected_csv, [row for row in flat_rows if not parse_bool(row.get("selected_clear_water"))])
    write_summary(summary_path, flat_rows, title=f"{prefix} clear-water filter summary", target_rate=args.target_rate)
    for latest, source in {
        out_dir / f"{prefix}_latest.jsonl": jsonl_path,
        out_dir / f"{prefix}_latest.csv": csv_path,
        out_dir / f"{prefix}_latest_selected.csv": selected_csv,
        out_dir / f"{prefix}_latest_rejected.csv": rejected_csv,
        out_dir / f"{prefix}_latest_summary.md": summary_path,
    }.items():
        shutil.copy2(source, latest)
    print(f"CSV={csv_path}")
    print(f"SELECTED_CSV={selected_csv}")
    print(f"SUMMARY={summary_path}")
    return 0


def create_contact_sheet(image_paths: list[Path], output: Path) -> Path | None:
    if not image_paths:
        return None
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font_big = ImageFont.truetype(str(font_path), 30) if font_path.exists() else ImageFont.load_default()
    font_small = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    thumb_w, thumb_h, label_h = 380, 214, 60
    cols = 4
    rows = (len(image_paths) + cols - 1) // cols
    margin, gap = 22, 16
    sheet = Image.new(
        "RGB",
        (margin * 2 + cols * thumb_w + (cols - 1) * gap, margin * 2 + rows * (thumb_h + label_h) + (rows - 1) * gap),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for idx, path in enumerate(image_paths, start=1):
        col = (idx - 1) % cols
        row = (idx - 1) // cols
        x = margin + col * (thumb_w + gap)
        y = margin + row * (thumb_h + label_h + gap)
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
        canvas.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        sheet.paste(canvas, (x, y))
        draw.rectangle([x, y, x + thumb_w - 1, y + thumb_h - 1], outline=(0, 120, 80), width=3)
        draw.rectangle([x, y, x + 58, y + 36], fill=(0, 0, 0))
        draw.text((x + 7, y + 2), f"#{idx}", fill="white", font=font_big)
        draw.text((x, y + thumb_h + 4), path.name[:44], fill=(0, 0, 0), font=font_small)
    sheet.save(output, quality=92)
    return output


def build_pack(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else csv_path.parent / f"clear_water_selected_pack_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = out_dir / "00_selected_clear_water"
    rejected_dir = out_dir / "01_rejected_or_review"
    selected_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    selected_paths: list[Path] = []

    for row in read_csv(csv_path):
        source = Path(row["file"])
        selected = parse_bool(row.get("selected_clear_water"))
        if args.selected_only and not selected:
            continue
        target_dir = selected_dir if selected else rejected_dir
        safe_name = f"{row.get('folder', 'unknown')}__{source.name}"
        dest = target_dir / safe_name
        shutil.copy2(source, dest)
        if selected:
            selected_paths.append(dest)
        manifest.append({**row, "pack_file": str(dest), "pack_bucket": target_dir.name})

    write_csv(out_dir / "manifest.csv", manifest)
    create_contact_sheet(selected_paths[: min(len(selected_paths), args.contact_sheet_limit)], out_dir / "selected_contact_sheet.jpg")
    archive_base = shutil.make_archive(str(out_dir), "zip", out_dir)
    print(f"PACK={out_dir}")
    print(f"ZIP={archive_base}")
    print(f"MANIFEST={out_dir / 'manifest.csv'}")
    return 0


def summarize_command(args: argparse.Namespace) -> int:
    rows = read_csv(Path(args.csv).resolve())
    write_summary(Path(args.out).resolve(), rows, title=args.title, target_rate=args.target_rate)
    print(f"SUMMARY={Path(args.out).resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter mixed images into a clear-water positive subset with a VLM.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_api_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--api-base-url", default=None)
        p.add_argument("--api-key", default=None)
        p.add_argument(
            "--model",
            default=(
                os.environ.get("QWEN_GEOMETRY_MODEL")
                or os.environ.get("QWEN397B_MODEL")
                or os.environ.get("SJTU_API_DEFAULT_MODEL")
                or "aios-smart-eye-vlm"
            ),
        )
        p.add_argument("--interval-seconds", type=float, default=6.8)
        p.add_argument("--timeout", type=int, default=90)
        p.add_argument("--max-retries", type=int, default=3)
        p.add_argument("--max-tokens", type=int, default=420)
        p.add_argument("--max-side", type=int, default=900)
        p.add_argument("--jpeg-quality", type=int, default=72)

    run = sub.add_parser("run")
    run.add_argument("--root", required=True)
    run.add_argument("--preset", choices=sorted(PRESETS), required=True)
    run.add_argument("--prompt-version", choices=sorted(PROMPTS), default="v3_recall_balanced")
    run.add_argument("--out-dir", default=None)
    run.add_argument("--prefix", default=None)
    run.add_argument("--sample-per-folder", type=int, default=None)
    run.add_argument("--seed", type=int, default=20260610)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--resume-jsonl", default=None)
    run.add_argument("--min-confidence", type=float, default=0.55)
    run.add_argument("--allow-weak", action="store_true")
    run.add_argument("--target-rate", type=float, default=0.65)
    add_api_args(run)
    run.set_defaults(func=run_filter)

    pack = sub.add_parser("build-pack")
    pack.add_argument("--csv", required=True)
    pack.add_argument("--out-dir", default=None)
    pack.add_argument("--contact-sheet-limit", type=int, default=120)
    pack.add_argument("--selected-only", action="store_true")
    pack.set_defaults(func=build_pack)

    summary = sub.add_parser("summarize")
    summary.add_argument("--csv", required=True)
    summary.add_argument("--out", required=True)
    summary.add_argument("--title", default="clear-water filter summary")
    summary.add_argument("--target-rate", type=float, default=0.65)
    summary.set_defaults(func=summarize_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
