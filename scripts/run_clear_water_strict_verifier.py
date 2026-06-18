from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

from run_clear_water_filter import (
    call_with_retries,
    get_api_config,
    load_jsonl_success,
    parse_bool,
    parse_float,
    write_csv,
    write_summary,
)


PROMPT_VERSION = "v4_strict_generalized_verifier"

PROMPT = """
Role:
You are the final visual quality-control verifier for clear-water anomaly images.

Important constraint:
Judge only from the image pixels. Do not assume anything from file names, folders, or prior labels.

Goal:
Accept only images that are safe to use as clear_water positive samples.
It is acceptable to lose usable positives. It is not acceptable to pass dry floors, wall-only water,
reflection-only cases, colored liquid, or unclear images.

Accept clear_water only when all conditions hold:
1. The image is usable and visually inspectable.
2. The liquid evidence is on the floor, not only on a wall or vertical surface.
3. The liquid looks transparent or nearly transparent, not green, pink, red, blue, yellow, milky, oily, or chemical.
4. There are at least two real floor-water cues, such as:
   - irregular puddle or wet boundary
   - local wet sheen that differs from normal glossy floor
   - transparent wet patch or thin film
   - pooling or spreading near equipment base, pipe, threshold, drain, or wall-floor junction
   - visible contact with floor surface
5. If the evidence is subtle, accept only when the visible cues support floor water directly.

Reject or review these hard negatives:
- wall_water_only: stains, seepage, dripping, damp marks, or wet pipe/wall with no floor water
- reflection_only: normal glossy floor, lamp/cabinet reflection, uniform shine, no irregular wet boundary
- dry_floor: shadows, dust, scratches, painted marks, texture, labels, grid lines, normal floor patterns
- colored_liquid: coolant, antifreeze, oil, chemical liquid, or any obviously non-clear liquid color
- unreadable: corrupted, too dark, too blurry, severe generated artifact, or impossible to inspect
- unclear: the image may contain water, but the visual evidence is weak or not localized on the floor

Decision policy:
- accept_clear_water: only for usable images with clear floor-level transparent water evidence and no hard negative.
- review: usable image, but the floor-water evidence is ambiguous or too weak.
- reject: hard negative, unreadable, non-industrial, colored liquid, or no floor clear water.

Return strict JSON only, no markdown, with exactly these keys:
can_see_image(boolean),
usable(boolean),
image_quality(good/fair/poor),
floor_clear_water_visible(boolean),
evidence_strength(strong/medium/weak/none),
floor_water_cues(array of short strings),
liquid_location(floor/wall/both/none/unclear),
hard_negative_type(wall_water_only/reflection_only/dry_floor/colored_liquid/unreadable/not_industrial/unclear/none),
decision(accept_clear_water/reject/review),
confidence(number),
reject_reason(string),
evidence_summary(string).
""".strip()


POSITIVE_FOLDERS = {"images_floor_clear_water", "images_clear_water"}
HARD_NEGATIVES = {
    "wall_water_only",
    "reflection_only",
    "dry_floor",
    "colored_liquid",
    "unreadable",
    "not_industrial",
    "unclear",
}


def now_id() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            loaded = json.loads(stripped)
            if isinstance(loaded, list):
                return [str(item).strip() for item in loaded if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [part.strip() for part in stripped.split(";") if part.strip()]
    return []


def row_is_weak_positive(row: dict[str, Any]) -> bool:
    folder = str(row.get("folder", "")).strip()
    weak_label = str(row.get("weak_label", "")).strip().lower()
    return folder in POSITIVE_FOLDERS or weak_label == "weak_positive_clear_water"


def stage2_rule(
    parsed: dict[str, Any],
    *,
    min_confidence: float,
    min_cues: int,
    strong_only: bool,
) -> tuple[bool, str]:
    if not parse_bool(parsed.get("can_see_image")):
        return False, "cannot_see_image"
    if not parse_bool(parsed.get("usable")):
        return False, "unusable"
    if not parse_bool(parsed.get("floor_clear_water_visible")):
        return False, "no_floor_clear_water"

    decision = str(parsed.get("decision", "")).strip().lower()
    if decision != "accept_clear_water":
        return False, f"decision:{decision or 'empty'}"

    location = str(parsed.get("liquid_location", "")).strip().lower()
    if location not in {"floor", "both"}:
        return False, f"not_floor_liquid:{location or 'empty'}"

    hard_negative = str(parsed.get("hard_negative_type", "")).strip().lower()
    if hard_negative in HARD_NEGATIVES:
        return False, f"hard_negative:{hard_negative}"
    if hard_negative and hard_negative != "none":
        return False, f"hard_negative:{hard_negative}"

    strength = str(parsed.get("evidence_strength", "")).strip().lower()
    allowed_strength = {"strong"} if strong_only else {"strong", "medium"}
    if strength not in allowed_strength:
        return False, f"evidence_strength:{strength or 'empty'}"

    confidence = parse_float(parsed.get("confidence"), 0.0)
    if confidence < min_confidence:
        return False, f"low_confidence:{confidence:.2f}"

    cues = as_list(parsed.get("floor_water_cues"))
    if len(cues) < min_cues:
        return False, f"too_few_cues:{len(cues)}"

    return True, f"stage2_accept_{strength}_conf>={min_confidence:.2f}_cues>={min_cues}"


def flatten_stage2(
    source: dict[str, Any],
    result: dict[str, Any],
    *,
    run_id: str,
    model: str,
    min_confidence: float,
    min_cues: int,
    strong_only: bool,
) -> dict[str, Any]:
    parsed = result.get("parsed") or {}
    selected, reason = (
        stage2_rule(
            parsed,
            min_confidence=min_confidence,
            min_cues=min_cues,
            strong_only=strong_only,
        )
        if result.get("ok")
        else (False, "api_error")
    )
    native = result.get("native_size") or ("", "")
    resized = result.get("resized_size") or ("", "")
    row = {
        "run_id": run_id,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "resolved_model": result.get("resolved_model"),
        "ok": result.get("ok"),
        "file": source.get("file"),
        "file_name": source.get("file_name"),
        "folder": source.get("folder"),
        "weak_label": source.get("weak_label"),
        "first_pass_selected_clear_water": source.get("selected_clear_water"),
        "first_pass_decision": source.get("decision"),
        "first_pass_confidence": source.get("confidence"),
        "first_pass_evidence_summary": source.get("evidence_summary"),
        "selected_clear_water": selected,
        "selection_reason": reason,
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
        "floor_water_cues",
        "liquid_location",
        "hard_negative_type",
        "decision",
        "confidence",
        "reject_reason",
        "evidence_summary",
    ):
        value = parsed.get(key)
        row[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
    return row


def load_candidate_rows(path: Path, only_selected: bool) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    if only_selected:
        rows = [row for row in rows if parse_bool(row.get("selected_clear_water"))]
    clean_rows = []
    for row in rows:
        file_path = Path(str(row.get("file", "")))
        if not file_path.exists():
            continue
        if not row.get("file_name"):
            row["file_name"] = file_path.name
        clean_rows.append(row)
    return clean_rows


def apply_rule_to_flat_rows(
    rows: list[dict[str, Any]],
    *,
    min_confidence: float,
    min_cues: int,
    strong_only: bool,
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for row in rows:
        parsed = {
            "can_see_image": row.get("can_see_image"),
            "usable": row.get("usable"),
            "floor_clear_water_visible": row.get("floor_clear_water_visible"),
            "decision": row.get("decision"),
            "liquid_location": row.get("liquid_location"),
            "hard_negative_type": row.get("hard_negative_type"),
            "evidence_strength": row.get("evidence_strength"),
            "confidence": row.get("confidence"),
            "floor_water_cues": row.get("floor_water_cues"),
        }
        selected, reason = stage2_rule(
            parsed,
            min_confidence=min_confidence,
            min_cues=min_cues,
            strong_only=strong_only,
        )
        new_row = dict(row)
        new_row["selected_clear_water"] = selected
        new_row["selection_reason"] = reason
        updated.append(new_row)
    return updated


def score_rows(rows: list[dict[str, Any]], *, selection_denominator: int | None = None) -> dict[str, Any]:
    total = len(rows)
    denominator = selection_denominator or total
    selected = [row for row in rows if parse_bool(row.get("selected_clear_water"))]
    weak_pos_selected = sum(1 for row in selected if row_is_weak_positive(row))
    weak_neg_selected = len(selected) - weak_pos_selected
    weak_pos_total = sum(1 for row in rows if row_is_weak_positive(row))
    return {
        "total": total,
        "selection_denominator": denominator,
        "selected": len(selected),
        "selected_rate": (len(selected) / denominator) if denominator else 0.0,
        "candidate_selected_rate": (len(selected) / total) if total else 0.0,
        "weak_pos_selected": weak_pos_selected,
        "weak_neg_selected": weak_neg_selected,
        "weak_precision": (weak_pos_selected / len(selected)) if selected else 0.0,
        "weak_recall": (weak_pos_selected / weak_pos_total) if weak_pos_total else 0.0,
    }


def build_calibration(
    rows: list[dict[str, Any]],
    *,
    target_max_rate: float,
    selection_denominator: int | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    thresholds = [round(x / 100, 2) for x in range(50, 100)]
    for strong_only in (True, False):
        for min_cues in (2, 1):
            for threshold in thresholds:
                ruled = apply_rule_to_flat_rows(
                    rows,
                    min_confidence=threshold,
                    min_cues=min_cues,
                    strong_only=strong_only,
                )
                score = score_rows(ruled, selection_denominator=selection_denominator)
                candidates.append(
                    {
                        "strong_only": strong_only,
                        "min_cues": min_cues,
                        "min_confidence": threshold,
                        "selected": score["selected"],
                        "selected_rate": score["selected_rate"],
                        "candidate_selected_rate": score["candidate_selected_rate"],
                        "selection_denominator": score["selection_denominator"],
                        "weak_neg_selected": score["weak_neg_selected"],
                        "weak_precision": score["weak_precision"],
                        "weak_recall": score["weak_recall"],
                        "fits_target_max_rate": score["selected_rate"] <= target_max_rate,
                    }
                )
    return sorted(
        candidates,
        key=lambda item: (
            item["weak_neg_selected"] > 0,
            item["selected_rate"] > target_max_rate,
            -item["selected"],
            -item["min_confidence"],
        ),
    )


def choose_zero_negative_rule(
    calibration: list[dict[str, Any]],
    *,
    target_max_rate: float,
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in calibration
        if row["weak_neg_selected"] == 0 and row["selected"] > 0 and row["selected_rate"] <= target_max_rate
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            row["selected"],
            row["weak_recall"],
            row["weak_precision"],
            row["strong_only"],
            -row["min_confidence"],
        ),
    )


def write_calibration(path: Path, rows: list[dict[str, Any]], *, limit: int = 80) -> None:
    write_csv(path, rows[:limit])


def write_generalization_note(
    path: Path,
    *,
    chosen: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    selection_denominator: int | None,
) -> None:
    score = score_rows(rows, selection_denominator=selection_denominator)
    folder_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        folder_counts[str(row.get("folder", ""))]["total"] += 1
        if parse_bool(row.get("selected_clear_water")):
            folder_counts[str(row.get("folder", ""))]["selected"] += 1

    lines = [
        "# Generalized clear_water verifier note",
        "",
        "This run uses source folders only for offline scoring. Runtime selection uses image-derived VLM fields only:",
        "`can_see_image`, `usable`, `floor_clear_water_visible`, `liquid_location`, `hard_negative_type`,",
        "`evidence_strength`, `confidence`, and `floor_water_cues`.",
        "",
        f"prompt_version: {PROMPT_VERSION}",
        f"selected_clear_water: {score['selected']} ({score['selected_rate']:.2%})",
        f"candidate_selected_rate: {score['candidate_selected_rate']:.2%}",
        f"selection_denominator: {score['selection_denominator']}",
        f"weak_negative_selected: {score['weak_neg_selected']}",
        f"weak_precision_by_folder_proxy: {score['weak_precision']:.2%}",
        f"weak_recall_on_clear_water_folder: {score['weak_recall']:.2%}",
    ]
    if chosen:
        lines.extend(
            [
                "",
                "## Chosen rule",
                f"strong_only: {chosen['strong_only']}",
                f"min_cues: {chosen['min_cues']}",
                f"min_confidence: {chosen['min_confidence']:.2f}",
                f"target_max_rate: {chosen['selected_rate']:.2%}",
                f"candidate_selected_rate: {chosen['candidate_selected_rate']:.2%}",
            ]
        )
    lines.extend(["", "## Folder audit, not runtime input", "| folder | selected | total | rate |", "| --- | ---: | ---: | ---: |"])
    for folder, counter in sorted(folder_counts.items()):
        total = counter["total"]
        selected = counter["selected"]
        lines.append(f"| {folder} | {selected} | {total} | {selected / total if total else 0:.2%} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    input_csv = Path(args.input_csv).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else input_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or "second_wave_clear_water_v4_generalized_verifier"
    run_id = now_id()
    base_url, api_key = get_api_config(args)

    rows = load_candidate_rows(input_csv, only_selected=args.only_selected)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no candidate images found in {input_csv}")

    partial_path = out_dir / f"{prefix}_{run_id}_partial.jsonl"
    resume_path = Path(args.resume_jsonl).resolve() if args.resume_jsonl else partial_path
    existing = load_jsonl_success(resume_path) if args.resume else {}
    flat_rows: list[dict[str, Any]] = []

    with partial_path.open("a", encoding="utf-8") as jf:
        pending_sources: list[dict[str, Any]] = []
        for source in rows:
            if source["file"] in existing:
                result = existing[source["file"]]
                flat = flatten_stage2(
                    source,
                    result,
                    run_id=run_id,
                    model=args.model,
                    min_confidence=args.min_confidence,
                    min_cues=args.min_cues,
                    strong_only=args.strong_only,
                )
                flat_rows.append(flat)
                print(
                    f"[{len(flat_rows)}/{len(rows)}] selected={flat['selected_clear_water']} "
                    f"folder={source.get('folder')} file={source.get('file_name')} reason={flat['selection_reason']} resume=True",
                    flush=True,
                )
                continue
            pending_sources.append(source)

        def process_source(source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            image_path = Path(str(source["file"]))
            result = call_with_retries(
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                prompt=PROMPT,
                image_path=image_path,
                max_side=args.max_side,
                jpeg_quality=args.jpeg_quality,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            return source, result

        if args.workers <= 1:
            for source in pending_sources:
                source, result = process_source(source)
                jf.write(json.dumps({**source, **result}, ensure_ascii=False) + "\n")
                jf.flush()
                flat = flatten_stage2(
                    source,
                    result,
                    run_id=run_id,
                    model=args.model,
                    min_confidence=args.min_confidence,
                    min_cues=args.min_cues,
                    strong_only=args.strong_only,
                )
                flat_rows.append(flat)
                print(
                    f"[{len(flat_rows)}/{len(rows)}] selected={flat['selected_clear_water']} "
                    f"folder={source.get('folder')} file={source.get('file_name')} reason={flat['selection_reason']}",
                    flush=True,
                )
                if args.interval_seconds:
                    time.sleep(args.interval_seconds)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures: dict[Any, dict[str, Any]] = {}
                next_index = 0
                last_submit_at = 0.0
                while next_index < len(pending_sources) or futures:
                    while next_index < len(pending_sources) and len(futures) < args.workers:
                        if args.interval_seconds and last_submit_at:
                            wait_s = args.interval_seconds - (time.time() - last_submit_at)
                            if wait_s > 0:
                                time.sleep(wait_s)
                        source = pending_sources[next_index]
                        futures[executor.submit(process_source, source)] = source
                        next_index += 1
                        last_submit_at = time.time()
                    done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                    if not done:
                        continue
                    for future in done:
                        source = futures.pop(future)
                        try:
                            source, result = future.result()
                        except Exception as exc:  # noqa: BLE001 - preserve batch progress.
                            result = {
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                                "attempt": 0,
                            }
                        jf.write(json.dumps({**source, **result}, ensure_ascii=False) + "\n")
                        jf.flush()
                        flat = flatten_stage2(
                            source,
                            result,
                            run_id=run_id,
                            model=args.model,
                            min_confidence=args.min_confidence,
                            min_cues=args.min_cues,
                            strong_only=args.strong_only,
                        )
                        flat_rows.append(flat)
                        print(
                            f"[{len(flat_rows)}/{len(rows)}] selected={flat['selected_clear_water']} "
                            f"folder={source.get('folder')} file={source.get('file_name')} reason={flat['selection_reason']}",
                            flush=True,
                        )

    chosen_rule: dict[str, Any] | None = None
    selection_denominator = args.target_denominator or len(rows)
    calibration = build_calibration(
        flat_rows,
        target_max_rate=args.target_max_rate,
        selection_denominator=selection_denominator,
    )
    if args.auto_zero_negative:
        chosen_rule = choose_zero_negative_rule(calibration, target_max_rate=args.target_max_rate)
        if chosen_rule:
            flat_rows = apply_rule_to_flat_rows(
                flat_rows,
                min_confidence=float(chosen_rule["min_confidence"]),
                min_cues=int(chosen_rule["min_cues"]),
                strong_only=parse_bool(chosen_rule["strong_only"]),
            )

    csv_path = out_dir / f"{prefix}_{run_id}.csv"
    selected_csv = out_dir / f"{prefix}_{run_id}_selected.csv"
    rejected_csv = out_dir / f"{prefix}_{run_id}_rejected.csv"
    summary_path = out_dir / f"{prefix}_{run_id}_summary.md"
    calibration_path = out_dir / f"{prefix}_{run_id}_calibration.csv"
    note_path = out_dir / f"{prefix}_{run_id}_generalization_note.md"
    jsonl_path = out_dir / f"{prefix}_{run_id}.jsonl"

    selected_rows = [row for row in flat_rows if parse_bool(row.get("selected_clear_water"))]
    rejected_rows = [row for row in flat_rows if not parse_bool(row.get("selected_clear_water"))]

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in flat_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(csv_path, flat_rows)
    write_csv(selected_csv, selected_rows)
    write_csv(rejected_csv, rejected_rows)
    write_summary(summary_path, flat_rows, title=f"{prefix} summary", target_rate=args.target_max_rate)
    write_calibration(calibration_path, calibration)
    write_generalization_note(
        note_path,
        chosen=chosen_rule,
        rows=flat_rows,
        selection_denominator=selection_denominator,
    )

    for latest, source in {
        out_dir / f"{prefix}_latest.jsonl": jsonl_path,
        out_dir / f"{prefix}_latest.csv": csv_path,
        out_dir / f"{prefix}_latest_selected.csv": selected_csv,
        out_dir / f"{prefix}_latest_rejected.csv": rejected_csv,
        out_dir / f"{prefix}_latest_summary.md": summary_path,
        out_dir / f"{prefix}_latest_calibration.csv": calibration_path,
        out_dir / f"{prefix}_latest_generalization_note.md": note_path,
    }.items():
        shutil.copy2(source, latest)

    print(f"CSV={csv_path}")
    print(f"SELECTED_CSV={selected_csv}")
    print(f"SUMMARY={summary_path}")
    print(f"CALIBRATION={calibration_path}")
    print(f"NOTE={note_path}")
    if chosen_rule:
        print(f"CHOSEN_RULE={chosen_rule}")
    else:
        print("CHOSEN_RULE=None")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Second-stage generalized clear-water verifier.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--only-selected", action="store_true", default=True)
    parser.add_argument("--include-all-input", dest="only_selected", action="store_false")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-jsonl", default=None)
    parser.add_argument(
        "--model",
        default=(
            os.environ.get("QWEN_GEOMETRY_MODEL")
            or os.environ.get("QWEN397B_MODEL")
            or os.environ.get("SJTU_API_DEFAULT_MODEL")
            or "aios-smart-eye-vlm"
        ),
    )
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--interval-seconds", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=520)
    parser.add_argument("--max-side", type=int, default=900)
    parser.add_argument("--jpeg-quality", type=int, default=72)
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--min-cues", type=int, default=2)
    parser.add_argument("--strong-only", action="store_true", default=True)
    parser.add_argument("--allow-medium", dest="strong_only", action="store_false")
    parser.add_argument("--auto-zero-negative", action="store_true")
    parser.add_argument("--target-max-rate", type=float, default=0.65)
    parser.add_argument("--target-denominator", type=int, default=None)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
