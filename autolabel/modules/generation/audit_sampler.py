from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from ...utils import ensure_dir, write_csv


def select_audit_samples(
    results: list[dict[str, Any]],
    sampling_ratio: float = 0.1,
    min_samples: int = 10,
) -> list[dict[str, Any]]:
    if not results:
        return []

    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_sample[str(row.get("sample_id", ""))].append(row)

    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    for sample_id, rows in sorted(by_sample.items()):
        if any(not bool(row.get("success")) for row in rows):
            failures.extend(rows)
        else:
            successes.extend(rows)

    target_count = max(min_samples, round(len(by_sample) * sampling_ratio))
    selected = list(failures)
    if len(selected) < target_count:
        selected.extend(successes[: max(0, target_count - len(selected))])
    return selected


def write_audit_sample_csv(
    results: list[dict[str, Any]],
    output_path: str | Path,
    sampling_ratio: float = 0.1,
    min_samples: int = 10,
) -> Path | None:
    selected = select_audit_samples(results, sampling_ratio=sampling_ratio, min_samples=min_samples)
    if not selected:
        return None

    target = Path(output_path)
    ensure_dir(target.parent)
    rows = [_serialize_audit_row(row) for row in selected]
    fieldnames = [
        "task_id",
        "sample_id",
        "object_id",
        "image_uri",
        "generated_image_uri",
        "localizer_used",
        "localizer",
        "attempt_role",
        "success",
        "passes_quality",
        "quality_reason",
        "reason",
        "failure_reason",
        "fallback_used",
        "final_bbox",
        "mask_uri",
        "crop_uri",
        "elapsed_ms",
        "anomaly_type",
        "background_preservation_score",
        "anomaly_visibility_score",
        "quality_score_bucket",
        "pgcd_component_score_bucket",
        "manual_label",
        "manual_accept",
        "manual_comment",
    ]
    write_csv(target, rows, fieldnames)
    return target


def _bucket_score(value: Any) -> str:
    if value in (None, ""):
        return "unknown"
    score = float(value)
    if score >= 0.75:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def _serialize_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    manual_accept = row.get("manual_accept")
    manual_label = row.get("manual_label")
    if manual_label in (None, "") and manual_accept is not None:
        manual_label = "accepted" if bool(manual_accept) else "rejected"

    return {
        "task_id": row.get("task_id"),
        "sample_id": row.get("sample_id"),
        "object_id": row.get("object_id"),
        "image_uri": row.get("image_uri"),
        "generated_image_uri": row.get("generated_image_uri"),
        "localizer_used": row.get("localizer_used"),
        "localizer": row.get("localizer"),
        "attempt_role": row.get("attempt_role"),
        "success": row.get("success"),
        "passes_quality": row.get("passes_quality"),
        "quality_reason": row.get("quality_reason"),
        "reason": row.get("reason"),
        "failure_reason": row.get("failure_reason") or row.get("reason"),
        "fallback_used": row.get("fallback_used"),
        "final_bbox": (
            json.dumps(row.get("final_bbox"), ensure_ascii=False)
            if row.get("final_bbox") is not None
            else ""
        ),
        "mask_uri": row.get("mask_uri"),
        "crop_uri": row.get("crop_uri"),
        "elapsed_ms": row.get("elapsed_ms"),
        "anomaly_type": row.get("anomaly_type"),
        "background_preservation_score": row.get("background_preservation_score"),
        "anomaly_visibility_score": row.get("anomaly_visibility_score"),
        "quality_score_bucket": _bucket_score(
            min(
                float(row.get("background_preservation_score", 0.0)),
                float(row.get("anomaly_visibility_score", 0.0)),
            )
            if row.get("background_preservation_score") is not None
            and row.get("anomaly_visibility_score") is not None
            else None
        ),
        "pgcd_component_score_bucket": _bucket_score(row.get("pgcd_component_score")),
        "manual_label": manual_label,
        "manual_accept": manual_accept,
        "manual_comment": row.get("manual_comment"),
    }
