from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ...utils import ensure_dir, write_csv, write_json


def bbox_iou(
    predicted: list[int] | tuple[int, int, int, int] | None,
    reference: list[int] | tuple[int, int, int, int] | None,
) -> float:
    if predicted is None or reference is None:
        return 0.0
    px1, py1, px2, py2 = [float(value) for value in predicted]
    rx1, ry1, rx2, ry2 = [float(value) for value in reference]
    inter_x1 = max(px1, rx1)
    inter_y1 = max(py1, ry1)
    inter_x2 = min(px2, rx2)
    inter_y2 = min(py2, ry2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    predicted_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    reference_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = predicted_area + reference_area - inter
    return inter / union if union > 0 else 0.0


def _load_mask(mask_path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(mask_path) as image:
        if image.size != image_size:
            image = image.resize(image_size, Image.NEAREST)
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def _mask_from_bbox(
    bbox: list[int] | tuple[int, int, int, int] | None,
    image_size: tuple[int, int],
) -> np.ndarray | None:
    if bbox is None:
        return None
    width, height = image_size
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1 = min(max(x1, 0), width)
    x2 = min(max(x2, 0), width)
    y1 = min(max(y1, 0), height)
    y2 = min(max(y2, 0), height)
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def compute_localizer_metrics(
    final_bbox: list[int] | tuple[int, int, int, int] | None,
    mask_path: str | Path | None,
    image_size: tuple[int, int] | None,
    reference_bbox: list[int] | tuple[int, int, int, int] | None = None,
    reference_mask_path: str | Path | None = None,
) -> dict[str, Any]:
    metrics = {
        "bbox_iou": bbox_iou(final_bbox, reference_bbox),
        "precision": 0.0,
        "recall": 0.0,
        "mask_iou": 0.0,
        "metric_reference_source": "none",
    }
    if image_size is None:
        return metrics

    reference_mask: np.ndarray | None = None
    if reference_mask_path and Path(reference_mask_path).exists():
        reference_mask = _load_mask(reference_mask_path, image_size)
        metrics["metric_reference_source"] = "reference_mask"
    elif reference_bbox is not None:
        reference_mask = _mask_from_bbox(reference_bbox, image_size)
        metrics["metric_reference_source"] = "reference_bbox"

    if mask_path is None or not Path(mask_path).exists() or reference_mask is None:
        return metrics

    predicted_mask = _load_mask(mask_path, image_size)
    predicted_positive = np.logical_and(predicted_mask, reference_mask)
    predicted_count = int(predicted_mask.sum())
    reference_count = int(reference_mask.sum())
    intersection = int(predicted_positive.sum())
    union = int(np.logical_or(predicted_mask, reference_mask).sum())

    metrics["precision"] = intersection / predicted_count if predicted_count else 0.0
    metrics["recall"] = intersection / reference_count if reference_count else 0.0
    metrics["mask_iou"] = intersection / union if union else 0.0
    return metrics


def summarize_localizer_results(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["localizer"])].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for localizer, rows in grouped.items():
        summary[localizer] = _summarize_group(rows)
    return summary


def summarize_localizer_results_by_anomaly_type(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row.get("anomaly_type") or "unknown")].append(row)
    return {anomaly_type: _summarize_group(rows) for anomaly_type, rows in grouped.items()}


def summarize_localizer_results_by_anomaly_type_and_localizer(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in results:
        anomaly_type = str(row.get("anomaly_type") or "unknown")
        localizer = str(row.get("localizer") or "unknown")
        grouped[anomaly_type][localizer].append(row)

    return {
        anomaly_type: {
            localizer: _summarize_group(rows)
            for localizer, rows in localizer_groups.items()
        }
        for anomaly_type, localizer_groups in grouped.items()
    }


def build_benchmark_aggregate_views(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_localizer": summarize_localizer_results(results),
        "by_anomaly_type": summarize_localizer_results_by_anomaly_type(results),
        "by_anomaly_type_and_localizer": summarize_localizer_results_by_anomaly_type_and_localizer(results),
    }


def _mean(values) -> float:
    numeric = [float(value) for value in values]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _manual_accept_value(row: dict[str, Any]) -> bool | None:
    for key in ("manual_accept", "manual_review_accepted", "manual_acceptance"):
        value = row.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "pass", "accepted"}:
            return True
        if normalized in {"0", "false", "no", "fail", "rejected"}:
            return False
    return None


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(rows)
    successes = [row for row in rows if bool(row.get("success"))]
    failure_reasons = Counter(str(row.get("reason")) for row in rows if row.get("reason"))
    manual_accept_values = [value for row in rows if (value := _manual_accept_value(row)) is not None]
    return {
        "sample_count": sample_count,
        "success_rate": (len(successes) / sample_count) if sample_count else 0.0,
        "fallback_rate": _mean(1.0 if bool(row.get("fallback_used")) else 0.0 for row in rows),
        "quality_pass_rate": _mean(
            1.0 if bool(row.get("passes_quality")) else 0.0
            for row in rows
            if row.get("passes_quality") is not None
        ),
        "mean_bbox_iou": _mean(row.get("bbox_iou", 0.0) for row in rows),
        "mean_precision": _mean(row.get("precision", 0.0) for row in rows),
        "mean_recall": _mean(row.get("recall", 0.0) for row in rows),
        "mean_mask_iou": _mean(row.get("mask_iou", 0.0) for row in rows),
        "avg_elapsed_ms": _mean(row.get("elapsed_ms", 0.0) for row in rows),
        "manual_accept_rate": (
            sum(1.0 if value else 0.0 for value in manual_accept_values) / len(manual_accept_values)
            if manual_accept_values
            else None
        ),
        "failure_reason_distribution": dict(failure_reasons),
    }


def write_localizer_benchmark_reports(
    results: list[dict[str, Any]],
    log_dir: str | Path,
    stem: str = "localizer_benchmark",
) -> dict[str, Path]:
    if not results:
        return {}

    target_dir = ensure_dir(log_dir)
    json_path = target_dir / f"{stem}.json"
    csv_path = target_dir / f"{stem}.csv"
    summary_path = target_dir / f"{stem}_summary.json"
    failure_path = target_dir / f"{stem}_failures.json"

    write_json(json_path, results)
    fieldnames = sorted({key for row in results for key in row.keys()})
    write_csv(csv_path, results, fieldnames)

    summary = summarize_localizer_results(results)
    aggregate_views = build_benchmark_aggregate_views(results)
    write_json(summary_path, summary)
    failures = {
        localizer: row["failure_reason_distribution"]
        for localizer, row in summary.items()
    }
    write_json(failure_path, failures)
    extra_failure_path: Path | None = None
    prefix = "localizer_benchmark_"
    if stem.startswith(prefix):
        suffix = stem[len(prefix):]
        extra_failure_path = target_dir / f"localizer_failure_summary_{suffix}.json"
        write_json(
            extra_failure_path,
            {
                "by_localizer": aggregate_views["by_localizer"],
                "by_anomaly_type": aggregate_views["by_anomaly_type"],
                "by_anomaly_type_and_localizer": aggregate_views["by_anomaly_type_and_localizer"],
                "failure_reason_distribution": {
                    "by_localizer": {
                        name: row["failure_reason_distribution"]
                        for name, row in aggregate_views["by_localizer"].items()
                    },
                    "by_anomaly_type": {
                        name: row["failure_reason_distribution"]
                        for name, row in aggregate_views["by_anomaly_type"].items()
                    },
                    "by_anomaly_type_and_localizer": {
                        anomaly_type: {
                            name: row["failure_reason_distribution"]
                            for name, row in localizer_rows.items()
                        }
                        for anomaly_type, localizer_rows in aggregate_views["by_anomaly_type_and_localizer"].items()
                    },
                },
            },
        )
    return {
        "results_json": json_path,
        "results_csv": csv_path,
        "summary_json": summary_path,
        "failures_json": failure_path,
        **({"failure_summary_json": extra_failure_path} if extra_failure_path is not None else {}),
    }
