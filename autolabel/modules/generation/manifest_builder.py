from __future__ import annotations

from pathlib import Path
from typing import Any

from ...utils import read_csv, write_csv


GENERATION_REQUIRED_FIELDS = [
    "sample_id",
    "image_id",
    "image_uri",
    "source_type",
    "task_mode",
    "task_key",
    "object_type",
    "anomaly_type",
]


def _fieldnames(input_rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in input_rows:
        for name in row.keys():
            if name not in names:
                names.append(name)
    for name in GENERATION_REQUIRED_FIELDS:
        if name not in names:
            names.append(name)
    return names


def build_water_leak_generation_rows(
    input_rows: list[dict[str, Any]],
    count: int = 1000,
    anomaly_type: str = "water_leak",
    object_type: str = "leakage_area",
    sample_prefix: str = "water_leak",
    allow_repeat: bool = False,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not input_rows:
        raise ValueError("input manifest has no rows")
    if count > len(input_rows) and not allow_repeat:
        raise ValueError(f"count {count} exceeds input rows {len(input_rows)}; use allow_repeat to cycle rows")

    rows: list[dict[str, Any]] = []
    for index in range(count):
        source = input_rows[index % len(input_rows)]
        original_sample_id = source.get("sample_id") or f"sample_{index + 1:06d}"
        original_image_id = source.get("image_id") or f"image_{index + 1:06d}"
        row = dict(source)
        row["sample_id"] = f"{sample_prefix}_{index + 1:04d}_{original_sample_id}"
        row["image_id"] = f"{sample_prefix}_{index + 1:04d}_{original_image_id}"
        row["source_type"] = row.get("source_type") or "manual_upload"
        row["task_mode"] = "generation"
        row["task_key"] = row.get("task_key") or ""
        row["object_type"] = object_type
        row["anomaly_type"] = anomaly_type
        rows.append(row)
    return rows


def write_water_leak_generation_manifest(
    input_path: str | Path,
    output_path: str | Path,
    count: int = 1000,
    anomaly_type: str = "water_leak",
    object_type: str = "leakage_area",
    sample_prefix: str = "water_leak",
    allow_repeat: bool = False,
) -> Path:
    input_rows = read_csv(input_path)
    rows = build_water_leak_generation_rows(
        input_rows,
        count=count,
        anomaly_type=anomaly_type,
        object_type=object_type,
        sample_prefix=sample_prefix,
        allow_repeat=allow_repeat,
    )
    write_csv(output_path, rows, _fieldnames(input_rows))
    return Path(output_path)
