from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

SUPPORTED_ANOMALY_TYPES = {"diesel_leak", "oil_leak", "coolant_leak", "water_leak", "water_leakage"}
ANOMALY_TYPE_ALIASES = {"water_leakage": "water_leak"}
SUPPORTED_SEVERITY_LEVELS = {"early", "moderate", "obvious_but_controlled"}


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    generated_images: Path
    grid_previews: Path
    crops: Path
    masks: Path
    metadata: Path
    labelstudio: Path
    logs: Path
    api_responses: Path
    transient_root: Path


def prepare_output_paths(output_root: str | Path) -> OutputPaths:
    root = Path(output_root)
    transient_root = Path(tempfile.mkdtemp(prefix="autolabel_generation_"))
    paths = OutputPaths(
        root=root,
        crops=root / "crops",
        masks=root / "masks",
        metadata=root / "metadata",
        generated_images=root / "metadata" / "images",
        grid_previews=transient_root / "grid_previews",
        labelstudio=root / "metadata",
        logs=root / "metadata" / "logs",
        api_responses=transient_root / "api_responses",
        transient_root=transient_root,
    )
    for path in (
        paths.crops,
        paths.masks,
        paths.metadata,
        paths.generated_images,
        paths.grid_previews,
        paths.labelstudio,
        paths.logs,
        paths.api_responses,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
    return paths


def parse_grid_layout(grid_layout: str) -> tuple[int, int]:
    value = grid_layout.lower().strip()
    if "x" not in value:
        raise ValueError(f"grid_layout must look like 4x4, got: {grid_layout!r}")
    rows_text, cols_text = value.split("x", 1)
    rows, cols = int(rows_text), int(cols_text)
    if rows < 1 or cols < 1 or rows > 26 or cols > 99:
        raise ValueError(f"Unsupported grid_layout: {grid_layout!r}")
    return rows, cols


def resolve_task_image_path(row: dict[str, Any], image_root: str | Path) -> Path:
    image_value = row.get("image_path") or row.get("image_uri")
    if not image_value:
        raise ValueError("task row is missing required image_path")
    image_path = Path(str(image_value))
    if image_path.is_absolute():
        return image_path
    if image_path.exists():
        return image_path
    return Path(image_root) / image_path


def normalize_task_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    normalized = {key: (value.strip() if isinstance(value, str) else value) for key, value in row.items()}
    task_id = normalized.get("task_id") or normalized.get("sample_id") or f"task_{index:06d}"
    normalized["task_id"] = str(task_id)
    normalized.setdefault("sample_id", f"sample_{task_id}")
    normalized.setdefault("image_id", f"image_{task_id}")
    normalized["source_type"] = normalized.get("source_type") or "generated"
    normalized["severity_level"] = normalized.get("severity_level") or "early"
    normalized["room_type"] = normalized.get("room_type") or "generator_room"
    if normalized["severity_level"] not in SUPPORTED_SEVERITY_LEVELS:
        raise ValueError(
            f"Unsupported severity_level for {task_id}: {normalized['severity_level']}. "
            f"Allowed: {sorted(SUPPORTED_SEVERITY_LEVELS)}"
        )
    anomaly_type = normalized.get("anomaly_type")
    if anomaly_type not in SUPPORTED_ANOMALY_TYPES:
        raise ValueError(
            f"Unsupported anomaly_type for {task_id}: {anomaly_type}. "
            f"Allowed: {sorted(SUPPORTED_ANOMALY_TYPES)}"
        )
    normalized["anomaly_type"] = ANOMALY_TYPE_ALIASES.get(str(anomaly_type), str(anomaly_type))
    return normalized


def positive_float(value: Any, name: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")
    return parsed


def positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return parsed
