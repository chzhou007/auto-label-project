from __future__ import annotations

import base64
import csv
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from PIL import Image


def ensure_output_dirs(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    dirs = {
        "root": root,
        "generated_images": root / "generated_images",
        "grid_previews": root / "grid_previews",
        "crops": root / "crops",
        "masks": root / "masks",
        "metadata": root / "metadata",
        "logs": root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(log_dir: str | Path) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(log_dir) / "pipeline.log", encoding="utf-8"),
        ],
    )


def read_tasks_csv(path: str | Path) -> list[dict[str, str]]:
    required = {"sample_id", "image_id", "image_uri", "anomaly_type", "source_type"}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"tasks.csv has no header: {path}")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"tasks.csv missing required columns: {missing}")
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def resolve_image_path(task: dict[str, str], image_root: str | Path) -> Path:
    image_uri = task.get("image_uri", "")
    candidates = [Path(image_uri)]
    if image_uri:
        candidates.append(Path(image_root) / Path(image_uri).name)
    candidates.append(Path(image_root) / task.get("image_id", ""))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"image not found for sample_id={task.get('sample_id')}: {image_uri}")


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def write_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def image_to_data_url(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def now_iso_shanghai() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def relative_uri(path: str | Path) -> str:
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted = dict(headers)
    if "Authorization" in redacted:
        redacted["Authorization"] = "Bearer ***"
    return redacted
