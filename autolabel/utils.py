from __future__ import annotations

import csv
import base64
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


SHANGHAI_TZ = timezone(timedelta(hours=8))


def now_iso_shanghai() -> str:
    return datetime.now(SHANGHAI_TZ).isoformat(timespec="seconds")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def slugify(value: str, fallback: str = "item") -> str:
    value = re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("._-")
    return value or fallback


def get_image_size(path: str | Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to read image dimensions.") from exc
    with Image.open(path) as image:
        return image.size


def image_to_data_url(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime = mime_map.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or base_dir is None:
        return candidate
    return Path(base_dir) / candidate


def maybe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def relpath_if_possible(path: str | Path, base_dir: str | Path | None = None) -> str:
    target = Path(path)
    if base_dir is None:
        return str(target)
    try:
        return str(target.relative_to(Path(base_dir)))
    except ValueError:
        return str(target)
