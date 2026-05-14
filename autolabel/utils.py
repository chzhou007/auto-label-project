from __future__ import annotations

import csv
import base64
import io
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


def image_to_data_url_with_dimensions(
    path: str | Path,
    max_side: int | None = None,
    jpeg_quality: int = 90,
    force_jpeg: bool = True,
) -> tuple[str, int, int, int, int]:
    source = Path(path)
    suffix = source.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime = mime_map.get(suffix, "image/jpeg")

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to encode images for VLM requests.") from exc

    with Image.open(source) as image:
        width, height = image.size
        request_width, request_height = width, height
        if max_side is not None and max_side > 0 and max(width, height) > max_side:
            scale = max_side / float(max(width, height))
            request_width = max(1, round(width * scale))
            request_height = max(1, round(height * scale))
            image = image.resize((request_width, request_height), Image.Resampling.LANCZOS)

        if force_jpeg or (request_width, request_height) != (width, height):
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=jpeg_quality)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}", width, height, request_width, request_height

    with open(source, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}", width, height, width, height


def image_to_data_url_with_size(
    path: str | Path,
    max_side: int | None = None,
    jpeg_quality: int = 90,
    force_jpeg: bool = True,
) -> tuple[str, int, int]:
    data_url, _, _, request_width, request_height = image_to_data_url_with_dimensions(
        path,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
        force_jpeg=force_jpeg,
    )
    return data_url, request_width, request_height


def image_to_data_url(path: str | Path, max_side: int | None = None, jpeg_quality: int = 90) -> str:
    data_url, _, _ = image_to_data_url_with_size(path, max_side=max_side, jpeg_quality=jpeg_quality)
    return data_url


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
