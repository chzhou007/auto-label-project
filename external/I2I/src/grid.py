from __future__ import annotations

import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import VALID_GRIDS


def _preview_max_side() -> int:
    raw = os.getenv("QWEN_GRID_PREVIEW_MAX_SIDE", "1280").strip()
    if not raw:
        return 1280
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("QWEN_GRID_PREVIEW_MAX_SIDE must be a positive integer") from exc
    if value <= 0:
        raise ValueError("QWEN_GRID_PREVIEW_MAX_SIDE must be a positive integer")
    return value


def _preview_quality() -> int:
    raw = os.getenv("QWEN_GRID_PREVIEW_JPEG_QUALITY", "80").strip()
    if not raw:
        return 80
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("QWEN_GRID_PREVIEW_JPEG_QUALITY must be an integer from 1 to 95") from exc
    return max(1, min(95, value))


def make_grid_preview(image_path: str, output_path: str) -> None:
    image = Image.open(image_path).convert("RGB")
    max_side = _preview_max_side()
    width, height = image.size
    scale = min(1.0, max_side / float(max(width, height)))
    if scale < 1.0:
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    line_width = max(2, min(width, height) // 300)
    font_size = max(24, min(width, height) // 18)
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for i in range(1, 4):
        x = round(width * i / 4)
        y = round(height * i / 4)
        draw.line([(x, 0), (x, height)], fill=(255, 0, 0), width=line_width)
        draw.line([(0, y), (width, y)], fill=(255, 0, 0), width=line_width)

    rows = "ABCD"
    for r in range(4):
        for c in range(4):
            label = f"{rows[r]}{c + 1}"
            cx = round((c + 0.5) * width / 4)
            cy = round((r + 0.5) * height / 4)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = max(4, font_size // 6)
            rect = [cx - tw // 2 - pad, cy - th // 2 - pad, cx + tw // 2 + pad, cy + th // 2 + pad]
            draw.rectangle(rect, fill=(255, 255, 255), outline=(255, 0, 0), width=max(1, line_width // 2))
            draw.text((cx - tw // 2, cy - th // 2), label, fill=(255, 0, 0), font=font)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=_preview_quality(), optimize=True)


def normalize_grid_id(grid_id: str) -> str:
    value = (grid_id or "").strip().upper()
    if re.fullmatch(r"[A-D][1-4]", value):
        return value
    return ""


def grid_id_to_bbox(grid_id: str, width: int, height: int) -> tuple[int, int, int, int]:
    grid = normalize_grid_id(grid_id)
    if grid not in VALID_GRIDS:
        raise ValueError(f"invalid grid_id: {grid_id}")
    row_index = "ABCD".index(grid[0])
    col_index = int(grid[1]) - 1
    x1 = int(col_index * width / 4)
    y1 = int(row_index * height / 4)
    x2 = int((col_index + 1) * width / 4)
    y2 = int((row_index + 1) * height / 4)
    return x1, y1, x2, y2


def expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    expanded_w = (x2 - x1) * ratio
    expanded_h = (y2 - y1) * ratio
    return (
        max(0, int(x1 - expanded_w)),
        max(0, int(y1 - expanded_h)),
        min(width, int(x2 + expanded_w)),
        min(height, int(y2 + expanded_h)),
    )


def bbox_to_box_dict(bbox: tuple[int, int, int, int]) -> dict[str, int | str]:
    x1, y1, x2, y2 = bbox
    return {"format": "xyxy", "x1": x1, "y1": y1, "x2": x2, "y2": y2}
