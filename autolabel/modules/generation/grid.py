from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .config import parse_grid_layout

BBox = tuple[int, int, int, int]


def grid_ids(grid_layout: str = "4x4") -> list[str]:
    rows, cols = parse_grid_layout(grid_layout)
    return [f"{chr(ord('A') + r)}{c + 1}" for r in range(rows) for c in range(cols)]


def validate_grid_id(grid_id: str, grid_layout: str = "4x4") -> None:
    if grid_id not in grid_ids(grid_layout):
        raise ValueError(f"Invalid grid id {grid_id!r}; expected one of {grid_ids(grid_layout)}")


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def make_grid_preview(image_path: str | Path, output_path: str | Path, grid_layout: str = "4x4") -> Path:
    rows, cols = parse_grid_layout(grid_layout)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    line_width = max(2, round(min(width, height) / 300))
    label_font = _font(max(14, round(min(width, height) / 24)))

    for col in range(cols + 1):
        x = round(col * width / cols)
        draw.line([(x, 0), (x, height)], fill=(255, 230, 0, 230), width=line_width)
    for row in range(rows + 1):
        y = round(row * height / rows)
        draw.line([(0, y), (width, y)], fill=(255, 230, 0, 230), width=line_width)

    for row in range(rows):
        for col in range(cols):
            label = f"{chr(ord('A') + row)}{col + 1}"
            x1, y1, x2, y2 = grid_id_to_bbox(label, width, height, grid_layout)
            tx, ty = x1 + 8, y1 + 8
            try:
                text_bbox = draw.textbbox((tx, ty), label, font=label_font)
                pad = 5
                draw.rectangle(
                    (text_bbox[0] - pad, text_bbox[1] - pad, text_bbox[2] + pad, text_bbox[3] + pad),
                    fill=(0, 0, 0, 145),
                )
            except Exception:
                pass
            draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=label_font)

    image.save(output, quality=95)
    return output


def grid_id_to_bbox(grid_id: str, image_width: int, image_height: int, grid_layout: str = "4x4") -> BBox:
    validate_grid_id(grid_id, grid_layout)
    rows, cols = parse_grid_layout(grid_layout)
    row_index = ord(grid_id[0].upper()) - ord("A")
    col_index = int(grid_id[1:]) - 1
    x1 = round(col_index * image_width / cols)
    y1 = round(row_index * image_height / rows)
    x2 = round((col_index + 1) * image_width / cols)
    y2 = round((row_index + 1) * image_height / rows)
    return clip_bbox((x1, y1, x2, y2), image_width, image_height)


def fine_grid_id_to_bbox(fine_grid_id: str, coarse_bbox: BBox, local_layout: str = "3x3") -> BBox:
    validate_grid_id(fine_grid_id, local_layout)
    x1, y1, x2, y2 = coarse_bbox
    local_width = x2 - x1
    local_height = y2 - y1
    fx1, fy1, fx2, fy2 = grid_id_to_bbox(fine_grid_id, local_width, local_height, local_layout)
    return (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)


def make_fine_grid_preview(
    image_path: str | Path,
    coarse_bbox: BBox,
    output_path: str | Path,
    local_layout: str = "3x3",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        crop = source.convert("RGB").crop(coarse_bbox)
    temp_path = output.with_suffix(".tmp.jpg")
    crop.save(temp_path, quality=95)
    make_grid_preview(temp_path, output, local_layout)
    temp_path.unlink(missing_ok=True)
    return output


def clip_bbox(bbox: Iterable[int | float], image_width: int, image_height: int) -> BBox:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(x1 + 1, min(image_width, x2))
    y2 = max(y1 + 1, min(image_height, y2))
    return x1, y1, x2, y2


def expand_bbox(bbox: BBox, image_width: int, image_height: int, ratio: float = 0.20) -> BBox:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    pad_x = width * ratio
    pad_y = height * ratio
    return clip_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), image_width, image_height)


def bbox_to_dict(bbox: BBox) -> dict[str, int | str]:
    x1, y1, x2, y2 = bbox
    return {"format": "xyxy", "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}


def bbox_area(bbox: BBox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_intersection_area(first: BBox, second: BBox) -> int:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def grid_ids_overlapping_bboxes(
    image_width: int,
    image_height: int,
    grid_layout: str,
    bboxes: Iterable[BBox],
    min_overlap_ratio: float = 0.0,
) -> list[str]:
    blocked: list[str] = []
    for grid in grid_ids(grid_layout):
        grid_bbox = grid_id_to_bbox(grid, image_width, image_height, grid_layout)
        grid_area = max(1, bbox_area(grid_bbox))
        for bbox in bboxes:
            overlap = bbox_intersection_area(grid_bbox, bbox)
            if overlap > 0 and (overlap / grid_area) >= min_overlap_ratio:
                blocked.append(grid)
                break
    return blocked
