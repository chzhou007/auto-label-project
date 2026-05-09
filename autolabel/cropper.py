from __future__ import annotations

from pathlib import Path
from typing import Any

from .sample_factory import make_box

CROP_FILE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cleanup_sample_crops(crop_dir: str | Path, sample_id: str) -> None:
    target_dir = Path(crop_dir)
    if not target_dir.exists():
        return
    prefix = f"{sample_id}_"
    for path in target_dir.iterdir():
        if path.is_file() and path.name.startswith(prefix) and path.suffix.lower() in CROP_FILE_SUFFIXES:
            path.unlink()


def expand_box(box: dict[str, Any], width: int, height: int, ratio: float) -> dict[str, int | str]:
    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(round(box_w * ratio))
    pad_y = int(round(box_h * ratio))
    return make_box(
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def crop_image_box(
    image_path: str | Path,
    box: dict[str, Any],
    output_path: str | Path,
    expand_ratio: float = 0.0,
) -> dict[str, Any]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to crop images.") from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        width, height = image.size
        crop_box = expand_box(box, width, height, expand_ratio)
        xyxy = (
            int(crop_box["x1"]),
            int(crop_box["y1"]),
            int(crop_box["x2"]),
            int(crop_box["y2"]),
        )
        image.crop(xyxy).convert("RGB").save(output, quality=95)

    return {
        "crop_uri": str(output),
        "crop_box": crop_box,
        "crop_expand_ratio": expand_ratio,
        "is_valid_crop": True,
    }


def attach_crops(sample: dict[str, Any], crop_dir: str | Path, expand_ratio: float = 0.0) -> None:
    image_uri = sample["image_asset"]["image_uri"]
    target_dir = Path(crop_dir)
    for obj in sample.get("objects", []):
        crop_id = f"{sample['sample_id']}_{obj['object_id']}"
        crop_path = target_dir / f"{crop_id}.jpg"
        crop_info = crop_image_box(image_uri, obj["box"], crop_path, expand_ratio)
        obj["crop"] = {
            "crop_id": crop_id,
            **crop_info,
        }
