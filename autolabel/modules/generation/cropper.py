from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .grid import BBox, bbox_to_dict, expand_bbox


def crop_with_expand(image_path: str | Path, bbox: BBox, output_path: str | Path, expand_ratio: float = 0.10) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        crop_bbox = expand_bbox(bbox, width, height, expand_ratio)
        image.crop(crop_bbox).save(output, quality=95)
    return {
        "crop_uri": str(output),
        "crop_box": bbox_to_dict(crop_bbox),
        "crop_expand_ratio": float(expand_ratio),
        "is_valid_crop": True,
    }
