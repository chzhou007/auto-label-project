from __future__ import annotations

from pathlib import Path

from PIL import Image

from grid import bbox_to_box_dict, expand_bbox


def crop_anomaly(
    image_path: str,
    final_bbox: dict,
    output_path: str,
    expand_ratio: float,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    bbox_tuple = (
        int(final_bbox["x1"]),
        int(final_bbox["y1"]),
        int(final_bbox["x2"]),
        int(final_bbox["y2"]),
    )
    crop_box_tuple = expand_bbox(bbox_tuple, width, height, expand_ratio)
    crop = image.crop(crop_box_tuple)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, quality=95)
    return {
        "crop_id": "crop_000001",
        "crop_uri": output_path,
        "crop_box": bbox_to_box_dict(crop_box_tuple),
        "crop_expand_ratio": expand_ratio,
        "is_valid_crop": True,
    }
