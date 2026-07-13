from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def load_aligned_images(
    original_image_path: str | Path,
    generated_image_path: str | Path,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    with Image.open(original_image_path) as original_image:
        original = original_image.convert("RGB")
    with Image.open(generated_image_path) as generated_image:
        generated = generated_image.convert("RGB")

    metrics: dict[str, Any] = {
        "alignment_size_matched": original.size == generated.size,
        "alignment_resize_applied": False,
        "alignment_warning": None,
    }
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.BILINEAR)
        metrics["alignment_resize_applied"] = True
        metrics["alignment_warning"] = "generated_resized_to_original"
    return original, generated, metrics
