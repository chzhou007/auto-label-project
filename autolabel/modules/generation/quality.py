from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops

from .grid import BBox, clip_bbox


def background_preservation_score(original_image_path: str | Path, generated_image_path: str | Path, roi_bbox: BBox) -> float:
    with Image.open(original_image_path) as original_source:
        original = original_source.convert("RGB")
    with Image.open(generated_image_path) as generated_source:
        generated = generated_source.convert("RGB")
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)
    width, height = original.size
    roi = clip_bbox(roi_bbox, width, height)
    diff = ImageChops.difference(original, generated)
    pixels = diff.load()
    step = max(1, int((width * height / 250000) ** 0.5))
    total = 0.0
    count = 0
    for y in range(0, height, step):
        for x in range(0, width, step):
            if roi[0] <= x < roi[2] and roi[1] <= y < roi[3]:
                continue
            r, g, b = pixels[x, y]
            total += (r + g + b) / 3.0
            count += 1
    if count == 0:
        return 1.0
    mean_abs = total / count
    return max(0.0, min(1.0, 1.0 - mean_abs / 50.0))


def anomaly_visibility_score(localization_metrics: dict[str, float | int | str | list[int]], review_score: float | None = None) -> float:
    roi_change_ratio = float(localization_metrics.get("roi_change_ratio", 0.0))
    bbox_area_ratio = float(localization_metrics.get("bbox_area_ratio", 0.0))
    mean_diff = float(localization_metrics.get("mean_diff_in_mask", 0.0))
    component_area = float(localization_metrics.get("selected_component_area", 0.0))
    roi_score = min(1.0, roi_change_ratio / 0.035)
    bbox_score = min(1.0, bbox_area_ratio / 0.012)
    diff_score = min(1.0, mean_diff / 55.0)
    component_score = 1.0 if component_area > 0 else 0.0
    score = 0.35 * roi_score + 0.25 * bbox_score + 0.30 * diff_score + 0.10 * component_score
    if review_score is not None:
        score = 0.85 * score + 0.15 * max(0.0, min(1.0, review_score))
    return max(0.0, min(1.0, score))


def passes_quality(
    background_score: float,
    visibility_score: float,
    min_background_score: float = 0.78,
    min_visibility_score: float = 0.12,
) -> tuple[bool, str | None]:
    if background_score < min_background_score:
        return False, f"background_changed_too_much: {background_score:.3f} < {min_background_score:.3f}"
    if visibility_score < min_visibility_score:
        return False, f"no_visible_change: {visibility_score:.3f} < {min_visibility_score:.3f}"
    return True, None
