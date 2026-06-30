from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops

from .grid import BBox, bbox_area, clip_bbox


@dataclass(frozen=True)
class LocalizationThresholds:
    min_bbox_area_ratio: float = 0.0001
    max_bbox_area_ratio: float = 0.08
    min_roi_change_ratio: float = 0.0005
    max_global_change_ratio: float = 0.20
    min_threshold: int = 18

    @classmethod
    def for_anomaly(cls, anomaly_type: str) -> "LocalizationThresholds":
        if anomaly_type == "oil_leak":
            return cls(min_bbox_area_ratio=0.00005, min_roi_change_ratio=0.0002, min_threshold=14)
        if anomaly_type in {"water_leak", "water_leakage"}:
            return cls(min_bbox_area_ratio=0.00002, min_roi_change_ratio=0.00012, min_threshold=7, max_bbox_area_ratio=0.10)
        if anomaly_type == "coolant_leak":
            return cls(min_bbox_area_ratio=0.00003, min_roi_change_ratio=0.00016, min_threshold=9, max_bbox_area_ratio=0.10)
        if anomaly_type == "diesel_leak":
            return cls(min_bbox_area_ratio=0.00003, min_roi_change_ratio=0.00018, min_threshold=10)
        return cls()


@dataclass(frozen=True)
class LocalizationResult:
    success: bool
    final_bbox: BBox | None
    mask_path: Path | None
    metrics: dict[str, Any]
    reason: str | None = None


def localize_difference(
    original_image_path: str | Path,
    generated_image_path: str | Path,
    roi_bbox: BBox,
    mask_output_path: str | Path,
    thresholds: LocalizationThresholds,
) -> LocalizationResult:
    with Image.open(original_image_path) as original_source:
        original = original_source.convert("RGB")
    with Image.open(generated_image_path) as generated_source:
        generated = generated_source.convert("RGB")
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)

    width, height = original.size
    roi = clip_bbox(roi_bbox, width, height)
    roi_area = bbox_area(roi)
    image_area = width * height
    if roi_area <= 0 or image_area <= 0:
        return LocalizationResult(False, None, None, {}, "invalid ROI")

    original_roi = original.crop(roi)
    generated_roi = generated.crop(roi)
    diff = ImageChops.difference(original_roi, generated_roi).convert("RGB")
    diff_values = _combined_diff_values(original_roi, generated_roi, diff)
    otsu = _otsu_threshold(diff_values)
    threshold_candidates = _threshold_ladder(otsu, thresholds.min_threshold)
    attempts: list[dict[str, float | int]] = []
    selected: dict[str, object] | None = None
    selected_threshold = threshold_candidates[0]
    selected_ratio = 0.0
    selected_global_ratio = 0.0
    selected_component_count = 0
    min_bbox_area = thresholds.min_bbox_area_ratio * image_area
    max_bbox_area = thresholds.max_bbox_area_ratio * image_area

    for threshold in threshold_candidates:
        binary = _values_to_binary(diff_values, diff.width, diff.height, threshold)
        binary = _open(_close(binary))
        binary = _dilate(binary)
        components = _connected_components(binary, diff_values, diff.width, diff.height)
        changed_pixels = sum(1 for value in diff_values if value >= threshold)
        roi_change_ratio = changed_pixels / roi_area
        global_change_ratio = changed_pixels / image_area
        attempts.append(
            {
                "threshold": int(threshold),
                "roi_change_ratio": float(roi_change_ratio),
                "global_change_ratio": float(global_change_ratio),
                "component_count": int(len(components)),
            }
        )
        if roi_change_ratio < thresholds.min_roi_change_ratio or global_change_ratio > thresholds.max_global_change_ratio:
            continue
        valid_components = []
        for component in components:
            local_bbox = component["bbox"]
            full_bbox = (
                roi[0] + local_bbox[0],
                roi[1] + local_bbox[1],
                roi[0] + local_bbox[2],
                roi[1] + local_bbox[3],
            )
            full_area = bbox_area(full_bbox)
            if min_bbox_area <= full_area <= max_bbox_area:
                valid_components.append({**component, "full_bbox": full_bbox, "full_bbox_area": full_area})
        if not valid_components:
            continue
        selected = _merge_components_if_reasonable(valid_components, image_area, max_bbox_area)
        selected_threshold = threshold
        selected_ratio = roi_change_ratio
        selected_global_ratio = global_change_ratio
        selected_component_count = len(components)
        break

    if selected is None:
        reason = "No valid connected component found in ROI"
        if attempts and all(float(item["roi_change_ratio"]) < thresholds.min_roi_change_ratio for item in attempts):
            reason = "ROI change ratio is too low"
        if attempts and any(float(item["global_change_ratio"]) > thresholds.max_global_change_ratio for item in attempts):
            reason = "Global change ratio is too high"
        return LocalizationResult(False, None, None, {"threshold_attempts": attempts, "otsu_threshold": int(otsu)}, reason)

    final_bbox = clip_bbox(selected["full_bbox"], width, height)
    selected_points = selected["points"]
    mask_output = Path(mask_output_path)
    mask_output.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (width, height), 0)
    mask_pixels = mask.load()
    for x, y in selected_points:
        mask_pixels[roi[0] + x, roi[1] + y] = 255
    mask.save(mask_output)

    mean_diff = selected["diff_sum"] / max(1, selected["area"])
    metrics: dict[str, float | int | str | list[int]] = {
        "threshold": int(selected_threshold),
        "threshold_attempts": attempts,  # type: ignore[dict-item]
        "otsu_threshold": int(otsu),
        "roi_change_ratio": float(selected_ratio),
        "global_change_ratio": float(selected_global_ratio),
        "component_count": int(selected_component_count),
        "selected_component_area": int(selected["area"]),
        "bbox_area_ratio": float(bbox_area(final_bbox) / image_area),
        "mean_diff_in_mask": float(mean_diff),
        "roi_bbox": list(map(int, roi)),
        "final_bbox": list(map(int, final_bbox)),
    }
    return LocalizationResult(True, final_bbox, mask_output, metrics)


def _otsu_threshold(values: Iterable[int]) -> int:
    hist = [0] * 256
    total = 0
    for value in values:
        hist[int(max(0, min(255, value)))] += 1
        total += 1
    if total == 0:
        return 0
    sum_total = sum(index * count for index, count in enumerate(hist))
    sum_background = 0.0
    weight_background = 0
    max_variance = -1.0
    threshold = 0
    for index, count in enumerate(hist):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += index * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = index
    return threshold


def _values_to_binary(values: list[int], width: int, height: int, threshold: int) -> list[list[bool]]:
    return [[values[y * width + x] >= threshold for x in range(width)] for y in range(height)]


def _dilate(binary: list[list[bool]]) -> list[list[bool]]:
    height = len(binary)
    width = len(binary[0]) if height else 0
    output = [[False] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            if not binary[y][x]:
                continue
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    output[ny][nx] = True
    return output


def _erode(binary: list[list[bool]]) -> list[list[bool]]:
    height = len(binary)
    width = len(binary[0]) if height else 0
    output = [[False] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            keep = True
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if not binary[ny][nx]:
                        keep = False
                        break
                if not keep:
                    break
            output[y][x] = keep
    return output


def _close(binary: list[list[bool]]) -> list[list[bool]]:
    return _erode(_dilate(binary))


def _open(binary: list[list[bool]]) -> list[list[bool]]:
    return _dilate(_erode(binary))


def _threshold_ladder(otsu_threshold: int, min_threshold: int) -> list[int]:
    candidates = [
        max(min_threshold, otsu_threshold),
        max(min_threshold, int(otsu_threshold * 0.78)),
        max(min_threshold, int(otsu_threshold * 0.58)),
        min_threshold,
        max(4, min_threshold - 3),
    ]
    result: list[int] = []
    for value in candidates:
        value = int(max(1, min(255, value)))
        if value not in result:
            result.append(value)
    return result


def _combined_diff_values(original_roi: Image.Image, generated_roi: Image.Image, rgb_diff: Image.Image) -> list[int]:
    rgb_bytes = rgb_diff.tobytes()
    lum_diff = ImageChops.difference(original_roi.convert("L"), generated_roi.convert("L")).tobytes()
    sat_diff = ImageChops.difference(original_roi.convert("HSV").split()[1], generated_roi.convert("HSV").split()[1]).tobytes()
    values: list[int] = []
    for idx in range(0, len(rgb_bytes), 3):
        pixel_index = idx // 3
        rgb_value = max(rgb_bytes[idx], rgb_bytes[idx + 1], rgb_bytes[idx + 2])
        lum_value = lum_diff[pixel_index]
        sat_value = sat_diff[pixel_index]
        # Transparent water often appears as a luminance/highlight shift with weak chroma change.
        values.append(max(rgb_value, int(lum_value * 1.20), int(sat_value * 0.80)))
    return values


def _connected_components(binary: list[list[bool]], diff_values: list[int], width: int, height: int) -> list[dict[str, object]]:
    visited = [[False] * width for _ in range(height)]
    components: list[dict[str, object]] = []
    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y][start_x] or not binary[start_y][start_x]:
                continue
            stack = [(start_x, start_y)]
            visited[start_y][start_x] = True
            points: list[tuple[int, int]] = []
            x_min = x_max = start_x
            y_min = y_max = start_y
            diff_sum = 0
            while stack:
                x, y = stack.pop()
                points.append((x, y))
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)
                diff_sum += diff_values[y * width + x]
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        if visited[ny][nx] or not binary[ny][nx]:
                            continue
                        visited[ny][nx] = True
                        stack.append((nx, ny))
            components.append(
                {
                    "points": points,
                    "area": len(points),
                    "bbox": (x_min, y_min, x_max + 1, y_max + 1),
                    "diff_sum": diff_sum,
                }
            )
    return sorted(components, key=lambda item: int(item["area"]), reverse=True)


def _merge_components_if_reasonable(components: list[dict[str, object]], image_area: int, max_bbox_area: float) -> dict[str, object]:
    top = components[:5]
    x1 = min(int(item["full_bbox"][0]) for item in top)  # type: ignore[index]
    y1 = min(int(item["full_bbox"][1]) for item in top)  # type: ignore[index]
    x2 = max(int(item["full_bbox"][2]) for item in top)  # type: ignore[index]
    y2 = max(int(item["full_bbox"][3]) for item in top)  # type: ignore[index]
    merged_area = max(0, x2 - x1) * max(0, y2 - y1)
    if len(top) > 1 and merged_area <= max_bbox_area:
        points: list[tuple[int, int]] = []
        diff_sum = 0
        area = 0
        for item in top:
            points.extend(item["points"])  # type: ignore[arg-type]
            diff_sum += int(item["diff_sum"])
            area += int(item["area"])
        return {"points": points, "diff_sum": diff_sum, "area": area, "full_bbox": (x1, y1, x2, y2)}
    return components[0]
