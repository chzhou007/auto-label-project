from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def clamp_bbox(bbox: list[int] | tuple[int, int, int, int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1 = min(max(x1, 0), width)
    x2 = min(max(x2, 0), width)
    y1 = min(max(y1, 0), height)
    y2 = min(max(y2, 0), height)
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]


def bbox_iou(a: list[int] | tuple[int, int, int, int], b: list[int] | tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_center(bbox: list[int] | tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def otsu_threshold(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    clipped = np.clip(values.astype(np.float32), 0.0, 1.0)
    if float(clipped.max()) <= 0.0:
        return 0.0
    hist, bin_edges = np.histogram(clipped, bins=256, range=(0.0, 1.0))
    total = clipped.size
    sum_total = np.dot(hist, bin_edges[:-1])
    sum_background = 0.0
    weight_background = 0.0
    max_variance = -1.0
    threshold = float(clipped.mean())
    for index, count in enumerate(hist):
        weight_background += count
        if weight_background <= 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground <= 0:
            break
        sum_background += bin_edges[index] * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = float(bin_edges[index])
    return threshold


def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool)
    for _ in range(max(iterations, 0)):
        padded = np.pad(result, 1, mode="constant")
        neighbors = [
            padded[0:-2, 0:-2],
            padded[0:-2, 1:-1],
            padded[0:-2, 2:],
            padded[1:-1, 0:-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, 0:-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ]
        result = np.logical_or.reduce(neighbors)
    return result


def _binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool)
    for _ in range(max(iterations, 0)):
        padded = np.pad(result, 1, mode="constant")
        neighbors = [
            padded[0:-2, 0:-2],
            padded[0:-2, 1:-1],
            padded[0:-2, 2:],
            padded[1:-1, 0:-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, 0:-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ]
        result = np.logical_and.reduce(neighbors)
    return result


def clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    opened = _binary_dilate(_binary_erode(mask, iterations=1), iterations=1)
    closed = _binary_erode(_binary_dilate(opened, iterations=1), iterations=1)
    return closed.astype(bool)


def connected_components(mask: np.ndarray) -> list[dict[str, Any]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []
    component_id = 1
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            x1 = x2 = x
            y1 = y2 = y
            while queue:
                cx, cy = queue.popleft()
                pixels.append((cx, cy))
                x1 = min(x1, cx)
                y1 = min(y1, cy)
                x2 = max(x2, cx)
                y2 = max(y2, cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
            bbox = [x1, y1, x2 + 1, y2 + 1]
            box_width = max(1, bbox[2] - bbox[0])
            box_height = max(1, bbox[3] - bbox[1])
            components.append(
                {
                    "component_id": component_id,
                    "bbox": bbox,
                    "area": len(pixels),
                    "center": [bbox[0] + box_width / 2.0, bbox[1] + box_height / 2.0],
                    "aspect_ratio": box_width / float(box_height),
                    "pixels": pixels,
                }
            )
            component_id += 1
    return components


def extract_components(
    heatmap: np.ndarray,
    prompt_box: list[int] | tuple[int, int, int, int],
    min_component_area: int = 100,
    threshold_mode: str = "otsu",
    fixed_threshold: float | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a 2D array")
    height, width = heatmap.shape
    prompt = clamp_bbox(prompt_box, width, height)
    px1, py1, px2, py2 = prompt
    roi = heatmap[py1:py2, px1:px2]
    if roi.size == 0:
        return np.zeros_like(heatmap, dtype=bool), [], 0.0
    if threshold_mode == "fixed":
        threshold = float(fixed_threshold if fixed_threshold is not None else 0.5)
    else:
        threshold = otsu_threshold(roi if np.any(roi > 0) else heatmap)
    if threshold <= 0:
        positive = heatmap[heatmap > 0]
        if positive.size == 0:
            return np.zeros_like(heatmap, dtype=bool), [], 0.0
        threshold = float(max(np.percentile(positive, 60), positive.mean() * 0.5, 0.05))

    binary = heatmap >= threshold
    binary = clean_binary_mask(binary)
    raw_components = connected_components(binary)
    components: list[dict[str, Any]] = []
    for component in raw_components:
        if component["area"] < max(1, int(min_component_area)):
            continue
        component["lpips_score"] = float(heatmap[binary_slice(component["bbox"])].mean())
        components.append(component)
    return binary, components, float(threshold)


def binary_slice(bbox: list[int] | tuple[int, int, int, int]) -> tuple[slice, slice]:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return (slice(y1, y2), slice(x1, x2))
