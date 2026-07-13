from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ....utils import ensure_dir
from .base import BaseLocalizer, LocalizationInput, LocalizationResult
from .components import binary_slice, clamp_bbox, extract_components
from .prompt_prior import select_best_component
from .image_alignment import load_aligned_images


def _save_mask(mask: np.ndarray, target: Path) -> str:
    ensure_dir(target.parent)
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    image.save(target)
    return str(target)


class RGBDiffLocalizer(BaseLocalizer):
    method_name = "rgb_diff"

    def __init__(self, roi_expand_pixels: int = 24, min_component_area: int = 80, **config: Any) -> None:
        super().__init__(**config)
        self.roi_expand_pixels = int(config.get("roi_expand_pixels", roi_expand_pixels))
        self.min_component_area = int(config.get("min_component_area", min_component_area))

    def localize(self, inp: LocalizationInput | None = None, **kwargs: Any) -> LocalizationResult:
        data = self.coerce_input(inp, **kwargs)
        original, generated, alignment_metrics = load_aligned_images(data.original_image_path, data.generated_image_path)

        original_arr = np.asarray(original, dtype=np.float32) / 255.0
        generated_arr = np.asarray(generated, dtype=np.float32) / 255.0
        diff_rgb = np.abs(generated_arr - original_arr)
        diff = 0.50 * diff_rgb.mean(axis=2) + 0.25 * np.max(diff_rgb, axis=2) + 0.25 * np.sqrt(np.mean(diff_rgb ** 2, axis=2))

        height, width = diff.shape
        prompt_box = clamp_bbox(data.prompt_box, width, height)
        x1, y1, x2, y2 = prompt_box
        expand = self.roi_expand_pixels
        roi = [
            max(0, x1 - expand),
            max(0, y1 - expand),
            min(width, x2 + expand),
            min(height, y2 + expand),
        ]
        rx1, ry1, rx2, ry2 = roi

        roi_diff = diff[ry1:ry2, rx1:rx2]
        prompt_diff = diff[y1:y2, x1:x2]
        change_ratio = float((prompt_diff > 0.04).mean()) if prompt_diff.size else 0.0
        global_change_ratio = float((diff > 0.04).mean()) if diff.size else 0.0
        metrics: dict[str, Any] = {
            "localizer_method": self.method_name,
            "change_ratio": change_ratio,
            "global_change_ratio": global_change_ratio,
            **alignment_metrics,
        }
        if float(roi_diff.max()) < 0.03:
            return LocalizationResult(False, None, None, self.method_name, metrics, reason="no_change")
        if global_change_ratio > float(self.config.get("max_global_change_ratio", 0.55)):
            return LocalizationResult(False, None, None, self.method_name, metrics, reason="global_change")

        masked_diff = np.zeros_like(diff, dtype=np.float32)
        masked_diff[ry1:ry2, rx1:rx2] = roi_diff
        binary_mask, components, threshold = extract_components(
            masked_diff,
            prompt_box=prompt_box,
            min_component_area=self.min_component_area,
            threshold_mode=str(self.config.get("threshold", "otsu")),
            fixed_threshold=self.config.get("fixed_threshold"),
        )
        metrics["rgb_diff_threshold"] = float(threshold)
        metrics["component_count"] = len(components)
        if not components:
            return LocalizationResult(False, None, None, self.method_name, metrics, reason="no_component")

        selected = select_best_component(components, prompt_box)
        selected_mask = np.zeros_like(binary_mask, dtype=bool)
        sy1, sx1 = selected["bbox"][1], selected["bbox"][0]
        sy2, sx2 = selected["bbox"][3], selected["bbox"][2]
        selected_mask[sy1:sy2, sx1:sx2] = binary_mask[sy1:sy2, sx1:sx2]
        if not np.any(selected_mask):
            region = binary_slice(selected["bbox"])
            selected_mask[region] = True
        mask_path = _save_mask(selected_mask, data.mask_output_path)
        metrics.update(
            {
                "selected_component_id": selected.get("component_id"),
                "selected_component_score": float(selected["component_score"]),
            }
        )
        return LocalizationResult(
            success=True,
            final_bbox=[int(value) for value in selected["bbox"]],
            mask_path=mask_path,
            method=self.method_name,
            metrics=metrics,
            components=components,
            selected_component_id=selected.get("component_id"),
        )
