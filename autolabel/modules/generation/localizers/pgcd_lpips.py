from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ....utils import ensure_dir
from .base import BaseLocalizer, LocalizationInput, LocalizationResult
from .components import clamp_bbox, extract_components
from .image_alignment import load_aligned_images
from .lpips_heatmap import SimpleHeatmapBackend, build_heatmap_backend
from .prompt_prior import DEFAULT_WEIGHTS, score_component, select_best_component


def _save_gray_image(array: np.ndarray, target: Path) -> str:
    ensure_dir(target.parent)
    clipped = np.clip(array, 0.0, 1.0)
    Image.fromarray((clipped * 255.0).astype(np.uint8), mode="L").save(target)
    return str(target)


def _save_binary_mask(mask: np.ndarray, target: Path) -> str:
    ensure_dir(target.parent)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(target)
    return str(target)


class PGCDLPIPSLocalizer(BaseLocalizer):
    method_name = "pgcd_lpips"

    def __init__(
        self,
        heatmap_backend: Any | None = None,
        min_component_area: int = 100,
        threshold_mode: str = "otsu",
        prompt_prior_weights: dict[str, float] | None = None,
        lpips_backbone: str = "alex",
        lpips_input_max_side: int = 768,
        heatmap_normalization: str = "percentile",
        heatmap_percentile_low: float = 1.0,
        heatmap_percentile_high: float = 99.0,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.heatmap_backend = heatmap_backend or config.get("heatmap_backend") or build_heatmap_backend(
            backbone=str(config.get("lpips_backbone", lpips_backbone)),
            input_max_side=int(config.get("lpips_input_max_side", lpips_input_max_side)),
            normalization=str(config.get("heatmap_normalization", heatmap_normalization)),
            percentile_low=float(config.get("heatmap_percentile_low", heatmap_percentile_low)),
            percentile_high=float(config.get("heatmap_percentile_high", heatmap_percentile_high)),
        )
        self.min_component_area = int(config.get("min_component_area", min_component_area))
        self.threshold_mode = str(config.get("threshold_mode", config.get("threshold", threshold_mode)))
        self.prompt_prior_weights = dict(DEFAULT_WEIGHTS)
        self.prompt_prior_weights.update(prompt_prior_weights or config.get("prompt_prior_weights") or {})

    def localize(self, inp: LocalizationInput | None = None, **kwargs: Any) -> LocalizationResult:
        data = self.coerce_input(inp, **kwargs)
        try:
            original, generated, alignment_metrics = load_aligned_images(data.original_image_path, data.generated_image_path)
            try:
                heatmap = self.heatmap_backend.compute(
                    original,
                    generated,
                    prompt_box=data.prompt_box,
                    anomaly_type=data.anomaly_type,
                ).astype(np.float32)
            except ModuleNotFoundError as exc:
                self.heatmap_backend = SimpleHeatmapBackend(reason=f"runtime_lpips_dependency_missing:{exc.name}")
                heatmap = self.heatmap_backend.compute(
                    original,
                    generated,
                    prompt_box=data.prompt_box,
                    anomaly_type=data.anomaly_type,
                ).astype(np.float32)
            if heatmap.ndim != 2:
                raise ValueError("heatmap backend must return a 2D array")

            max_value = float(np.max(heatmap)) if heatmap.size else 0.0
            if max_value <= 0.0:
                metrics = {
                    "localizer_method": self.method_name,
                    **alignment_metrics,
                    "pgcd_heatmap_max": 0.0,
                    "pgcd_heatmap_mean": 0.0,
                    "pgcd_heatmap_p95": 0.0,
                    "pgcd_heatmap_path": None,
                }
                return LocalizationResult(
                    False,
                    None,
                    None,
                    self.method_name,
                    metrics,
                    reason="empty_heatmap",
                )
            if max_value > 1.0:
                heatmap = heatmap / max_value

            height, width = heatmap.shape
            prompt_box = clamp_bbox(data.prompt_box, width, height)
            binary_mask, components, threshold = extract_components(
                heatmap,
                prompt_box=prompt_box,
                min_component_area=self.min_component_area,
                threshold_mode=self.threshold_mode,
                fixed_threshold=self.config.get("fixed_threshold"),
            )
            backend_info = self.heatmap_backend.describe() if hasattr(self.heatmap_backend, "describe") else {}
            metrics: dict[str, Any] = {
                "localizer_method": self.method_name,
                "pgcd_heatmap_max": float(heatmap.max()),
                "pgcd_heatmap_mean": float(heatmap.mean()),
                "pgcd_heatmap_p95": float(np.percentile(heatmap, 95)),
                "pgcd_heatmap_threshold": float(threshold),
                "pgcd_heatmap_path": None,
                "pgcd_selected_component_count": 0,
                **backend_info,
                **alignment_metrics,
            }
            debug_artifacts: dict[str, str] = {}
            if not components:
                if self.debug and data.debug_dir:
                    debug_dir = ensure_dir(data.debug_dir)
                    debug_artifacts["heatmap"] = _save_gray_image(heatmap, debug_dir / "pgcd_heatmap.png")
                    debug_artifacts["binary_mask"] = _save_binary_mask(binary_mask, debug_dir / "pgcd_binary_mask.png")
                    metrics["pgcd_heatmap_path"] = debug_artifacts["heatmap"]
                return LocalizationResult(False, None, None, self.method_name, metrics, reason="no_component", debug_artifacts=debug_artifacts)

            scored_components = [score_component(component, prompt_box, weights=self.prompt_prior_weights) for component in components]
            selected = select_best_component(components, prompt_box, weights=self.prompt_prior_weights)
            metrics.update(
                {
                    "pgcd_component_score": float(selected["component_score"]),
                    "pgcd_area_score": float(selected["area_score"]),
                    "pgcd_lpips_score": float(selected["lpips_score"]),
                    "pgcd_prompt_iou_score": float(selected["prompt_iou_score"]),
                    "pgcd_distance_penalty": float(selected["distance_penalty"]),
                    "pgcd_selected_component_count": 1,
                }
            )

            final_mask = np.zeros_like(binary_mask, dtype=bool)
            x1, y1, x2, y2 = [int(value) for value in selected["bbox"]]
            final_mask[y1:y2, x1:x2] = binary_mask[y1:y2, x1:x2]
            if not np.any(final_mask):
                final_mask[y1:y2, x1:x2] = True

            mask_path = _save_binary_mask(final_mask, data.mask_output_path)
            if self.debug and data.debug_dir:
                debug_dir = ensure_dir(data.debug_dir)
                debug_artifacts["heatmap"] = _save_gray_image(heatmap, debug_dir / "pgcd_heatmap.png")
                debug_artifacts["binary_mask"] = _save_binary_mask(binary_mask, debug_dir / "pgcd_binary_mask.png")
                metrics["pgcd_heatmap_path"] = debug_artifacts["heatmap"]
                components_path = debug_dir / "pgcd_components.json"
                components_path.write_text(json.dumps(scored_components, ensure_ascii=False, indent=2), encoding="utf-8")
                debug_artifacts["components"] = str(components_path)
                selected_path = debug_dir / "pgcd_selected_component.json"
                selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
                debug_artifacts["selected_component"] = str(selected_path)

            return LocalizationResult(
                success=True,
                final_bbox=[x1, y1, x2, y2],
                mask_path=mask_path,
                method=self.method_name,
                metrics=metrics,
                components=scored_components,
                selected_component_id=selected.get("component_id"),
                debug_artifacts=debug_artifacts,
            )
        except Exception as exc:
            return LocalizationResult(
                success=False,
                final_bbox=None,
                mask_path=None,
                method=self.method_name,
                metrics={
                    "localizer_method": self.method_name,
                    "localizer_error": type(exc).__name__,
                    "localizer_error_message": str(exc),
                },
                reason="localizer_error",
            )
