from __future__ import annotations

import numpy as np
from PIL import Image

from typing import Any

from .base import LocalizationInput, LocalizationResult
from .pgcd_lpips import PGCDLPIPSLocalizer
from ..sam2_refiner import SAM2Refiner


class PGCDLPIPSSAM2Localizer(PGCDLPIPSLocalizer):
    method_name = "pgcd_lpips_sam2"

    def __init__(self, sam2_required: bool = False, sam2_model: str = "tiny", sam2_backend: Any | None = None, **config: Any) -> None:
        self.sam2_required = sam2_required
        self.sam2_model = sam2_model
        self.refiner = SAM2Refiner(backend=sam2_backend, model=sam2_model)
        self.sam2_available = self.refiner.available
        if not self.sam2_available and sam2_required:
            raise ModuleNotFoundError("sam2 is required but not available")
        super().__init__(**config)

    @staticmethod
    def _bbox_area(bbox: list[int] | tuple[int, int, int, int] | None) -> float:
        if bbox is None:
            return 0.0
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def localize(self, inp: LocalizationInput | None = None, **kwargs: Any) -> LocalizationResult:
        data = self.coerce_input(inp, **kwargs)
        result = super().localize(data)
        result.metrics = dict(result.metrics)
        result.metrics.setdefault("sam2_enabled", self.sam2_available)
        result.metrics.setdefault("sam2_model", self.sam2_model)
        result.metrics.setdefault("sam2_success", False)
        result.metrics.setdefault("sam2_refined_area_ratio", 0.0)
        result.metrics.setdefault("sam2_bbox_delta_ratio", 0.0)
        if self.sam2_available and result.success and result.mask_path and result.final_bbox:
            try:
                coarse_bbox = [int(value) for value in result.final_bbox]
                coarse_area = self._bbox_area(coarse_bbox)
                refined_mask, refined_bbox = self.refiner.refine_mask(
                    image_path=str(data.generated_image_path),
                    coarse_bbox=coarse_bbox,
                    coarse_mask_path=result.mask_path,
                )
                Image.fromarray(np.where(refined_mask, 255, 0).astype(np.uint8), mode="L").save(result.mask_path)
                result.final_bbox = [int(value) for value in refined_bbox]
                result.method = self.method_name
                refined_area = self._bbox_area(refined_bbox)
                result.metrics["sam2_enabled"] = True
                result.metrics["sam2_success"] = True
                result.metrics["sam2_refined"] = True
                result.metrics["sam2_refined_area_ratio"] = refined_area / coarse_area if coarse_area > 0 else 0.0
                result.metrics["sam2_bbox_delta_ratio"] = (
                    abs(refined_area - coarse_area) / coarse_area if coarse_area > 0 else 0.0
                )
                return result
            except Exception as exc:
                result.metrics["sam2_enabled"] = True
                result.metrics["sam2_success"] = False
                result.metrics["sam2_refined"] = False
                result.metrics["sam2_error"] = str(exc)
                result.method = f"{self.method_name}_fallback_coarse"
                return result
        result.metrics["sam2_enabled"] = self.sam2_available
        result.metrics["sam2_success"] = False
        result.metrics["sam2_refined"] = False
        if result.success:
            result.method = f"{self.method_name}_fallback_coarse"
        return result
