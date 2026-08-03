from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class SAM2Refiner:
    """Optional SAM2 integration point with coarse-mask fallback behavior."""

    def __init__(self, backend: Any | None = None, model: str = "tiny") -> None:
        self.model = model
        self.backend = backend or self._load_module_backend()
        self.available = self.backend is not None

    def _coerce_backend(self, candidate: Any) -> Any | None:
        if candidate is None:
            return None
        if hasattr(candidate, "refine_mask"):
            return candidate
        return None

    def _load_module_backend(self) -> Any | None:
        try:
            sam2_module = __import__("sam2")
        except ModuleNotFoundError:
            return None

        for attr_name in ("build_refiner", "load_refiner"):
            builder = getattr(sam2_module, attr_name, None)
            if callable(builder):
                backend = self._coerce_backend(builder(model=self.model))
                if backend is not None:
                    return backend

        refiner_cls = getattr(sam2_module, "SAM2Refiner", None)
        if callable(refiner_cls):
            backend = self._coerce_backend(refiner_cls(model=self.model))
            if backend is not None:
                return backend
        return None

    def refine_mask(
        self,
        image_path: str | Path,
        coarse_bbox: list[int],
        coarse_mask_path: str | Path | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        if self.backend is not None and hasattr(self.backend, "refine_mask"):
            return self.backend.refine_mask(image_path=image_path, coarse_bbox=coarse_bbox, coarse_mask_path=coarse_mask_path)
        if not self.available:
            raise ModuleNotFoundError("sam2 is not available")
        if coarse_mask_path is None:
            raise RuntimeError("coarse_mask_path is required when no explicit SAM2 backend is provided")

        with Image.open(coarse_mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L")) > 0
        return mask, [int(value) for value in coarse_bbox]
