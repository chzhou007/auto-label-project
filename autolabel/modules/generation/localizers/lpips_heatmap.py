from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def normalize_heatmap(
    heatmap: np.ndarray,
    mode: str = "percentile",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    array = np.asarray(heatmap, dtype=np.float32)
    if array.size == 0:
        return array.astype(np.float32)

    normalized_mode = (mode or "percentile").strip().lower()
    if normalized_mode == "none":
        clipped = np.clip(array, 0.0, None)
        max_value = float(clipped.max())
        if max_value <= 0:
            return np.zeros_like(clipped, dtype=np.float32)
        return (clipped / max_value).astype(np.float32)

    if normalized_mode == "minmax":
        low = float(array.min())
        high = float(array.max())
    else:
        low = float(np.percentile(array, percentile_low))
        high = float(np.percentile(array, percentile_high))

    if high - low <= 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _resize_for_model(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_side:
        return image
    scale = max_side / float(longest_side)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size, Image.BICUBIC)


def _resize_heatmap(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if array.shape == (size[1], size[0]):
        return array.astype(np.float32)
    resized = Image.fromarray(array.astype(np.float32), mode="F").resize(size, Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


class RGBDiffHeatmapBackend:
    backend_name = "rgb_diff_fallback"

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason

    def compute(self, original: Image.Image, generated: Image.Image, **kwargs: Any) -> np.ndarray:
        original_arr = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
        generated_arr = np.asarray(generated.convert("RGB"), dtype=np.float32) / 255.0
        diff = np.abs(generated_arr - original_arr)
        heatmap = 0.60 * diff.mean(axis=2) + 0.40 * np.max(diff, axis=2)
        max_value = float(heatmap.max())
        if max_value > 0:
            heatmap = heatmap / max_value
        return heatmap.astype(np.float32)

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"pgcd_heatmap_backend": self.backend_name}
        if self.reason:
            payload["pgcd_heatmap_backend_reason"] = self.reason
        return payload


class SimpleHeatmapBackend(RGBDiffHeatmapBackend):
    pass


class LPIPSHeatmapBackend:
    backend_name = "lpips"

    def __init__(
        self,
        backbone: str = "alex",
        input_max_side: int = 768,
        normalization: str = "percentile",
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        prefer_gpu: bool = True,
        model: Any | None = None,
        torch_module: Any | None = None,
        lpips_module: Any | None = None,
    ) -> None:
        self.backbone = backbone
        self.input_max_side = int(input_max_side)
        self.normalization = normalization
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)
        self.prefer_gpu = prefer_gpu
        self._model = model
        self._torch = torch_module
        self._lpips_module = lpips_module
        self.device = "cpu"

    def _ensure_runtime(self) -> tuple[Any, Any]:
        if self._torch is None:
            import torch

            self._torch = torch
        if self._model is None:
            if self._lpips_module is None:
                import lpips

                self._lpips_module = lpips
            self.device = "cuda" if self.prefer_gpu and self._torch.cuda.is_available() else "cpu"
            model = self._lpips_module.LPIPS(net=self.backbone, spatial=True)
            model = model.to(self.device)
            model.eval()
            self._model = model
        return self._torch, self._model

    def _image_to_tensor(self, image: Image.Image, torch_module: Any) -> Any:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch_module.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0
        return tensor.to(self.device)

    def compute(self, original: Image.Image, generated: Image.Image, **kwargs: Any) -> np.ndarray:
        torch_module, model = self._ensure_runtime()
        resized_original = _resize_for_model(original, self.input_max_side)
        resized_generated = _resize_for_model(generated, self.input_max_side)
        if resized_generated.size != resized_original.size:
            resized_generated = resized_generated.resize(resized_original.size, Image.BICUBIC)

        with torch_module.no_grad():
            original_tensor = self._image_to_tensor(resized_original, torch_module)
            generated_tensor = self._image_to_tensor(resized_generated, torch_module)
            heatmap = model(original_tensor, generated_tensor)

        if isinstance(heatmap, (list, tuple)):
            heatmap = heatmap[0]
        if hasattr(heatmap, "detach"):
            heatmap = heatmap.detach()
        if hasattr(heatmap, "float"):
            heatmap = heatmap.float()
        if hasattr(heatmap, "cpu"):
            heatmap = heatmap.cpu()
        if hasattr(heatmap, "numpy"):
            heatmap = heatmap.numpy()
        heatmap_array = np.asarray(heatmap, dtype=np.float32).squeeze()
        if heatmap_array.ndim == 0:
            heatmap_array = np.full(
                (resized_original.size[1], resized_original.size[0]),
                float(heatmap_array),
                dtype=np.float32,
            )
        heatmap_array = _resize_heatmap(heatmap_array, original.size)
        return normalize_heatmap(
            heatmap_array,
            mode=self.normalization,
            percentile_low=self.percentile_low,
            percentile_high=self.percentile_high,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "pgcd_heatmap_backend": self.backend_name,
            "pgcd_lpips_backbone": self.backbone,
            "pgcd_lpips_input_max_side": self.input_max_side,
            "pgcd_heatmap_normalization": self.normalization,
            "pgcd_heatmap_percentile_low": self.percentile_low,
            "pgcd_heatmap_percentile_high": self.percentile_high,
        }


def build_heatmap_backend(
    backbone: str = "alex",
    input_max_side: int = 768,
    normalization: str = "percentile",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    prefer_gpu: bool = True,
) -> LPIPSHeatmapBackend | RGBDiffHeatmapBackend:
    try:
        backend = LPIPSHeatmapBackend(
            backbone=backbone,
            input_max_side=input_max_side,
            normalization=normalization,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            prefer_gpu=prefer_gpu,
        )
        backend._ensure_runtime()
        return backend
    except ModuleNotFoundError as exc:
        return RGBDiffHeatmapBackend(reason=f"optional_lpips_dependency_missing:{exc.name}")
