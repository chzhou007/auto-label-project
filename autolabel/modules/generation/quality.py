from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .localizers.components import bbox_iou


DEFAULT_QUALITY_PROFILES: dict[str, dict[str, float]] = {
    "default": {
        "background_preservation_min": 0.60,
        "anomaly_visibility_min": 0.02,
        "prompt_iou_min": 0.01,
        "visibility_scale": 0.08,
    },
    "water_leak": {
        "background_preservation_min": 0.65,
        "anomaly_visibility_min": 0.02,
        "prompt_iou_min": 0.02,
        "visibility_scale": 0.08,
    },
    "coolant_leak": {
        "background_preservation_min": 0.65,
        "anomaly_visibility_min": 0.02,
        "prompt_iou_min": 0.02,
        "visibility_scale": 0.08,
    },
    "oil_leak": {
        "background_preservation_min": 0.55,
        "anomaly_visibility_min": 0.02,
        "prompt_iou_min": 0.01,
        "visibility_scale": 0.08,
    },
    "diesel_leak": {
        "background_preservation_min": 0.55,
        "anomaly_visibility_min": 0.02,
        "prompt_iou_min": 0.01,
        "visibility_scale": 0.08,
    },
}


def _load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _load_mask(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != size:
            image = image.resize(size, Image.NEAREST)
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def _merge_profile_maps(base: dict[str, dict[str, float]], override: dict[str, Any]) -> dict[str, dict[str, float]]:
    merged = deepcopy(base)
    for name, profile in override.items():
        if not isinstance(profile, dict):
            continue
        target = dict(merged.get(name, {}))
        for key, value in profile.items():
            target[str(key)] = float(value)
        merged[str(name)] = target
    return merged


def resolve_quality_profile(
    anomaly_type: str,
    quality_config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, float]]:
    profiles = deepcopy(DEFAULT_QUALITY_PROFILES)
    if isinstance(quality_config, dict):
        if isinstance(quality_config.get("profiles"), dict):
            profiles = _merge_profile_maps(profiles, quality_config["profiles"])
        else:
            ad_hoc_profiles = {
                key: value
                for key, value in quality_config.items()
                if isinstance(value, dict)
            }
            if ad_hoc_profiles:
                profiles = _merge_profile_maps(profiles, ad_hoc_profiles)

    profile_name = anomaly_type if anomaly_type in profiles else "default"
    return profile_name, profiles[profile_name]


def build_quality_failure(
    reason: str,
    anomaly_type: str,
    quality_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_name, _profile = resolve_quality_profile(anomaly_type, quality_config=quality_config)
    return {
        "background_preservation_score": 0.0,
        "anomaly_visibility_score": 0.0,
        "passes_quality": False,
        "quality_reason": reason,
        "quality_threshold_profile": profile_name,
        "foreground_change": 0.0,
        "background_change": 0.0,
        "prompt_iou": 0.0,
    }


def evaluate_localization_quality(
    original_image_path: str | Path,
    generated_image_path: str | Path,
    final_bbox: list[int] | tuple[int, int, int, int] | None,
    mask_path: str | Path | None,
    anomaly_type: str,
    prompt_box: list[int] | tuple[int, int, int, int] | None = None,
    quality_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_name, profile = resolve_quality_profile(anomaly_type, quality_config=quality_config)
    if final_bbox is None or mask_path is None:
        return build_quality_failure("missing_localizer_output", anomaly_type, quality_config=quality_config)

    try:
        original = _load_rgb(original_image_path)
        generated = _load_rgb(generated_image_path)
        if generated.size != original.size:
            generated = generated.resize(original.size, Image.BICUBIC)

        mask = _load_mask(mask_path, original.size)
        if not np.any(mask):
            return build_quality_failure("empty_mask", anomaly_type, quality_config=quality_config)

        original_array = np.asarray(original, dtype=np.float32) / 255.0
        generated_array = np.asarray(generated, dtype=np.float32) / 255.0
        diff_map = np.abs(generated_array - original_array).mean(axis=2)

        foreground_change = float(diff_map[mask].mean()) if np.any(mask) else 0.0
        background_mask = ~mask
        background_change = float(diff_map[background_mask].mean()) if np.any(background_mask) else 0.0

        denom = max(foreground_change, 1e-6)
        background_preservation_score = float(np.clip(1.0 - (background_change / denom), 0.0, 1.0))

        visibility_scale = max(float(profile.get("visibility_scale", 0.08)), 1e-6)
        anomaly_visibility_score = float(np.clip(foreground_change / visibility_scale, 0.0, 1.0))

        prompt_iou = float(bbox_iou(final_bbox, prompt_box)) if prompt_box is not None else 1.0

        reasons: list[str] = []
        if background_preservation_score < float(profile.get("background_preservation_min", 0.60)):
            reasons.append("background_preservation_low")
        if anomaly_visibility_score < float(profile.get("anomaly_visibility_min", 0.02)):
            reasons.append("anomaly_visibility_low")
        if prompt_box is not None and prompt_iou < float(profile.get("prompt_iou_min", 0.01)):
            reasons.append("prompt_alignment_low")

        return {
            "background_preservation_score": background_preservation_score,
            "anomaly_visibility_score": anomaly_visibility_score,
            "passes_quality": not reasons,
            "quality_reason": None if not reasons else ",".join(reasons),
            "quality_threshold_profile": profile_name,
            "foreground_change": foreground_change,
            "background_change": background_change,
            "prompt_iou": prompt_iou,
        }
    except Exception as exc:
        return {
            **build_quality_failure(f"quality_error:{type(exc).__name__}", anomaly_type, quality_config=quality_config),
            "quality_error_message": str(exc),
        }
