from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import logging
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import I2IServiceConfig, PipelineConfig
from cropper import crop_anomaly
from diff_localizer import localize_change_bbox
from grid import bbox_to_box_dict, expand_bbox, grid_id_to_bbox, make_grid_preview
from metadata_builder import build_autolabel_sample, build_classification_labels
from prompts import NEGATIVE_PROMPT, build_wan_prompt, write_prompt_files
from qwen_vlm_client import QwenVLMClient
from utils import (
    ensure_output_dirs,
    image_size,
    load_dotenv_if_available,
    read_tasks_csv,
    relative_uri,
    resolve_image_path,
    setup_logging,
    write_json,
    copy_file,
)
from validators import ValidationError, validate_required_fields, validate_task
from wan_image_client import WanImageClient

logger = logging.getLogger(__name__)


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="VLM grid + image edit industrial anomaly autolabel pipeline")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--vlm-model", default="qwen3.6-27b")
    parser.add_argument("--image-model", default="doubao-seedream-5-0-pro-260628")
    parser.add_argument("--grid-layout", default="4x4", choices=["4x4"])
    parser.add_argument("--edit-bbox-expand-ratio", type=float, default=0.20)
    parser.add_argument("--crop-expand-ratio", type=float, default=0.03)
    parser.add_argument("--vlm-min-confidence", type=float, default=0.45)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Run without external model APIs using deterministic local stubs.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N tasks. Useful for API smoke tests.")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent samples to process.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples with existing valid metadata JSON.")
    parser.add_argument("--seedream-mode", choices=["single_image_edit", "boxed_single_edit", "boxed_fusion"], default=None)
    parser.add_argument("--water-reference-dir", default=None)
    parser.add_argument("--red-box-max-size", type=int, default=200)
    parser.add_argument("--red-box-min-size", type=int, default=200)
    args = parser.parse_args()
    return PipelineConfig(**vars(args))


def _candidate_grids(vlm_result: dict) -> list[str]:
    grids = [vlm_result["selected_grid"]]
    for item in vlm_result.get("top_candidates", []):
        grid = item.get("grid")
        if grid and grid not in grids:
            grids.append(grid)
    return grids[:3]


def _write_failure(log_dir: Path, sample_id: str, stage: str, error: Exception | str, context: dict | None = None) -> None:
    payload = {
        "sample_id": sample_id,
        "status": "failed",
        "stage": stage,
        "error": str(error),
        "context": context or {},
    }
    write_json(log_dir / f"{sample_id}_failure.json", payload)
    logger.error("%s failed at %s: %s", sample_id, stage, error)


def _write_skip(log_dir: Path, sample_id: str, reason: str, context: dict | None = None) -> None:
    payload = {
        "sample_id": sample_id,
        "status": "skipped",
        "stage": "floor_region_precheck",
        "reason": reason,
        "context": context or {},
    }
    write_json(log_dir / f"{sample_id}_skip.json", payload)
    logger.info("%s skipped at floor_region_precheck: %s", sample_id, reason)


def _failure_artifact_context(
    sample_id: str,
    dirs: dict[str, Path],
    generated_path: Path,
    request_log_path: Path,
    response_log_path: Path,
    mask_path: Path,
    crop_path: Path,
    extra_artifact_paths: dict[str, Path] | None = None,
) -> dict:
    artifacts: dict[str, str] = {}
    if generated_path.exists():
        failed_image_path = dirs["failed_generated_images"] / generated_path.name
        copy_file(generated_path, failed_image_path)
        artifacts["failed_generated_image_uri"] = relative_uri(failed_image_path)
        artifacts["generated_image_uri"] = relative_uri(generated_path)
    if request_log_path.exists():
        artifacts["request_log_uri"] = relative_uri(request_log_path)
    if response_log_path.exists():
        artifacts["response_log_uri"] = relative_uri(response_log_path)
    if mask_path.exists():
        artifacts["mask_uri"] = relative_uri(mask_path)
    if crop_path.exists():
        artifacts["crop_uri"] = relative_uri(crop_path)
    for key, path in (extra_artifact_paths or {}).items():
        if path.exists():
            artifacts[key] = relative_uri(path)
    return {"failure_artifacts": artifacts} if artifacts else {}


def _has_valid_metadata(metadata_path: Path) -> bool:
    if not metadata_path.exists():
        return False
    try:
        import json

        with open(metadata_path, encoding="utf-8") as f:
            sample = json.load(f)
        validate_required_fields(sample)
        for obj in sample.get("objects", []):
            params = obj.get("geometry_detail", {}).get("generation_params", {})
            vlm_selection = params.get("vlm_selection", {})
            raw_response = vlm_selection.get("raw_response", {})
            if isinstance(raw_response, dict) and raw_response.get("dry_run"):
                return False
        return True
    except Exception:
        return False


def _normalize_generated_size(image_path: Path, target_size: tuple[int, int]) -> None:
    with Image.open(image_path) as image:
        if image.size == target_size:
            return
        resized = image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
        resized.save(image_path)


def _compose_seedream_source_preserving_output(
    original_path: Path,
    raw_generated_path: Path,
    output_path: Path,
    edit_bbox: tuple[int, int, int, int],
    water_mask_output_path: Path | None = None,
    composition_mask_output_path: Path | None = None,
    anomaly_type: str | None = None,
) -> dict:
    with Image.open(original_path) as original_image, Image.open(raw_generated_path) as generated_image:
        original = original_image.convert("RGB")
        generated = generated_image.convert("RGB")
        if generated.size != original.size:
            generated = generated.resize(original.size, Image.Resampling.LANCZOS)

    width, height = original.size
    x1, y1, x2, y2 = edit_bbox
    x1 = max(0, min(width, int(x1)))
    x2 = max(0, min(width, int(x2)))
    y1 = max(0, min(height, int(y1)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty Seedream composition bbox")

    original_array = np.asarray(original, dtype=np.int16)
    generated_array = np.asarray(generated, dtype=np.int16)
    original_roi = original_array[y1:y2, x1:x2].astype(np.uint8)
    generated_roi = generated_array[y1:y2, x1:x2].astype(np.uint8)
    water = _extract_seedream_water_mask(original_roi, generated_roi, anomaly_type)
    local_water_mask = water["mask"]
    wx1, wy1, wx2, wy2 = water["local_bbox"]

    roi_diff = np.max(np.abs(generated_array[y1:y2, x1:x2] - original_array[y1:y2, x1:x2]), axis=2)
    threshold = int(os.getenv("SEEDREAM_COMPOSE_DIFF_THRESHOLD", "8"))
    raw_diff_mask = np.where(roi_diff > threshold, 255, 0).astype(np.uint8)
    raw_changed_ratio = float((raw_diff_mask > 0).mean()) if raw_diff_mask.size else 0.0
    water_mask_area = int((local_water_mask > 0).sum())
    red_box_area = max(1, (x2 - x1) * (y2 - y1))
    water_mask_coverage_ratio = float(water_mask_area / red_box_area)
    water_bbox = (x1 + wx1, y1 + wy1, x1 + wx2, y1 + wy2)
    bbox_red_box_iou = _bbox_tuple_iou(water_bbox, (x1, y1, x2, y2))
    patch_like_score = max(water_mask_coverage_ratio, bbox_red_box_iou)

    mask = Image.new("L", original.size, 0)
    water_mask_image = Image.fromarray(local_water_mask, mode="L")
    mask.paste(water_mask_image, (x1, y1))
    composition_mask = Image.new("L", original.size, 0)
    composition_local_mask = water_mask_image.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(radius=2))
    composition_mask.paste(composition_local_mask, (x1, y1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.composite(generated, original, composition_mask).save(output_path)
    if water_mask_output_path is not None:
        water_mask_output_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(water_mask_output_path)
    if composition_mask_output_path is not None:
        composition_mask_output_path.parent.mkdir(parents=True, exist_ok=True)
        composition_mask.save(composition_mask_output_path)
    return {
        "seedream_composition_mode": "source_preserving_water_mask_blend",
        "seedream_raw_output_uri": relative_uri(raw_generated_path),
        "seedream_composition_bbox": [x1, y1, x2, y2],
        "seedream_composition_mask_bbox": list(water_bbox),
        "seedream_water_mask_bbox": list(water_bbox),
        "seedream_water_mask_area": water_mask_area,
        "seedream_mask_coverage_ratio": water_mask_coverage_ratio,
        "seedream_bbox_red_box_iou": bbox_red_box_iou,
        "seedream_patch_like_score": patch_like_score,
        "seedream_raw_changed_ratio": raw_changed_ratio,
        "seedream_composition_changed_ratio": raw_changed_ratio,
        "seedream_water_mask_uri": relative_uri(water_mask_output_path) if water_mask_output_path is not None else None,
        "seedream_composition_mask_uri": (
            relative_uri(composition_mask_output_path) if composition_mask_output_path is not None else None
        ),
    }


def _prepare_seedream_raw_candidate_output(
    original_path: Path,
    raw_generated_path: Path,
    output_path: Path,
    edit_bbox: tuple[int, int, int, int],
    water_mask_output_path: Path | None = None,
    raw_diff_mask_output_path: Path | None = None,
    anomaly_type: str | None = None,
) -> dict:
    with Image.open(original_path) as original_image, Image.open(raw_generated_path) as generated_image:
        original = original_image.convert("RGB")
        generated = generated_image.convert("RGB")
        if generated.size != original.size:
            generated = generated.resize(original.size, Image.Resampling.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated.save(output_path)

    original_array = np.asarray(original, dtype=np.uint8)
    generated_array = np.asarray(generated, dtype=np.uint8)
    localization = _localize_seedream_raw_water_change(
        original_array,
        generated_array,
        edit_bbox,
        anomaly_type=anomaly_type,
    )
    mask = Image.fromarray(localization["mask"], mode="L")
    if water_mask_output_path is not None:
        water_mask_output_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(water_mask_output_path)
    if raw_diff_mask_output_path is not None:
        raw_diff_mask_output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(localization["candidate_mask"], mode="L").save(raw_diff_mask_output_path)

    return {
        "seedream_composition_mode": "raw_candidate_background_gated",
        "seedream_final_image_source": "seedream_raw_output",
        "seedream_raw_output_uri": relative_uri(raw_generated_path),
        "seedream_composition_bbox": list(localization["prompt_box"]),
        "seedream_composition_mask_bbox": list(localization["bbox"]),
        "seedream_water_mask_bbox": list(localization["bbox"]),
        "seedream_water_mask_area": int(localization["area"]),
        "seedream_mask_coverage_ratio": float(localization["mask_coverage_ratio"]),
        "seedream_bbox_red_box_iou": float(localization["bbox_red_box_iou"]),
        "seedream_patch_like_score": float(localization["patch_like_score"]),
        "seedream_raw_changed_ratio": float(localization["raw_changed_ratio"]),
        "seedream_composition_changed_ratio": float(localization["raw_changed_ratio"]),
        "seedream_water_mask_uri": relative_uri(water_mask_output_path) if water_mask_output_path is not None else None,
        "seedream_raw_diff_mask_uri": relative_uri(raw_diff_mask_output_path) if raw_diff_mask_output_path is not None else None,
        "seedream_raw_localizer_metrics": localization["metrics"],
    }


def _bbox_tuple_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _clip_bbox_tuple(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    clipped = max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"empty bbox after clipping: {bbox}")
    return clipped


def _connected_component_candidates(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return labels, stats, centroids, num_labels


def _localize_seedream_raw_water_change(
    original_array: np.ndarray,
    generated_array: np.ndarray,
    edit_bbox: tuple[int, int, int, int],
    *,
    anomaly_type: str | None = None,
) -> dict:
    if original_array.shape != generated_array.shape or original_array.size == 0:
        raise ValueError("Seedream raw localization failed: image shape mismatch")

    height, width = original_array.shape[:2]
    x1, y1, x2, y2 = _clip_bbox_tuple(edit_bbox, width, height)
    original_roi = original_array[y1:y2, x1:x2]
    generated_roi = generated_array[y1:y2, x1:x2]
    roi_h, roi_w = original_roi.shape[:2]
    roi_area = max(1, roi_w * roi_h)

    original_lab = cv2.cvtColor(original_roi, cv2.COLOR_RGB2LAB).astype(np.float32)
    generated_lab = cv2.cvtColor(generated_roi, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_diff = np.linalg.norm(generated_lab - original_lab, axis=2)
    rgb_diff = np.abs(generated_roi.astype(np.int16) - original_roi.astype(np.int16))
    max_rgb_diff = rgb_diff.max(axis=2)
    raw_changed = (lab_diff > float(os.getenv("SEEDREAM_RAW_DIFF_THRESHOLD", "10"))) | (max_rgb_diff > 10)
    raw_changed_ratio = float(raw_changed.mean()) if raw_changed.size else 0.0

    fallback_candidate = raw_changed
    if anomaly_type != "water_leak":
        candidate = raw_changed
    else:
        hsv = cv2.cvtColor(generated_roi, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        original_gray = cv2.cvtColor(original_roi, cv2.COLOR_RGB2GRAY).astype(np.int16)
        generated_gray = cv2.cvtColor(generated_roi, cv2.COLOR_RGB2GRAY).astype(np.int16)
        gray_delta = generated_gray - original_gray
        gray_abs_delta = np.abs(gray_delta)
        generated_gray_float = generated_gray.astype(np.float32) / 255.0
        gx = np.zeros_like(generated_gray_float)
        gy = np.zeros_like(generated_gray_float)
        gx[:, 1:] = np.abs(generated_gray_float[:, 1:] - generated_gray_float[:, :-1])
        gy[1:, :] = np.abs(generated_gray_float[1:, :] - generated_gray_float[:-1, :])
        edge_strength = np.maximum(gx, gy)

        low_sat = saturation <= int(os.getenv("SEEDREAM_RAW_WATER_MAX_SATURATION", "135"))
        not_white_panel = value < int(os.getenv("SEEDREAM_RAW_WATER_MAX_VALUE", "245"))
        not_hard_edge = edge_strength < float(os.getenv("SEEDREAM_RAW_WATER_MAX_EDGE", "0.12"))
        wet_dark = gray_delta <= -int(os.getenv("SEEDREAM_RAW_WATER_DARK_DELTA", "4"))
        wet_reflection = gray_delta >= int(os.getenv("SEEDREAM_RAW_WATER_BRIGHT_DELTA", "10"))
        changed_enough = raw_changed | (gray_abs_delta >= int(os.getenv("SEEDREAM_RAW_WATER_ABS_DELTA", "8")))
        red_residual = (generated_roi[:, :, 0] > 160) & (generated_roi[:, :, 1] < 100) & (generated_roi[:, :, 2] < 100)
        candidate = changed_enough & low_sat & not_white_panel & not_hard_edge & (wet_dark | wet_reflection | (gray_abs_delta >= 12))
        candidate &= ~red_residual
        fallback_candidate = raw_changed & ~red_residual

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    configured_min_area = int(os.getenv("SEEDREAM_RAW_WATER_MIN_AREA", "300"))
    min_area = max(24, min(configured_min_area, int(roi_area * 0.01)))
    max_area = int(roi_area * float(os.getenv("SEEDREAM_RAW_WATER_MAX_AREA_RATIO", "0.55")))
    max_bbox_area_ratio = float(os.getenv("SEEDREAM_RAW_WATER_MAX_BBOX_AREA_RATIO", "0.85"))
    configured_min_side = int(os.getenv("SEEDREAM_RAW_WATER_MIN_BBOX_SIDE", "24"))
    min_side = max(8, min(configured_min_side, int(min(roi_w, roi_h) * 0.35)))

    def build_component_candidates(mask_source: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[float, int, tuple[int, int, int, int], int]]]:
        mask = np.where(mask_source, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        component_labels, component_stats, _centroids, component_count = _connected_component_candidates(mask)
        found: list[tuple[float, int, tuple[int, int, int, int], int]] = []
        for label in range(1, component_count):
            area = int(component_stats[label, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            lx = int(component_stats[label, cv2.CC_STAT_LEFT])
            ly = int(component_stats[label, cv2.CC_STAT_TOP])
            lw = int(component_stats[label, cv2.CC_STAT_WIDTH])
            lh = int(component_stats[label, cv2.CC_STAT_HEIGHT])
            if lw < min_side or lh < min_side:
                continue
            bbox_area = max(1, lw * lh)
            bbox_area_ratio = bbox_area / float(roi_area)
            if bbox_area_ratio > max_bbox_area_ratio:
                continue
            component = component_labels == label
            component_diff = float(lab_diff[component].mean()) if np.any(component) else 0.0
            lower_bias = (ly + lh / 2.0) / max(1.0, roi_h)
            extent = area / float(bbox_area)
            aspect = max(lw / max(1, lh), lh / max(1, lw))
            touches_side = lx <= 1 or lx + lw >= roi_w - 1
            touches_top = ly <= 1
            score = 1.3 * min(1.0, area / max(1.0, roi_area * 0.12))
            score += 0.7 * min(1.0, component_diff / 45.0)
            score += 0.35 * lower_bias
            score += 0.25 * min(1.0, extent / 0.65)
            score -= 0.12 * max(0.0, aspect - 4.0)
            if touches_top:
                score -= 0.4
            if touches_side:
                score -= 0.1
            found.append((score, label, (lx, ly, lx + lw, ly + lh), area))
        return mask, component_labels, found

    candidate_mask, labels, candidates = build_component_candidates(candidate)
    used_fallback_changed_components = False
    if not candidates and anomaly_type == "water_leak":
        candidate_mask, labels, candidates = build_component_candidates(fallback_candidate)
        used_fallback_changed_components = bool(candidates)

    if not candidates:
        raise ValueError("Seedream raw localization failed: no water-like changed components")

    candidates.sort(reverse=True, key=lambda item: item[0])
    best_score = candidates[0][0]
    keep = np.zeros(candidate_mask.shape, dtype=np.uint8)
    kept_boxes: list[tuple[int, int, int, int]] = []
    kept_area = 0
    for score, label, box, area in candidates[:5]:
        if kept_boxes and score < best_score - float(os.getenv("SEEDREAM_RAW_WATER_COMPONENT_SCORE_WINDOW", "0.55")):
            continue
        if kept_area + area > max_area:
            continue
        keep[labels == label] = 255
        kept_boxes.append(box)
        kept_area += area

    if not kept_boxes:
        raise ValueError("Seedream raw localization failed: all water-like candidates rejected")

    bx1 = min(box[0] for box in kept_boxes)
    by1 = min(box[1] for box in kept_boxes)
    bx2 = max(box[2] for box in kept_boxes)
    by2 = max(box[3] for box in kept_boxes)
    if bx2 - bx1 < min_side or by2 - by1 < min_side:
        raise ValueError(f"Seedream raw localization failed: bbox too small {(bx1, by1, bx2, by2)}")

    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = keep
    full_candidate_mask = np.zeros((height, width), dtype=np.uint8)
    full_candidate_mask[y1:y2, x1:x2] = candidate_mask
    final_bbox = (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)
    coverage_ratio = kept_area / float(roi_area)
    bbox_iou = _bbox_tuple_iou(final_bbox, (x1, y1, x2, y2))
    patch_like_score = max(coverage_ratio, bbox_iou)
    return {
        "bbox": final_bbox,
        "prompt_box": (x1, y1, x2, y2),
        "mask": full_mask,
        "candidate_mask": full_candidate_mask,
        "area": kept_area,
        "mask_coverage_ratio": coverage_ratio,
        "bbox_red_box_iou": bbox_iou,
        "patch_like_score": patch_like_score,
        "raw_changed_ratio": raw_changed_ratio,
        "metrics": {
            "candidate_component_count": len(candidates),
            "kept_component_count": len(kept_boxes),
            "best_component_score": float(best_score),
            "raw_changed_ratio": raw_changed_ratio,
            "mask_coverage_ratio": coverage_ratio,
            "bbox_red_box_iou": bbox_iou,
            "used_fallback_changed_components": used_fallback_changed_components,
        },
    }


def _bbox_iou(a: dict, b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = int(a["x1"]), int(a["y1"]), int(a["x2"]), int(a["y2"])
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _extract_seedream_water_mask(original_roi: np.ndarray, generated_roi: np.ndarray, anomaly_type: str | None) -> dict:
    if original_roi.shape != generated_roi.shape or original_roi.size == 0:
        raise ValueError("empty Seedream water mask roi")

    diff = np.abs(generated_roi.astype(np.int16) - original_roi.astype(np.int16))
    max_diff = diff.max(axis=2)
    mean_diff = diff.mean(axis=2)
    threshold = int(os.getenv("SEEDREAM_COMPOSE_DIFF_THRESHOLD", "8"))
    changed = (max_diff > threshold) | (mean_diff > max(4.0, threshold * 0.6))

    if anomaly_type != "water_leak":
        mask = np.where(changed, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            raise ValueError("Seedream mask extraction failed: no changed pixels")
        return {
            "mask": mask,
            "local_bbox": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            "area": int((mask > 0).sum()),
            "coverage_ratio": float((mask > 0).mean()),
            "bbox_area_ratio": float(((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / mask.size),
        }

    hsv = cv2.cvtColor(generated_roi, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    original_gray = cv2.cvtColor(original_roi, cv2.COLOR_RGB2GRAY).astype(np.int16)
    generated_gray = cv2.cvtColor(generated_roi, cv2.COLOR_RGB2GRAY).astype(np.int16)
    gray_delta = generated_gray - original_gray

    max_saturation = int(os.getenv("SEEDREAM_WATER_MAX_SATURATION", "110"))
    dark_delta = int(os.getenv("SEEDREAM_WATER_DARK_DELTA", "5"))
    bright_delta = int(os.getenv("SEEDREAM_WATER_BRIGHT_DELTA", "14"))
    low_saturation = saturation <= max_saturation
    wet_dark = gray_delta <= -dark_delta
    subtle_reflection = (gray_delta >= bright_delta) & (saturation <= max_saturation - 20) & (value < 245)
    water_like = changed & low_saturation & (wet_dark | subtle_reflection | (np.abs(gray_delta) >= bright_delta))

    if float(water_like.mean()) < 0.001:
        water_like = changed & low_saturation & (np.abs(gray_delta) >= max(8, bright_delta // 2))

    red_residual = (generated_roi[:, :, 0] > 160) & (generated_roi[:, :, 1] < 100) & (generated_roi[:, :, 2] < 100)
    mask = np.where(water_like & ~red_residual, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    roi_h, roi_w = mask.shape[:2]
    roi_area = max(1, roi_w * roi_h)
    min_area = max(24, int(roi_area * float(os.getenv("SEEDREAM_WATER_MIN_AREA_RATIO", "0.0015"))))
    max_area = int(roi_area * float(os.getenv("SEEDREAM_WATER_MAX_AREA_RATIO", "0.45")))
    max_bbox_area = roi_area * float(os.getenv("SEEDREAM_WATER_MAX_BBOX_AREA_RATIO", "0.70"))

    candidates: list[tuple[float, int, tuple[int, int, int, int]]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        lx = int(stats[label, cv2.CC_STAT_LEFT])
        ly = int(stats[label, cv2.CC_STAT_TOP])
        lw = int(stats[label, cv2.CC_STAT_WIDTH])
        lh = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox_area = max(1, lw * lh)
        if bbox_area > max_bbox_area:
            continue
        extent = area / float(bbox_area)
        aspect = max(lw / max(1, lh), lh / max(1, lw))
        touches_border = lx <= 1 or ly <= 1 or lx + lw >= roi_w - 1 or ly + lh >= roi_h - 1
        component = labels == label
        dark_fraction = float((wet_dark & component).sum() / max(1, area))
        low_sat_fraction = float((low_saturation & component).sum() / max(1, area))
        lower_bias = (ly + lh / 2.0) / max(1.0, roi_h)
        area_score = min(1.0, area / max(1.0, roi_area * 0.08))
        score = 1.0 * area_score + 0.9 * dark_fraction + 0.5 * low_sat_fraction + 0.25 * lower_bias
        score -= 0.8 * max(0.0, extent - 0.72)
        score -= 0.25 * max(0.0, aspect - 5.0)
        if touches_border:
            score -= 0.35
        candidates.append((score, label, (lx, ly, lx + lw, ly + lh)))

    if not candidates:
        raise ValueError("Seedream water mask extraction failed: no compact water-like components")

    candidates.sort(reverse=True, key=lambda item: item[0])
    best_score = candidates[0][0]
    keep = np.zeros(mask.shape, dtype=np.uint8)
    kept_boxes: list[tuple[int, int, int, int]] = []
    kept_area = 0
    for score, label, box in candidates[:4]:
        area = int(stats[label, cv2.CC_STAT_AREA])
        if score < best_score - 0.65 and kept_boxes:
            continue
        if (kept_area + area) > max_area:
            continue
        keep[labels == label] = 255
        kept_boxes.append(box)
        kept_area += area

    if not kept_boxes:
        raise ValueError("Seedream water mask extraction failed: all water-like components rejected")
    bx1 = min(box[0] for box in kept_boxes)
    by1 = min(box[1] for box in kept_boxes)
    bx2 = max(box[2] for box in kept_boxes)
    by2 = max(box[3] for box in kept_boxes)
    if bx2 - bx1 < 8 or by2 - by1 < 8:
        raise ValueError(f"Seedream water mask extraction failed: bbox too small {(bx1, by1, bx2, by2)}")

    coverage_ratio = kept_area / float(roi_area)
    bbox_area_ratio = ((bx2 - bx1) * (by2 - by1)) / float(roi_area)
    if coverage_ratio > float(os.getenv("SEEDREAM_WATER_MAX_AREA_RATIO", "0.45")):
        raise ValueError(f"Seedream water mask extraction failed: mask too large ({coverage_ratio:.3f})")
    if bbox_area_ratio > float(os.getenv("SEEDREAM_WATER_MAX_BBOX_AREA_RATIO", "0.70")):
        raise ValueError(f"Seedream water mask extraction failed: bbox too large ({bbox_area_ratio:.3f})")

    return {
        "mask": keep,
        "local_bbox": (bx1, by1, bx2, by2),
        "area": kept_area,
        "coverage_ratio": coverage_ratio,
        "bbox_area_ratio": bbox_area_ratio,
    }


def _is_refined_final_bbox(final_box: dict, expanded_bbox: tuple[int, int, int, int]) -> bool:
    expanded_box = bbox_to_box_dict(expanded_bbox)
    return final_box != expanded_box and _bbox_iou(final_box, expanded_bbox) < 0.95


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _stable_rng(sample_id: str) -> random.Random:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _choose_seedream_red_box(
    sample_id: str,
    grid_bbox: tuple[int, int, int, int],
    min_size: int,
    max_size: int,
    anomaly_type: str | None = None,
) -> tuple[int, int, int, int]:
    gx1, gy1, gx2, gy2 = grid_bbox
    grid_width = max(1, gx2 - gx1)
    grid_height = max(1, gy2 - gy1)
    upper = max(1, min(int(max_size), grid_width, grid_height))
    lower = max(1, min(int(min_size), upper))
    rng = _stable_rng(sample_id)
    box_width = rng.randint(lower, upper)
    box_height = rng.randint(lower, upper)
    x1 = rng.randint(gx1, max(gx1, gx2 - box_width))
    y_min = gy1
    y_max = max(gy1, gy2 - box_height)
    if anomaly_type == "water_leak":
        preferred_y_min = gy1 + int(grid_height * 0.45)
        y_min = min(preferred_y_min, y_max)
    y1 = rng.randint(y_min, y_max)
    return (x1, y1, x1 + box_width, y1 + box_height)


def _choose_seedream_water_leak_box(
    image_path: str | Path,
    sample_id: str,
    grid_bbox: tuple[int, int, int, int],
    min_size: int,
    max_size: int,
) -> tuple[int, int, int, int]:
    fallback = _choose_seedream_red_box(sample_id, grid_bbox, min_size, max_size, "water_leak")
    gx1, gy1, gx2, gy2 = grid_bbox
    grid_width = max(1, gx2 - gx1)
    grid_height = max(1, gy2 - gy1)
    upper = max(1, min(int(max_size), grid_width, grid_height))
    lower = max(1, min(int(min_size), upper))
    box_size = upper if upper >= lower else lower
    if grid_width < box_size or grid_height < box_size:
        return fallback

    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
    except Exception:
        return fallback

    return _choose_seedream_water_leak_floor_box(rgb, sample_id, grid_bbox, min_size, max_size)


def _choose_seedream_water_leak_floor_box(
    rgb: Image.Image,
    sample_id: str,
    grid_bbox: tuple[int, int, int, int],
    min_size: int,
    max_size: int,
) -> tuple[int, int, int, int]:
    fallback = _choose_seedream_red_box(sample_id, grid_bbox, min_size, max_size, "water_leak")
    gx1, gy1, gx2, gy2 = grid_bbox
    grid_width = max(1, gx2 - gx1)
    grid_height = max(1, gy2 - gy1)
    upper = max(1, min(int(max_size), grid_width, grid_height))
    lower = max(1, min(int(min_size), upper))
    box_size = upper if upper >= lower else lower
    if grid_width < box_size or grid_height < box_size:
        raise ValueError("no_visible_floor_region: selected grid is smaller than required water edit box")

    x_stop = max(gx1, gx2 - box_size)
    y_stop = max(gy1, gy2 - box_size)
    y_start = min(gy1 + int(grid_height * 0.45), y_stop)
    step = max(8, box_size // 5)
    best_box = fallback
    best_score = float("-inf")
    best_metrics: dict[str, float] = {}
    for y1 in range(y_start, y_stop + 1, step):
        for x1 in range(gx1, x_stop + 1, step):
            x2 = x1 + box_size
            y2 = y1 + box_size
            crop = np.asarray(rgb.crop((x1, y1, x2, y2)), dtype=np.float32) / 255.0
            maxc = crop.max(axis=2)
            minc = crop.min(axis=2)
            saturation = np.divide(maxc - minc, maxc, out=np.zeros_like(maxc), where=maxc > 0.001)
            brightness = maxc
            mean_sat = float(saturation.mean())
            mean_brightness = float(brightness.mean())
            gray = (0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2])
            gx = np.zeros_like(gray)
            gy = np.zeros_like(gray)
            gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
            gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
            edge_density = float((np.maximum(gx, gy) > 0.07).mean())
            smooth_ratio = float((np.maximum(gx, gy) < 0.025).mean())
            red_yellow_green = (
                ((crop[:, :, 0] > 0.45) & (crop[:, :, 0] > crop[:, :, 1] * 1.15))
                | ((crop[:, :, 0] > 0.45) & (crop[:, :, 1] > 0.35) & (crop[:, :, 2] < 0.25))
                | ((crop[:, :, 1] > 0.35) & (crop[:, :, 1] > crop[:, :, 0] * 1.15))
            )
            saturated_equipment_ratio = float((red_yellow_green & (saturation > 0.20)).mean())
            too_dark_ratio = float((brightness < 0.18).mean())
            max_floor_brightness = float(os.getenv("SEEDREAM_MAX_FLOOR_BRIGHTNESS", "0.86"))
            too_bright_ratio = float((brightness > max_floor_brightness).mean())
            low_saturation_ratio = float((saturation < 0.18).mean())
            floor_like_ratio = float(
                (
                    (saturation < 0.22)
                    & (brightness > 0.18)
                    & (brightness < max_floor_brightness)
                    & (np.maximum(gx, gy) < 0.06)
                ).mean()
            )
            y_bias = (y1 - gy1) / max(1, grid_height - box_size)
            score = (
                (1.8 * floor_like_ratio)
                + (1.0 * low_saturation_ratio)
                + (0.8 * smooth_ratio)
                + (0.6 * y_bias)
                + (0.2 * mean_brightness)
            )
            score -= 2.5 * saturated_equipment_ratio
            score -= 0.8 * too_dark_ratio
            score -= 0.6 * too_bright_ratio
            score -= 1.6 * edge_density
            if score > best_score:
                best_score = score
                best_box = (x1, y1, x2, y2)
                best_metrics = {
                    "floor_score": float(score),
                    "floor_like_ratio": floor_like_ratio,
                    "low_saturation_ratio": low_saturation_ratio,
                    "smooth_ratio": smooth_ratio,
                    "edge_density": edge_density,
                    "saturated_equipment_ratio": saturated_equipment_ratio,
                    "too_dark_ratio": too_dark_ratio,
                    "too_bright_ratio": too_bright_ratio,
                }
    min_score = float(os.getenv("SEEDREAM_MIN_FLOOR_SCORE", "1.35"))
    min_floor_ratio = float(os.getenv("SEEDREAM_MIN_FLOOR_LIKE_RATIO", "0.45"))
    max_edge_density = float(os.getenv("SEEDREAM_MAX_FLOOR_EDGE_DENSITY", "0.18"))
    if (
        best_score < min_score
        or best_metrics.get("floor_like_ratio", 0.0) < min_floor_ratio
        or best_metrics.get("edge_density", 1.0) > max_edge_density
    ):
        raise ValueError(
            "no_visible_floor_region: selected grid has no reliable floor patch "
            f"(score={best_score:.3f}, floor_like_ratio={best_metrics.get('floor_like_ratio', 0.0):.3f}, "
            f"edge_density={best_metrics.get('edge_density', 1.0):.3f})"
        )
    return best_box


def _draw_seedream_red_box(
    original_path: str | Path,
    output_path: str | Path,
    bbox: tuple[int, int, int, int],
) -> None:
    with Image.open(original_path) as image:
        guide = image.convert("RGB")
    draw = ImageDraw.Draw(guide)
    draw.rectangle(bbox, outline=(255, 0, 0), width=4)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    guide.save(output_path, quality=95)


def _water_reference_files(reference_dir: str | Path | None) -> list[Path]:
    if not reference_dir:
        return []
    root = Path(reference_dir)
    if not root.exists() or not root.is_dir():
        return []
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in allowed)


def _select_water_reference(reference_dir: str | Path | None, sample_id: str) -> Path:
    candidates = _water_reference_files(reference_dir)
    if not candidates:
        raise ValueError(f"boxed_fusion requires water reference images under: {reference_dir}")
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return candidates[int(digest[:8], 16) % len(candidates)]


def _seedream_reference_copy_path(reference_path: Path, sample_id: str, output_dir: Path) -> Path:
    suffix = reference_path.suffix.lower() if reference_path.suffix else ".png"
    digest = hashlib.sha256(f"{sample_id}|{reference_path.name}".encode("utf-8")).hexdigest()[:16]
    return output_dir / f"ref_{digest}{suffix}"


def _red_pixel_ratio(image_path: str | Path, bbox: tuple[int, int, int, int]) -> float:
    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = array[y1:y2, x1:x2]
    red = (crop[:, :, 0] > 180) & (crop[:, :, 1] < 90) & (crop[:, :, 2] < 90)
    return float(red.mean())


def _outside_change_ratio(
    original_path: str | Path,
    generated_path: str | Path,
    allowed_bbox: tuple[int, int, int, int],
    threshold: float = 30.0,
) -> float:
    with Image.open(original_path) as original_image, Image.open(generated_path) as generated_image:
        original = original_image.convert("RGB")
        generated = generated_image.convert("RGB")
        if generated.size != original.size:
            generated = generated.resize(original.size, Image.Resampling.LANCZOS)
        original_array = np.asarray(original, dtype=np.int16)
        generated_array = np.asarray(generated, dtype=np.int16)
    diff = np.abs(generated_array - original_array).mean(axis=2)
    height, width = diff.shape
    x1, y1, x2, y2 = allowed_bbox
    mask = np.ones((height, width), dtype=bool)
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = False
    if not np.any(mask):
        return 0.0
    return float((diff[mask] > threshold).mean())


def _edge_map(rgb_array: np.ndarray, threshold: float) -> np.ndarray:
    gray = (
        0.299 * rgb_array[:, :, 0].astype(np.float32)
        + 0.587 * rgb_array[:, :, 1].astype(np.float32)
        + 0.114 * rgb_array[:, :, 2].astype(np.float32)
    ) / 255.0
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    return np.maximum(gx, gy) > threshold


def _outside_structure_change_ratio(
    original_path: str | Path,
    generated_path: str | Path,
    allowed_bbox: tuple[int, int, int, int],
) -> float:
    with Image.open(original_path) as original_image, Image.open(generated_path) as generated_image:
        original = original_image.convert("RGB")
        generated = generated_image.convert("RGB")
        if generated.size != original.size:
            generated = generated.resize(original.size, Image.Resampling.LANCZOS)

        max_side = int(os.getenv("SEEDREAM_STRUCTURE_MAX_SIDE", "640"))
        scale = min(1.0, max_side / float(max(original.size)))
        if scale < 1.0:
            resized_size = (max(1, int(original.size[0] * scale)), max(1, int(original.size[1] * scale)))
            original = original.resize(resized_size, Image.Resampling.BICUBIC)
            generated = generated.resize(resized_size, Image.Resampling.BICUBIC)

    original_array = np.asarray(original, dtype=np.uint8)
    generated_array = np.asarray(generated, dtype=np.uint8)
    height, width = original_array.shape[:2]
    x1, y1, x2, y2 = allowed_bbox
    if scale < 1.0:
        x1, x2 = int(x1 * scale), int(x2 * scale)
        y1, y2 = int(y1 * scale), int(y2 * scale)
    mask = np.ones((height, width), dtype=bool)
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = False
    if not np.any(mask):
        return 0.0

    edge_threshold = float(os.getenv("SEEDREAM_STRUCTURE_EDGE_THRESHOLD", "0.08"))
    original_edges = _edge_map(original_array, edge_threshold)
    generated_edges = _edge_map(generated_array, edge_threshold)
    changed_edges = np.logical_xor(original_edges, generated_edges)
    return float(changed_edges[mask].mean())


def _validate_seedream_experiment_output(
    original_path: str | Path,
    generated_path: str | Path,
    allowed_bbox: tuple[int, int, int, int],
    seedream_mode: str | None,
    composition_metadata: dict | None = None,
) -> dict:
    result = {
        "passes_quality": True,
        "quality_reason": None,
        "outside_change_ratio": _outside_change_ratio(original_path, generated_path, allowed_bbox),
        "outside_structure_change_ratio": _outside_structure_change_ratio(original_path, generated_path, allowed_bbox),
        "red_box_residual_ratio": 0.0,
        "seedream_mask_coverage_ratio": None,
        "seedream_bbox_red_box_iou": None,
        "seedream_patch_like_score": None,
        "seedream_raw_changed_ratio": None,
    }
    if composition_metadata:
        result["seedream_mask_coverage_ratio"] = composition_metadata.get("seedream_mask_coverage_ratio")
        result["seedream_bbox_red_box_iou"] = composition_metadata.get("seedream_bbox_red_box_iou")
        result["seedream_patch_like_score"] = composition_metadata.get("seedream_patch_like_score")
        result["seedream_raw_changed_ratio"] = composition_metadata.get("seedream_raw_changed_ratio")
    max_outside_change = float(str(os.getenv("SEEDREAM_MAX_OUTSIDE_CHANGE_RATIO", "0.20")).strip())
    max_structure_change = float(str(os.getenv("SEEDREAM_MAX_OUTSIDE_STRUCTURE_CHANGE_RATIO", "0.05")).strip())
    reasons = []
    if (
        result["outside_change_ratio"] > max_outside_change
        and result["outside_structure_change_ratio"] > max_structure_change
    ):
        reasons.append("seedream_outside_region_change_high")
    if seedream_mode in {"boxed_single_edit", "boxed_fusion"}:
        result["red_box_residual_ratio"] = _red_pixel_ratio(generated_path, allowed_bbox)
        max_red_ratio = float(str(os.getenv("SEEDREAM_MAX_RED_RESIDUAL_RATIO", "0.02")).strip())
        if result["red_box_residual_ratio"] > max_red_ratio:
            reasons.append("seedream_red_box_residual")
    if composition_metadata:
        max_mask_coverage = float(str(os.getenv("SEEDREAM_MAX_WATER_MASK_COVERAGE_RATIO", "0.45")).strip())
        max_bbox_iou = float(str(os.getenv("SEEDREAM_MAX_WATER_BBOX_RED_BOX_IOU", "0.75")).strip())
        mask_coverage = float(composition_metadata.get("seedream_mask_coverage_ratio") or 0.0)
        bbox_iou = float(composition_metadata.get("seedream_bbox_red_box_iou") or 0.0)
        if mask_coverage > max_mask_coverage or bbox_iou > max_bbox_iou:
            reasons.append("seedream_patch_like_region")
    if reasons:
        result["passes_quality"] = False
        result["quality_reason"] = ",".join(reasons)
    return result


def process_task(task: dict, cfg: PipelineConfig, dirs: dict[str, Path], vlm: QwenVLMClient, wan: WanImageClient) -> bool | None:
    sample_id = task["sample_id"]
    stale_metadata = dirs["metadata"] / f"{sample_id}.json"
    failure_log = dirs["logs"] / f"{sample_id}_failure.json"
    if stale_metadata.exists():
        stale_metadata.unlink()
    if failure_log.exists():
        failure_log.unlink()
    validate_task(task)
    original_path = resolve_image_path(task, cfg.image_root)
    width, height = image_size(original_path)

    grid_preview_path = dirs["grid_previews"] / f"{sample_id}_grid.jpg"
    make_grid_preview(str(original_path), str(grid_preview_path))

    vlm_result = None
    last_error: Exception | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            vlm_result = vlm.select_grid_with_qwen(
                str(grid_preview_path),
                task["anomaly_type"],
                str(dirs["logs"] / f"{sample_id}_vlm_response.json"),
            )
            if vlm_result["confidence"] < cfg.vlm_min_confidence:
                raise ValueError(f"VLM confidence below threshold: {vlm_result['confidence']}")
            break
        except Exception as exc:
            last_error = exc
            logger.warning("%s VLM attempt %s failed: %s", sample_id, attempt + 1, exc)
            if attempt < cfg.max_retries:
                sleep_seconds = min(60.0, 5.0 * (2**attempt)) + random.uniform(0.0, 3.0)
                time.sleep(sleep_seconds)
    if vlm_result is None:
        _write_failure(dirs["logs"], sample_id, "vlm", last_error or "unknown VLM failure")
        return False

    prompt = build_wan_prompt(task["anomaly_type"])
    generated_path = dirs["generated_images"] / f"{sample_id}.png"
    seedream_raw_path = dirs["seedream_raw_outputs"] / f"{sample_id}_seedream_raw.png"
    mask_path = dirs["masks"] / f"{sample_id}_obj_000001_mask.png"
    crop_path = dirs["crops"] / f"{sample_id}_obj_000001_crop.jpg"
    raw_diff_mask_path = dirs["seedream_raw_diff_masks"] / f"{sample_id}_raw_diff_mask.png"
    request_log_path = dirs["requests"] / f"{sample_id}_wan_request.json"
    response_log_path = dirs["responses"] / f"{sample_id}_wan_response.json"
    seedream_quality = None

    candidate_grids = _candidate_grids(vlm_result)
    grid_id = str(vlm_result["selected_grid"]).strip().upper()
    try:
        seedream_reference_paths: list[str] = []
        seedream_metadata: dict = {}
        red_box_bbox: tuple[int, int, int, int] | None = None
        floor_rejections: list[dict[str, str]] = []
        if cfg.seedream_mode in {"boxed_single_edit", "boxed_fusion"} and task["anomaly_type"] == "water_leak":
            for candidate_grid in candidate_grids:
                candidate_grid = str(candidate_grid).strip().upper()
                try:
                    candidate_grid_bbox = grid_id_to_bbox(candidate_grid, width, height)
                    candidate_red_box = _choose_seedream_water_leak_box(
                        original_path,
                        sample_id,
                        candidate_grid_bbox,
                        cfg.red_box_min_size,
                        cfg.red_box_max_size,
                    )
                    grid_id = candidate_grid
                    grid_bbox = candidate_grid_bbox
                    red_box_bbox = candidate_red_box
                    break
                except ValueError as exc:
                    if "no_visible_floor_region" not in str(exc):
                        raise
                    floor_rejections.append({"grid": candidate_grid, "reason": str(exc)})
            else:
                _write_skip(
                    dirs["logs"],
                    sample_id,
                    "no_visible_floor_region",
                    {
                        "vlm_result": vlm_result,
                        "candidate_grids": candidate_grids,
                        "floor_rejections": floor_rejections,
                    },
                )
                return None
        else:
            grid_bbox = grid_id_to_bbox(grid_id, width, height)

        expanded_bbox = expand_bbox(grid_bbox, width, height, cfg.edit_bbox_expand_ratio)
        request_image_path = original_path
        request_bbox = expanded_bbox
        if cfg.seedream_mode in {"boxed_single_edit", "boxed_fusion"}:
            if red_box_bbox is None:
                red_box_bbox = _choose_seedream_red_box(
                    sample_id,
                    grid_bbox,
                    cfg.red_box_min_size,
                    cfg.red_box_max_size,
                    task["anomaly_type"],
                )
            guide_path = dirs["seedream_guides"] / f"{sample_id}_red_box_guide.jpg"
            _draw_seedream_red_box(original_path, guide_path, red_box_bbox)
            request_image_path = guide_path
            request_bbox = red_box_bbox
            seedream_metadata = {
                "experimental_seedream": True,
                "seedream_mode": cfg.seedream_mode,
                "red_box_bbox": list(red_box_bbox),
                "red_box_guide_uri": relative_uri(guide_path),
            }
            if cfg.seedream_mode == "boxed_fusion":
                reference_path = _select_water_reference(cfg.water_reference_dir, sample_id)
                reference_copy_path = _seedream_reference_copy_path(
                    reference_path,
                    sample_id,
                    dirs["seedream_references"],
                )
                copy_file(reference_path, reference_copy_path)
                seedream_reference_paths = [str(reference_copy_path)]
                seedream_metadata["water_reference_uri"] = relative_uri(reference_copy_path)
                seedream_metadata["water_reference_source_uri"] = str(reference_path)
        elif cfg.seedream_mode == "single_image_edit":
            seedream_metadata = {
                "experimental_seedream": True,
                "seedream_mode": cfg.seedream_mode,
            }
        request_output_path = seedream_raw_path if cfg.seedream_mode else generated_path
        wan.edit_image_with_wan(
            image_path=str(request_image_path),
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            bbox=request_bbox,
            output_path=str(request_output_path),
            anomaly_type=task["anomaly_type"],
            request_log_path=str(request_log_path),
            response_log_path=str(response_log_path),
            seedream_mode=cfg.seedream_mode,
            seedream_reference_paths=seedream_reference_paths,
        )
        if cfg.seedream_mode:
            seedream_metadata.update(
                _prepare_seedream_raw_candidate_output(
                    original_path,
                    seedream_raw_path,
                    generated_path,
                    request_bbox,
                    water_mask_output_path=mask_path,
                    raw_diff_mask_output_path=raw_diff_mask_path,
                    anomaly_type=task["anomaly_type"],
                )
            )
        gen_width, gen_height = image_size(generated_path)
        if (gen_width, gen_height) != (width, height):
            logger.info("%s generated image size differs; resizing generated image to original dimensions", sample_id)
            _normalize_generated_size(generated_path, (width, height))
            gen_width, gen_height = image_size(generated_path)
        seedream_quality = None
        if cfg.seedream_mode and not cfg.dry_run:
            seedream_quality = _validate_seedream_experiment_output(
                original_path,
                generated_path,
                request_bbox,
                cfg.seedream_mode,
                seedream_metadata,
            )
            if not seedream_quality["passes_quality"]:
                raise RuntimeError(
                    "Seedream experiment quality failed: "
                    f"{seedream_quality['quality_reason']} "
                    f"(outside_change_ratio={seedream_quality.get('outside_change_ratio')}, "
                    f"outside_structure_change_ratio={seedream_quality.get('outside_structure_change_ratio')}, "
                    f"red_box_residual_ratio={seedream_quality.get('red_box_residual_ratio')}, "
                    f"seedream_mask_coverage_ratio={seedream_quality.get('seedream_mask_coverage_ratio')}, "
                    f"seedream_bbox_red_box_iou={seedream_quality.get('seedream_bbox_red_box_iou')})"
                )
        elif cfg.seedream_mode:
            seedream_quality = {
                "passes_quality": True,
                "quality_reason": "dry_run_not_evaluated",
                "outside_change_ratio": None,
                "outside_structure_change_ratio": None,
                "red_box_residual_ratio": None,
            }
        if cfg.seedream_mode:
            water_mask_bbox = seedream_metadata.get("seedream_water_mask_bbox")
            if not isinstance(water_mask_bbox, list) or len(water_mask_bbox) != 4:
                raise RuntimeError("Seedream water mask bbox missing after composition")
            diff = {
                "bbox": bbox_to_box_dict(tuple(int(value) for value in water_mask_bbox)),
                "area": int(seedream_metadata.get("seedream_water_mask_area") or 0),
                "mask_uri": str(mask_path),
                "status": "ok",
                "diff_method": "seedream_water_mask_within_selected_region",
            }
        else:
            diff = localize_change_bbox(
                str(original_path),
                str(generated_path),
                request_bbox,
                task["anomaly_type"],
                str(mask_path),
            )
        refined_bbox = _is_refined_final_bbox(diff["bbox"], request_bbox)
        if not refined_bbox:
            logger.warning(
                "%s localization produced coarse bbox matching the edit region; "
                "writing metadata for in-repo localizer postprocess",
                sample_id,
            )
        final_bbox_source = (
            "seedream_water_mask_within_selected_region"
            if cfg.seedream_mode
            else (
                "image_difference_within_selected_grid"
                if refined_bbox
                else "coarse_expanded_edit_bbox_requires_localizer_postprocess"
            )
        )
        crop_info = crop_anomaly(
            str(generated_path),
            diff["bbox"],
            str(crop_path),
            cfg.crop_expand_ratio,
        )
        generation_params = {
            "localization_pipeline": "vlm_grid_selection_then_image_edit_then_image_diff",
            "vlm_model": cfg.vlm_model,
            "image_generation_model": cfg.image_model,
            "grid_layout": cfg.grid_layout,
            "selected_grid": grid_id,
            "candidate_grids": candidate_grids,
            "grid_bbox": list(grid_bbox),
            "expanded_edit_bbox": list(expanded_bbox),
            "prompt_box": list(request_bbox),
            "final_bbox_source": final_bbox_source,
            "coarse_bbox_requires_postprocess": False if cfg.seedream_mode else not refined_bbox,
            "diff_method": diff["diff_method"],
            "mask_uri": relative_uri(mask_path),
            "vlm_selection": vlm_result,
            **seedream_metadata,
        }
        if seedream_quality is not None:
            generation_params["seedream_quality_gate"] = seedream_quality
        sample = build_autolabel_sample(
            task=task,
            generated_image_uri=relative_uri(generated_path),
            width=gen_width,
            height=gen_height,
            object_box=diff["bbox"],
            crop_info={**crop_info, "crop_uri": relative_uri(crop_path)},
            classification_labels=build_classification_labels(task["anomaly_type"]),
            generation_params=generation_params,
            generation_prompt=prompt,
            image_model=cfg.image_model,
            vlm_model=cfg.vlm_model,
        )
        validate_required_fields(sample)
        write_json(dirs["metadata"] / f"{sample_id}.json", sample)
        if failure_log.exists():
            failure_log.unlink()
        logger.info("%s succeeded with grid=%s final_bbox=%s", sample_id, grid_id, diff["bbox"])
        return True
    except Exception as exc:
        last_error = exc
        logger.warning("%s generation/localization failed for selected grid %s: %s", sample_id, grid_id, exc)

    failure_context = {
        "vlm_result": vlm_result,
        "candidate_grids": candidate_grids,
        "selected_grid": grid_id,
        **_failure_artifact_context(
            sample_id,
            dirs,
            generated_path,
            request_log_path,
            response_log_path,
            mask_path,
            crop_path,
            extra_artifact_paths={
                "seedream_raw_output_uri": seedream_raw_path,
                "seedream_raw_diff_mask_uri": raw_diff_mask_path,
            },
        ),
    }
    if seedream_quality is not None:
        failure_context["seedream_quality_gate"] = seedream_quality
    _write_failure(
        dirs["logs"],
        sample_id,
        "generation_or_diff",
        last_error or "unknown generation/diff failure",
        failure_context,
    )
    return False


def main() -> int:
    load_dotenv_if_available()
    cfg = parse_args()
    dirs = ensure_output_dirs(cfg.output_root)
    setup_logging(dirs["logs"])
    write_prompt_files("data/prompts")

    services = I2IServiceConfig.from_env()

    tasks = read_tasks_csv(cfg.tasks)
    if cfg.limit is not None:
        tasks = tasks[: cfg.limit]
    skipped = 0
    runnable_tasks = []
    if cfg.skip_existing:
        for task in tasks:
            metadata_path = dirs["metadata"] / f"{task.get('sample_id', '')}.json"
            if _has_valid_metadata(metadata_path):
                skipped += 1
                continue
            runnable_tasks.append(task)
    else:
        runnable_tasks = tasks

    ok = 0
    failed = 0
    floor_skipped = 0
    workers = max(1, cfg.workers)
    logger.info("processing %s tasks with workers=%s skipped_existing=%s", len(runnable_tasks), workers, skipped)

    def run_one(task: dict) -> tuple[str, bool | None, int, int]:
        vlm = QwenVLMClient(cfg.vlm_model, services.vlm, dry_run=cfg.dry_run)
        wan = WanImageClient(cfg.image_model, services.image, dry_run=cfg.dry_run)
        try:
            success = process_task(task, cfg, dirs, vlm, wan)
            return (
                task.get("sample_id", "unknown"),
                success,
                int(getattr(wan, "model_call_count", 0)),
                int(getattr(wan, "model_generated_count", 0)),
            )
        except (ValidationError, Exception) as exc:
            _write_failure(dirs["logs"], task.get("sample_id", "unknown"), "task", exc, {"task": task})
            return (
                task.get("sample_id", "unknown"),
                False,
                int(getattr(wan, "model_call_count", 0)),
                int(getattr(wan, "model_generated_count", 0)),
            )

    model_call_count = 0
    model_generated_count = 0
    if workers == 1:
        for task in runnable_tasks:
            _, success, calls, generated = run_one(task)
            model_call_count += calls
            model_generated_count += generated
            if success is True:
                ok += 1
            elif success is None:
                floor_skipped += 1
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, task) for task in runnable_tasks]
            completed = 0
            for future in as_completed(futures):
                sample_id, success, calls, generated = future.result()
                model_call_count += calls
                model_generated_count += generated
                completed += 1
                if success is True:
                    ok += 1
                elif success is None:
                    floor_skipped += 1
                else:
                    failed += 1
                logger.info(
                    "progress completed=%s/%s succeeded=%s failed=%s skipped=%s last=%s",
                    completed,
                    len(runnable_tasks),
                    ok,
                    failed,
                    skipped + floor_skipped,
                    sample_id,
                )

    final_generated_count = _count_files(dirs["generated_images"])
    debug_artifact_count = _count_files(dirs["debug"])
    summary = {
        "total": len(tasks),
        "processed": len(runnable_tasks),
        "succeeded": ok,
        "failed": failed,
        "skipped": skipped + floor_skipped,
        "skipped_existing": skipped,
        "skipped_no_floor_region": floor_skipped,
        "model_call_count": model_call_count,
        "model_generated_count": model_generated_count,
        "final_generated_count": final_generated_count,
        "debug_artifact_count": debug_artifact_count,
    }
    write_json(dirs["logs"] / "run_summary.json", summary)
    logger.info("run summary: %s", summary)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
