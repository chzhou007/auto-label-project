from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import logging
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw

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
    parser.add_argument("--image-model", default="doubao-seedream-5.0-lite")
    parser.add_argument("--grid-layout", default="4x4", choices=["4x4"])
    parser.add_argument("--edit-bbox-expand-ratio", type=float, default=0.20)
    parser.add_argument("--crop-expand-ratio", type=float, default=0.10)
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


def _failure_artifact_context(
    sample_id: str,
    dirs: dict[str, Path],
    generated_path: Path,
    request_log_path: Path,
    response_log_path: Path,
    mask_path: Path,
    crop_path: Path,
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

    x_stop = max(gx1, gx2 - box_size)
    y_stop = max(gy1, gy2 - box_size)
    y_start = min(gy1 + int(grid_height * 0.45), y_stop)
    step = max(8, box_size // 5)
    best_box = fallback
    best_score = float("-inf")
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
            red_yellow_green = (
                ((crop[:, :, 0] > 0.45) & (crop[:, :, 0] > crop[:, :, 1] * 1.15))
                | ((crop[:, :, 0] > 0.45) & (crop[:, :, 1] > 0.35) & (crop[:, :, 2] < 0.25))
                | ((crop[:, :, 1] > 0.35) & (crop[:, :, 1] > crop[:, :, 0] * 1.15))
            )
            saturated_equipment_ratio = float((red_yellow_green & (saturation > 0.20)).mean())
            too_dark_ratio = float((brightness < 0.18).mean())
            y_bias = (y1 - gy1) / max(1, grid_height - box_size)
            score = (1.8 * (1.0 - mean_sat)) + (0.8 * mean_brightness) + (0.7 * y_bias)
            score -= 2.5 * saturated_equipment_ratio
            score -= 0.8 * too_dark_ratio
            if score > best_score:
                best_score = score
                best_box = (x1, y1, x2, y2)
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
) -> dict:
    result = {
        "passes_quality": True,
        "quality_reason": None,
        "outside_change_ratio": _outside_change_ratio(original_path, generated_path, allowed_bbox),
        "outside_structure_change_ratio": _outside_structure_change_ratio(original_path, generated_path, allowed_bbox),
        "red_box_residual_ratio": 0.0,
    }
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
    if reasons:
        result["passes_quality"] = False
        result["quality_reason"] = ",".join(reasons)
    return result


def process_task(task: dict, cfg: PipelineConfig, dirs: dict[str, Path], vlm: QwenVLMClient, wan: WanImageClient) -> bool:
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
    mask_path = dirs["masks"] / f"{sample_id}_obj_000001_mask.png"
    crop_path = dirs["crops"] / f"{sample_id}_obj_000001_crop.jpg"
    request_log_path = dirs["requests"] / f"{sample_id}_wan_request.json"
    response_log_path = dirs["responses"] / f"{sample_id}_wan_response.json"
    seedream_quality = None

    candidate_grids = _candidate_grids(vlm_result)
    grid_id = str(vlm_result["selected_grid"]).strip().upper()
    try:
        grid_bbox = grid_id_to_bbox(grid_id, width, height)
        expanded_bbox = expand_bbox(grid_bbox, width, height, cfg.edit_bbox_expand_ratio)
        request_image_path = original_path
        request_bbox = expanded_bbox
        seedream_reference_paths: list[str] = []
        seedream_metadata: dict = {}
        if cfg.seedream_mode in {"boxed_single_edit", "boxed_fusion"}:
            if task["anomaly_type"] == "water_leak":
                red_box_bbox = _choose_seedream_water_leak_box(
                    original_path,
                    sample_id,
                    grid_bbox,
                    cfg.red_box_min_size,
                    cfg.red_box_max_size,
                )
            else:
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
                reference_copy_path = dirs["seedream_references"] / f"{sample_id}_{reference_path.name}"
                copy_file(reference_path, reference_copy_path)
                seedream_reference_paths = [str(reference_copy_path)]
                seedream_metadata["water_reference_uri"] = relative_uri(reference_copy_path)
        elif cfg.seedream_mode == "single_image_edit":
            seedream_metadata = {
                "experimental_seedream": True,
                "seedream_mode": cfg.seedream_mode,
            }
        wan.edit_image_with_wan(
            image_path=str(request_image_path),
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            bbox=request_bbox,
            output_path=str(generated_path),
            anomaly_type=task["anomaly_type"],
            request_log_path=str(request_log_path),
            response_log_path=str(response_log_path),
            seedream_mode=cfg.seedream_mode,
            seedream_reference_paths=seedream_reference_paths,
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
            )
            if not seedream_quality["passes_quality"]:
                raise RuntimeError(
                    "Seedream experiment quality failed: "
                    f"{seedream_quality['quality_reason']} "
                    f"(outside_change_ratio={seedream_quality.get('outside_change_ratio')}, "
                    f"outside_structure_change_ratio={seedream_quality.get('outside_structure_change_ratio')}, "
                    f"red_box_residual_ratio={seedream_quality.get('red_box_residual_ratio')})"
                )
        elif cfg.seedream_mode:
            seedream_quality = {
                "passes_quality": True,
                "quality_reason": "dry_run_not_evaluated",
                "outside_change_ratio": None,
                "outside_structure_change_ratio": None,
                "red_box_residual_ratio": None,
            }
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
                "%s diff localization produced coarse bbox matching expanded_edit_bbox; "
                "writing metadata for in-repo localizer postprocess",
                sample_id,
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
            "final_bbox_source": (
                "image_difference_within_selected_grid"
                if refined_bbox
                else "coarse_expanded_edit_bbox_requires_localizer_postprocess"
            ),
            "coarse_bbox_requires_postprocess": not refined_bbox,
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
    workers = max(1, cfg.workers)
    logger.info("processing %s tasks with workers=%s skipped_existing=%s", len(runnable_tasks), workers, skipped)

    def run_one(task: dict) -> tuple[str, bool, int, int]:
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
            if success:
                ok += 1
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
                if success:
                    ok += 1
                else:
                    failed += 1
                logger.info(
                    "progress completed=%s/%s succeeded=%s failed=%s skipped=%s last=%s",
                    completed,
                    len(runnable_tasks),
                    ok,
                    failed,
                    skipped,
                    sample_id,
                )

    final_generated_count = _count_files(dirs["generated_images"])
    debug_artifact_count = _count_files(dirs["debug"])
    summary = {
        "total": len(tasks),
        "processed": len(runnable_tasks),
        "succeeded": ok,
        "failed": failed,
        "skipped": skipped,
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
