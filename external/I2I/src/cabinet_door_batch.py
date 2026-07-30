from __future__ import annotations

import csv
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from config import I2IServiceConfig
from qwen_vlm_client import QwenVLMClient
from utils import cv2_imread, cv2_imwrite, now_iso_shanghai, setup_logging, write_json
from wan_image_client import WanImageClient

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class CabinetDoorBatchConfig:
    image_dir: Path
    output_root: Path
    manifest: Path | None = None
    vlm_model: str = "qwen3.6-27b"
    image_model: str = "doubao-seedream-5-0-pro-260628"
    min_confidence: float = 0.70
    preview_max_side: int = 1280
    preview_jpeg_quality: int = 82
    max_outside_mean_abs_diff: float = 30.0
    max_outside_structure_change_ratio: float = 0.12
    dry_run: bool = False
    skip_existing: bool = False
    limit: int | None = None
    workers: int = 1


def ensure_cabinet_door_output_dirs(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    dirs = {
        "root": root,
        "generated_images": root / "generated_images",
        "metadata": root / "metadata",
        "debug": root / "debug",
        "vlm_previews": root / "debug" / "vlm_previews",
        "seedream_raw_outputs": root / "debug" / "seedream_raw_outputs",
        "crops": root / "debug" / "crops",
        "failed_generated_images": root / "debug" / "failed_generated_images",
        "logs": root / "logs",
        "qwen": root / "logs" / "qwen",
        "seedream_requests": root / "logs" / "seedream_requests",
        "seedream_responses": root / "logs" / "seedream_responses",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def list_source_images(image_dir: str | Path, manifest: str | Path | None = None) -> list[Path]:
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"image directory not found: {root}")

    manifest_path = Path(manifest) if manifest else root / "manifest.csv"
    images: list[Path] = []
    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = str(row.get("image_name") or row.get("image_uri") or "").strip()
                if not name:
                    continue
                candidate = root / Path(name).name
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                    images.append(candidate)
    if not images:
        images = sorted(
            path for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )

    deduped: list[Path] = []
    seen: set[str] = set()
    for image_path in images:
        key = str(image_path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(image_path)
    return deduped


def make_vlm_preview(
    source_path: str | Path,
    preview_path: str | Path,
    *,
    max_side: int = 1280,
    jpeg_quality: int = 82,
) -> tuple[int, int]:
    source_path = Path(source_path)
    preview_path = Path(preview_path)
    with Image.open(source_path) as source_image:
        image = source_image.convert("RGB")
        original_size = image.size
        if max(image.size) > max_side:
            scale = max_side / float(max(image.size))
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(preview_path, "JPEG", quality=jpeg_quality, optimize=True)
    return original_size


def expand_door_edit_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    hinge_side: str = "unknown",
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    door_width = max(1, x2 - x1)
    door_height = max(1, y2 - y1)
    left_margin = door_width
    right_margin = door_width
    if hinge_side == "left":
        left_margin = round(door_width * 0.35)
    elif hinge_side == "right":
        right_margin = round(door_width * 0.35)
    vertical_margin = round(door_height * 0.12)
    return (
        max(0, x1 - left_margin),
        max(0, y1 - vertical_margin),
        min(width, x2 + right_margin),
        min(height, y2 + vertical_margin),
    )


def _resize_generated_to_source(
    raw_path: str | Path,
    source_size: tuple[int, int],
    output_path: str | Path,
) -> None:
    with Image.open(raw_path) as raw_image:
        generated = raw_image.convert("RGB")
        if generated.size != source_size:
            generated = generated.resize(source_size, Image.Resampling.LANCZOS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated.save(output_path, "PNG")


def _bbox_from_change_mask(
    original: np.ndarray,
    generated: np.ndarray,
    allowed_bbox: tuple[int, int, int, int],
    fallback_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    gray_a = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
    delta = cv2.absdiff(gray_a, gray_b)
    mask = np.zeros(delta.shape, dtype=np.uint8)
    x1, y1, x2, y2 = allowed_bbox
    roi = delta[y1:y2, x1:x2]
    if roi.size == 0:
        return fallback_bbox
    roi_mask = (cv2.GaussianBlur(roi, (5, 5), 0) >= 18).astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    mask[y1:y2, x1:x2] = roi_mask
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 100:
        return fallback_bbox
    bx, by, bw, bh = cv2.boundingRect(points)
    return bx, by, bx + bw, by + bh


def evaluate_cabinet_door_edit(
    source_path: str | Path,
    generated_path: str | Path,
    selected_bbox: tuple[int, int, int, int],
    *,
    hinge_side: str = "unknown",
    max_outside_mean_abs_diff: float = 30.0,
    max_outside_structure_change_ratio: float = 0.12,
) -> dict[str, Any]:
    original = cv2_imread(source_path, cv2.IMREAD_COLOR)
    generated = cv2_imread(generated_path, cv2.IMREAD_COLOR)
    if original is None or generated is None:
        raise ValueError("failed to read source or generated image for cabinet-door quality evaluation")
    if generated.shape[:2] != original.shape[:2]:
        raise ValueError(f"generated size does not match source: {generated.shape[:2]} != {original.shape[:2]}")

    height, width = original.shape[:2]
    allowed_bbox = expand_door_edit_bbox(selected_bbox, (width, height), hinge_side=hinge_side)
    allowed_mask = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = allowed_bbox
    allowed_mask[y1:y2, x1:x2] = True
    outside = ~allowed_mask

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    generated_gray = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
    abs_diff = cv2.absdiff(original_gray, generated_gray)
    outside_mean_abs_diff = float(abs_diff[outside].mean()) if outside.any() else 0.0
    outside_change_ratio = float((abs_diff[outside] >= 20).mean()) if outside.any() else 0.0

    original_edges = cv2.Canny(original_gray, 80, 160)
    generated_edges = cv2.Canny(generated_gray, 80, 160)
    edge_change = cv2.bitwise_xor(original_edges, generated_edges) > 0
    outside_structure_change_ratio = float(edge_change[outside].mean()) if outside.any() else 0.0
    passes = (
        outside_mean_abs_diff <= max_outside_mean_abs_diff
        and outside_structure_change_ratio <= max_outside_structure_change_ratio
    )
    changed_bbox = _bbox_from_change_mask(original, generated, allowed_bbox, selected_bbox)
    return {
        "passes_quality": passes,
        "quality_reason": "success" if passes else "seedream_background_structure_drift",
        "selected_closed_door_bbox": list(selected_bbox),
        "allowed_edit_bbox": list(allowed_bbox),
        "generated_change_bbox": list(changed_bbox),
        "outside_mean_abs_diff": outside_mean_abs_diff,
        "outside_change_ratio": outside_change_ratio,
        "outside_structure_change_ratio": outside_structure_change_ratio,
        "thresholds": {
            "max_outside_mean_abs_diff": max_outside_mean_abs_diff,
            "max_outside_structure_change_ratio": max_outside_structure_change_ratio,
        },
    }


def _save_crop(
    image_path: str | Path,
    bbox: tuple[int, int, int, int],
    crop_path: str | Path,
    *,
    expand_ratio: float = 0.05,
) -> None:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        x1, y1, x2, y2 = bbox
        pad_x = round((x2 - x1) * expand_ratio)
        pad_y = round((y2 - y1) * expand_ratio)
        crop_bbox = (max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y))
        crop = rgb.crop(crop_bbox)
    crop_path = Path(crop_path)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(crop_path, "JPEG", quality=94)


def _metadata_base(image_path: Path, config: CabinetDoorBatchConfig) -> dict[str, Any]:
    return {
        "sample_id": image_path.stem,
        "source_image": str(image_path),
        "task": "cabinet_door_open",
        "architecture": "qwen_bbox_then_seedream_single_edit",
        "vlm_model": config.vlm_model,
        "image_model": config.image_model,
        "created_at": now_iso_shanghai(),
    }


def process_cabinet_door_image(
    image_path: Path,
    config: CabinetDoorBatchConfig,
    dirs: dict[str, Path],
    *,
    qwen_client: QwenVLMClient | None = None,
    image_client: WanImageClient | None = None,
) -> dict[str, Any]:
    sample_id = image_path.stem
    metadata_path = dirs["metadata"] / f"{sample_id}.json"
    if config.skip_existing and metadata_path.is_file():
        return {"sample_id": sample_id, "status": "skipped_existing", "qwen_call_count": 0, "seedream_call_count": 0}

    metadata = _metadata_base(image_path, config)
    preview_path = dirs["vlm_previews"] / f"{sample_id}.jpg"
    raw_path = dirs["seedream_raw_outputs"] / f"{sample_id}.png"
    final_path = dirs["generated_images"] / f"{sample_id}.png"
    crop_path = dirs["crops"] / f"{sample_id}_open_door.jpg"
    qwen_log = dirs["qwen"] / f"{sample_id}.json"
    seedream_request_log = dirs["seedream_requests"] / f"{sample_id}.json"
    seedream_response_log = dirs["seedream_responses"] / f"{sample_id}.json"
    qwen_call_count = 0
    seedream_call_count = 0

    try:
        source_size = make_vlm_preview(
            image_path,
            preview_path,
            max_side=config.preview_max_side,
            jpeg_quality=config.preview_jpeg_quality,
        )
        if config.dry_run:
            selection = QwenVLMClient(
                config.vlm_model,
                I2IServiceConfig.from_env().vlm,
                dry_run=True,
            ).select_cabinet_door_bbox(str(preview_path), source_size, str(qwen_log))
        else:
            qwen = qwen_client or QwenVLMClient(config.vlm_model, I2IServiceConfig.from_env().vlm)
            qwen_call_count = 1
            selection = qwen.select_cabinet_door_bbox(
                str(preview_path),
                source_size,
                str(qwen_log),
                min_confidence=config.min_confidence,
            )
        metadata["selection"] = selection
        metadata["qwen_call_count"] = qwen_call_count
        if selection["status"] != "selected":
            metadata.update(
                {
                    "status": "skipped",
                    "stage": "qwen_bbox_selection",
                    "skip_reason": selection.get("reason") or "no suitable closed cabinet door",
                    "seedream_call_count": 0,
                }
            )
            write_json(metadata_path, metadata)
            return metadata

        selected_bbox = tuple(int(value) for value in selection["bbox"])
        if config.dry_run:
            shutil.copy2(image_path, raw_path)
        else:
            seedream = image_client or WanImageClient(config.image_model, I2IServiceConfig.from_env().image)
            seedream_call_count = 1
            seedream.edit_image_with_wan(
                str(image_path),
                "Open the selected closed equipment cabinet door.",
                "Do not change any other door, cabinet, object, camera, timestamp, or background.",
                selected_bbox,
                str(raw_path),
                "cabinet_door_open",
                request_log_path=str(seedream_request_log),
                response_log_path=str(seedream_response_log),
                seedream_mode="cabinet_door_open",
            )
        _resize_generated_to_source(raw_path, source_size, final_path)
        quality = evaluate_cabinet_door_edit(
            image_path,
            final_path,
            selected_bbox,
            hinge_side=str(selection.get("hinge_side") or "unknown"),
            max_outside_mean_abs_diff=config.max_outside_mean_abs_diff,
            max_outside_structure_change_ratio=config.max_outside_structure_change_ratio,
        )
        generated_change_bbox = tuple(int(value) for value in quality["generated_change_bbox"])
        _save_crop(final_path, generated_change_bbox, crop_path)
        if not quality["passes_quality"] and not config.dry_run:
            rejected_path = dirs["failed_generated_images"] / final_path.name
            shutil.move(str(final_path), rejected_path)
            final_uri: str | None = None
            rejected_uri: str | None = str(rejected_path)
            status = "rejected"
        else:
            final_uri = str(final_path)
            rejected_uri = None
            status = "accepted"

        metadata.update(
            {
                "status": status,
                "stage": "complete" if status == "accepted" else "quality_gate",
                "seedream_call_count": seedream_call_count,
                "model_generated_count": 0 if config.dry_run else 1,
                "generated_image": final_uri,
                "rejected_image": rejected_uri,
                "seedream_raw_image": str(raw_path),
                "crop_image": str(crop_path),
                "quality": quality,
            }
        )
        write_json(metadata_path, metadata)
        return metadata
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "stage": "generation",
                "error": str(exc),
                "qwen_call_count": qwen_call_count,
                "seedream_call_count": seedream_call_count,
            }
        )
        write_json(metadata_path, metadata)
        logger.exception("%s failed: %s", sample_id, exc)
        return metadata


def run_cabinet_door_batch(
    config: CabinetDoorBatchConfig,
    *,
    qwen_factory: Callable[[], QwenVLMClient] | None = None,
    image_factory: Callable[[], WanImageClient] | None = None,
) -> dict[str, Any]:
    dirs = ensure_cabinet_door_output_dirs(config.output_root)
    setup_logging(dirs["logs"])
    images = list_source_images(config.image_dir, config.manifest)
    if config.limit is not None:
        images = images[: max(0, config.limit)]
    if not images:
        raise ValueError(f"no source images found under {config.image_dir}")

    if not config.dry_run:
        services = I2IServiceConfig.from_env()
        if not services.vlm.api_key:
            raise RuntimeError("QWEN397B_API_KEY is required")
        if not services.image.api_key:
            raise RuntimeError("ARK_API_KEY is required")

    def run_one(image_path: Path) -> dict[str, Any]:
        return process_cabinet_door_image(
            image_path,
            config,
            dirs,
            qwen_client=qwen_factory() if qwen_factory else None,
            image_client=image_factory() if image_factory else None,
        )

    results: list[dict[str, Any]] = []
    workers = max(1, config.workers)
    if workers == 1:
        for index, image_path in enumerate(images, start=1):
            logger.info("processing %s/%s: %s", index, len(images), image_path.name)
            results.append(run_one(image_path))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, image_path): image_path for image_path in images}
            for future in as_completed(futures):
                results.append(future.result())

    summary = {
        "task": "cabinet_door_open",
        "total": len(images),
        "accepted": sum(result.get("status") == "accepted" for result in results),
        "rejected": sum(result.get("status") == "rejected" for result in results),
        "skipped": sum(result.get("status") == "skipped" for result in results),
        "skipped_existing": sum(result.get("status") == "skipped_existing" for result in results),
        "failed": sum(result.get("status") == "failed" for result in results),
        "qwen_call_count": sum(int(result.get("qwen_call_count", 0)) for result in results),
        "seedream_call_count": sum(int(result.get("seedream_call_count", 0)) for result in results),
        "model_generated_count": sum(int(result.get("model_generated_count", 0)) for result in results),
        "dry_run": config.dry_run,
        "output_root": str(config.output_root),
        "completed_at": now_iso_shanghai(),
    }
    write_json(dirs["logs"] / "run_summary.json", summary)
    logger.info("run summary: %s", summary)
    return summary
