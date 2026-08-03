from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from config import I2IServiceConfig, ModelServiceConfig
from qwen_vlm_client import QwenVLMClient
from utils import cv2_imread, cv2_imwrite, now_iso_shanghai, read_json, setup_logging, write_json
from wan_image_client import WanImageClient, validate_api_key_for_http_header

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_ARK_RESPONSES_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/responses"
DEFAULT_ARK_VLM_MODEL = "doubao-seed-2-1-pro-260628"
DEFAULT_ELECTRICAL_ROOM_TYPE_REGEX = (
    r"高压配电室|低压配电室|发电机配电房|电池室|精密空调间|冷冻站|水设备间|蓄水泵房"
)


@dataclass(frozen=True)
class CabinetDoorBatchConfig:
    image_dir: Path
    output_root: Path
    manifest: Path | None = None
    priority_image_dir: Path | None = None
    priority_only: bool = False
    reviewed_all_close: bool = False
    selector_backend: str = "segmentation_candidates"
    candidate_predictions_csv: Path | None = None
    image_predictions_csv: Path | None = None
    allowed_room_type_regex: str = DEFAULT_ELECTRICAL_ROOM_TYPE_REGEX
    minimum_segmentation_confidence: float = 0.50
    maximum_closed_probability: float = 0.20
    maximum_not_door_probability: float = 0.05
    minimum_bbox_area_ratio: float = 0.002
    maximum_bbox_area_ratio: float = 0.10
    minimum_bbox_short_side: int = 50
    maximum_bbox_edge_density: float = 0.12
    maximum_dark_neutral_ratio: float = 0.45
    maximum_existing_open_overlap_ratio: float = 0.05
    vlm_provider: str = "volcengine_ark_responses"
    vlm_model: str = DEFAULT_ARK_VLM_MODEL
    vlm_endpoint: str | None = None
    vlm_api_key_env: str = "ARK_API_KEY"
    vlm_timeout_seconds: float = 300.0
    vlm_max_retries: int = 0
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


def _csv_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _segmentation_bbox(
    row: dict[str, str],
    image_size: tuple[int, int],
) -> tuple[tuple[int, int, int, int], str]:
    width, height = image_size
    polygon_text = str(row.get("polygon_json") or "").strip()
    if polygon_text:
        try:
            polygon = json.loads(polygon_text)
            points = [
                point
                for point in polygon
                if isinstance(point, (list, tuple))
                and len(point) >= 2
                and all(isinstance(value, (int, float)) for value in point[:2])
            ]
            if len(points) >= 3:
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                bbox = (
                    max(0, int(np.floor(min(xs)))),
                    max(0, int(np.floor(min(ys)))),
                    min(width, int(np.ceil(max(xs))) + 1),
                    min(height, int(np.ceil(max(ys))) + 1),
                )
                return bbox, "polygon_bounds"
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return (
        (
            max(0, int(float(row["bbox_x1"]))),
            max(0, int(float(row["bbox_y1"]))),
            min(width, int(float(row["bbox_x2"]))),
            min(height, int(float(row["bbox_y2"]))),
        ),
        "candidate_crop_bbox",
    )


def _intersection_over_second_bbox(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = float((x2 - x1) * (y2 - y1))
    second_area = float(max(1, (second[2] - second[0]) * (second[3] - second[1])))
    return intersection / second_area


class SegmentationCandidateSelector:
    def __init__(self, config: CabinetDoorBatchConfig):
        self.config = config
        run_root = config.image_dir.parent
        self.candidate_predictions_csv = config.candidate_predictions_csv or run_root / "candidate_predictions.csv"
        self.image_predictions_csv = config.image_predictions_csv or run_root / "image_predictions.csv"
        if not self.candidate_predictions_csv.is_file():
            raise FileNotFoundError(f"candidate predictions CSV not found: {self.candidate_predictions_csv}")
        if not self.image_predictions_csv.is_file():
            raise FileNotFoundError(f"image predictions CSV not found: {self.image_predictions_csv}")
        try:
            self.allowed_room_pattern = re.compile(config.allowed_room_type_regex)
        except re.error as exc:
            raise ValueError(f"invalid allowed_room_type_regex: {config.allowed_room_type_regex}") from exc
        self.image_context = self._load_image_context()
        self.candidates_by_image = self._load_candidates()

    def _load_image_context(self) -> dict[str, dict[str, str]]:
        context: dict[str, dict[str, str]] = {}
        with self.image_predictions_csv.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                image_name = str(row.get("image_name") or "").strip()
                if image_name:
                    context[image_name] = {key: str(value or "").strip() for key, value in row.items()}
        return context

    def _load_candidates(self) -> dict[str, list[dict[str, str]]]:
        candidates: dict[str, list[dict[str, str]]] = {}
        with self.candidate_predictions_csv.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                image_name = str(row.get("image_name") or "").strip()
                if image_name:
                    candidates.setdefault(image_name, []).append(
                        {key: str(value or "").strip() for key, value in row.items()}
                    )
        return candidates

    def select(self, image_path: Path, image_size: tuple[int, int]) -> dict[str, Any]:
        image_name = image_path.name
        context = self.image_context.get(image_name, {})
        room_type = context.get("room_type", "")
        if not self.config.reviewed_all_close and not self.allowed_room_pattern.search(room_type):
            return {
                "status": "skipped",
                "reason": f"room_type_not_allowed:{room_type or 'unknown'}",
                "selection_backend": "segmentation_candidates",
                "bbox": None,
                "room_type": room_type,
            }

        width, height = image_size
        image_area = float(width * height)
        valid: list[dict[str, Any]] = []
        rejection_counts: dict[str, int] = {}
        source_bgr: np.ndarray | None = None
        source_gray: np.ndarray | None = None
        all_rows = self.candidates_by_image.get(image_name, [])
        if self.config.reviewed_all_close:
            reviewed_rows = [
                row
                for row in all_rows
                if _csv_bool(row.get("fused_open")) or _csv_bool(row.get("passed_state_gate"))
            ]
            rows = reviewed_rows or all_rows
        else:
            rows = all_rows
        known_open_bboxes: list[tuple[int, int, int, int]] = []
        for row in rows if not self.config.reviewed_all_close else []:
            try:
                open_probability = float(row.get("state_open_probability", 0.0))
                is_known_open = _csv_bool(row.get("fused_open")) or (
                    _csv_bool(row.get("passed_state_gate")) and open_probability >= 0.50
                )
                if is_known_open:
                    known_open_bboxes.append(_segmentation_bbox(row, image_size)[0])
            except (KeyError, TypeError, ValueError):
                continue

        def reject(reason: str) -> None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        for row in rows:
            if (
                not self.config.reviewed_all_close
                and not _csv_bool(row.get("passed_segmentation_gate"))
            ):
                reject("segmentation_gate")
                continue
            deduplication_value = row.get("passed_candidate_deduplication")
            if (
                deduplication_value is not None
                and str(deduplication_value).strip()
                and not _csv_bool(deduplication_value)
            ):
                reject("deduplication_gate")
                continue
            if (
                not self.config.reviewed_all_close
                and not _csv_bool(row.get("passed_not_door_gate"))
            ):
                reject("not_door_gate")
                continue
            try:
                segmentation_confidence = float(row.get("segmentation_confidence", 0.0))
                open_probability = float(row.get("state_open_probability", 1.0))
                not_door_probability = float(row.get("gate_not_door_probability", 1.0))
                bbox, bbox_source = _segmentation_bbox(row, image_size)
            except (KeyError, TypeError, ValueError):
                reject("invalid_values")
                continue
            if (
                not self.config.reviewed_all_close
                and segmentation_confidence < self.config.minimum_segmentation_confidence
            ):
                reject("low_segmentation_confidence")
                continue
            if (
                not self.config.reviewed_all_close
                and open_probability > self.config.maximum_closed_probability
            ):
                reject("not_closed")
                continue
            if (
                not self.config.reviewed_all_close
                and not_door_probability > self.config.maximum_not_door_probability
            ):
                reject("not_door_probability")
                continue
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            if bbox_width <= 0 or bbox_height <= 0:
                reject("empty_bbox")
                continue
            area_ratio = (bbox_width * bbox_height) / image_area
            if (
                not self.config.reviewed_all_close
                and min(bbox_width, bbox_height) < self.config.minimum_bbox_short_side
            ):
                reject("bbox_too_small")
                continue
            if (
                not self.config.reviewed_all_close
                and area_ratio < self.config.minimum_bbox_area_ratio
            ):
                reject("bbox_area_too_small")
                continue
            if (
                not self.config.reviewed_all_close
                and area_ratio > self.config.maximum_bbox_area_ratio
            ):
                reject("bbox_area_too_large")
                continue
            aspect_ratio = bbox_width / float(bbox_height)
            if (
                not self.config.reviewed_all_close
                and (aspect_ratio < 0.20 or aspect_ratio > 4.0)
            ):
                reject("bbox_aspect_invalid")
                continue
            if source_gray is None:
                source_bgr = cv2_imread(image_path, cv2.IMREAD_COLOR)
                if source_bgr is None:
                    reject("source_image_unreadable")
                    continue
                source_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
            x1, y1, x2, y2 = bbox
            roi = source_gray[y1:y2, x1:x2]
            if roi.size == 0:
                reject("empty_bbox")
                continue
            edge_density = float((cv2.Canny(roi, 80, 160) > 0).mean())
            if (
                not self.config.reviewed_all_close
                and edge_density > self.config.maximum_bbox_edge_density
            ):
                reject("bbox_texture_too_high")
                continue
            color_roi = source_bgr[y1:y2, x1:x2]
            saturation = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)[:, :, 1]
            dark_neutral_ratio = float(((roi < 60) & (saturation < 60)).mean())
            if (
                not self.config.reviewed_all_close
                and dark_neutral_ratio > self.config.maximum_dark_neutral_ratio
            ):
                reject("dark_neutral_rack_like")
                continue
            context_bbox = expand_door_edit_bbox(bbox, image_size)
            existing_open_overlap_ratio = max(
                (
                    _intersection_over_second_bbox(context_bbox, open_bbox)
                    for open_bbox in known_open_bboxes
                ),
                default=0.0,
            )
            if (
                not self.config.reviewed_all_close
                and existing_open_overlap_ratio > self.config.maximum_existing_open_overlap_ratio
            ):
                reject("existing_open_door_in_context")
                continue
            score = segmentation_confidence * (1.0 - not_door_probability)
            if not self.config.reviewed_all_close:
                score *= 1.0 - open_probability
            valid.append(
                {
                    "status": "selected",
                    "selection_backend": "segmentation_candidates",
                    "bbox": list(bbox),
                    "bbox_source": bbox_source,
                    "confidence": score,
                    "segmentation_confidence": segmentation_confidence,
                    "state_open_probability": open_probability,
                    "gate_not_door_probability": not_door_probability,
                    "candidate_index": int(row.get("candidate_index", 0) or 0),
                    "polygon_json": row.get("polygon_json") or None,
                    "room": context.get("room") or row.get("room"),
                    "room_type": room_type,
                    "camera_position": context.get("camera_position") or row.get("camera_position"),
                    "bbox_area_ratio": area_ratio,
                    "bbox_edge_density": edge_density,
                    "dark_neutral_ratio": dark_neutral_ratio,
                    "existing_open_overlap_ratio": existing_open_overlap_ratio,
                    "reason": (
                        "manually reviewed close target from false-positive open candidate"
                        if self.config.reviewed_all_close
                        else "closed electrical equipment-door candidate from segmentation/classification"
                    ),
                    "manual_close_override": self.config.reviewed_all_close,
                    "hinge_side": "unknown",
                }
            )
        if not valid:
            return {
                "status": "skipped",
                "reason": "no_eligible_closed_electrical_cabinet_candidate",
                "selection_backend": "segmentation_candidates",
                "bbox": None,
                "room_type": room_type,
                "candidate_count": len(all_rows),
                "rejection_counts": rejection_counts,
            }
        valid.sort(
            key=lambda item: (
                item["confidence"],
                item["segmentation_confidence"],
                -item["bbox_area_ratio"],
                -item["candidate_index"],
            ),
            reverse=True,
        )
        result = valid[0]
        result["eligible_candidate_count"] = len(valid)
        result["candidate_count"] = len(all_rows)
        result["rejection_counts"] = rejection_counts
        return result


def build_cabinet_door_selector_config(config: CabinetDoorBatchConfig) -> ModelServiceConfig:
    provider = config.vlm_provider.strip().lower()
    if provider in {"volcengine_ark_responses", "ark_responses", "ark-responses"}:
        api_key_env = config.vlm_api_key_env or "ARK_API_KEY"
        endpoint = (
            config.vlm_endpoint
            or os.getenv("ARK_VLM_RESPONSES_URL")
            or os.getenv("ARK_RESPONSES_URL")
            or DEFAULT_ARK_RESPONSES_ENDPOINT
        )
        return ModelServiceConfig(
            api_key=os.getenv(api_key_env) or os.getenv("ARK_API_KEY"),
            provider="volcengine_ark_responses",
            endpoint=endpoint,
            api_key_env=api_key_env,
            endpoint_env="ARK_VLM_RESPONSES_URL",
        )
    if provider in {"qwen_openai_compatible", "openai_compatible", "qwen"}:
        legacy = I2IServiceConfig.from_env().vlm
        return ModelServiceConfig(
            api_key=legacy.api_key,
            provider="openai_compatible",
            endpoint=config.vlm_endpoint or legacy.endpoint,
            api_key_env=legacy.api_key_env,
            endpoint_env=legacy.endpoint_env,
        )
    raise ValueError(
        "unsupported VLM provider; expected volcengine_ark_responses or qwen_openai_compatible, "
        f"got {config.vlm_provider!r}"
    )


def ensure_cabinet_door_output_dirs(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    dirs = {
        "root": root,
        "generated_images": root / "generated_images",
        "close_images": root / "pairs" / "close",
        "open_images": root / "pairs" / "open",
        "pair_annotations": root / "pairs" / "annotations",
        "metadata": root / "metadata",
        "debug": root / "debug",
        "vlm_previews": root / "debug" / "vlm_previews",
        "seedream_input_crops": root / "debug" / "seedream_input_crops",
        "selection_overlays": root / "debug" / "selection_overlays",
        "seedream_raw_outputs": root / "debug" / "seedream_raw_outputs",
        "seedream_open_candidates": root / "debug" / "seedream_open_candidates",
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


def prioritize_source_images(
    images: list[Path],
    priority_image_dir: str | Path | None,
    *,
    priority_only: bool = False,
) -> tuple[list[Path], int]:
    if priority_image_dir is None:
        return images, 0
    priority_root = Path(priority_image_dir)
    if not priority_root.is_dir():
        raise FileNotFoundError(f"priority image directory not found: {priority_root}")
    priority_names = {
        path.name
        for path in priority_root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    }
    prioritized = [image for image in images if image.name in priority_names]
    if priority_only:
        return prioritized, len(prioritized)
    remaining = [image for image in images if image.name not in priority_names]
    return prioritized + remaining, len(prioritized)


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
    margin_scale: float = 1.0,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    door_width = max(1, x2 - x1)
    door_height = max(1, y2 - y1)
    left_margin = round(door_width * margin_scale)
    right_margin = round(door_width * margin_scale)
    if hinge_side == "left":
        left_margin = round(door_width * 0.35 * margin_scale)
    elif hinge_side == "right":
        right_margin = round(door_width * 0.35 * margin_scale)
    vertical_margin = round(door_height * 0.12 * margin_scale)
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
    allowed_bbox = expand_door_edit_bbox(
        selected_bbox,
        (width, height),
        hinge_side=hinge_side,
        margin_scale=0.75,
    )
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


def _save_exact_crop(
    image_path: str | Path,
    bbox: tuple[int, int, int, int],
    output_path: str | Path,
) -> tuple[int, int]:
    with Image.open(image_path) as image:
        crop = image.convert("RGB").crop(bbox)
        crop_size = crop.size
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, "PNG")
    return crop_size


def _bbox_to_crop_coordinates(
    bbox: tuple[int, int, int, int],
    crop_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    crop_x1, crop_y1, _crop_x2, _crop_y2 = crop_bbox
    x1, y1, x2, y2 = bbox
    return x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1


def _bbox_to_source_coordinates(
    bbox: tuple[int, int, int, int],
    crop_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    crop_x1, crop_y1, _crop_x2, _crop_y2 = crop_bbox
    x1, y1, x2, y2 = bbox
    return x1 + crop_x1, y1 + crop_y1, x2 + crop_x1, y2 + crop_y1


def _save_selection_overlay(
    image_path: str | Path,
    bbox: tuple[int, int, int, int],
    output_path: str | Path,
    selection: dict[str, Any],
) -> None:
    image = cv2_imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read source image for selection overlay: {image_path}")
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 3)
    label = (
        f"closed electrical door conf={float(selection.get('segmentation_confidence', 0.0)):.3f} "
        f"open_p={float(selection.get('state_open_probability', 0.0)):.3f}"
    )
    cv2.putText(
        image,
        label,
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )
    cv2_imwrite(output_path, image)


def _metadata_base(image_path: Path, config: CabinetDoorBatchConfig) -> dict[str, Any]:
    segmentation_backend = config.selector_backend == "segmentation_candidates"
    return {
        "sample_id": image_path.stem,
        "pair_id": image_path.stem,
        "source_image": str(image_path),
        "task": "cabinet_door_open",
        "architecture": (
            "segmentation_closed_door_context_crop_then_seedream_edit"
            if segmentation_backend
            else "configurable_vlm_context_crop_then_seedream_edit"
        ),
        "selector_backend": config.selector_backend,
        "reviewed_all_close": config.reviewed_all_close,
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
    preselected_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_id = image_path.stem
    metadata_path = dirs["metadata"] / f"{sample_id}.json"
    if config.skip_existing and metadata_path.is_file():
        previous = read_json(metadata_path)
        if previous.get("status") in {"accepted", "rejected", "skipped"}:
            return {
                "sample_id": sample_id,
                "status": "skipped_existing",
                "previous_status": previous.get("status"),
                "qwen_call_count": 0,
                "seedream_call_count": 0,
            }

    metadata = _metadata_base(image_path, config)
    preview_path = dirs["vlm_previews"] / f"{sample_id}.jpg"
    seedream_input_path = dirs["seedream_input_crops"] / f"{sample_id}.png"
    raw_path = dirs["seedream_raw_outputs"] / f"{sample_id}.png"
    open_candidate_path = dirs["seedream_open_candidates"] / f"{sample_id}.png"
    crop_path = dirs["crops"] / f"{sample_id}_open_door_change.jpg"
    overlay_path = dirs["selection_overlays"] / f"{sample_id}_closed_door_bbox.jpg"
    close_pair_path = dirs["close_images"] / f"{sample_id}.png"
    open_pair_path = dirs["open_images"] / f"{sample_id}.png"
    pair_annotation_path = dirs["pair_annotations"] / f"{sample_id}.json"
    qwen_log = dirs["qwen"] / f"{sample_id}.json"
    seedream_request_log = dirs["seedream_requests"] / f"{sample_id}.json"
    seedream_response_log = dirs["seedream_responses"] / f"{sample_id}.json"
    qwen_call_count = 0
    seedream_call_count = 0
    current_stage = "selector"

    try:
        if preselected_selection is not None:
            with Image.open(image_path) as source_image:
                source_size = source_image.size
            selection = preselected_selection
        else:
            source_size = make_vlm_preview(
                image_path,
                preview_path,
                max_side=config.preview_max_side,
                jpeg_quality=config.preview_jpeg_quality,
            )
            if config.dry_run:
                selection = QwenVLMClient(
                    config.vlm_model,
                    build_cabinet_door_selector_config(config),
                    dry_run=True,
                    request_timeout_seconds=config.vlm_timeout_seconds,
                    max_retries=config.vlm_max_retries,
                ).select_cabinet_door_bbox(str(preview_path), source_size, str(qwen_log))
            else:
                qwen = qwen_client or QwenVLMClient(
                    config.vlm_model,
                    build_cabinet_door_selector_config(config),
                    request_timeout_seconds=config.vlm_timeout_seconds,
                    max_retries=config.vlm_max_retries,
                )
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
                    "stage": (
                        "segmentation_bbox_selection"
                        if preselected_selection is not None
                        else "qwen_bbox_selection"
                    ),
                    "skip_reason": selection.get("reason") or "no suitable closed cabinet door",
                    "seedream_call_count": 0,
                }
            )
            write_json(metadata_path, metadata)
            return metadata

        selected_bbox = tuple(int(value) for value in selection["bbox"])
        _save_selection_overlay(image_path, selected_bbox, overlay_path, selection)
        context_crop_bbox = expand_door_edit_bbox(
            selected_bbox,
            source_size,
            hinge_side=str(selection.get("hinge_side") or "unknown"),
        )
        crop_size = _save_exact_crop(image_path, context_crop_bbox, seedream_input_path)
        crop_closed_door_bbox = _bbox_to_crop_coordinates(selected_bbox, context_crop_bbox)
        current_stage = "seedream"
        if config.dry_run:
            shutil.copy2(seedream_input_path, raw_path)
        else:
            seedream = image_client or WanImageClient(config.image_model, I2IServiceConfig.from_env().image)
            seedream_call_count = 1
            seedream.edit_image_with_wan(
                str(seedream_input_path),
                "Open the selected closed equipment cabinet door.",
                "Do not change any other door, cabinet, object, perspective, or crop background.",
                crop_closed_door_bbox,
                str(raw_path),
                "cabinet_door_open",
                request_log_path=str(seedream_request_log),
                response_log_path=str(seedream_response_log),
                seedream_mode="cabinet_door_open",
            )
        _resize_generated_to_source(raw_path, crop_size, open_candidate_path)
        current_stage = "quality_gate"
        quality = evaluate_cabinet_door_edit(
            seedream_input_path,
            open_candidate_path,
            crop_closed_door_bbox,
            hinge_side=str(selection.get("hinge_side") or "unknown"),
            max_outside_mean_abs_diff=config.max_outside_mean_abs_diff,
            max_outside_structure_change_ratio=config.max_outside_structure_change_ratio,
        )
        generated_change_bbox = tuple(int(value) for value in quality["generated_change_bbox"])
        generated_change_bbox_source = _bbox_to_source_coordinates(
            generated_change_bbox,
            context_crop_bbox,
        )
        _save_crop(open_candidate_path, generated_change_bbox, crop_path)
        if not quality["passes_quality"] and not config.dry_run:
            rejected_path = dirs["failed_generated_images"] / open_candidate_path.name
            shutil.move(str(open_candidate_path), rejected_path)
            final_uri: str | None = None
            rejected_uri: str | None = str(rejected_path)
            status = "rejected"
        else:
            shutil.copy2(seedream_input_path, close_pair_path)
            shutil.copy2(open_candidate_path, open_pair_path)
            final_uri = str(open_pair_path)
            rejected_uri = None
            status = "accepted"
            pair_annotation = {
                "pair_id": sample_id,
                "object_type": "electrical_equipment_cabinet_door",
                "source_image": str(image_path),
                "close_image": str(close_pair_path),
                "open_image": str(open_pair_path),
                "source_closed_door_bbox": list(selected_bbox),
                "source_context_crop_bbox": list(context_crop_bbox),
                "crop_closed_door_bbox": list(crop_closed_door_bbox),
                "crop_size": list(crop_size),
                "bbox_format": "xyxy",
                "selection_backend": selection.get("selection_backend", config.selector_backend),
                "candidate_index": selection.get("candidate_index"),
                "segmentation_confidence": selection.get("segmentation_confidence"),
                "state_open_probability": selection.get("state_open_probability"),
                "gate_not_door_probability": selection.get("gate_not_door_probability"),
                "bbox_edge_density": selection.get("bbox_edge_density"),
                "dark_neutral_ratio": selection.get("dark_neutral_ratio"),
                "existing_open_overlap_ratio": selection.get("existing_open_overlap_ratio"),
                "manual_close_override": selection.get("manual_close_override", False),
                "room": selection.get("room"),
                "room_type": selection.get("room_type"),
            }
            write_json(pair_annotation_path, pair_annotation)

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
                "seedream_input_crop": str(seedream_input_path),
                "selection_overlay": str(overlay_path),
                "source_closed_door_bbox": list(selected_bbox),
                "source_context_crop_bbox": list(context_crop_bbox),
                "crop_closed_door_bbox": list(crop_closed_door_bbox),
                "crop_size": list(crop_size),
                "generated_change_bbox_crop": list(generated_change_bbox),
                "generated_change_bbox_source": list(generated_change_bbox_source),
                "bbox_format": "xyxy",
                "close_image": str(close_pair_path) if status == "accepted" else None,
                "open_image": str(open_pair_path) if status == "accepted" else None,
                "pair_annotation": str(pair_annotation_path) if status == "accepted" else None,
                "quality": quality,
            }
        )
        write_json(metadata_path, metadata)
        return metadata
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "stage": current_stage,
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
    images, priority_image_count = prioritize_source_images(
        images,
        config.priority_image_dir,
        priority_only=config.priority_only,
    )
    if not images:
        raise ValueError(f"no source images found under {config.image_dir}")

    selector: SegmentationCandidateSelector | None = None
    prepared: list[tuple[Path, dict[str, Any] | None]]
    selection_results: list[dict[str, Any]] = []
    if config.selector_backend == "segmentation_candidates":
        selector = SegmentationCandidateSelector(config)
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    selection = selector.select(image_path, image.size)
            except Exception as exc:
                selection = {
                    "status": "skipped",
                    "reason": f"selection_input_error:{exc}",
                    "selection_backend": "segmentation_candidates",
                    "bbox": None,
                }
            selection_results.append(
                {
                    "sample_id": image_path.stem,
                    "image_name": image_path.name,
                    **selection,
                }
            )
        selected_by_name = {
            result["image_name"]: result
            for result in selection_results
            if result.get("status") == "selected"
        }
        selected_images = [image for image in images if image.name in selected_by_name]
        if config.limit is not None:
            selected_images = selected_images[: max(0, config.limit)]
        prepared = [(image, selected_by_name[image.name]) for image in selected_images]
        selection_report_path = dirs["logs"] / "segmentation_selections.jsonl"
        with selection_report_path.open("w", encoding="utf-8") as handle:
            for result in selection_results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    elif config.selector_backend == "vlm":
        if config.limit is not None:
            images = images[: max(0, config.limit)]
        prepared = [(image, None) for image in images]
    else:
        raise ValueError(
            "unsupported selector_backend; expected segmentation_candidates or vlm, "
            f"got {config.selector_backend!r}"
        )

    if not prepared:
        raise ValueError("no eligible closed electrical cabinet-door candidates were selected")

    if not config.dry_run:
        services = I2IServiceConfig.from_env()
        if config.selector_backend == "vlm":
            selector_config = build_cabinet_door_selector_config(config)
            if not selector_config.api_key:
                raise RuntimeError(f"{selector_config.api_key_env} is required for the configured VLM selector")
        if not services.image.api_key:
            raise RuntimeError("ARK_API_KEY is required")
        validate_api_key_for_http_header(
            services.image.api_key,
            env_name=services.image.api_key_env or "ARK_API_KEY",
        )

    def run_one(item: tuple[Path, dict[str, Any] | None]) -> dict[str, Any]:
        image_path, preselected_selection = item
        return process_cabinet_door_image(
            image_path,
            config,
            dirs,
            qwen_client=(
                qwen_factory()
                if config.selector_backend == "vlm" and qwen_factory
                else None
            ),
            image_client=image_factory() if image_factory else None,
            preselected_selection=preselected_selection,
        )

    results: list[dict[str, Any]] = []
    workers = max(1, config.workers)
    if workers == 1:
        for index, item in enumerate(prepared, start=1):
            logger.info("processing %s/%s: %s", index, len(prepared), item[0].name)
            results.append(run_one(item))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, item): item[0] for item in prepared}
            for future in as_completed(futures):
                results.append(future.result())

    pair_manifest_path = config.output_root / "pairs" / "manifest.csv"
    accepted_results = [result for result in results if result.get("status") == "accepted"]
    with pair_manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "pair_id",
            "close_image",
            "open_image",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "context_x1",
            "context_y1",
            "context_x2",
            "context_y2",
            "crop_bbox_x1",
            "crop_bbox_y1",
            "crop_bbox_x2",
            "crop_bbox_y2",
            "room",
            "room_type",
            "candidate_index",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in accepted_results:
            bbox = result["source_closed_door_bbox"]
            context_bbox = result["source_context_crop_bbox"]
            crop_bbox = result["crop_closed_door_bbox"]
            selection = result.get("selection") or {}
            writer.writerow(
                {
                    "pair_id": result["pair_id"],
                    "close_image": result["close_image"],
                    "open_image": result["open_image"],
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "context_x1": context_bbox[0],
                    "context_y1": context_bbox[1],
                    "context_x2": context_bbox[2],
                    "context_y2": context_bbox[3],
                    "crop_bbox_x1": crop_bbox[0],
                    "crop_bbox_y1": crop_bbox[1],
                    "crop_bbox_x2": crop_bbox[2],
                    "crop_bbox_y2": crop_bbox[3],
                    "room": selection.get("room"),
                    "room_type": selection.get("room_type"),
                    "candidate_index": selection.get("candidate_index"),
                }
            )

    summary = {
        "task": "cabinet_door_open",
        "input_total": len(images),
        "priority_image_count": priority_image_count,
        "priority_only": config.priority_only,
        "reviewed_all_close": config.reviewed_all_close,
        "total": len(prepared),
        "accepted": sum(result.get("status") == "accepted" for result in results),
        "rejected": sum(result.get("status") == "rejected" for result in results),
        "skipped": sum(result.get("status") == "skipped" for result in results),
        "skipped_existing": sum(result.get("status") == "skipped_existing" for result in results),
        "failed": sum(result.get("status") == "failed" for result in results),
        "qwen_call_count": sum(int(result.get("qwen_call_count", 0)) for result in results),
        "selector_call_count": sum(int(result.get("qwen_call_count", 0)) for result in results),
        "seedream_call_count": sum(int(result.get("seedream_call_count", 0)) for result in results),
        "model_generated_count": sum(int(result.get("model_generated_count", 0)) for result in results),
        "selector_backend": config.selector_backend,
        "segmentation_candidate_count": sum(
            result.get("status") == "selected" for result in selection_results
        ),
        "segmentation_skipped_count": sum(
            result.get("status") != "selected" for result in selection_results
        ),
        "dry_run": config.dry_run,
        "vlm_provider": config.vlm_provider if config.selector_backend == "vlm" else None,
        "vlm_model": config.vlm_model if config.selector_backend == "vlm" else None,
        "pair_manifest": str(pair_manifest_path),
        "output_root": str(config.output_root),
        "completed_at": now_iso_shanghai(),
    }
    write_json(dirs["logs"] / "run_summary.json", summary)
    logger.info("run summary: %s", summary)
    return summary
