from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import random
import time

from PIL import Image

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


def _validate_final_bbox_is_refined(final_box: dict, expanded_bbox: tuple[int, int, int, int]) -> None:
    expanded_box = bbox_to_box_dict(expanded_bbox)
    if final_box == expanded_box or _bbox_iou(final_box, expanded_bbox) >= 0.95:
        raise ValueError(
            "diff localization failed: final_bbox is identical or nearly identical to expanded_edit_bbox"
        )


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

    candidates = _candidate_grids(vlm_result)
    for grid_id in candidates:
        try:
            grid_bbox = grid_id_to_bbox(grid_id, width, height)
            expanded_bbox = expand_bbox(grid_bbox, width, height, cfg.edit_bbox_expand_ratio)
            wan.edit_image_with_wan(
                image_path=str(original_path),
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                bbox=expanded_bbox,
                output_path=str(generated_path),
                anomaly_type=task["anomaly_type"],
                request_log_path=str(dirs["logs"] / f"{sample_id}_wan_request.json"),
                response_log_path=str(dirs["logs"] / f"{sample_id}_wan_response.json"),
            )
            gen_width, gen_height = image_size(generated_path)
            if (gen_width, gen_height) != (width, height):
                logger.info("%s generated image size differs; resizing generated image to original dimensions", sample_id)
                _normalize_generated_size(generated_path, (width, height))
                gen_width, gen_height = image_size(generated_path)
            diff = localize_change_bbox(
                str(original_path),
                str(generated_path),
                expanded_bbox,
                task["anomaly_type"],
                str(mask_path),
            )
            _validate_final_bbox_is_refined(diff["bbox"], expanded_bbox)
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
                "grid_bbox": list(grid_bbox),
                "expanded_edit_bbox": list(expanded_bbox),
                "final_bbox_source": "image_difference_within_selected_grid",
                "diff_method": diff["diff_method"],
                "mask_uri": relative_uri(mask_path),
                "vlm_selection": vlm_result,
            }
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
            logger.warning("%s generation/localization failed for grid %s: %s", sample_id, grid_id, exc)

    _write_failure(
        dirs["logs"],
        sample_id,
        "generation_or_diff",
        last_error or "unknown generation/diff failure",
        {"vlm_result": vlm_result, "candidate_grids": candidates},
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

    def run_one(task: dict) -> tuple[str, bool]:
        vlm = QwenVLMClient(cfg.vlm_model, services.vlm, dry_run=cfg.dry_run)
        wan = WanImageClient(cfg.image_model, services.image, dry_run=cfg.dry_run)
        try:
            return task.get("sample_id", "unknown"), process_task(task, cfg, dirs, vlm, wan)
        except (ValidationError, Exception) as exc:
            _write_failure(dirs["logs"], task.get("sample_id", "unknown"), "task", exc, {"task": task})
            return task.get("sample_id", "unknown"), False

    if workers == 1:
        for task in runnable_tasks:
            _, success = run_one(task)
            if success:
                ok += 1
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, task) for task in runnable_tasks]
            completed = 0
            for future in as_completed(futures):
                sample_id, success = future.result()
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

    summary = {"total": len(tasks), "processed": len(runnable_tasks), "succeeded": ok, "failed": failed, "skipped": skipped}
    write_json(dirs["logs"] / "run_summary.json", summary)
    logger.info("run summary: %s", summary)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
