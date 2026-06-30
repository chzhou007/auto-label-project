from __future__ import annotations

import argparse
import concurrent.futures
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from autolabel.utils import now_iso_shanghai, read_csv, read_json, write_json

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from autolabel.modules.generation.benchmark import BenchmarkRecorder
    from autolabel.modules.generation.config import normalize_task_row, positive_float, positive_int, prepare_output_paths, resolve_task_image_path
    from autolabel.modules.generation.cropper import crop_with_expand
    from autolabel.modules.generation.diff_localizer import LocalizationThresholds, localize_difference
    from autolabel.modules.generation.env_loader import load_env_files
    from autolabel.modules.generation.exporter_labelstudio import export_metadata_dir
    from autolabel.modules.generation.grid import bbox_to_dict, clip_bbox, expand_bbox, fine_grid_id_to_bbox, grid_id_to_bbox, grid_ids_overlapping_bboxes, make_fine_grid_preview, make_grid_preview
    from autolabel.modules.generation.metadata_builder import build_autolabel_sample
    from autolabel.modules.generation.prompts import build_negative_prompt, build_wan_edit_prompt
    from autolabel.modules.generation.quality import anomaly_visibility_score, background_preservation_score, passes_quality
    from autolabel.modules.generation.qwen_vlm_client import GridCandidate, GridSelection, QwenVLMClient
    from autolabel.modules.generation.utils import JsonlLogger
    from autolabel.modules.generation.validators import validate_autolabel_sample
    from autolabel.modules.generation.wan_image_client import WanImageClient
else:
    from .benchmark import BenchmarkRecorder
    from .config import normalize_task_row, positive_float, positive_int, prepare_output_paths, resolve_task_image_path
    from .cropper import crop_with_expand
    from .diff_localizer import LocalizationThresholds, localize_difference
    from .env_loader import load_env_files
    from .exporter_labelstudio import export_metadata_dir
    from .grid import bbox_to_dict, clip_bbox, expand_bbox, fine_grid_id_to_bbox, grid_id_to_bbox, grid_ids_overlapping_bboxes, make_fine_grid_preview, make_grid_preview
    from .metadata_builder import build_autolabel_sample
    from .prompts import build_negative_prompt, build_wan_edit_prompt
    from .quality import anomaly_visibility_score, background_preservation_score, passes_quality
    from .qwen_vlm_client import GridCandidate, GridSelection, QwenVLMClient
    from .utils import JsonlLogger
    from .validators import validate_autolabel_sample
    from .wan_image_client import WanImageClient


@dataclass
class EvaluatedGeneration:
    score: float
    sample: dict[str, Any]
    metadata_path: Path
    background_score: float
    visibility_score: float
    selected_grid: str
    generated_image_path: Path
    coarse_candidate_rank: int


@dataclass(frozen=True)
class RunLimits:
    vlm: threading.Semaphore
    wan_submit: threading.Semaphore
    wan_poll: threading.Semaphore
    download: threading.Semaphore
    local_cpu_workers: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM grid selection + Wan local image editing + diff AutoLabel pipeline.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--vlm-model", default="qwen3.6-plus")
    parser.add_argument("--image-model", default="wan2.7-image-pro")
    parser.add_argument("--mode", default="balanced", choices=["speed", "balanced", "quality"])
    parser.add_argument("--vlm-request-image-max-side", type=lambda value: positive_int(value, "vlm-request-image-max-side"), default=512)
    parser.add_argument("--vlm-timeout-seconds", type=lambda value: positive_int(value, "vlm-timeout-seconds"), default=25)
    parser.add_argument("--vlm-json-retry-count", type=int, default=1)
    parser.add_argument("--vlm-max-tokens", type=lambda value: positive_int(value, "vlm-max-tokens"), default=256)
    parser.add_argument("--disable-vlm-fallback", action="store_true")
    parser.add_argument("--wan-timeout-seconds", type=lambda value: positive_int(value, "wan-timeout-seconds"), default=300)
    parser.add_argument("--wan-poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--wan-max-poll-seconds", type=lambda value: positive_int(value, "wan-max-poll-seconds"), default=300)
    parser.add_argument("--grid-layout", default="4x4")
    parser.add_argument("--edit-bbox-expand-ratio", type=lambda value: positive_float(value, "edit-bbox-expand-ratio"), default=0.20)
    parser.add_argument("--crop-expand-ratio", type=lambda value: positive_float(value, "crop-expand-ratio"), default=0.10)
    parser.add_argument("--num-generations-per-candidate", type=lambda value: positive_int(value, "num-generations-per-candidate"), default=1)
    parser.add_argument("--enable-fine-grid", action="store_true")
    parser.add_argument("--disable-fine-grid", action="store_true", help="Compatibility switch; fine grid is disabled by default.")
    parser.add_argument("--enable-vlm-review", action="store_true")
    parser.add_argument("--export-labelstudio", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-candidate-grids", type=lambda value: positive_int(value, "max-candidate-grids"), default=3)
    parser.add_argument("--candidate-grid-count", type=lambda value: positive_int(value, "candidate-grid-count"), default=3)
    parser.add_argument("--topk-coarse-grids", type=lambda value: positive_int(value, "topk-coarse-grids"), default=None)
    parser.add_argument("--allow-sensitive-overlays", action="store_true", help="Allow candidate grids that overlap CCTV timestamp/site-label overlays.")
    parser.add_argument("--exclude-grids", default="", help="Comma-separated coarse grid IDs to block during selection, e.g. A1,D4.")
    parser.add_argument("--sensitive-top-left-ratio", default="0.30,0.12", help="width,height ratios for the top-left timestamp exclusion box.")
    parser.add_argument("--sensitive-bottom-right-ratio", default="0.42,0.12", help="width,height ratios for the bottom-right site-label exclusion box.")
    parser.add_argument("--sensitive-grid-min-overlap-ratio", type=float, default=0.0)
    parser.add_argument("--candidate-strategy", default="sequential", choices=["sequential", "speculative"])
    parser.add_argument("--speculative-top-k", type=lambda value: positive_int(value, "speculative-top-k"), default=2)
    parser.add_argument("--reuse-vlm-cache", action="store_true")
    parser.add_argument("--refresh-vlm-cache", action="store_true")
    parser.add_argument("--reuse-wan-cache", action="store_true", help="Reserved for future Wan result cache; disabled by default.")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--debug-save-intermediates", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--severity-level", default="early", choices=["early", "moderate", "obvious_but_controlled"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4, help="Number of task-level workers. Tune with API rate limits.")
    parser.add_argument("--vlm-concurrency", type=lambda value: positive_int(value, "vlm-concurrency"), default=2)
    parser.add_argument("--wan-submit-concurrency", type=lambda value: positive_int(value, "wan-submit-concurrency"), default=4)
    parser.add_argument("--wan-poll-concurrency", type=lambda value: positive_int(value, "wan-poll-concurrency"), default=8)
    parser.add_argument("--download-concurrency", type=lambda value: positive_int(value, "download-concurrency"), default=8)
    parser.add_argument("--local-cpu-workers", default="auto")
    parser.add_argument(
        "--baseline-mode",
        default="ours_qwen_grid_wan_diff",
        choices=["baseline_random_grid", "baseline_no_diff_use_grid_bbox", "ours_qwen_grid_wan_diff"],
    )
    return parser.parse_args(argv)


def normalize_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.topk_coarse_grids is not None:
        args.candidate_grid_count = args.topk_coarse_grids
    args.candidate_grid_count = max(1, int(args.candidate_grid_count))
    args.max_candidate_grids = min(max(1, int(args.max_candidate_grids)), args.candidate_grid_count)
    if args.disable_fine_grid:
        args.enable_fine_grid = False
    if args.mode == "speed":
        args.enable_vlm_review = False
        args.vlm_timeout_seconds = min(args.vlm_timeout_seconds, 20)
    elif args.mode == "quality":
        args.vlm_request_image_max_side = max(args.vlm_request_image_max_side, 640)
        args.vlm_timeout_seconds = max(args.vlm_timeout_seconds, 35)
    if str(args.local_cpu_workers).lower() == "auto":
        args.local_cpu_workers = max(1, min(8, int(args.workers)))
    else:
        args.local_cpu_workers = max(1, int(args.local_cpu_workers))
    return args


def benchmark_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "workers": args.workers,
        "vlm_concurrency": args.vlm_concurrency,
        "wan_submit_concurrency": args.wan_submit_concurrency,
        "wan_poll_concurrency": args.wan_poll_concurrency,
        "download_concurrency": args.download_concurrency,
        "local_cpu_workers": args.local_cpu_workers,
        "candidate_grid_count": args.candidate_grid_count,
        "max_candidate_grids": args.max_candidate_grids,
        "candidate_strategy": args.candidate_strategy,
        "speculative_top_k": args.speculative_top_k,
        "allow_sensitive_overlays": bool(args.allow_sensitive_overlays),
        "exclude_grids": args.exclude_grids,
        "sensitive_top_left_ratio": args.sensitive_top_left_ratio,
        "sensitive_bottom_right_ratio": args.sensitive_bottom_right_ratio,
        "sensitive_grid_min_overlap_ratio": args.sensitive_grid_min_overlap_ratio,
        "enable_fine_grid": bool(args.enable_fine_grid),
        "enable_vlm_review": bool(args.enable_vlm_review),
        "reuse_vlm_cache": bool(args.reuse_vlm_cache),
        "refresh_vlm_cache": bool(args.refresh_vlm_cache),
    }


def sensitive_overlay_bboxes(image_width: int, image_height: int, args: argparse.Namespace) -> list[tuple[int, int, int, int]]:
    if args.allow_sensitive_overlays:
        return []
    top_width, top_height = parse_ratio_pair(args.sensitive_top_left_ratio)
    bottom_width, bottom_height = parse_ratio_pair(args.sensitive_bottom_right_ratio)
    bboxes: list[tuple[int, int, int, int]] = []
    if top_width > 0 and top_height > 0:
        bboxes.append(clip_bbox((0, 0, image_width * top_width, image_height * top_height), image_width, image_height))
    if bottom_width > 0 and bottom_height > 0:
        bboxes.append(
            clip_bbox(
                (
                    image_width * (1.0 - bottom_width),
                    image_height * (1.0 - bottom_height),
                    image_width,
                    image_height,
                ),
                image_width,
                image_height,
            )
        )
    return bboxes


def excluded_selection_grids(
    image_width: int,
    image_height: int,
    grid_layout: str,
    args: argparse.Namespace,
) -> list[str]:
    manual = [item.strip().upper() for item in str(args.exclude_grids or "").split(",") if item.strip()]
    overlay_bboxes = sensitive_overlay_bboxes(image_width, image_height, args)
    overlay_grids = grid_ids_overlapping_bboxes(
        image_width,
        image_height,
        grid_layout,
        overlay_bboxes,
        min_overlap_ratio=max(0.0, float(args.sensitive_grid_min_overlap_ratio)),
    )
    excluded: list[str] = []
    for grid in [*manual, *overlay_grids]:
        if grid not in excluded:
            excluded.append(grid)
    return excluded


def parse_ratio_pair(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 2:
        raise ValueError(f"ratio pair must look like width,height, got {value!r}")
    width, height = float(parts[0]), float(parts[1])
    return max(0.0, min(1.0, width)), max(0.0, min(1.0, height))


def run_pipeline(args: argparse.Namespace) -> int:
    args = normalize_runtime_args(args)
    load_env_files()
    outputs = prepare_output_paths(args.output_root)
    logger = JsonlLogger(outputs.logs / f"run_{now_iso_shanghai().replace(':', '').replace('-', '').replace('+', '_')}.jsonl")
    recorder = BenchmarkRecorder(args.benchmark or args.profile, outputs.logs, benchmark_config(args))
    limits = RunLimits(
        vlm=threading.Semaphore(args.vlm_concurrency),
        wan_submit=threading.Semaphore(args.wan_submit_concurrency),
        wan_poll=threading.Semaphore(args.wan_poll_concurrency),
        download=threading.Semaphore(args.download_concurrency),
        local_cpu_workers=int(args.local_cpu_workers),
    )
    qwen = QwenVLMClient(
        model_name=args.vlm_model,
        dry_run=args.dry_run,
        request_image_max_side=args.vlm_request_image_max_side,
        request_timeout_seconds=args.vlm_timeout_seconds,
        json_retry_count=args.vlm_json_retry_count,
        enable_grid_fallback=not args.disable_vlm_fallback,
        candidate_grid_count=args.candidate_grid_count,
        cache_dir=outputs.metadata / "vlm_cache",
        reuse_cache=args.reuse_vlm_cache,
        refresh_cache=args.refresh_vlm_cache,
        max_tokens=args.vlm_max_tokens,
    )
    wan = WanImageClient(
        model_name=args.image_model,
        dry_run=args.dry_run,
        request_timeout_seconds=args.wan_timeout_seconds,
        poll_interval_seconds=args.wan_poll_interval_seconds,
        max_poll_seconds=args.wan_max_poll_seconds,
        submit_semaphore=limits.wan_submit,
        poll_semaphore=limits.wan_poll,
        download_semaphore=limits.download,
        stage_timer=recorder.stage,
    )
    if not args.dry_run:
        wan_ok, wan_reason = wan.is_configured()
        if not wan_ok:
            logger.log("preflight", "wan_config", "failed", reason=wan_reason)
            logger.close()
            print(f"FAILED preflight: {wan_reason}", file=sys.stderr)
            return 1
        if args.disable_vlm_fallback and not qwen.api_key:
            reason = "Qwen API key is not configured and --disable-vlm-fallback was set."
            logger.log("preflight", "qwen_config", "failed", reason=reason)
            logger.close()
            print(f"FAILED preflight: {reason}", file=sys.stderr)
            return 1
        if not qwen.api_key:
            logger.log("preflight", "qwen_config", "warning", reason="Qwen API key is not configured; grid fallback will be used.")
    rows = read_csv(args.tasks)
    if args.limit is not None:
        rows = rows[: args.limit]
    success_count = 0
    try:
        worker_count = max(1, int(args.workers))
        if worker_count == 1:
            for index, raw_row in enumerate(rows, 1):
                task_id, success, error = process_raw_row(index, raw_row, args, outputs, qwen, wan, logger, recorder, limits)
                success_count += int(success)
                if error:
                    print(f"FAILED {task_id}: {error}", file=sys.stderr)
                    if args.fail_fast:
                        break
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_row = {
                    executor.submit(process_raw_row, index, raw_row, args, outputs, qwen, wan, logger, recorder, limits): (index, raw_row)
                    for index, raw_row in enumerate(rows, 1)
                }
                for future in concurrent.futures.as_completed(future_to_row):
                    index, raw_row = future_to_row[future]
                    try:
                        task_id, success, error = future.result()
                    except Exception as exc:  # noqa: BLE001 - defensive guard; process_raw_row should catch row failures.
                        task_id = str(raw_row.get("task_id") or f"row_{index}")
                        success = False
                        error = str(exc)
                        logger.log(task_id, "task", "failed", reason=error, row=raw_row)
                    success_count += int(success)
                    if error:
                        print(f"FAILED {task_id}: {error}", file=sys.stderr)
                        if args.fail_fast:
                            break
        if args.export_labelstudio:
            export_path = outputs.labelstudio / "import.json"
            tasks = export_metadata_dir(outputs.metadata, export_path)
            print(f"Wrote {len(tasks)} Label Studio tasks to {export_path}")
    finally:
        summary_path = recorder.write_summary()
        logger.close()
    if summary_path:
        print(f"Benchmark summary: {summary_path}")
    print(f"Completed {success_count}/{len(rows)} tasks. Output root: {outputs.root}")
    return 0 if success_count == len(rows) else 1


def process_raw_row(
    index: int,
    raw_row: dict[str, Any],
    args: argparse.Namespace,
    outputs: Any,
    qwen: QwenVLMClient,
    wan: WanImageClient,
    logger: JsonlLogger,
    recorder: BenchmarkRecorder,
    limits: RunLimits,
) -> tuple[str, bool, str | None]:
    started = time.perf_counter()
    anomaly_type = str(raw_row.get("anomaly_type") or "unknown")
    try:
        task = normalize_task_row(raw_row, index)
        anomaly_type = str(task["anomaly_type"])
        task["severity_level"] = raw_row.get("severity_level") or args.severity_level
        task["severity_level"] = task["severity_level"] or "early"
        if task["severity_level"] not in {"early", "moderate", "obvious_but_controlled"}:
            raise ValueError(f"Unsupported severity_level: {task['severity_level']}")
        success, failure_reason, attempted, success_rank = process_task(task, args, outputs, qwen, wan, logger, recorder, limits)
        recorder.record_task(anomaly_type, success, time.perf_counter() - started, failure_reason, attempted, success_rank)
        return str(task["task_id"]), success, None
    except Exception as exc:  # noqa: BLE001 - row-level failure should be logged and continue.
        task_id = str(raw_row.get("task_id") or f"row_{index}")
        logger.log(task_id, "task", "failed", reason=str(exc), row=raw_row)
        recorder.record_task(anomaly_type, False, time.perf_counter() - started, str(exc), 0, None)
        return task_id, False, str(exc)


def process_task(
    task: dict[str, Any],
    args: argparse.Namespace,
    outputs: Any,
    qwen: QwenVLMClient,
    wan: WanImageClient,
    logger: JsonlLogger,
    recorder: BenchmarkRecorder,
    limits: RunLimits,
) -> tuple[bool, str | None, int, int | None]:
    task_id = task["task_id"]
    existing_metadata_path = outputs.metadata / f"{task.get('sample_id') or f'sample_{task_id}'}.json"
    if args.skip_existing and existing_metadata_path.exists():
        ok, errors = existing_metadata_is_valid(existing_metadata_path)
        if ok:
            logger.log(task_id, "metadata", "skipped", metadata_path=str(existing_metadata_path), validation="passed")
            return True, None, 0, None
        logger.log(task_id, "metadata", "rerun_invalid_existing", metadata_path=str(existing_metadata_path), errors=errors)

    with recorder.stage("load_image"):
        image_path = resolve_task_image_path(task, args.image_root)
        if not image_path.exists():
            logger.log(task_id, "input", "failed", reason="image not found", image_path=str(image_path))
            return False, "image_not_found", 0, None

        with Image.open(image_path) as source:
            image_width, image_height = source.size
    excluded_grids = excluded_selection_grids(image_width, image_height, args.grid_layout, args)
    with recorder.stage("grid_preview"):
        grid_preview = outputs.grid_previews / f"{task_id}_grid.jpg"
        make_grid_preview(image_path, grid_preview, args.grid_layout)
    logger.log(task_id, "grid_preview", "success", image_path=str(image_path), grid_preview=str(grid_preview))

    with recorder.stage("vlm_selection"):
        with limits.vlm:
            selection = qwen.select_grid(
                grid_preview,
                task["anomaly_type"],
                args.grid_layout,
                candidate_count=args.candidate_grid_count,
                excluded_grids=excluded_grids,
            )
    recorder.incr("qwen_grid_calls")
    if selection.raw_response.get("fallback"):
        recorder.incr("qwen_fallback_count")
    if selection.raw_response.get("cache_hit"):
        recorder.incr("qwen_cache_hits")
    selection_status = "fallback" if selection.raw_response.get("fallback") else "success"
    logger.log(
        task_id,
        "vlm_grid_selection",
        selection_status,
        selected_grid=selection.selected_grid,
        candidate_grids=[candidate.__dict__ for candidate in selection.candidate_grids],
        fallback_reason=selection.raw_response.get("fallback_reason"),
        candidate_grid_count=args.candidate_grid_count,
        excluded_grids=excluded_grids,
        fine_grid_enabled=bool(args.enable_fine_grid),
    )

    candidate_pool = selection.candidate_grids[: args.max_candidate_grids]
    best_overall: EvaluatedGeneration | None
    if args.candidate_strategy == "speculative":
        best_overall, attempted_count = evaluate_candidates_speculative(
            task,
            candidate_pool,
            selection,
            image_path,
            image_width,
            image_height,
            args,
            outputs,
            qwen,
            wan,
            logger,
            recorder,
            limits,
        )
    else:
        best_overall = None
        attempted_count = 0
        for rank, candidate in enumerate(candidate_pool, 1):
            attempted_count += 1
            candidate_best = evaluate_candidate(
                task=task,
                candidate=candidate,
                coarse_candidate_rank=rank,
                selection=selection,
                original_image_path=image_path,
                image_width=image_width,
                image_height=image_height,
                args=args,
                outputs=outputs,
                qwen=qwen,
                wan=wan,
                logger=logger,
                recorder=recorder,
                limits=limits,
            )
            recorder.record_grid_rank(rank, candidate_best is not None)
            if candidate_best is not None:
                best_overall = candidate_best
                break

    if best_overall is None:
        logger.log(task_id, "task", "failed", reason="all VLM candidate grids failed")
        return False, "all_candidate_grids_failed", attempted_count, None

    if args.skip_existing and best_overall.metadata_path.exists():
        ok, errors = existing_metadata_is_valid(best_overall.metadata_path)
        if ok:
            logger.log(task_id, "metadata", "skipped", metadata_path=str(best_overall.metadata_path), validation="passed")
            return True, None, attempted_count, best_overall.coarse_candidate_rank
        logger.log(task_id, "metadata", "overwriting_invalid_existing", metadata_path=str(best_overall.metadata_path), errors=errors)
    with recorder.stage("metadata_write"):
        write_json(best_overall.metadata_path, best_overall.sample)
    logger.log(
        task_id,
        "task",
        "success",
        metadata_path=str(best_overall.metadata_path),
        generated_image_path=str(best_overall.generated_image_path),
        selected_grid=best_overall.selected_grid,
        background_preservation_score=best_overall.background_score,
        anomaly_visibility_score=best_overall.visibility_score,
        coarse_candidate_rank=best_overall.coarse_candidate_rank,
        topk_attempted_candidates=attempted_count,
    )
    return True, None, attempted_count, best_overall.coarse_candidate_rank


def evaluate_candidates_speculative(
    task: dict[str, Any],
    candidate_pool: list[GridCandidate],
    selection: GridSelection,
    original_image_path: Path,
    image_width: int,
    image_height: int,
    args: argparse.Namespace,
    outputs: Any,
    qwen: QwenVLMClient,
    wan: WanImageClient,
    logger: JsonlLogger,
    recorder: BenchmarkRecorder,
    limits: RunLimits,
) -> tuple[EvaluatedGeneration | None, int]:
    if not candidate_pool:
        return None, 0
    speculative_count = min(len(candidate_pool), max(1, int(args.speculative_top_k)))
    attempted_count = speculative_count
    best: EvaluatedGeneration | None = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(speculative_count, limits.local_cpu_workers)) as executor:
        future_to_rank = {
            executor.submit(
                evaluate_candidate,
                task,
                candidate,
                rank,
                selection,
                original_image_path,
                image_width,
                image_height,
                args,
                outputs,
                qwen,
                wan,
                logger,
                recorder,
                limits,
            ): rank
            for rank, candidate in enumerate(candidate_pool[:speculative_count], 1)
        }
        for future in concurrent.futures.as_completed(future_to_rank):
            rank = future_to_rank[future]
            try:
                evaluated = future.result()
            except Exception as exc:  # noqa: BLE001 - one speculative candidate must not fail the task.
                logger.log(task["task_id"], "candidate_generation", "failed", coarse_candidate_rank=rank, reason=str(exc))
                evaluated = None
            recorder.record_grid_rank(rank, evaluated is not None)
            if evaluated is not None and (best is None or evaluated.score > best.score):
                best = evaluated
    if best is not None:
        return best, attempted_count

    for rank, candidate in enumerate(candidate_pool[speculative_count:], speculative_count + 1):
        attempted_count += 1
        evaluated = evaluate_candidate(
            task,
            candidate,
            rank,
            selection,
            original_image_path,
            image_width,
            image_height,
            args,
            outputs,
            qwen,
            wan,
            logger,
            recorder,
            limits,
        )
        recorder.record_grid_rank(rank, evaluated is not None)
        if evaluated is not None:
            return evaluated, attempted_count
    return None, attempted_count


def existing_metadata_is_valid(metadata_path: Path) -> tuple[bool, list[str]]:
    try:
        sample = read_json(metadata_path)
    except Exception as exc:  # noqa: BLE001 - invalid JSON should trigger rerun.
        return False, [f"metadata JSON read failed: {exc}"]
    return validate_autolabel_sample(sample)


def evaluate_candidate(
    task: dict[str, Any],
    candidate: GridCandidate,
    coarse_candidate_rank: int,
    selection: GridSelection,
    original_image_path: Path,
    image_width: int,
    image_height: int,
    args: argparse.Namespace,
    outputs: Any,
    qwen: QwenVLMClient,
    wan: WanImageClient,
    logger: JsonlLogger,
    recorder: BenchmarkRecorder,
    limits: RunLimits,
) -> EvaluatedGeneration | None:
    task_id = task["task_id"]
    coarse_bbox = grid_id_to_bbox(candidate.grid, image_width, image_height, args.grid_layout)
    grid_bbox = coarse_bbox
    selected_grid_label = candidate.grid
    fine_selection_raw: dict[str, Any] | None = None
    if args.enable_fine_grid:
        recorder.incr("fine_grid_calls")
        fine_preview = outputs.grid_previews / f"{task_id}_{candidate.grid}_fine_grid.jpg"
        make_fine_grid_preview(original_image_path, coarse_bbox, fine_preview, "3x3")
        with recorder.stage("vlm_selection"):
            with limits.vlm:
                fine_selection = qwen.select_fine_grid(fine_preview, task["anomaly_type"], candidate.grid)
        fine_grid = fine_selection.candidate_grids[0].grid
        grid_bbox = fine_grid_id_to_bbox(fine_grid, coarse_bbox, "3x3")
        selected_grid_label = f"{candidate.grid}/{fine_grid}"
        fine_selection_raw = fine_selection.raw_response
        fine_status = "fallback" if fine_selection.raw_response.get("fallback") else "success"
        logger.log(task_id, "vlm_fine_grid_selection", fine_status, coarse_grid=candidate.grid, fine_grid=fine_grid)

    expanded_edit_bbox = expand_bbox(grid_bbox, image_width, image_height, args.edit_bbox_expand_ratio)
    prompt = build_wan_edit_prompt(task["anomaly_type"], selection.edit_region_hint or candidate.reason, task["severity_level"])
    negative_prompt = build_negative_prompt()
    thresholds = LocalizationThresholds.for_anomaly(task["anomaly_type"])
    best: EvaluatedGeneration | None = None

    for generation_index in range(1, args.num_generations_per_candidate + 1):
        suffix = f"{task_id}_r{coarse_candidate_rank:02d}_{selected_grid_label.replace('/', '_')}_g{generation_index}"
        generated_path = outputs.generated_images / f"{suffix}.jpg"
        response_path = outputs.api_responses / f"{suffix}_wan_response.json"
        mask_path = outputs.masks / f"{suffix}_mask.png"
        crop_path = outputs.crops / f"{suffix}_crop.jpg"
        try:
            wan_result = wan.generate(
                original_image_path=original_image_path,
                bbox_list=[expanded_edit_bbox],
                prompt=prompt,
                negative_prompt=negative_prompt,
                output_path=generated_path,
                response_path=response_path,
                anomaly_type=task["anomaly_type"],
            )
            recorder.incr("wan_generation_calls")
            with recorder.stage("generated_size_check"):
                with limits.download:
                    resized_image: Image.Image | None = None
                    with Image.open(generated_path) as generated_source:
                        if generated_source.size != (image_width, image_height):
                            logger.log(
                                task_id,
                                "generated_size",
                                "warning",
                                selected_grid=selected_grid_label,
                                generated_size=generated_source.size,
                                expected_size=(image_width, image_height),
                                action="resize_to_original_size",
                            )
                            resized_image = generated_source.convert("RGB").resize((image_width, image_height), Image.Resampling.LANCZOS)
                    if resized_image is not None:
                        resized_image.save(generated_path, quality=95)
            with recorder.stage("diff_localization"):
                localization = localize_difference(original_image_path, generated_path, expanded_edit_bbox, mask_path, thresholds)
            if not localization.success or localization.final_bbox is None or localization.mask_path is None:
                logger.log(task_id, "diff_localization", "failed", selected_grid=selected_grid_label, reason=localization.reason, metrics=localization.metrics)
                continue
            with recorder.stage("quality"):
                background_score = background_preservation_score(original_image_path, generated_path, expanded_edit_bbox)
                visibility_score = anomaly_visibility_score(localization.metrics)
                min_background_score = min_background_quality_score(task["anomaly_type"])
                min_visibility_score = min_visibility_quality_score(task["anomaly_type"])
                quality_ok, quality_reason = passes_quality(
                    background_score,
                    visibility_score,
                    min_background_score=min_background_score,
                    min_visibility_score=min_visibility_score,
                )
            if not quality_ok:
                logger.log(
                    task_id,
                    "quality",
                    "failed",
                    selected_grid=selected_grid_label,
                    reason=quality_reason,
                    background_preservation_score=background_score,
                    anomaly_visibility_score=visibility_score,
                    min_background_score=min_background_score,
                    min_visibility_score=min_visibility_score,
                )
                continue
            with recorder.stage("crop"):
                crop_info = crop_with_expand(generated_path, localization.final_bbox, crop_path, args.crop_expand_ratio)
            review_payload = None
            review_score = None
            if args.enable_vlm_review:
                with recorder.stage("vlm_selection"):
                    with limits.vlm:
                        review = qwen.review_generation(crop_path, task["anomaly_type"])
                recorder.incr("qwen_review_calls")
                review_payload = review.raw_response
                review_score = review.score
                if not (review.is_valid and review.anomaly_type_match and review.location_reasonable and not review.visual_artifact):
                    logger.log(task_id, "vlm_review", "failed", selected_grid=selected_grid_label, review=review.raw_response)
                    continue
                visibility_score = anomaly_visibility_score(localization.metrics, review_score=review.score)
            combined_score = 0.50 * visibility_score + 0.35 * background_score + 0.15 * (review_score if review_score is not None else 1.0)
            generation_params = {
                "localization_pipeline": "vlm_grid_selection_wan_edit_diff_localization",
                "grid_layout": args.grid_layout,
                "enable_fine_grid": bool(args.enable_fine_grid),
                "fine_grid_enabled": bool(args.enable_fine_grid),
                "fine_grid_removed_or_skipped": not bool(args.enable_fine_grid),
                "selected_grid": selected_grid_label,
                "selected_coarse_grid": candidate.grid,
                "coarse_candidate_rank": coarse_candidate_rank,
                "coarse_candidate_grids_topk": [item.grid for item in selection.candidate_grids[: args.candidate_grid_count]],
                "coarse_candidate_grids_top3": [item.grid for item in selection.candidate_grids[:3]],
                "candidate_grid_count": args.candidate_grid_count,
                "excluded_sensitive_grids": list(selection.raw_response.get("excluded_grids") or []),
                "sensitive_overlay_exclusion_enabled": not bool(args.allow_sensitive_overlays),
                "topk_strategy": args.candidate_strategy,
                "candidate_generation_index": generation_index,
                "candidate_grids": [item.grid for item in selection.candidate_grids],
                "grid_bbox": list(map(int, grid_bbox)),
                "expanded_edit_bbox": list(map(int, expanded_edit_bbox)),
                "final_bbox_source": "image_difference_connected_components",
                "vlm_model": args.vlm_model,
                "image_generation_model": args.image_model,
                "severity_level": task["severity_level"],
                "prompt_version": "v1.0",
                "negative_prompt_version": "v2.0",
                "background_preservation_score": background_score,
                "anomaly_visibility_score": visibility_score,
                "min_background_score": min_background_score,
                "min_visibility_score": min_visibility_score,
                "vlm_review_score": review_score,
                "wan_prompt": prompt,
                "negative_prompt": negative_prompt,
                "wan_input_image_uri": str(original_image_path),
                "wan_input_is_clean_original": True,
                "wan_response_path": str(response_path),
                "wan_raw_response": wan_result.raw_response,
                "qwen_grid_selection": selection.raw_response,
                "qwen_grid_selection_fallback": bool(selection.raw_response.get("fallback")),
                "qwen_fine_grid_selection": fine_selection_raw,
                "qwen_fine_grid_selection_fallback": bool(fine_selection_raw and fine_selection_raw.get("fallback")),
                "qwen_review": review_payload,
                "localization_metrics": localization.metrics,
                "baseline_mode": args.baseline_mode,
            }
            sample = build_autolabel_sample(
                task=task,
                generated_image_path=generated_path,
                original_image_path=original_image_path,
                image_width=image_width,
                image_height=image_height,
                final_bbox=localization.final_bbox,
                crop_info=crop_info,
                mask_path=localization.mask_path,
                generation_params=generation_params,
            )
            with recorder.stage("metadata_validation"):
                ok, errors = validate_autolabel_sample(sample)
            if not ok:
                logger.log(task_id, "required_fields_v1", "failed", selected_grid=selected_grid_label, errors=errors)
                continue
            logger.log(
                task_id,
                "candidate_generation",
                "success",
                selected_grid=selected_grid_label,
                coarse_candidate_rank=coarse_candidate_rank,
                generation_index=generation_index,
                final_bbox=bbox_to_dict(localization.final_bbox),
                mask_uri=str(localization.mask_path),
                background_preservation_score=background_score,
                anomaly_visibility_score=visibility_score,
            )
            sample_id = sample["sample_id"]
            metadata_path = outputs.metadata / f"{sample_id}.json"
            evaluated = EvaluatedGeneration(
                score=combined_score,
                sample=sample,
                metadata_path=metadata_path,
                background_score=background_score,
                visibility_score=visibility_score,
                selected_grid=selected_grid_label,
                generated_image_path=generated_path,
                coarse_candidate_rank=coarse_candidate_rank,
            )
            if best is None or evaluated.score > best.score:
                best = evaluated
        except Exception as exc:  # noqa: BLE001 - candidate fallback continues after logging.
            logger.log(task_id, "candidate_generation", "failed", selected_grid=selected_grid_label, generation_index=generation_index, reason=str(exc))
    return best


def min_background_quality_score(anomaly_type: str) -> float:
    if anomaly_type in {"water_leak", "water_leakage"}:
        return 0.72
    if anomaly_type == "coolant_leak":
        return 0.74
    return 0.78


def min_visibility_quality_score(anomaly_type: str) -> float:
    if anomaly_type in {"water_leak", "water_leakage"}:
        return 0.075
    if anomaly_type == "coolant_leak":
        return 0.085
    if anomaly_type == "diesel_leak":
        return 0.10
    if anomaly_type == "oil_leak":
        return 0.12
    return 0.10


def main(argv: list[str] | None = None) -> int:
    return run_pipeline(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
