from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ...model_config import resolve_generation_runtime
from ...utils import read_csv, write_csv
from .base import GenerationRunResult


class VLMWanAutoLabelGenerationModule:
    """Internal VLM grid + Wan edit + diff-localization generation backend."""

    def __init__(self, pipeline_config: dict[str, Any], module_config: dict[str, Any] | None = None) -> None:
        self.pipeline_config = pipeline_config
        self.module_config = module_config or {}
        self.generation_root = Path(__file__).resolve().parent
        self.repo_root = Path(__file__).resolve().parents[3]
        self.main_py = self.generation_root / "main.py"

    def prepare_tasks(self, tasks_csv: str | Path, output_root: str | Path) -> Path | None:
        rows = read_csv(tasks_csv)
        if any("task_mode" in row for row in rows):
            rows = [row for row in rows if row.get("task_mode") == "generation"]
        if not rows:
            return None
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        target = Path(output_root) / "generation_tasks.filtered.csv"
        write_csv(target, rows, fieldnames)
        return target

    def run(
        self,
        tasks_csv: str | Path,
        image_root: str | Path,
        output_root: str | Path,
        dry_run: bool = False,
        skip_existing: bool = False,
        limit: int | None = None,
    ) -> GenerationRunResult:
        if not self.main_py.exists():
            raise FileNotFoundError(f"Internal VLM/Wan entrypoint not found: {self.main_py}")
        filtered_tasks = self.prepare_tasks(tasks_csv, output_root)
        if filtered_tasks is None:
            return GenerationRunResult(
                returncode=0,
                output_root=Path(output_root),
                stdout="No generation rows found; skipped internal VLM/Wan generation module.\n",
                skipped=True,
            )

        generation_cfg = self.pipeline_config.get("generation", {})
        runtime = resolve_generation_runtime(self.pipeline_config)
        cmd = [
            sys.executable,
            str(self.main_py),
            "--tasks",
            str(filtered_tasks.resolve()),
            "--image-root",
            str(Path(image_root).resolve()),
            "--output-root",
            str(Path(output_root).resolve()),
            "--vlm-model",
            str(self.module_config.get("vlm_model") or runtime["vlm_model_name"] or "qwen3.6-plus"),
            "--image-model",
            str(self.module_config.get("image_model") or runtime["image_model_name"] or "wan2.7-image-pro"),
            "--grid-layout",
            str(generation_cfg.get("grid_layout", "4x4")),
            "--edit-bbox-expand-ratio",
            str(generation_cfg.get("edit_bbox_expand_ratio", 0.20)),
            "--crop-expand-ratio",
            str(generation_cfg.get("crop_expand_ratio", 0.10)),
            "--num-generations-per-candidate",
            str(generation_cfg.get("num_generations_per_candidate", 1)),
            "--max-candidate-grids",
            str(generation_cfg.get("max_candidate_grids", 9)),
            "--candidate-grid-count",
            str(generation_cfg.get("candidate_grid_count", generation_cfg.get("topk_coarse_grids", 9))),
            "--mode",
            str(generation_cfg.get("mode", "balanced")),
            "--severity-level",
            str(generation_cfg.get("severity_level", "early")),
            "--workers",
            str(generation_cfg.get("workers", 4)),
            "--vlm-concurrency",
            str(generation_cfg.get("vlm_concurrency", 2)),
            "--wan-submit-concurrency",
            str(generation_cfg.get("wan_submit_concurrency", 4)),
            "--wan-poll-concurrency",
            str(generation_cfg.get("wan_poll_concurrency", 8)),
            "--download-concurrency",
            str(generation_cfg.get("download_concurrency", 8)),
            "--candidate-strategy",
            str(generation_cfg.get("candidate_strategy", "sequential")),
            "--speculative-top-k",
            str(generation_cfg.get("speculative_top_k", 2)),
        ]
        if dry_run or bool(generation_cfg.get("dry_run", False)):
            cmd.append("--dry-run")
        if skip_existing:
            cmd.append("--skip-existing")
        if limit is not None:
            cmd.extend(["--limit", str(limit)])
        if bool(generation_cfg.get("enable_fine_grid", False)):
            cmd.append("--enable-fine-grid")
        if bool(generation_cfg.get("enable_vlm_review", False)):
            cmd.append("--enable-vlm-review")
        if bool(generation_cfg.get("export_labelstudio", False)):
            cmd.append("--export-labelstudio")
        if bool(generation_cfg.get("benchmark", False)):
            cmd.append("--benchmark")
        if bool(generation_cfg.get("reuse_vlm_cache", False)):
            cmd.append("--reuse-vlm-cache")
        if bool(generation_cfg.get("refresh_vlm_cache", False)):
            cmd.append("--refresh-vlm-cache")

        subprocess_env = os.environ.copy()
        subprocess_env.update({key: value for key, value in runtime.get("env", {}).items() if value not in ("", None)})
        completed = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            text=True,
            capture_output=True,
            check=False,
            env=subprocess_env,
        )
        return GenerationRunResult(
            returncode=completed.returncode,
            output_root=Path(output_root),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
