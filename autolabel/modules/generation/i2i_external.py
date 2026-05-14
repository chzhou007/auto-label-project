from __future__ import annotations

from pathlib import Path
from typing import Any

from ...adapters.i2i_generator import I2IGenerator
from ...model_config import resolve_generation_runtime
from ...utils import read_csv, write_csv
from .base import GenerationRunResult


I2I_TASK_FIELDS = [
    "sample_id",
    "image_id",
    "image_uri",
    "anomaly_type",
    "source_type",
    "site",
    "building",
    "floor",
    "room_name",
    "room_type",
    "collection_batch",
    "camera_id",
    "capture_time",
]


class ExternalI2IGenerationModule:
    """Generation module backed by the existing I2I project.

    This keeps the DAG modular while preserving the current I2I code as a
    backend. If the I2I code is moved into this repository later, this module
    can be replaced without changing the pipeline.
    """

    def __init__(self, pipeline_config: dict[str, Any], module_config: dict[str, Any] | None = None) -> None:
        self.pipeline_config = pipeline_config
        self.module_config = module_config or {}

    def prepare_tasks(self, tasks_csv: str | Path, output_root: str | Path) -> Path | None:
        rows = read_csv(tasks_csv)
        if any("task_mode" in row for row in rows):
            rows = [row for row in rows if row.get("task_mode") == "generation"]
        if not rows:
            return None

        target = Path(output_root) / "generation_tasks.filtered.csv"
        write_csv(target, rows, I2I_TASK_FIELDS)
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
        generation_cfg = self.pipeline_config.get("generation", {})
        filtered_tasks = self.prepare_tasks(tasks_csv, output_root)
        if filtered_tasks is None:
            return GenerationRunResult(
                returncode=0,
                output_root=Path(output_root),
                stdout="No generation rows found; skipped generation module.\n",
                skipped=True,
            )

        runtime = resolve_generation_runtime(self.pipeline_config)
        project_dir = (
            self.module_config.get("project_dir")
            or self.pipeline_config.get("paths", {}).get("i2i_project")
        )
        runner = I2IGenerator(project_dir)
        completed = runner.run(
            tasks_csv=filtered_tasks,
            image_root=image_root,
            output_root=output_root,
            vlm_model=runtime["vlm_model_name"],
            image_model=runtime["image_model_name"],
            grid_layout=generation_cfg.get("grid_layout", "4x4"),
            edit_bbox_expand_ratio=float(generation_cfg.get("edit_bbox_expand_ratio", 0.20)),
            crop_expand_ratio=float(generation_cfg.get("crop_expand_ratio", 0.10)),
            workers=int(generation_cfg.get("workers", 1)),
            dry_run=dry_run or bool(generation_cfg.get("dry_run", False)),
            skip_existing=skip_existing or bool(generation_cfg.get("skip_existing", False)),
            limit=limit,
            env=runtime["env"],
        )
        return GenerationRunResult(
            returncode=completed.returncode,
            output_root=Path(output_root),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
