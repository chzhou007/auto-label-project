from __future__ import annotations

from pathlib import Path
from typing import Any

from ...adapters.i2i_generator import I2IGenerator
from ...model_config import resolve_generation_runtime
from ...utils import slugify, write_csv
from .base import GenerationRunResult
from .config import filter_generation_rows, group_generation_rows_by_localizer_policy


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

    def filtered_generation_rows(self, tasks_csv: str | Path) -> list[dict[str, Any]]:
        return filter_generation_rows(str(tasks_csv))

    def prepare_tasks(self, rows: list[dict[str, Any]], output_path: str | Path) -> Path | None:
        if not rows:
            return None
        target = Path(output_path)
        write_csv(target, rows, I2I_TASK_FIELDS)
        return target

    def pass_localizer_cli_args(self) -> bool:
        return bool(self.module_config.get("pass_localizer_cli_args", False))

    def build_runtime_groups(self, rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
        if not self.pass_localizer_cli_args():
            return [("default", rows)]
        return group_generation_rows_by_localizer_policy(self.pipeline_config, rows)

    def build_extra_cli_args(self, runtime: dict[str, Any]) -> list[str]:
        args = [str(arg) for arg in runtime.get("extra_cli_args", []) if str(arg)]
        if self.pass_localizer_cli_args():
            args.extend(str(arg) for arg in runtime.get("localizer_cli_args", []) if str(arg))
        return args

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
        rows = self.filtered_generation_rows(tasks_csv)
        if not rows:
            return GenerationRunResult(
                returncode=0,
                output_root=Path(output_root),
                stdout="No generation rows found; skipped generation module.\n",
                skipped=True,
            )

        project_dir = (
            self.module_config.get("project_dir")
            or self.pipeline_config.get("paths", {}).get("i2i_project")
        )
        runner = I2IGenerator(project_dir)
        combined_stdout: list[str] = []
        combined_stderr: list[str] = []
        for group_name, group_rows in self.build_runtime_groups(rows):
            filtered_tasks = self.prepare_tasks(
                group_rows,
                Path(output_root) / f"generation_tasks.{slugify(group_name)}.csv",
            )
            if filtered_tasks is None:
                continue
            anomaly_type = group_rows[0].get("anomaly_type") if group_rows else None
            runtime = resolve_generation_runtime(self.pipeline_config, anomaly_type=anomaly_type)
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
                extra_cli_args=self.build_extra_cli_args(runtime),
            )
            if completed.stdout:
                combined_stdout.append(completed.stdout)
            if completed.stderr:
                combined_stderr.append(completed.stderr)
            if completed.returncode != 0:
                return GenerationRunResult(
                    returncode=completed.returncode,
                    output_root=Path(output_root),
                    stdout="".join(combined_stdout),
                    stderr="".join(combined_stderr),
                )
        return GenerationRunResult(
            returncode=0,
            output_root=Path(output_root),
            stdout="".join(combined_stdout),
            stderr="".join(combined_stderr),
        )
