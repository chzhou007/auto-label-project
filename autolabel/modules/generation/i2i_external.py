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

DEFAULT_ANOMALY_TYPE_ALIASES: dict[str, str] = {}


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

    def anomaly_type_aliases(self) -> dict[str, str]:
        configured = self.module_config.get("anomaly_type_aliases")
        if not isinstance(configured, dict):
            return dict(DEFAULT_ANOMALY_TYPE_ALIASES)
        aliases = dict(DEFAULT_ANOMALY_TYPE_ALIASES)
        aliases.update(
            {
                str(source).strip(): str(target).strip()
                for source, target in configured.items()
                if str(source).strip() and str(target).strip()
            }
        )
        return aliases

    def normalize_rows_for_backend(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
        aliases = self.anomaly_type_aliases()
        normalized_rows: list[dict[str, Any]] = []
        remapped: dict[str, str] = {}
        for row in rows:
            normalized = dict(row)
            anomaly_type = str(normalized.get("anomaly_type") or "").strip()
            backend_anomaly_type = aliases.get(anomaly_type, anomaly_type)
            if backend_anomaly_type:
                normalized["anomaly_type"] = backend_anomaly_type
            if backend_anomaly_type and backend_anomaly_type != anomaly_type:
                remapped[anomaly_type] = backend_anomaly_type
            normalized_rows.append(normalized)
        return normalized_rows, remapped

    def prepare_tasks(self, rows: list[dict[str, Any]], output_path: str | Path) -> Path | None:
        if not rows:
            return None
        rows, _ = self.normalize_rows_for_backend(rows)
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
        args = [str(arg) for arg in runtime.get("selector_cli_args", []) if str(arg)]
        args.extend(str(arg) for arg in runtime.get("extra_cli_args", []) if str(arg))
        seedream_cfg = self.pipeline_config.get("generation", {}).get("seedream", {})
        if isinstance(seedream_cfg, dict):
            mode = str(seedream_cfg.get("mode") or "").strip()
            if mode:
                args.extend(["--seedream-mode", mode])
                water_reference_dir = str(seedream_cfg.get("water_reference_dir") or "").strip()
                if water_reference_dir:
                    args.extend(["--water-reference-dir", water_reference_dir])
                if seedream_cfg.get("red_box_max_size") not in (None, ""):
                    args.extend(["--red-box-max-size", str(int(seedream_cfg["red_box_max_size"]))])
                if seedream_cfg.get("red_box_min_size") not in (None, ""):
                    args.extend(["--red-box-min-size", str(int(seedream_cfg["red_box_min_size"]))])
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
            _, remapped = self.normalize_rows_for_backend(group_rows)
            if remapped:
                aliases = ", ".join(f"{source}->{target}" for source, target in sorted(remapped.items()))
                combined_stdout.append(f"Normalized backend anomaly types: {aliases}\n")
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
                vlm_model=runtime.get("vlm_model_name") or "selector_not_vlm",
                image_model=runtime["image_model_name"],
                grid_layout=generation_cfg.get("grid_layout", "4x4"),
                edit_bbox_expand_ratio=float(generation_cfg.get("edit_bbox_expand_ratio", 0.20)),
                crop_expand_ratio=float(generation_cfg.get("crop_expand_ratio", 0.03)),
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
