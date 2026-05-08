from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path
from typing import Any

from ..utils import read_json
from ..validators import validate_sample_contract


class I2IGenerator:
    def __init__(self, i2i_project: str | Path) -> None:
        self.i2i_project = Path(i2i_project)

    @property
    def main_py(self) -> Path:
        return self.i2i_project / "src" / "main.py"

    def run(
        self,
        tasks_csv: str | Path,
        image_root: str | Path,
        output_root: str | Path,
        vlm_model: str = "aios-smart-eye-vlm",
        image_model: str = "wan2.7-image-pro",
        grid_layout: str = "4x4",
        edit_bbox_expand_ratio: float = 0.20,
        crop_expand_ratio: float = 0.10,
        workers: int = 1,
        dry_run: bool = False,
        skip_existing: bool = False,
        limit: int | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if not self.main_py.exists():
            raise FileNotFoundError(f"I2I entrypoint not found: {self.main_py}")

        cmd = [
            sys.executable,
            str(self.main_py),
            "--tasks",
            str(Path(tasks_csv).resolve()),
            "--image-root",
            str(Path(image_root).resolve()),
            "--output-root",
            str(Path(output_root).resolve()),
            "--vlm-model",
            vlm_model,
            "--image-model",
            image_model,
            "--grid-layout",
            grid_layout,
            "--edit-bbox-expand-ratio",
            str(edit_bbox_expand_ratio),
            "--crop-expand-ratio",
            str(crop_expand_ratio),
            "--workers",
            str(workers),
        ]
        if dry_run:
            cmd.append("--dry-run")
        if skip_existing:
            cmd.append("--skip-existing")
        if limit is not None:
            cmd.extend(["--limit", str(limit)])

        subprocess_env = os.environ.copy()
        if env:
            subprocess_env.update({key: value for key, value in env.items() if value not in ("", None)})

        return subprocess.run(
            cmd,
            cwd=str(self.i2i_project),
            text=True,
            capture_output=True,
            check=False,
            env=subprocess_env,
        )


def iter_generated_metadata(output_root: str | Path) -> list[Path]:
    metadata_dir = Path(output_root) / "metadata"
    if not metadata_dir.exists():
        return []
    return sorted(metadata_dir.glob("*.json"))


def load_generated_samples(output_root: str | Path, validate: bool = True) -> list[dict[str, Any]]:
    samples = []
    for path in iter_generated_metadata(output_root):
        sample = read_json(path)
        if validate:
            validate_sample_contract(sample)
        samples.append(sample)
    return samples
