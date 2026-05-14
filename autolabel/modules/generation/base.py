from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class GenerationRunResult:
    returncode: int
    output_root: Path
    stdout: str = ""
    stderr: str = ""
    skipped: bool = False


class GenerationModule(Protocol):
    def run(
        self,
        tasks_csv: str | Path,
        image_root: str | Path,
        output_root: str | Path,
        dry_run: bool = False,
        skip_existing: bool = False,
        limit: int | None = None,
    ) -> GenerationRunResult:
        ...
