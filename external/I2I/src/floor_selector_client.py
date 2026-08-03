from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from utils import resolve_image_path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_mmseg_floor_prepass(
    tasks: list[dict[str, str]],
    image_root: str,
    dirs: dict[str, Path],
    *,
    python_executable: str,
    worker_path: str,
    config_path: str,
    checkpoint_path: str,
    device: str,
    model_name: str,
    road_class_id: int,
    line_class_id: int,
    box_size: int,
    road_coverage_min: float,
    line_coverage_max: float,
    dry_run: bool,
) -> dict[str, dict[str, Any]]:
    requests_path = dirs["logs"] / "floor_selection_requests.jsonl"
    results_path = dirs["logs"] / "floor_selection.jsonl"
    requests = []
    for task in tasks:
        sample_id = str(task["sample_id"])
        requests.append(
            {
                "sample_id": sample_id,
                "image_path": str(resolve_image_path(task, image_root).resolve()),
                "mask_path": str((dirs["floor_masks"] / f"{sample_id}_road.png").resolve()),
                "overlay_path": str((dirs["floor_overlays"] / f"{sample_id}_floor_box.jpg").resolve()),
                "box_size": int(box_size),
            }
        )
    _write_jsonl(requests_path, requests)

    command = [
        str(python_executable),
        str(Path(worker_path).resolve()),
        "--requests-jsonl",
        str(requests_path.resolve()),
        "--results-jsonl",
        str(results_path.resolve()),
        "--config",
        str(Path(config_path).resolve()),
        "--checkpoint",
        str(Path(checkpoint_path).resolve()) if checkpoint_path else "",
        "--device",
        str(device),
        "--model-name",
        str(model_name),
        "--road-class-id",
        str(int(road_class_id)),
        "--line-class-id",
        str(int(line_class_id)),
        "--box-size",
        str(int(box_size)),
        "--road-coverage-min",
        str(float(road_coverage_min)),
        "--line-coverage-max",
        str(float(line_coverage_max)),
    ]
    if dry_run:
        command.append("--dry-run")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0 and not results_path.exists():
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(f"MMSeg floor prepass failed: {detail}")
    rows = _read_jsonl(results_path)
    failed = [row for row in rows if row.get("status") == "failed"]
    if failed:
        preview = "; ".join(f"{row.get('sample_id')}: {row.get('error')}" for row in failed[:5])
        raise RuntimeError(f"MMSeg floor prepass failed for {len(failed)} images: {preview}")
    return {str(row["sample_id"]): row for row in rows}
