from __future__ import annotations

import importlib.util
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "external" / "floor_segmentation" / "worker.py"
I2I_SRC = ROOT / "external" / "I2I" / "src"


def _load_worker_module():
    spec = importlib.util.spec_from_file_location("floor_segmentation_worker", WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_floor_box_uses_only_road_and_is_deterministic() -> None:
    worker = _load_worker_module()
    labels = np.zeros((600, 800), dtype=np.uint8)
    labels[260:590, 80:760] = 2
    labels[400:405, 80:760] = 1

    first = worker.select_floor_box(labels, "sample-1", box_size=200)
    second = worker.select_floor_box(labels, "sample-1", box_size=200)

    assert first["status"] == "selected"
    assert first["bbox"] == second["bbox"]
    x1, y1, x2, y2 = first["bbox"]
    selected = labels[y1:y2, x1:x2]
    assert float((selected == 2).mean()) >= 0.95
    assert float((selected == 1).mean()) <= 0.05
    assert 0 < x1 < x2 < labels.shape[1]
    assert 0 < y1 < y2 < labels.shape[0]


def test_floor_box_skips_background_and_line_only_masks() -> None:
    worker = _load_worker_module()
    labels = np.zeros((500, 700), dtype=np.uint8)
    labels[200:480, 50:650] = 1

    result = worker.select_floor_box(labels, "sample-no-road", box_size=200)

    assert result["status"] == "skipped"
    assert result["reason"] == "no_visible_floor_region"


def test_floor_box_skips_when_road_cannot_fit_required_size() -> None:
    worker = _load_worker_module()
    labels = np.zeros((500, 700), dtype=np.uint8)
    labels[250:390, 100:600] = 2

    result = worker.select_floor_box(labels, "sample-narrow-road", box_size=200)

    assert result["status"] == "skipped"
    assert result["reason"] == "no_visible_floor_region"


def test_floor_prepass_worker_loads_once_for_batch_in_dry_run(tmp_path: Path) -> None:
    sys.path.insert(0, str(I2I_SRC))
    try:
        from floor_selector_client import run_mmseg_floor_prepass
        from utils import ensure_output_dirs
    finally:
        sys.path.remove(str(I2I_SRC))

    image_root = tmp_path / "images"
    image_root.mkdir()
    tasks = []
    for index in range(2):
        image_path = image_root / f"image_{index}.jpg"
        Image.new("RGB", (640, 480), (110, 110, 110)).save(image_path)
        tasks.append(
            {
                "sample_id": f"sample_{index}",
                "image_id": f"image_{index}",
                "image_uri": image_path.name,
                "anomaly_type": "water_leak",
                "source_type": "manual_upload",
            }
        )
    dirs = ensure_output_dirs(tmp_path / "output")

    results = run_mmseg_floor_prepass(
        tasks,
        str(image_root),
        dirs,
        python_executable=sys.executable,
        worker_path=str(WORKER_PATH),
        config_path=str(ROOT / "external" / "floor_segmentation" / "configs" / "segformer_mit-b0_roadline_inference.py"),
        checkpoint_path="",
        device="cpu",
        model_name="segformer-test",
        road_class_id=2,
        line_class_id=1,
        box_size=200,
        road_coverage_min=0.95,
        line_coverage_max=0.05,
        dry_run=True,
    )

    assert set(results) == {"sample_0", "sample_1"}
    assert all(result["status"] == "selected" for result in results.values())
    assert len(list(dirs["floor_masks"].glob("*.png"))) == 2
    assert len(list(dirs["floor_overlays"].glob("*.jpg"))) == 2
    rows = [
        json.loads(line)
        for line in (dirs["logs"] / "floor_selection.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2


def test_i2i_mmseg_dry_run_bypasses_qwen_and_writes_floor_metadata(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    reference_root = tmp_path / "references"
    image_root.mkdir()
    reference_root.mkdir()
    image_path = image_root / "source.jpg"
    Image.new("RGB", (640, 480), (115, 115, 115)).save(image_path)
    Image.new("RGB", (256, 256), (85, 95, 100)).save(reference_root / "water.png")
    tasks_path = tmp_path / "tasks.csv"
    with open(tasks_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["sample_id", "image_id", "image_uri", "anomaly_type", "source_type"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample_floor",
                "image_id": "image_floor",
                "image_uri": image_path.name,
                "anomaly_type": "water_leak",
                "source_type": "manual_upload",
            }
        )
    output_root = tmp_path / "output"
    env = os.environ.copy()
    for name in ("QWEN397B_API_KEY", "DASHSCOPE_API_KEY", "QWEN397B_API_URL", "DASHSCOPE_VLM_ENDPOINT"):
        env.pop(name, None)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "external" / "I2I" / "src" / "main.py"),
            "--tasks",
            str(tasks_path),
            "--image-root",
            str(image_root),
            "--output-root",
            str(output_root),
            "--selector-backend",
            "mmseg_floor_selector",
            "--selector-model",
            "segformer-test",
            "--image-model",
            "doubao-seedream-5-0-pro-260628",
            "--floor-python",
            sys.executable,
            "--floor-worker",
            str(WORKER_PATH),
            "--floor-config",
            str(ROOT / "external" / "floor_segmentation" / "configs" / "segformer_mit-b0_roadline_inference.py"),
            "--floor-device",
            "cpu",
            "--seedream-mode",
            "boxed_fusion",
            "--water-reference-dir",
            str(reference_root),
            "--red-box-min-size",
            "200",
            "--red-box-max-size",
            "200",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert not list((output_root / "debug" / "grid_previews").glob("*"))
    assert not list((output_root / "logs").glob("*_vlm_response.json"))
    metadata = json.loads((output_root / "metadata" / "sample_floor.json").read_text(encoding="utf-8"))
    params = metadata["objects"][0]["geometry_detail"]["generation_params"]
    assert params["selection_backend"] == "mmseg_floor_selector"
    assert params["selected_grid"] is None
    assert params["selected_floor_bbox"] == params["red_box_bbox"]
    assert params["water_bbox_floor_overlap"] >= 0.8
    assert Path(params["floor_mask_uri"]).is_file()
    summary = json.loads((output_root / "logs" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["selector_inference_count"] == 1
    assert summary["selector_selected_count"] == 1
    assert summary["skipped_no_floor_region"] == 0


def test_mmseg_preflight_reports_missing_trained_checkpoint(tmp_path: Path, monkeypatch) -> None:
    from autolabel.config_loader import load_config
    from autolabel.modules.generation.preflight import GenerationPreflightError, run_generation_preflight

    image_root = tmp_path / "images"
    reference_root = tmp_path / "references"
    image_root.mkdir()
    reference_root.mkdir()
    image_path = image_root / "source.jpg"
    Image.new("RGB", (640, 480), (100, 100, 100)).save(image_path)
    Image.new("RGB", (64, 64), (80, 90, 100)).save(reference_root / "water.png")
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_id",
                "image_id",
                "image_uri",
                "anomaly_type",
                "source_type",
                "task_mode",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample",
                "image_id": "image",
                "image_uri": image_path.name,
                "anomaly_type": "water_leak",
                "source_type": "manual_upload",
                "task_mode": "generation",
            }
        )
    config = load_config(ROOT / "configs" / "autolabel.yaml")
    config["generation"]["seedream"] = {
        "mode": "boxed_fusion",
        "allow_experimental_generation": True,
        "water_reference_dir": str(reference_root),
    }
    config["generation"]["floor_selector"] = {
        "python": sys.executable,
        "checkpoint": str(tmp_path / "best_mIoU_iter_3000.pth"),
        "device": "cpu",
    }
    monkeypatch.setenv("ARK_API_KEY", "test-key")

    with pytest.raises(GenerationPreflightError, match="checkpoint not found"):
        run_generation_preflight(
            config,
            tasks_csv=manifest,
            image_root=image_root,
            output_root=tmp_path / "output",
            require_credentials=True,
        )


def test_mmseg_preflight_does_not_require_qwen_credentials(tmp_path: Path, monkeypatch) -> None:
    from autolabel.config_loader import load_config
    from autolabel.modules.generation.preflight import run_generation_preflight

    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "source.jpg"
    Image.new("RGB", (640, 480), (100, 100, 100)).save(image_path)
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_id",
                "image_id",
                "image_uri",
                "anomaly_type",
                "source_type",
                "task_mode",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample",
                "image_id": "image",
                "image_uri": image_path.name,
                "anomaly_type": "water_leak",
                "source_type": "manual_upload",
                "task_mode": "generation",
            }
        )
    for name in ("QWEN397B_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config = load_config(ROOT / "configs" / "autolabel.yaml")

    report = run_generation_preflight(
        config,
        tasks_csv=manifest,
        image_root=image_root,
        output_root=tmp_path / "output",
        require_credentials=False,
    )

    assert report["runtime_by_anomaly"]["water_leak"]["selector_backend"] == "mmseg_floor_selector"
    assert len(report["credential_checks"]) == 1
    assert report["credential_checks"][0]["api_key_env"] == "ARK_API_KEY"


def test_skipped_floor_selection_never_calls_seedream(tmp_path: Path) -> None:
    sys.path.insert(0, str(I2I_SRC))
    try:
        from config import PipelineConfig
        from main import process_task
        from utils import ensure_output_dirs
    finally:
        sys.path.remove(str(I2I_SRC))

    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "source.jpg"
    Image.new("RGB", (640, 480), (110, 110, 110)).save(image_path)
    dirs = ensure_output_dirs(tmp_path / "output")
    cfg = PipelineConfig(
        tasks=str(tmp_path / "tasks.csv"),
        image_root=str(image_root),
        output_root=str(tmp_path / "output"),
        selector_backend="mmseg_floor_selector",
        selector_model="segformer-test",
        seedream_mode="boxed_fusion",
    )

    class NeverCalledWan:
        model_call_count = 0
        model_generated_count = 0

        def edit_image_with_wan(self, *args, **kwargs):
            raise AssertionError("Seedream must not be called for a skipped floor selection")

    result = process_task(
        {
            "sample_id": "sample_skip",
            "image_id": "image_skip",
            "image_uri": image_path.name,
            "anomaly_type": "water_leak",
            "source_type": "manual_upload",
        },
        cfg,
        dirs,
        None,
        NeverCalledWan(),
        selector_result={
            "sample_id": "sample_skip",
            "status": "skipped",
            "reason": "no_visible_floor_region",
        },
    )

    assert result is None
    assert not list(dirs["generated_images"].glob("*"))


def test_water_mask_floor_overlap_rejects_non_floor_pixels(tmp_path: Path) -> None:
    sys.path.insert(0, str(I2I_SRC))
    try:
        from main import _water_mask_floor_overlap
    finally:
        sys.path.remove(str(I2I_SRC))

    water_path = tmp_path / "water.png"
    floor_path = tmp_path / "floor.png"
    water = np.zeros((100, 100), dtype=np.uint8)
    water[20:80, 20:80] = 255
    floor = np.zeros((100, 100), dtype=np.uint8)
    floor[20:80, 20:50] = 255
    Image.fromarray(water, mode="L").save(water_path)
    Image.fromarray(floor, mode="L").save(floor_path)

    assert _water_mask_floor_overlap(water_path, floor_path) == pytest.approx(0.5)
