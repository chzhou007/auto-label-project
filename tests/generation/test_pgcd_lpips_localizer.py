from __future__ import annotations

import importlib
import json
from pathlib import Path

from PIL import Image

from .conftest import FakeHeatmapBackend, ZeroHeatmapBackend, assert_bbox_valid, bbox_iou, result_to_dict


def make_pgcd_localizer(**kwargs):
    registry = importlib.import_module("autolabel.modules.generation.localizers.registry")
    return registry.create_localizer("pgcd_lpips", **kwargs)


def test_pgcd_lpips_localizes_from_injected_heatmap(simple_change_pair) -> None:
    backend = FakeHeatmapBackend([simple_change_pair["gt_bbox"]], image_size=simple_change_pair["image_size"])
    localizer = make_pgcd_localizer(heatmap_backend=backend, min_component_area=100, debug=True)
    mask_path = Path(simple_change_pair["output_dir"]) / "pgcd_mask.png"
    result = localizer.localize(
        original_image_path=str(simple_change_pair["original_path"]),
        generated_image_path=str(simple_change_pair["generated_path"]),
        prompt_box=list(simple_change_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(mask_path),
        debug_dir=str(simple_change_pair["output_dir"]),
    )
    data = result_to_dict(result)
    assert data["success"], data.get("reason")
    assert data["method"] in {"pgcd_lpips", "pgcd_lpips_cc"}
    assert_bbox_valid(data["final_bbox"], simple_change_pair["image_size"])
    assert bbox_iou(data["final_bbox"], simple_change_pair["gt_bbox"]) >= 0.60
    assert Path(data["mask_path"]).exists()
    assert Image.open(data["mask_path"]).size == simple_change_pair["image_size"]
    for key in ["pgcd_component_score", "pgcd_selected_component_count", "pgcd_heatmap_threshold", "pgcd_heatmap_p95"]:
        assert key in data["metrics"]
    assert data["metrics"]["pgcd_heatmap_p95"] >= 0.0
    assert data["metrics"]["pgcd_heatmap_path"] is not None
    assert Path(data["metrics"]["pgcd_heatmap_path"]).exists()


def test_pgcd_prompt_prior_ignores_larger_outside_change(prompt_prior_pair) -> None:
    # 同时给 heatmap 两个热点：外部大扰动 + prompt_box 内异常。
    backend = FakeHeatmapBackend(
        [prompt_prior_pair["outside_change_bbox"], prompt_prior_pair["gt_bbox"]],
        image_size=prompt_prior_pair["image_size"],
    )
    localizer = make_pgcd_localizer(heatmap_backend=backend, min_component_area=100, debug=True)
    mask_path = Path(prompt_prior_pair["output_dir"]) / "pgcd_prior_mask.png"
    result = localizer.localize(
        original_image_path=str(prompt_prior_pair["original_path"]),
        generated_image_path=str(prompt_prior_pair["generated_path"]),
        prompt_box=list(prompt_prior_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(mask_path),
        debug_dir=str(prompt_prior_pair["output_dir"]),
    )
    data = result_to_dict(result)
    assert data["success"], data.get("reason")
    inside_iou = bbox_iou(data["final_bbox"], prompt_prior_pair["gt_bbox"])
    outside_iou = bbox_iou(data["final_bbox"], prompt_prior_pair["outside_change_bbox"])
    assert inside_iou > outside_iou, "Prompt Box Prior 应优先选择 prompt_box 内异常组件"


def test_pgcd_zero_heatmap_returns_failure(no_change_pair) -> None:
    localizer = make_pgcd_localizer(heatmap_backend=ZeroHeatmapBackend(), min_component_area=100, debug=True)
    mask_path = Path(no_change_pair["output_dir"]) / "pgcd_zero_mask.png"
    result = localizer.localize(
        original_image_path=str(no_change_pair["original_path"]),
        generated_image_path=str(no_change_pair["generated_path"]),
        prompt_box=list(no_change_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(mask_path),
        debug_dir=str(no_change_pair["output_dir"]),
    )
    data = result_to_dict(result)
    assert not data["success"]
    assert data.get("reason") in {"no_change", "no_component", "empty_heatmap", "below_threshold"} or data.get("reason")


def test_pgcd_debug_artifacts_are_written(simple_change_pair) -> None:
    backend = FakeHeatmapBackend([simple_change_pair["gt_bbox"]], image_size=simple_change_pair["image_size"])
    localizer = make_pgcd_localizer(heatmap_backend=backend, min_component_area=100, debug=True)
    debug_dir = Path(simple_change_pair["output_dir"]) / "debug"
    debug_dir.mkdir()
    mask_path = Path(simple_change_pair["output_dir"]) / "pgcd_debug_mask.png"
    result = localizer.localize(
        original_image_path=str(simple_change_pair["original_path"]),
        generated_image_path=str(simple_change_pair["generated_path"]),
        prompt_box=list(simple_change_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(mask_path),
        debug_dir=str(debug_dir),
    )
    data = result_to_dict(result)
    assert data["success"], data.get("reason")
    names = {p.name for p in debug_dir.iterdir()}
    assert any("heatmap" in name for name in names), names
    assert any("component" in name for name in names), names
    selected_component_path = debug_dir / "pgcd_selected_component.json"
    components_path = debug_dir / "pgcd_components.json"
    assert selected_component_path.exists()
    assert components_path.exists()

    selected_component = json.loads(selected_component_path.read_text(encoding="utf-8"))
    components = json.loads(components_path.read_text(encoding="utf-8"))
    assert "component_score" in selected_component
    assert "prompt_iou_score" in selected_component
    assert components
    assert "component_score" in components[0]
    assert "prompt_iou_score" in components[0]


def test_pgcd_handles_resized_generated_image(resized_generated_pair) -> None:
    backend = FakeHeatmapBackend([resized_generated_pair["gt_bbox"]], image_size=resized_generated_pair["image_size"])
    localizer = make_pgcd_localizer(heatmap_backend=backend, min_component_area=100, debug=True)
    mask_path = Path(resized_generated_pair["output_dir"]) / "pgcd_resized_mask.png"
    result = localizer.localize(
        original_image_path=str(resized_generated_pair["original_path"]),
        generated_image_path=str(resized_generated_pair["generated_path"]),
        prompt_box=list(resized_generated_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(mask_path),
        debug_dir=str(resized_generated_pair["output_dir"]),
    )
    data = result_to_dict(result)
    assert data["success"], data.get("reason")
    assert_bbox_valid(data["final_bbox"], resized_generated_pair["image_size"])
    assert Image.open(data["mask_path"]).size == resized_generated_pair["image_size"]
    assert data["metrics"]["alignment_resize_applied"] is True
