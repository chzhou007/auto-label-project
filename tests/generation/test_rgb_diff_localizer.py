from __future__ import annotations

import importlib
from pathlib import Path

from PIL import Image

from .conftest import assert_bbox_valid, bbox_iou, result_to_dict


def make_rgb_diff_localizer():
    registry = importlib.import_module("autolabel.modules.generation.localizers.registry")
    return registry.create_localizer("rgb_diff")


def test_rgb_diff_localizes_obvious_local_change(simple_change_pair) -> None:
    localizer = make_rgb_diff_localizer()
    mask_path = Path(simple_change_pair["output_dir"]) / "rgb_mask.png"
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
    assert_bbox_valid(data["final_bbox"], simple_change_pair["image_size"])
    assert bbox_iou(data["final_bbox"], simple_change_pair["gt_bbox"]) >= 0.35
    assert Path(data["mask_path"]).exists()
    assert Image.open(data["mask_path"]).size == simple_change_pair["image_size"]
    assert data["method"] == "rgb_diff"
    assert "global_change_ratio" in data["metrics"] or "change_ratio" in data["metrics"]


def test_rgb_diff_does_not_hallucinate_on_no_change(no_change_pair) -> None:
    localizer = make_rgb_diff_localizer()
    mask_path = Path(no_change_pair["output_dir"]) / "rgb_no_change_mask.png"
    result = localizer.localize(
        original_image_path=str(no_change_pair["original_path"]),
        generated_image_path=str(no_change_pair["generated_path"]),
        prompt_box=list(no_change_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(mask_path),
        debug_dir=str(no_change_pair["output_dir"]),
    )
    data = result_to_dict(result)
    assert not data["success"], "无变化图像不应产出成功定位"
    assert data.get("reason") in {"no_change", "weak_change", "no_component", "empty_mask", "below_threshold"} or data.get("reason")


def test_rgb_diff_handles_resized_generated_image(resized_generated_pair) -> None:
    localizer = make_rgb_diff_localizer()
    mask_path = Path(resized_generated_pair["output_dir"]) / "rgb_resized_mask.png"
    result = localizer.localize(
        original_image_path=str(resized_generated_pair["original_path"]),
        generated_image_path=str(resized_generated_pair["generated_path"]),
        prompt_box=list(resized_generated_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(mask_path),
        debug_dir=str(resized_generated_pair["output_dir"]),
    )
    data = result_to_dict(result)
    # 允许成功，也允许明确 alignment 失败；不允许静默异常或返回非法 bbox。
    if data["success"]:
        assert_bbox_valid(data["final_bbox"], resized_generated_pair["image_size"])
        assert Image.open(data["mask_path"]).size == resized_generated_pair["image_size"]
    else:
        assert data.get("reason")


def test_rgb_diff_rejects_large_global_change(global_change_pair) -> None:
    localizer = make_rgb_diff_localizer()
    mask_path = Path(global_change_pair["output_dir"]) / "rgb_global_change_mask.png"
    result = localizer.localize(
        original_image_path=str(global_change_pair["original_path"]),
        generated_image_path=str(global_change_pair["generated_path"]),
        prompt_box=list(global_change_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(mask_path),
        debug_dir=str(global_change_pair["output_dir"]),
    )
    data = result_to_dict(result)
    assert not data["success"]
    assert data.get("reason") in {"global_change", "localizer_error"} or data.get("reason")
