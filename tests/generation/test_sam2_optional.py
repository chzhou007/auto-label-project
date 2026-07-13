from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from PIL import Image

import pytest

from .conftest import FakeHeatmapBackend, result_to_dict


def test_sam2_localizer_is_optional_and_can_fallback(simple_change_pair) -> None:
    registry = importlib.import_module("autolabel.modules.generation.localizers.registry")
    try:
        localizer = registry.create_localizer("pgcd_lpips_sam2", sam2_required=False)
    except ModuleNotFoundError:
        pytest.fail("pgcd_lpips_sam2 创建时不应因 SAM2 缺失直接崩溃；应支持 optional fallback")
    result = localizer.localize(
        original_image_path=str(simple_change_pair["original_path"]),
        generated_image_path=str(simple_change_pair["generated_path"]),
        prompt_box=list(simple_change_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(simple_change_pair["output_dir"] / "sam2_optional_mask.png"),
        debug_dir=str(simple_change_pair["output_dir"]),
    )
    data = result_to_dict(result)
    assert "sam2" in data["method"] or data["method"] in {"pgcd_lpips", "pgcd_lpips_cc", "rgb_diff"}
    if not data["success"]:
        assert data.get("reason")


class FakeSAM2Backend:
    def __init__(self, bbox: list[int], image_size: tuple[int, int]):
        self.bbox = bbox
        self.image_size = image_size

    def refine_mask(self, image_path, coarse_bbox, coarse_mask_path=None):
        mask = np.zeros((self.image_size[1], self.image_size[0]), dtype=bool)
        x1, y1, x2, y2 = self.bbox
        mask[y1:y2, x1:x2] = True
        return mask, list(self.bbox)


class FailingSAM2Backend:
    def refine_mask(self, image_path, coarse_bbox, coarse_mask_path=None):
        raise RuntimeError("sam2 refine failed")


def test_sam2_backend_can_refine_mask_and_bbox(simple_change_pair) -> None:
    registry = importlib.import_module("autolabel.modules.generation.localizers.registry")
    refined_bbox = [104, 98, 148, 130]
    localizer = registry.create_localizer(
        "pgcd_lpips_sam2",
        sam2_required=False,
        sam2_backend=FakeSAM2Backend(refined_bbox, simple_change_pair["image_size"]),
        heatmap_backend=FakeHeatmapBackend([tuple(simple_change_pair["gt_bbox"])], image_size=simple_change_pair["image_size"]),
        min_component_area=100,
    )
    mask_path = Path(simple_change_pair["output_dir"]) / "sam2_refined_mask.png"
    result = localizer.localize(
        original_image_path=str(simple_change_pair["original_path"]),
        generated_image_path=str(simple_change_pair["generated_path"]),
        prompt_box=list(simple_change_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(mask_path),
        debug_dir=str(simple_change_pair["output_dir"]),
    )
    data = result_to_dict(result)
    assert data["success"] is True
    assert data["method"] == "pgcd_lpips_sam2"
    assert data["metrics"]["sam2_enabled"] is True
    assert data["metrics"]["sam2_model"] == "tiny"
    assert data["metrics"]["sam2_success"] is True
    assert data["final_bbox"] == refined_bbox
    assert data["metrics"]["sam2_refined"] is True
    assert data["metrics"]["sam2_refined_area_ratio"] > 0.0
    assert data["metrics"]["sam2_bbox_delta_ratio"] >= 0.0
    assert Image.open(data["mask_path"]).size == simple_change_pair["image_size"]


def test_sam2_failure_keeps_coarse_pgcd_result(simple_change_pair) -> None:
    registry = importlib.import_module("autolabel.modules.generation.localizers.registry")
    localizer = registry.create_localizer(
        "pgcd_lpips_sam2",
        sam2_required=False,
        sam2_backend=FailingSAM2Backend(),
        heatmap_backend=FakeHeatmapBackend([tuple(simple_change_pair["gt_bbox"])], image_size=simple_change_pair["image_size"]),
        min_component_area=100,
    )
    mask_path = Path(simple_change_pair["output_dir"]) / "sam2_fail_mask.png"
    result = localizer.localize(
        original_image_path=str(simple_change_pair["original_path"]),
        generated_image_path=str(simple_change_pair["generated_path"]),
        prompt_box=list(simple_change_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(mask_path),
        debug_dir=str(simple_change_pair["output_dir"]),
    )
    data = result_to_dict(result)
    assert data["success"] is True
    assert data["method"] == "pgcd_lpips_sam2_fallback_coarse"
    assert data["metrics"]["sam2_enabled"] is True
    assert data["metrics"]["sam2_model"] == "tiny"
    assert data["metrics"]["sam2_success"] is False
    assert data["metrics"]["sam2_refined"] is False
    assert data["metrics"]["sam2_error"] == "sam2 refine failed"
