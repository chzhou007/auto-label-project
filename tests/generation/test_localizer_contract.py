from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from PIL import Image

from .conftest import assert_bbox_valid, result_to_dict


def import_required(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"缺少必需模块 {module_name}，请先完成 Localizer 抽象层实现: {exc}")


def test_localization_result_contract_fields() -> None:
    base = import_required("autolabel.modules.generation.localizers.base")
    assert hasattr(base, "LocalizationResult"), "base.py 需要定义 LocalizationResult"
    result = base.LocalizationResult(
        success=True,
        final_bbox=[10, 20, 80, 100],
        mask_path="/tmp/mask.png",
        method="rgb_diff",
        metrics={"score": 0.9},
        reason=None,
        fallback_used=False,
    )
    data = result_to_dict(result)
    for field in ["success", "final_bbox", "mask_path", "method", "metrics", "reason", "fallback_used"]:
        assert field in data
    assert data["success"] is True
    assert data["method"] == "rgb_diff"


def test_registry_can_create_core_localizers() -> None:
    registry = import_required("autolabel.modules.generation.localizers.registry")
    assert hasattr(registry, "create_localizer"), "registry.py 需要提供 create_localizer(name, **kwargs)"
    rgb = registry.create_localizer("rgb_diff")
    pgcd = registry.create_localizer("pgcd_lpips")
    assert hasattr(rgb, "localize")
    assert hasattr(pgcd, "localize")


def test_registry_rejects_unknown_localizer() -> None:
    registry = import_required("autolabel.modules.generation.localizers.registry")
    with pytest.raises((ValueError, KeyError)):
        registry.create_localizer("unknown_localizer")


def test_localizer_result_is_pipeline_compatible(simple_change_pair) -> None:
    registry = import_required("autolabel.modules.generation.localizers.registry")
    localizer = registry.create_localizer("rgb_diff")
    mask_path = Path(simple_change_pair["output_dir"]) / "contract_mask.png"
    result = localizer.localize(
        original_image_path=str(simple_change_pair["original_path"]),
        generated_image_path=str(simple_change_pair["generated_path"]),
        prompt_box=list(simple_change_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(mask_path),
        debug_dir=str(simple_change_pair["output_dir"]),
    )
    data = result_to_dict(result)
    assert isinstance(data["success"], bool)
    assert data["method"]
    assert isinstance(data["metrics"], dict)
    if data["success"]:
        assert_bbox_valid(data["final_bbox"], simple_change_pair["image_size"])
        assert Path(data["mask_path"]).exists()
        assert Image.open(data["mask_path"]).size == simple_change_pair["image_size"]
