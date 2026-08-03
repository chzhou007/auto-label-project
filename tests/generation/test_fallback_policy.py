from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from .conftest import result_to_dict


@dataclass
class FakeResult:
    success: bool
    final_bbox: list[int] | None
    mask_path: str | None
    method: str
    metrics: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    fallback_used: bool = False


class FailingLocalizer:
    def __init__(self):
        self.called = False

    def localize(self, **kwargs):
        self.called = True
        return FakeResult(False, None, None, "pgcd_lpips", {"x": 1}, "no_component")


class SuccessfulLocalizer:
    def __init__(self):
        self.called = False

    def localize(self, **kwargs):
        self.called = True
        return FakeResult(True, [10, 20, 60, 80], kwargs.get("mask_output_path"), "rgb_diff", {"y": 2}, None)


class ExceptionLocalizer:
    def __init__(self):
        self.called = False

    def localize(self, **kwargs):
        self.called = True
        raise RuntimeError("backend exploded")


class FailingFallbackLocalizer:
    def __init__(self):
        self.called = False

    def localize(self, **kwargs):
        self.called = True
        return FakeResult(False, None, None, "rgb_diff", {"z": 3}, "global_change")


def get_fallback_module():
    return importlib.import_module("autolabel.modules.generation.localizers.fallback")


def test_fallback_runs_when_primary_fails(simple_change_pair) -> None:
    fallback_module = get_fallback_module()
    primary = FailingLocalizer()
    fallback = SuccessfulLocalizer()
    policy = fallback_module.FallbackLocalizer(primary=primary, fallback=fallback)
    result = policy.localize(
        original_image_path=str(simple_change_pair["original_path"]),
        generated_image_path=str(simple_change_pair["generated_path"]),
        prompt_box=list(simple_change_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(simple_change_pair["output_dir"] / "fallback_mask.png"),
    )
    data = result_to_dict(result)
    assert primary.called is True
    assert fallback.called is True
    assert data["success"] is True
    assert data["fallback_used"] is True
    assert data["method"] == "rgb_diff"
    assert data["metrics"].get("primary_method") == "pgcd_lpips"
    assert data["metrics"].get("fallback_method") == "rgb_diff"


def test_fallback_not_called_when_primary_succeeds(simple_change_pair) -> None:
    fallback_module = get_fallback_module()
    primary = SuccessfulLocalizer()
    fallback = SuccessfulLocalizer()
    policy = fallback_module.FallbackLocalizer(primary=primary, fallback=fallback)
    result = policy.localize(
        original_image_path=str(simple_change_pair["original_path"]),
        generated_image_path=str(simple_change_pair["generated_path"]),
        prompt_box=list(simple_change_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(simple_change_pair["output_dir"] / "primary_mask.png"),
    )
    data = result_to_dict(result)
    assert primary.called is True
    assert fallback.called is False
    assert data["success"] is True
    assert data["fallback_used"] is False


def test_fallback_runs_when_primary_raises(simple_change_pair) -> None:
    fallback_module = get_fallback_module()
    primary = ExceptionLocalizer()
    fallback = SuccessfulLocalizer()
    policy = fallback_module.FallbackLocalizer(primary=primary, fallback=fallback)
    result = policy.localize(
        original_image_path=str(simple_change_pair["original_path"]),
        generated_image_path=str(simple_change_pair["generated_path"]),
        prompt_box=list(simple_change_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(simple_change_pair["output_dir"] / "exception_fallback_mask.png"),
    )
    data = result_to_dict(result)
    assert primary.called is True
    assert fallback.called is True
    assert data["success"] is True
    assert data["fallback_used"] is True
    assert data["metrics"].get("primary_reason") == "localizer_exception"


def test_fallback_preserves_both_failure_reasons(simple_change_pair) -> None:
    fallback_module = get_fallback_module()
    primary = FailingLocalizer()
    fallback = FailingFallbackLocalizer()
    policy = fallback_module.FallbackLocalizer(primary=primary, fallback=fallback)
    result = policy.localize(
        original_image_path=str(simple_change_pair["original_path"]),
        generated_image_path=str(simple_change_pair["generated_path"]),
        prompt_box=list(simple_change_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(simple_change_pair["output_dir"] / "double_fail_mask.png"),
    )
    data = result_to_dict(result)
    assert primary.called is True
    assert fallback.called is True
    assert data["success"] is False
    assert data["fallback_used"] is True
    assert data["reason"] == "global_change"
    assert data["metrics"].get("primary_reason") == "no_component"
    assert data["metrics"].get("fallback_method") == "rgb_diff"
