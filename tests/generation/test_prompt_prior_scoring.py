from __future__ import annotations

import importlib

import pytest


def get_prompt_prior_module():
    try:
        return importlib.import_module("autolabel.modules.generation.localizers.prompt_prior")
    except ModuleNotFoundError as exc:
        pytest.fail(f"缺少 prompt_prior.py，请实现 component scoring: {exc}")


def test_component_closer_to_prompt_center_scores_higher() -> None:
    prior = get_prompt_prior_module()
    assert hasattr(prior, "score_component"), "prompt_prior.py 需要提供 score_component(component, prompt_box, weights=None)"
    prompt_box = [80, 80, 170, 150]
    near_component = {
        "bbox": [95, 95, 145, 130],
        "area": 1750,
        "lpips_score": 0.75,
    }
    far_component = {
        "bbox": [10, 10, 60, 45],
        "area": 1750,
        "lpips_score": 0.75,
    }
    near_score = prior.score_component(near_component, prompt_box)["component_score"]
    far_score = prior.score_component(far_component, prompt_box)["component_score"]
    assert near_score > far_score


def test_component_with_higher_prompt_iou_scores_higher() -> None:
    prior = get_prompt_prior_module()
    prompt_box = [80, 80, 170, 150]
    inside_component = {
        "bbox": [92, 92, 150, 132],
        "area": 2320,
        "lpips_score": 0.65,
    }
    outside_component = {
        "bbox": [178, 86, 236, 126],
        "area": 2320,
        "lpips_score": 0.65,
    }
    inside = prior.score_component(inside_component, prompt_box)
    outside = prior.score_component(outside_component, prompt_box)
    assert inside["prompt_iou_score"] > outside["prompt_iou_score"]
    assert inside["component_score"] > outside["component_score"]


def test_select_best_component_prefers_prompt_consistent_component() -> None:
    prior = get_prompt_prior_module()
    assert hasattr(prior, "select_best_component"), "prompt_prior.py 需要提供 select_best_component(components, prompt_box, **kwargs)"
    prompt_box = [80, 80, 170, 150]
    components = [
        {"bbox": [10, 12, 74, 62], "area": 3200, "lpips_score": 0.82},
        {"bbox": [100, 96, 152, 132], "area": 1872, "lpips_score": 0.70},
    ]
    selected = prior.select_best_component(components, prompt_box)
    assert selected["bbox"] == [100, 96, 152, 132]
    for key in ["area_score", "lpips_score", "prompt_iou_score", "distance_penalty", "component_score"]:
        assert key in selected
