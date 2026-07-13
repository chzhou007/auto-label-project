from __future__ import annotations

from typing import Any

from .components import bbox_center, bbox_iou

DEFAULT_WEIGHTS = {
    "area": 0.30,
    "lpips": 0.40,
    "iou": 0.20,
    "distance": 0.10,
}


def score_component(
    component: dict[str, Any],
    prompt_box: list[int] | tuple[int, int, int, int],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    resolved_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        resolved_weights.update(weights)

    prompt_area = max(1.0, float((prompt_box[2] - prompt_box[0]) * (prompt_box[3] - prompt_box[1])))
    area = max(1.0, float(component.get("area", 0)))
    area_score = min(area / prompt_area, 1.0)
    lpips_score = float(component.get("lpips_score", component.get("score", 0.0)) or 0.0)
    prompt_iou_score = bbox_iou(component["bbox"], prompt_box)

    component_center = bbox_center(component["bbox"])
    prompt_center = bbox_center(prompt_box)
    dx = component_center[0] - prompt_center[0]
    dy = component_center[1] - prompt_center[1]
    prompt_diag = max(1.0, ((prompt_box[2] - prompt_box[0]) ** 2 + (prompt_box[3] - prompt_box[1]) ** 2) ** 0.5)
    distance_penalty = min(((dx * dx + dy * dy) ** 0.5) / prompt_diag, 1.0)

    component_score = (
        resolved_weights["area"] * area_score
        + resolved_weights["lpips"] * lpips_score
        + resolved_weights["iou"] * prompt_iou_score
        - resolved_weights["distance"] * distance_penalty
    )
    enriched = dict(component)
    enriched.update(
        {
            "area_score": float(area_score),
            "lpips_score": float(lpips_score),
            "prompt_iou_score": float(prompt_iou_score),
            "distance_penalty": float(distance_penalty),
            "component_score": float(component_score),
        }
    )
    return enriched


def select_best_component(
    components: list[dict[str, Any]],
    prompt_box: list[int] | tuple[int, int, int, int],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not components:
        raise ValueError("components must not be empty")
    scored = [score_component(component, prompt_box, weights=weights) for component in components]
    scored.sort(key=lambda item: (item["component_score"], item["prompt_iou_score"], item["area"]), reverse=True)
    return scored[0]
