from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BBox = tuple[int, int, int, int]


@dataclass
class LocalizationInput:
    original_image_path: Path
    generated_image_path: Path
    prompt_box: list[int]
    anomaly_type: str
    output_mask_path: Path
    grid_bbox: list[int] | None = None
    sample_id: str | None = None
    object_id: str | None = None
    task_id: str | None = None
    attempt_role: str | None = None
    debug_dir: Path | None = None
    image_size: tuple[int, int] | None = None
    generation_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "LocalizationInput":
        return cls(
            original_image_path=Path(kwargs["original_image_path"]),
            generated_image_path=Path(kwargs["generated_image_path"]),
            prompt_box=[int(value) for value in kwargs["prompt_box"]],
            anomaly_type=str(kwargs.get("anomaly_type", "unknown")),
            output_mask_path=Path(kwargs.get("mask_output_path") or kwargs.get("output_mask_path")),
            grid_bbox=(
                [int(value) for value in kwargs["grid_bbox"]]
                if kwargs.get("grid_bbox") is not None
                else None
            ),
            sample_id=kwargs.get("sample_id"),
            object_id=kwargs.get("object_id"),
            task_id=kwargs.get("task_id"),
            attempt_role=kwargs.get("attempt_role"),
            debug_dir=Path(kwargs["debug_dir"]) if kwargs.get("debug_dir") else None,
            image_size=kwargs.get("image_size"),
            generation_params=dict(kwargs.get("generation_params") or {}),
        )

    @property
    def mask_output_path(self) -> Path:
        return self.output_mask_path


@dataclass
class ComponentRecord:
    component_id: int
    bbox: list[int]
    area: int
    center: list[float]
    aspect_ratio: float
    lpips_score: float | None = None
    area_score: float | None = None
    prompt_iou_score: float | None = None
    distance_penalty: float | None = None
    component_score: float | None = None
    reject_reason: str | None = None


@dataclass
class LocalizationResult:
    success: bool
    final_bbox: list[int] | None
    mask_path: str | Path | None
    method: str
    metrics: dict[str, Any]
    reason: str | None = None
    fallback_used: bool = False
    components: list[dict[str, Any]] = field(default_factory=list)
    selected_component_id: int | None = None
    debug_artifacts: dict[str, str] = field(default_factory=dict)


class BaseLocalizer:
    method_name = "base"

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)
        self.debug = bool(self.config.get("debug", False))

    def coerce_input(self, inp: LocalizationInput | None = None, **kwargs: Any) -> LocalizationInput:
        if inp is not None:
            return inp
        return LocalizationInput.from_kwargs(**kwargs)

    def localize(self, inp: LocalizationInput | None = None, **kwargs: Any) -> LocalizationResult:
        raise NotImplementedError
