from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from autolabel.modules.generation.localizers.base import LocalizationResult
from autolabel.modules.generation.metadata_builder import apply_localizer_postprocess
from autolabel.sample_factory import make_box
from autolabel.utils import read_json

from .conftest import add_water_patch, make_base_image


def _build_sample(generated_path: Path) -> dict:
    return {
        "sample_id": "sample_generation",
        "image_asset": {
            "image_id": "image_generation",
            "image_uri": str(generated_path),
            "width": 256,
            "height": 192,
            "source_type": "generated",
        },
        "objects": [
            {
                "object_id": "leak_000001",
                "object_type": "leakage_area",
                "box": make_box(80, 80, 170, 150),
                "geometry_source": "synthetic_generator",
                "geometry_model": {"model_name": "wan", "model_version": "test", "confidence": 0.9},
                "geometry_detail": {
                    "polygon": None,
                    "mask_uri": None,
                    "mask_format": None,
                    "generation_params": {
                        "prompt_box": [80, 80, 170, 150],
                    },
                },
                "crop": {
                    "crop_id": "leak_000001_crop",
                    "crop_uri": "pending",
                    "crop_box": None,
                    "crop_expand_ratio": None,
                    "is_valid_crop": False,
                },
                "classification": {
                    "multi_labels": [
                        {
                            "label_key": "anomaly_type",
                            "label_value": "water_leak",
                            "confidence": None,
                            "evidence": "generated label",
                        }
                    ],
                    "classifier_type": "vlm",
                    "classifier_name": "i2i_generation",
                    "classifier_version": "test",
                    "prompt_version": "test",
                    "raw_response": None,
                },
                "quality_check": None,
            }
        ],
        "workflow": {"workflow_status": "classified", "updated_time": "2026-07-01T00:00:00+08:00"},
        "export": {"export_format": "labelstudio", "export_status": "not_exported"},
    }


def _build_pipeline_config() -> dict:
    return {
        "generation": {"crop_expand_ratio": 0.05},
        "modules": {
            "generation": {
                "localizer": {
                    "primary": "rgb_diff",
                    "fallback": "none",
                    "debug": False,
                    "allow_quality_fallback": False,
                    "sidecar_eval": None,
                    "pgcd": {"min_component_area": 10, "max_global_change_ratio": 1.0},
                }
            }
        },
    }


def _write_box_mask(path: str | Path, bbox: tuple[int, int, int, int], size: tuple[int, int] = (256, 192)) -> None:
    image = Image.new("L", size, 0)
    x1, y1, x2, y2 = bbox
    for x in range(x1, x2):
        for y in range(y1, y2):
            image.putpixel((x, y), 255)
    image.save(path)


class StaticLocalizer:
    def __init__(self, method: str, bbox: tuple[int, int, int, int]) -> None:
        self.method_name = method
        self.method = method
        self.bbox = bbox

    def localize(self, **kwargs):
        _write_box_mask(kwargs["mask_output_path"], self.bbox)
        return LocalizationResult(
            success=True,
            final_bbox=list(self.bbox),
            mask_path=kwargs["mask_output_path"],
            method=self.method,
            metrics={"localizer_method": self.method},
        )


class StaticLocalizerWithArtifacts(StaticLocalizer):
    def localize(self, **kwargs):
        result = super().localize(**kwargs)
        debug_dir = Path(kwargs["debug_dir"])
        debug_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = debug_dir / "pgcd_heatmap.png"
        components_path = debug_dir / "pgcd_components.json"
        Image.new("L", (256, 192), 128).save(heatmap_path)
        components_path.write_text(json.dumps([{"component_score": 0.9}]), encoding="utf-8")
        result.metrics = {
            "localizer_method": self.method,
            "pgcd_heatmap_path": str(heatmap_path),
        }
        result.debug_artifacts = {
            "heatmap": str(heatmap_path),
            "components": str(components_path),
        }
        return result


def test_apply_localizer_postprocess_computes_real_metrics(tmp_path) -> None:
    original = make_base_image()
    generated = add_water_patch(original, (98, 96, 152, 132))
    original_path = tmp_path / "original.jpg"
    generated_path = tmp_path / "generated.jpg"
    original.save(original_path)
    generated.save(generated_path)

    sample = _build_sample(generated_path)
    config = _build_pipeline_config()

    processed, rows = apply_localizer_postprocess(
        sample,
        pipeline_config=config,
        source_row={"sample_id": "sample_generation", "image_uri": str(original_path), "anomaly_type": "water_leak"},
        processed_root=tmp_path / "processed",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["bbox_iou"] > 0.0
    assert row["precision"] > 0.0
    assert row["recall"] > 0.0
    assert row["passes_quality"] is True

    generation_params = processed["objects"][0]["geometry_detail"]["generation_params"]
    assert generation_params["quality"]["passes_quality"] is True
    assert generation_params["localizer"]["benchmark"]["bbox_iou"] > 0.0


def test_apply_localizer_postprocess_uses_quality_fallback(tmp_path) -> None:
    original = make_base_image()
    good_bbox = (98, 96, 152, 132)
    generated = add_water_patch(original, good_bbox)
    original_path = tmp_path / "original.jpg"
    generated_path = tmp_path / "generated.jpg"
    original.save(original_path)
    generated.save(generated_path)

    sample = _build_sample(generated_path)
    config = _build_pipeline_config()
    config["modules"]["generation"]["localizer"].update(
        {
            "primary": "pgcd_lpips",
            "fallback": "rgb_diff",
            "allow_quality_fallback": True,
            "quality": {
                "profiles": {
                    "water_leak": {
                        "background_preservation_min": 0.20,
                        "anomaly_visibility_min": 0.01,
                        "prompt_iou_min": 0.10,
                    }
                }
            },
        }
    )

    def fake_create_localizer(name: str, **_kwargs):
        if name == "pgcd_lpips":
            return StaticLocalizer("pgcd_lpips", (10, 10, 40, 40))
        if name == "rgb_diff":
            return StaticLocalizer("rgb_diff", good_bbox)
        raise AssertionError(name)

    with patch("autolabel.modules.generation.metadata_builder.create_localizer", side_effect=fake_create_localizer):
        processed, rows = apply_localizer_postprocess(
            sample,
            pipeline_config=config,
            source_row={"sample_id": "sample_generation", "image_uri": str(original_path), "anomaly_type": "water_leak"},
            processed_root=tmp_path / "processed",
        )

    localizer_info = processed["objects"][0]["geometry_detail"]["generation_params"]["localizer"]
    assert localizer_info["used"] == "rgb_diff"
    assert localizer_info["fallback_used"] is True
    assert localizer_info["fallback_trigger"] == "quality_failed"
    assert processed["objects"][0]["box"] == make_box(*good_bbox)
    assert len(rows) == 2
    assert rows[0]["success"] is False
    assert rows[0]["reason"].startswith("quality_failed:")
    assert rows[1]["success"] is True


def test_apply_localizer_postprocess_supports_multiple_sidecars(tmp_path) -> None:
    original = make_base_image()
    generated = add_water_patch(original, (98, 96, 152, 132))
    original_path = tmp_path / "original.jpg"
    generated_path = tmp_path / "generated.jpg"
    original.save(original_path)
    generated.save(generated_path)

    sample = _build_sample(generated_path)
    config = _build_pipeline_config()
    config["modules"]["generation"]["localizer"].update(
        {
            "primary": "rgb_diff",
            "sidecar_eval": "pgcd_lpips,pgcd_lpips_sam2",
        }
    )

    def fake_create_localizer(name: str, **_kwargs):
        return StaticLocalizer(name, (98, 96, 152, 132))

    with patch("autolabel.modules.generation.metadata_builder.create_localizer", side_effect=fake_create_localizer):
        processed, rows = apply_localizer_postprocess(
            sample,
            pipeline_config=config,
            source_row={"sample_id": "sample_generation", "image_uri": str(original_path), "anomaly_type": "water_leak"},
            processed_root=tmp_path / "processed",
        )

    localizer_info = processed["objects"][0]["geometry_detail"]["generation_params"]["localizer"]
    assert len(rows) == 3
    assert [row["attempt_role"] for row in rows].count("sidecar") == 2
    assert len(localizer_info["sidecars"]) == 2
    assert localizer_info["sidecars"][0]["name"] == "pgcd_lpips"
    assert localizer_info["sidecars"][1]["name"] == "pgcd_lpips_sam2"
    assert localizer_info["sidecar"]["name"] == "pgcd_lpips"


def test_apply_localizer_postprocess_persists_wan_response_path(tmp_path) -> None:
    original = make_base_image()
    generated = add_water_patch(original, (98, 96, 152, 132))
    original_path = tmp_path / "original.jpg"
    generated_path = tmp_path / "generated.jpg"
    original.save(original_path)
    generated.save(generated_path)

    wan_response_path = tmp_path / "wan_response.json"
    wan_response_path.write_text('{"job_id":"wan-job-1","status":"succeeded"}', encoding="utf-8")

    sample = _build_sample(generated_path)
    sample["objects"][0]["geometry_detail"]["generation_params"]["wan"] = {
        "bbox_list": [[80, 80, 170, 150]],
        "wan_response_path": str(wan_response_path),
    }
    config = _build_pipeline_config()

    with patch(
        "autolabel.modules.generation.metadata_builder.create_localizer",
        side_effect=lambda name, **_kwargs: StaticLocalizer(name, (98, 96, 152, 132)),
    ):
        processed, rows = apply_localizer_postprocess(
            sample,
            pipeline_config=config,
            source_row={"sample_id": "sample_generation", "image_uri": str(original_path), "anomaly_type": "water_leak"},
            processed_root=tmp_path / "processed",
        )

    assert len(rows) == 1
    generation_params = processed["objects"][0]["geometry_detail"]["generation_params"]
    persisted_relative = generation_params["wan"]["wan_response_path"]
    assert not Path(persisted_relative).is_absolute()
    persisted_path = tmp_path / "processed" / persisted_relative
    assert persisted_path.exists()
    assert persisted_path.parent == tmp_path / "processed" / "metadata" / "api_responses"
    assert persisted_path.name == "sample_generation_leak_000001_wan_response.json"
    assert read_json(persisted_path) == {"job_id": "wan-job-1", "status": "succeeded"}


def test_apply_localizer_postprocess_relativizes_metadata_paths(tmp_path) -> None:
    original = make_base_image()
    generated = add_water_patch(original, (98, 96, 152, 132))
    original_path = tmp_path / "original.jpg"
    generated_path = tmp_path / "generated.jpg"
    original.save(original_path)
    generated.save(generated_path)

    wan_response_path = tmp_path / "wan_response.json"
    wan_response_path.write_text('{"job_id":"wan-job-2","status":"succeeded"}', encoding="utf-8")

    sample = _build_sample(generated_path)
    sample["objects"][0]["geometry_detail"]["generation_params"]["wan"] = {
        "bbox_list": [[80, 80, 170, 150]],
        "wan_response_path": str(wan_response_path),
    }
    config = _build_pipeline_config()
    config["modules"]["generation"]["localizer"].update(
        {
            "primary": "pgcd_lpips",
            "debug": True,
        }
    )

    processed_root = tmp_path / "processed"
    with patch(
        "autolabel.modules.generation.metadata_builder.create_localizer",
        side_effect=lambda name, **_kwargs: StaticLocalizerWithArtifacts(name, (98, 96, 152, 132)),
    ):
        processed, rows = apply_localizer_postprocess(
            sample,
            pipeline_config=config,
            source_row={"sample_id": "sample_generation", "image_uri": str(original_path), "anomaly_type": "water_leak", "task_id": "task_001"},
            processed_root=processed_root,
        )

    assert len(rows) == 1
    row = rows[0]
    obj = processed["objects"][0]
    generation_params = obj["geometry_detail"]["generation_params"]
    localizer_info = generation_params["localizer"]

    assert row["task_id"] == "task_001"
    assert row["localizer_used"] == "pgcd_lpips"
    assert not Path(row["mask_uri"]).is_absolute()
    assert (processed_root / row["mask_uri"]).exists()

    assert not Path(obj["geometry_detail"]["mask_uri"]).is_absolute()
    assert (processed_root / obj["geometry_detail"]["mask_uri"]).exists()
    assert not Path(obj["crop"]["crop_uri"]).is_absolute()
    assert (processed_root / obj["crop"]["crop_uri"]).exists()

    assert not Path(generation_params["wan"]["wan_response_path"]).is_absolute()
    assert (processed_root / generation_params["wan"]["wan_response_path"]).exists()

    assert not Path(localizer_info["metrics"]["pgcd_heatmap_path"]).is_absolute()
    assert (processed_root / localizer_info["metrics"]["pgcd_heatmap_path"]).exists()
    assert not Path(localizer_info["debug_artifacts"]["heatmap"]).is_absolute()
    assert (processed_root / localizer_info["debug_artifacts"]["heatmap"]).exists()
    assert not Path(localizer_info["attempts"][0]["metrics"]["pgcd_heatmap_path"]).is_absolute()
    assert (processed_root / localizer_info["attempts"][0]["metrics"]["pgcd_heatmap_path"]).exists()


def test_apply_localizer_postprocess_enables_debug_artifacts_for_benchmark(tmp_path) -> None:
    original = make_base_image()
    generated = add_water_patch(original, (98, 96, 152, 132))
    original_path = tmp_path / "original.jpg"
    generated_path = tmp_path / "generated.jpg"
    original.save(original_path)
    generated.save(generated_path)

    sample = _build_sample(generated_path)
    config = _build_pipeline_config()
    config["modules"]["generation"]["localizer"].update(
        {
            "primary": "pgcd_lpips",
            "debug": False,
            "benchmark": True,
        }
    )

    processed_root = tmp_path / "processed"

    def fake_create_localizer(name: str, **kwargs):
        assert name == "pgcd_lpips"
        assert kwargs["debug"] is True
        return StaticLocalizerWithArtifacts(name, (98, 96, 152, 132))

    with patch(
        "autolabel.modules.generation.metadata_builder.create_localizer",
        side_effect=fake_create_localizer,
    ):
        processed, rows = apply_localizer_postprocess(
            sample,
            pipeline_config=config,
            source_row={"sample_id": "sample_generation", "image_uri": str(original_path), "anomaly_type": "water_leak"},
            processed_root=processed_root,
        )

    assert len(rows) == 1
    localizer_info = processed["objects"][0]["geometry_detail"]["generation_params"]["localizer"]
    assert not Path(localizer_info["metrics"]["pgcd_heatmap_path"]).is_absolute()
    assert (processed_root / localizer_info["metrics"]["pgcd_heatmap_path"]).exists()
    assert not Path(localizer_info["debug_artifacts"]["heatmap"]).is_absolute()
    assert (processed_root / localizer_info["debug_artifacts"]["heatmap"]).exists()
