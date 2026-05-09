from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from autolabel.adapters.classification_script import labels_from_boolean_response
from autolabel.adapters.crop_reviewer import apply_crop_review_result, build_crop_review_config, parse_review_payload
from autolabel.adapters.vlm_labelstudio_detector import (
    labelstudio_value_to_xyxy,
    labelstudio_payload_to_objects,
    parse_detector_payload,
    parse_json_output,
    percent_box_to_xyxy,
)
from autolabel.config_loader import load_config
from autolabel.contract_normalizer import normalize_autolabel_sample
from autolabel.model_config import build_detector_runtime_config, resolve_classification_runtime, resolve_generation_runtime
from autolabel.modules.classification.dry_run import DryRunClassificationModule
from autolabel.preprocess import estimate_extracted_frame_count
from autolabel.exporters.labelstudio import build_labelstudio_config, sample_to_labelstudio_task
from autolabel.sample_factory import make_object
from autolabel.utils import read_json, write_csv, write_json
from autolabel.validators import validate_sample_contract


ROOT = Path(__file__).resolve().parents[1]


def load_builtin_classification_module():
    script_path = ROOT / "scripts" / "classification.py"
    spec = importlib.util.spec_from_file_location("builtin_classification_for_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContractTests(unittest.TestCase):
    def test_schema_file_is_valid_json(self) -> None:
        with open(ROOT / "schemas" / "autolabel_sample.schema.json", encoding="utf-8") as f:
            json.load(f)

    def test_example_sample_passes_lightweight_validator(self) -> None:
        sample = read_json(ROOT / "schemas" / "autolabel_sample.example.json")
        validate_sample_contract(sample)

    def test_boolean_classifier_response_maps_to_multi_labels(self) -> None:
        labels = labels_from_boolean_response(
            {
                "hard_hat": True,
                "reflective_vest": False,
                "smoking": False,
                "notes": {
                    "reflective_vest": "not visible"
                },
            }
        )
        self.assertIn(
            {"label_key": "helmet", "label_value": "wearing_helmet", "confidence": None, "evidence": "vlm classification"},
            labels,
        )
        self.assertIn(
            {
                "label_key": "reflective_vest",
                "label_value": "no_reflective_vest",
                "confidence": None,
                "evidence": "not visible",
            },
            labels,
        )

    def test_builtin_classifier_parses_prose_boolean_output(self) -> None:
        module = load_builtin_classification_module()
        raw = """Based on the visual analysis:
        1. **safety_harness**: No visible straps. (false)
        2. **hard_hat**: The person is wearing a hard hat. -> true
        3. **reflective_vest**: No reflective vest is visible. (false)
        """
        parsed = module.parse_boolean_text_output(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["safety_harness"], False)
        self.assertEqual(parsed["hard_hat"], True)
        self.assertEqual(parsed["reflective_vest"], False)
        self.assertEqual(parsed["_parse_mode"], "text_fallback")

    def test_builtin_classifier_normalizes_missing_labels(self) -> None:
        module = load_builtin_classification_module()
        parsed = module.normalize_boolean_response({"hard_hat": True, "notes": {}})
        self.assertTrue(parsed["hard_hat"])
        self.assertFalse(parsed["safety_shoes"])
        self.assertIn("safety_shoes", parsed["notes"])

    def test_builtin_classifier_skips_invalid_crop_geometry(self) -> None:
        from PIL import Image

        module = load_builtin_classification_module()
        with tempfile.TemporaryDirectory() as tmp:
            crop_path = Path(tmp) / "thin.jpg"
            Image.new("RGB", (351, 1), color=(255, 255, 255)).save(crop_path)
            raw = module.process_image(
                crop_path,
                client=None,
                classifier_config={
                    "min_crop_width": 4,
                    "min_crop_height": 4,
                    "max_crop_aspect_ratio": 20.0,
                },
            )
            self.assertEqual(raw["error"], "invalid_crop_geometry")
            self.assertEqual(raw["image_width"], 351)
            self.assertEqual(raw["image_height"], 1)

    def test_labelstudio_export_shape(self) -> None:
        sample = read_json(ROOT / "schemas" / "autolabel_sample.example.json")
        task = sample_to_labelstudio_task(sample)
        self.assertEqual(task["data"]["sample_id"], sample["sample_id"])
        self.assertEqual(task["predictions"][0]["result"][0]["type"], "rectanglelabels")
        self.assertEqual(task["predictions"][0]["result"][0]["value"]["rectanglelabels"], ["Person"])
        self.assertIn("classification", task["predictions"][0]["result"][0]["meta"])
        choices = [item for item in task["predictions"][0]["result"] if item["type"] == "choices"]
        self.assertTrue(choices)
        self.assertEqual(choices[0]["id"], task["predictions"][0]["result"][0]["id"])
        self.assertTrue(choices[0]["from_name"].startswith("cls_"))

    def test_labelstudio_config_has_per_region_choices(self) -> None:
        sample = read_json(ROOT / "schemas" / "autolabel_sample.example.json")
        config_xml = build_labelstudio_config([sample])
        self.assertIn('<RectangleLabels name="label" toName="image">', config_xml)
        self.assertIn('<Choices name="cls_helmet" toName="image" perRegion="true"', config_xml)
        self.assertIn('<Choice value="wearing_helmet"/>', config_xml)

    def test_yaml_model_selection_is_resolved_from_config(self) -> None:
        config = load_config(ROOT / "configs" / "autolabel.yaml")
        generation = resolve_generation_runtime(config)
        classification = resolve_classification_runtime(config)
        detector = build_detector_runtime_config(config)
        preprocess = config["preprocess"]
        direct = config["direct_annotation"]
        self.assertEqual(generation["vlm_model_name"], "aios-smart-eye-vlm")
        self.assertIn("model", classification)
        self.assertIn("model_profiles", detector)
        self.assertEqual(detector["services"]["ppe_person"]["model_ref"], "ppe_person_vlm_labelstudio_detector")
        self.assertEqual(detector["model_profiles"]["ppe_person_vlm_labelstudio_detector"]["parse_retry_count"], 0)
        self.assertTrue(detector["model_profiles"]["ppe_person_vlm_labelstudio_detector"]["fail_on_parse_error"])
        self.assertTrue(detector["model_profiles"]["ppe_person_vlm_labelstudio_detector"]["use_response_format"])
        self.assertEqual(detector["model_profiles"]["ppe_person_vlm_labelstudio_detector"]["request_image_max_side"], 1280)
        self.assertEqual(detector["model_profiles"]["ppe_person_vlm_labelstudio_detector"]["coordinate_units"], "auto")
        self.assertTrue(detector["model_profiles"]["ppe_person_vlm_labelstudio_detector"]["auto_detect_coordinate_units"])
        self.assertEqual(
            detector["model_profiles"]["ppe_person_vlm_labelstudio_detector"]["response_format_type"],
            "json_object",
        )
        self.assertEqual(classification["max_tokens"], 2000)
        self.assertEqual(classification["request_image_max_side"], 1024)
        self.assertEqual(classification["min_crop_width"], 4)
        self.assertEqual(classification["min_crop_height"], 4)
        self.assertEqual(classification["max_crop_aspect_ratio"], 20.0)
        self.assertEqual(classification["parse_retry_count"], 0)
        self.assertTrue(classification["use_response_format"])
        self.assertFalse(classification["text_fallback_enabled"])
        self.assertEqual(preprocess["video_decode_mode"], "cpu")
        self.assertEqual(preprocess["video_error_policy"], "skip")
        self.assertEqual(direct["batch_size"], 1)
        self.assertEqual(direct["workers"], 1)
        self.assertEqual(direct["json_retry_attempts"], 3)
        self.assertEqual(direct["min_box_width"], 4)
        self.assertEqual(direct["min_box_height"], 4)
        self.assertEqual(direct["min_box_area"], 16)
        self.assertEqual(direct["max_box_aspect_ratio"], 20.0)
        self.assertTrue(direct["cleanup_existing_crops"])
        self.assertFalse(direct["crop_review"]["enabled"])
        self.assertEqual(direct["crop_review"]["model_ref"], "ppe_person_vlm_labelstudio_detector")
        self.assertEqual(build_crop_review_config(config, detector), {"enabled": False})

    def test_crop_review_config_uses_detector_model_profile(self) -> None:
        config = deepcopy(load_config(ROOT / "configs" / "autolabel.yaml"))
        config["direct_annotation"]["crop_review"]["enabled"] = True
        detector = build_detector_runtime_config(config)
        detector["services"]["ppe_person"]["dry_run"] = True

        review = build_crop_review_config(config, detector)
        self.assertTrue(review["enabled"])
        self.assertTrue(review["dry_run"])
        self.assertEqual(review["model_name"], "aios-smart-eye-vlm")
        self.assertEqual(review["model_ref"], "ppe_person_vlm_labelstudio_detector")

    def test_crop_review_failure_updates_object_quality_check(self) -> None:
        obj = {"object_id": "person_000001", "object_type": "person", "quality_check": None}
        apply_crop_review_result(
            obj,
            {
                "contains_person": True,
                "is_complete_visible_person": False,
                "missing_parts": ["feet"],
                "reason": "脚部被 crop 截断",
            },
            {
                "failed_issue_flag": "incomplete_person_crop",
                "reviewer": "vlm_crop_reviewer",
                "prompt_version": "crop_full_person_review_v1",
            },
        )

        quality_check = obj["quality_check"]
        self.assertEqual(quality_check["qc_status"], "failed")
        self.assertIn("incomplete_person_crop", quality_check["issue_flags"])
        self.assertIn("missing_feet", quality_check["issue_flags"])
        self.assertEqual(quality_check["reviewer"], "vlm_crop_reviewer")
        self.assertIn("脚部被 crop 截断", quality_check["comment"])

    def test_crop_review_parser_reads_json_object_from_response(self) -> None:
        parsed = parse_review_payload(
            '结果：{"contains_person": true, "is_complete_visible_person": true, "missing_parts": [], "reason": ""}'
        )
        self.assertTrue(parsed["contains_person"])
        self.assertTrue(parsed["is_complete_visible_person"])

    def test_video_frame_stride_count_estimate(self) -> None:
        self.assertEqual(estimate_extracted_frame_count(3000, 30), 100)
        self.assertEqual(estimate_extracted_frame_count(2500, 30), 84)
        self.assertEqual(estimate_extracted_frame_count(3000, 30, max_frames=10), 10)

    def test_labelstudio_percent_box_converts_to_xyxy_pixels(self) -> None:
        box = percent_box_to_xyxy({"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0}, 1000, 500)
        self.assertEqual(box, {"format": "xyxy", "x1": 100, "y1": 100, "x2": 400, "y2": 300})

    def test_vlm_box_converter_auto_detects_pixel_xywh(self) -> None:
        box, coordinate_unit = labelstudio_value_to_xyxy(
            {"x": 500.0, "y": 100.0, "width": 50.0, "height": 200.0},
            1000,
            500,
        )
        self.assertEqual(box, {"format": "xyxy", "x1": 500, "y1": 100, "x2": 550, "y2": 300})
        self.assertEqual(coordinate_unit, "pixel_xywh_auto_detected")

    def test_vlm_box_converter_maps_resized_pixel_xywh_back_to_original(self) -> None:
        box, coordinate_unit = labelstudio_value_to_xyxy(
            {"x": 640.0, "y": 100.0, "width": 100.0, "height": 300.0},
            3840,
            2160,
            request_width=1280,
            request_height=720,
        )
        self.assertEqual(box, {"format": "xyxy", "x1": 1920, "y1": 300, "x2": 2220, "y2": 1200})
        self.assertEqual(coordinate_unit, "pixel_xywh_auto_detected")

    def test_vlm_box_converter_keeps_valid_percent_xywh(self) -> None:
        box, coordinate_unit = labelstudio_value_to_xyxy(
            {"x": 80.0, "y": 20.0, "width": 15.0, "height": 40.0},
            1000,
            500,
        )
        self.assertEqual(box, {"format": "xyxy", "x1": 800, "y1": 100, "x2": 950, "y2": 300})
        self.assertEqual(coordinate_unit, "labelstudio_percent_xywh")

    def test_vlm_json_parser_ignores_explanatory_suffix(self) -> None:
        payload = parse_json_output('结果如下：[{"predictions": [{"result": []}]}]\n说明文字')
        self.assertEqual(payload, [{"predictions": [{"result": []}]}])

    def test_vlm_detector_parse_failure_can_return_none(self) -> None:
        self.assertIsNone(parse_detector_payload("I found one person but cannot provide JSON.", log_error=False))

    def test_vlm_detector_requests_json_object_response_format(self) -> None:
        from autolabel.adapters.vlm_labelstudio_detector import VLMLabelStudioDetector

        calls = []

        class Message:
            content = '{"predictions": [{"result": []}]}'

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return Response()

        class Chat:
            completions = Completions()

        class Client:
            chat = Chat()

        detector = VLMLabelStudioDetector(
            {
                "use_response_format": True,
                "response_format_type": "json_object",
            }
        )
        text = detector._request_json_text(Client(), "aios-smart-eye-vlm", "data:image/jpeg;base64,abc", "prompt")
        self.assertEqual(text, '{"predictions": [{"result": []}]}')
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})

    def test_vlm_labelstudio_payload_maps_to_autolabel_objects(self) -> None:
        payload = [
            {
                "data": {"image": "image.jpg"},
                "predictions": [
                    {
                        "model_version": "vlm-pre-annotation-v1",
                        "score": 0.95,
                        "result": [
                            {
                                "id": "person_001",
                                "type": "rectanglelabels",
                                "value": {
                                    "rotation": 0,
                                    "x": 10.0,
                                    "y": 20.0,
                                    "width": 30.0,
                                    "height": 40.0,
                                    "rectanglelabels": ["Person"],
                                },
                            }
                        ],
                    }
                ],
            }
        ]
        service = {
            "geometry_source": "detector",
            "model_name": "aios-smart-eye-vlm",
            "model_version": "vlm-pre-annotation-v1",
            "object_type_map": {"Person": "person"},
            "default_object_type": "person",
        }
        objects = labelstudio_payload_to_objects(payload, "image.jpg", 1000, 500, service)
        self.assertEqual(objects[0]["object_id"], "person_001")
        self.assertEqual(objects[0]["object_type"], "person")
        self.assertEqual(objects[0]["box"], {"format": "xyxy", "x1": 100, "y1": 100, "x2": 400, "y2": 300})

    def test_vlm_detection_objects_align_with_autolabel_sample_contract(self) -> None:
        payload = [
            {
                "data": {"image": "image.jpg"},
                "predictions": [
                    {
                        "model_version": "vlm-pre-annotation-v1",
                        "score": 0.95,
                        "result": [
                            {
                                "id": "自动生成的唯一ID",
                                "type": "rectanglelabels",
                                "from_name": "label",
                                "to_name": "image",
                                "image_rotation": 0,
                                "value": {
                                    "rotation": 0,
                                    "x": 10.0,
                                    "y": 20.0,
                                    "width": 30.0,
                                    "height": 40.0,
                                    "rectanglelabels": ["Person"],
                                },
                            }
                        ],
                    }
                ],
            }
        ]
        service = {
            "geometry_source": "detector",
            "model_name": "aios-smart-eye-vlm",
            "model_version": "vlm-pre-annotation-v1",
            "prompt_version": "person_labelstudio_bbox_v1",
            "object_type_map": {"Person": "person"},
            "default_object_type": "person",
            "object_id_prefix": "person",
        }
        objects = labelstudio_payload_to_objects(payload, "image.jpg", 1000, 500, service)
        sample = normalize_autolabel_sample(
            {
                "sample_id": "sample_person_vlm_001",
                "image_asset": {
                    "image_id": "image_001",
                    "image_uri": "image.jpg",
                    "width": 1000,
                    "height": 500,
                    "source_type": "manual_upload",
                },
                "objects": objects,
                "workflow": {"workflow_status": "boxed"},
                "export": {"export_format": "labelstudio", "export_status": "not_exported"},
            }
        )
        validate_sample_contract(sample)
        obj = sample["objects"][0]
        self.assertEqual(obj["object_id"], "person_000001")
        self.assertEqual(obj["object_type"], "person")
        self.assertEqual(obj["geometry_source"], "detector")
        self.assertEqual(obj["geometry_model"]["model_name"], "aios-smart-eye-vlm")
        self.assertEqual(obj["geometry_model"]["model_version"], "vlm-pre-annotation-v1")
        self.assertEqual(obj["geometry_model"]["confidence"], 0.95)
        self.assertEqual(obj["geometry_detail"]["polygon"], None)
        self.assertEqual(obj["geometry_detail"]["mask_uri"], None)
        self.assertEqual(obj["geometry_detail"]["mask_format"], None)
        self.assertEqual(
            obj["geometry_detail"]["generation_params"]["output_contract"],
            "AutoLabelSample.objects[]",
        )
        self.assertEqual(obj["crop"]["crop_id"], "person_000001_crop")
        self.assertEqual(obj["classification"]["multi_labels"], [])
        self.assertEqual(obj["quality_check"], None)

    def test_dry_run_classifier_populates_classification_contract(self) -> None:
        sample = normalize_autolabel_sample(
            {
                "sample_id": "sample_dry_classification",
                "image_asset": {
                    "image_id": "image_001",
                    "image_uri": "image.jpg",
                    "width": 100,
                    "height": 100,
                    "source_type": "manual_upload",
                },
                "objects": [
                    {
                        "object_id": "person_000001",
                        "object_type": "person",
                        "box": {"format": "xyxy", "x1": 10, "y1": 10, "x2": 90, "y2": 90},
                        "geometry_source": "detector",
                        "crop": {
                            "crop_id": "sample_dry_classification_person_000001",
                            "crop_uri": "crop.jpg",
                        },
                    }
                ],
                "workflow": {"workflow_status": "cropped"},
                "export": {"export_format": "labelstudio", "export_status": "not_exported"},
            }
        )
        module = DryRunClassificationModule({})
        sample = module.classify_sample(sample)
        validate_sample_contract(sample)
        self.assertEqual(sample["objects"][0]["classification"]["classifier_name"], "dry_run_rule_classifier")
        self.assertTrue(sample["objects"][0]["classification"]["multi_labels"])

    def test_direct_pipeline_filters_bad_boxes_and_cleans_stale_crops(self) -> None:
        from PIL import Image

        from autolabel.pipeline import run_direct_pipeline

        class TinyAndValidDetector:
            def __init__(self, _config: dict) -> None:
                pass

            def detect(self, _image_uri: str, _task_key: str | None = None) -> list[dict]:
                return [
                    make_object(
                        object_id="person_tiny",
                        object_type="person",
                        box={"format": "xyxy", "x1": 1, "y1": 1, "x2": 2, "y2": 2},
                        geometry_source="detector",
                        geometry_model={"model_name": "test", "model_version": "1", "confidence": 0.9},
                    ),
                    make_object(
                        object_id="person_valid",
                        object_type="person",
                        box={"format": "xyxy", "x1": 10, "y1": 10, "x2": 40, "y2": 70},
                        geometry_source="detector",
                        geometry_model={"model_name": "test", "model_version": "1", "confidence": 0.9},
                    ),
                    make_object(
                        object_id="person_thin",
                        object_type="person",
                        box={"format": "xyxy", "x1": 50, "y1": 1, "x2": 54, "y2": 79},
                        geometry_source="detector",
                        geometry_model={"model_name": "test", "model_version": "1", "confidence": 0.9},
                    ),
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            Image.new("RGB", (100, 80), color=(255, 255, 255)).save(image_path)
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_tiny_crop",
                        "image_id": "image_tiny_crop",
                        "image_uri": str(image_path),
                        "source_type": "manual_upload",
                        "task_mode": "direct",
                        "width": 100,
                        "height": 80,
                    }
                ],
                [
                    "sample_id",
                    "image_id",
                    "image_uri",
                    "source_type",
                    "task_mode",
                    "width",
                    "height",
                ],
            )

            pipeline_config = {
                "pipeline_id": "test_pipeline",
                "pipeline_version": "test",
                "direct_annotation": {
                    "default_task_key": "ppe_person",
                    "min_box_confidence": 0.0,
                    "min_box_width": 4,
                    "min_box_height": 4,
                    "min_box_area": 16,
                    "max_box_aspect_ratio": 10.0,
                    "cleanup_existing_crops": True,
                    "json_retry_attempts": 3,
                    "batch_size": 1,
                    "workers": 1,
                },
                "classification": {"enabled": False},
                "export": {"export_format": "labelstudio", "export_status": "not_exported"},
            }
            detector_config = {"default_task_key": "ppe_person", "services": {"ppe_person": {}}}
            output_root = root / "processed"
            stale_tiny_crop = output_root / "crops" / "sample_tiny_crop_person_tiny.jpg"
            stale_tiny_crop.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1, 1), color=(0, 0, 0)).save(stale_tiny_crop)

            with patch("autolabel.pipeline.DetectorServiceClient", TinyAndValidDetector):
                written = run_direct_pipeline(
                    manifest_path,
                    pipeline_config,
                    detector_config,
                    output_root,
                    classify=False,
                )

            self.assertEqual(len(written), 1)
            sample = read_json(written[0])
            self.assertEqual([obj["object_id"] for obj in sample["objects"]], ["person_valid"])
            self.assertFalse(stale_tiny_crop.exists())
            valid_crop = output_root / "crops" / "sample_tiny_crop_person_valid.jpg"
            self.assertTrue(valid_crop.exists())
            with Image.open(valid_crop) as crop:
                self.assertEqual(crop.size, (30, 60))

    def test_direct_pipeline_does_not_write_metadata_after_json_parse_failures(self) -> None:
        from PIL import Image

        from autolabel.adapters.vlm_labelstudio_detector import VLMJsonParseError
        from autolabel.pipeline import run_direct_pipeline

        class FailingDetector:
            def __init__(self, _config: dict) -> None:
                pass

            def detect(self, _image_uri: str, _task_key: str | None = None) -> list[dict]:
                raise VLMJsonParseError("bad detector json")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            Image.new("RGB", (100, 80), color=(255, 255, 255)).save(image_path)
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_bad_json",
                        "image_id": "image_bad_json",
                        "image_uri": str(image_path),
                        "source_type": "manual_upload",
                        "task_mode": "direct",
                        "width": 100,
                        "height": 80,
                    }
                ],
                [
                    "sample_id",
                    "image_id",
                    "image_uri",
                    "source_type",
                    "task_mode",
                    "width",
                    "height",
                ],
            )

            pipeline_config = {
                "pipeline_id": "test_pipeline",
                "pipeline_version": "test",
                "direct_annotation": {
                    "default_task_key": "ppe_person",
                    "json_retry_attempts": 3,
                    "batch_size": 1,
                    "workers": 1,
                },
                "classification": {"enabled": False},
                "export": {"export_format": "labelstudio", "export_status": "not_exported"},
            }
            detector_config = {"default_task_key": "ppe_person", "services": {"ppe_person": {}}}
            output_root = root / "processed"
            write_json(output_root / "metadata" / "sample_bad_json.json", {"stale": True})
            write_json(output_root / "retry_failures" / "sample_bad_json.json", {"stale": True})

            with patch("autolabel.pipeline.DetectorServiceClient", FailingDetector):
                written = run_direct_pipeline(
                    manifest_path,
                    pipeline_config,
                    detector_config,
                    output_root,
                    classify=False,
                )

            self.assertEqual(written, [])
            self.assertFalse((output_root / "metadata" / "sample_bad_json.json").exists())
            self.assertFalse((output_root / "retry_failures" / "sample_bad_json.json").exists())


if __name__ == "__main__":
    unittest.main()
