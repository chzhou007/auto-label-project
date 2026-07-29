from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from autolabel.adapters.classification_script import labels_from_boolean_response
from autolabel.adapters.crop_reviewer import (
    VLMCropReviewer,
    apply_crop_review_result,
    build_crop_review_config,
    parse_review_payload,
)
from autolabel.adapters.vlm_labelstudio_detector import (
    labelstudio_value_to_xyxy,
    labelstudio_payload_to_objects,
    parse_detector_payload,
    parse_json_output,
    percent_box_to_xyxy,
)
from autolabel.asset_qc import infer_crop_source, run_asset_qc
from autolabel.config_loader import load_config
from autolabel.contract_normalizer import normalize_autolabel_sample
from autolabel.model_config import build_detector_runtime_config, resolve_classification_runtime, resolve_generation_runtime
from autolabel.modules.classification.dry_run import DryRunClassificationModule
from autolabel.preprocess import estimate_extracted_frame_count
from autolabel.exporters.labelstudio import build_labelstudio_config, sample_to_labelstudio_task
from autolabel.qc_agent import parse_vlm_qc_payload, resolve_local_uri, run_qc_agent
from autolabel.sample_factory import make_object
from autolabel.utils import read_csv, read_json, write_csv, write_json
from autolabel.validators import ValidationError, validate_sample_contract


ROOT = Path(__file__).resolve().parents[1]


def load_builtin_classification_module():
    script_path = ROOT / "scripts" / "classification.py"
    spec = importlib.util.spec_from_file_location("builtin_classification_for_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_script_module(script_name: str):
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", "_for_test"), script_path)
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

    def test_builtin_classifier_api_error_raises_for_pipeline_retry(self) -> None:
        from PIL import Image

        module = load_builtin_classification_module()

        class Completions:
            def create(self, **_kwargs):
                raise RuntimeError("connection reset")

        class Chat:
            completions = Completions()

        class Client:
            chat = Chat()

        with tempfile.TemporaryDirectory() as tmp:
            crop_path = Path(tmp) / "person.jpg"
            Image.new("RGB", (40, 120), color=(255, 255, 255)).save(crop_path)
            with self.assertRaises(module.ClassificationJsonParseError):
                module.process_image(
                    crop_path,
                    Client(),
                    classifier_config={
                        "min_crop_width": 4,
                        "min_crop_height": 4,
                        "max_crop_aspect_ratio": 20.0,
                    },
                )

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
        self.assertFalse(detector["model_profiles"]["ppe_person_vlm_labelstudio_detector"]["use_response_format"])
        self.assertEqual(
            detector["model_profiles"]["ppe_person_vlm_labelstudio_detector"]["model_version"],
            "vlm-pre-annotation-v2",
        )
        self.assertEqual(
            detector["model_profiles"]["ppe_person_vlm_labelstudio_detector"]["prompt_version"],
            "person_labelstudio_full_body_bbox_v3",
        )
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
        self.assertEqual(direct["min_box_width"], 12)
        self.assertEqual(direct["min_box_height"], 24)
        self.assertEqual(direct["min_box_area"], 300)
        self.assertEqual(direct["max_box_aspect_ratio"], 8.0)
        self.assertTrue(direct["cleanup_existing_crops"])
        self.assertTrue(direct["crop_review"]["enabled"])
        self.assertEqual(direct["crop_review"]["model_ref"], "ppe_person_vlm_labelstudio_detector")
        review_config = build_crop_review_config(config, detector)
        self.assertTrue(review_config["enabled"])
        self.assertTrue(review_config["drop_failed"])
        self.assertFalse(review_config["drop_incomplete_person"])

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

    def test_crop_review_can_drop_no_person_objects(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            crop_path = Path(tmp) / "sample_person_false.jpg"
            Image.new("RGB", (40, 120), color=(0, 0, 0)).save(crop_path)
            sample = {
                "sample_id": "sample_crop_review_drop",
                "objects": [
                    {
                        "object_id": "person_false",
                        "object_type": "person",
                        "crop": {"crop_uri": str(crop_path)},
                    }
                ],
            }
            reviewer = VLMCropReviewer({"drop_failed": True})
            with patch.object(
                reviewer,
                "review_crop",
                return_value={
                    "contains_person": False,
                    "is_complete_visible_person": False,
                    "missing_parts": [],
                    "reason": "只有柜门",
                },
            ):
                reviewer.review_sample(sample)
            self.assertEqual(sample["objects"], [])
            self.assertFalse(crop_path.exists())

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

    def test_generation_branch_skips_direct_only_manifest(self) -> None:
        from autolabel.orchestrator import run_generation_branch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            image_path.write_bytes(b"placeholder")
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_direct_only",
                        "image_id": "image_direct_only",
                        "image_uri": str(image_path),
                        "source_type": "manual_upload",
                        "task_mode": "direct",
                    }
                ],
                ["sample_id", "image_id", "image_uri", "source_type", "task_mode"],
            )
            config = {
                "paths": {
                    "metadata_dir": str(root / "metadata"),
                    "image_sequence_dir": str(root),
                },
                "modules": {"generation": {"backend": "i2i_external", "backends": {"i2i_external": {}}}},
            }
            code = run_generation_branch(config, tasks_csv=manifest_path, output_root=root / "i2i")
            self.assertEqual(code, 0)

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

    def test_image_to_data_url_with_size_uses_one_resized_jpeg_payload(self) -> None:
        from PIL import Image
        from autolabel.utils import image_to_data_url_with_size

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "wide.png"
            Image.new("RGB", (200, 100), color=(255, 0, 0)).save(image_path)
            data_url, width, height = image_to_data_url_with_size(image_path, max_side=100)
            self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
            self.assertEqual((width, height), (100, 50))

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
        objects[0]["crop"] = {
            "crop_id": "person_000001_crop",
            "crop_uri": "crop.jpg",
            "crop_box": None,
            "crop_expand_ratio": None,
            "is_valid_crop": False,
        }
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

    def test_validator_rejects_pending_crop_uri(self) -> None:
        sample = normalize_autolabel_sample(
            {
                "sample_id": "sample_pending_crop",
                "image_asset": {
                    "image_id": "image_001",
                    "image_uri": "image.jpg",
                    "width": 100,
                    "height": 100,
                    "source_type": "manual_upload",
                },
                "objects": [
                    make_object(
                        object_id="person_000001",
                        object_type="person",
                        box={"format": "xyxy", "x1": 10, "y1": 10, "x2": 90, "y2": 90},
                        geometry_source="detector",
                    )
                ],
                "workflow": {"workflow_status": "boxed"},
                "export": {"export_format": "labelstudio", "export_status": "not_exported"},
            }
        )
        with self.assertRaises(ValidationError):
            validate_sample_contract(sample)

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

    def test_filter_detected_objects_drops_degenerate_boxes_when_thresholds_are_zero(self) -> None:
        from autolabel.pipeline import filter_detected_objects

        objects = [
            make_object(
                object_id="person_zero_width",
                object_type="person",
                box={"format": "xyxy", "x1": 10, "y1": 10, "x2": 10, "y2": 50},
                geometry_source="detector",
            ),
            make_object(
                object_id="person_valid",
                object_type="person",
                box={"format": "xyxy", "x1": 20, "y1": 10, "x2": 40, "y2": 70},
                geometry_source="detector",
            ),
        ]
        kept = filter_detected_objects(
            objects,
            {
                "min_box_confidence": 0,
                "min_box_width": 0,
                "min_box_height": 0,
                "min_box_area": 0,
                "max_box_aspect_ratio": 20,
            },
            "sample_degenerate",
        )
        self.assertEqual([obj["object_id"] for obj in kept], ["person_valid"])

    def test_direct_pipeline_uses_configured_metadata_and_crop_dirs(self) -> None:
        from PIL import Image

        from autolabel.pipeline import run_direct_pipeline

        class OneBoxDetector:
            def __init__(self, _config: dict) -> None:
                pass

            def detect(self, _image_uri: str, _task_key: str | None = None) -> list[dict]:
                return [
                    make_object(
                        object_id="person_valid",
                        object_type="person",
                        box={"format": "xyxy", "x1": 10, "y1": 10, "x2": 40, "y2": 70},
                        geometry_source="detector",
                    )
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
                        "sample_id": "sample_custom_paths",
                        "image_id": "image_custom_paths",
                        "image_uri": str(image_path),
                        "source_type": "manual_upload",
                        "task_mode": "direct",
                        "width": 100,
                        "height": 80,
                    }
                ],
                ["sample_id", "image_id", "image_uri", "source_type", "task_mode", "width", "height"],
            )
            metadata_dir = root / "custom_meta"
            crop_dir = root / "custom_crops"
            pipeline_config = {
                "pipeline_id": "test_pipeline",
                "pipeline_version": "test",
                "paths": {
                    "metadata_dir": str(metadata_dir),
                    "crop_dir": str(crop_dir),
                },
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

            with patch("autolabel.pipeline.DetectorServiceClient", OneBoxDetector):
                written = run_direct_pipeline(
                    manifest_path,
                    pipeline_config,
                    detector_config,
                    classify=False,
                )

            self.assertEqual(written, [metadata_dir / "sample_custom_paths.json"])
            self.assertTrue((crop_dir / "sample_custom_paths_person_valid.jpg").exists())

    def test_direct_pipeline_falls_back_to_image_size_when_manifest_dimensions_are_zero(self) -> None:
        from PIL import Image

        from autolabel.pipeline import run_direct_pipeline

        class NoBoxDetector:
            def __init__(self, _config: dict) -> None:
                pass

            def detect(self, _image_uri: str, _task_key: str | None = None) -> list[dict]:
                return []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            Image.new("RGB", (123, 45), color=(255, 255, 255)).save(image_path)
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_zero_dims",
                        "image_id": "image_zero_dims",
                        "image_uri": str(image_path),
                        "source_type": "manual_upload",
                        "task_mode": "direct",
                        "width": "0",
                        "height": "0",
                    }
                ],
                ["sample_id", "image_id", "image_uri", "source_type", "task_mode", "width", "height"],
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

            with patch("autolabel.pipeline.DetectorServiceClient", NoBoxDetector):
                written = run_direct_pipeline(
                    manifest_path,
                    pipeline_config,
                    detector_config,
                    root / "processed",
                    classify=False,
                )

            sample = read_json(written[0])
            self.assertEqual(sample["image_asset"]["width"], 123)
            self.assertEqual(sample["image_asset"]["height"], 45)

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

    def test_qc_agent_passes_valid_sample_and_writes_reports(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "images" / "image.jpg"
            crop_path = root / "crops" / "person.jpg"
            image_path.parent.mkdir(parents=True)
            crop_path.parent.mkdir(parents=True)
            Image.new("RGB", (100, 80), color=(255, 255, 255)).save(image_path)
            Image.new("RGB", (30, 60), color=(255, 255, 255)).save(crop_path)
            sample = normalize_autolabel_sample(
                {
                    "sample_id": "sample_qc_valid",
                    "image_asset": {
                        "image_id": "image_qc_valid",
                        "image_uri": "images/image.jpg",
                        "width": 100,
                        "height": 80,
                        "source_type": "manual_upload",
                    },
                    "objects": [
                        make_object(
                            object_id="person_000001",
                            object_type="person",
                            box={"format": "xyxy", "x1": 10, "y1": 10, "x2": 40, "y2": 70},
                            geometry_source="detector",
                            crop={
                                "crop_id": "person_000001_crop",
                                "crop_uri": "crops/person.jpg",
                                "crop_box": {"format": "xyxy", "x1": 10, "y1": 10, "x2": 40, "y2": 70},
                                "crop_expand_ratio": 0.0,
                                "is_valid_crop": True,
                            },
                        )
                    ],
                    "workflow": {"workflow_status": "classified"},
                    "export": {"export_format": "labelstudio", "export_status": "not_exported"},
                }
            )
            metadata_path = root / "metadata" / "sample_qc_valid.json"
            write_json(metadata_path, sample)

            report = run_qc_agent(
                pipeline_config={"qc_agent": {"output_dir": str(root / "qc"), "manual_sampling_ratio": 0}},
                sample_paths=[metadata_path],
                asset_base_dirs=[root],
            )

            self.assertEqual(report["summary"]["passed_samples"], 1)
            self.assertEqual(report["summary"]["manual_review_queue_size"], 0)
            sample_report = report["samples"][0]
            self.assertEqual(sample_report["field_snapshot"]["image_asset"]["image_id"], "image_qc_valid")
            object_report = sample_report["objects"][0]
            self.assertEqual(object_report["field_snapshot"]["object_type"], "person")
            self.assertEqual(object_report["field_snapshot"]["crop"]["crop_uri"], "crops/person.jpg")
            self.assertTrue(Path(report["report_path"]).exists())
            self.assertTrue(Path(report["manual_review_queue_path"]).exists())

    def test_qc_agent_resolves_windows_uri_from_posix_asset_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "metadata" / "sample.json"
            image_path = root / "i2i_outputs" / "generated_images" / "sample.png"
            image_path.parent.mkdir(parents=True)
            image_path.touch()

            resolved, exists, _ = resolve_local_uri(
                r"C:\Users\annotator\batch\i2i_outputs\generated_images\sample.png",
                metadata_path,
                [root / "i2i_outputs"],
            )

            self.assertTrue(exists)
            self.assertEqual(resolved, image_path)

    def test_qc_agent_flags_tiny_box_and_missing_crop(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "images" / "image.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (100, 80), color=(255, 255, 255)).save(image_path)
            sample = normalize_autolabel_sample(
                {
                    "sample_id": "sample_qc_bad_crop",
                    "image_asset": {
                        "image_id": "image_qc_bad_crop",
                        "image_uri": "images/image.jpg",
                        "width": 100,
                        "height": 80,
                        "source_type": "manual_upload",
                    },
                    "objects": [
                        make_object(
                            object_id="person_000001",
                            object_type="person",
                            box={"format": "xyxy", "x1": 10, "y1": 10, "x2": 11, "y2": 11},
                            geometry_source="detector",
                            crop={
                                "crop_id": "person_000001_crop",
                                "crop_uri": "crops/missing.jpg",
                                "crop_box": {"format": "xyxy", "x1": 10, "y1": 10, "x2": 11, "y2": 11},
                                "crop_expand_ratio": 0.0,
                                "is_valid_crop": True,
                            },
                        )
                    ],
                    "workflow": {"workflow_status": "classified"},
                    "export": {"export_format": "labelstudio", "export_status": "not_exported"},
                }
            )
            metadata_path = root / "metadata" / "sample_qc_bad_crop.json"
            write_json(metadata_path, sample)

            report = run_qc_agent(
                pipeline_config={"qc_agent": {"output_dir": str(root / "qc"), "manual_sampling_ratio": 0}},
                sample_paths=[metadata_path],
                asset_base_dirs=[root],
            )

            self.assertEqual(report["summary"]["failed_samples"], 1)
            issue_codes = {
                issue["code"]
                for obj in report["samples"][0]["objects"]
                for issue in obj["issues"]
            }
            self.assertIn("tiny_box", issue_codes)
            self.assertIn("crop_file_missing", issue_codes)
            field_paths = {
                issue["field_path"]
                for obj in report["samples"][0]["objects"]
                for issue in obj["issues"]
            }
            self.assertIn("objects[].box", field_paths)
            self.assertIn("objects[].crop.crop_uri", field_paths)
            queue_rows = read_csv(report["manual_review_queue_path"])
            self.assertEqual(queue_rows[0]["object_type"], "person")
            self.assertEqual(queue_rows[0]["source_type"], "manual_upload")
            self.assertIn("objects[].box", queue_rows[0]["issue_field_paths"])

    def test_qc_vlm_payload_parser_normalizes_response(self) -> None:
        parsed = parse_vlm_qc_payload(
            '结果：{"visible_target": true, "box_quality": "under_inclusive", '
            '"label_match": true, "needs_human_review": true, '
            '"issue_flags": ["missing_feet"], "reason": "脚部被红框截断"}'
        )
        self.assertTrue(parsed["visible_target"])
        self.assertEqual(parsed["box_quality"], "under_inclusive")
        self.assertTrue(parsed["needs_human_review"])
        self.assertEqual(parsed["issue_flags"], ["missing_feet"])

    def test_anomaly_bbox_qc_crop_evidence_fields(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        parsed = module.add_crop_evidence_fields(
            {
                "visible_target": False,
                "box_quality": "good",
                "label_match": True,
                "needs_human_review": False,
                "issue_flags": [],
                "raw": {
                    "inside_has_target_anomaly": "true",
                    "outside_has_same_anomaly": "true",
                    "outside_extension": "true",
                    "containment_quality": "under_inclusive",
                    "target_location": "floor",
                    "outside_evidence_summary": "same wet patch continues outside the red box",
                },
            }
        )
        self.assertTrue(parsed["inside_has_target_anomaly"])
        self.assertTrue(parsed["visible_target"])
        self.assertTrue(parsed["outside_extension"])
        self.assertEqual(parsed["box_quality"], "under_inclusive")
        self.assertEqual(parsed["containment_quality"], "under_inclusive")

    def test_anomaly_bbox_qc_training_triage_fields(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        parsed = module.add_crop_evidence_fields(
            {
                "visible_target": True,
                "box_quality": "good",
                "label_match": True,
                "needs_human_review": False,
                "issue_flags": [],
                "raw": {
                    "image_has_positive_candidate": "true",
                    "visual_positive_class": "true",
                    "bbox_usable_for_training": True,
                    "weak_label_consistent": True,
                    "label_conflict": False,
                    "training_action": "auto_accept_positive",
                    "training_priority": "high",
                    "rework_type": "none",
                    "visual_class": "floor_clear_water",
                    "positive_candidate_location": "inside_bbox",
                    "positive_candidate_relation_to_bbox": "contained",
                    "hard_negative_type": "none",
                },
            }
        )
        self.assertTrue(parsed["image_has_positive_candidate"])
        self.assertTrue(parsed["visual_positive_class"])
        self.assertTrue(parsed["bbox_usable_for_training"])
        self.assertFalse(parsed["label_conflict"])
        self.assertEqual(parsed["training_action"], "auto_accept_positive")
        self.assertEqual(parsed["training_priority"], "high")
        self.assertEqual(parsed["rework_type"], "none")
        self.assertEqual(parsed["positive_candidate_location"], "inside_bbox")

    def test_anomaly_bbox_qc_training_guard_blocks_negative_auto_accept(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        parsed = module.apply_training_triage_guard(
            {
                "visual_positive_class": True,
                "training_action": "auto_accept_positive",
                "label_conflict": False,
                "needs_human_review": False,
                "training_priority": "high",
                "issue_flags": [],
            },
            "negative_other_or_label_conflict_candidate",
        )
        self.assertEqual(parsed["training_action"], "label_conflict_review")
        self.assertTrue(parsed["label_conflict"])
        self.assertTrue(parsed["needs_human_review"])
        self.assertIn("weak_label_conflict", parsed["issue_flags"])

    def test_anomaly_bbox_qc_training_guard_blocks_negative_rework(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        parsed = module.apply_training_triage_guard(
            {
                "image_has_positive_candidate": True,
                "visual_positive_class": False,
                "training_action": "rework_bbox_positive_candidate",
                "label_conflict": False,
                "needs_human_review": True,
                "training_priority": "high",
                "issue_flags": [],
            },
            "negative_wall_water",
        )
        self.assertEqual(parsed["training_action"], "label_conflict_review")
        self.assertTrue(parsed["label_conflict"])
        self.assertIn("weak_label_conflict", parsed["issue_flags"])

    def test_anomaly_bbox_qc_training_guard_keeps_positive_bbox_rework_candidate(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        parsed = module.apply_training_triage_guard(
            {
                "image_has_positive_candidate": True,
                "inside_has_target_anomaly": False,
                "visual_positive_class": False,
                "bbox_usable_for_training": False,
                "box_quality": "wrong_target",
                "outside_extension": False,
                "training_action": "reject_from_positive_training",
                "label_conflict": True,
                "needs_human_review": False,
                "training_priority": "reject",
                "rework_type": "none",
                "issue_flags": [],
            },
            "positive_floor_clear_water",
        )
        self.assertEqual(parsed["training_action"], "rework_bbox_positive_candidate")
        self.assertEqual(parsed["rework_type"], "move_bbox")
        self.assertFalse(parsed["bbox_usable_for_training"])
        self.assertFalse(parsed["label_conflict"])
        self.assertTrue(parsed["needs_human_review"])
        self.assertIn("positive_candidate_needs_bbox_rework", parsed["issue_flags"])

    def test_anomaly_bbox_qc_v5_parses_independent_stage_payloads(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        semantic = module.parse_semantic_v5_payload(
            '{"semantic_verdict":"clear_positive","candidate_location":"bottom_right",'
            '"candidate_surface":"floor",'
            '"positive_evidence":["irregular_wet_dry_boundary"],'
            '"hard_negative_type":"shadow","reason":"visible puddle"}'
        )
        geometry = module.parse_geometry_v5_payload(
            '{"inside_target_status":"no_target","outside_relation":"separate_target",'
            '"target_crosses_left_edge":false,"target_crosses_top_edge":false,'
            '"target_crosses_right_edge":false,"target_crosses_bottom_edge":false,'
            '"target_area_fraction":"none","background_excess":false,'
            '"box_quality":"good","target_surface":"equipment_surface",'
            '"dominant_confounder":"equipment_surface","reason":"box misses water"}'
        )
        self.assertEqual(semantic["semantic_verdict"], "clear_positive")
        self.assertEqual(semantic["candidate_surface"], "floor")
        self.assertEqual(semantic["hard_negative_type"], "none")
        self.assertEqual(geometry["inside_target_status"], "no_target")
        self.assertEqual(geometry["box_quality"], "wrong_target")

    def test_anomaly_bbox_qc_v5_fusion_recovers_positive_wrong_box(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        parsed = module.fuse_training_triage_v5(
            {
                "semantic_verdict": "clear_positive",
                "candidate_location": "middle_right",
                "positive_evidence": ["pooled_liquid"],
                "hard_negative_type": "none",
                "semantic_reason": "water is visible away from the current box",
            },
            {
                "inside_target_status": "no_target",
                "outside_relation": "separate_target",
                "box_quality": "wrong_target",
                "target_surface": "equipment_surface",
                "dominant_confounder": "equipment_surface",
                "geometry_reason": "current box contains equipment",
            },
            "positive_floor_clear_water",
        )
        self.assertEqual(parsed["training_action"], "rework_bbox_positive_candidate")
        self.assertEqual(parsed["rework_type"], "move_bbox")
        self.assertTrue(parsed["image_has_positive_candidate"])
        self.assertFalse(parsed["bbox_usable_for_training"])

    def test_anomaly_bbox_qc_v5_edge_checks_force_under_inclusive(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        parsed = module.parse_geometry_v5_payload(
            '{"inside_target_status":"clear_target","outside_relation":"none",'
            '"target_crosses_left_edge":true,"target_crosses_top_edge":false,'
            '"target_crosses_right_edge":false,"target_crosses_bottom_edge":true,'
            '"target_area_fraction":"50_to_75_percent","background_excess":false,'
            '"box_quality":"good","target_surface":"floor",'
            '"dominant_confounder":"none","reason":"water crosses two edges"}'
        )
        self.assertEqual(parsed["outside_relation"], "continuous_main_target")
        self.assertEqual(parsed["box_quality"], "under_inclusive")
        self.assertEqual(parsed["truncated_edges"], ["left", "bottom"])

        over = module.parse_geometry_v5_payload(
            '{"inside_target_status":"clear_target","outside_relation":"none",'
            '"target_crosses_left_edge":false,"target_crosses_top_edge":false,'
            '"target_crosses_right_edge":false,"target_crosses_bottom_edge":false,'
            '"target_area_fraction":"10_to_25_percent","background_excess":false,'
            '"box_quality":"good","target_surface":"floor",'
            '"dominant_confounder":"none","reason":"target occupies a small part of the box"}'
        )
        self.assertEqual(over["box_quality"], "over_inclusive")

        incomplete = module.parse_geometry_v5_payload(
            '{"inside_target_status":"clear_target","outside_relation":"none",'
            '"box_quality":"good","target_surface":"floor",'
            '"dominant_confounder":"none","reason":"edge checks omitted"}'
        )
        self.assertEqual(incomplete["box_quality"], "uncertain")
        self.assertFalse(incomplete["edge_checks_complete"])

    def test_anomaly_bbox_qc_v5_fusion_keeps_negative_visual_positive_in_conflict(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        parsed = module.fuse_training_triage_v5(
            {
                "semantic_verdict": "clear_positive",
                "candidate_location": "bottom_center",
                "positive_evidence": ["wet_texture_change"],
                "hard_negative_type": "none",
                "semantic_reason": "floor water is visible",
            },
            {
                "inside_target_status": "clear_target",
                "outside_relation": "continuous_main_target",
                "box_quality": "under_inclusive",
                "target_surface": "floor",
                "dominant_confounder": "none",
                "geometry_reason": "main water body is cut off",
            },
            "negative_wall_water_or_hard_negative",
        )
        self.assertEqual(parsed["training_action"], "label_conflict_review")
        self.assertEqual(parsed["rework_type"], "expand_bbox")
        self.assertTrue(parsed["label_conflict"])
        self.assertFalse(parsed["bbox_usable_for_training"])

    def test_anomaly_bbox_qc_v5_auto_accept_requires_clear_semantics_and_bbox(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        geometry = {
            "inside_target_status": "clear_target",
            "outside_relation": "minor_same_target_fringe",
            "box_quality": "good",
            "target_surface": "floor_equipment_junction",
            "dominant_confounder": "none",
            "geometry_reason": "main target is contained",
        }
        clear = module.fuse_training_triage_v5(
            {
                "semantic_verdict": "clear_positive",
                "candidate_location": "bottom_left",
                "positive_evidence": ["droplet_or_flow_trail"],
                "hard_negative_type": "none",
                "semantic_reason": "clear liquid trail",
            },
            geometry,
            "positive_floor_clear_water",
        )
        probable = module.fuse_training_triage_v5(
            {
                "semantic_verdict": "probable_positive",
                "candidate_location": "bottom_left",
                "positive_evidence": ["reflection_with_liquid_boundary"],
                "hard_negative_type": "none",
                "semantic_reason": "possible liquid but reflection remains plausible",
            },
            geometry,
            "positive_floor_clear_water",
        )
        self.assertEqual(clear["training_action"], "auto_accept_positive")
        self.assertTrue(clear["bbox_usable_for_training"])
        self.assertEqual(probable["training_action"], "manual_review")
        self.assertFalse(probable["bbox_usable_for_training"])

    def test_anomaly_bbox_qc_v5_stage_disagreement_preserves_asset_for_review(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        parsed = module.fuse_training_triage_v5(
            {
                "semantic_verdict": "hard_negative",
                "candidate_location": "none",
                "candidate_surface": "none",
                "positive_evidence": [],
                "hard_negative_type": "uniform_specular_reflection",
                "semantic_reason": "full image looks dry",
            },
            {
                "inside_target_status": "clear_target",
                "outside_relation": "none",
                "box_quality": "over_inclusive",
                "target_surface": "floor",
                "dominant_confounder": "none",
                "background_excess": True,
                "truncated_edges": [],
                "geometry_reason": "crop shows clear water with excess background",
            },
            "positive_floor_clear_water",
        )
        self.assertEqual(parsed["training_action"], "manual_review")
        self.assertEqual(parsed["rework_type"], "shrink_bbox")
        self.assertTrue(parsed["stage_disagreement"])
        self.assertIn("semantic_bbox_stage_disagreement", parsed["issue_flags"])

    def test_anomaly_bbox_qc_v5_semantic_prompt_is_label_blind(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        prompt = module.semantic_prompt_v5()
        self.assertNotIn("weak_label", prompt)
        self.assertNotIn("sample_id", prompt)
        self.assertIn("无标注、无红框", prompt)

    def test_anomaly_bbox_qc_v5_vetoes_liquid_on_equipment_surface(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        parsed = module.parse_semantic_v5_payload(
            '{"semantic_verdict":"clear_positive","candidate_location":"bottom_left",'
            '"candidate_surface":"equipment_surface",'
            '"positive_evidence":["pooled_liquid"],"hard_negative_type":"none",'
            '"reason":"liquid is visible on top of a cabinet"}'
        )
        self.assertEqual(parsed["semantic_verdict"], "hard_negative")
        self.assertEqual(parsed["hard_negative_type"], "equipment_surface")

        geometry = module.parse_geometry_v5_payload(
            '{"inside_target_status":"clear_target","outside_relation":"continuous_main_target",'
            '"target_crosses_left_edge":true,"target_crosses_top_edge":false,'
            '"target_crosses_right_edge":false,"target_crosses_bottom_edge":false,'
            '"target_area_fraction":"10_to_25_percent","background_excess":true,'
            '"box_quality":"good","target_surface":"equipment_surface",'
            '"dominant_confounder":"none","reason":"liquid is on a cabinet top"}'
        )
        self.assertEqual(geometry["box_quality"], "wrong_target")
        self.assertEqual(module._v5_rework_type("wrong_target", "continuous_main_target", True), "move_bbox")

    def test_anomaly_bbox_qc_v5_clean_verifier_vetoes_provisional_auto(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        primary = {
            "training_action": "auto_accept_positive",
            "training_priority": "high",
            "bbox_usable_for_training": True,
            "needs_human_review": False,
            "image_has_positive_candidate": True,
            "box_quality": "good",
            "issue_flags": [],
            "reason": "primary accepted",
        }
        verifier = {
            "inside_target_status": "clear_target",
            "outside_relation": "continuous_main_target",
            "box_quality": "under_inclusive",
            "target_surface": "floor",
            "dominant_confounder": "dry_floor",
            "edge_checks_complete": True,
            "truncated_edges": ["bottom"],
            "background_excess": False,
            "target_area_fraction": "50_to_75_percent",
            "geometry_reason": "same pool crosses the bottom edge",
            "geometry_raw_response_text": "{}",
        }
        result = module.apply_clean_verifier_v5(
            primary,
            verifier,
            verifier_model="qwen3.6-27b",
        )
        self.assertFalse(result["clean_verifier_passed"])
        self.assertEqual(result["training_action"], "rework_bbox_positive_candidate")
        self.assertEqual(result["rework_type"], "expand_bbox")
        self.assertEqual(result["box_quality"], "under_inclusive")
        self.assertFalse(result["bbox_usable_for_training"])
        self.assertIn("clean_verifier_veto", result["issue_flags"])

    def test_anomaly_bbox_qc_v5_clean_verifier_requires_all_clean_fields(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        primary = {
            "training_action": "auto_accept_positive",
            "training_priority": "high",
            "bbox_usable_for_training": True,
            "needs_human_review": False,
            "image_has_positive_candidate": True,
            "box_quality": "good",
            "issue_flags": [],
            "reason": "primary accepted",
        }
        clean = {
            "inside_target_status": "clear_target",
            "outside_relation": "none",
            "box_quality": "good",
            "target_surface": "floor",
            "dominant_confounder": "none",
            "edge_checks_complete": True,
            "truncated_edges": [],
            "background_excess": False,
            "target_area_fraction": "gt_75_percent",
            "geometry_reason": "tight clean target",
        }
        accepted = module.apply_clean_verifier_v5(primary, clean, verifier_model="qwen3.6-27b")
        unavailable = module.apply_clean_verifier_v5(
            primary,
            None,
            verifier_model="qwen3.6-27b",
            verifier_error="timeout",
        )
        self.assertTrue(accepted["clean_verifier_passed"])
        self.assertEqual(accepted["training_action"], "auto_accept_positive")
        self.assertEqual(unavailable["training_action"], "manual_review")
        self.assertIn("clean_verifier_unavailable", unavailable["issue_flags"])
        self.assertEqual(module.TRIAGE_PROMPT_REVISIONS["v5"], "v5.7")
        self.assertIn("独立反方复核员", module.clean_verifier_prompt_v5())

    def test_anomaly_bbox_qc_builds_crop_evidence_panel(self) -> None:
        from PIL import Image, ImageDraw

        module = load_script_module("run_anomaly_bbox_qc.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.jpg"
            image = Image.new("RGB", (120, 90), color=(220, 220, 220))
            draw = ImageDraw.Draw(image)
            draw.ellipse((30, 30, 55, 48), fill=(180, 200, 210))
            image.save(image_path)

            output_path = root / "panel.jpg"
            data_url, metadata = module.save_crop_evidence_panel(
                image_path,
                {"format": "xyxy", "x1": 25, "y1": 24, "x2": 62, "y2": 55},
                output_path,
                context_expand_ratio=0.5,
                context_min_pad=10,
                request_image_max_side=360,
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
            self.assertEqual(metadata["bbox_clipped"], [25, 24, 62, 55])
            self.assertLessEqual(metadata["context_box"][0], 25)
            self.assertLessEqual(metadata["context_box"][1], 24)
            self.assertGreaterEqual(metadata["context_box"][2], 62)
            self.assertGreaterEqual(metadata["context_box"][3], 55)

            geometry_urls = module.build_geometry_evidence_urls(
                image_path,
                {"format": "xyxy", "x1": 25, "y1": 24, "x2": 62, "y2": 55},
                context_expand_ratio=0.5,
                context_min_pad=10,
                max_side=360,
            )
            self.assertEqual(len(geometry_urls), 6)
            self.assertTrue(all(url.startswith("data:image/jpeg;base64,") for url in geometry_urls))

    def test_anomaly_bbox_qc_resolves_image_by_basename_from_asset_base(self) -> None:
        from PIL import Image

        module = load_script_module("run_anomaly_bbox_qc.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_dir = root / "metadata"
            image_dir = root / "second_screen" / "images_floor_clear_water"
            metadata_dir.mkdir()
            image_dir.mkdir(parents=True)
            image_path = image_dir / "sample_001.jpg"
            Image.new("RGB", (20, 20), color=(255, 255, 255)).save(image_path)

            resolved, exists, _ = module.resolve_image_for_qc(
                "data/processed_baoshan_speed/metadata/images/sample_001.jpg",
                metadata_dir / "sample_001.json",
                [root / "second_screen"],
            )

            self.assertTrue(exists)
            self.assertEqual(resolved, image_path)

    def test_anomaly_bbox_qc_resolves_windows_image_uri_from_asset_base(self) -> None:
        from PIL import Image

        module = load_script_module("run_anomaly_bbox_qc.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_dir = root / "metadata"
            image_dir = root / "i2i_outputs" / "generated_images"
            metadata_dir.mkdir()
            image_dir.mkdir(parents=True)
            image_path = image_dir / "sample_001.png"
            Image.new("RGB", (20, 20), color=(255, 255, 255)).save(image_path)

            resolved, exists, _ = module.resolve_image_for_qc(
                r"C:\Users\annotator\batch\i2i_outputs\generated_images\sample_001.png",
                metadata_dir / "sample_001.json",
                [root / "i2i_outputs"],
            )

            self.assertTrue(exists)
            self.assertEqual(resolved, image_path)

    def test_anomaly_bbox_qc_infers_positive_hint_from_metadata(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        sample = {
            "image_asset": {"scene_context": {"inspection_content": "water_leak"}},
            "objects": [
                {
                    "classification": {
                        "multi_labels": [
                            {"label_key": "anomaly_type", "label_value": "water_leak"}
                        ]
                    }
                }
            ],
        }

        hint = module.weak_label_hint_from_sample(Path("metadata/sample.json"), sample)

        self.assertEqual(hint, "positive_floor_clear_water")

    def test_anomaly_bbox_qc_keeps_directory_hint_over_metadata(self) -> None:
        module = load_script_module("run_anomaly_bbox_qc.py")
        sample = {
            "image_asset": {"scene_context": {"inspection_content": "water_leak"}},
            "objects": [],
        }

        hint = module.weak_label_hint_from_sample(
            Path("images_wall_water/metadata/sample.json"),
            sample,
        )

        self.assertEqual(hint, "negative_wall_water_or_hard_negative")

    def test_bbox_triage_summarizer_loads_direct_run_root(self) -> None:
        module = load_script_module("summarize_bbox_triage_runs.py")
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "single_run"
            run_root.mkdir()
            (run_root / "bbox_qc_results.jsonl").write_text(
                json.dumps({"sample_id": "sample", "object_id": "obj"}) + "\n",
                encoding="utf-8",
            )

            records = module.load_records(run_root)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["_run"], "single_run")

    def test_bbox_triage_summarizer_validates_and_classifies_transitions(self) -> None:
        module = load_script_module("summarize_bbox_triage_runs.py")
        current = [
            {
                "_run": "first_clear_water",
                "sample_id": "rescued",
                "object_id": "obj_1",
                "weak_label_hint": "positive_floor_clear_water",
                "training_action": "rework_bbox_positive_candidate",
                "box_quality": "wrong_target",
                "rework_type": "move_bbox",
                "triage_prompt_revision": "v5.6",
                "edge_checks_complete": True,
                "target_area_fraction": "none",
                "reason": "target is outside <script>alert(1)</script>",
            },
            {
                "_run": "first_others",
                "sample_id": "negative",
                "object_id": "obj_1",
                "weak_label_hint": "negative_other_or_label_conflict_candidate",
                "training_action": "reject_from_positive_training",
                "box_quality": "wrong_target",
                "rework_type": "relabeled_negative_or_other",
                "triage_prompt_revision": "v5.6",
                "edge_checks_complete": True,
                "target_area_fraction": "none",
            },
        ]
        baseline = [
            {
                **current[0],
                "training_action": "reject_from_positive_training",
                "triage_prompt_revision": "v4",
            },
            {
                **current[1],
                "training_action": "reject_from_positive_training",
                "triage_prompt_revision": "v4",
            },
        ]

        integrity = module.validate_records(current, expected_total=2, expected_revision="v5.6")
        self.assertTrue(integrity["valid"])
        transitions, issues = module.build_transition_rows(current, baseline)
        self.assertFalse(issues)
        self.assertEqual(transitions[0]["transition"], "unchanged")
        self.assertEqual(transitions[1]["transition"], "rescued_from_reject")

        replacement = {
            **current[0],
            "training_action": "auto_accept_positive",
            "triage_prompt_revision": "v5.7",
        }
        merged = module.merge_replacements(current, [replacement])
        self.assertEqual(merged[0]["training_action"], "auto_accept_positive")
        self.assertEqual(merged[0]["replaced_triage_prompt_revision"], "v5.6")
        with self.assertRaises(ValueError):
            module.merge_replacements(current, [{**replacement, "sample_id": "unknown"}])

        with tempfile.TemporaryDirectory() as tmp:
            gallery = Path(tmp) / "gallery.html"
            module.write_gallery(current, module.metric_counts(current), gallery, transitions)
            rendered = gallery.read_text(encoding="utf-8")
            self.assertIn("BBox QC v5.6", rendered)
            self.assertIn("rescued_from_reject", rendered)
            self.assertNotIn("<script>alert(1)</script>", rendered)

    def test_bbox_triage_summarizer_rejects_incomplete_v5_records(self) -> None:
        module = load_script_module("summarize_bbox_triage_runs.py")
        record = {
            "sample_id": "sample",
            "object_id": "obj",
            "weak_label_hint": "positive_floor_clear_water",
            "training_action": "manual_review",
            "triage_prompt_revision": "v5.6",
            "edge_checks_complete": False,
            "target_area_fraction": "uncertain",
        }
        integrity = module.validate_records([record, dict(record)], expected_total=2, expected_revision="v5.6")
        self.assertFalse(integrity["valid"])
        self.assertEqual(integrity["unique_key_count"], 1)
        self.assertEqual(integrity["incomplete_edge_check_count"], 2)
        self.assertEqual(integrity["missing_target_area_count"], 2)

    def test_bbox_cross_model_audit_builds_action_comparison(self) -> None:
        module = load_script_module("summarize_bbox_cross_model_audit.py")
        primary = {
            ("sample", "obj"): {
                "sample_id": "sample",
                "object_id": "obj",
                "weak_label_hint": "positive_floor_clear_water",
                "training_action": "rework_bbox_positive_candidate",
                "box_quality": "wrong_target",
            }
        }
        audit = {
            ("sample", "obj"): {
                "_run": "first_clear_water",
                "sample_id": "sample",
                "object_id": "obj",
                "weak_label_hint": "positive_floor_clear_water",
                "training_action": "manual_review",
                "box_quality": "uncertain",
            }
        }
        selection = [
            {
                "sample_id": "sample",
                "object_id": "obj",
                "selection_reasons": ["rescued_v4_reject"],
            }
        ]
        rows = module.build_rows(primary, audit, selection)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["exact_action_agreement"])
        self.assertEqual(rows[0]["primary_route"], "rework")
        self.assertEqual(rows[0]["audit_route"], "review")
        self.assertEqual(rows[0]["selection_reasons"], "rescued_v4_reject")

    def test_bbox_mask_geometry_audit_detects_derived_bbox(self) -> None:
        from PIL import Image, ImageDraw

        module = load_script_module("audit_bbox_mask_geometry.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "images_clear_water"
            metadata_dir = root / "metadata"
            mask_dir = root / "masks"
            metadata_dir.mkdir(parents=True)
            mask_dir.mkdir()
            mask_path = mask_dir / "sample_mask.png"
            mask = Image.new("L", (10, 10), color=0)
            ImageDraw.Draw(mask).rectangle((2, 3, 5, 7), fill=255)
            mask.save(mask_path)
            metadata_path = metadata_dir / "sample.json"
            sample = {"sample_id": "sample", "image_asset": {"width": 10, "height": 10}}
            obj = {
                "object_id": "obj",
                "box": {"format": "xyxy", "x1": 2, "y1": 3, "x2": 6, "y2": 8},
                "geometry_detail": {
                    "mask_uri": "data/masks/sample_mask.png",
                    "generation_params": {"final_bbox_source": "image_difference_connected_components"},
                },
            }
            row = module.inspect_object(metadata_path, sample, obj)
            self.assertTrue(row["mask_exists"])
            self.assertTrue(row["mask_bbox_exact_match"])
            self.assertEqual(row["mask_inside_bbox_ratio"], 1.0)
            self.assertEqual(row["mask_fill_ratio"], 1.0)
            self.assertEqual(row["issue_flags"], "")

    def test_bbox_mask_geometry_audit_resolves_windows_mask_uri(self) -> None:
        from PIL import Image, ImageDraw

        module = load_script_module("audit_bbox_mask_geometry.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "metadata" / "sample.json"
            mask_path = root / "i2i_outputs" / "debug" / "masks" / "sample_mask.png"
            metadata_path.parent.mkdir()
            mask_path.parent.mkdir(parents=True)
            mask = Image.new("L", (10, 10), color=0)
            ImageDraw.Draw(mask).rectangle((2, 3, 5, 7), fill=255)
            mask.save(mask_path)
            sample = {"sample_id": "sample", "image_asset": {"width": 10, "height": 10}}
            obj = {
                "object_id": "obj",
                "box": {"format": "xyxy", "x1": 2, "y1": 3, "x2": 6, "y2": 8},
                "geometry_detail": {
                    "mask_uri": r"C:\batch\i2i_outputs\debug\masks\sample_mask.png"
                },
            }

            row = module.inspect_object(
                metadata_path,
                sample,
                obj,
                [root / "i2i_outputs"],
            )

            self.assertTrue(row["mask_exists"])
            self.assertEqual(Path(row["mask_path"]).resolve(), mask_path.resolve())
            self.assertTrue(row["mask_bbox_exact_match"])

    def test_asset_qc_maps_crop_filename_to_manifest(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "frames"
            crop_dir = root / "crops"
            image_dir.mkdir()
            crop_dir.mkdir()
            frame_path = image_dir / "001.mp4_20dc896e_frame_000000.jpg"
            crop_path = crop_dir / "sample_001.mp4_20dc896e_frame_000000_person_1.jpg"
            Image.new("RGB", (100, 80), color=(255, 255, 255)).save(frame_path)
            Image.new("RGB", (30, 60), color=(255, 255, 255)).save(crop_path)
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_001.mp4_20dc896e_frame_000000",
                        "image_id": "001.mp4_20dc896e_frame_000000",
                        "image_uri": "frames/001.mp4_20dc896e_frame_000000.jpg",
                        "source_type": "cctv",
                    }
                ],
                ["sample_id", "image_id", "image_uri", "source_type"],
            )

            crop_info = infer_crop_source(crop_path)
            self.assertEqual(crop_info["sample_id"], "sample_001.mp4_20dc896e_frame_000000")
            self.assertEqual(crop_info["object_type"], "person")
            bbox_info = infer_crop_source(crop_dir / "sample_001.mp4_20dc896e_frame_000000_bbox_2.jpg")
            self.assertEqual(bbox_info["sample_id"], "sample_001.mp4_20dc896e_frame_000000")
            self.assertEqual(bbox_info["object_type"], "bbox")
            bare_info = infer_crop_source(crop_dir / "sample_001.mp4_20dc896e_frame_000000_2.jpg")
            self.assertEqual(bare_info["sample_id"], "sample_001.mp4_20dc896e_frame_000000")
            self.assertEqual(bare_info["object_type"], "unknown")
            bbox_person_info = infer_crop_source(crop_dir / "sample_001.mp4_20dc896e_frame_000000_bbox_person_3.jpg")
            self.assertEqual(bbox_person_info["sample_id"], "sample_001.mp4_20dc896e_frame_000000")
            self.assertEqual(bbox_person_info["object_type"], "person")

            report = run_asset_qc(
                image_paths=[image_dir],
                crop_paths=[crop_dir],
                manifest_csv=manifest_path,
                output_dir=root / "qc",
            )
            self.assertEqual(report["summary"]["total_assets"], 2)
            self.assertEqual(report["summary"]["failed_assets"], 0)
            crop_record = [item for item in report["assets"] if item["asset_kind"] == "crop"][0]
            self.assertEqual(crop_record["sample_id"], "sample_001.mp4_20dc896e_frame_000000")
            self.assertEqual(crop_record["source_type"], "cctv")
            self.assertEqual(crop_record["object_index"], "1")

    def test_asset_qc_maps_generated_crop_from_metadata(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "generated_images"
            crop_dir = root / "crops"
            image_dir.mkdir()
            crop_dir.mkdir()
            image_path = image_dir / "sample_000001.png"
            crop_path = crop_dir / "sample_000001_obj_000001_crop.jpg"
            Image.new("RGB", (100, 80), color=(255, 255, 255)).save(image_path)
            Image.new("RGB", (30, 60), color=(255, 255, 255)).save(crop_path)
            sample = normalize_autolabel_sample(
                {
                    "sample_id": "sample_000001",
                    "image_asset": {
                        "image_id": "image_generated_001",
                        "image_uri": "generated_images/sample_000001.png",
                        "width": 100,
                        "height": 80,
                        "source_type": "generated",
                    },
                    "objects": [
                        make_object(
                            object_id="obj_000001",
                            object_type="leakage_area",
                            box={"format": "xyxy", "x1": 10, "y1": 10, "x2": 40, "y2": 70},
                            geometry_source="synthetic_generator",
                            crop={
                                "crop_id": "crop_000001",
                                "crop_uri": "crops/sample_000001_obj_000001_crop.jpg",
                                "crop_box": {"format": "xyxy", "x1": 10, "y1": 10, "x2": 40, "y2": 70},
                                "crop_expand_ratio": 0.0,
                                "is_valid_crop": True,
                            },
                        )
                    ],
                    "workflow": {"workflow_status": "classified"},
                    "export": {"export_format": "labelstudio", "export_status": "not_exported"},
                }
            )
            metadata_dir = root / "metadata"
            write_json(metadata_dir / "sample_000001.json", sample)

            report = run_asset_qc(
                image_paths=[image_dir],
                crop_paths=[crop_dir],
                metadata_dir=metadata_dir,
                output_dir=root / "qc",
            )
            self.assertEqual(report["summary"]["failed_assets"], 0)
            self.assertEqual(report["summary"]["needs_human_review_assets"], 0)
            crop_record = [item for item in report["assets"] if item["asset_kind"] == "crop"][0]
            self.assertEqual(crop_record["sample_id"], "sample_000001")
            self.assertEqual(crop_record["object_type"], "leakage_area")

    def test_clear_water_filter_rule_accepts_floor_water_and_rejects_hard_negative(self) -> None:
        module = load_script_module("run_clear_water_filter.py")
        accepted, reason = module.selected_by_rule(
            {
                "can_see_image": True,
                "usable": True,
                "hard_negative_type": "none",
                "liquid_location": "floor",
                "floor_clear_water_visible": True,
                "confidence": 0.82,
                "evidence_strength": "medium",
                "decision": "accept_clear_water",
            },
            min_confidence=0.55,
            allow_weak=False,
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "model_accept_strong_or_medium")

        accepted, reason = module.selected_by_rule(
            {
                "can_see_image": True,
                "usable": True,
                "hard_negative_type": "wall_water_only",
                "liquid_location": "wall",
                "floor_clear_water_visible": False,
                "confidence": 0.95,
                "evidence_strength": "strong",
                "decision": "accept_clear_water",
            },
            min_confidence=0.55,
            allow_weak=True,
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "hard_negative:wall_water_only")

    def test_visual_calibrator_threshold_keeps_weak_negative_out(self) -> None:
        import numpy as np

        module = load_script_module("run_clear_water_visual_calibrator.py")
        rows = [
            {
                "folder": "images_floor_clear_water",
                "decision": "accept_clear_water",
                "can_see_image": "true",
                "usable": "true",
                "floor_clear_water_visible": "true",
                "hard_negative_type": "none",
            },
            {
                "folder": "images_wall_water",
                "decision": "accept_clear_water",
                "can_see_image": "true",
                "usable": "true",
                "floor_clear_water_visible": "true",
                "hard_negative_type": "none",
            },
            {
                "folder": "images_floor_clear_water",
                "decision": "accept_clear_water",
                "can_see_image": "true",
                "usable": "true",
                "floor_clear_water_visible": "true",
                "hard_negative_type": "none",
            },
        ]
        threshold, stats = module.choose_threshold(
            rows,
            np.asarray([0.91, 0.72, 0.84], dtype=np.float32),
            require_vlm_accept=True,
            target_denominator=3,
            target_max_rate=1.0,
        )
        self.assertGreater(threshold, 0.72)
        self.assertEqual(stats["weak_negative_selected"], 0)
        self.assertEqual(stats["weak_positive_selected"], 2)


if __name__ == "__main__":
    unittest.main()
