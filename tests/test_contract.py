from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
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
from autolabel.config_loader import load_config
from autolabel.contract_normalizer import normalize_autolabel_sample
from autolabel.model_config import build_detector_runtime_config, resolve_classification_runtime, resolve_generation_runtime
from autolabel.modules.classification.dry_run import DryRunClassificationModule
from autolabel.modules.generation.manifest_builder import write_water_leak_generation_manifest
from autolabel.modules.generation.preflight import run_generation_preflight
from autolabel.preprocess import estimate_extracted_frame_count
from autolabel.exporters.labelstudio import build_labelstudio_config, export_metadata_dir, sample_to_labelstudio_task
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

    def test_labelstudio_generated_quality_gate_rejects_failed_generated_samples(self) -> None:
        def generated_sample(sample_id: str, postprocess_status: str, passes_quality: bool) -> dict:
            sample = deepcopy(read_json(ROOT / "schemas" / "autolabel_sample.example.json"))
            sample["sample_id"] = sample_id
            sample["image_asset"]["source_type"] = "generated"
            sample["objects"][0]["geometry_detail"]["generation_params"] = {
                "localizer": {
                    "postprocess_status": postprocess_status,
                    "used": "pgcd_lpips",
                    "fallback_used": False,
                    "quality": {
                        "passes_quality": passes_quality,
                        "quality_reason": None if passes_quality else "prompt_alignment_low",
                    },
                    "debug_artifacts": {"heatmap": "metadata/debug/heatmap.png"},
                    "metrics": {"pgcd_component_score": 0.7},
                }
            }
            return sample

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_dir = root / "metadata"
            write_json(metadata_dir / "good.json", generated_sample("sample_good", "success", True))
            write_json(metadata_dir / "bad.json", generated_sample("sample_bad", "quality_failed", False))
            direct_sample = read_json(ROOT / "schemas" / "autolabel_sample.example.json")
            direct_sample["sample_id"] = "sample_direct"
            write_json(metadata_dir / "direct.json", direct_sample)

            output_path = root / "export" / "import.json"
            rejected_path = root / "export" / "rejected.json"
            tasks = export_metadata_dir(
                metadata_dir,
                output_path,
                generated_quality_gate=True,
                rejected_report_path=rejected_path,
            )

            self.assertEqual({task["data"]["sample_id"] for task in tasks}, {"sample_good", "sample_direct"})
            report = read_json(rejected_path)
            self.assertEqual(report["generated_samples"], 2)
            self.assertEqual(report["exported_generated_samples"], 1)
            self.assertEqual(report["rejected_samples"], 1)
            self.assertEqual(report["rejections"][0]["sample_id"], "sample_bad")
            self.assertIn("quality_", report["rejections"][0]["object_rejections"][0]["reason"])

    def test_yaml_model_selection_is_resolved_from_config(self) -> None:
        config = load_config(ROOT / "configs" / "autolabel.yaml")
        generation = resolve_generation_runtime(config)
        classification = resolve_classification_runtime(config)
        detector = build_detector_runtime_config(config)
        preprocess = config["preprocess"]
        direct = config["direct_annotation"]
        self.assertEqual(generation["vlm_model_name"], "qwen3.6-27b")
        self.assertEqual(generation["image_model_name"], "doubao-seedream-5-0-pro-260628")
        self.assertIn("model", classification)
        self.assertIn("model_profiles", detector)
        self.assertEqual(classification["model"], "qwen3.6-27b")
        self.assertEqual(classification["api_url"], "https://deepseek.gds-services.com/v1")
        self.assertEqual(detector["services"]["ppe_person"]["model_ref"], "ppe_person_vlm_labelstudio_detector")
        self.assertEqual(detector["model_profiles"]["ppe_person_vlm_labelstudio_detector"]["model_name"], "qwen3.6-27b")
        self.assertEqual(
            detector["model_profiles"]["ppe_person_vlm_labelstudio_detector"]["base_url"],
            "https://deepseek.gds-services.com/v1",
        )
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

    def test_seedream5_generation_profile_is_resolved_from_config(self) -> None:
        config = load_config(ROOT / "configs" / "autolabel.yaml")
        self.assertIn("seedream5_image_editor", config["models"]["generation"]["image_generators"])

        config["generation"]["image_model_key"] = "seedream5_image_editor"
        generation = resolve_generation_runtime(config)

        self.assertEqual(generation["image_model_name"], "doubao-seedream-5-0-pro-260628")
        self.assertEqual(generation["image_profile"]["api_key_env"], "ARK_API_KEY")
        self.assertEqual(
            generation["image_profile"]["endpoint"],
            "https://ark.cn-beijing.volces.com/api/v3",
        )

    def test_generation_run_overrides_can_switch_vlm_and_image_models(self) -> None:
        from autolabel.orchestrator import apply_generation_run_overrides

        config = load_config(ROOT / "configs" / "autolabel.yaml")
        config["models"]["generation"]["vlm"]["private_grid_selector"] = {
            "provider": "custom",
            "model_name": "private-vlm-grid",
            "credential_ref": "local_or_private_model",
            "api_key_env": "PRIVATE_MODEL_API_KEY",
            "endpoint_env": "CUSTOM_VLM_ENDPOINT",
            "endpoint": "http://127.0.0.1:9000/vlm",
        }

        apply_generation_run_overrides(
            config,
            vlm_model_key="private_grid_selector",
            image_model_key="seedream5_image_editor",
            workers=3,
        )
        runtime = resolve_generation_runtime(config, anomaly_type="water_leak")

        self.assertEqual(runtime["vlm_model_name"], "private-vlm-grid")
        self.assertEqual(runtime["image_model_name"], "doubao-seedream-5-0-pro-260628")
        self.assertEqual(config["generation"]["workers"], 3)

    def test_generation_runtime_keeps_localizer_cli_args_separate(self) -> None:
        config = load_config(ROOT / "configs" / "autolabel.yaml")
        generation = resolve_generation_runtime(config)
        localizer_cli_args = generation["localizer_cli_args"]

        self.assertEqual(generation["extra_cli_args"], [])
        self.assertIn("--localizer", localizer_cli_args)
        self.assertIn("pgcd_lpips", localizer_cli_args)
        self.assertIn("--localizer-fallback", localizer_cli_args)
        self.assertIn("rgb_diff", localizer_cli_args)
        self.assertIn("--pgcd-threshold", localizer_cli_args)
        self.assertIn("--pgcd-min-component-area", localizer_cli_args)
        self.assertIn("--pgcd-max-global-change-ratio", localizer_cli_args)
        self.assertIn("--sam2-model", localizer_cli_args)

    def test_config_includes_localizer_policy_examples(self) -> None:
        config = load_config(ROOT / "configs" / "autolabel.yaml")
        generation_module = config["modules"]["generation"]

        self.assertEqual(generation_module["backend"], "vlm_wan_autolabel")
        self.assertEqual(generation_module["backends"]["vlm_wan_autolabel"]["project_dir"], "external/I2I")
        self.assertFalse(generation_module["backends"]["vlm_wan_autolabel"]["pass_localizer_cli_args"])
        self.assertEqual(config["models"]["generation"]["active_image_generator"], "seedream5_image_editor")
        self.assertEqual(config["generation"]["image_model_key"], "seedream5_image_editor")
        self.assertEqual(generation_module["localizer_policy"]["water_leak"]["primary"], "pgcd_lpips")
        self.assertEqual(generation_module["localizer_policy"]["coolant_leak"]["primary"], "pgcd_lpips")
        self.assertEqual(generation_module["localizer_policy"]["diesel_leak"]["primary"], "rgb_diff")
        self.assertEqual(generation_module["localizer_policy"]["oil_leak"]["fallback"], "pgcd_lpips")

    def test_i2i_generator_appends_extra_cli_args(self) -> None:
        from autolabel.adapters.i2i_generator import I2IGenerator

        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            entrypoint = project_dir / "src" / "main.py"
            entrypoint.parent.mkdir(parents=True, exist_ok=True)
            entrypoint.write_text("print('ok')\n", encoding="utf-8")

            recorded = {}

            def fake_run(cmd, **kwargs):
                recorded["cmd"] = cmd
                recorded["kwargs"] = kwargs
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("autolabel.adapters.i2i_generator.subprocess.run", side_effect=fake_run):
                I2IGenerator(project_dir).run(
                    tasks_csv=project_dir / "tasks.csv",
                    image_root=project_dir / "images",
                    output_root=project_dir / "output",
                    extra_cli_args=["--localizer", "pgcd_lpips", "--localizer-debug"],
                )

            self.assertIn("--localizer", recorded["cmd"])
            self.assertIn("pgcd_lpips", recorded["cmd"])
            self.assertIn("--localizer-debug", recorded["cmd"])

    def test_i2i_generator_resolves_relative_project_dir_before_subprocess(self) -> None:
        from autolabel.adapters.i2i_generator import I2IGenerator

        project_dir = ROOT / "external" / "I2I"
        recorded = {}

        def fake_run(cmd, **kwargs):
            recorded["cmd"] = cmd
            recorded["kwargs"] = kwargs
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("autolabel.adapters.i2i_generator.subprocess.run", side_effect=fake_run):
            I2IGenerator(Path("external") / "I2I").run(
                tasks_csv=ROOT / "configs" / "task_manifest.example.csv",
                image_root=ROOT,
                output_root=ROOT / "data" / "processed" / "i2i_outputs",
                dry_run=True,
            )

        self.assertEqual(Path(recorded["cmd"][1]), project_dir / "src" / "main.py")
        self.assertEqual(Path(recorded["kwargs"]["cwd"]), project_dir)

    def test_i2i_diff_localizer_reads_unicode_windows_paths(self) -> None:
        from PIL import Image, ImageDraw

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from diff_localizer import localize_change_bbox
        finally:
            sys.path.remove(str(src_dir))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_path = root / "A1-制冷方舱原图.jpg"
            edited_path = root / "A1-制冷方舱生成.jpg"
            mask_path = root / "输出" / "掩码.png"

            original = Image.new("RGB", (96, 80), (80, 80, 80))
            edited = original.copy()
            draw = ImageDraw.Draw(edited)
            draw.rectangle((30, 28, 58, 48), fill=(225, 225, 225))
            original.save(original_path)
            edited.save(edited_path)

            result = localize_change_bbox(
                str(original_path),
                str(edited_path),
                (0, 0, 96, 80),
                "water_leak",
                str(mask_path),
            )

            self.assertTrue(mask_path.exists())
            self.assertEqual(result["status"], "ok")
            self.assertLessEqual(result["bbox"]["x1"], 35)
            self.assertGreaterEqual(result["bbox"]["x2"], 55)

    def test_i2i_allows_coarse_bbox_for_repo_localizer_postprocess(self) -> None:
        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from main import _is_refined_final_bbox
        finally:
            sys.path.remove(str(src_dir))

        expanded_bbox = (10, 20, 110, 120)
        coarse_box = {"format": "xyxy", "x1": 10, "y1": 20, "x2": 110, "y2": 120}
        refined_box = {"format": "xyxy", "x1": 36, "y1": 42, "x2": 66, "y2": 72}

        self.assertFalse(_is_refined_final_bbox(coarse_box, expanded_bbox))
        self.assertTrue(_is_refined_final_bbox(refined_box, expanded_bbox))

    def test_seedream_client_pastes_generated_crop_back_into_original(self) -> None:
        from PIL import Image

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from config import ModelServiceConfig
            from wan_image_client import WanImageClient
        finally:
            sys.path.remove(str(src_dir))

        class FakeResponse:
            ok = True
            status_code = 200

            def json(self):
                return {"data": [{"url": "https://example.invalid/generated.png"}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_path = root / "original.jpg"
            output_path = root / "generated.png"
            request_log_path = root / "request.json"
            response_log_path = root / "response.json"
            Image.new("RGB", (100, 80), (70, 80, 90)).save(original_path)

            def fake_download(_value, target):
                Image.new("RGB", (256, 256), (220, 235, 245)).save(target)

            client = WanImageClient(
                "doubao-seedream-5-0-pro-260628",
                ModelServiceConfig(
                    api_key="test-key",
                    provider="volcengine_ark",
                    endpoint="https://ark.cn-beijing.volces.com/api/plan/v3/images/generations",
                    api_key_env="ARK_API_KEY",
                    endpoint_env="SEEDREAM_BASE_URL",
                ),
            )

            with (
                patch.dict("os.environ", {"SEEDREAM_SIZE": "auto", "SEEDREAM_N": "4", "SEEDREAM_ALLOW_REFERENCE_GENERATION_DEBUG": "1"}),
                patch("wan_image_client._generate_seedream_with_openai_sdk", return_value={"data": [{"url": "https://example.invalid/generated.png"}]}),
                patch("wan_image_client._download_or_decode_image", side_effect=fake_download),
            ):
                client.edit_image_with_wan(
                    image_path=str(original_path),
                    prompt="make a small water leak",
                    negative_prompt="",
                    bbox=(20, 10, 60, 50),
                    output_path=str(output_path),
                    anomaly_type="water_leak",
                    request_log_path=str(request_log_path),
                    response_log_path=str(response_log_path),
                )

            with Image.open(original_path) as original_image, Image.open(output_path) as generated_image:
                original = original_image.convert("RGB")
                generated = generated_image.convert("RGB")
                self.assertEqual(generated.size, original.size)
                self.assertEqual(generated.getpixel((5, 5)), original.getpixel((5, 5)))
                self.assertEqual(generated.getpixel((95, 75)), original.getpixel((95, 75)))
                self.assertNotEqual(generated.getpixel((30, 20)), original.getpixel((30, 20)))

            request_log = read_json(request_log_path)
            body = request_log["body"]
            self.assertTrue(body["seedream_local_crop_mode"])
            self.assertEqual(body["source_edit_bbox"], [20, 10, 60, 50])
            self.assertEqual(body["edit_bbox"], [0, 0, 40, 40])
            self.assertEqual(body["size"], "1920x1920")
            self.assertEqual(body["n"], 1)
            self.assertTrue(body["image"].endswith("seedream_input_crop.jpg"))
            response_log = read_json(response_log_path)
            self.assertEqual(response_log["local_edit"]["mode"], "crop_then_paste")

    def test_seedream_reference_endpoint_is_rejected_for_production_local_edit(self) -> None:
        from PIL import Image

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from config import ModelServiceConfig
            from wan_image_client import WanImageClient
        finally:
            sys.path.remove(str(src_dir))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_path = root / "original.jpg"
            Image.new("RGB", (100, 80), (70, 80, 90)).save(original_path)
            client = WanImageClient(
                "doubao-seedream-5-0-pro-260628",
                ModelServiceConfig(
                    api_key="test-key",
                    provider="volcengine_ark",
                    endpoint="https://ark.cn-beijing.volces.com/api/plan/v3/images/generations",
                    api_key_env="ARK_API_KEY",
                    endpoint_env="SEEDREAM_BASE_URL",
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "reference-generation-only"):
                client.edit_image_with_wan(
                    image_path=str(original_path),
                    prompt="make a small water leak",
                    negative_prompt="",
                    bbox=(20, 10, 60, 50),
                    output_path=str(root / "generated.png"),
                    anomaly_type="water_leak",
                )

    def test_seedream_single_image_experiment_uses_one_full_image_input(self) -> None:
        from PIL import Image

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from config import ModelServiceConfig
            from wan_image_client import WanImageClient
        finally:
            sys.path.remove(str(src_dir))

        class FakeResponse:
            ok = True
            status_code = 200

            def json(self):
                return {"data": [{"url": "https://example.invalid/generated.png"}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_path = root / "original.jpg"
            output_path = root / "generated.png"
            request_log_path = root / "request.json"
            Image.new("RGB", (100, 80), (70, 80, 90)).save(original_path)

            def fake_download(_value, target):
                Image.new("RGB", (100, 80), (90, 100, 110)).save(target)

            client = WanImageClient(
                "doubao-seedream-5-0-pro-260628",
                ModelServiceConfig(
                    api_key="test-key",
                    provider="volcengine_ark",
                    endpoint="https://ark.cn-beijing.volces.com/api/plan/v3/images/generations",
                    api_key_env="ARK_API_KEY",
                    endpoint_env="SEEDREAM_BASE_URL",
                ),
            )

            with (
                patch.dict("os.environ", {"SEEDREAM_SIZE": "2k"}, clear=False),
                patch("wan_image_client._generate_seedream_with_openai_sdk", return_value={"data": [{"url": "https://example.invalid/generated.png"}]}),
                patch("wan_image_client._download_or_decode_image", side_effect=fake_download),
            ):
                client.edit_image_with_wan(
                    image_path=str(original_path),
                    prompt="make a small water leak",
                    negative_prompt="",
                    bbox=(20, 10, 60, 50),
                    output_path=str(output_path),
                    anomaly_type="water_leak",
                    request_log_path=str(request_log_path),
                    seedream_mode="single_image_edit",
                )

            body = read_json(request_log_path)["body"]
            self.assertEqual(body["seedream_mode"], "single_image_edit")
            self.assertTrue(body["experimental_seedream"])
            self.assertEqual(body["n"], 1)
            self.assertEqual(body["model"], "doubao-seedream-5-0-pro-260628")
            self.assertEqual(body["image"], str(original_path))
            self.assertNotIn("seedream_local_crop_mode", body)
            self.assertIn("80-160 px", body["prompt"])
            self.assertIn("Do not redraw the room", body["prompt"])
            self.assertNotIn("or inside the selected grid", body["prompt"])
            self.assertNotIn("Additional anomaly detail", body["prompt"])
            self.assertNotIn("生成一张真实的工业监控异常图像", body["prompt"])

    def test_seedream_boxed_fusion_uses_guide_and_reference_inputs(self) -> None:
        from PIL import Image

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from config import ModelServiceConfig
            from wan_image_client import WanImageClient
        finally:
            sys.path.remove(str(src_dir))

        class FakeResponse:
            ok = True
            status_code = 200

            def json(self):
                return {"data": [{"url": "https://example.invalid/generated.png"}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guide_path = root / "guide.jpg"
            reference_path = root / "water_ref.png"
            request_log_path = root / "request.json"
            Image.new("RGB", (100, 80), (70, 80, 90)).save(guide_path)
            Image.new("RGB", (40, 30), (160, 180, 190)).save(reference_path)

            def fake_download(_value, target):
                Image.new("RGB", (100, 80), (90, 100, 110)).save(target)

            seedream_payloads = []

            def fake_generate(_endpoint, _api_key, request_payload):
                seedream_payloads.append(request_payload)
                return {"data": [{"url": "https://example.invalid/generated.png"}]}

            client = WanImageClient(
                "doubao-seedream-5-0-pro-260628",
                ModelServiceConfig(
                    api_key="test-key",
                    provider="volcengine_ark",
                    endpoint="https://ark.cn-beijing.volces.com/api/plan/v3/images/generations",
                    api_key_env="ARK_API_KEY",
                    endpoint_env="SEEDREAM_BASE_URL",
                ),
            )

            with (
                patch.dict("os.environ", {"SEEDREAM_SIZE": "2k"}, clear=False),
                patch("wan_image_client._generate_seedream_with_openai_sdk", side_effect=fake_generate),
                patch("wan_image_client._download_or_decode_image", side_effect=fake_download),
            ):
                client.edit_image_with_wan(
                    image_path=str(guide_path),
                    prompt="fuse a water stain",
                    negative_prompt="",
                    bbox=(20, 10, 60, 50),
                    output_path=str(root / "generated.png"),
                    anomaly_type="water_leak",
                    request_log_path=str(request_log_path),
                    seedream_mode="boxed_fusion",
                    seedream_reference_paths=[str(reference_path)],
                )

            body = read_json(request_log_path)["body"]
            self.assertEqual(body["seedream_mode"], "boxed_fusion")
            self.assertEqual(body["n"], 1)
            self.assertEqual(body["model"], "doubao-seedream-5-0-pro-260628")
            self.assertEqual(body["image"], [str(guide_path), str(reference_path)])
            self.assertNotIn("images", body)
            self.assertIn("80-160 px", body["prompt"])
            self.assertIn("Do not cover or repaint the whole rectangle", body["prompt"])
            self.assertNotIn("Additional anomaly detail", body["prompt"])
            self.assertNotIn("生成一张真实的工业监控异常图像", body["prompt"])
            self.assertEqual(len(seedream_payloads), 1)
            self.assertEqual(len(seedream_payloads[0]["image"]), 2)
            self.assertTrue(seedream_payloads[0]["image"][0].startswith("data:image/jpeg;base64,"))
            self.assertTrue(seedream_payloads[0]["image"][1].startswith("data:image/png;base64,"))
            self.assertNotIn("images", seedream_payloads[0])

    def test_seedream_reference_copy_path_uses_short_stable_name(self) -> None:
        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from main import _seedream_reference_copy_path
        finally:
            sys.path.remove(str(src_dir))

        long_sample_id = "water_leak_0003_" + ("sample_A1-_A1-CH-A-03-03_AI__10.72.214.212_" * 3)
        long_reference = Path(
            "water_leak_0001_sample_A1-_A1-CH-A-01-03_AI__10.72.214.210_17840825901473.png"
        )
        copy_path = _seedream_reference_copy_path(long_reference, long_sample_id, Path("seedream_references"))

        self.assertEqual(copy_path.parent, Path("seedream_references"))
        self.assertEqual(copy_path.suffix, ".png")
        self.assertLessEqual(len(copy_path.name), len("ref_") + 16 + len(".png"))
        self.assertEqual(copy_path, _seedream_reference_copy_path(long_reference, long_sample_id, Path("seedream_references")))

    def test_seedream_source_preserving_composition_keeps_outside_bbox_unchanged(self) -> None:
        import numpy as np
        from PIL import Image, ImageDraw

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from main import _compose_seedream_source_preserving_output
        finally:
            sys.path.remove(str(src_dir))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_path = root / "original.jpg"
            raw_path = root / "raw.png"
            output_path = root / "generated.png"
            original = Image.new("RGB", (120, 90), (80, 90, 100))
            ImageDraw.Draw(original).rectangle((0, 0, 119, 20), fill=(20, 30, 40))
            original.save(original_path)
            raw = Image.new("RGB", (240, 180), (160, 170, 180))
            draw = ImageDraw.Draw(raw)
            draw.rectangle((80, 80, 150, 135), fill=(210, 230, 240))
            raw.save(raw_path)

            metadata = _compose_seedream_source_preserving_output(
                original_path,
                raw_path,
                output_path,
                (40, 40, 75, 68),
            )

            original_out = Image.open(original_path).convert("RGB")
            composed = Image.open(output_path).convert("RGB")
            original_array = np.asarray(original_out)
            composed_array = np.asarray(composed)
            outside_mask = np.ones(original_array.shape[:2], dtype=bool)
            outside_mask[40:68, 40:75] = False
            self.assertTrue(np.array_equal(original_array[outside_mask], composed_array[outside_mask]))
            self.assertEqual(metadata["seedream_composition_mode"], "source_preserving_water_mask_blend")
            self.assertEqual(metadata["seedream_composition_bbox"], [40, 40, 75, 68])

    def test_seedream_water_mask_returns_tight_bbox_and_rejects_patch_like_region(self) -> None:
        from PIL import Image, ImageDraw

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from main import _prepare_seedream_raw_candidate_output, _validate_seedream_experiment_output
        finally:
            sys.path.remove(str(src_dir))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_path = root / "original.jpg"
            raw_path = root / "raw.png"
            output_path = root / "generated.png"
            water_mask_path = root / "water_mask.png"
            raw_diff_mask_path = root / "raw_diff_mask.png"

            original = Image.new("RGB", (240, 180), (130, 132, 128))
            original.save(original_path)
            raw = original.copy()
            draw = ImageDraw.Draw(raw)
            draw.point((8, 8), fill=(90, 95, 98))
            draw.rectangle((80, 70, 200, 150), fill=(138, 140, 136))
            draw.ellipse((105, 100, 160, 128), fill=(70, 78, 80))
            raw.save(raw_path)

            metadata = _prepare_seedream_raw_candidate_output(
                original_path,
                raw_path,
                output_path,
                (70, 60, 210, 160),
                water_mask_output_path=water_mask_path,
                raw_diff_mask_output_path=raw_diff_mask_path,
                anomaly_type="water_leak",
            )

            self.assertTrue(water_mask_path.exists())
            self.assertTrue(raw_diff_mask_path.exists())
            self.assertEqual(metadata["seedream_composition_mode"], "raw_candidate_background_gated")
            self.assertEqual(metadata["seedream_final_image_source"], "seedream_raw_output")
            self.assertEqual(Image.open(output_path).convert("RGB").getpixel((8, 8)), (90, 95, 98))
            self.assertLess(metadata["seedream_bbox_red_box_iou"], 0.35)
            self.assertLess(metadata["seedream_mask_coverage_ratio"], 0.20)
            water_bbox = metadata["seedream_water_mask_bbox"]
            self.assertLessEqual(abs(water_bbox[0] - 105), 2)
            self.assertLessEqual(abs(water_bbox[1] - 100), 2)
            self.assertLessEqual(abs(water_bbox[2] - 161), 2)
            self.assertLessEqual(abs(water_bbox[3] - 129), 2)
            quality = _validate_seedream_experiment_output(
                original_path,
                output_path,
                (70, 60, 210, 160),
                "boxed_fusion",
                metadata,
            )
            self.assertTrue(quality["passes_quality"])

            patch_metadata = {
                "seedream_mask_coverage_ratio": 0.60,
                "seedream_bbox_red_box_iou": 0.90,
                "seedream_patch_like_score": 0.90,
            }
            patch_quality = _validate_seedream_experiment_output(
                original_path,
                output_path,
                (70, 60, 210, 160),
                "boxed_fusion",
                patch_metadata,
            )
            self.assertFalse(patch_quality["passes_quality"])
            self.assertIn("seedream_patch_like_region", patch_quality["quality_reason"])

    def test_seedream_boxed_single_edit_uses_only_guide_input(self) -> None:
        from PIL import Image

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from config import ModelServiceConfig
            from wan_image_client import WanImageClient
        finally:
            sys.path.remove(str(src_dir))

        class FakeResponse:
            ok = True
            status_code = 200

            def json(self):
                return {"data": [{"url": "https://example.invalid/generated.png"}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guide_path = root / "guide.jpg"
            reference_path = root / "water_ref.png"
            request_log_path = root / "request.json"
            Image.new("RGB", (100, 80), (70, 80, 90)).save(guide_path)
            Image.new("RGB", (40, 30), (160, 180, 190)).save(reference_path)

            def fake_download(_value, target):
                Image.new("RGB", (100, 80), (90, 100, 110)).save(target)

            client = WanImageClient(
                "doubao-seedream-5-0-pro-260628",
                ModelServiceConfig(
                    api_key="test-key",
                    provider="volcengine_ark",
                    endpoint="https://ark.cn-beijing.volces.com/api/plan/v3/images/generations",
                    api_key_env="ARK_API_KEY",
                    endpoint_env="SEEDREAM_BASE_URL",
                ),
            )

            with (
                patch.dict("os.environ", {"SEEDREAM_SIZE": "2k"}, clear=False),
                patch("wan_image_client._generate_seedream_with_openai_sdk", return_value={"data": [{"url": "https://example.invalid/generated.png"}]}),
                patch("wan_image_client._download_or_decode_image", side_effect=fake_download),
            ):
                client.edit_image_with_wan(
                    image_path=str(guide_path),
                    prompt="add a water stain in the red box",
                    negative_prompt="",
                    bbox=(20, 10, 60, 50),
                    output_path=str(root / "generated.png"),
                    anomaly_type="water_leak",
                    request_log_path=str(request_log_path),
                    seedream_mode="boxed_single_edit",
                    seedream_reference_paths=[str(reference_path)],
                )

            body = read_json(request_log_path)["body"]
            self.assertEqual(body["seedream_mode"], "boxed_single_edit")
            self.assertEqual(body["n"], 1)
            self.assertEqual(body["model"], "doubao-seedream-5-0-pro-260628")
            self.assertEqual(body["image"], str(guide_path))
            self.assertIn("80-160 px", body["prompt"])
            self.assertIn("Do not fill the whole rectangle", body["prompt"])
            self.assertIn("[20, 10, 60, 50]", body["prompt"])
            self.assertNotIn("images", body)

    def test_seedream_red_box_is_bounded_and_deterministic(self) -> None:
        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from main import _choose_seedream_red_box
        finally:
            sys.path.remove(str(src_dir))

        grid_bbox = (100, 200, 500, 600)
        box_a = _choose_seedream_red_box("sample_a", grid_bbox, 200, 200)
        box_b = _choose_seedream_red_box("sample_a", grid_bbox, 200, 200)
        self.assertEqual(box_a, box_b)
        self.assertGreaterEqual(box_a[0], grid_bbox[0])
        self.assertGreaterEqual(box_a[1], grid_bbox[1])
        self.assertLessEqual(box_a[2], grid_bbox[2])
        self.assertLessEqual(box_a[3], grid_bbox[3])
        self.assertEqual(box_a[2] - box_a[0], 200)
        self.assertEqual(box_a[3] - box_a[1], 200)
        water_box = _choose_seedream_red_box("sample_a", grid_bbox, 200, 200, "water_leak")
        self.assertGreaterEqual(water_box[1], grid_bbox[1] + int((grid_bbox[3] - grid_bbox[1]) * 0.45))

    def test_seedream_water_leak_box_prefers_low_saturation_floor_area(self) -> None:
        from PIL import Image, ImageDraw

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from main import _choose_seedream_water_leak_box
        finally:
            sys.path.remove(str(src_dir))

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "source.jpg"
            image = Image.new("RGB", (640, 640), (210, 40, 30))
            draw = ImageDraw.Draw(image)
            draw.rectangle((100, 400, 300, 620), fill=(150, 150, 145))
            image.save(image_path)

            box = _choose_seedream_water_leak_box(image_path, "sample_a", (100, 200, 500, 600), 200, 200)

        self.assertLessEqual(box[0], 140)
        self.assertGreaterEqual(box[1], 380)
        self.assertEqual(box[2] - box[0], 200)
        self.assertEqual(box[3] - box[1], 200)

    def test_seedream_water_leak_box_rejects_grid_without_floor(self) -> None:
        from PIL import Image, ImageDraw

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from main import _choose_seedream_water_leak_box
        finally:
            sys.path.remove(str(src_dir))

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "cabinet.jpg"
            image = Image.new("RGB", (640, 640), (230, 230, 225))
            draw = ImageDraw.Draw(image)
            for x in range(100, 500, 40):
                draw.line((x, 200, x, 600), fill=(40, 40, 40), width=3)
            for y in range(220, 600, 36):
                draw.line((100, y, 500, y), fill=(40, 40, 40), width=2)
            for x in range(120, 500, 80):
                draw.rectangle((x, 250, x + 45, 285), fill=(30, 160, 45))
                draw.rectangle((x, 330, x + 55, 390), fill=(245, 200, 20))
            image.save(image_path)

            with self.assertRaisesRegex(ValueError, "no_visible_floor_region"):
                _choose_seedream_water_leak_box(image_path, "sample_panel", (100, 200, 500, 600), 200, 200)

    def test_qwen_grid_preview_is_downscaled_for_vlm_payload(self) -> None:
        from PIL import Image

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from grid import make_grid_preview
        finally:
            sys.path.remove(str(src_dir))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "large.jpg"
            preview_path = root / "preview.jpg"
            Image.new("RGB", (2560, 1440), (120, 130, 140)).save(source_path)

            with patch.dict("os.environ", {"QWEN_GRID_PREVIEW_MAX_SIDE": "640", "QWEN_GRID_PREVIEW_JPEG_QUALITY": "70"}):
                make_grid_preview(str(source_path), str(preview_path))

            with Image.open(preview_path) as preview:
                self.assertLessEqual(max(preview.size), 640)
            self.assertLess(preview_path.stat().st_size, source_path.stat().st_size)

    def test_qwen_vlm_payload_records_and_limits_image_bytes(self) -> None:
        from PIL import Image

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from config import ModelServiceConfig
            from qwen_vlm_client import QwenVLMClient
        finally:
            sys.path.remove(str(src_dir))

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "preview.jpg"
            Image.new("RGB", (100, 80), (90, 100, 110)).save(image_path, quality=90)
            client = QwenVLMClient(
                "qwen3.6-27b",
                ModelServiceConfig(
                    api_key="qwen-test",
                    provider="openai_compatible",
                    endpoint="https://deepseek.gds-services.com/v1",
                    api_key_env="QWEN397B_API_KEY",
                    endpoint_env="QWEN397B_API_URL",
                ),
            )

            with patch.dict("os.environ", {"QWEN397B_IMAGE_MAX_BYTES": "1000000"}):
                _endpoints, _body, log_body = client._build_request_payloads(str(image_path), "select grid")
            self.assertEqual(log_body["image_bytes"], image_path.stat().st_size)
            self.assertEqual(log_body["image_max_bytes"], 1000000)

            with patch.dict("os.environ", {"QWEN397B_IMAGE_MAX_BYTES": "1"}):
                with self.assertRaisesRegex(RuntimeError, "request image too large"):
                    client._build_request_payloads(str(image_path), "select grid")

    def test_seedream_experiment_quality_rejects_red_residual_and_scene_drift(self) -> None:
        from PIL import Image, ImageDraw

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from main import _validate_seedream_experiment_output
        finally:
            sys.path.remove(str(src_dir))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_path = root / "original.jpg"
            red_path = root / "red.jpg"
            drift_path = root / "drift.jpg"
            tone_shift_path = root / "tone_shift.jpg"
            original = Image.new("RGB", (100, 80), (70, 80, 90))
            original_draw = ImageDraw.Draw(original)
            original_draw.rectangle((5, 5, 18, 70), fill=(20, 20, 20))
            original_draw.rectangle((70, 10, 92, 70), fill=(130, 130, 130))
            original.save(original_path)
            red_image = Image.new("RGB", (100, 80), (70, 80, 90))
            ImageDraw.Draw(red_image).rectangle((20, 10, 60, 50), fill=(255, 0, 0))
            red_image.save(red_path)
            tone_shift = Image.new("RGB", (100, 80), (125, 135, 145))
            tone_draw = ImageDraw.Draw(tone_shift)
            tone_draw.rectangle((5, 5, 18, 70), fill=(75, 75, 75))
            tone_draw.rectangle((70, 10, 92, 70), fill=(185, 185, 185))
            tone_shift.save(tone_shift_path)
            drift = Image.new("RGB", (100, 80), (220, 220, 220))
            drift_draw = ImageDraw.Draw(drift)
            drift_draw.rectangle((35, 5, 48, 70), fill=(20, 20, 20))
            drift_draw.rectangle((8, 10, 30, 70), fill=(130, 130, 130))
            drift.save(drift_path)

            red_quality = _validate_seedream_experiment_output(original_path, red_path, (20, 10, 60, 50), "boxed_fusion")
            tone_quality = _validate_seedream_experiment_output(original_path, tone_shift_path, (20, 10, 60, 50), "single_image_edit")
            drift_quality = _validate_seedream_experiment_output(original_path, drift_path, (20, 10, 60, 50), "single_image_edit")

            self.assertFalse(red_quality["passes_quality"])
            self.assertIn("seedream_red_box_residual", red_quality["quality_reason"])
            self.assertGreater(tone_quality["outside_change_ratio"], 0.20)
            self.assertLess(tone_quality["outside_structure_change_ratio"], 0.25)
            self.assertTrue(tone_quality["passes_quality"])
            self.assertFalse(drift_quality["passes_quality"])
            self.assertIn("seedream_outside_region_change_high", drift_quality["quality_reason"])

    def test_i2i_failure_artifacts_copy_failed_generated_image(self) -> None:
        from PIL import Image

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from main import _failure_artifact_context
            from utils import ensure_output_dirs, write_json as i2i_write_json
        finally:
            sys.path.remove(str(src_dir))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = ensure_output_dirs(root)
            sample_id = "sample_failed"
            generated_path = dirs["generated_images"] / f"{sample_id}.png"
            request_log_path = dirs["requests"] / f"{sample_id}_wan_request.json"
            response_log_path = dirs["responses"] / f"{sample_id}_wan_response.json"
            mask_path = dirs["masks"] / f"{sample_id}_obj_000001_mask.png"
            crop_path = dirs["crops"] / f"{sample_id}_obj_000001_crop.jpg"
            Image.new("RGB", (32, 24), (90, 100, 110)).save(generated_path)
            i2i_write_json(request_log_path, {"request": True})
            i2i_write_json(response_log_path, {"response": True})

            context = _failure_artifact_context(
                sample_id,
                dirs,
                generated_path,
                request_log_path,
                response_log_path,
                mask_path,
                crop_path,
            )

            artifacts = context["failure_artifacts"]
            failed_image = dirs["failed_generated_images"] / generated_path.name
            self.assertTrue(failed_image.exists())
            self.assertIn("debug", artifacts["failed_generated_image_uri"])
            self.assertIn("generated_images", artifacts["generated_image_uri"])
            self.assertIn("requests", artifacts["request_log_uri"])
            self.assertIn("responses", artifacts["response_log_uri"])

    def test_seedream_debug_reference_generation_rejects_framed_scene_crop(self) -> None:
        from PIL import Image, ImageDraw

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from config import ModelServiceConfig
            from wan_image_client import WanImageClient
        finally:
            sys.path.remove(str(src_dir))

        class FakeResponse:
            ok = True
            status_code = 200

            def json(self):
                return {"data": [{"url": "https://example.invalid/generated.png"}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_path = root / "original.jpg"
            Image.new("RGB", (120, 90), (70, 80, 90)).save(original_path)

            def fake_download(_value, target):
                image = Image.new("RGB", (320, 240), (160, 170, 180))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 319, 239), outline=(0, 0, 0), width=20)
                image.save(target)

            client = WanImageClient(
                "doubao-seedream-5-0-pro-260628",
                ModelServiceConfig(
                    api_key="test-key",
                    provider="volcengine_ark",
                    endpoint="https://ark.cn-beijing.volces.com/api/plan/v3/images/generations",
                    api_key_env="ARK_API_KEY",
                    endpoint_env="SEEDREAM_BASE_URL",
                ),
            )

            with (
                patch.dict("os.environ", {"SEEDREAM_ALLOW_REFERENCE_GENERATION_DEBUG": "1"}),
                patch("wan_image_client._generate_seedream_with_openai_sdk", return_value={"data": [{"url": "https://example.invalid/generated.png"}]}),
                patch("wan_image_client._download_or_decode_image", side_effect=fake_download),
            ):
                with self.assertRaisesRegex(RuntimeError, "framed full-scene"):
                    client.edit_image_with_wan(
                        image_path=str(original_path),
                        prompt="make a small water leak",
                        negative_prompt="",
                        bbox=(20, 10, 80, 60),
                        output_path=str(root / "generated.png"),
                        anomaly_type="water_leak",
                    )

    def test_seedream_default_size_scales_crop_aspect_to_min_pixels(self) -> None:
        from PIL import Image

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from wan_image_client import _seedream_size_for_image
        finally:
            sys.path.remove(str(src_dir))

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "crop.jpg"
            Image.new("RGB", (672, 378), (70, 80, 90)).save(image_path)
            with patch.dict("os.environ", {"SEEDREAM_SIZE": "auto"}):
                self.assertEqual(_seedream_size_for_image(image_path), "2560x1440")

    def test_seedream_client_allows_explicit_valid_size_override(self) -> None:
        from PIL import Image

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from config import ModelServiceConfig
            from wan_image_client import WanImageClient
        finally:
            sys.path.remove(str(src_dir))

        class FakeResponse:
            ok = True
            status_code = 200

            def json(self):
                return {"data": [{"url": "https://example.invalid/generated.png"}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_path = root / "original.jpg"
            request_log_path = root / "request.json"
            Image.new("RGB", (100, 80), (70, 80, 90)).save(original_path)
            client = WanImageClient(
                "doubao-seedream-5-0-pro-260628",
                ModelServiceConfig(
                    api_key="test-key",
                    provider="volcengine_ark",
                    endpoint="https://ark.cn-beijing.volces.com/api/plan/v3/images/generations",
                    api_key_env="ARK_API_KEY",
                    endpoint_env="SEEDREAM_BASE_URL",
                ),
            )

            def fake_download(_value, target):
                Image.new("RGB", (256, 256), (220, 235, 245)).save(target)

            with (
                patch.dict("os.environ", {"SEEDREAM_SIZE": "2k", "SEEDREAM_ALLOW_REFERENCE_GENERATION_DEBUG": "1"}),
                patch("wan_image_client._generate_seedream_with_openai_sdk", return_value={"data": [{"url": "https://example.invalid/generated.png"}]}),
                patch("wan_image_client._download_or_decode_image", side_effect=fake_download),
            ):
                client.edit_image_with_wan(
                    image_path=str(original_path),
                    prompt="make a small water leak",
                    negative_prompt="",
                    bbox=(20, 10, 60, 50),
                    output_path=str(root / "generated.png"),
                    anomaly_type="water_leak",
                    request_log_path=str(request_log_path),
                )

            body = read_json(request_log_path)["body"]
            self.assertEqual(body["size"], "2k")

    def test_seedream_client_rejects_invalid_size_before_request(self) -> None:
        from PIL import Image

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from config import ModelServiceConfig
            from wan_image_client import WanImageClient
        finally:
            sys.path.remove(str(src_dir))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_path = root / "original.jpg"
            Image.new("RGB", (100, 80), (70, 80, 90)).save(original_path)
            client = WanImageClient(
                "doubao-seedream-5-0-pro-260628",
                ModelServiceConfig(
                    api_key="test-key",
                    provider="volcengine_ark",
                    endpoint="https://ark.cn-beijing.volces.com/api/plan/v3/images/generations",
                    api_key_env="ARK_API_KEY",
                    endpoint_env="SEEDREAM_BASE_URL",
                ),
            )

            with patch.dict("os.environ", {"SEEDREAM_SIZE": "bad-size", "SEEDREAM_ALLOW_REFERENCE_GENERATION_DEBUG": "1"}):
                with self.assertRaisesRegex(ValueError, "SEEDREAM_SIZE"):
                    client.edit_image_with_wan(
                        image_path=str(original_path),
                        prompt="make a small water leak",
                        negative_prompt="",
                        bbox=(20, 10, 60, 50),
                        output_path=str(root / "generated.png"),
                        anomaly_type="water_leak",
                    )

    def test_generation_preflight_rejects_seedream_reference_generation_endpoint(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            image_root.mkdir()
            image_path = image_root / "source.jpg"
            Image.new("RGB", (100, 80), (70, 80, 90)).save(image_path)
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_water",
                        "image_id": "source",
                        "image_uri": image_path.name,
                        "anomaly_type": "water_leak",
                        "source_type": "manual_upload",
                        "task_mode": "generation",
                    }
                ],
                ["sample_id", "image_id", "image_uri", "anomaly_type", "source_type", "task_mode"],
            )

            with patch.dict("os.environ", {"ARK_API_KEY": "ark-test", "QWEN397B_API_KEY": "qwen-test"}, clear=False):
                config = load_config(ROOT / "configs" / "autolabel.yaml")
                config["models"]["generation"]["image_generators"]["seedream5_image_editor"][
                    "endpoint"
                ] = "https://ark.cn-beijing.volces.com/api/plan/v3/images/generations"
                with self.assertRaisesRegex(Exception, "reference-generation endpoint"):
                    run_generation_preflight(
                        config,
                        tasks_csv=manifest_path,
                        image_root=image_root,
                        output_root=root / "out",
                    )

    def test_generation_preflight_rejects_boxed_fusion_without_reference_images(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            image_root.mkdir()
            image_path = image_root / "source.jpg"
            Image.new("RGB", (100, 80), (70, 80, 90)).save(image_path)
            empty_reference_dir = root / "refs"
            empty_reference_dir.mkdir()
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_water",
                        "image_id": "source",
                        "image_uri": image_path.name,
                        "anomaly_type": "water_leak",
                        "source_type": "manual_upload",
                        "task_mode": "generation",
                    }
                ],
                ["sample_id", "image_id", "image_uri", "anomaly_type", "source_type", "task_mode"],
            )

            config = load_config(ROOT / "configs" / "autolabel.yaml")
            config["generation"]["seedream"] = {
                "mode": "boxed_fusion",
                "allow_experimental_generation": True,
                "water_reference_dir": str(empty_reference_dir),
            }
            with self.assertRaisesRegex(Exception, "no image files"):
                run_generation_preflight(
                    config,
                    tasks_csv=manifest_path,
                    image_root=image_root,
                    output_root=root / "out",
                    require_credentials=False,
                )

    def test_i2i_dry_run_summary_separates_final_images_from_debug_artifacts(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            output_root = root / "out"
            image_root.mkdir()
            image_path = image_root / "source.jpg"
            Image.new("RGB", (160, 120), (80, 80, 80)).save(image_path)
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_dry_run",
                        "image_id": "source",
                        "image_uri": image_path.name,
                        "anomaly_type": "water_leak",
                        "source_type": "manual_upload",
                        "task_mode": "generation",
                    }
                ],
                ["sample_id", "image_id", "image_uri", "anomaly_type", "source_type", "task_mode"],
            )

            cmd = [
                sys.executable,
                str(ROOT / "external" / "I2I" / "src" / "main.py"),
                "--tasks",
                str(manifest_path),
                "--image-root",
                str(image_root),
                "--output-root",
                str(output_root),
                "--dry-run",
                "--limit",
                "1",
            ]
            completed = subprocess.run(cmd, cwd=ROOT / "external" / "I2I", text=True, capture_output=True, check=False)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = read_json(output_root / "logs" / "run_summary.json")
            self.assertEqual(summary["final_generated_count"], 1)
            self.assertEqual(summary["model_call_count"], 0)
            self.assertEqual(summary["model_generated_count"], 0)
            self.assertGreaterEqual(summary["debug_artifact_count"], 3)
            self.assertEqual(len(list((output_root / "generated_images").glob("*.png"))), 1)
            self.assertTrue((output_root / "debug" / "grid_previews").exists())

    def test_i2i_seedream_boxed_fusion_dry_run_outputs_one_final_image(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            reference_root = root / "refs"
            output_root = root / "out"
            image_root.mkdir()
            reference_root.mkdir()
            image_path = image_root / "source.jpg"
            reference_path = reference_root / "water_ref.png"
            Image.new("RGB", (160, 120), (80, 80, 80)).save(image_path)
            reference = Image.new("RGBA", (64, 64), (160, 180, 190, 0))
            for x in range(14, 50):
                for y in range(22, 42):
                    reference.putpixel((x, y), (45, 55, 58, 210))
            reference.save(reference_path)
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_seedream_dry_run",
                        "image_id": "source",
                        "image_uri": image_path.name,
                        "anomaly_type": "water_leak",
                        "source_type": "manual_upload",
                        "task_mode": "generation",
                    }
                ],
                ["sample_id", "image_id", "image_uri", "anomaly_type", "source_type", "task_mode"],
            )

            cmd = [
                sys.executable,
                str(ROOT / "external" / "I2I" / "src" / "main.py"),
                "--tasks",
                str(manifest_path),
                "--image-root",
                str(image_root),
                "--output-root",
                str(output_root),
                "--dry-run",
                "--seedream-mode",
                "boxed_fusion",
                "--water-reference-dir",
                str(reference_root),
                "--limit",
                "1",
            ]
            completed = subprocess.run(cmd, cwd=ROOT / "external" / "I2I", text=True, capture_output=True, check=False)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = read_json(output_root / "logs" / "run_summary.json")
            self.assertEqual(summary["final_generated_count"], 1)
            self.assertEqual(summary["model_call_count"], 0)
            self.assertEqual(summary["model_generated_count"], 0)
            self.assertEqual(len(list((output_root / "generated_images").glob("*.png"))), 1)
            self.assertEqual(len(list((output_root / "debug" / "seedream_guides").glob("*.jpg"))), 1)
            self.assertEqual(len(list((output_root / "debug" / "seedream_references").glob("*"))), 1)
            self.assertTrue((output_root / "debug" / "crops").exists())
            self.assertTrue((output_root / "debug" / "masks").exists())

    def test_i2i_process_task_generates_once_for_selected_grid_only(self) -> None:
        from PIL import Image, ImageDraw

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from config import PipelineConfig
            from main import process_task
            from utils import ensure_output_dirs
        finally:
            sys.path.remove(str(src_dir))

        class FakeVLM:
            def select_grid_with_qwen(self, *_args, **_kwargs):
                return {
                    "selected_grid": "C2",
                    "confidence": 0.98,
                    "top_candidates": [
                        {"grid": "C2", "score": 0.98},
                        {"grid": "D2", "score": 0.90},
                        {"grid": "B2", "score": 0.85},
                    ],
                }

        class FakeWan:
            def __init__(self):
                self.calls = []

            def edit_image_with_wan(self, **kwargs):
                self.calls.append(kwargs)
                with Image.open(kwargs["image_path"]) as image:
                    generated = image.convert("RGB")
                x1, y1, x2, y2 = kwargs["bbox"]
                draw = ImageDraw.Draw(generated)
                draw.ellipse((x1 + 15, y1 + 12, x1 + 38, y1 + 28), fill=(225, 235, 240))
                Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
                generated.save(kwargs["output_path"])
                return {"fake": True}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            output_root = root / "out"
            image_root.mkdir()
            original_path = image_root / "source.jpg"
            Image.new("RGB", (160, 120), (80, 80, 80)).save(original_path)
            dirs = ensure_output_dirs(output_root)
            cfg = PipelineConfig(
                tasks=str(root / "tasks.csv"),
                image_root=str(image_root),
                output_root=str(output_root),
                dry_run=False,
                max_retries=0,
            )
            task = {
                "sample_id": "sample_once",
                "image_id": "source",
                "image_uri": str(original_path),
                "anomaly_type": "water_leak",
                "source_type": "manual_upload",
            }
            fake_wan = FakeWan()

            self.assertTrue(process_task(task, cfg, dirs, FakeVLM(), fake_wan))
            self.assertEqual(len(fake_wan.calls), 1)
            written = read_json(dirs["metadata"] / "sample_once.json")
            params = written["objects"][0]["geometry_detail"]["generation_params"]
            self.assertEqual(params["selected_grid"], "C2")
            self.assertEqual(params["candidate_grids"], ["C2", "D2", "B2"])

    def test_i2i_seedream_water_leak_skips_without_visible_floor(self) -> None:
        from PIL import Image, ImageDraw

        src_dir = ROOT / "external" / "I2I" / "src"
        sys.path.insert(0, str(src_dir))
        try:
            from config import PipelineConfig
            from main import process_task
            from utils import ensure_output_dirs
        finally:
            sys.path.remove(str(src_dir))

        class FakeVLM:
            def select_grid_with_qwen(self, *_args, **_kwargs):
                return {
                    "selected_grid": "C2",
                    "confidence": 0.98,
                    "top_candidates": [
                        {"grid": "C2", "score": 0.98},
                        {"grid": "D2", "score": 0.90},
                        {"grid": "B2", "score": 0.85},
                    ],
                }

        class FakeWan:
            model_call_count = 0
            model_generated_count = 0

            def edit_image_with_wan(self, **_kwargs):
                raise AssertionError("Seedream should not be called when no floor region is available")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            output_root = root / "out"
            image_root.mkdir()
            original_path = image_root / "source.jpg"
            image = Image.new("RGB", (160, 120), (235, 235, 230))
            draw = ImageDraw.Draw(image)
            for x in range(0, 160, 12):
                draw.line((x, 0, x, 120), fill=(30, 30, 30), width=1)
            for y in range(0, 120, 10):
                draw.line((0, y, 160, y), fill=(30, 30, 30), width=1)
            draw.rectangle((40, 55, 95, 95), fill=(245, 190, 20))
            image.save(original_path)
            dirs = ensure_output_dirs(output_root)
            cfg = PipelineConfig(
                tasks=str(root / "tasks.csv"),
                image_root=str(image_root),
                output_root=str(output_root),
                dry_run=False,
                max_retries=0,
                seedream_mode="boxed_single_edit",
                red_box_min_size=30,
                red_box_max_size=30,
            )
            task = {
                "sample_id": "sample_no_floor",
                "image_id": "source",
                "image_uri": str(original_path),
                "anomaly_type": "water_leak",
                "source_type": "manual_upload",
            }

            self.assertIsNone(process_task(task, cfg, dirs, FakeVLM(), FakeWan()))
            skip_log = read_json(dirs["logs"] / "sample_no_floor_skip.json")
            self.assertEqual(skip_log["status"], "skipped")
            self.assertEqual(skip_log["reason"], "no_visible_floor_region")
            self.assertFalse((dirs["metadata"] / "sample_no_floor.json").exists())

    def test_generation_runtime_uses_anomaly_type_localizer_policy(self) -> None:
        config = load_config(ROOT / "configs" / "autolabel.yaml")
        config["modules"]["generation"]["localizer_policy"] = {
            "oil_leak": {
                "primary": "rgb_diff",
                "fallback": "none",
            }
        }

        runtime = resolve_generation_runtime(config, anomaly_type="oil_leak")
        self.assertEqual(runtime["extra_cli_args"], [])
        self.assertIn("--localizer", runtime["localizer_cli_args"])
        localizer_index = runtime["localizer_cli_args"].index("--localizer")
        self.assertEqual(runtime["localizer_cli_args"][localizer_index + 1], "rgb_diff")
        self.assertIn("--localizer-fallback", runtime["localizer_cli_args"])
        fallback_index = runtime["localizer_cli_args"].index("--localizer-fallback")
        self.assertEqual(runtime["localizer_cli_args"][fallback_index + 1], "none")

    def test_generation_module_does_not_pass_localizer_cli_args_by_default(self) -> None:
        from autolabel.modules.generation.i2i_external import ExternalI2IGenerationModule

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_water",
                        "image_id": "image_water",
                        "image_uri": str(root / "water.jpg"),
                        "anomaly_type": "water_leak",
                        "source_type": "manual_upload",
                        "task_mode": "generation",
                    },
                    {
                        "sample_id": "sample_oil",
                        "image_id": "image_oil",
                        "image_uri": str(root / "oil.jpg"),
                        "anomaly_type": "oil_leak",
                        "source_type": "manual_upload",
                        "task_mode": "generation",
                    },
                ],
                ["sample_id", "image_id", "image_uri", "anomaly_type", "source_type", "task_mode"],
            )
            config = load_config(ROOT / "configs" / "autolabel.yaml")
            config["modules"]["generation"]["localizer_policy"] = {
                "oil_leak": {
                    "primary": "rgb_diff",
                    "fallback": "none",
                }
            }

            calls = []

            class FakeCompleted:
                returncode = 0
                stdout = "ok\n"
                stderr = ""

            def fake_run(self, **kwargs):
                calls.append(kwargs)
                return FakeCompleted()

            with patch("autolabel.modules.generation.i2i_external.I2IGenerator.run", new=fake_run):
                result = ExternalI2IGenerationModule(config, {}).run(
                    tasks_csv=manifest_path,
                    image_root=root,
                    output_root=root / "out",
                )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["extra_cli_args"], [])

    def test_generation_module_passes_seedream_model_to_i2i(self) -> None:
        from autolabel.modules.generation.i2i_external import ExternalI2IGenerationModule

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_water",
                        "image_id": "image_water",
                        "image_uri": str(root / "water.jpg"),
                        "anomaly_type": "water_leak",
                        "source_type": "manual_upload",
                        "task_mode": "generation",
                    }
                ],
                ["sample_id", "image_id", "image_uri", "anomaly_type", "source_type", "task_mode"],
            )
            config = load_config(ROOT / "configs" / "autolabel.yaml")
            config["generation"]["image_model_key"] = "seedream5_image_editor"
            calls = []

            class FakeCompleted:
                returncode = 0
                stdout = "ok\n"
                stderr = ""

            def fake_run(self, **kwargs):
                calls.append(kwargs)
                return FakeCompleted()

            with patch("autolabel.modules.generation.i2i_external.I2IGenerator.run", new=fake_run):
                result = ExternalI2IGenerationModule(config, {}).run(
                    tasks_csv=manifest_path,
                    image_root=root,
                    output_root=root / "out",
                )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(calls[0]["image_model"], "doubao-seedream-5-0-pro-260628")

    def test_generation_module_passes_vlm_model_to_i2i(self) -> None:
        from autolabel.modules.generation.i2i_external import ExternalI2IGenerationModule

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_water",
                        "image_id": "image_water",
                        "image_uri": str(root / "water.jpg"),
                        "anomaly_type": "water_leak",
                        "source_type": "manual_upload",
                        "task_mode": "generation",
                    }
                ],
                ["sample_id", "image_id", "image_uri", "anomaly_type", "source_type", "task_mode"],
            )
            config = load_config(ROOT / "configs" / "autolabel.yaml")
            config["models"]["generation"]["vlm"]["private_grid_selector"] = {
                "provider": "custom",
                "model_name": "private-vlm-grid",
                "credential_ref": "local_or_private_model",
            }
            config["generation"]["vlm_model_key"] = "private_grid_selector"
            calls = []

            class FakeCompleted:
                returncode = 0
                stdout = "ok\n"
                stderr = ""

            def fake_run(self, **kwargs):
                calls.append(kwargs)
                return FakeCompleted()

            with patch("autolabel.modules.generation.i2i_external.I2IGenerator.run", new=fake_run):
                result = ExternalI2IGenerationModule(config, {}).run(
                    tasks_csv=manifest_path,
                    image_root=root,
                    output_root=root / "out",
                )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(calls[0]["vlm_model"], "private-vlm-grid")

    def test_generation_module_normalizes_water_leak_for_external_backend_alias(self) -> None:
        from autolabel.modules.generation.i2i_external import ExternalI2IGenerationModule

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_water",
                        "image_id": "image_water",
                        "image_uri": str(root / "water.jpg"),
                        "anomaly_type": "water_leak",
                        "source_type": "manual_upload",
                        "task_mode": "generation",
                    }
                ],
                ["sample_id", "image_id", "image_uri", "anomaly_type", "source_type", "task_mode"],
            )
            config = load_config(ROOT / "configs" / "autolabel.yaml")
            calls = []

            class FakeCompleted:
                returncode = 0
                stdout = "ok\n"
                stderr = ""

            def fake_run(self, **kwargs):
                calls.append(kwargs)
                return FakeCompleted()

            with patch("autolabel.modules.generation.i2i_external.I2IGenerator.run", new=fake_run):
                result = ExternalI2IGenerationModule(
                    config,
                    {"anomaly_type_aliases": {"water_leak": "coolant_leak"}},
                ).run(
                    tasks_csv=manifest_path,
                    image_root=root,
                    output_root=root / "out",
                )

            self.assertEqual(result.returncode, 0)
            self.assertIn("water_leak->coolant_leak", result.stdout)
            prepared_rows = read_csv(calls[0]["tasks_csv"])
            self.assertEqual(prepared_rows[0]["anomaly_type"], "coolant_leak")

    def test_generation_module_can_opt_in_to_localizer_cli_passthrough(self) -> None:
        from autolabel.modules.generation.i2i_external import ExternalI2IGenerationModule

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_water",
                        "image_id": "image_water",
                        "image_uri": str(root / "water.jpg"),
                        "anomaly_type": "water_leak",
                        "source_type": "manual_upload",
                        "task_mode": "generation",
                    },
                    {
                        "sample_id": "sample_oil",
                        "image_id": "image_oil",
                        "image_uri": str(root / "oil.jpg"),
                        "anomaly_type": "oil_leak",
                        "source_type": "manual_upload",
                        "task_mode": "generation",
                    },
                ],
                ["sample_id", "image_id", "image_uri", "anomaly_type", "source_type", "task_mode"],
            )
            config = load_config(ROOT / "configs" / "autolabel.yaml")
            config["modules"]["generation"]["localizer_policy"] = {
                "oil_leak": {
                    "primary": "rgb_diff",
                    "fallback": "none",
                }
            }

            calls = []

            class FakeCompleted:
                returncode = 0
                stdout = "ok\n"
                stderr = ""

            def fake_run(self, **kwargs):
                calls.append(kwargs)
                return FakeCompleted()

            with patch("autolabel.modules.generation.i2i_external.I2IGenerator.run", new=fake_run):
                result = ExternalI2IGenerationModule(config, {"pass_localizer_cli_args": True}).run(
                    tasks_csv=manifest_path,
                    image_root=root,
                    output_root=root / "out",
                )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(len(calls), 2)
            first_args = calls[0]["extra_cli_args"]
            second_args = calls[1]["extra_cli_args"]
            self.assertNotEqual(first_args, second_args)
            self.assertTrue(any(arg == "rgb_diff" for arg in first_args + second_args))

    def test_generation_registry_supports_vlm_wan_autolabel_backend(self) -> None:
        from autolabel.modules.generation import build_generation_module
        from autolabel.modules.generation.internal_vlm_wan import InternalVLMWanGenerationModule

        config = load_config(ROOT / "configs" / "autolabel.yaml")
        config["modules"]["generation"]["backend"] = "vlm_wan_autolabel"
        config["modules"]["generation"]["backends"]["vlm_wan_autolabel"] = {}
        module = build_generation_module(config)
        self.assertIsInstance(module, InternalVLMWanGenerationModule)

    def test_ingest_generated_metadata_applies_localizer_and_writes_benchmark(self) -> None:
        from PIL import Image, ImageDraw, ImageFilter

        from autolabel.pipeline import ingest_generated_metadata

        def make_base_image(size=(256, 192)):
            image = Image.new("RGB", size, (132, 132, 132))
            draw = ImageDraw.Draw(image)
            for y in range(0, size[1], 16):
                color = 126 if (y // 16) % 2 == 0 else 138
                draw.line([(0, y), (size[0], y)], fill=(color, color, color), width=1)
            return image

        def add_water_patch(img, bbox):
            out = img.convert("RGBA")
            overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.ellipse(bbox, fill=(190, 205, 210, 120))
            overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
            return Image.alpha_composite(out, overlay).convert("RGB")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            i2i_output_root = root / "i2i_outputs"
            metadata_input_dir = i2i_output_root / "metadata"
            metadata_input_dir.mkdir(parents=True)
            processed_root = root / "processed"
            metadata_dir = processed_root / "metadata"
            original_path = root / "original.jpg"
            generated_path = root / "generated.jpg"
            base = make_base_image()
            generated = add_water_patch(base, (98, 96, 152, 132))
            base.save(original_path)
            generated.save(generated_path)

            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_generated_localizer",
                        "image_id": "image_generated_localizer",
                        "image_uri": original_path.name,
                        "anomaly_type": "water_leak",
                        "source_type": "manual_upload",
                        "task_mode": "generation",
                    }
                ],
                ["sample_id", "image_id", "image_uri", "anomaly_type", "source_type", "task_mode"],
            )

            sample = normalize_autolabel_sample(
                {
                    "sample_id": "sample_generated_localizer",
                    "image_asset": {
                        "image_id": "image_generated_localizer",
                        "image_uri": str(generated_path),
                        "width": 256,
                        "height": 192,
                        "source_type": "generated",
                    },
                    "objects": [
                        {
                            "object_id": "leak_000001",
                            "object_type": "leakage_area",
                            "box": {"format": "xyxy", "x1": 80, "y1": 80, "x2": 170, "y2": 150},
                            "geometry_source": "synthetic_generator",
                            "geometry_model": {"model_name": "wan", "model_version": "test", "confidence": 0.9},
                            "geometry_detail": {
                                "polygon": None,
                                "mask_uri": None,
                                "mask_format": None,
                                "generation_params": {
                                    "prompt_box": [80, 80, 170, 150],
                                    "output_contract": "AutoLabelSample.objects[]",
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
                    "workflow": {"workflow_status": "classified"},
                    "export": {"export_format": "labelstudio", "export_status": "not_exported"},
                },
                pipeline_id="autolabel_dag_v1",
                pipeline_version="0.1.0",
            )
            write_json(metadata_input_dir / "sample_generated_localizer.json", sample)

            config = load_config(ROOT / "configs" / "autolabel.yaml")
            config["modules"]["generation"]["localizer"]["primary"] = "rgb_diff"
            config["modules"]["generation"]["localizer"]["fallback"] = "pgcd_lpips"
            config["modules"]["generation"]["localizer"]["debug"] = True
            config["modules"]["generation"]["localizer"]["sidecar_eval"] = "pgcd_lpips"
            config["modules"]["generation"]["localizer_policy"]["water_leak"] = {
                "primary": "rgb_diff",
                "fallback": "pgcd_lpips",
            }
            config["generation"]["crop_expand_ratio"] = 0.05
            config["modules"]["generation"]["localizer"]["pgcd"]["min_component_area"] = 50

            written = ingest_generated_metadata(
                i2i_output_root,
                metadata_dir,
                pipeline_config=config,
                tasks_csv=manifest_path,
                image_root=root,
            )

            self.assertEqual(written, [metadata_dir / "sample_generated_localizer.json"])
            ingested = read_json(written[0])
            obj = ingested["objects"][0]
            generation_params = obj["geometry_detail"]["generation_params"]
            localizer_info = obj["geometry_detail"]["generation_params"]["localizer"]
            self.assertEqual(generation_params["localizer_strategy"], "rgb_diff")
            self.assertEqual(generation_params["localizer_fallback"], "pgcd_lpips")
            self.assertTrue(generation_params["localizer_debug"])
            self.assertEqual(localizer_info["strategy"], "rgb_diff")
            self.assertEqual(localizer_info["fallback"], "pgcd_lpips")
            self.assertEqual(localizer_info["used"], "rgb_diff")
            self.assertIn("metrics", localizer_info)
            self.assertIn("benchmark", localizer_info)
            self.assertIn("quality", localizer_info)
            processed_root = metadata_dir.parent
            self.assertFalse(Path(obj["geometry_detail"]["mask_uri"]).is_absolute())
            self.assertTrue((processed_root / obj["geometry_detail"]["mask_uri"]).exists())
            self.assertEqual(obj["geometry_detail"]["mask_format"], "png")
            self.assertFalse(Path(obj["crop"]["crop_uri"]).is_absolute())
            self.assertTrue((processed_root / obj["crop"]["crop_uri"]).exists())
            self.assertIn("sidecar", localizer_info)
            self.assertIn("quality", generation_params)
            self.assertIn("attempts", localizer_info)
            self.assertFalse(Path(localizer_info["sidecar"]["debug_artifacts"]["heatmap"]).is_absolute())
            self.assertTrue((processed_root / localizer_info["sidecar"]["debug_artifacts"]["heatmap"]).exists())
            self.assertFalse(Path(localizer_info["sidecar"]["metrics"]["pgcd_heatmap_path"]).is_absolute())
            self.assertTrue((processed_root / localizer_info["sidecar"]["metrics"]["pgcd_heatmap_path"]).exists())
            log_dir = metadata_dir / "logs"
            self.assertTrue(any(log_dir.glob("localizer_benchmark_*.json")))
            self.assertTrue(any(log_dir.glob("localizer_benchmark_*.csv")))
            self.assertTrue(any(log_dir.glob("localizer_benchmark_*_summary.json")))
            self.assertTrue(any(log_dir.glob("localizer_failure_summary_*.json")))
            self.assertTrue(any(log_dir.glob("audit_sample_list_*.csv")))
            summary_json = read_json(next(log_dir.glob("localizer_benchmark_*_summary.json")))
            self.assertIn("manual_accept_rate", summary_json["rgb_diff"])
            audit_csv = next(log_dir.glob("audit_sample_list_*.csv"))
            audit_rows = read_csv(audit_csv)
            self.assertTrue(audit_rows)
            self.assertIn("task_id", audit_rows[0])
            self.assertIn("generated_image_uri", audit_rows[0])
            self.assertIn("localizer_used", audit_rows[0])
            self.assertIn("manual_accept", audit_rows[0])

    def test_ingest_generated_metadata_exports_trusted_i2i_seedream_bbox_and_crop(self) -> None:
        from PIL import Image, ImageDraw

        from autolabel.exporters.labelstudio import export_metadata_dir
        from autolabel.pipeline import ingest_generated_metadata

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            i2i_output_root = root / "i2i_outputs"
            metadata_input_dir = i2i_output_root / "metadata"
            metadata_input_dir.mkdir(parents=True)
            mask_dir = i2i_output_root / "debug" / "masks"
            mask_dir.mkdir(parents=True)
            processed_root = root / "processed"
            metadata_dir = processed_root / "metadata"
            original_path = root / "original.jpg"
            generated_path = i2i_output_root / "generated_images" / "sample_seedream.png"
            generated_path.parent.mkdir(parents=True)
            Image.new("RGB", (256, 192), (120, 120, 120)).save(original_path)
            generated = Image.new("RGB", (256, 192), (120, 120, 120))
            draw = ImageDraw.Draw(generated)
            draw.rectangle((100, 95, 150, 130), fill=(170, 190, 200))
            generated.save(generated_path)
            mask_path = mask_dir / "sample_seedream_obj_000001_mask.png"
            mask = Image.new("L", (256, 192), 0)
            ImageDraw.Draw(mask).rectangle((100, 95, 150, 130), fill=255)
            mask.save(mask_path)

            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_seedream",
                        "image_id": "image_seedream",
                        "image_uri": str(original_path),
                        "anomaly_type": "water_leak",
                        "source_type": "manual_upload",
                        "task_mode": "generation",
                    }
                ],
                ["sample_id", "image_id", "image_uri", "anomaly_type", "source_type", "task_mode"],
            )

            sample = normalize_autolabel_sample(
                {
                    "sample_id": "sample_seedream",
                    "image_asset": {
                        "image_id": "image_seedream",
                        "image_uri": str(generated_path),
                        "width": 256,
                        "height": 192,
                        "source_type": "generated",
                    },
                    "objects": [
                        {
                            "object_id": "obj_000001",
                            "object_type": "leakage_area",
                            "box": {"format": "xyxy", "x1": 100, "y1": 95, "x2": 150, "y2": 130},
                            "geometry_source": "synthetic_generator",
                            "geometry_model": {"model_name": "seedream+i2i_diff", "model_version": "test"},
                            "geometry_detail": {
                                "polygon": None,
                                "mask_uri": str(mask_path),
                                "mask_format": "png",
                                "generation_params": {
                                    "prompt_box": [96, 90, 160, 140],
                                    "final_bbox_source": "image_difference_within_selected_grid",
                                    "coarse_bbox_requires_postprocess": False,
                                    "seedream_quality_gate": {
                                        "passes_quality": True,
                                        "quality_reason": None,
                                        "outside_change_ratio": 0.02,
                                    },
                                },
                            },
                            "crop": {
                                "crop_id": "crop_000001",
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
                                        "confidence": 1.0,
                                        "evidence": "generated label",
                                    }
                                ],
                                "classifier_type": "rule",
                                "classifier_name": "synthetic_label_rule",
                                "classifier_version": "test",
                                "prompt_version": "test",
                                "raw_response": None,
                            },
                            "quality_check": None,
                        }
                    ],
                    "workflow": {"workflow_status": "classified"},
                    "export": {"export_format": "labelstudio", "export_status": "not_exported"},
                },
                pipeline_id="autolabel_dag_v1",
                pipeline_version="0.1.0",
            )
            write_json(metadata_input_dir / "sample_seedream.json", sample)

            config = load_config(ROOT / "configs" / "autolabel.yaml")
            with patch("autolabel.modules.generation.metadata_builder.create_localizer") as create_localizer:
                create_localizer.side_effect = AssertionError("trusted i2i bbox should not rerun localizer")
                written = ingest_generated_metadata(
                    i2i_output_root,
                    metadata_dir,
                    pipeline_config=config,
                    tasks_csv=manifest_path,
                    image_root=root,
                )

            ingested = read_json(written[0])
            obj = ingested["objects"][0]
            localizer = obj["geometry_detail"]["generation_params"]["localizer"]
            self.assertEqual(localizer["used"], "i2i_diff")
            self.assertEqual(localizer["postprocess_status"], "success")
            self.assertTrue(localizer["quality"]["passes_quality"])
            self.assertEqual(obj["box"], {"format": "xyxy", "x1": 100, "y1": 95, "x2": 150, "y2": 130})
            self.assertFalse(Path(obj["crop"]["crop_uri"]).is_absolute())
            self.assertTrue((processed_root / obj["crop"]["crop_uri"]).exists())
            self.assertFalse(Path(obj["geometry_detail"]["mask_uri"]).is_absolute())
            self.assertTrue((processed_root / obj["geometry_detail"]["mask_uri"]).exists())

            export_path = processed_root / "exports" / "labelstudio" / "import.json"
            tasks = export_metadata_dir(metadata_dir, export_path, generated_quality_gate=True)
            self.assertEqual(len(tasks), 1)
            result = tasks[0]["predictions"][0]["result"][0]
            self.assertEqual(result["meta"]["box_xyxy"], obj["box"])
            self.assertEqual(result["meta"]["crop"]["crop_uri"], obj["crop"]["crop_uri"])

    def test_crop_review_config_uses_detector_model_profile(self) -> None:
        config = deepcopy(load_config(ROOT / "configs" / "autolabel.yaml"))
        config["direct_annotation"]["crop_review"]["enabled"] = True
        detector = build_detector_runtime_config(config)
        detector["services"]["ppe_person"]["dry_run"] = True

        review = build_crop_review_config(config, detector)
        self.assertTrue(review["enabled"])
        self.assertTrue(review["dry_run"])
        self.assertEqual(review["model_name"], "qwen3.6-27b")
        self.assertEqual(review["model_ref"], "ppe_person_vlm_labelstudio_detector")

    def test_ingest_generated_metadata_can_disable_benchmark_outputs(self) -> None:
        from autolabel.pipeline import ingest_generated_metadata

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_dir = root / "processed" / "metadata"
            sample = normalize_autolabel_sample(
                {
                    "sample_id": "sample_no_benchmark",
                    "image_asset": {
                        "image_id": "image_no_benchmark",
                        "image_uri": "metadata/images/generated.jpg",
                        "width": 100,
                        "height": 80,
                        "source_type": "generated",
                    },
                    "objects": [],
                    "workflow": {"workflow_status": "classified"},
                    "export": {"export_format": "labelstudio", "export_status": "not_exported"},
                }
            )
            rows = [
                {
                    "sample_id": "sample_no_benchmark",
                    "object_id": "obj_001",
                    "localizer": "rgb_diff",
                    "success": True,
                }
            ]

            config = {
                "modules": {
                    "generation": {
                        "localizer": {
                            "benchmark": False,
                        }
                    }
                }
            }

            with (
                patch("autolabel.pipeline.load_generated_samples", return_value=[sample]),
                patch("autolabel.pipeline.apply_localizer_postprocess", return_value=(sample, rows)),
                patch("autolabel.pipeline.write_localizer_benchmark_reports") as write_benchmark,
                patch("autolabel.pipeline.write_audit_sample_csv") as write_audit,
            ):
                written = ingest_generated_metadata(
                    i2i_output_root=root / "i2i_outputs",
                    metadata_dir=metadata_dir,
                    pipeline_config=config,
                )

            self.assertEqual(written, [metadata_dir / "sample_no_benchmark.json"])
            self.assertFalse(write_benchmark.called)
            self.assertFalse(write_audit.called)

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

    def test_prepare_water_leak_manifest_writes_generation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "manifest.csv"
            output_path = root / "water_leak_generation_2.csv"
            write_csv(
                input_path,
                [
                    {
                        "sample_id": "sample_a",
                        "image_id": "image_a",
                        "image_uri": "a.jpg",
                        "source_type": "manual_upload",
                        "task_mode": "direct",
                        "width": "100",
                        "height": "80",
                    },
                    {
                        "sample_id": "sample_b",
                        "image_id": "image_b",
                        "image_uri": "b.jpg",
                        "source_type": "manual_upload",
                        "task_mode": "direct",
                        "width": "120",
                        "height": "90",
                    },
                ],
                ["sample_id", "image_id", "image_uri", "source_type", "task_mode", "width", "height"],
            )

            write_water_leak_generation_manifest(input_path, output_path, count=2)
            rows = read_csv(output_path)

            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0]["sample_id"].startswith("water_leak_0001_"))
            self.assertEqual({row["task_mode"] for row in rows}, {"generation"})
            self.assertEqual({row["anomaly_type"] for row in rows}, {"water_leak"})
            self.assertEqual({row["object_type"] for row in rows}, {"leakage_area"})

    def test_generation_preflight_validates_water_leak_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "water.jpg"
            image_path.write_bytes(b"placeholder")
            i2i_entrypoint = root / "I2I" / "src" / "main.py"
            i2i_entrypoint.parent.mkdir(parents=True)
            i2i_entrypoint.write_text("print('ok')\n", encoding="utf-8")
            manifest_path = root / "manifest.csv"
            write_csv(
                manifest_path,
                [
                    {
                        "sample_id": "sample_water",
                        "image_id": "image_water",
                        "image_uri": str(image_path),
                        "anomaly_type": "water_leak",
                        "source_type": "manual_upload",
                        "task_mode": "generation",
                    }
                ],
                ["sample_id", "image_id", "image_uri", "anomaly_type", "source_type", "task_mode"],
            )
            config = load_config(ROOT / "configs" / "autolabel.yaml")
            config["generation"]["image_model_key"] = "seedream5_image_editor"
            config["modules"]["generation"]["backends"]["vlm_wan_autolabel"]["project_dir"] = str(root / "I2I")

            report = run_generation_preflight(
                config,
                tasks_csv=manifest_path,
                image_root=root,
                output_root=root / "run" / "i2i_outputs",
                require_credentials=False,
            )

            self.assertFalse(report["skipped"])
            self.assertEqual(report["generation_rows"], 1)
            self.assertEqual(
                report["runtime_by_anomaly"]["water_leak"]["image_model_name"],
                "doubao-seedream-5-0-pro-260628",
            )

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


if __name__ == "__main__":
    unittest.main()
