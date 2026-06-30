from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

from PIL import Image, ImageDraw

from autolabel.modules.generation.env_loader import load_env_files
from autolabel.modules.generation.grid import expand_bbox, grid_id_to_bbox
from autolabel.modules.generation.main import main as vlm_wan_main
from autolabel.modules.generation.prompts import build_negative_prompt, build_qwen_grid_prompt, build_wan_edit_prompt
from autolabel.modules.generation.qwen_vlm_client import QwenVLMClient
from autolabel.modules.generation.config import normalize_task_row
from autolabel.utils import read_json
from autolabel.validators import validate_sample_contract


class VLMWanAutoLabelTests(unittest.TestCase):
    def test_grid_bbox_and_expansion(self) -> None:
        self.assertEqual(grid_id_to_bbox("A1", 400, 200, "4x4"), (0, 0, 100, 50))
        self.assertEqual(grid_id_to_bbox("D4", 400, 200, "4x4"), (300, 150, 400, 200))
        self.assertEqual(expand_bbox((100, 50, 200, 100), 400, 200, 0.2), (80, 40, 220, 110))

    def test_water_leak_prompts_are_distinct_from_oil_diesel_coolant(self) -> None:
        qwen_prompt = build_qwen_grid_prompt("water_leak", excluded_grids=["A1", "D4"])
        wan_prompt = build_wan_edit_prompt("water_leak", "near a drain pipe and equipment base", "early")
        negative_prompt = build_negative_prompt()
        self.assertIn("漏水 / 清水泄漏", qwen_prompt)
        self.assertIn("top-3", qwen_prompt)
        self.assertIn("禁止选择这些网格：A1, D4", qwen_prompt)
        self.assertIn("裸露透明水渍", qwen_prompt)
        self.assertIn("不需要制造明确", wan_prompt)
        self.assertIn("极少量、早期、受控、局部", wan_prompt)
        self.assertIn("浅薄水膜", wan_prompt)
        self.assertIn("不是黄色柴油", wan_prompt)
        self.assertIn("银绿色防冻液", negative_prompt)
        self.assertIn("黑褐色油污", negative_prompt)
        self.assertIn("高压喷水", negative_prompt)
        self.assertIn("大面积积水", negative_prompt)
        self.assertIn("天花板漏水", negative_prompt)
        leakage_prompt = build_wan_edit_prompt("water_leakage", "small bare wet mark", "early")
        self.assertIn("极少量、早期、受控、局部", leakage_prompt)

    def test_water_leakage_alias_normalizes_to_water_leak(self) -> None:
        task = normalize_task_row({"task_id": "t1", "image_path": "a.jpg", "anomaly_type": "water_leakage"}, 1)
        self.assertEqual(task["anomaly_type"], "water_leak")

    def test_qwen_grid_selection_falls_back_on_timeout(self) -> None:
        class TimeoutQwen(QwenVLMClient):
            def _openai_compatible_vision_call(self, image_path: str | Path, prompt: str) -> str:
                raise TimeoutError("simulated timeout")

        client = TimeoutQwen(dry_run=False, api_key="test-key", request_timeout_seconds=1)
        selection = client.select_grid("missing-grid.jpg", "water_leak", "4x4")
        self.assertTrue(selection.raw_response["fallback"])
        self.assertIn("simulated timeout", selection.raw_response["fallback_reason"])
        self.assertEqual(len(selection.candidate_grids), 3)

    def test_qwen_top3_deduplicates_completes_and_excludes_sensitive_grids(self) -> None:
        client = QwenVLMClient(dry_run=True, candidate_grid_count=3)
        payload = {
            "selected_grid": "A1",
            "confidence": 0.9,
            "candidate_grids": ["A1", "A1", {"grid": "B2", "score": 0.8}, {"grid": "Z9", "score": 0.7}],
            "edit_region_hint": "small floor wet mark",
        }
        selection = client._parse_grid_selection(payload, "4x4", 3, excluded_grids=["A1", "D4"])  # noqa: SLF001
        grids = [candidate.grid for candidate in selection.candidate_grids]
        self.assertEqual(len(grids), 3)
        self.assertEqual(len(set(grids)), 3)
        self.assertNotIn("A1", grids)
        self.assertNotIn("D4", grids)
        self.assertIn("B2", grids)

    def test_env_loader_reads_dotenv_without_overriding_existing_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "DASHSCOPE_API_KEY=file-key\n"
                "QWEN_VLM_BASE_URL=https://example.test/v1\n"
                "export DASHSCOPE_WAN_ENDPOINT='https://wan.example.test/generation'\n",
                encoding="utf-8",
            )
            with mock.patch.dict("os.environ", {"DASHSCOPE_API_KEY": "shell-key"}, clear=True):
                loaded = load_env_files([env_path])
                import os

                self.assertEqual(loaded, [env_path])
                self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "shell-key")
                self.assertEqual(os.environ["QWEN_VLM_BASE_URL"], "https://example.test/v1")
                self.assertEqual(os.environ["DASHSCOPE_WAN_ENDPOINT"], "https://wan.example.test/generation")

    def test_dry_run_writes_valid_metadata_with_diff_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            output_root = root / "outputs"
            image_root.mkdir()
            image_path = image_root / "engine_room.jpg"
            image = Image.new("RGB", (360, 240), (115, 118, 120))
            draw = ImageDraw.Draw(image)
            draw.rectangle((40, 50, 320, 170), fill=(80, 84, 86), outline=(160, 160, 160), width=3)
            draw.line((130, 105, 300, 105), fill=(185, 150, 42), width=10)
            draw.rectangle((0, 170, 360, 240), fill=(92, 94, 96))
            image.save(image_path, quality=95)
            tasks = root / "tasks.csv"
            tasks.write_text(
                "task_id,image_path,anomaly_type,source_type\n"
                "task_0001,engine_room.jpg,water_leak,generated\n",
                encoding="utf-8",
            )

            code = vlm_wan_main(
                [
                    "--tasks",
                    str(tasks),
                    "--image-root",
                    str(image_root),
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                    "--export-labelstudio",
                    "--benchmark",
                ]
            )
            self.assertEqual(code, 0)
            metadata_path = output_root / "metadata" / "sample_task_0001.json"
            sample = read_json(metadata_path)
            validate_sample_contract(sample)
            obj = sample["objects"][0]
            params = obj["geometry_detail"]["generation_params"]
            self.assertTrue(Path(obj["geometry_detail"]["mask_uri"]).exists())
            self.assertEqual(obj["geometry_detail"]["mask_format"], "png")
            self.assertNotEqual(
                [obj["box"]["x1"], obj["box"]["y1"], obj["box"]["x2"], obj["box"]["y2"]],
                params["expanded_edit_bbox"],
            )
            self.assertIn("background_preservation_score", params)
            self.assertIn("anomaly_visibility_score", params)
            self.assertEqual(len(params["coarse_candidate_grids_top3"]), 3)
            self.assertEqual(params["candidate_grid_count"], 3)
            self.assertIn("excluded_sensitive_grids", params)
            self.assertFalse(params["fine_grid_enabled"])
            self.assertTrue(params["fine_grid_removed_or_skipped"])
            self.assertEqual(params["selected_coarse_grid"], params["selected_grid"])
            self.assertIn("coarse_candidate_rank", params)
            self.assertEqual(obj["classification"]["multi_labels"][0]["label_value"], "water_leak")
            self.assertTrue((output_root / "metadata" / "import.json").exists())
            summaries = list((output_root / "metadata" / "logs").glob("run_summary_*.json"))
            self.assertEqual(len(summaries), 1)
            summary = read_json(summaries[0])
            self.assertEqual(summary["total_tasks"], 1)
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(summary["counters"]["fine_grid_calls"], 0)
            self.assertIn("topk_avg_attempt_count", summary)
            top_level_dirs = {path.name for path in output_root.iterdir() if path.is_dir()}
            self.assertEqual(top_level_dirs, {"crops", "masks", "metadata"})

    def test_skip_existing_reruns_invalid_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            output_root = root / "outputs"
            metadata_root = output_root / "metadata"
            image_root.mkdir()
            metadata_root.mkdir(parents=True)
            image_path = image_root / "engine_room.jpg"
            image = Image.new("RGB", (320, 220), (100, 104, 106))
            draw = ImageDraw.Draw(image)
            draw.rectangle((30, 40, 280, 155), fill=(78, 80, 82), outline=(150, 150, 150), width=3)
            draw.rectangle((0, 155, 320, 220), fill=(88, 90, 92))
            image.save(image_path, quality=95)
            (metadata_root / "sample_task_0001.json").write_text('{"invalid": true}', encoding="utf-8")
            tasks = root / "tasks.csv"
            tasks.write_text(
                "task_id,image_path,anomaly_type,source_type\n"
                "task_0001,engine_room.jpg,water_leak,generated\n",
                encoding="utf-8",
            )

            code = vlm_wan_main(
                [
                    "--tasks",
                    str(tasks),
                    "--image-root",
                    str(image_root),
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                    "--skip-existing",
                ]
            )
            self.assertEqual(code, 0)
            sample = read_json(metadata_root / "sample_task_0001.json")
            validate_sample_contract(sample)


if __name__ == "__main__":
    unittest.main()
