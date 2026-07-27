from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from autolabel.qc_service import (
    EVIDENCE_MODE,
    QCServiceError,
    discover_bundle,
    run_qc_bundle,
    safe_extract_zip,
)


def sample_metadata() -> dict:
    return {
        "sample_id": "water_leak_sample_001",
        "image_asset": {
            "image_id": "water_leak_001",
            "image_uri": r"C:\batch\i2i_outputs\generated_images\sample.png",
            "width": 100,
            "height": 80,
            "source_type": "generated",
            "scene_context": {
                "inspection_content": "water_leak",
                "task_group": "equipment_environment_anomaly",
            },
        },
        "objects": [
            {
                "object_id": "obj_000001",
                "object_type": "leakage_area",
                "box": {
                    "format": "xyxy",
                    "x1": 20,
                    "y1": 30,
                    "x2": 60,
                    "y2": 60,
                },
                "geometry_source": "synthetic_generator",
                "geometry_detail": {
                    "generation_params": {
                        "selected_grid": "C2",
                        "grid_bbox": [0, 20, 80, 80],
                        "expanded_edit_bbox": [0, 10, 90, 80],
                        "final_bbox_source": "image_difference_within_selected_grid",
                        "localization_pipeline": "test_pipeline",
                    }
                },
                "classification": {
                    "multi_labels": [
                        {
                            "label_key": "anomaly_type",
                            "label_value": "water_leak",
                            "confidence": 1.0,
                        }
                    ]
                },
            }
        ],
        "workflow": {"workflow_status": "classified"},
        "export": {"export_format": "labelstudio", "export_status": "not_exported"},
    }


def write_test_bundle(root: Path) -> Path:
    source = root / "source" / "i2i_outputs"
    metadata_dir = source / "metadata"
    image_dir = source / "generated_images"
    metadata_dir.mkdir(parents=True)
    image_dir.mkdir()
    (metadata_dir / "sample.json").write_text(
        json.dumps(sample_metadata(), ensure_ascii=False),
        encoding="utf-8",
    )
    image = Image.new("RGB", (100, 80), color=(210, 210, 210))
    image.save(image_dir / "sample.png")
    archive = root / "bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                handle.write(path, Path("i2i_outputs") / path.relative_to(source))
    return archive


class QCServiceTests(unittest.TestCase):
    def test_safe_extract_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../outside.txt", "unsafe")

            with self.assertRaises(QCServiceError):
                safe_extract_zip(archive, root / "extract")

            self.assertFalse((root / "outside.txt").exists())

    def test_discover_bundle_finds_metadata_and_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = write_test_bundle(root)
            extract_root = safe_extract_zip(archive, root / "extract")

            bundle = discover_bundle(extract_root)

            self.assertEqual(bundle.metadata_count, 1)
            self.assertEqual(bundle.object_count, 1)
            self.assertEqual(bundle.metadata_dir.name, "metadata")
            self.assertEqual(bundle.asset_root.name, "i2i_outputs")

    def test_failed_job_removes_extracted_input_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../outside.txt", "unsafe")

            with patch.dict("os.environ", {"QC_KEEP_INPUTS": "false"}):
                with self.assertRaises(QCServiceError):
                    run_qc_bundle(
                        archive,
                        work_root=root / "runs",
                        mode=EVIDENCE_MODE,
                    )

            run_dirs = list((root / "runs").iterdir())
            self.assertEqual(len(run_dirs), 1)
            self.assertFalse((run_dirs[0] / "input").exists())
            self.assertFalse((root / "outside.txt").exists())

    def test_evidence_only_bundle_runs_end_to_end_and_packages_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = write_test_bundle(root)

            result = run_qc_bundle(
                archive,
                work_root=root / "runs",
                mode=EVIDENCE_MODE,
            )

            self.assertTrue(result.archive_path.is_file())
            self.assertEqual(len(result.table_rows), 1)
            self.assertEqual(len(result.gallery_items), 1)
            self.assertFalse((result.run_dir / "input").exists())
            with zipfile.ZipFile(result.archive_path) as handle:
                names = set(handle.namelist())
            self.assertIn("bbox_qc/bbox_qc_results.jsonl", names)
            self.assertIn("bbox_qc/evidence_panels/water_leak_sample_001_obj_000001_crop_evidence.jpg", names)
            self.assertIn("rule_qc", {name.split("/", 1)[0] for name in names})
            self.assertIn("run.log", names)


if __name__ == "__main__":
    unittest.main()
