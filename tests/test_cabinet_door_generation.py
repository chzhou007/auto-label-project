from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
I2I_SRC = ROOT / "external" / "I2I" / "src"
sys.path.insert(0, str(I2I_SRC))

from cabinet_door_batch import (
    CabinetDoorBatchConfig,
    ensure_cabinet_door_output_dirs,
    expand_door_edit_bbox,
    list_source_images,
    process_cabinet_door_image,
)
from config import ModelServiceConfig
from qwen_vlm_client import validate_cabinet_door_selection
from wan_image_client import WanImageClient


def test_cabinet_door_bbox_maps_normalized_coordinates_to_original_pixels() -> None:
    selected = validate_cabinet_door_selection(
        {
            "status": "selected",
            "bbox_1000": [250, 100, 500, 900],
            "confidence": 0.92,
            "hinge_side": "left",
        },
        (1920, 1080),
    )

    assert selected["status"] == "selected"
    assert selected["bbox"] == [480, 108, 960, 972]
    assert selected["hinge_side"] == "left"


def test_cabinet_door_selector_skips_no_target_and_low_confidence() -> None:
    no_target = validate_cabinet_door_selection(
        {"status": "skip", "bbox_1000": None, "confidence": 0.95, "reason": "only open racks"},
        (1920, 1080),
    )
    low_confidence = validate_cabinet_door_selection(
        {"status": "selected", "bbox_1000": [100, 100, 500, 900], "confidence": 0.42},
        (1920, 1080),
    )

    assert no_target["status"] == "skipped"
    assert low_confidence["status"] == "skipped"


def test_seedream_cabinet_door_mode_uses_one_original_image() -> None:
    image_config = ModelServiceConfig(
        api_key="test",
        provider="volcengine_ark",
        endpoint="https://ark.cn-beijing.volces.com/api/v3",
        api_key_env="ARK_API_KEY",
        endpoint_env="SEEDREAM_BASE_URL",
    )
    client = WanImageClient("doubao-seedream-5-0-pro-260628", image_config)

    source = ROOT / "tests" / "test_result" / "cabinet_door_payload_source.jpg"
    source.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), (90, 100, 110)).save(source)
    try:
        _endpoint, payload, log_payload, is_seedream = client._build_request_payloads(
            str(source),
            "open selected door",
            "",
            (100, 80, 300, 420),
            seedream_mode="cabinet_door_open",
            anomaly_type="cabinet_door_open",
        )
    finally:
        source.unlink(missing_ok=True)

    assert is_seedream is True
    assert payload["n"] == 1
    assert isinstance(payload["image"], str)
    assert payload["image"].startswith("data:image/jpeg;base64,")
    assert "Open exactly one currently closed equipment-cabinet door" in payload["prompt"]
    assert log_payload["seedream_mode"] == "cabinet_door_open"


class _FakeQwen:
    def __init__(self, status: str = "selected"):
        self.status = status
        self.calls = 0

    def select_cabinet_door_bbox(self, _preview, image_size, _log, *, min_confidence):
        self.calls += 1
        if self.status == "skipped":
            return {
                "status": "skipped",
                "confidence": 0.9,
                "reason": "no closed equipment door",
                "bbox": None,
                "bbox_1000": None,
            }
        width, height = image_size
        return {
            "status": "selected",
            "confidence": max(0.9, min_confidence),
            "reason": "clear closed cabinet door",
            "bbox": [width // 3, height // 5, width // 2, height * 4 // 5],
            "bbox_1000": [333, 200, 500, 800],
            "hinge_side": "left",
        }


class _FakeSeedream:
    def __init__(self):
        self.calls = 0

    def edit_image_with_wan(
        self,
        image_path,
        _prompt,
        _negative_prompt,
        _bbox,
        output_path,
        _anomaly_type,
        **kwargs,
    ):
        self.calls += 1
        assert kwargs["seedream_mode"] == "cabinet_door_open"
        Image.open(image_path).convert("RGB").save(output_path)
        return {"data": [{"url": "fake"}]}


def test_one_selected_image_calls_qwen_and_seedream_once(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    Image.new("RGB", (640, 480), (90, 100, 110)).save(image_path)
    output_root = tmp_path / "output"
    dirs = ensure_cabinet_door_output_dirs(output_root)
    config = CabinetDoorBatchConfig(image_dir=image_dir, output_root=output_root)
    qwen = _FakeQwen()
    seedream = _FakeSeedream()

    result = process_cabinet_door_image(
        image_path,
        config,
        dirs,
        qwen_client=qwen,
        image_client=seedream,
    )

    assert result["status"] == "accepted"
    assert result["qwen_call_count"] == 1
    assert result["seedream_call_count"] == 1
    assert qwen.calls == 1
    assert seedream.calls == 1
    assert Path(result["generated_image"]).is_file()
    assert Path(result["crop_image"]).is_file()


def test_skipped_image_never_calls_seedream(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    Image.new("RGB", (640, 480), (90, 100, 110)).save(image_path)
    output_root = tmp_path / "output"
    dirs = ensure_cabinet_door_output_dirs(output_root)
    config = CabinetDoorBatchConfig(image_dir=image_dir, output_root=output_root)
    qwen = _FakeQwen(status="skipped")
    seedream = _FakeSeedream()

    result = process_cabinet_door_image(
        image_path,
        config,
        dirs,
        qwen_client=qwen,
        image_client=seedream,
    )

    assert result["status"] == "skipped"
    assert result["qwen_call_count"] == 1
    assert result["seedream_call_count"] == 0
    assert qwen.calls == 1
    assert seedream.calls == 0


def test_manifest_order_and_hinge_aware_edit_expansion(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("b.jpg", "a.jpg"):
        Image.new("RGB", (100, 100), (0, 0, 0)).save(image_dir / name)
    (image_dir / "manifest.csv").write_text("image_name\nb.jpg\na.jpg\n", encoding="utf-8")

    images = list_source_images(image_dir)
    expanded = expand_door_edit_bbox((400, 200, 600, 800), (1920, 1080), hinge_side="left")

    assert [path.name for path in images] == ["b.jpg", "a.jpg"]
    assert expanded == (330, 128, 800, 872)
