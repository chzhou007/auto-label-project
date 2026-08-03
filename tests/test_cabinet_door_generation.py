from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw
import requests

ROOT = Path(__file__).resolve().parents[1]
I2I_SRC = ROOT / "external" / "I2I" / "src"
sys.path.insert(0, str(I2I_SRC))

from cabinet_door_batch import (
    CabinetDoorBatchConfig,
    SegmentationCandidateSelector,
    _segmentation_bbox,
    build_cabinet_door_selector_config,
    ensure_cabinet_door_output_dirs,
    expand_door_edit_bbox,
    list_source_images,
    prioritize_source_images,
    process_cabinet_door_image,
    run_cabinet_door_batch,
)
from config import ModelServiceConfig
from qwen_vlm_client import QwenVLMClient, _extract_text_from_response, validate_cabinet_door_selection
from wan_image_client import (
    WanImageClient,
    _generate_seedream_with_openai_sdk,
    validate_api_key_for_http_header,
)


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


def test_ark_responses_selector_payload_uses_input_image_and_input_text(tmp_path: Path) -> None:
    image_path = tmp_path / "preview.jpg"
    Image.new("RGB", (320, 180), (90, 100, 110)).save(image_path)
    selector_config = ModelServiceConfig(
        api_key="ark-test",
        provider="volcengine_ark_responses",
        endpoint="https://ark.cn-beijing.volces.com/api/v3/responses",
        api_key_env="ARK_API_KEY",
        endpoint_env="ARK_VLM_RESPONSES_URL",
    )
    client = QwenVLMClient("doubao-seed-2-1-pro-260628", selector_config)

    endpoints, payload, log_payload = client._build_request_payloads(str(image_path), "select one door")

    assert endpoints == ["https://ark.cn-beijing.volces.com/api/v3/responses"]
    assert payload["model"] == "doubao-seed-2-1-pro-260628"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["store"] is False
    assert payload["max_output_tokens"] == 512
    content = payload["input"][0]["content"]
    assert content[0]["type"] == "input_image"
    assert content[0]["image_url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "input_text", "text": "select one door"}
    assert log_payload["input"][0]["content"][0]["image_url"] == str(image_path)


def test_ark_responses_output_text_is_extracted() -> None:
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"status":"skip","confidence":0.9}',
                    }
                ],
            }
        ]
    }

    assert _extract_text_from_response(response) == '{"status":"skip","confidence":0.9}'


def test_default_selector_uses_ark_key_and_allows_model_override(tmp_path: Path) -> None:
    config = CabinetDoorBatchConfig(
        image_dir=tmp_path,
        output_root=tmp_path / "output",
        vlm_model="custom-vision-model",
    )
    with patch.dict("os.environ", {"ARK_API_KEY": "ark-secret"}, clear=False):
        service = build_cabinet_door_selector_config(config)

    assert service.provider == "volcengine_ark_responses"
    assert service.api_key == "ark-secret"
    assert service.endpoint == "https://ark.cn-beijing.volces.com/api/v3/responses"
    assert config.vlm_model == "custom-vision-model"


def test_ark_selector_timeout_is_configurable_and_overwrites_error_log(tmp_path: Path) -> None:
    image_path = tmp_path / "preview.jpg"
    log_path = tmp_path / "selector.json"
    Image.new("RGB", (320, 180), (90, 100, 110)).save(image_path)
    selector_config = ModelServiceConfig(
        api_key="ark-test",
        provider="volcengine_ark_responses",
        endpoint="https://ark.cn-beijing.volces.com/api/v3/responses",
        api_key_env="ARK_API_KEY",
        endpoint_env="ARK_VLM_RESPONSES_URL",
    )
    client = QwenVLMClient(
        "doubao-seed-2-1-pro-260628",
        selector_config,
        request_timeout_seconds=321,
        max_retries=0,
    )
    log_path.write_text('{"stale": true}', encoding="utf-8")

    with patch(
        "qwen_vlm_client.requests.post",
        side_effect=requests.ReadTimeout("selector timed out"),
    ) as post:
        try:
            client.select_cabinet_door_bbox(str(image_path), (1920, 1080), str(log_path))
        except RuntimeError as exc:
            assert "Configured read timeout=321s" in str(exc)
        else:
            raise AssertionError("expected selector timeout")

    assert post.call_args.kwargs["timeout"] == (15, 321.0)
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert "stale" not in log
    assert log["error"]["type"] == "ReadTimeout"
    assert log["request"]["failed_attempts"][0]["timeout_seconds"] == 321.0


def test_seedream_cabinet_door_mode_uses_one_context_crop() -> None:
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
    assert "Edit the input context crop itself" in payload["prompt"]
    assert "Open exactly that one currently closed electrical control-cabinet" in payload["prompt"]
    assert "Do not turn it into a server rack" in payload["prompt"]
    assert log_payload["seedream_mode"] == "cabinet_door_open"


def test_seedream_uses_direct_http_when_openai_package_is_missing() -> None:
    response = Mock()
    response.ok = True
    response.status_code = 200
    response.json.return_value = {"data": [{"url": "https://example.invalid/generated.png"}]}
    response.text = ""
    payload = {
        "model": "doubao-seedream-5-0-pro-260628",
        "prompt": "open one cabinet door",
        "image": "data:image/jpeg;base64,AAAA",
        "size": "2K",
        "n": 1,
        "response_format": "url",
        "watermark": False,
    }

    with (
        patch.dict(sys.modules, {"openai": None}),
        patch("wan_image_client.requests.post", return_value=response) as post,
    ):
        result = _generate_seedream_with_openai_sdk(
            "https://ark.cn-beijing.volces.com/api/v3",
            "ark-test",
            payload,
        )

    assert result["_transport"] == "requests_fallback"
    assert post.call_args.args[0] == "https://ark.cn-beijing.volces.com/api/v3/images/generations"
    assert post.call_args.kwargs["json"] == payload
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer ark-test"
    assert post.call_args.kwargs["timeout"] == (15, 300.0)


def test_seedream_api_key_preflight_rejects_placeholder_and_non_ascii() -> None:
    for invalid in ("<你的 Ark API Key>", "ark-密钥", "ark key with spaces"):
        try:
            validate_api_key_for_http_header(invalid)
        except RuntimeError as exc:
            assert "ARK_API_KEY" in str(exc)
        else:
            raise AssertionError(f"expected invalid API key to be rejected: {invalid!r}")

    assert validate_api_key_for_http_header("ark-valid_ascii-123") == "ark-valid_ascii-123"


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
        self.input_size = None
        self.edit_bbox = None

    def edit_image_with_wan(
        self,
        image_path,
        _prompt,
        _negative_prompt,
        bbox,
        output_path,
        _anomaly_type,
        **kwargs,
    ):
        self.calls += 1
        assert kwargs["seedream_mode"] == "cabinet_door_open"
        with Image.open(image_path) as image:
            self.input_size = image.size
            image.convert("RGB").save(output_path)
        self.edit_bbox = tuple(bbox)
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


def test_preselected_segmentation_bbox_skips_qwen_and_writes_close_open_pair(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "electrical_room.jpg"
    Image.new("RGB", (640, 480), (90, 100, 110)).save(image_path)
    output_root = tmp_path / "output"
    dirs = ensure_cabinet_door_output_dirs(output_root)
    config = CabinetDoorBatchConfig(image_dir=image_dir, output_root=output_root)
    qwen = _FakeQwen()
    seedream = _FakeSeedream()
    selection = {
        "status": "selected",
        "selection_backend": "segmentation_candidates",
        "bbox": [180, 80, 360, 400],
        "confidence": 0.91,
        "segmentation_confidence": 0.95,
        "state_open_probability": 0.01,
        "gate_not_door_probability": 0.02,
        "candidate_index": 3,
        "room": "A1 high-voltage room",
        "room_type": "高压配电室(HVDR)",
        "hinge_side": "unknown",
    }

    result = process_cabinet_door_image(
        image_path,
        config,
        dirs,
        qwen_client=qwen,
        image_client=seedream,
        preselected_selection=selection,
    )

    assert result["status"] == "accepted"
    assert result["qwen_call_count"] == 0
    assert result["seedream_call_count"] == 1
    assert qwen.calls == 0
    assert seedream.calls == 1
    assert result["source_closed_door_bbox"] == [180, 80, 360, 400]
    assert result["source_context_crop_bbox"] == [0, 42, 540, 438]
    assert result["crop_closed_door_bbox"] == [180, 38, 360, 358]
    assert result["crop_size"] == [540, 396]
    assert seedream.input_size == (540, 396)
    assert seedream.edit_bbox == (180, 38, 360, 358)
    assert Path(result["close_image"]).is_file()
    assert Path(result["open_image"]).is_file()
    with Image.open(result["close_image"]) as close_image, Image.open(result["open_image"]) as open_image:
        assert close_image.size == open_image.size == (540, 396)
    assert not any(dirs["generated_images"].iterdir())
    pair_annotation = json.loads(Path(result["pair_annotation"]).read_text(encoding="utf-8"))
    assert pair_annotation["pair_id"] == "electrical_room"
    assert pair_annotation["source_closed_door_bbox"] == [180, 80, 360, 400]


def _write_segmentation_csvs(root: Path, image_names: list[str]) -> tuple[Path, Path]:
    candidate_csv = root / "candidate_predictions.csv"
    image_csv = root / "image_predictions.csv"
    image_csv.write_text(
        "image_name,room,room_type,camera_position\n"
        f"{image_names[0]},A1 HV room,高压配电室(HVDR),camera-1\n"
        f"{image_names[1]},A1 module room,模块机房(M),camera-2\n",
        encoding="utf-8-sig",
    )
    candidate_csv.write_text(
        "image_name,candidate_index,segmentation_confidence,state_open_probability,"
        "gate_not_door_probability,passed_segmentation_gate,passed_candidate_deduplication,"
        "passed_not_door_gate,bbox_x1,bbox_y1,bbox_x2,bbox_y2,polygon_json\n"
        f"{image_names[0]},0,0.96,0.02,0.01,True,True,True,100,80,260,260,[]\n"
        f"{image_names[0]},1,0.99,0.01,0.01,True,True,True,0,0,640,480,[]\n"
        f"{image_names[0]},2,0.98,0.90,0.01,True,True,True,320,60,520,420,[]\n"
        f"{image_names[1]},0,0.97,0.01,0.01,True,True,True,100,60,300,420,[]\n",
        encoding="utf-8-sig",
    )
    return candidate_csv, image_csv


def test_segmentation_selector_uses_closed_electrical_candidate_and_rejects_rack_room(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "false_positive_originals_dedup_190"
    image_dir.mkdir()
    image_names = ["electrical.jpg", "rack.jpg"]
    for name in image_names:
        Image.new("RGB", (640, 480), (90, 100, 110)).save(image_dir / name)
    candidate_csv, image_csv = _write_segmentation_csvs(tmp_path, image_names)
    config = CabinetDoorBatchConfig(
        image_dir=image_dir,
        output_root=tmp_path / "output",
        candidate_predictions_csv=candidate_csv,
        image_predictions_csv=image_csv,
    )
    selector = SegmentationCandidateSelector(config)

    selected = selector.select(image_dir / image_names[0], (640, 480))
    skipped = selector.select(image_dir / image_names[1], (640, 480))

    assert selected["status"] == "selected"
    assert selected["bbox"] == [100, 80, 260, 260]
    assert selected["candidate_index"] == 0
    assert selected["selection_backend"] == "segmentation_candidates"
    assert skipped["status"] == "skipped"
    assert skipped["reason"].startswith("room_type_not_allowed:")

    striped = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(striped)
    for x in range(0, 640, 4):
        draw.line((x, 0, x, 480), fill="black", width=2)
    striped.save(image_dir / image_names[0])
    textured = selector.select(image_dir / image_names[0], (640, 480))
    assert textured["status"] == "skipped"
    assert textured["rejection_counts"]["bbox_texture_too_high"] == 1

    Image.new("RGB", (640, 480), (35, 35, 35)).save(image_dir / image_names[0])
    dark_neutral = selector.select(image_dir / image_names[0], (640, 480))
    assert dark_neutral["status"] == "skipped"
    assert dark_neutral["rejection_counts"]["dark_neutral_rack_like"] == 1


def test_segmentation_bbox_prefers_tight_polygon_over_expanded_candidate_crop() -> None:
    bbox, source = _segmentation_bbox(
        {
            "bbox_x1": "100",
            "bbox_y1": "50",
            "bbox_x2": "600",
            "bbox_y2": "450",
            "polygon_json": "[[220.4, 130.2], [381.6, 132.0], [379.1, 390.8], [218.9, 388.5]]",
        },
        (640, 480),
    )

    assert bbox == (218, 130, 383, 392)
    assert source == "polygon_bounds"


def test_priority_review_directory_reorders_clean_sources_without_using_annotated_files(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "clean"
    review_dir = tmp_path / "review"
    source_dir.mkdir()
    review_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        Image.new("RGB", (32, 32), "gray").save(source_dir / name)
    Image.new("RGB", (32, 32), "red").save(review_dir / "c.jpg")
    Image.new("RGB", (32, 32), "red").save(review_dir / "missing.jpg")
    images = list_source_images(source_dir)

    prioritized, matched = prioritize_source_images(images, review_dir)
    priority_only, priority_only_matched = prioritize_source_images(
        images,
        review_dir,
        priority_only=True,
    )

    assert [path.name for path in prioritized] == ["c.jpg", "a.jpg", "b.jpg"]
    assert [path.parent for path in prioritized] == [source_dir, source_dir, source_dir]
    assert matched == 1
    assert [path.name for path in priority_only] == ["c.jpg"]
    assert priority_only_matched == 1


def test_legacy_csv_without_dedup_field_rejects_crop_containing_existing_open_door(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_name = "battery_room.jpg"
    Image.new("RGB", (640, 480), (90, 100, 110)).save(image_dir / image_name)
    image_csv = tmp_path / "image_predictions.csv"
    image_csv.write_text(
        "image_name,room,room_type,camera_position\n"
        f"{image_name},A1 battery room,电池室(BR),camera-1\n",
        encoding="utf-8-sig",
    )
    candidate_csv = tmp_path / "candidate_predictions.csv"
    candidate_csv.write_text(
        "image_name,candidate_index,segmentation_confidence,state_open_probability,"
        "gate_not_door_probability,passed_segmentation_gate,passed_state_gate,"
        "passed_not_door_gate,fused_open,bbox_x1,bbox_y1,bbox_x2,bbox_y2,polygon_json\n"
        f"{image_name},0,0.96,0.02,0.01,True,False,True,False,100,80,260,260,[]\n"
        f"{image_name},1,0.95,0.99,0.01,True,True,True,True,250,80,400,260,[]\n",
        encoding="utf-8-sig",
    )
    selector = SegmentationCandidateSelector(
        CabinetDoorBatchConfig(
            image_dir=image_dir,
            output_root=tmp_path / "output",
            candidate_predictions_csv=candidate_csv,
            image_predictions_csv=image_csv,
        )
    )

    result = selector.select(image_dir / image_name, (640, 480))

    assert result["status"] == "skipped"
    assert result["rejection_counts"]["existing_open_door_in_context"] == 1
    assert result["rejection_counts"]["not_closed"] == 1

    candidate_csv.write_text(
        "image_name,candidate_index,segmentation_confidence,state_open_probability,"
        "gate_not_door_probability,passed_segmentation_gate,passed_state_gate,"
        "passed_not_door_gate,fused_open,bbox_x1,bbox_y1,bbox_x2,bbox_y2,polygon_json\n"
        f"{image_name},1,0.10,0.99,0.99,False,True,False,True,250,80,400,260,[]\n",
        encoding="utf-8-sig",
    )
    reviewed_selector = SegmentationCandidateSelector(
        CabinetDoorBatchConfig(
            image_dir=image_dir,
            output_root=tmp_path / "reviewed-output",
            candidate_predictions_csv=candidate_csv,
            image_predictions_csv=image_csv,
            allowed_room_type_regex=r"NEVER_MATCH",
            reviewed_all_close=True,
        )
    )
    reviewed = reviewed_selector.select(image_dir / image_name, (640, 480))
    assert reviewed["status"] == "selected"
    assert reviewed["candidate_index"] == 1
    assert reviewed["state_open_probability"] == 0.99
    assert reviewed["segmentation_confidence"] == 0.10
    assert reviewed["gate_not_door_probability"] == 0.99
    assert reviewed["manual_close_override"] is True


def test_segmentation_batch_limit_counts_selected_pairs_and_never_creates_qwen_client(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "false_positive_originals_dedup_190"
    image_dir.mkdir()
    image_names = ["electrical.jpg", "rack.jpg"]
    for name in image_names:
        Image.new("RGB", (640, 480), (90, 100, 110)).save(image_dir / name)
    candidate_csv, image_csv = _write_segmentation_csvs(tmp_path, image_names)
    output_root = tmp_path / "output"
    config = CabinetDoorBatchConfig(
        image_dir=image_dir,
        output_root=output_root,
        candidate_predictions_csv=candidate_csv,
        image_predictions_csv=image_csv,
        dry_run=True,
        limit=1,
    )

    summary = run_cabinet_door_batch(
        config,
        qwen_factory=Mock(side_effect=AssertionError("Qwen must not be created")),
    )

    assert summary["input_total"] == 2
    assert summary["total"] == 1
    assert summary["accepted"] == 1
    assert summary["qwen_call_count"] == 0
    assert summary["selector_call_count"] == 0
    assert summary["seedream_call_count"] == 0
    assert summary["segmentation_candidate_count"] == 1
    assert summary["segmentation_skipped_count"] == 1
    assert (output_root / "pairs" / "manifest.csv").is_file()


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


def test_skip_existing_reruns_failed_metadata(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    Image.new("RGB", (640, 480), (90, 100, 110)).save(image_path)
    output_root = tmp_path / "output"
    dirs = ensure_cabinet_door_output_dirs(output_root)
    (dirs["metadata"] / "sample.json").write_text(
        '{"sample_id":"sample","status":"failed"}',
        encoding="utf-8",
    )
    config = CabinetDoorBatchConfig(image_dir=image_dir, output_root=output_root, skip_existing=True)
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
    assert qwen.calls == 1
    assert seedream.calls == 1


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
