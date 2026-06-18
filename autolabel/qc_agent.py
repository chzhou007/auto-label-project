from __future__ import annotations

import base64
import io
import hashlib
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from .adapters.crop_reviewer import parse_review_payload
from .adapters.vlm_labelstudio_detector import VLMJsonParseError, bool_config, preview_text
from .model_config import deep_merge, get_credentials
from .utils import get_image_size, now_iso_shanghai, read_json, write_csv, write_json
from .validators import ValidationError, validate_sample_contract


DEFAULT_QC_AGENT_CONFIG: dict[str, Any] = {
    "output_dir": "data/qc",
    "manual_sampling_ratio": 0.05,
    "random_seed": 20260601,
    "rules": {
        "check_image_dimensions": True,
        "require_crop_file": True,
        "min_box_width": 4,
        "min_box_height": 4,
        "min_box_area": 16,
        "max_box_aspect_ratio": 20.0,
        "min_crop_width": 4,
        "min_crop_height": 4,
        "max_crop_aspect_ratio": 20.0,
        "crop_size_tolerance_px": 2,
        "duplicate_iou_threshold": 0.98,
        "require_classification_labels": False,
        "require_mask_for_generated": False,
        "check_generation_region_alignment": True,
        "generation_region_tolerance_px": 2,
        "min_selected_grid_iou": 0.01,
        "check_mask_box_alignment": True,
        "min_mask_box_iou": 0.35,
        "min_mask_coverage_by_box": 0.8,
        "max_mask_box_center_shift_ratio": 0.5,
        "max_mask_box_area_ratio": 5.0,
    },
    "vlm_review": {
        "enabled": False,
        "model_ref": "ppe_person_vlm_labelstudio_detector",
        "reviewer": "vlm_annotation_qc_agent",
        "prompt_version": "annotation_qc_v1",
        "request_image_max_side": 1280,
        "temperature": 0.0,
        "max_tokens": 800,
        "use_response_format": True,
        "response_format_type": "json_object",
        "store_raw_response": False,
    },
}


QC_REPORT_FIELDS = [
    "sample_id",
    "image_id",
    "source_type",
    "collection_batch",
    "camera_id",
    "capture_time",
    "site",
    "building",
    "floor",
    "room_name",
    "room_type",
    "task_group",
    "inspection_content",
    "object_id",
    "object_type",
    "geometry_source",
    "geometry_model_name",
    "geometry_model_version",
    "geometry_confidence",
    "classifier_type",
    "classifier_name",
    "classifier_version",
    "label_summary",
    "workflow_status",
    "pipeline_id",
    "pipeline_version",
    "export_format",
    "export_status",
    "qc_batch_id",
    "status",
    "risk_score",
    "issue_codes",
    "issue_field_paths",
    "issue_messages",
    "metadata_uri",
    "image_uri",
    "resolved_image_uri",
    "crop_uri",
    "resolved_crop_uri",
    "manual_review_reason",
]


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    object_id: str | None = None,
    field_path: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "object_id": object_id,
        "field_path": field_path,
        "evidence": evidence or {},
    }


def _is_remote_or_data_uri(value: str) -> bool:
    return value.startswith(("http://", "https://", "data:"))


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path.absolute())
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _candidate_roots(metadata_path: Path, extra_roots: list[Path] | None = None) -> list[Path]:
    roots: list[Path] = []
    roots.extend(extra_roots or [])
    roots.append(Path.cwd())
    roots.append(metadata_path.parent)
    roots.extend(metadata_path.parent.parents[:8])
    return _dedupe_paths(roots)


def resolve_local_uri(
    uri: str | None,
    metadata_path: Path,
    extra_roots: list[Path] | None = None,
) -> tuple[Path | None, bool, list[str]]:
    if not uri or _is_remote_or_data_uri(uri):
        return None, bool(uri), []

    value = Path(uri)
    if value.is_absolute():
        return value, value.exists(), [str(value)]

    candidates = [root / value for root in _candidate_roots(metadata_path, extra_roots)]
    for candidate in candidates:
        if candidate.exists():
            return candidate, True, [str(path) for path in candidates]
    return candidates[0] if candidates else value, False, [str(path) for path in candidates]


def _box_numbers(box: Any) -> tuple[int, int, int, int] | None:
    try:
        if isinstance(box, dict):
            return int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        if isinstance(box, (list, tuple)) and len(box) == 4:
            return int(box[0]), int(box[1]), int(box[2]), int(box[3])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _box_from_numbers(values: tuple[int, int, int, int]) -> dict[str, int | str]:
    x1, y1, x2, y2 = values
    return {"format": "xyxy", "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _box_size(box: Any) -> tuple[int, int] | None:
    values = _box_numbers(box)
    if values is None:
        return None
    x1, y1, x2, y2 = values
    return x2 - x1, y2 - y1


def _box_contains(outer: Any, inner: Any) -> bool:
    outer_values = _box_numbers(outer)
    inner_values = _box_numbers(inner)
    if outer_values is None or inner_values is None:
        return False
    ox1, oy1, ox2, oy2 = outer_values
    ix1, iy1, ix2, iy2 = inner_values
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def _padded_box(box: Any, tolerance: int) -> dict[str, int | str] | None:
    values = _box_numbers(box)
    if values is None:
        return None
    x1, y1, x2, y2 = values
    return _box_from_numbers((x1 - tolerance, y1 - tolerance, x2 + tolerance, y2 + tolerance))


def _box_area(box: Any) -> float:
    values = _box_numbers(box)
    if values is None:
        return 0.0
    x1, y1, x2, y2 = values
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _box_intersection_area(left: Any, right: Any) -> float:
    left_values = _box_numbers(left)
    right_values = _box_numbers(right)
    if left_values is None or right_values is None:
        return 0.0
    ax1, ay1, ax2, ay2 = left_values
    bx1, by1, bx2, by2 = right_values
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    return float(inter_w * inter_h)


def _box_center_shift_ratio(reference: Any, candidate: Any) -> tuple[float, float] | None:
    ref_values = _box_numbers(reference)
    candidate_values = _box_numbers(candidate)
    if ref_values is None or candidate_values is None:
        return None
    rx1, ry1, rx2, ry2 = ref_values
    cx1, cy1, cx2, cy2 = candidate_values
    ref_w = max(1.0, float(rx2 - rx1))
    ref_h = max(1.0, float(ry2 - ry1))
    ref_cx = (rx1 + rx2) / 2.0
    ref_cy = (ry1 + ry2) / 2.0
    candidate_cx = (cx1 + cx2) / 2.0
    candidate_cy = (cy1 + cy2) / 2.0
    return abs(candidate_cx - ref_cx) / ref_w, abs(candidate_cy - ref_cy) / ref_h


def _mask_content_bbox(mask_path: Path) -> dict[str, int | str] | None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency is optional at import time.
        raise RuntimeError("Pillow is required for mask content QC") from exc

    with Image.open(mask_path) as image:
        bbox = image.convert("L").point(lambda pixel: 255 if pixel else 0).getbbox()
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    return _box_from_numbers((left, top, right, bottom))


def _box_iou(left: Any, right: Any) -> float:
    left_values = _box_numbers(left)
    right_values = _box_numbers(right)
    if left_values is None or right_values is None:
        return 0.0
    ax1, ay1, ax2, ay2 = left_values
    bx1, by1, bx2, by2 = right_values
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    left_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    right_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _issue_risk(issues: list[dict[str, Any]]) -> float:
    weights = {"error": 1.0, "warning": 0.35, "info": 0.05}
    return min(1.0, sum(weights.get(issue.get("severity"), 0.1) for issue in issues))


def _status_from_issues(issues: list[dict[str, Any]]) -> str:
    if any(issue.get("severity") == "error" for issue in issues):
        return "failed"
    if any(issue.get("severity") == "warning" for issue in issues):
        return "needs_human_review"
    return "passed"


def _stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _preview(value: Any, limit: int = 300) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _classification_labels(classification: Any) -> list[dict[str, Any]]:
    if not isinstance(classification, dict):
        return []
    labels = classification.get("multi_labels")
    return labels if isinstance(labels, list) else []


def _label_summary(classification: Any) -> str:
    parts = []
    for label in _classification_labels(classification):
        if not isinstance(label, dict):
            continue
        key = label.get("label_key")
        value = label.get("label_value")
        if key and value:
            parts.append(f"{key}={value}")
    return ";".join(parts)


def _sample_field_snapshot(sample: dict[str, Any]) -> dict[str, Any]:
    image_asset = sample.get("image_asset") if isinstance(sample.get("image_asset"), dict) else {}
    source_context = image_asset.get("source_context") if isinstance(image_asset.get("source_context"), dict) else {}
    scene_context = image_asset.get("scene_context") if isinstance(image_asset.get("scene_context"), dict) else {}
    generation_prompt = source_context.get("generation_prompt")
    return {
        "sample_id": sample.get("sample_id"),
        "image_asset": {
            "image_id": image_asset.get("image_id"),
            "image_uri": image_asset.get("image_uri"),
            "width": image_asset.get("width"),
            "height": image_asset.get("height"),
            "source_type": image_asset.get("source_type"),
            "source_context": {
                "generation_model": source_context.get("generation_model"),
                "collection_batch": source_context.get("collection_batch"),
                "camera_id": source_context.get("camera_id"),
                "capture_time": source_context.get("capture_time"),
                "generation_prompt_present": bool(generation_prompt),
                "generation_prompt_chars": len(str(generation_prompt)) if generation_prompt else 0,
                "generation_prompt_preview": _preview(generation_prompt),
            },
            "scene_context": {
                "site": scene_context.get("site"),
                "building": scene_context.get("building"),
                "floor": scene_context.get("floor"),
                "room_name": scene_context.get("room_name"),
                "room_type": scene_context.get("room_type"),
                "task_group": scene_context.get("task_group"),
                "inspection_content": scene_context.get("inspection_content"),
            },
        },
        "qc_policy": deepcopy(sample.get("qc_policy")),
        "workflow": deepcopy(sample.get("workflow")),
        "export": deepcopy(sample.get("export")),
    }


def _object_field_snapshot(obj: dict[str, Any]) -> dict[str, Any]:
    geometry_detail = obj.get("geometry_detail") if isinstance(obj.get("geometry_detail"), dict) else {}
    generation_params = geometry_detail.get("generation_params")
    classification = obj.get("classification") if isinstance(obj.get("classification"), dict) else {}
    raw_response = classification.get("raw_response")
    return {
        "object_id": obj.get("object_id"),
        "object_type": obj.get("object_type"),
        "box": deepcopy(obj.get("box")),
        "geometry_source": obj.get("geometry_source"),
        "geometry_model": deepcopy(obj.get("geometry_model")),
        "geometry_detail": {
            "polygon": deepcopy(geometry_detail.get("polygon")),
            "mask_uri": geometry_detail.get("mask_uri"),
            "mask_format": geometry_detail.get("mask_format"),
            "generation_params_present": isinstance(generation_params, dict),
            "generation_params_keys": sorted(generation_params.keys()) if isinstance(generation_params, dict) else [],
            "selected_grid": generation_params.get("selected_grid") if isinstance(generation_params, dict) else None,
            "final_bbox_source": generation_params.get("final_bbox_source") if isinstance(generation_params, dict) else None,
            "localization_pipeline": generation_params.get("localization_pipeline") if isinstance(generation_params, dict) else None,
        },
        "crop": deepcopy(obj.get("crop")),
        "classification": {
            "multi_labels": deepcopy(_classification_labels(classification)),
            "classifier_type": classification.get("classifier_type"),
            "classifier_name": classification.get("classifier_name"),
            "classifier_version": classification.get("classifier_version"),
            "prompt_version": classification.get("prompt_version"),
            "raw_response_present": raw_response is not None,
        },
        "quality_check": deepcopy(obj.get("quality_check")),
    }


def _flatten_sample_context(sample: dict[str, Any]) -> dict[str, Any]:
    snapshot = sample.get("field_snapshot") or {}
    image_asset = snapshot.get("image_asset") if isinstance(snapshot.get("image_asset"), dict) else {}
    source_context = image_asset.get("source_context") if isinstance(image_asset.get("source_context"), dict) else {}
    scene_context = image_asset.get("scene_context") if isinstance(image_asset.get("scene_context"), dict) else {}
    workflow = snapshot.get("workflow") if isinstance(snapshot.get("workflow"), dict) else {}
    export = snapshot.get("export") if isinstance(snapshot.get("export"), dict) else {}
    qc_policy = snapshot.get("qc_policy") if isinstance(snapshot.get("qc_policy"), dict) else {}
    return {
        "image_id": image_asset.get("image_id"),
        "source_type": image_asset.get("source_type") or sample.get("source_type"),
        "collection_batch": source_context.get("collection_batch"),
        "camera_id": source_context.get("camera_id"),
        "capture_time": source_context.get("capture_time"),
        "site": scene_context.get("site"),
        "building": scene_context.get("building"),
        "floor": scene_context.get("floor"),
        "room_name": scene_context.get("room_name"),
        "room_type": scene_context.get("room_type"),
        "task_group": scene_context.get("task_group"),
        "inspection_content": scene_context.get("inspection_content"),
        "workflow_status": workflow.get("workflow_status"),
        "pipeline_id": workflow.get("pipeline_id"),
        "pipeline_version": workflow.get("pipeline_version"),
        "export_format": export.get("export_format"),
        "export_status": export.get("export_status"),
        "qc_batch_id": qc_policy.get("qc_batch_id"),
    }


def _flatten_object_context(obj: dict[str, Any] | None) -> dict[str, Any]:
    if not obj:
        return {
            "object_type": "",
            "geometry_source": "",
            "geometry_model_name": "",
            "geometry_model_version": "",
            "geometry_confidence": "",
            "classifier_type": "",
            "classifier_name": "",
            "classifier_version": "",
            "label_summary": "",
        }
    snapshot = obj.get("field_snapshot") or {}
    geometry_model = snapshot.get("geometry_model") if isinstance(snapshot.get("geometry_model"), dict) else {}
    classification = snapshot.get("classification") if isinstance(snapshot.get("classification"), dict) else {}
    return {
        "object_type": snapshot.get("object_type") or obj.get("object_type"),
        "geometry_source": snapshot.get("geometry_source"),
        "geometry_model_name": geometry_model.get("model_name"),
        "geometry_model_version": geometry_model.get("model_version"),
        "geometry_confidence": geometry_model.get("confidence"),
        "classifier_type": classification.get("classifier_type"),
        "classifier_name": classification.get("classifier_name"),
        "classifier_version": classification.get("classifier_version"),
        "label_summary": _label_summary(classification),
    }


def parse_vlm_qc_payload(text: str) -> dict[str, Any]:
    parsed = parse_review_payload(text)
    box_quality = str(parsed.get("box_quality") or "uncertain").strip().lower()
    if box_quality not in {"good", "over_inclusive", "under_inclusive", "wrong_target", "uncertain"}:
        box_quality = "uncertain"
    return {
        "visible_target": _parse_bool(parsed.get("visible_target")),
        "box_quality": box_quality,
        "label_match": _parse_bool(parsed.get("label_match"), default=True),
        "needs_human_review": _parse_bool(parsed.get("needs_human_review")),
        "issue_flags": _string_list(parsed.get("issue_flags")),
        "reason": str(parsed.get("reason") or "").strip(),
        "raw": parsed,
    }


def build_qc_agent_config(config: dict[str, Any]) -> dict[str, Any]:
    qc_cfg = deep_merge(DEFAULT_QC_AGENT_CONFIG, config.get("qc_agent", {}) or {})
    vlm_cfg = dict(qc_cfg.get("vlm_review") or {})
    model_ref = vlm_cfg.get("model_ref")
    profile: dict[str, Any] = {}
    if model_ref:
        profile = deepcopy(config.get("models", {}).get("geometry", {}).get("candidates", {}).get(model_ref, {}))
    merged_vlm = deep_merge(profile, vlm_cfg)
    credential = get_credentials(config, merged_vlm.get("credential_ref"))
    merged_vlm["api_key"] = merged_vlm.get("api_key") or credential.get("api_key")
    merged_vlm["base_url"] = (
        merged_vlm.get("base_url")
        or merged_vlm.get("api_url")
        or credential.get("base_url")
        or credential.get("api_url")
    )
    merged_vlm["model_name"] = merged_vlm.get("model_name") or merged_vlm.get("model")
    qc_cfg["vlm_review"] = merged_vlm
    return qc_cfg


def collect_metadata_paths(
    metadata_dir: str | Path | None = None,
    sample_paths: list[str | Path] | None = None,
    limit: int | None = None,
) -> list[Path]:
    if sample_paths:
        paths = [Path(path) for path in sample_paths]
    elif metadata_dir:
        paths = sorted(Path(metadata_dir).glob("*.json"))
    else:
        raise ValueError("metadata_dir or sample_paths is required")
    if limit is not None:
        paths = paths[: max(0, int(limit))]
    return paths


def build_overlay_data_url(
    image_path: Path,
    box: dict[str, Any],
    *,
    max_side: int | None = None,
    jpeg_quality: int = 90,
) -> str:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("Pillow is required to build QC overlay images.") from exc

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        x1, y1, x2, y2 = _box_numbers(box) or (0, 0, 1, 1)
        line_width = max(3, round(min(image.size) / 180))
        for offset in range(line_width):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=(255, 0, 0))
        if max_side is not None and max_side > 0 and max(image.size) > max_side:
            scale = max_side / float(max(image.size))
            resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(resized, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class VLMAnnotationQCReviewer:
    def __init__(self, review_cfg: dict[str, Any]) -> None:
        self.review_cfg = review_cfg
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("openai is required for VLM annotation QC.") from exc
            api_key = self.review_cfg.get("api_key")
            base_url = self.review_cfg.get("base_url") or self.review_cfg.get("api_url")
            if not api_key:
                raise RuntimeError("QC VLM api_key is not configured.")
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def review_object(self, sample: dict[str, Any], obj: dict[str, Any], image_path: Path) -> dict[str, Any]:
        max_side = self.review_cfg.get("request_image_max_side")
        max_side = int(max_side) if max_side not in (None, "") else None
        image_url = build_overlay_data_url(image_path, obj.get("box", {}), max_side=max_side)
        prompt = self._prompt(sample, obj)
        raw_text = self._request_json_text(image_url, prompt)
        parsed = parse_vlm_qc_payload(raw_text)
        if self.review_cfg.get("store_raw_response"):
            parsed["raw_response_text"] = raw_text
        return parsed

    def _prompt(self, sample: dict[str, Any], obj: dict[str, Any]) -> str:
        labels = obj.get("classification", {}).get("multi_labels") or []
        label_text = ", ".join(
            f"{item.get('label_key')}={item.get('label_value')}"
            for item in labels
            if isinstance(item, dict) and item.get("label_key")
        ) or "无分类标签"
        scene = sample.get("image_asset", {}).get("scene_context") or {}
        source_type = sample.get("image_asset", {}).get("source_type")
        return f"""你是工业异常检测标注质检员。图片上的红色矩形是待复核的异常标注框。

这次只质检“异常区域框”本身。红框必须框住图中真实可见的异常痕迹，例如漏油、漏水、湿痕、液膜、积液、污渍、反光水迹、异常颜色或异常形态。阀门、螺栓、管道、接头、设备边缘、地面空白、阴影和正常反光都不能当成异常目标。

样本信息：
- sample_id: {sample.get("sample_id")}
- source_type: {source_type}
- object_id: {obj.get("object_id")}
- object_type: {obj.get("object_type")}
- labels: {label_text}
- inspection_content: {scene.get("inspection_content")}
- task_group: {scene.get("task_group")}

判定要求：
1. visible_target 只表示红框内是否能看到与 labels / inspection_content 对应的可见异常痕迹；只看到设备部件时必须为 false。
2. label_match 表示红框内的异常痕迹是否和标签一致。比如 diesel_leak 应该有柴油/油渍/湿润油膜特征，coolant_leak 应该有冷却液或对应液体痕迹。
3. box_quality 只能从 good、over_inclusive、under_inclusive、wrong_target、uncertain 中选一个。
4. good：红框基本覆盖主要异常区域，允许少量边缘留白。
5. over_inclusive：红框包含异常，但背景、设备或正常区域明显过多。
6. under_inclusive：红框只框住异常的一小部分，或者图中主异常明显延伸到红框外。
7. wrong_target：红框框到了设备、螺栓、管道、阴影、文字、正常反光、空白区域，或框内没有可见异常。
8. uncertain：画质差、异常太弱、遮挡严重，无法稳定判断。
9. 如果图中有明显异常但红框没有覆盖主要异常，按 under_inclusive 或 wrong_target 处理，并设置 needs_human_review=true。
10. wrong_target 直接 needs_human_review=true；under_inclusive、over_inclusive、uncertain 也需要复核。
11. 只输出合法 JSON object，不要 markdown，不要解释文字。

输出格式：
{{
  "visible_target": true,
  "box_quality": "good",
  "label_match": true,
  "needs_human_review": false,
  "issue_flags": [],
  "reason": ""
}}
"""

    def _request_json_text(self, image_url: str, prompt: str) -> str:
        model_name = self.review_cfg.get("model_name")
        if not model_name:
            raise RuntimeError("QC VLM model_name is not configured.")
        messages = [
            {
                "role": "system",
                "content": self.review_cfg.get(
                    "system_message",
                    "You are a strict JSON API for visual annotation quality control. Return JSON only.",
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": float(self.review_cfg.get("temperature", 0.0)),
            "max_tokens": max(1, int(self.review_cfg.get("max_tokens", 800))),
        }
        if bool_config(self.review_cfg.get("use_response_format"), default=True):
            kwargs["response_format"] = {
                "type": self.review_cfg.get("response_format_type", "json_object"),
            }
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if "response_format" not in message and "json_object" not in message and "guided" not in message:
                raise
            kwargs.pop("response_format", None)
            response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


class QCAnnotationAgent:
    def __init__(self, qc_cfg: dict[str, Any], asset_base_dirs: list[str | Path] | None = None) -> None:
        self.qc_cfg = qc_cfg
        self.rules = qc_cfg.get("rules", {}) or {}
        self.asset_base_dirs = [Path(path) for path in (asset_base_dirs or [])]
        self.vlm_reviewer = (
            VLMAnnotationQCReviewer(qc_cfg.get("vlm_review", {}))
            if bool_config((qc_cfg.get("vlm_review") or {}).get("enabled"), default=False)
            else None
        )

    def review_metadata_path(self, metadata_path: str | Path) -> dict[str, Any]:
        path = Path(metadata_path)
        sample_issues: list[dict[str, Any]] = []
        object_results: list[dict[str, Any]] = []
        resolved_image_uri = None
        sample: dict[str, Any] = {}

        try:
            sample = read_json(path)
        except Exception as exc:
            issues = [_issue("error", "invalid_json", f"metadata JSON cannot be read: {exc}", field_path="$")]
            return self._record_for_unreadable_path(path, issues)

        try:
            validate_sample_contract(sample)
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            sample_issues.append(
                _issue(
                    "error",
                    "contract_validation_failed",
                    str(exc),
                    field_path="$",
                    evidence={"validator": "validate_sample_contract"},
                )
            )

        image_asset = sample.get("image_asset") if isinstance(sample.get("image_asset"), dict) else {}
        image_uri = image_asset.get("image_uri")
        image_path, image_exists, image_candidates = resolve_local_uri(image_uri, path, self.asset_base_dirs)
        if image_path is not None:
            resolved_image_uri = str(image_path)
        if image_uri and image_path is not None and not image_exists:
            sample_issues.append(
                _issue(
                    "error",
                    "image_file_missing",
                    f"image file not found: {image_uri}",
                    field_path="image_asset.image_uri",
                    evidence={"candidates": image_candidates[:5]},
                )
            )

        actual_image_size = None
        if image_path is not None and image_exists:
            try:
                actual_image_size = get_image_size(image_path)
            except Exception as exc:
                sample_issues.append(_issue("error", "image_file_unreadable", str(exc), field_path="image_asset.image_uri"))
            if actual_image_size and bool_config(self.rules.get("check_image_dimensions"), default=True):
                expected = (int(image_asset.get("width", 0) or 0), int(image_asset.get("height", 0) or 0))
                if expected != actual_image_size:
                    sample_issues.append(
                        _issue(
                            "error",
                            "image_dimension_mismatch",
                            f"metadata image size {expected} != actual image size {actual_image_size}",
                            field_path="image_asset.width,image_asset.height",
                        )
                    )

        objects = sample.get("objects") if isinstance(sample.get("objects"), list) else []
        sample_issues.extend(self._duplicate_object_id_issues(objects))
        sample_issues.extend(self._duplicate_box_issues(objects))

        for obj in objects:
            object_issues = self._review_object_rules(sample, obj, path, actual_image_size)
            if self.vlm_reviewer is not None and image_path is not None and image_exists and _box_numbers(obj.get("box")):
                object_issues.extend(self._review_object_with_vlm(sample, obj, image_path))
            status = _status_from_issues(object_issues)
            object_results.append(
                {
                    "object_id": obj.get("object_id"),
                    "object_type": obj.get("object_type"),
                    "status": status,
                    "risk_score": _issue_risk(object_issues),
                    "issues": object_issues,
                    "field_snapshot": _object_field_snapshot(obj),
                    "crop_uri": obj.get("crop", {}).get("crop_uri") if isinstance(obj.get("crop"), dict) else None,
                    "resolved_crop_uri": self._resolved_crop_uri(obj, path),
                }
            )

        all_issues = sample_issues + [issue for obj in object_results for issue in obj["issues"]]
        status = _status_from_issues(all_issues)
        return {
            "sample_id": sample.get("sample_id") or path.stem,
            "status": status,
            "risk_score": _issue_risk(all_issues),
            "manual_review": status != "passed",
            "manual_review_reason": "issues" if status != "passed" else None,
            "metadata_uri": str(path),
            "image_uri": image_uri,
            "resolved_image_uri": resolved_image_uri,
            "source_type": image_asset.get("source_type"),
            "object_count": len(objects),
            "field_snapshot": _sample_field_snapshot(sample),
            "issues": sample_issues,
            "objects": object_results,
        }

    def _record_for_unreadable_path(self, path: Path, issues: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "sample_id": path.stem,
            "status": "failed",
            "risk_score": _issue_risk(issues),
            "manual_review": True,
            "manual_review_reason": "issues",
            "metadata_uri": str(path),
            "image_uri": None,
            "resolved_image_uri": None,
            "source_type": None,
            "object_count": 0,
            "field_snapshot": {},
            "issues": issues,
            "objects": [],
        }

    def _duplicate_object_id_issues(self, objects: list[Any]) -> list[dict[str, Any]]:
        counts = Counter(
            obj.get("object_id")
            for obj in objects
            if isinstance(obj, dict) and obj.get("object_id")
        )
        return [
            _issue(
                "error",
                "duplicate_object_id",
                f"object_id appears {count} times: {object_id}",
                field_path="objects[].object_id",
                evidence={"object_id": object_id, "count": count},
            )
            for object_id, count in counts.items()
            if count > 1
        ]

    def _duplicate_box_issues(self, objects: list[Any]) -> list[dict[str, Any]]:
        threshold = float(self.rules.get("duplicate_iou_threshold", 0.98) or 0)
        if threshold <= 0:
            return []
        issues: list[dict[str, Any]] = []
        clean_objects = [obj for obj in objects if isinstance(obj, dict)]
        for idx, left in enumerate(clean_objects):
            for right in clean_objects[idx + 1 :]:
                if left.get("object_type") != right.get("object_type"):
                    continue
                iou = _box_iou(left.get("box"), right.get("box"))
                if iou >= threshold:
                    issues.append(
                        _issue(
                            "warning",
                            "duplicate_or_overlapping_box",
                            f"two {left.get('object_type')} boxes overlap heavily: IoU={iou:.3f}",
                            object_id=left.get("object_id"),
                            field_path="objects[].box",
                            evidence={"other_object_id": right.get("object_id"), "iou": round(iou, 4)},
                        )
                    )
        return issues

    def _review_object_rules(
        self,
        sample: dict[str, Any],
        obj: Any,
        metadata_path: Path,
        actual_image_size: tuple[int, int] | None,
    ) -> list[dict[str, Any]]:
        if not isinstance(obj, dict):
            return [_issue("error", "object_not_object", "object entry must be a JSON object", field_path="objects[]")]
        object_id = obj.get("object_id")
        issues: list[dict[str, Any]] = []
        issues.extend(self._box_rule_issues(obj, actual_image_size))
        issues.extend(self._generation_region_issues(obj))
        issues.extend(self._crop_rule_issues(obj, metadata_path))
        issues.extend(self._mask_rule_issues(sample, obj, metadata_path, actual_image_size))
        issues.extend(self._classification_rule_issues(obj))
        issues.extend(self._existing_quality_check_issues(obj))
        for issue in issues:
            issue.setdefault("object_id", object_id)
        return issues

    def _box_rule_issues(
        self,
        obj: dict[str, Any],
        actual_image_size: tuple[int, int] | None,
    ) -> list[dict[str, Any]]:
        object_id = obj.get("object_id")
        box = obj.get("box")
        values = _box_numbers(box)
        if values is None:
            return [
                _issue(
                    "error",
                    "invalid_box_shape",
                    "object.box must contain numeric x1/y1/x2/y2",
                    object_id=object_id,
                    field_path="objects[].box",
                )
            ]
        x1, y1, x2, y2 = values
        if x2 <= x1 or y2 <= y1:
            return [
                _issue(
                    "error",
                    "invalid_box_geometry",
                    f"box has non-positive size: {box}",
                    object_id=object_id,
                    field_path="objects[].box",
                )
            ]
        if actual_image_size is not None:
            width, height = actual_image_size
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                return [
                    _issue(
                        "error",
                        "box_outside_image",
                        f"box is outside image bounds {actual_image_size}: {box}",
                        object_id=object_id,
                        field_path="objects[].box",
                    )
                ]
        box_w, box_h = x2 - x1, y2 - y1
        min_w = float(self.rules.get("min_box_width", 4) or 0)
        min_h = float(self.rules.get("min_box_height", 4) or 0)
        min_area = float(self.rules.get("min_box_area", 16) or 0)
        max_ratio = float(self.rules.get("max_box_aspect_ratio", 20.0) or 0)
        issues = []
        if box_w < min_w or box_h < min_h or box_w * box_h < min_area:
            issues.append(
                _issue(
                    "error",
                    "tiny_box",
                    f"box is smaller than QC thresholds: width={box_w}, height={box_h}, area={box_w * box_h}",
                    object_id=object_id,
                    field_path="objects[].box",
                )
            )
        if max_ratio and max(box_w / box_h, box_h / box_w) > max_ratio:
            issues.append(
                _issue(
                    "error",
                    "extreme_box_aspect_ratio",
                    f"box aspect ratio is too extreme: width={box_w}, height={box_h}",
                    object_id=object_id,
                    field_path="objects[].box",
                )
            )
        return issues

    def _generation_region_issues(self, obj: dict[str, Any]) -> list[dict[str, Any]]:
        if not bool_config(self.rules.get("check_generation_region_alignment"), default=True):
            return []
        object_id = obj.get("object_id")
        object_box = obj.get("box")
        if _box_numbers(object_box) is None:
            return []

        detail = obj.get("geometry_detail") if isinstance(obj.get("geometry_detail"), dict) else {}
        params = detail.get("generation_params") if isinstance(detail.get("generation_params"), dict) else {}
        if not params:
            return []

        issues: list[dict[str, Any]] = []
        tolerance = int(self.rules.get("generation_region_tolerance_px", 2) or 0)
        expanded_edit_bbox = params.get("expanded_edit_bbox")
        grid_bbox = params.get("grid_bbox")

        if expanded_edit_bbox is not None:
            expanded_values = _box_numbers(expanded_edit_bbox)
            if expanded_values is None:
                issues.append(
                    _issue(
                        "warning",
                        "invalid_expanded_edit_bbox",
                        "generation_params.expanded_edit_bbox must be [x1,y1,x2,y2]",
                        object_id=object_id,
                        field_path="objects[].geometry_detail.generation_params.expanded_edit_bbox",
                        evidence={"expanded_edit_bbox": expanded_edit_bbox},
                    )
                )
            elif not _box_contains(_padded_box(expanded_values, tolerance), object_box):
                issues.append(
                    _issue(
                        "warning",
                        "box_outside_expanded_edit_region",
                        "object.box is outside the generation expanded edit region",
                        object_id=object_id,
                        field_path="objects[].box",
                        evidence={
                            "object_box": object_box,
                            "expanded_edit_bbox": list(expanded_values),
                            "tolerance_px": tolerance,
                            "final_bbox_source": params.get("final_bbox_source"),
                        },
                    )
                )

        if grid_bbox is not None:
            grid_values = _box_numbers(grid_bbox)
            if grid_values is None:
                issues.append(
                    _issue(
                        "warning",
                        "invalid_selected_grid_bbox",
                        "generation_params.grid_bbox must be [x1,y1,x2,y2]",
                        object_id=object_id,
                        field_path="objects[].geometry_detail.generation_params.grid_bbox",
                        evidence={"grid_bbox": grid_bbox, "selected_grid": params.get("selected_grid")},
                    )
                )
            else:
                grid_iou = _box_iou(object_box, grid_values)
                min_grid_iou = float(self.rules.get("min_selected_grid_iou", 0.01) or 0)
                if grid_iou < min_grid_iou:
                    issues.append(
                        _issue(
                            "warning",
                            "box_disconnected_from_selected_grid",
                            f"object.box barely overlaps selected grid: IoU={grid_iou:.3f}",
                            object_id=object_id,
                            field_path="objects[].box",
                            evidence={
                                "object_box": object_box,
                                "grid_bbox": list(grid_values),
                                "selected_grid": params.get("selected_grid"),
                                "grid_layout": params.get("grid_layout"),
                                "grid_iou": round(grid_iou, 4),
                                "min_selected_grid_iou": min_grid_iou,
                                "final_bbox_source": params.get("final_bbox_source"),
                            },
                        )
                    )

        return issues

    def _crop_rule_issues(self, obj: dict[str, Any], metadata_path: Path) -> list[dict[str, Any]]:
        object_id = obj.get("object_id")
        crop = obj.get("crop") if isinstance(obj.get("crop"), dict) else {}
        crop_uri = crop.get("crop_uri")
        crop_path, crop_exists, crop_candidates = resolve_local_uri(crop_uri, metadata_path, self.asset_base_dirs)
        issues: list[dict[str, Any]] = []
        if bool_config(self.rules.get("require_crop_file"), default=True) and crop_path is not None and not crop_exists:
            issues.append(
                _issue(
                    "error",
                    "crop_file_missing",
                    f"crop file not found: {crop_uri}",
                    object_id=object_id,
                    field_path="objects[].crop.crop_uri",
                    evidence={"candidates": crop_candidates[:5]},
                )
            )
            return issues
        if crop_path is None or not crop_exists:
            return issues

        try:
            crop_size = get_image_size(crop_path)
        except Exception as exc:
            return [
                _issue(
                    "error",
                    "crop_file_unreadable",
                    str(exc),
                    object_id=object_id,
                    field_path="objects[].crop.crop_uri",
                )
            ]

        crop_w, crop_h = crop_size
        min_w = float(self.rules.get("min_crop_width", 4) or 0)
        min_h = float(self.rules.get("min_crop_height", 4) or 0)
        max_ratio = float(self.rules.get("max_crop_aspect_ratio", 20.0) or 0)
        if crop_w < min_w or crop_h < min_h:
            issues.append(
                _issue(
                    "error",
                    "tiny_crop",
                    f"crop is smaller than QC thresholds: width={crop_w}, height={crop_h}",
                    object_id=object_id,
                    field_path="objects[].crop.crop_uri",
                )
            )
        if max_ratio and max(crop_w / crop_h, crop_h / crop_w) > max_ratio:
            issues.append(
                _issue(
                    "error",
                    "extreme_crop_aspect_ratio",
                    f"crop aspect ratio is too extreme: width={crop_w}, height={crop_h}",
                    object_id=object_id,
                    field_path="objects[].crop.crop_uri",
                )
            )

        crop_box = crop.get("crop_box")
        crop_box_size = _box_size(crop_box)
        if crop_box_size is not None:
            tolerance = int(self.rules.get("crop_size_tolerance_px", 2) or 0)
            expected_w, expected_h = crop_box_size
            if abs(crop_w - expected_w) > tolerance or abs(crop_h - expected_h) > tolerance:
                issues.append(
                    _issue(
                        "warning",
                        "crop_size_mismatch",
                        f"crop image size {crop_size} does not match crop_box size {(expected_w, expected_h)}",
                        object_id=object_id,
                        field_path="objects[].crop.crop_box",
                    )
                )
            if not _box_contains(crop_box, obj.get("box")):
                issues.append(
                    _issue(
                        "error",
                        "crop_does_not_cover_box",
                        "crop_box does not fully contain object.box",
                        object_id=object_id,
                        field_path="objects[].crop.crop_box",
                    )
                )
        return issues

    def _mask_rule_issues(
        self,
        sample: dict[str, Any],
        obj: dict[str, Any],
        metadata_path: Path,
        actual_image_size: tuple[int, int] | None,
    ) -> list[dict[str, Any]]:
        object_id = obj.get("object_id")
        detail = obj.get("geometry_detail") if isinstance(obj.get("geometry_detail"), dict) else {}
        mask_uri = detail.get("mask_uri")
        source_type = sample.get("image_asset", {}).get("source_type")
        if not mask_uri:
            if source_type == "generated" and bool_config(self.rules.get("require_mask_for_generated"), default=False):
                return [
                    _issue(
                        "warning",
                        "generated_mask_missing",
                        "generated sample has no mask_uri",
                        object_id=object_id,
                        field_path="objects[].geometry_detail.mask_uri",
                    )
                ]
            return []
        mask_path, mask_exists, mask_candidates = resolve_local_uri(mask_uri, metadata_path, self.asset_base_dirs)
        if mask_path is not None and not mask_exists:
            return [
                _issue(
                    "error",
                    "mask_file_missing",
                    f"mask file not found: {mask_uri}",
                    object_id=object_id,
                    field_path="objects[].geometry_detail.mask_uri",
                    evidence={"candidates": mask_candidates[:5]},
                )
            ]
        if mask_path is None or not mask_exists or actual_image_size is None:
            return []
        try:
            mask_size = get_image_size(mask_path)
        except Exception as exc:
            return [
                _issue(
                    "error",
                    "mask_file_unreadable",
                    str(exc),
                    object_id=object_id,
                    field_path="objects[].geometry_detail.mask_uri",
                )
            ]
        if mask_size != actual_image_size:
            return [
                _issue(
                    "warning",
                    "mask_dimension_mismatch",
                    f"mask size {mask_size} does not match image size {actual_image_size}",
                    object_id=object_id,
                    field_path="objects[].geometry_detail.mask_uri",
                )
            ]
        if not bool_config(self.rules.get("check_mask_box_alignment"), default=True):
            return []

        try:
            mask_bbox = _mask_content_bbox(mask_path)
        except Exception as exc:
            return [
                _issue(
                    "warning",
                    "mask_content_unreadable",
                    f"mask content bbox cannot be computed: {exc}",
                    object_id=object_id,
                    field_path="objects[].geometry_detail.mask_uri",
                )
            ]
        if mask_bbox is None:
            return [
                _issue(
                    "warning",
                    "empty_mask_content",
                    "mask file has no non-zero pixels",
                    object_id=object_id,
                    field_path="objects[].geometry_detail.mask_uri",
                )
            ]

        object_box = obj.get("box")
        object_values = _box_numbers(object_box)
        if object_values is None:
            return []

        intersection = _box_intersection_area(object_box, mask_bbox)
        mask_area = _box_area(mask_bbox)
        object_area = _box_area(object_box)
        mask_coverage_by_box = intersection / mask_area if mask_area else 0.0
        mask_box_iou = _box_iou(object_box, mask_bbox)
        center_shift = _box_center_shift_ratio(object_box, mask_bbox) or (0.0, 0.0)
        max_center_shift = max(center_shift)
        area_ratio = max(object_area / mask_area, mask_area / object_area) if mask_area and object_area else 0.0

        min_mask_box_iou = float(self.rules.get("min_mask_box_iou", 0.35) or 0)
        min_mask_coverage = float(self.rules.get("min_mask_coverage_by_box", 0.8) or 0)
        max_center_shift_ratio = float(self.rules.get("max_mask_box_center_shift_ratio", 0.5) or 0)
        max_area_ratio = float(self.rules.get("max_mask_box_area_ratio", 5.0) or 0)
        evidence = {
            "object_box": object_box,
            "mask_bbox": mask_bbox,
            "mask_box_iou": round(mask_box_iou, 4),
            "mask_coverage_by_box": round(mask_coverage_by_box, 4),
            "center_shift_x_ratio": round(center_shift[0], 4),
            "center_shift_y_ratio": round(center_shift[1], 4),
            "area_ratio": round(area_ratio, 4),
            "thresholds": {
                "min_mask_box_iou": min_mask_box_iou,
                "min_mask_coverage_by_box": min_mask_coverage,
                "max_mask_box_center_shift_ratio": max_center_shift_ratio,
                "max_mask_box_area_ratio": max_area_ratio,
            },
        }

        issues: list[dict[str, Any]] = []
        if mask_coverage_by_box < min_mask_coverage:
            issues.append(
                _issue(
                    "warning",
                    "box_misses_mask_content",
                    "object.box does not cover enough non-zero mask content",
                    object_id=object_id,
                    field_path="objects[].box",
                    evidence=evidence,
                )
            )
        if (
            mask_box_iou < min_mask_box_iou
            or (max_center_shift_ratio and max_center_shift > max_center_shift_ratio)
            or (max_area_ratio and area_ratio > max_area_ratio)
        ):
            issues.append(
                _issue(
                    "warning",
                    "mask_box_alignment_suspect",
                    "object.box and mask content bbox are weakly aligned",
                    object_id=object_id,
                    field_path="objects[].box",
                    evidence=evidence,
                )
            )
        return issues

    def _classification_rule_issues(self, obj: dict[str, Any]) -> list[dict[str, Any]]:
        if not bool_config(self.rules.get("require_classification_labels"), default=False):
            return []
        labels = obj.get("classification", {}).get("multi_labels") if isinstance(obj.get("classification"), dict) else None
        if not labels:
            return [
                _issue(
                    "warning",
                    "classification_labels_missing",
                    "object has no classification labels",
                    object_id=obj.get("object_id"),
                    field_path="objects[].classification.multi_labels",
                )
            ]
        return []

    def _existing_quality_check_issues(self, obj: dict[str, Any]) -> list[dict[str, Any]]:
        qc = obj.get("quality_check")
        if not isinstance(qc, dict):
            return []
        status = qc.get("qc_status")
        if status == "failed":
            return [
                _issue(
                    "warning",
                    "existing_object_qc_failed",
                    qc.get("comment") or "object already has failed quality_check",
                    object_id=obj.get("object_id"),
                    field_path="objects[].quality_check.qc_status",
                    evidence={"issue_flags": qc.get("issue_flags", [])},
                )
            ]
        if status == "pending":
            return [
                _issue(
                    "warning",
                    "existing_object_qc_pending",
                    "object quality_check is still pending",
                    object_id=obj.get("object_id"),
                    field_path="objects[].quality_check.qc_status",
                )
            ]
        if status in {"discarded"}:
            return [
                _issue(
                    "error",
                    "existing_object_qc_discarded",
                    "object quality_check is discarded",
                    object_id=obj.get("object_id"),
                    field_path="objects[].quality_check.qc_status",
                )
            ]
        return []

    def _review_object_with_vlm(self, sample: dict[str, Any], obj: dict[str, Any], image_path: Path) -> list[dict[str, Any]]:
        object_id = obj.get("object_id")
        try:
            result = self.vlm_reviewer.review_object(sample, obj, image_path)  # type: ignore[union-attr]
        except (VLMJsonParseError, ValueError) as exc:
            return [
                _issue(
                    "warning",
                    "vlm_qc_parse_error",
                    str(exc),
                    object_id=object_id,
                    field_path="objects[].box",
                )
            ]
        except Exception as exc:
            return [
                _issue(
                    "warning",
                    "vlm_qc_error",
                    f"VLM QC failed: {preview_text(str(exc), 200)}",
                    object_id=object_id,
                    field_path="objects[].box",
                )
            ]

        evidence = {
            "reviewer": (self.qc_cfg.get("vlm_review") or {}).get("reviewer"),
            "prompt_version": (self.qc_cfg.get("vlm_review") or {}).get("prompt_version"),
            "box_quality": result.get("box_quality"),
            "issue_flags": result.get("issue_flags", []),
            "reason": result.get("reason"),
        }
        if self.qc_cfg.get("vlm_review", {}).get("store_raw_response"):
            evidence["raw_response_text"] = result.get("raw_response_text")
        if not result.get("visible_target") or not result.get("label_match") or result.get("box_quality") == "wrong_target":
            return [
                _issue(
                    "error",
                    "vlm_qc_target_mismatch",
                    result.get("reason") or "VLM QC says the red box does not match the target",
                    object_id=object_id,
                    field_path="objects[].box",
                    evidence=evidence,
                )
            ]
        if result.get("needs_human_review") or result.get("box_quality") in {"over_inclusive", "under_inclusive", "uncertain"}:
            return [
                _issue(
                    "warning",
                    "vlm_qc_needs_human_review",
                    result.get("reason") or f"VLM QC box_quality={result.get('box_quality')}",
                    object_id=object_id,
                    field_path="objects[].box",
                    evidence=evidence,
                )
            ]
        return []

    def _resolved_crop_uri(self, obj: dict[str, Any], metadata_path: Path) -> str | None:
        crop_uri = obj.get("crop", {}).get("crop_uri") if isinstance(obj.get("crop"), dict) else None
        crop_path, _, _ = resolve_local_uri(crop_uri, metadata_path, self.asset_base_dirs)
        return str(crop_path) if crop_path is not None else None


def _apply_manual_sampling(samples: list[dict[str, Any]], ratio: float, seed: int) -> None:
    if ratio <= 0:
        return
    for sample in samples:
        if sample.get("manual_review"):
            continue
        fraction = _stable_fraction(str(sample.get("sample_id")), seed)
        if fraction < ratio:
            sample["manual_review"] = True
            sample["manual_review_reason"] = "sampling"


def summarize_qc_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(sample.get("status") for sample in samples)
    issue_counts = Counter(
        issue.get("code")
        for sample in samples
        for issue in sample.get("issues", [])
    )
    for sample in samples:
        for obj in sample.get("objects", []):
            for issue in obj.get("issues", []):
                issue_counts[issue.get("code")] += 1
    return {
        "total_samples": len(samples),
        "total_objects": sum(int(sample.get("object_count") or 0) for sample in samples),
        "passed_samples": status_counts.get("passed", 0),
        "failed_samples": status_counts.get("failed", 0),
        "needs_human_review_samples": status_counts.get("needs_human_review", 0),
        "manual_review_queue_size": sum(1 for sample in samples if sample.get("manual_review")),
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def build_manual_review_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in report.get("samples", []):
        if not sample.get("manual_review"):
            continue
        sample_issues = sample.get("issues", [])
        object_rows_written = False
        for obj in sample.get("objects", []):
            object_issues = obj.get("issues", [])
            if sample.get("manual_review_reason") == "sampling" and not object_issues:
                continue
            if not object_issues and not sample_issues:
                continue
            rows.append(_manual_row(sample, obj, sample_issues + object_issues))
            object_rows_written = True
        if not object_rows_written:
            rows.append(_manual_row(sample, None, sample_issues))
    return rows


def _manual_row(
    sample: dict[str, Any],
    obj: dict[str, Any] | None,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    row = {
        "sample_id": sample.get("sample_id"),
        **_flatten_sample_context(sample),
        "object_id": obj.get("object_id") if obj else "",
        **_flatten_object_context(obj),
        "status": obj.get("status") if obj else sample.get("status"),
        "risk_score": obj.get("risk_score") if obj else sample.get("risk_score"),
        "issue_codes": ";".join(str(issue.get("code")) for issue in issues),
        "issue_field_paths": ";".join(str(issue.get("field_path") or "") for issue in issues),
        "issue_messages": " | ".join(str(issue.get("message")) for issue in issues),
        "metadata_uri": sample.get("metadata_uri"),
        "image_uri": sample.get("image_uri"),
        "resolved_image_uri": sample.get("resolved_image_uri"),
        "crop_uri": obj.get("crop_uri") if obj else "",
        "resolved_crop_uri": obj.get("resolved_crop_uri") if obj else "",
        "manual_review_reason": sample.get("manual_review_reason"),
    }
    return row


def write_qc_outputs(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    run_id = report["qc_run_id"]
    report_path = target_dir / f"{run_id}_report.json"
    queue_path = target_dir / f"{run_id}_manual_review_queue.csv"
    write_json(report_path, report)
    write_csv(queue_path, build_manual_review_rows(report), QC_REPORT_FIELDS)
    return report_path, queue_path


def run_qc_agent(
    *,
    pipeline_config: dict[str, Any],
    metadata_dir: str | Path | None = None,
    sample_paths: list[str | Path] | None = None,
    output_dir: str | Path | None = None,
    asset_base_dirs: list[str | Path] | None = None,
    sampling_ratio: float | None = None,
    enable_vlm: bool | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    qc_cfg = build_qc_agent_config(pipeline_config)
    if output_dir is not None:
        qc_cfg["output_dir"] = str(output_dir)
    if sampling_ratio is not None:
        qc_cfg["manual_sampling_ratio"] = float(sampling_ratio)
    if enable_vlm is not None:
        qc_cfg.setdefault("vlm_review", {})["enabled"] = bool(enable_vlm)

    paths = collect_metadata_paths(metadata_dir, sample_paths, limit=limit)
    agent = QCAnnotationAgent(qc_cfg, asset_base_dirs=asset_base_dirs)
    samples = [agent.review_metadata_path(path) for path in paths]
    _apply_manual_sampling(
        samples,
        ratio=float(qc_cfg.get("manual_sampling_ratio", 0.0) or 0.0),
        seed=int(qc_cfg.get("random_seed", 20260601) or 20260601),
    )
    run_id = f"qc_{now_iso_shanghai().replace(':', '').replace('+', '_').replace('-', '')}"
    report = {
        "qc_run_id": run_id,
        "created_time": now_iso_shanghai(),
        "metadata_dir": str(metadata_dir) if metadata_dir else None,
        "sample_paths": [str(path) for path in sample_paths or []],
        "qc_config": {
            "manual_sampling_ratio": qc_cfg.get("manual_sampling_ratio"),
            "random_seed": qc_cfg.get("random_seed"),
            "vlm_review_enabled": bool_config(qc_cfg.get("vlm_review", {}).get("enabled"), default=False),
        },
        "summary": summarize_qc_report(samples),
        "samples": samples,
    }
    report_path, queue_path = write_qc_outputs(report, qc_cfg.get("output_dir", "data/qc"))
    report["report_path"] = str(report_path)
    report["manual_review_queue_path"] = str(queue_path)
    write_json(report_path, report)
    return report
