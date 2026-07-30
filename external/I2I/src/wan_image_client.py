from __future__ import annotations

import base64
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import requests

from config import DashScopeConfig, I2IServiceConfig, ModelServiceConfig
from prompts import NEGATIVE_PROMPT
from utils import cv2_imread, cv2_imwrite, image_to_data_url, redact_headers, write_json

logger = logging.getLogger(__name__)

DEFAULT_SEEDREAM_MIN_PIXELS = 3_686_400
DEFAULT_SEEDREAM_SIZE = "2K"
REFERENCE_GENERATION_ERROR = (
    "Seedream endpoint appears to be reference-generation-only and is not allowed for production local editing. "
    "Configure a real Seedream local edit/inpaint endpoint or set SEEDREAM_ALLOW_REFERENCE_GENERATION_DEBUG=1 for debug-only experiments."
)


def _save_base64_image(data: str, output_path: str) -> None:
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    image_bytes = base64.b64decode(data)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(image_bytes)


def _extract_image_url_or_base64(response: dict[str, Any]) -> str:
    def walk(value: Any) -> str | None:
        if isinstance(value, dict):
            for key in ("image", "image_url", "url", "image_base64", "b64_json"):
                item = value.get(key)
                if isinstance(item, str) and item:
                    return item
            for item in value.values():
                found = walk(item)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = walk(item)
                if found:
                    return found
        elif isinstance(value, str) and (
            value.startswith("http://")
            or value.startswith("https://")
            or value.startswith("data:image/")
        ):
            return value
        return None

    found = walk(response)
    if found:
        return found
    output = response.get("output", {})
    for key in ("image_url", "url", "image", "image_base64", "b64_json"):
        value = output.get(key)
        if isinstance(value, str) and value:
            return value
    results = output.get("results") or output.get("images")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            for key in ("url", "image_url", "image", "image_base64", "b64_json"):
                value = first.get(key)
                if isinstance(value, str) and value:
                    return value
    raise ValueError(f"unable to extract generated image from response: {response}")


def _download_or_decode_image(value: str, output_path: str) -> None:
    if value.startswith("http://") or value.startswith("https://"):
        resp = requests.get(value, timeout=180)
        resp.raise_for_status()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(resp.content)
    else:
        _save_base64_image(value, output_path)


def _clip_bbox_to_image(bbox: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x1, y1, x2, y2 = bbox
    clipped = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"empty edit bbox after clipping: {bbox}")
    return clipped


def _paste_generated_crop(
    original_image_path: str,
    generated_crop_path: str,
    output_path: str,
    bbox: tuple[int, int, int, int],
) -> None:
    with Image.open(original_image_path) as original_image:
        original = original_image.convert("RGB")
    x1, y1, x2, y2 = _clip_bbox_to_image(bbox, original.size)
    target_size = (x2 - x1, y2 - y1)
    with Image.open(generated_crop_path) as crop_image:
        generated_crop = crop_image.convert("RGB")
        if generated_crop.size != target_size:
            generated_crop = generated_crop.resize(target_size, Image.Resampling.LANCZOS)
    composed = original.copy()
    composed.paste(generated_crop, (x1, y1))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path)


def _looks_like_framed_scene(image_path: str | Path) -> bool:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width < 32 or height < 32:
            return False
        border = max(4, min(width, height) // 40)
        gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    inner = gray[border : height - border, border : width - border]
    if inner.size == 0:
        return False
    border_dark_ratio = float((gray[border_mask] < 28).mean())
    inner_dark_ratio = float((inner < 28).mean())
    return border_dark_ratio >= 0.65 and inner_dark_ratio <= 0.35


def _dry_run_edit(image_path: str, bbox: tuple[int, int, int, int], anomaly_type: str, output_path: str) -> None:
    image = cv2_imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    center = (x1 + int(w * 0.52), y1 + int(h * 0.62))
    axes = (max(12, int(w * 0.12)), max(8, int(h * 0.06)))
    overlay = image.copy()
    if anomaly_type == "oil_leak":
        color = (20, 18, 16)
        alpha = 0.62
    elif anomaly_type == "water_leak":
        color = (210, 235, 245)
        alpha = 0.32
    elif anomaly_type == "coolant_leak":
        color = (150, 210, 120)
        alpha = 0.45
    else:
        color = (80, 190, 225)
        alpha = 0.38
    cv2.ellipse(overlay, center, axes, -12, 0, 360, color, -1)
    if anomaly_type != "oil_leak":
        cv2.line(
            overlay,
            (center[0] - axes[0] // 2, max(y1, center[1] - axes[1] * 4)),
            center,
            color,
            max(2, min(w, h) // 80),
        )
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2_imwrite(output_path, image)


def _service_endpoint(config: ModelServiceConfig | DashScopeConfig) -> str:
    endpoint = getattr(config, "endpoint", None) or getattr(config, "wan_endpoint", None)
    if not endpoint:
        raise ValueError("image generation endpoint is required")
    return str(endpoint)


def _seedream_openai_base_url(endpoint: str) -> str:
    override = os.getenv("SEEDREAM_OPENAI_BASE_URL", "").strip()
    if override:
        return override.rstrip("/")
    normalized = endpoint.rstrip("/") or "https://ark.cn-beijing.volces.com/api/v3"
    suffix = "/images/generations"
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]
    if normalized.endswith("/api/plan/v3"):
        normalized = normalized[: -len("/api/plan/v3")] + "/api/v3"
    return normalized


def _is_seedream_provider(config: ModelServiceConfig | DashScopeConfig, endpoint: str, model: str) -> bool:
    provider = str(getattr(config, "provider", "") or "").lower()
    if provider in {"volcengine_ark", "ark", "seedream"}:
        return True
    marker = f"{endpoint} {model}".lower()
    return "seedream" in marker or "ark.cn-beijing.volces.com" in marker or "/images/generations" in marker


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _is_seedream_reference_generation_endpoint(endpoint: str) -> bool:
    normalized = endpoint.lower().rstrip("/")
    return normalized.endswith("/images/generations") or "/images/generations" in normalized


VALID_SEEDREAM_EXPERIMENT_MODES = {
    "single_image_edit",
    "boxed_single_edit",
    "boxed_fusion",
    "cabinet_door_open",
}


def _allow_seedream_reference_generation_debug() -> bool:
    return _truthy_env("SEEDREAM_ALLOW_REFERENCE_GENERATION_DEBUG")


def validate_seedream_local_edit_capability(
    endpoint: str,
    *,
    dry_run: bool = False,
    seedream_mode: str | None = None,
) -> None:
    if dry_run:
        return
    if seedream_mode in VALID_SEEDREAM_EXPERIMENT_MODES:
        return
    if _is_seedream_reference_generation_endpoint(endpoint) and not _allow_seedream_reference_generation_debug():
        raise RuntimeError(REFERENCE_GENERATION_ERROR)


def _seedream_prompt(
    prompt: str,
    bbox: tuple[int, int, int, int],
    *,
    seedream_mode: str | None = None,
    reference_count: int = 1,
    anomaly_type: str | None = None,
) -> str:
    x1, y1, x2, y2 = bbox
    if seedream_mode == "cabinet_door_open":
        return (
            "Edit the input image itself. Open exactly one currently closed equipment-cabinet door whose closed "
            f"door leaf is inside pixel bbox [{x1}, {y1}, {x2}, {y2}]. Open it naturally by about 45-70 degrees "
            "around its existing hinge. Preserve the same cabinet identity, door material, color, thickness, "
            "handle, hinge geometry, perspective, lighting, camera position, timestamp, resolution, and all "
            "surrounding equipment. Reveal only a plausible dark cabinet interior directly behind this door. "
            "The opened door may extend immediately to the left or right of the bbox according to its hinge, "
            "but do not alter any other cabinet door or object. Return one full-frame edited CCTV image. "
            "No second scene, no inset, no pasted rectangle, no border, no duplicated cabinet, no detached or "
            "floating door, no people, and no new text."
        )
    if seedream_mode == "boxed_single_edit":
        return (
            f"Use the input image as the only source scene. The red rectangle marks the allowed edit region "
            f"[{x1}, {y1}, {x2}, {y2}]. Add one subtle natural clear-water leak stain inside that region, "
            "roughly 80-160 px across. Do not fill the whole rectangle. Feather the edges into the floor or "
            "nearby equipment surface. Remove the red rectangle in the final image. Preserve the camera, "
            "timestamp, equipment, background, lighting, and all pixels outside the rectangle as much as possible. "
            "No inset image, no pasted panel, no border, no duplicated scene."
        )
    if seedream_mode == "single_image_edit":
        return (
            f"Use the input image as the only source scene. Add one subtle natural clear-water leak stain near "
            f"pixel region [{x1}, {y1}, {x2}, {y2}], roughly 80-160 px across. Preserve the original camera, "
            "timestamp, equipment positions, background, lighting, and scene layout. Do not redraw the room, "
            "do not change viewpoint, and do not create a pasted patch or duplicated scene."
        )
    if seedream_mode == "boxed_fusion":
        return (
            f"Use image 1 as the only source scene. Use image 2 only as a water-stain texture reference. "
            f"The red rectangle in image 1 is a placement guide for region [{x1}, {y1}, {x2}, {y2}]. "
            "Create one realistic clear-water leak stain inside the red rectangle, roughly 80-160 px across. "
            "Do not cover or repaint the whole rectangle. The stain should be transparent, low contrast, "
            "irregular, with wet reflections and feathered edges that blend into the original floor or nearby "
            "equipment surface. Remove the red rectangle completely in the final image. Preserve image 1's "
            "camera view, timestamp, equipment positions, background, lighting, and scene layout. Return one "
            "full-size edited source-scene image. No second image, no inset, no pasted panel, no border, no "
            "visible layer, no duplicated room, no new scene."
        )
    return (
        f"{prompt}\n\n"
        "Edit region constraint: use the clean input image as the source image and edit only inside "
        f"pixel bbox [x1={x1}, y1={y1}, x2={x2}, y2={y2}]. Keep the camera, background, equipment, "
        "lighting, timestamp, and all content outside this box unchanged. The final anomaly must be a "
        "small realistic early-stage leak located inside this box."
    )


def _seedream_anomaly_detail(anomaly_type: str | None) -> str:
    if anomaly_type == "water_leak":
        return (
            "异常目标：仅生成清水漏水水渍。不要生成油污、泡沫、蒸汽、喷射水柱、人员、车辆、新设备、新房间、新时间戳或场景级变化。"
        )
    return (
        "Target anomaly: one small realistic industrial anomaly only inside the specified bbox. Keep all non-target "
        "content unchanged and avoid any scene-level change."
    )


def _seedream_image_field(seedream_mode: str | None, image_count: int) -> str:
    override = os.getenv("SEEDREAM_IMAGE_FIELD", "").strip()
    if override:
        return override
    if seedream_mode == "boxed_fusion" or image_count > 1:
        return "image"
    return "image"


def _with_seedream_image_input(
    payload: dict[str, Any],
    image_data_urls: list[str],
    *,
    seedream_mode: str | None = None,
) -> dict[str, Any]:
    field = _seedream_image_field(seedream_mode, len(image_data_urls))
    if field in {"image", "image_url"}:
        payload[field] = image_data_urls if len(image_data_urls) > 1 else image_data_urls[0]
    elif field in {"images", "image_urls"}:
        payload[field] = image_data_urls
    else:
        payload[field] = image_data_urls if len(image_data_urls) > 1 else image_data_urls[0]
    return payload


def _seedream_payload_model(model: str, seedream_mode: str | None) -> str:
    override = os.getenv("SEEDREAM_IMAGE_MODEL", "").strip()
    if override:
        return override
    if seedream_mode in {"single_image_edit", "boxed_single_edit", "cabinet_door_open"}:
        return os.getenv("SEEDREAM_SINGLE_IMAGE_MODEL", model).strip() or model
    return model


def _response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    data = getattr(response, "data", None)
    if data is not None:
        items = []
        for item in data:
            if hasattr(item, "model_dump"):
                items.append(item.model_dump())
            elif hasattr(item, "dict"):
                items.append(item.dict())
            else:
                items.append({key: getattr(item, key) for key in ("url", "b64_json") if getattr(item, key, None)})
        return {"data": items}
    raise TypeError(f"unsupported image generation response type: {type(response)!r}")


def _generate_seedream_with_openai_sdk(endpoint: str, api_key: str, request_payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai is required for Seedream image generation.") from exc

    payload = dict(request_payload)
    model = str(payload.pop("model"))
    prompt = str(payload.pop("prompt"))
    size = str(payload.pop("size", DEFAULT_SEEDREAM_SIZE))
    n = int(payload.pop("n", 1))
    response_format = str(payload.pop("response_format", os.getenv("SEEDREAM_RESPONSE_FORMAT", "url")))
    client = OpenAI(base_url=_seedream_openai_base_url(endpoint), api_key=api_key)
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        n=n,
        response_format=response_format,
        extra_body=payload,
    )
    return _response_to_dict(response)


def _is_seedream_auth_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 401:
        return True
    message = str(exc).lower()
    return "401" in message and ("unauthorized" in message or "authentication" in message or "api key" in message)


def _seedream_auth_error_message(config: ModelServiceConfig | DashScopeConfig) -> str:
    env_name = getattr(config, "api_key_env", None) or "ARK_API_KEY"
    return (
        "Seedream authentication failed: Ark returned HTTP 401. "
        f"Check that {env_name} is set to a valid Volcengine Ark API key for the current shell. "
        "If you use an alias, SEEDREAM_API_KEY is also supported by the bundled config."
    )


def _seedream_size_from_env() -> str | None:
    size = os.getenv("SEEDREAM_SIZE", "").strip()
    if not size:
        return None
    if size.lower() == "auto":
        return "auto"
    normalized = size.lower()
    if normalized in {"2k", "3k", "4k"}:
        return size
    parts = normalized.split("x", 1)
    if len(parts) == 2 and all(part.isdigit() and int(part) > 0 for part in parts):
        return normalized
    raise ValueError("SEEDREAM_SIZE must be one of WIDTHxHEIGHT, 2k, 3k, or 4k")


def _seedream_min_pixels() -> int:
    raw = os.getenv("SEEDREAM_MIN_PIXELS", "").strip()
    if not raw:
        return DEFAULT_SEEDREAM_MIN_PIXELS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("SEEDREAM_MIN_PIXELS must be a positive integer") from exc
    if value <= 0:
        raise ValueError("SEEDREAM_MIN_PIXELS must be a positive integer")
    return value


def _scale_size_to_min_pixels(width: int, height: int, min_pixels: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid Seedream image size: {width}x{height}")
    if width * height >= min_pixels:
        return width, height

    scale = math.sqrt(min_pixels / float(width * height))
    scaled_width = max(1, math.ceil(width * scale))
    scaled_height = max(1, math.ceil(height * scale))
    while scaled_width * scaled_height < min_pixels:
        if scaled_width / width <= scaled_height / height:
            scaled_width += 1
        else:
            scaled_height += 1
    return scaled_width, scaled_height


def _seedream_size_for_image(image_path: str | Path) -> str:
    override = _seedream_size_from_env()
    if override and override != "auto":
        return override
    default_size = os.getenv("SEEDREAM_DEFAULT_SIZE", DEFAULT_SEEDREAM_SIZE).strip()
    if not override and default_size:
        return default_size
    with Image.open(image_path) as image:
        width, height = image.size
    width, height = _scale_size_to_min_pixels(width, height, _seedream_min_pixels())
    return f"{width}x{height}"


class WanImageClient:
    def __init__(self, model: str, dashscope_config: ModelServiceConfig | DashScopeConfig, dry_run: bool = False):
        self.model = model
        self.config = dashscope_config
        self.dry_run = dry_run

    def _build_request_payloads(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str,
        bbox: tuple[int, int, int, int],
        seedream_mode: str | None = None,
        seedream_reference_paths: list[str] | None = None,
        anomaly_type: str | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, Any], bool]:
        endpoint = _service_endpoint(self.config)
        full_prompt = prompt
        if negative_prompt:
            full_prompt = f"{prompt}\n\nNegative prompt: {negative_prompt}"

        if _is_seedream_provider(self.config, endpoint, self.model):
            reference_paths = [
                str(path) for path in (seedream_reference_paths or []) if str(path) and seedream_mode == "boxed_fusion"
            ]
            request_image_paths = [image_path, *reference_paths]
            seedream_prompt = _seedream_prompt(
                full_prompt,
                bbox,
                seedream_mode=seedream_mode,
                reference_count=len(request_image_paths),
                anomaly_type=anomaly_type,
            )
            request_payload: dict[str, Any] = {
                "model": _seedream_payload_model(self.model, seedream_mode),
                "prompt": seedream_prompt,
                "n": 1,
                "output_format": os.getenv("SEEDREAM_OUTPUT_FORMAT", "png"),
                "watermark": False,
            }
            request_payload["response_format"] = os.getenv("SEEDREAM_RESPONSE_FORMAT", "url")
            request_payload["size"] = _seedream_size_for_image(image_path)
            request_payload = _with_seedream_image_input(
                request_payload,
                [image_to_data_url(path) for path in request_image_paths],
                seedream_mode=seedream_mode,
            )

            request_log_payload = dict(request_payload)
            for image_key in ("image", "image_url"):
                if image_key in request_log_payload:
                    request_log_payload[image_key] = (
                        request_image_paths if isinstance(request_payload.get(image_key), list) else image_path
                    )
            for image_key in ("images", "image_urls"):
                if image_key in request_log_payload:
                    request_log_payload[image_key] = request_image_paths
            request_log_payload["edit_bbox"] = list(bbox)
            if seedream_mode:
                request_log_payload["seedream_mode"] = seedream_mode
                request_log_payload["experimental_seedream"] = True
            return endpoint, request_payload, request_log_payload, True

        request_payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": image_to_data_url(image_path)},
                            {"text": full_prompt},
                        ],
                    }
                ]
            },
            "parameters": {
                "bbox_list": [[list(bbox)]],
                "n": 1,
                "watermark": False,
                "thinking_mode": True,
            },
        }
        request_log_payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image_path": image_path},
                            {"text": full_prompt},
                        ],
                    }
                ]
            },
            "parameters": {
                "bbox_list": [[list(bbox)]],
                "n": 1,
                "watermark": False,
                "thinking_mode": True,
            },
        }
        return endpoint, request_payload, request_log_payload, False

    def edit_image_with_wan(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str,
        bbox: tuple[int, int, int, int],
        output_path: str,
        anomaly_type: str,
        request_log_path: str | None = None,
        response_log_path: str | None = None,
        seedream_mode: str | None = None,
        seedream_reference_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        endpoint = _service_endpoint(self.config)
        is_seedream = _is_seedream_provider(self.config, endpoint, self.model)
        self.model_call_count = getattr(self, "model_call_count", 0)
        self.model_generated_count = getattr(self, "model_generated_count", 0)
        request_payload: dict[str, Any]
        request_log_payload: dict[str, Any]
        if not (is_seedream and not self.dry_run):
            endpoint, request_payload, request_log_payload, is_seedream = self._build_request_payloads(
                image_path,
                prompt,
                negative_prompt,
                bbox,
                seedream_mode=seedream_mode,
                seedream_reference_paths=seedream_reference_paths,
                anomaly_type=anomaly_type,
            )
        else:
            request_payload = {}
            request_log_payload = {}
        if request_log_path:
            write_json(
                request_log_path,
                {"endpoint": endpoint, "provider": self.config.provider, "body": request_log_payload},
            )

        if self.dry_run:
            _dry_run_edit(image_path, bbox, anomaly_type, output_path)
            response = {"dry_run": True, "output_path": output_path, "bbox_list": [[list(bbox)]]}
            if response_log_path:
                write_json(response_log_path, response)
            return response

        if not self.config.api_key:
            raise RuntimeError("image generation API key is required unless --dry-run is used; set ARK_API_KEY")

        if is_seedream:
            validate_seedream_local_edit_capability(endpoint, dry_run=self.dry_run, seedream_mode=seedream_mode)

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
        request_image_path = image_path
        request_bbox = bbox
        compose_seedream_crop = False
        seedream_crop_bbox: tuple[int, int, int, int] | None = None
        if is_seedream and seedream_mode not in VALID_SEEDREAM_EXPERIMENT_MODES:
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="seedream_i2i_")
            temp_dir = Path(temp_dir_obj.name)
            with Image.open(image_path) as original_image:
                original = original_image.convert("RGB")
                seedream_crop_bbox = _clip_bbox_to_image(bbox, original.size)
                crop = original.crop(seedream_crop_bbox)
            request_image_path = str(temp_dir / "seedream_input_crop.jpg")
            crop.save(request_image_path, quality=95)
            request_bbox = (0, 0, crop.size[0], crop.size[1])
            endpoint, request_payload, request_log_payload, is_seedream = self._build_request_payloads(
                request_image_path,
                prompt,
                negative_prompt,
                request_bbox,
                seedream_mode=seedream_mode,
                seedream_reference_paths=seedream_reference_paths,
                anomaly_type=anomaly_type,
            )
            request_log_payload["source_image_path"] = image_path
            request_log_payload["source_edit_bbox"] = list(seedream_crop_bbox)
            request_log_payload["seedream_local_crop_mode"] = True
            compose_seedream_crop = True
        elif is_seedream:
            endpoint, request_payload, request_log_payload, is_seedream = self._build_request_payloads(
                request_image_path,
                prompt,
                negative_prompt,
                request_bbox,
                seedream_mode=seedream_mode,
                seedream_reference_paths=seedream_reference_paths,
                anomaly_type=anomaly_type,
            )
            request_log_payload["source_image_path"] = image_path
            request_log_payload["seedream_full_image_experiment"] = True

        if request_log_path:
            write_json(
                request_log_path,
                {
                    "endpoint": endpoint,
                    "provider": self.config.provider,
                    "headers": redact_headers(headers),
                    "body": request_log_payload,
                },
            )
        try:
            self.model_call_count += 1
            if is_seedream:
                try:
                    raw = _generate_seedream_with_openai_sdk(endpoint, self.config.api_key, request_payload)
                except Exception as exc:
                    if _is_seedream_auth_error(exc):
                        auth_error = {
                            "error": "seedream_authentication_failed",
                            "status_code": 401,
                            "message": _seedream_auth_error_message(self.config),
                            "api_key_env": getattr(self.config, "api_key_env", None),
                            "original_error": str(exc),
                        }
                        if response_log_path:
                            write_json(response_log_path, auth_error)
                        raise RuntimeError(auth_error["message"]) from exc
                    raise
            else:
                response = requests.post(endpoint, headers=headers, json=request_payload, timeout=180)
                try:
                    raw = response.json()
                except Exception:
                    raw = {"status_code": response.status_code, "text": response.text}
                if not response.ok:
                    if response_log_path:
                        write_json(response_log_path, raw)
                    raise RuntimeError(f"image generation request failed: HTTP {response.status_code}")

            final_response = self._resolve_async_if_needed(raw, headers, is_seedream)
            image_value = _extract_image_url_or_base64(final_response)
            download_target = output_path
            if compose_seedream_crop:
                download_target = str(Path(temp_dir_obj.name) / "seedream_generated_crop.png")  # type: ignore[union-attr]
            _download_or_decode_image(image_value, download_target)
            self.model_generated_count += 1
            if compose_seedream_crop:
                if seedream_crop_bbox is None:
                    raise RuntimeError("missing Seedream crop bbox for local composition")
                if _looks_like_framed_scene(download_target):
                    raise RuntimeError("Seedream generated crop appears to contain a framed full-scene image; rejecting debug reference-generation output")
                _paste_generated_crop(image_path, download_target, output_path, seedream_crop_bbox)
                final_response.setdefault("local_edit", {})
                final_response["local_edit"].update(
                    {
                        "mode": "crop_then_paste",
                        "source_edit_bbox": list(seedream_crop_bbox),
                        "request_bbox": list(request_bbox),
                        "output_path": output_path,
                    }
                )
        finally:
            if temp_dir_obj is not None:
                temp_dir_obj.cleanup()
        if response_log_path:
            write_json(response_log_path, final_response)
        return final_response

    def _resolve_async_if_needed(
        self,
        raw: dict[str, Any],
        headers: dict[str, str],
        is_seedream: bool = False,
    ) -> dict[str, Any]:
        task_id = raw.get("output", {}).get("task_id") or raw.get("task_id")
        if not task_id:
            return raw
        if is_seedream:
            template = os.getenv("SEEDREAM_TASK_URL_TEMPLATE")
            if not template:
                return raw
            task_url = template.format(task_id=task_id)
            for _ in range(60):
                time.sleep(5)
                poll = requests.get(task_url, headers=headers, timeout=60)
                data = poll.json()
                status = data.get("status") or data.get("output", {}).get("task_status") or data.get("task_status")
                if status in {"SUCCEEDED", "succeeded", "SUCCESS", "success", "completed"}:
                    return data
                if status in {"FAILED", "failed", "CANCELED", "canceled", "cancelled"}:
                    raise RuntimeError(f"Seedream async task failed: {data}")
            raise TimeoutError(f"Seedream async task timed out: {task_id}")

        endpoint = _service_endpoint(self.config)
        base = endpoint.split("/api/v1/", 1)[0]
        task_url = f"{base}/api/v1/tasks/{task_id}"
        for _ in range(60):
            time.sleep(5)
            poll = requests.get(task_url, headers=headers, timeout=60)
            data = poll.json()
            status = data.get("output", {}).get("task_status") or data.get("task_status")
            if status in {"SUCCEEDED", "succeeded", "SUCCESS"}:
                return data
            if status in {"FAILED", "failed", "CANCELED", "canceled"}:
                raise RuntimeError(f"image generation async task failed: {data}")
        raise TimeoutError(f"image generation async task timed out: {task_id}")


def edit_image_with_wan(
    image_path: str,
    prompt: str,
    negative_prompt: str,
    bbox: tuple[int, int, int, int],
    output_path: str,
) -> dict[str, Any]:
    client = WanImageClient("doubao-seedream-5.0-lite", I2IServiceConfig.from_env().image)
    return client.edit_image_with_wan(image_path, prompt, negative_prompt or NEGATIVE_PROMPT, bbox, output_path, "")
