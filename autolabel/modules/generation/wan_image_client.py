from __future__ import annotations

import base64
import json
import os
import random
import shutil
import socket
import ssl
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager

from PIL import Image, ImageDraw

from autolabel.utils import image_to_data_url

from .grid import BBox, clip_bbox


@dataclass(frozen=True)
class WanGenerationResult:
    output_path: Path
    raw_response: dict[str, Any]


class WanImageClient:
    def __init__(
        self,
        model_name: str = "wan2.7-image-pro",
        dry_run: bool = False,
        api_key: str | None = None,
        request_timeout_seconds: int = 300,
        poll_interval_seconds: float = 2.0,
        max_poll_seconds: int = 300,
        max_retries: int = 3,
        use_async_api: bool | None = None,
        submit_semaphore: Any | None = None,
        poll_semaphore: Any | None = None,
        download_semaphore: Any | None = None,
        stage_timer: Callable[[str], ContextManager[Any]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.dry_run = dry_run
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("WAN_IMAGE_API_KEY")
        self.endpoint = os.getenv("DASHSCOPE_WAN_ENDPOINT") or os.getenv("WAN_IMAGE_EDIT_ENDPOINT")
        if self.endpoint:
            self.endpoint = "".join(str(self.endpoint).split())
        self.request_timeout_seconds = request_timeout_seconds
        self.poll_interval_seconds = max(0.5, float(poll_interval_seconds))
        self.max_poll_seconds = max(1, int(max_poll_seconds))
        self.max_retries = max(0, int(max_retries))
        self.submit_semaphore = submit_semaphore
        self.poll_semaphore = poll_semaphore
        self.download_semaphore = download_semaphore
        self.stage_timer = stage_timer
        if use_async_api is None:
            env_value = str(os.getenv("WAN_USE_ASYNC_API", "")).strip().lower()
            use_async_api = env_value in {"1", "true", "yes", "on"} or bool(
                self.endpoint and "/image-generation/generation" in self.endpoint
            )
        self.use_async_api = bool(use_async_api)

    def is_configured(self) -> tuple[bool, str | None]:
        if self.dry_run:
            return True, None
        if not self.api_key:
            return False, "Wan API key is not configured. Set DASHSCOPE_API_KEY or WAN_IMAGE_API_KEY."
        if not self.endpoint:
            return False, "Wan endpoint is not configured. Set DASHSCOPE_WAN_ENDPOINT or WAN_IMAGE_EDIT_ENDPOINT."
        return True, None

    def generate(
        self,
        original_image_path: str | Path,
        bbox_list: list[BBox],
        prompt: str,
        negative_prompt: str,
        output_path: str | Path,
        response_path: str | Path,
        anomaly_type: str,
    ) -> WanGenerationResult:
        output = Path(output_path)
        response_output = Path(response_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        response_output.parent.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            with self._stage("wan_submit"):
                raw = self._generate_dry_run(original_image_path, bbox_list, output, anomaly_type)
        else:
            raw = self._generate_live(original_image_path, bbox_list, prompt, negative_prompt, output)
            self._validate_downloaded_image(output)
        response_output.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        return WanGenerationResult(output_path=output, raw_response=raw)

    def _generate_live(
        self,
        original_image_path: str | Path,
        bbox_list: list[BBox],
        prompt: str,
        negative_prompt: str,
        output_path: Path,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Wan API key is not configured. Use --dry-run or set DASHSCOPE_API_KEY/WAN_IMAGE_API_KEY.")
        if not self.endpoint:
            raise RuntimeError(
                "WAN_IMAGE_EDIT_ENDPOINT/DASHSCOPE_WAN_ENDPOINT is not configured. "
                "This wrapper expects an endpoint returning image_base64, image_url, or local_path; use --dry-run to validate locally."
            )
        if "dashscope" in self.endpoint and (
            "/multimodal-generation/generation" in self.endpoint or "/image-generation/generation" in self.endpoint
        ):
            return self._generate_dashscope_official(original_image_path, bbox_list, prompt, negative_prompt, output_path)

        with open(original_image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
        request_payload = {
            "model": self.model_name,
            "image_base64": image_b64,
            "bbox_list": [list(map(int, bbox)) for bbox in bbox_list],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with self._stage("wan_submit"):
            with _maybe_semaphore(self.submit_semaphore):
                payload = self._urlopen_json(request)

        image_b64_response = payload.get("image_base64") or payload.get("generated_image_base64")
        if image_b64_response:
            with self._stage("download"):
                output_path.write_bytes(base64.b64decode(image_b64_response))
        elif payload.get("local_path"):
            with self._stage("download"):
                shutil.copyfile(payload["local_path"], output_path)
        elif payload.get("image_url"):
            self.download_image(payload["image_url"], output_path)
        else:
            raise RuntimeError("Wan endpoint response did not include image_base64, image_url, or local_path.")
        payload["saved_output_path"] = str(output_path)
        payload["input_was_clean_original"] = True
        return payload

    def _generate_dashscope_official(
        self,
        original_image_path: str | Path,
        bbox_list: list[BBox],
        prompt: str,
        negative_prompt: str,
        output_path: Path,
    ) -> dict[str, Any]:
        with Image.open(original_image_path) as source:
            width, height = source.size
        image_data_url = image_to_data_url(original_image_path, max_side=None, jpeg_quality=95)
        prompt_with_negative = f"{prompt}\n\nNegative constraints: {negative_prompt}"
        request_payload = {
            "model": self.model_name,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": image_data_url},
                            {"text": prompt_with_negative},
                        ],
                    }
                ]
            },
            "parameters": {
                "bbox_list": [[list(map(int, bbox)) for bbox in bbox_list]],
                "size": f"{width}*{height}",
                "n": 1,
                "watermark": False,
                "thinking_mode": False,
            },
        }
        if self.use_async_api:
            payload = self.submit_task(request_payload)
            task_id = self._extract_task_id(payload)
            if not task_id:
                raise RuntimeError(f"DashScope Wan async submit response did not include task_id: {payload}")
            final_payload = self.poll_task(task_id)
            image_url = self._extract_dashscope_image_url(final_payload)
            if not image_url:
                raise RuntimeError(f"DashScope Wan async result did not include an output image URL: {final_payload}")
            self.download_image(image_url, output_path)
            final_payload["submit_response"] = payload
            final_payload["task_id"] = task_id
            payload = final_payload
        else:
            payload = self.submit_sync(request_payload)
            image_url = self._extract_dashscope_image_url(payload)
            if not image_url:
                raise RuntimeError(f"DashScope Wan response did not include an output image URL: {payload}")
            self.download_image(image_url, output_path)
        payload["saved_output_path"] = str(output_path)
        payload["input_was_clean_original"] = True
        payload["request_bbox_list"] = request_payload["parameters"]["bbox_list"]
        payload["request_size"] = request_payload["parameters"]["size"]
        return payload

    def submit_sync(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with self._stage("wan_submit"):
            with _maybe_semaphore(self.submit_semaphore):
                return self._urlopen_json(request)

    def submit_task(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "X-DashScope-Async": "enable",
            },
            method="POST",
        )
        with self._stage("wan_submit"):
            with _maybe_semaphore(self.submit_semaphore):
                return self._urlopen_json(request)

    def poll_task(self, task_id: str) -> dict[str, Any]:
        with self._stage("wan_poll"):
            if not self.endpoint:
                raise RuntimeError("Wan endpoint is not configured.")
            base = self.endpoint.split("/services/", 1)[0].rstrip("/")
            task_url = f"{base}/tasks/{task_id}"
            deadline = time.monotonic() + self.max_poll_seconds
            last_payload: dict[str, Any] = {}
            while time.monotonic() < deadline:
                request = urllib.request.Request(
                    task_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    method="GET",
                )
                with _maybe_semaphore(self.poll_semaphore):
                    payload = self._urlopen_json(request)
                last_payload = payload
                status = str((payload.get("output") or {}).get("task_status") or payload.get("task_status") or "").upper()
                if status in {"SUCCEEDED", "SUCCESS", "FINISHED", "COMPLETED"}:
                    return payload
                if status in {"FAILED", "ERROR", "CANCELED", "CANCELLED"}:
                    raise RuntimeError(f"DashScope Wan task {task_id} failed: {payload}")
                time.sleep(self.poll_interval_seconds)
            raise TimeoutError(f"DashScope Wan task {task_id} timed out after {self.max_poll_seconds}s: {last_payload}")

    def download_image(self, image_url: str, output_path: str | Path) -> None:
        with self._stage("download"):
            with _maybe_semaphore(self.download_semaphore):
                self._download_url(image_url, Path(output_path))

    @staticmethod
    def _extract_dashscope_image_url(payload: dict[str, Any]) -> str | None:
        output = payload.get("output") or {}
        for choice in output.get("choices") or []:
            message = choice.get("message") or {}
            for item in message.get("content") or []:
                if isinstance(item, dict) and item.get("image"):
                    return str(item["image"])
        for result in output.get("results") or []:
            if isinstance(result, dict) and result.get("url"):
                return str(result["url"])
        return None

    @staticmethod
    def _extract_task_id(payload: dict[str, Any]) -> str | None:
        output = payload.get("output") or {}
        task_id = output.get("task_id") or payload.get("task_id")
        return str(task_id) if task_id else None

    def _urlopen_json(self, request: urllib.request.Request) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:  # noqa: S310
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = RuntimeError(f"Wan HTTP {exc.code}: {body}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise last_error from exc
            except (TimeoutError, socket.timeout, ssl.SSLError, urllib.error.URLError) as exc:
                last_error = RuntimeError(f"Wan network error: {exc}")
                if attempt >= self.max_retries:
                    raise last_error from exc
            time.sleep(min(10.0, 0.6 * (2**attempt)))
        raise RuntimeError(f"Wan request failed: {last_error}")

    def _download_url(self, image_url: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(image_url, timeout=self.request_timeout_seconds) as image_response:  # noqa: S310
                    data = image_response.read()
                if len(data) < 128:
                    raise RuntimeError(f"downloaded image is too small: {len(data)} bytes")
                output_path.write_bytes(data)
                self._validate_downloaded_image(output_path)
                return
            except Exception as exc:  # noqa: BLE001 - retry handles transient URL/download/Pillow failures.
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(10.0, 0.6 * (2**attempt)))
        raise RuntimeError(f"Wan image download failed: {last_error}")

    @staticmethod
    def _validate_downloaded_image(output_path: Path) -> None:
        with Image.open(output_path) as image:
            image.verify()

    def _stage(self, name: str) -> ContextManager[Any]:
        return self.stage_timer(name) if self.stage_timer is not None else nullcontext()

    def _generate_dry_run(self, original_image_path: str | Path, bbox_list: list[BBox], output_path: Path, anomaly_type: str) -> dict[str, Any]:
        with Image.open(original_image_path) as source:
            base = source.convert("RGB")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        width, height = base.size
        rng = random.Random(str(output_path))
        for bbox in bbox_list:
            x1, y1, x2, y2 = clip_bbox(bbox, width, height)
            bw, bh = x2 - x1, y2 - y1
            color = self._leak_color(anomaly_type)
            center_x = x1 + int(bw * (0.45 + rng.random() * 0.15))
            center_y = y1 + int(bh * (0.62 + rng.random() * 0.20))
            if anomaly_type in {"water_leak", "water_leakage"}:
                puddle_w = max(5, int(bw * (0.08 + rng.random() * 0.05)))
                puddle_h = max(3, int(bh * (0.025 + rng.random() * 0.025)))
                blob_count = 3
            else:
                puddle_w = max(6, int(bw * (0.18 + rng.random() * 0.12)))
                puddle_h = max(4, int(bh * (0.06 + rng.random() * 0.08)))
                blob_count = 5
            for idx in range(blob_count):
                dx = int((rng.random() - 0.5) * puddle_w * 0.9)
                dy = int((rng.random() - 0.5) * puddle_h * 0.9)
                scale = 1.0 - idx * 0.12
                box = (
                    center_x + dx - int(puddle_w * scale / 2),
                    center_y + dy - int(puddle_h * scale / 2),
                    center_x + dx + int(puddle_w * scale / 2),
                    center_y + dy + int(puddle_h * scale / 2),
                )
                draw.ellipse(box, fill=color)
            if anomaly_type not in {"water_leak", "water_leakage"}:
                trail_points = []
                for idx in range(5):
                    px = center_x - int(puddle_w * 0.45) + int(idx * puddle_w / 4)
                    py = center_y - int(puddle_h * 1.6) + int(rng.random() * puddle_h)
                    trail_points.append((px, py))
                draw.line(trail_points, fill=color, width=max(2, int(min(bw, bh) * 0.015)))
            else:
                highlight = (255, 255, 255, 90)
                draw.arc(
                    (
                        center_x - puddle_w // 3,
                        center_y - puddle_h // 2,
                        center_x + puddle_w // 3,
                        center_y + puddle_h // 2,
                    ),
                    start=195,
                    end=330,
                    fill=highlight,
                    width=1,
                )
        generated = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        generated.save(output_path, quality=95)
        return {
            "dry_run": True,
            "model": self.model_name,
            "bbox_list": [list(map(int, bbox)) for bbox in bbox_list],
            "saved_output_path": str(output_path),
            "input_was_clean_original": True,
            "note": "Synthetic local leak drawn for offline validation; no grid/text/box annotations added.",
        }

    @staticmethod
    def _leak_color(anomaly_type: str) -> tuple[int, int, int, int]:
        if anomaly_type == "oil_leak":
            return (25, 18, 12, 150)
        if anomaly_type == "coolant_leak":
            return (115, 210, 170, 115)
        if anomaly_type in {"water_leak", "water_leakage"}:
            return (245, 250, 255, 72)
        return (228, 184, 75, 105)


def _maybe_semaphore(semaphore: Any | None) -> ContextManager[Any]:
    return semaphore if semaphore is not None else nullcontext()
