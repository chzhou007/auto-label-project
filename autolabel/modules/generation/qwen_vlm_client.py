from __future__ import annotations

import json
import os
import re
import socket
import ssl
import hashlib
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

from autolabel.utils import image_to_data_url, read_json, write_json

from .grid import grid_ids, validate_grid_id
from .prompts import build_qwen_fine_grid_prompt, build_qwen_grid_prompt, build_qwen_review_prompt


QWEN_GRID_PROMPT_VERSION = "qwen-grid-top3-sensitive-v3.0"


class VLMResponseError(RuntimeError):
    pass


class VLMNetworkError(VLMResponseError):
    pass


class VLMJsonError(VLMResponseError):
    pass


@dataclass(frozen=True)
class GridCandidate:
    grid: str
    score: float
    reason: str


@dataclass(frozen=True)
class GridSelection:
    selected_grid: str
    confidence: float
    candidate_grids: list[GridCandidate]
    edit_region_hint: str
    risk_note: str
    rejected_region_reason: str
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class ReviewResult:
    is_valid: bool
    anomaly_type_match: bool
    location_reasonable: bool
    visual_artifact: bool
    score: float
    reason: str
    raw_response: dict[str, Any]


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _as_score(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


class QwenVLMClient:
    def __init__(
        self,
        model_name: str = "qwen3.6-plus",
        dry_run: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        request_image_max_side: int = 768,
        request_timeout_seconds: int = 60,
        json_retry_count: int = 1,
        enable_grid_fallback: bool = True,
        jpeg_quality: int = 82,
        candidate_grid_count: int = 3,
        cache_dir: str | Path | None = None,
        reuse_cache: bool = False,
        refresh_cache: bool = False,
        max_tokens: int = 256,
        max_http_retries: int = 2,
    ) -> None:
        self.model_name = model_name
        self.dry_run = dry_run
        self.api_key = api_key or os.getenv("QWEN_VLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("QWEN_VLM_BASE_URL")
            or os.getenv("DASHSCOPE_VLM_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.base_url = "".join(str(self.base_url).split())
        self.request_image_max_side = request_image_max_side
        self.request_timeout_seconds = request_timeout_seconds
        self.json_retry_count = max(0, json_retry_count)
        self.enable_grid_fallback = enable_grid_fallback
        self.jpeg_quality = jpeg_quality
        self.candidate_grid_count = max(1, int(candidate_grid_count))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.reuse_cache = reuse_cache
        self.refresh_cache = refresh_cache
        self.max_tokens = max(64, int(max_tokens))
        self.max_http_retries = max(0, int(max_http_retries))
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def select_grid(
        self,
        grid_preview_image: str | Path,
        anomaly_type: str,
        grid_layout: str = "4x4",
        candidate_count: int | None = None,
        excluded_grids: list[str] | None = None,
    ) -> GridSelection:
        excluded = _normalize_excluded_grids(excluded_grids, grid_layout)
        legal_count = max(1, len(grid_ids(grid_layout)) - len(excluded))
        target_count = min(max(1, int(candidate_count or self.candidate_grid_count)), legal_count)
        prompt = build_qwen_grid_prompt(anomaly_type, candidate_count=target_count, excluded_grids=excluded)
        if self.dry_run:
            return self._dummy_grid_selection(
                grid_layout,
                edit_region_hint=f"dry-run {anomaly_type} top-{target_count} candidate",
                candidate_count=target_count,
                image_path=grid_preview_image,
                excluded_grids=excluded,
            )
        cache_path = self._cache_path(grid_preview_image, anomaly_type, grid_layout, target_count, excluded)
        if cache_path and self.reuse_cache and not self.refresh_cache and cache_path.exists():
            try:
                payload = read_json(cache_path)
                selection = self._parse_grid_selection(payload, grid_layout, target_count, image_path=grid_preview_image, excluded_grids=excluded)
                raw = dict(selection.raw_response)
                raw["cache_hit"] = True
                return GridSelection(
                    selected_grid=selection.selected_grid,
                    confidence=selection.confidence,
                    candidate_grids=selection.candidate_grids,
                    edit_region_hint=selection.edit_region_hint,
                    risk_note=selection.risk_note,
                    rejected_region_reason=selection.rejected_region_reason,
                    raw_response=raw,
                )
            except Exception:
                pass
        try:
            payload = self._call_json_model(grid_preview_image, prompt, retry_prompt=prompt + "\nReturn JSON only. No prose.")
            try:
                selection = self._parse_grid_selection(payload, grid_layout, target_count, image_path=grid_preview_image, excluded_grids=excluded)
            except (VLMResponseError, ValueError) as schema_exc:
                repair_prompt = (
                    prompt
                    + "\nYour previous JSON did not satisfy the schema. Return exactly "
                    + str(target_count)
                    + " unique legal coarse grid IDs from the 4x4 grid. JSON only."
                )
                payload = self._call_json_model(grid_preview_image, repair_prompt, retry_prompt=repair_prompt)
                selection = self._parse_grid_selection(payload, grid_layout, target_count, image_path=grid_preview_image, excluded_grids=excluded)
                selection = GridSelection(
                    selected_grid=selection.selected_grid,
                    confidence=selection.confidence,
                    candidate_grids=selection.candidate_grids,
                    edit_region_hint=selection.edit_region_hint,
                    risk_note=selection.risk_note,
                    rejected_region_reason=selection.rejected_region_reason,
                    raw_response={**selection.raw_response, "schema_repair_retry": True, "schema_error": str(schema_exc)},
                )
            if cache_path:
                write_json(cache_path, selection.raw_response)
            return selection
        except (VLMResponseError, ValueError, json.JSONDecodeError) as exc:
            if self.enable_grid_fallback:
                selection = self._dummy_grid_selection(
                    grid_layout,
                    edit_region_hint=f"fallback {anomaly_type} candidate after Qwen failure",
                    fallback_reason=str(exc),
                    candidate_count=target_count,
                    image_path=grid_preview_image,
                    excluded_grids=excluded,
                )
                if cache_path:
                    write_json(cache_path, selection.raw_response)
                return selection
            if isinstance(exc, VLMResponseError):
                raise
            raise VLMResponseError(f"Qwen VLM grid selection failed: {exc}") from exc

    def select_fine_grid(self, fine_grid_preview_image: str | Path, anomaly_type: str, coarse_grid: str) -> GridSelection:
        prompt = build_qwen_fine_grid_prompt(anomaly_type, coarse_grid)
        if self.dry_run:
            return self._dummy_grid_selection("3x3", edit_region_hint=f"dry-run fine selection within {coarse_grid}")
        try:
            payload = self._call_json_model(fine_grid_preview_image, prompt, retry_prompt=prompt + "\nReturn JSON only. No prose.")
            return self._parse_grid_selection(payload, "3x3", 3, image_path=fine_grid_preview_image)
        except (VLMResponseError, ValueError, json.JSONDecodeError) as exc:
            if self.enable_grid_fallback:
                return self._dummy_grid_selection(
                    "3x3",
                    edit_region_hint=f"fallback fine selection within {coarse_grid} after Qwen failure",
                    fallback_reason=str(exc),
                    candidate_count=3,
                    image_path=fine_grid_preview_image,
                )
            if isinstance(exc, VLMResponseError):
                raise
            raise VLMResponseError(f"Qwen VLM fine grid selection failed: {exc}") from exc

    def review_generation(self, image_path: str | Path, anomaly_type: str) -> ReviewResult:
        prompt = build_qwen_review_prompt(anomaly_type)
        if self.dry_run:
            raw = {
                "is_valid": True,
                "anomaly_type_match": True,
                "location_reasonable": True,
                "visual_artifact": False,
                "score": 0.88,
                "reason": "dry-run review accepts deterministic synthetic anomaly",
            }
            return self._parse_review(raw)
        payload = self._call_json_model(image_path, prompt, retry_prompt=prompt + "\nReturn JSON only. No prose.")
        return self._parse_review(payload)

    def _call_json_model(self, image_path: str | Path, prompt: str, retry_prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise VLMResponseError("Qwen VLM API key is not configured. Use --dry-run or set DASHSCOPE_API_KEY/QWEN_VLM_API_KEY.")
        prompts = [prompt] + [retry_prompt] * self.json_retry_count
        last_error: Exception | None = None
        for attempt_index, current_prompt in enumerate(prompts, 1):
            try:
                raw_text = self._openai_compatible_vision_call(image_path, current_prompt)
                return _extract_json_object(raw_text)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = VLMNetworkError(f"Qwen VLM HTTP {exc.code}: {body}")
                if attempt_index < len(prompts) and exc.code in {408, 409, 429, 500, 502, 503, 504}:
                    time.sleep(_backoff_seconds(attempt_index))
                    continue
                raise last_error from exc
            except (TimeoutError, socket.timeout, ssl.SSLError, urllib.error.URLError) as exc:
                last_error = VLMNetworkError(f"Qwen VLM network error: {exc}")
                if attempt_index < len(prompts):
                    time.sleep(_backoff_seconds(attempt_index))
                    continue
                raise last_error from exc
            except json.JSONDecodeError as exc:
                last_error = exc
                if attempt_index < len(prompts):
                    continue
            except Exception as exc:  # noqa: BLE001 - record retry cause for API/debug flow.
                last_error = exc
                if attempt_index < len(prompts):
                    continue
        raise VLMJsonError(f"Qwen VLM response was not valid JSON after {len(prompts)} attempt(s): {last_error}")

    def _openai_compatible_vision_call(self, image_path: str | Path, prompt: str) -> str:
        data_url = image_to_data_url(image_path, max_side=self.request_image_max_side, jpeg_quality=self.jpeg_quality)
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        image_url: dict[str, Any] = {"url": data_url}
        if self.request_image_max_side:
            image_url["max_pixels"] = int(self.request_image_max_side * self.request_image_max_side)
        request_payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a strict JSON API. Return only valid JSON."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": image_url},
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        payload = self._urlopen_json(request)
        content = payload.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            raise VLMResponseError(f"Qwen VLM returned an empty response: {payload}")
        return content

    def _urlopen_json(self, request: urllib.request.Request) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_http_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:  # noqa: S310 - endpoint is user-configured.
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = VLMNetworkError(f"Qwen VLM HTTP {exc.code}: {body}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= self.max_http_retries:
                    raise last_error from exc
            except (TimeoutError, socket.timeout, ssl.SSLError, urllib.error.URLError) as exc:
                last_error = VLMNetworkError(f"Qwen VLM network error: {exc}")
                if attempt >= self.max_http_retries:
                    raise last_error from exc
            time.sleep(_backoff_seconds(attempt + 1))
        raise VLMNetworkError(f"Qwen VLM request failed: {last_error}")

    def _parse_grid_selection(
        self,
        payload: dict[str, Any],
        grid_layout: str,
        candidate_count: int,
        image_path: str | Path | None = None,
        excluded_grids: list[str] | None = None,
    ) -> GridSelection:
        excluded = set(_normalize_excluded_grids(excluded_grids, grid_layout))
        selected_grid = str(payload.get("selected_grid", "")).strip().upper()
        raw_candidates = payload.get("candidate_grids")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise VLMResponseError("candidate_grids must be a non-empty list")

        candidates: list[GridCandidate] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_candidates):
            if isinstance(item, str):
                grid = item.strip().upper()
                score = max(0.01, 1.0 - index * 0.06)
                reason = "Qwen candidate grid"
            elif isinstance(item, dict):
                grid = str(item.get("grid", "")).strip().upper()
                score = _as_score(item.get("score"), default=max(0.01, 1.0 - index * 0.06))
                reason = str(item.get("reason") or "")
            else:
                continue
            try:
                validate_grid_id(grid, grid_layout)
            except ValueError:
                continue
            if grid in excluded:
                continue
            if grid in seen:
                continue
            seen.add(grid)
            candidates.append(GridCandidate(grid=grid, score=score, reason=reason))
        if not selected_grid:
            selected_grid = candidates[0].grid if candidates else ""
        try:
            validate_grid_id(selected_grid, grid_layout)
        except ValueError:
            selected_grid = candidates[0].grid if candidates else self._fallback_grid_order(grid_layout, image_path, excluded_grids=list(excluded))[0]
        if selected_grid in excluded:
            selected_grid = candidates[0].grid if candidates else self._fallback_grid_order(grid_layout, image_path, excluded_grids=list(excluded))[0]
        if selected_grid and selected_grid not in seen:
            candidates.insert(0, GridCandidate(selected_grid, _as_score(payload.get("confidence")), "selected_grid fallback"))
            seen.add(selected_grid)
        if len(candidates) < candidate_count:
            for grid in self._fallback_grid_order(grid_layout, image_path, excluded_grids=list(excluded)):
                if grid not in {candidate.grid for candidate in candidates}:
                    score = max(0.01, 0.45 - len(candidates) * 0.03)
                    candidates.append(GridCandidate(grid, score, "automatic structure-aware fallback to complete top-k coarse grids"))
                if len(candidates) >= candidate_count:
                    break
        candidates = sorted(candidates, key=lambda item: item.score, reverse=True)[:candidate_count]
        raw_response = dict(payload)
        raw_response["candidate_grid_count"] = candidate_count
        raw_response["excluded_grids"] = sorted(excluded)
        raw_response["prompt_version"] = QWEN_GRID_PROMPT_VERSION
        raw_response["repaired_or_completed"] = len(raw_candidates) != len(candidates) or len(candidates) < candidate_count
        return GridSelection(
            selected_grid=selected_grid,
            confidence=_as_score(payload.get("confidence")),
            candidate_grids=candidates,
            edit_region_hint=str(payload.get("edit_region_hint") or ""),
            risk_note=str(payload.get("risk_note") or ""),
            rejected_region_reason=str(payload.get("rejected_region_reason") or ""),
            raw_response=raw_response,
        )

    def _parse_review(self, payload: dict[str, Any]) -> ReviewResult:
        return ReviewResult(
            is_valid=bool(payload.get("is_valid")),
            anomaly_type_match=bool(payload.get("anomaly_type_match")),
            location_reasonable=bool(payload.get("location_reasonable")),
            visual_artifact=bool(payload.get("visual_artifact")),
            score=_as_score(payload.get("score")),
            reason=str(payload.get("reason") or ""),
            raw_response=payload,
        )

    def _dummy_grid_selection(
        self,
        grid_layout: str,
        edit_region_hint: str,
        fallback_reason: str | None = None,
        candidate_count: int | None = None,
        image_path: str | Path | None = None,
        excluded_grids: list[str] | None = None,
    ) -> GridSelection:
        target_count = max(1, int(candidate_count or self.candidate_grid_count))
        candidates = self._fallback_grid_order(grid_layout, image_path, excluded_grids=excluded_grids)[:target_count]
        scores = [max(0.01, 0.86 - idx * 0.05) for idx in range(target_count)]
        raw_candidates = [
            {"grid": grid, "score": scores[idx], "reason": "deterministic fallback candidate" if fallback_reason else "deterministic dry-run candidate"}
            for idx, grid in enumerate(candidates)
        ]
        raw = {
            "selected_grid": candidates[0],
            "confidence": scores[0],
            "candidate_grids": raw_candidates,
            "edit_region_hint": edit_region_hint,
            "risk_note": "Qwen grid selection fallback was used" if fallback_reason else "dry-run does not use an external VLM",
            "rejected_region_reason": "fallback skips visual semantics and validates pipeline continuity" if fallback_reason else "dry-run skips visual semantics and validates pipeline structure",
            "fallback": bool(fallback_reason),
            "fallback_reason": fallback_reason,
            "excluded_grids": _normalize_excluded_grids(excluded_grids, grid_layout),
        }
        return self._parse_grid_selection(raw, grid_layout, target_count, image_path=image_path, excluded_grids=excluded_grids)

    def _cache_path(
        self,
        image_path: str | Path,
        anomaly_type: str,
        grid_layout: str,
        candidate_count: int,
        excluded_grids: list[str] | None = None,
    ) -> Path | None:
        if not self.cache_dir:
            return None
        try:
            image_digest = _file_sha256(image_path)
        except OSError:
            image_digest = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()
        key_parts = [
            image_digest,
            anomaly_type,
            grid_layout,
            str(candidate_count),
            str(self.request_image_max_side),
            QWEN_GRID_PROMPT_VERSION,
            ",".join(_normalize_excluded_grids(excluded_grids, grid_layout)),
        ]
        digest = hashlib.sha256("|".join(key_parts).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _fallback_grid_order(
        self,
        grid_layout: str,
        image_path: str | Path | None = None,
        excluded_grids: list[str] | None = None,
    ) -> list[str]:
        ids = grid_ids(grid_layout)
        excluded = set(_normalize_excluded_grids(excluded_grids, grid_layout))
        if image_path is None:
            ordered = _default_grid_order(grid_layout)
        else:
            try:
                ordered = _score_fallback_grids(image_path, grid_layout)
            except Exception:
                ordered = _default_grid_order(grid_layout)
        filtered = [grid for grid in ordered if grid not in excluded]
        return filtered or ids


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_excluded_grids(excluded_grids: list[str] | None, grid_layout: str) -> list[str]:
    if not excluded_grids:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in excluded_grids:
        grid = str(value).strip().upper()
        if not grid or grid in seen:
            continue
        try:
            validate_grid_id(grid, grid_layout)
        except ValueError:
            continue
        normalized.append(grid)
        seen.add(grid)
    return normalized


def _backoff_seconds(attempt_index: int) -> float:
    return min(8.0, 0.4 * (2 ** max(0, attempt_index - 1)))


def _default_grid_order(grid_layout: str) -> list[str]:
    ids = grid_ids(grid_layout)
    rows_cols = grid_layout.lower().split("x")
    rows, cols = int(rows_cols[0]), int(rows_cols[1])
    scored: list[tuple[float, str]] = []
    for grid in ids:
        row = ord(grid[0]) - ord("A")
        col = int(grid[1:]) - 1
        lower_mid = row / max(1, rows - 1)
        center = 1.0 - abs((col + 0.5) / cols - 0.5) * 1.4
        score = 0.55 * lower_mid + 0.30 * center
        if row == 0:
            score -= 0.35
        scored.append((score, grid))
    return [grid for _, grid in sorted(scored, key=lambda item: item[0], reverse=True)]


def _score_fallback_grids(image_path: str | Path, grid_layout: str) -> list[str]:
    ids = grid_ids(grid_layout)
    rows_cols = grid_layout.lower().split("x")
    rows, cols = int(rows_cols[0]), int(rows_cols[1])
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scored: list[tuple[float, str]] = []
    for grid in ids:
        row = ord(grid[0]) - ord("A")
        col = int(grid[1:]) - 1
        x1 = round(col * width / cols)
        y1 = round(row * height / rows)
        x2 = round((col + 1) * width / cols)
        y2 = round((row + 1) * height / rows)
        crop = image.crop((x1, y1, x2, y2)).resize((64, 64), Image.Resampling.BILINEAR)
        gray = crop.convert("L")
        stat = ImageStat.Stat(gray)
        mean = float(stat.mean[0])
        variance = float(stat.var[0])
        # Favor structured lower/middle cells while avoiding pure blank ceiling/wall/floor areas.
        structure = min(1.0, variance / 1800.0)
        tonal_penalty = 0.35 if mean < 25 or mean > 235 else 0.0
        lower = row / max(1, rows - 1)
        mid_col = 1.0 - abs((col + 0.5) / cols - 0.5) * 1.2
        score = 0.42 * structure + 0.36 * lower + 0.18 * mid_col - tonal_penalty
        if row == 0:
            score -= 0.25
        scored.append((score, grid))
    return [grid for _, grid in sorted(scored, key=lambda item: item[0], reverse=True)]
