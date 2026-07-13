from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .base import LocalizationResult


def _coerce_result(result: Any) -> LocalizationResult:
    if isinstance(result, LocalizationResult):
        return result
    if is_dataclass(result):
        payload = asdict(result)
    else:
        payload = {
            "success": getattr(result, "success"),
            "final_bbox": getattr(result, "final_bbox"),
            "mask_path": getattr(result, "mask_path"),
            "method": getattr(result, "method"),
            "metrics": dict(getattr(result, "metrics", {}) or {}),
            "reason": getattr(result, "reason", None),
            "fallback_used": getattr(result, "fallback_used", False),
        }
    return LocalizationResult(**payload)


class FallbackLocalizer:
    def __init__(self, primary: Any, fallback: Any) -> None:
        self.primary = primary
        self.fallback = fallback

    def localize(self, **kwargs: Any) -> LocalizationResult:
        try:
            primary_result = _coerce_result(self.primary.localize(**kwargs))
        except Exception as exc:
            primary_method = getattr(self.primary, "method_name", self.primary.__class__.__name__)
            primary_result = LocalizationResult(
                success=False,
                final_bbox=None,
                mask_path=None,
                method=str(primary_method),
                metrics={
                    "localizer_method": str(primary_method),
                    "localizer_error": type(exc).__name__,
                    "localizer_error_message": str(exc),
                },
                reason="localizer_exception",
            )
        if primary_result.success:
            primary_result.metrics = dict(primary_result.metrics)
            primary_result.metrics.setdefault("primary_method", primary_result.method)
            primary_result.fallback_used = False
            return primary_result

        try:
            fallback_result = _coerce_result(self.fallback.localize(**kwargs))
        except Exception as exc:
            fallback_method = getattr(self.fallback, "method_name", self.fallback.__class__.__name__)
            fallback_result = LocalizationResult(
                success=False,
                final_bbox=None,
                mask_path=None,
                method=str(fallback_method),
                metrics={
                    "localizer_method": str(fallback_method),
                    "localizer_error": type(exc).__name__,
                    "localizer_error_message": str(exc),
                },
                reason="fallback_exception",
            )
        fallback_result.metrics = dict(fallback_result.metrics)
        fallback_result.metrics["primary_method"] = primary_result.method
        fallback_result.metrics["fallback_method"] = fallback_result.method
        fallback_result.metrics["primary_reason"] = primary_result.reason
        fallback_result.fallback_used = True
        return fallback_result
