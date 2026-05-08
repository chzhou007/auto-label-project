from __future__ import annotations

from typing import Any, Protocol


class ClassificationModule(Protocol):
    def classify_sample(
        self,
        sample: dict[str, Any],
        skip_source_types: set[str] | None = None,
    ) -> dict[str, Any]:
        ...
