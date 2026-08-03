from __future__ import annotations

import json
from typing import Any

from ...model_config import resolve_localizer_config
from ...utils import read_csv


def filter_generation_rows(tasks_csv: str) -> list[dict[str, Any]]:
    rows = read_csv(tasks_csv)
    if any("task_mode" in row for row in rows):
        rows = [row for row in rows if row.get("task_mode") == "generation"]
    return rows


def has_localizer_policy(config: dict[str, Any]) -> bool:
    generation_cfg = config.get("generation", {})
    module_cfg = config.get("modules", {}).get("generation", {})
    return isinstance(generation_cfg.get("localizer_policy"), dict) or isinstance(module_cfg.get("localizer_policy"), dict)


def group_generation_rows_by_localizer_policy(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    if not has_localizer_policy(config):
        return [("default", rows)]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        anomaly_type = row.get("anomaly_type") or "default"
        localizer_cfg = resolve_localizer_config(config, anomaly_type=anomaly_type)
        key = json.dumps(localizer_cfg, ensure_ascii=False, sort_keys=True)
        grouped.setdefault(key, []).append(row)
    return [(f"group_{index + 1}", group_rows) for index, group_rows in enumerate(grouped.values())]
