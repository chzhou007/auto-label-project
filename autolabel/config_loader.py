from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        return os.getenv(name, default if default is not None else "")

    return ENV_PATTERN.sub(replace, value)


def expand_env_values(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, list):
        return [expand_env_values(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_values(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    with open(config_path, encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to read YAML config files. Run: pip install PyYAML") from exc
            data = yaml.safe_load(f) or {}
        elif suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {config_path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object: {config_path}")
    return expand_env_values(data)
