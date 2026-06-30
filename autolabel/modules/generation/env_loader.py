from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def default_env_paths() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    module_root = Path(__file__).resolve().parent
    return [repo_root / ".env", module_root / ".env"]


def load_env_files(paths: Iterable[str | Path] | None = None, *, override: bool = False) -> list[Path]:
    loaded: list[Path] = []
    for raw_path in paths or default_env_paths():
        path = Path(raw_path)
        if not path.exists():
            continue
        _load_one_env_file(path, override=override)
        loaded.append(path)
    return loaded


def _load_one_env_file(path: Path, *, override: bool) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = _parse_env_value(value)


def _parse_env_value(value: str) -> str:
    parsed = value.strip()
    if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {"'", '"'}:
        return parsed[1:-1]
    return parsed
