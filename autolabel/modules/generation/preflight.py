from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ...model_config import get_credentials, resolve_generation_runtime
from ...utils import read_csv, resolve_path
from .config import filter_generation_rows


class GenerationPreflightError(RuntimeError):
    pass


def _resolve_existing_image_path(image_uri: str, image_root: str | Path) -> Path | None:
    candidates = [
        resolve_path(image_uri),
        resolve_path(image_uri, image_root),
        Path(image_root) / Path(image_uri).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _generation_module_config(config: dict[str, Any]) -> dict[str, Any]:
    generation_module = config.get("modules", {}).get("generation", {})
    backend_name = generation_module.get("backend", "i2i_external")
    backends = generation_module.get("backends", {}) if isinstance(generation_module.get("backends"), dict) else {}
    backend_config = backends.get(backend_name, {}) if isinstance(backends.get(backend_name), dict) else {}
    return {
        "backend": backend_name,
        "project_dir": backend_config.get("project_dir") or config.get("paths", {}).get("i2i_project"),
        "entrypoint": backend_config.get("entrypoint", "src/main.py"),
    }


def _credential_report(config: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    credential = get_credentials(config, profile.get("credential_ref"))
    api_key_env_names = []
    for value in (profile.get("api_key_env") or credential.get("api_key_env"), profile.get("api_key_env_aliases")):
        if value in ("", None, False):
            continue
        if isinstance(value, str):
            candidates = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple, set)):
            candidates = [str(item).strip() for item in value if str(item).strip()]
        else:
            candidates = [str(value)]
        for candidate in candidates:
            if candidate not in api_key_env_names:
                api_key_env_names.append(candidate)
    configured_value = profile.get("api_key") or credential.get("api_key")
    return {
        "api_key_env": api_key_env_names[0] if api_key_env_names else None,
        "api_key_env_aliases": api_key_env_names[1:],
        "present": bool(configured_value) or any(os.getenv(str(name)) for name in api_key_env_names),
    }


def run_generation_preflight(
    config: dict[str, Any],
    tasks_csv: str | Path,
    image_root: str | Path,
    output_root: str | Path,
    limit: int | None = None,
    require_credentials: bool = True,
) -> dict[str, Any]:
    tasks_path = Path(tasks_csv)
    if not tasks_path.exists():
        raise GenerationPreflightError(f"Generation manifest not found: {tasks_path}")

    all_rows = read_csv(tasks_path)
    rows = filter_generation_rows(str(tasks_path))
    if not rows:
        return {
            "skipped": True,
            "manifest_rows": len(all_rows),
            "generation_rows": 0,
        }
    if limit is not None and limit > len(rows):
        raise GenerationPreflightError(
            f"Generation limit {limit} exceeds available generation rows {len(rows)} in {tasks_path}"
        )

    missing_required = []
    missing_images = []
    for row in rows[:limit]:
        sample_id = row.get("sample_id") or "<missing sample_id>"
        for field in ("sample_id", "image_id", "image_uri", "anomaly_type"):
            if not row.get(field):
                missing_required.append(f"{sample_id}:{field}")
        image_uri = row.get("image_uri")
        if image_uri and _resolve_existing_image_path(image_uri, image_root) is None:
            missing_images.append(f"{sample_id}:{image_uri}")

    if missing_required:
        preview = ", ".join(missing_required[:10])
        raise GenerationPreflightError(f"Generation manifest has missing required values: {preview}")
    if missing_images:
        preview = ", ".join(missing_images[:10])
        raise GenerationPreflightError(f"Generation manifest references missing images: {preview}")

    module_config = _generation_module_config(config)
    project_dir = Path(str(module_config.get("project_dir") or "external/I2I"))
    entrypoint = project_dir / str(module_config.get("entrypoint") or "src/main.py")
    if not entrypoint.exists():
        raise GenerationPreflightError(f"I2I entrypoint not found: {entrypoint}")

    output_path = Path(output_root).resolve()
    image_root_path = Path(image_root).resolve()
    if output_path == image_root_path:
        raise GenerationPreflightError(f"Generation output root must be isolated from image root: {output_path}")

    anomaly_types = sorted({row.get("anomaly_type") or "default" for row in rows[:limit]})
    runtime_by_anomaly: dict[str, dict[str, Any]] = {}
    credential_checks: list[dict[str, Any]] = []
    for anomaly_type in anomaly_types:
        runtime = resolve_generation_runtime(config, anomaly_type=anomaly_type)
        runtime_by_anomaly[anomaly_type] = {
            "vlm_model_name": runtime.get("vlm_model_name"),
            "image_model_name": runtime.get("image_model_name"),
        }
        credential_checks.append(_credential_report(config, runtime.get("vlm_profile", {})))
        credential_checks.append(_credential_report(config, runtime.get("image_profile", {})))

    missing_credentials = [
        check.get("api_key_env") or "<inline api_key>"
        for check in credential_checks
        if not check.get("present")
    ]
    if require_credentials and missing_credentials:
        unique_missing = sorted({str(value) for value in missing_credentials})
        raise GenerationPreflightError(
            "Missing generation API credentials for: " + ", ".join(unique_missing)
        )

    return {
        "skipped": False,
        "manifest_rows": len(all_rows),
        "generation_rows": len(rows),
        "effective_generation_rows": min(len(rows), limit) if limit is not None else len(rows),
        "anomaly_types": anomaly_types,
        "runtime_by_anomaly": runtime_by_anomaly,
        "i2i_entrypoint": str(entrypoint),
        "output_root": str(output_path),
        "credential_checks": credential_checks,
    }
