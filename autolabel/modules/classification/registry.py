from __future__ import annotations

from typing import Any

from .dry_run import DryRunClassificationModule
from .external_script import ExternalScriptClassificationModule


def build_classification_module(config: dict[str, Any]) -> ExternalScriptClassificationModule | DryRunClassificationModule:
    module_cfg = config.get("modules", {}).get("classification", {})
    backend = module_cfg.get("backend", "external_script")
    backend_cfg = module_cfg.get("backends", {}).get(backend, {})
    if backend == "external_script":
        return ExternalScriptClassificationModule(config, backend_cfg)
    if backend == "dry_run_rule":
        return DryRunClassificationModule(config, backend_cfg)
    raise ValueError(f"Unsupported classification backend: {backend}")
