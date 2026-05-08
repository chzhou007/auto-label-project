from __future__ import annotations

from typing import Any

from .i2i_external import ExternalI2IGenerationModule


def build_generation_module(config: dict[str, Any]) -> ExternalI2IGenerationModule:
    module_cfg = config.get("modules", {}).get("generation", {})
    backend = module_cfg.get("backend", "i2i_external")
    backend_cfg = module_cfg.get("backends", {}).get(backend, {})
    if backend == "i2i_external":
        return ExternalI2IGenerationModule(config, backend_cfg)
    raise ValueError(f"Unsupported generation backend: {backend}")
