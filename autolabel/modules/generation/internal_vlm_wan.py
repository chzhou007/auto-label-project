from __future__ import annotations

from typing import Any

from .i2i_external import ExternalI2IGenerationModule


class InternalVLMWanGenerationModule(ExternalI2IGenerationModule):
    """Compatibility backend name for the roadmap generation stack.

    The current repository version still delegates image generation to the
    existing I2I project, then applies in-repo localizer postprocess during
    metadata ingestion.
    """

    def __init__(self, pipeline_config: dict[str, Any], module_config: dict[str, Any] | None = None) -> None:
        super().__init__(pipeline_config, module_config=module_config)
