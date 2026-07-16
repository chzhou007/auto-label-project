from __future__ import annotations

import os
from dataclasses import dataclass


SUPPORTED_ANOMALY_TYPES = {"diesel_leak", "oil_leak", "coolant_leak", "water_leak"}
VALID_GRIDS = {f"{row}{col}" for row in "ABCD" for col in range(1, 5)}

DEFAULT_QWEN_API_URL = "https://deepseek.gds-services.com/v1"
DEFAULT_DASHSCOPE_GENERATION_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)
DEFAULT_SEEDREAM_GENERATION_ENDPOINT = "https://ark.cn-beijing.volces.com/api/plan/v3/images/generations"


@dataclass(frozen=True)
class PipelineConfig:
    tasks: str
    image_root: str
    output_root: str
    vlm_model: str = "qwen3.6-27b"
    image_model: str = "doubao-seedream-5.0-lite"
    grid_layout: str = "4x4"
    edit_bbox_expand_ratio: float = 0.20
    crop_expand_ratio: float = 0.10
    vlm_min_confidence: float = 0.45
    max_retries: int = 2
    dry_run: bool = False
    limit: int | None = None
    workers: int = 1
    skip_existing: bool = False
    seedream_mode: str | None = None
    water_reference_dir: str | None = None
    red_box_max_size: int = 200
    red_box_min_size: int = 200


@dataclass(frozen=True)
class ModelServiceConfig:
    api_key: str | None
    provider: str
    endpoint: str
    api_key_env: str
    endpoint_env: str


@dataclass(frozen=True)
class I2IServiceConfig:
    vlm: ModelServiceConfig
    image: ModelServiceConfig

    @classmethod
    def from_env(cls) -> "I2IServiceConfig":
        qwen_api_key = os.getenv("QWEN397B_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        qwen_endpoint = os.getenv("QWEN397B_API_URL") or os.getenv("DASHSCOPE_VLM_ENDPOINT") or DEFAULT_QWEN_API_URL

        seedream_api_key = os.getenv("ARK_API_KEY") or os.getenv("SEEDREAM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        seedream_endpoint = (
            os.getenv("SEEDREAM_BASE_URL")
            or os.getenv("SEEDREAM_ENDPOINT")
            or os.getenv("ARK_BASE_URL")
            or os.getenv("DASHSCOPE_WAN_ENDPOINT")
            or DEFAULT_SEEDREAM_GENERATION_ENDPOINT
        )

        image_provider = "volcengine_ark"
        if "dashscope.aliyuncs.com" in seedream_endpoint:
            image_provider = "dashscope"

        return cls(
            vlm=ModelServiceConfig(
                api_key=qwen_api_key,
                provider="openai_compatible",
                endpoint=qwen_endpoint,
                api_key_env="QWEN397B_API_KEY",
                endpoint_env="QWEN397B_API_URL",
            ),
            image=ModelServiceConfig(
                api_key=seedream_api_key,
                provider=image_provider,
                endpoint=seedream_endpoint,
                api_key_env="ARK_API_KEY" if image_provider == "volcengine_ark" else "DASHSCOPE_API_KEY",
                endpoint_env="SEEDREAM_BASE_URL" if image_provider == "volcengine_ark" else "DASHSCOPE_WAN_ENDPOINT",
            ),
        )


@dataclass(frozen=True)
class DashScopeConfig:
    """Compatibility wrapper for old imports/tests that expect one DashScope object."""

    api_key: str | None
    provider: str
    vlm_endpoint: str
    wan_endpoint: str

    @classmethod
    def from_env(cls) -> "DashScopeConfig":
        services = I2IServiceConfig.from_env()
        return cls(
            api_key=services.vlm.api_key,
            provider=services.vlm.provider,
            vlm_endpoint=services.vlm.endpoint,
            wan_endpoint=services.image.endpoint,
        )
