from __future__ import annotations

from copy import deepcopy
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _non_empty(value: Any) -> Any:
    return None if value in ("", None) else value


def get_credentials(config: dict[str, Any], credential_ref: str | None) -> dict[str, Any]:
    if not credential_ref:
        return {}
    credentials = config.get("credentials") or config.get("secrets") or {}
    credential = credentials.get(credential_ref, {})
    if not isinstance(credential, dict):
        return {"api_key": credential}
    return credential


def get_model_profile(
    config: dict[str, Any],
    group: str,
    collection: str,
    active_key_field: str,
    explicit_key: str | None = None,
) -> tuple[str, dict[str, Any]]:
    group_config = config.get("models", {}).get(group, {})
    active_key = explicit_key or group_config.get(active_key_field)
    if not active_key:
        raise ValueError(f"models.{group}.{active_key_field} is required")
    profiles = group_config.get(collection, {})
    profile = profiles.get(active_key)
    if not isinstance(profile, dict):
        raise KeyError(f"Model profile not found: models.{group}.{collection}.{active_key}")
    return active_key, profile


def resolve_generation_runtime(config: dict[str, Any]) -> dict[str, Any]:
    generation_cfg = config.get("generation", {})
    _, vlm_profile = get_model_profile(
        config,
        group="generation",
        collection="vlm",
        active_key_field="active_vlm",
        explicit_key=generation_cfg.get("vlm_model_key"),
    )
    _, image_profile = get_model_profile(
        config,
        group="generation",
        collection="image_generators",
        active_key_field="active_image_generator",
        explicit_key=generation_cfg.get("image_model_key"),
    )

    env = {}
    for profile in (vlm_profile, image_profile):
        credential = get_credentials(config, profile.get("credential_ref"))
        api_key = _non_empty(profile.get("api_key")) or _non_empty(credential.get("api_key"))
        api_key_env = profile.get("api_key_env") or credential.get("api_key_env")
        if api_key and api_key_env:
            env[api_key_env] = api_key

        endpoint = _non_empty(profile.get("endpoint"))
        endpoint_env = profile.get("endpoint_env")
        if endpoint and endpoint_env:
            env[endpoint_env] = endpoint

    return {
        "vlm_model_name": vlm_profile.get("model_name"),
        "image_model_name": image_profile.get("model_name"),
        "vlm_profile": vlm_profile,
        "image_profile": image_profile,
        "env": env,
    }


def resolve_classification_runtime(config: dict[str, Any]) -> dict[str, Any]:
    classification_cfg = config.get("classification", {})
    _, profile = get_model_profile(
        config,
        group="classification",
        collection="candidates",
        active_key_field="active_model",
        explicit_key=classification_cfg.get("model_key"),
    )
    credential = get_credentials(config, profile.get("credential_ref"))
    merged = deep_merge(profile, classification_cfg)
    merged["api_key"] = _non_empty(merged.get("api_key")) or _non_empty(credential.get("api_key"))
    merged["api_url"] = (
        _non_empty(merged.get("api_url"))
        or _non_empty(merged.get("base_url"))
        or _non_empty(credential.get("api_url"))
        or _non_empty(credential.get("base_url"))
    )
    merged["model"] = _non_empty(merged.get("model")) or _non_empty(merged.get("model_name"))
    merged["classifier_name"] = (
        _non_empty(merged.get("classifier_name"))
        or _non_empty(merged.get("model_name"))
        or _non_empty(merged.get("name"))
    )
    return merged


def build_detector_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    detector_config = deepcopy(config.get("detector_services", {}))
    geometry_profiles = deepcopy(config.get("models", {}).get("geometry", {}).get("candidates", {}))
    for profile in geometry_profiles.values():
        if not isinstance(profile, dict):
            continue
        credential = get_credentials(config, profile.get("credential_ref"))
        profile["api_key"] = _non_empty(profile.get("api_key")) or _non_empty(credential.get("api_key"))
        profile["base_url"] = (
            _non_empty(profile.get("base_url"))
            or _non_empty(profile.get("api_url"))
            or _non_empty(credential.get("base_url"))
            or _non_empty(credential.get("api_url"))
        )
    detector_config["model_profiles"] = geometry_profiles
    return detector_config
