from __future__ import annotations

from copy import deepcopy
import os
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


def _normalize_name_list(value: Any) -> list[str]:
    if value in ("", None, False):
        return []
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized.lower() in {"false", "none", "null"}:
            return []
        return [item.strip() for item in normalized.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        names: list[str] = []
        for item in value:
            for name in _normalize_name_list(item):
                if name not in names:
                    names.append(name)
        return names
    return [str(value)]


def _normalize_cli_args(value: Any) -> list[str]:
    if value in ("", None, False):
        return []
    if isinstance(value, str):
        return [item for item in value.split() if item]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _normalize_env_names(value: Any) -> list[str]:
    names: list[str] = []
    for name in _normalize_name_list(value):
        if name not in names:
            names.append(name)
    return names


def _first_env_value(names: list[str]) -> str | None:
    for name in names:
        value = _non_empty(os.getenv(name))
        if value:
            return str(value)
    return None


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


def resolve_localizer_config(config: dict[str, Any], anomaly_type: str | None = None) -> dict[str, Any]:
    generation_cfg = config.get("generation", {})
    module_generation_cfg = config.get("modules", {}).get("generation", {})

    localizer_cfg: dict[str, Any] = {}
    if isinstance(module_generation_cfg.get("localizer"), dict):
        localizer_cfg = deep_merge(localizer_cfg, module_generation_cfg["localizer"])
    if isinstance(generation_cfg.get("localizer"), dict):
        localizer_cfg = deep_merge(localizer_cfg, generation_cfg["localizer"])

    policy_cfg: dict[str, Any] = {}
    if isinstance(module_generation_cfg.get("localizer_policy"), dict):
        policy_cfg = deep_merge(policy_cfg, module_generation_cfg["localizer_policy"])
    if isinstance(generation_cfg.get("localizer_policy"), dict):
        policy_cfg = deep_merge(policy_cfg, generation_cfg["localizer_policy"])

    if anomaly_type and isinstance(policy_cfg.get(anomaly_type), dict):
        localizer_cfg = deep_merge(localizer_cfg, policy_cfg[anomaly_type])
    return localizer_cfg


def resolve_generation_runtime(config: dict[str, Any], anomaly_type: str | None = None) -> dict[str, Any]:
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
        api_key_env_names = _normalize_env_names(profile.get("api_key_env") or credential.get("api_key_env"))
        api_key_env_names.extend(
            name for name in _normalize_env_names(profile.get("api_key_env_aliases")) if name not in api_key_env_names
        )
        api_key = (
            _non_empty(profile.get("api_key"))
            or _non_empty(credential.get("api_key"))
            or _first_env_value(api_key_env_names)
        )
        if api_key:
            for api_key_env in api_key_env_names:
                env[api_key_env] = api_key

        endpoint = _non_empty(profile.get("endpoint"))
        endpoint_env_names = _normalize_env_names(profile.get("endpoint_env"))
        endpoint_env_names.extend(
            name for name in _normalize_env_names(profile.get("endpoint_env_aliases")) if name not in endpoint_env_names
        )
        if endpoint:
            for endpoint_env in endpoint_env_names:
                env[endpoint_env] = endpoint

    localizer_cfg = resolve_localizer_config(config, anomaly_type=anomaly_type)

    return {
        "vlm_model_name": vlm_profile.get("model_name"),
        "image_model_name": image_profile.get("model_name"),
        "vlm_profile": vlm_profile,
        "image_profile": image_profile,
        "env": env,
        "localizer_config": localizer_cfg,
        "localizer_cli_args": build_generation_extra_cli_args(localizer_cfg),
        # Localizer is applied during metadata ingest inside this repo. Keep
        # external generation CLI args separate so old I2I entrypoints do not
        # fail on repo-local flags such as --localizer.
        "extra_cli_args": _normalize_cli_args(generation_cfg.get("extra_cli_args")),
    }


def build_generation_extra_cli_args(localizer_cfg: dict[str, Any]) -> list[str]:
    if not localizer_cfg:
        return []

    args: list[str] = []

    primary = _non_empty(localizer_cfg.get("primary")) or _non_empty(localizer_cfg.get("name"))
    if primary:
        args.extend(["--localizer", str(primary)])

    fallback = _non_empty(localizer_cfg.get("fallback"))
    if fallback:
        args.extend(["--localizer-fallback", str(fallback)])

    if bool(localizer_cfg.get("debug")):
        args.append("--localizer-debug")

    sidecar_eval = _normalize_name_list(localizer_cfg.get("sidecar_eval"))
    if sidecar_eval:
        args.extend(["--localizer-sidecar-eval", ",".join(sidecar_eval)])

    pgcd_cfg = localizer_cfg.get("pgcd", {}) if isinstance(localizer_cfg.get("pgcd"), dict) else {}
    if _non_empty(pgcd_cfg.get("threshold")):
        args.extend(["--pgcd-threshold", str(pgcd_cfg["threshold"])])
    if _non_empty(pgcd_cfg.get("min_component_area")) is not None:
        args.extend(["--pgcd-min-component-area", str(pgcd_cfg["min_component_area"])])
    if _non_empty(pgcd_cfg.get("max_global_change_ratio")) is not None:
        args.extend(["--pgcd-max-global-change-ratio", str(pgcd_cfg["max_global_change_ratio"])])

    prior_weights = (
        pgcd_cfg.get("prompt_prior_weights", {})
        if isinstance(pgcd_cfg.get("prompt_prior_weights"), dict)
        else {}
    )
    if _non_empty(prior_weights.get("area")) is not None:
        args.extend(["--pgcd-prompt-prior-weight-area", str(prior_weights["area"])])
    if _non_empty(prior_weights.get("lpips")) is not None:
        args.extend(["--pgcd-prompt-prior-weight-lpips", str(prior_weights["lpips"])])
    if _non_empty(prior_weights.get("iou")) is not None:
        args.extend(["--pgcd-prompt-prior-weight-iou", str(prior_weights["iou"])])
    if _non_empty(prior_weights.get("distance")) is not None:
        args.extend(["--pgcd-prompt-prior-weight-distance", str(prior_weights["distance"])])

    sam2_cfg = localizer_cfg.get("sam2", {}) if isinstance(localizer_cfg.get("sam2"), dict) else {}
    sam2_enabled = bool(sam2_cfg.get("enabled")) or primary == "pgcd_lpips_sam2"
    if sam2_enabled:
        args.append("--sam2-enabled")
    if _non_empty(sam2_cfg.get("model")):
        args.extend(["--sam2-model", str(sam2_cfg["model"])])

    return args


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
