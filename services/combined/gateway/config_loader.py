from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


_DEFAULTS: dict[str, Any] = {
    "DB_PATH": "/data/controlplane.db",
    "GLOBAL_MAX_CONCURRENT": 128,
    "MAX_BODY_BYTES": 1024 * 1024,
    "VLLM_PORT": 8001,
    "UPSTREAM_TIMEOUT_SECONDS": 300.0,
    "BUILD_SHA": "dev",
    "BUILD_TIME": "dev",
    "DEFAULT_MAX_CONCURRENT": 4,
    "DEFAULT_RPM_LIMIT": 120,
    "DEFAULT_TPM_LIMIT": 120000,
    "DEFAULT_MAX_CONTEXT_TOKENS": 8192,
    "DEFAULT_MAX_OUTPUT_TOKENS": 512,
    "TRUST_REMOTE_CODE": False,
    "METRICS_TENANT_LABELS": "on",
    "GPU_METRICS_POLL_INTERVAL_SECONDS": 2.0,
    "GPU_HOURLY_RATE": 0.0,
    "ADMIN_AUTH_MODE": "static_token",
    "ROLE_CLAIM": "role",
    "GROUPS_CLAIM": "groups",
    "METRICS_ENABLED": True,
    "UI_ENABLED": True,
    "PROFILES_ENABLED": True,
    "PROFILES_PATH": "/app/profiles",
    "UI_PATH": "/app/gateway/ui",
}


class ConfigError(RuntimeError):
    pass


def _normalize_yaml_data(data: Any) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError("config.yaml must be a mapping at the top level")

    out: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        out[key] = value
        out[key.upper()] = value
    return out


def _parse_bool(raw: Any, key: str) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in _BOOL_TRUE:
        return True
    if text in _BOOL_FALSE:
        return False
    raise ConfigError(f"{key} must be a boolean-like value")


def _parse_int(raw: Any, key: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be an integer") from exc


def _parse_float(raw: Any, key: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be a number") from exc


def _read_yaml_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"failed to parse config file {config_path}: {exc}") from exc

    return _normalize_yaml_data(payload)


def _resolve_value(key: str, yaml_cfg: dict[str, Any]) -> Any:
    if key in os.environ:
        return os.environ[key]
    if key in yaml_cfg:
        return yaml_cfg[key]
    if key.upper() in yaml_cfg:
        return yaml_cfg[key.upper()]
    return _DEFAULTS.get(key)


def load_config() -> dict[str, Any]:
    config_path = os.getenv("CONFIG_PATH", "/app/config/config.yaml")
    yaml_cfg = _read_yaml_config(config_path)

    config: dict[str, Any] = {}
    config["CONFIG_PATH"] = config_path

    model_id = _resolve_value("MODEL_ID", yaml_cfg)
    if model_id is None or not str(model_id).strip():
        raise ConfigError("MODEL_ID is required (env or config.yaml)")
    config["MODEL_ID"] = str(model_id).strip()

    for key in [
        "DB_PATH",
        "BUILD_SHA",
        "BUILD_TIME",
        "METRICS_TENANT_LABELS",
        "ADMIN_AUTH_MODE",
        "ADMIN_TOKEN",
        "JWKS_URL",
        "OIDC_ISSUER",
        "OIDC_AUDIENCE",
        "ROLE_CLAIM",
        "GROUPS_CLAIM",
        "ADMIN_GROUP",
        "PROFILES_PATH",
        "UI_PATH",
    ]:
        value = _resolve_value(key, yaml_cfg)
        config[key] = None if value is None else str(value)

    for key in [
        "GLOBAL_MAX_CONCURRENT",
        "MAX_BODY_BYTES",
        "VLLM_PORT",
        "DEFAULT_MAX_CONCURRENT",
        "DEFAULT_RPM_LIMIT",
        "DEFAULT_TPM_LIMIT",
        "DEFAULT_MAX_CONTEXT_TOKENS",
        "DEFAULT_MAX_OUTPUT_TOKENS",
    ]:
        config[key] = _parse_int(_resolve_value(key, yaml_cfg), key)

    for key in [
        "UPSTREAM_TIMEOUT_SECONDS",
        "GPU_METRICS_POLL_INTERVAL_SECONDS",
        "GPU_HOURLY_RATE",
    ]:
        config[key] = _parse_float(_resolve_value(key, yaml_cfg), key)

    for key in [
        "TRUST_REMOTE_CODE",
        "METRICS_ENABLED",
        "UI_ENABLED",
        "PROFILES_ENABLED",
    ]:
        config[key] = _parse_bool(_resolve_value(key, yaml_cfg), key)

    raw_metrics_labels = config["METRICS_TENANT_LABELS"]
    if isinstance(raw_metrics_labels, bool):
        metrics_tenant_labels = "on" if raw_metrics_labels else "off"
    else:
        metrics_tenant_labels = str(raw_metrics_labels).strip().lower()
        if metrics_tenant_labels in _BOOL_TRUE:
            metrics_tenant_labels = "on"
        elif metrics_tenant_labels in _BOOL_FALSE:
            metrics_tenant_labels = "off"

    if metrics_tenant_labels not in {"on", "off"}:
        raise ConfigError("METRICS_TENANT_LABELS must be one of: on, off")
    config["METRICS_TENANT_LABELS"] = metrics_tenant_labels

    admin_auth_mode = str(config["ADMIN_AUTH_MODE"] or "static_token").strip().lower()
    if admin_auth_mode not in {"static_token", "oidc"}:
        raise ConfigError("ADMIN_AUTH_MODE must be static_token or oidc")
    config["ADMIN_AUTH_MODE"] = admin_auth_mode

    if config["GLOBAL_MAX_CONCURRENT"] < 1:
        raise ConfigError("GLOBAL_MAX_CONCURRENT must be >= 1")
    if config["MAX_BODY_BYTES"] < 1024:
        raise ConfigError("MAX_BODY_BYTES must be >= 1024")
    if config["UPSTREAM_TIMEOUT_SECONDS"] <= 0:
        raise ConfigError("UPSTREAM_TIMEOUT_SECONDS must be > 0")
    if config["GPU_METRICS_POLL_INTERVAL_SECONDS"] <= 0:
        raise ConfigError("GPU_METRICS_POLL_INTERVAL_SECONDS must be > 0")
    if config["GPU_HOURLY_RATE"] < 0:
        raise ConfigError("GPU_HOURLY_RATE must be >= 0")

    return config
