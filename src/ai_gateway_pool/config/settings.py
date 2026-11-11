"""Process settings (environment variables) and YAML loading.

Split of responsibilities, with no overlap:

* environment variables (``POOL_*``) say WHICH file to load, which environment
  the process expects, and carry secrets (admin token);
* the YAML file (:mod:`.schema`) describes nodes and policies.

Startup fails if the two disagree on the environment.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_gateway_pool.config.schema import ConfigStatus, PoolConfig
from ai_gateway_pool.errors import ConfigurationInvalidError
from ai_gateway_pool.nodes.models import Environment


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POOL_", extra="ignore")

    # Required: no default environment, in particular never a production default.
    environment: Environment
    config_file: Path
    # When unset, the /internal/* routes are disabled (404).
    admin_token: SecretStr | None = None
    # Deployment manifests set this to 1: `status: example` files are refused.
    require_approved_config: bool = False
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR)$")
    # Local runs only; containers bind 0.0.0.0 inside their own network namespace.
    bind_host: str = "127.0.0.1"
    bind_port: int = Field(default=18080, ge=1024, le=65535)


def load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        missing = ", ".join(
            "POOL_" + str(e["loc"][0]).upper() for e in exc.errors() if e.get("loc")
        )
        raise ConfigurationInvalidError(f"invalid or missing settings: {missing}") from exc


def load_pool_config(path: Path) -> PoolConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationInvalidError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationInvalidError(f"config file is not valid YAML: {path}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationInvalidError(f"config file must be a mapping: {path}")
    try:
        return PoolConfig.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigurationInvalidError(f"{path}: {details}") from exc


def load_for_settings(settings: Settings) -> PoolConfig:
    config = load_pool_config(settings.config_file)
    if config.environment != settings.environment:
        raise ConfigurationInvalidError(
            f"POOL_ENVIRONMENT={settings.environment} but {settings.config_file} declares "
            f"environment={config.environment}"
        )
    if settings.require_approved_config and config.status is not ConfigStatus.APPROVED:
        raise ConfigurationInvalidError(
            f"{settings.config_file} has status={config.status}; deployments require an "
            "approved configuration"
        )
    return config
