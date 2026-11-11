"""Shared test helpers. No test contacts a real GPU host or any real network host."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from ai_gateway_pool.config.schema import PoolConfig
from ai_gateway_pool.config.settings import Settings
from ai_gateway_pool.main import create_app

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
ADMIN_TOKEN = "test-admin-token-not-a-secret-0123456789"


def gateway_fixture(name: str) -> Any:
    return json.loads((FIXTURES / "gateway" / name).read_text())


def node(
    node_id: str = "gw-staging",
    *,
    env: str = "staging",
    url: str = "http://gw-staging.invalid:8200",
    host: str = "gpu-host-a",
    **extra: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "node_id": node_id,
        "physical_host_id": host,
        "display_name": node_id,
        "environment": env,
        "gateway_instance": env,
        "base_url": url,
        "capabilities": ["GENERATE", "CHAT", "EMBEDDINGS", "RERANK", "ASR"],
        "models": [
            {"model_id": "legacy-768", "capabilities": ["EMBEDDINGS"], "dimension": 768},
            {"model_id": "llama-chat", "capabilities": ["GENERATE", "CHAT"]},
            {"model_id": "rerank-default", "capabilities": ["RERANK"]},
            {"model_id": "asr-whisper-turbo", "capabilities": ["ASR"]},
        ],
    }
    base.update(extra)
    return base


def raw_config(
    env: str = "staging", nodes: list[dict[str, Any]] | None = None, **extra: Any
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "schema_version": 1,
        "status": "example",
        "environment": env,
        "routing": {"policy": "SINGLE_NODE"},
        "nodes": nodes if nodes is not None else [node(env=env)],
    }
    cfg.update(extra)
    return copy.deepcopy(cfg)


def build_config(**kwargs: Any) -> PoolConfig:
    return PoolConfig.model_validate(raw_config(**kwargs))


def write_yaml(tmp_path: Path, data: dict[str, Any], name: str = "pool.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def staging_config() -> PoolConfig:
    return build_config()


def make_client(config: PoolConfig, admin_token: str | None = ADMIN_TOKEN) -> TestClient:
    settings = Settings(
        environment=config.environment,
        config_file=Path("/nonexistent/in-memory.yaml"),
        admin_token=admin_token,
    )
    return TestClient(create_app(settings, config))


@pytest.fixture
def client(staging_config: PoolConfig) -> TestClient:
    return make_client(staging_config)
