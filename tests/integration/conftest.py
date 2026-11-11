"""Fixtures: a real fake-Gateway HTTP server and a real pool HTTP server in front."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_gateway_pool.config.schema import PoolConfig
from ai_gateway_pool.config.settings import Settings
from ai_gateway_pool.main import create_app
from tests.conftest import ADMIN_TOKEN, node, raw_config
from tests.integration.live_servers import FakeState, ServerThread, fake_gateway_app

# Every capability that has a route in gateway-api/1.0 (STRUCTURED_EXTRACT has none: Open edition).
ALL_CAPS = ["GENERATE", "CHAT", "EMBEDDINGS", "RERANK", "VISION", "ASR"]
MODELS = [
    {"model_id": "legacy-768", "capabilities": ["EMBEDDINGS"], "dimension": 768},
    {"model_id": "nomic-embed-text:latest", "capabilities": ["EMBEDDINGS"], "dimension": 768},
    {"model_id": "llama-chat", "capabilities": ["GENERATE", "CHAT"]},
    {"model_id": "llama3.1:8b", "capabilities": ["GENERATE", "CHAT"]},
    {"model_id": "rerank-default", "capabilities": ["RERANK"]},
    {"model_id": "vision-large", "capabilities": ["VISION", "GENERATE", "CHAT"]},
    {"model_id": "asr-whisper-turbo", "capabilities": ["ASR"]},
]


# POOL-3: implicit-model requests between several nodes require one identical
# published default per capability, so the fake nodes declare theirs.
DEFAULTS = {
    "EMBEDDINGS": "legacy-768",
    "GENERATE": "llama-chat",
    "RERANK": "rerank-default",
    "VISION": "vision-large",
    "ASR": "asr-whisper-turbo",
}


def full_node(url: str, node_id: str = "fake-staging", env: str = "staging") -> dict[str, Any]:
    return node(
        node_id,
        env=env,
        url=url,
        capabilities=ALL_CAPS,
        models=MODELS,
        default_models=dict(DEFAULTS),
    )


def pool_config(
    fake_url: str, tmp_path: Path, *, env: str = "staging", backend: str = "sqlite", **extra: Any
) -> PoolConfig:
    ja: dict[str, Any] = {"backend": backend}
    if backend == "sqlite":
        ja["sqlite_path"] = str(tmp_path / "affinity.sqlite3")
    data = raw_config(
        env=env, nodes=[full_node(fake_url, f"fake-{env}", env=env)], job_affinity=ja, **extra
    )
    return PoolConfig.model_validate(data)


def pool_server(config: PoolConfig, admin_token: str | None = ADMIN_TOKEN) -> ServerThread:
    settings = Settings(
        environment=config.environment,
        config_file=Path("/nonexistent/in-memory.yaml"),
        admin_token=admin_token,
        log_level="WARNING",
    )
    return ServerThread(create_app(settings, config))


@pytest.fixture
def fake() -> Iterator[tuple[FakeState, ServerThread]]:
    state = FakeState(node="fake-staging")
    with ServerThread(fake_gateway_app(state)) as srv:
        yield state, srv


@pytest.fixture
def make_pool(tmp_path: Path) -> Iterator[Callable[..., ServerThread]]:
    started: list[ServerThread] = []

    def _make(fake_url: str, **kw: Any) -> ServerThread:
        srv = pool_server(pool_config(fake_url, tmp_path, **kw)).__enter__()
        started.append(srv)
        return srv

    yield _make
    for srv in reversed(started):
        srv.__exit__(None, None, None)


@pytest.fixture
def pool(fake: tuple[FakeState, ServerThread], make_pool: Callable[..., ServerThread]) -> str:
    return make_pool(fake[1].url).url


@pytest.fixture
def http() -> Iterator[httpx.Client]:
    with httpx.Client(timeout=30, trust_env=False) as client:
        yield client
