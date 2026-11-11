"""Build the configured JobAffinityStore (or None when ASR routing is disabled)."""

from __future__ import annotations

import os
from typing import Any

from ai_gateway_pool.config.schema import PoolConfig
from ai_gateway_pool.errors import ConfigurationInvalidError
from ai_gateway_pool.jobs.affinity import InMemoryJobAffinityStore, RedisJobAffinityStore
from ai_gateway_pool.jobs.contracts import JobAffinityStore
from ai_gateway_pool.jobs.sqlite_store import SqliteJobAffinityStore


def build_job_store(config: PoolConfig) -> JobAffinityStore | None:
    ja = config.job_affinity
    if ja.backend == "none":
        return None
    if ja.backend == "memory":
        return InMemoryJobAffinityStore()
    if ja.backend == "sqlite":
        assert ja.sqlite_path is not None  # noqa: S101 - schema guarantees it
        return SqliteJobAffinityStore(ja.sqlite_path)
    # redis
    assert ja.redis_url_env is not None  # noqa: S101 - schema guarantees it
    url = os.environ.get(ja.redis_url_env)
    if not url:
        raise ConfigurationInvalidError(f"job_affinity: env var {ja.redis_url_env} is not set")
    if not url.startswith(("redis://", "rediss://", "unix://")):
        raise ConfigurationInvalidError(f"job_affinity: {ja.redis_url_env} is not a redis URL")
    client = redis_client_from_url(url, timeout_seconds=ja.redis_timeout_seconds)
    return RedisJobAffinityStore(client, key_prefix=ja.redis_key_prefix)


def redis_client_from_url(
    url: str, *, timeout_seconds: float, client_name: str | None = None
) -> Any:
    """The ONE place where the pool's Redis client is configured (tests reuse it).

    Bounded timeouts: a dead Redis must surface as AFFINITY_STORE_UNAVAILABLE
    quickly, never hang a request. No library-level retry (``from_url`` yields
    connections with ``retries=0``) and no fallback store.
    """

    import redis.asyncio as aioredis

    kwargs: dict[str, Any] = {
        "socket_connect_timeout": timeout_seconds,
        "socket_timeout": timeout_seconds,
        "health_check_interval": 15,
        "retry_on_timeout": False,
    }
    if client_name is not None:
        kwargs["client_name"] = client_name
    return aioredis.from_url(url, **kwargs)
