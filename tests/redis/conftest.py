"""Fixtures for tests against a REAL Redis server.

``AIGP_TEST_REDIS_URL`` must point at an isolated qualification Redis (never a
shared or production one). ``AIGP_TEST_REDIS_CONTAINER`` (optional) names its
Docker container so outage/recovery can be simulated with ``docker pause``.
Each test uses a random key prefix and never flushes the database.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import subprocess
from collections.abc import AsyncIterator, Iterator

import pytest
import redis.asyncio as aioredis

from ai_gateway_pool.jobs.affinity import RedisJobAffinityStore

REDIS_URL = os.environ.get("AIGP_TEST_REDIS_URL")
CONTAINER = os.environ.get("AIGP_TEST_REDIS_CONTAINER")

pytestmark = pytest.mark.redis_real

requires_redis = pytest.mark.skipif(not REDIS_URL, reason="AIGP_TEST_REDIS_URL not set")
requires_container = pytest.mark.skipif(
    not (REDIS_URL and CONTAINER), reason="AIGP_TEST_REDIS_CONTAINER not set"
)


@pytest.fixture
def prefix() -> str:
    # "aigp*" prefix: the staging Redis ACL only grants ~aigp* keys to the pool user.
    return "aigpq" + secrets.token_hex(6)


def client() -> aioredis.Redis:
    assert REDIS_URL
    return aioredis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=1)


@pytest.fixture
async def redis_client(prefix: str) -> AsyncIterator[aioredis.Redis]:
    c = client()
    yield c
    keys = [k async for k in c.scan_iter(f"{prefix}:*")]
    if keys:
        await c.delete(*keys)
    await c.aclose()


@pytest.fixture
def store(redis_client: aioredis.Redis, prefix: str) -> RedisJobAffinityStore:
    return RedisJobAffinityStore(redis_client, key_prefix=prefix)


@contextlib.contextmanager
def pause_redis() -> Iterator[None]:
    """Pause ONLY the isolated qualification container during the block."""

    assert CONTAINER
    subprocess.run(["docker", "pause", CONTAINER], check=True, capture_output=True)
    try:
        yield
    finally:
        subprocess.run(["docker", "unpause", CONTAINER], check=True, capture_output=True)
