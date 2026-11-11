"""POOL-PROD-2: stale Redis connection after a server-side disconnect (REAL Redis).

Deterministic reproduction of the production observation "first affinity
operation after a Redis restart -> 503": a connection of the pool's client is
closed by the SERVER (``CLIENT KILL``, exactly what a restart does to every
established connection) while it sits idle in the client's connection pool.

The isolated qualification Redis must allow ``CLIENT LIST`` / ``CLIENT KILL``
(true for ``check-local.sh``'s throw-away Redis; the tests skip otherwise).
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import pytest
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from ai_gateway_pool.config.schema import PoolConfig
from ai_gateway_pool.errors import PoolError, PoolErrorCode
from ai_gateway_pool.jobs.affinity import RedisJobAffinityStore
from ai_gateway_pool.jobs.contracts import AffinityStoreUnavailable
from ai_gateway_pool.jobs.factory import redis_client_from_url
from ai_gateway_pool.nodes.models import Environment
from tests.conftest import raw_config
from tests.integration.conftest import full_node, pool_server
from tests.integration.live_servers import FakeState, ServerThread, fake_gateway_app
from tests.redis.conftest import (
    REDIS_URL,
    client,
    pause_redis,
    requires_container,
    requires_redis,
)

pytestmark = [pytest.mark.redis_real, requires_redis]
PRD = Environment.PRODUCTION


async def kill_clients(name: str) -> int:
    """Server-side close of every connection whose CLIENT name is ``name``."""

    admin = client()
    try:
        killed = 0
        for c in await admin.client_list():
            if c.get("name") == name:
                await admin.client_kill_filter(_id=c["id"])
                killed += 1
        return killed
    except ResponseError as exc:  # pragma: no cover - restricted Redis
        pytest.skip(f"CLIENT KILL not allowed on this Redis: {exc}")
    finally:
        await admin.aclose()


@pytest.fixture
async def victim(prefix: str) -> AsyncIterator[tuple[RedisJobAffinityStore, aioredis.Redis, str]]:
    """A store built EXACTLY like production (``redis_client_from_url``), named for KILL."""

    assert REDIS_URL
    name = "aigpq-victim-" + secrets.token_hex(4)
    c = redis_client_from_url(REDIS_URL, timeout_seconds=1, client_name=name)
    yield RedisJobAffinityStore(c, key_prefix=prefix), c, name
    await c.aclose()
    cleaner = client()
    keys = [k async for k in cleaner.scan_iter(f"{prefix}:*")]
    if keys:
        await cleaner.delete(*keys)
    await cleaner.aclose()


async def test_stale_connection_is_reproduced_at_library_level(
    victim: tuple[RedisJobAffinityStore, aioredis.Redis, str],
) -> None:
    """The defect itself (independent of the fix): a server-closed pooled connection
    makes the next command raise ConnectionError; redis-py does not retry (retries=0)."""

    _, raw, name = victim
    assert await raw.ping()  # REDIS_CONNECTION = ESTABLISHED (pooled, idle)
    conn = await raw.connection_pool.get_connection()
    assert conn.retry._retries == 0  # no hidden library retry with from_url
    await raw.connection_pool.release(conn)
    assert await kill_clients(name) == 1  # REDIS_SERVER side close -> CLIENT_CONNECTION = STALE
    with pytest.raises(RedisConnectionError, match="Connection closed by server"):
        await raw.get("aigpq-any")  # FIRST_REDIS_OPERATION = CONNECTION_ERROR
    assert await raw.get("aigpq-any") is None  # the broken connection was discarded


async def test_lookup_recovers_after_server_side_close(
    victim: tuple[RedisJobAffinityStore, aioredis.Redis, str],
) -> None:
    store, _, name = victim
    await store.register_job(PRD, "job-1", "node-a", 600)
    assert await kill_clients(name) >= 1
    found = await store.lookup_job(PRD, "job-1")  # was: 503 AFFINITY_STORE_UNAVAILABLE
    assert found is not None and found.node_id == "node-a"
    assert store.reconnects == 1


async def test_ping_recovers_after_server_side_close(
    victim: tuple[RedisJobAffinityStore, aioredis.Redis, str],
) -> None:
    store, _, name = victim
    await store.ping()
    await kill_clients(name)
    await store.ping()  # readiness and the ASR pre-check no longer flap to 503
    assert store.reconnects == 1


async def test_register_after_server_side_close_writes_exactly_one_mapping(
    victim: tuple[RedisJobAffinityStore, aioredis.Redis, str], prefix: str
) -> None:
    store, _, name = victim
    await store.ping()
    await kill_clients(name)
    item = await store.register_job(PRD, "job-2", "node-a", 600)
    assert item.node_id == "node-a"
    c = client()
    keys = [k async for k in c.scan_iter(f"{prefix}:*")]
    await c.aclose()
    assert keys == [f"{prefix}:production:asr-job:job-2".encode()]


async def test_forget_after_server_side_close_keeps_identity_condition(
    victim: tuple[RedisJobAffinityStore, aioredis.Redis, str],
) -> None:
    store, _, name = victim
    await store.register_job(PRD, "job-3", "node-a", 600)
    await kill_clients(name)
    assert await store.forget_job(PRD, "job-3", node_id="node-b") is False  # other node: kept
    assert (await store.lookup_job(PRD, "job-3")) is not None
    await kill_clients(name)
    assert await store.forget_job(PRD, "job-3", node_id="node-a") is True
    assert await store.lookup_job(PRD, "job-3") is None


async def test_concurrent_requests_after_all_connections_went_stale(
    victim: tuple[RedisJobAffinityStore, aioredis.Redis, str],
) -> None:
    store, _, name = victim
    for i in range(8):
        await store.register_job(PRD, f"c-{i}", "node-a", 600)
    # grow the pool to several idle connections, then close all of them server-side
    await asyncio.gather(*(store.lookup_job(PRD, f"c-{i}") for i in range(8)))
    assert await kill_clients(name) >= 2
    found = await asyncio.gather(*(store.lookup_job(PRD, f"c-{i}") for i in range(8)))
    assert [f.node_id if f else None for f in found] == ["node-a"] * 8


async def test_wrong_password_fails_closed_without_reconnect() -> None:
    assert REDIS_URL
    parts = urlsplit(REDIS_URL)
    netloc = "aigpq-nobody:wrong@" + parts.netloc.rsplit("@", 1)[-1]
    bad = redis_client_from_url(urlunsplit(parts._replace(netloc=netloc)), timeout_seconds=1)
    store = RedisJobAffinityStore(bad, key_prefix="aigpqauth")
    with pytest.raises(AffinityStoreUnavailable, match="AuthenticationError"):
        await store.ping()
    assert store.reconnects == 0
    await bad.aclose()


@requires_container
async def test_redis_down_still_fails_closed_with_one_bounded_attempt(
    victim: tuple[RedisJobAffinityStore, aioredis.Redis, str],
) -> None:
    store, _, _ = victim
    await store.ping()
    t = time.perf_counter()
    with pause_redis(), pytest.raises(AffinityStoreUnavailable, match="TimeoutError"):
        await store.lookup_job(PRD, "x")  # timeout: ambiguous -> never retried
    assert time.perf_counter() - t < 1 + 1.5  # one socket timeout (1 s), no second wait
    assert store.reconnects == 0


async def test_asr_post_after_stale_connection_is_sent_once_and_announced(
    prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the real pool: the ASR pre-check hits the stale
    connection, recovers, and the Gateway receives exactly ONE POST."""

    assert REDIS_URL
    monkeypatch.setenv("POOL_TEST_REDIS_URL", REDIS_URL)
    state = FakeState(node="fake-staging")
    with ServerThread(fake_gateway_app(state)) as gw:
        data: dict[str, Any] = raw_config(
            nodes=[full_node(gw.url)],
            job_affinity={
                "backend": "redis",
                "redis_url_env": "POOL_TEST_REDIS_URL",
                "redis_key_prefix": prefix,
            },
        )
        with pool_server(PoolConfig.model_validate(data)) as pool:
            files = {
                "request": (None, b'{"model":"asr-whisper-turbo"}', "application/json"),
                "file": ("a", b"\x00synthetic", "application/octet-stream"),
            }
            async with httpx.AsyncClient(timeout=10, trust_env=False) as http:
                first = await http.post(pool.url + "/v1/audio/transcriptions", files=files)
                assert first.status_code == 202
                admin = client()
                killed = await admin.client_kill_filter(_type="normal", skipme=True)
                await admin.aclose()
                assert killed >= 1
                posts_before = sum(1 for r in state.received if r.method == "POST")
                second = await http.post(pool.url + "/v1/audio/transcriptions", files=files)
                posts_after = sum(1 for r in state.received if r.method == "POST")
                assert second.status_code == 202, second.text  # was: 503 before node selection
                assert posts_after - posts_before == 1  # HTTP_REQUEST_REPLAY = NEVER
                job = second.json()["job_id"]
                got = await http.get(f"{pool.url}/v1/audio/transcriptions/{job}")
                assert got.status_code == 200  # affinity registered and resolvable
    cleaner = client()
    keys = [k async for k in cleaner.scan_iter(f"{prefix}:*")]
    if keys:
        await cleaner.delete(*keys)
    await cleaner.aclose()


async def test_job_conflict_is_unchanged_after_reconnect(
    victim: tuple[RedisJobAffinityStore, aioredis.Redis, str],
) -> None:
    store, _, name = victim
    await store.register_job(PRD, "dup", "node-a", 600)
    await kill_clients(name)
    with pytest.raises(PoolError) as exc:
        await store.register_job(PRD, "dup", "node-b", 600)
    assert exc.value.code is PoolErrorCode.JOB_AFFINITY_CONFLICT
