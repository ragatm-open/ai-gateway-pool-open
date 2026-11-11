"""RedisJobAffinityStore against a REAL Redis server (no mock)."""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import time

import pytest
import redis.asyncio as aioredis

from ai_gateway_pool.errors import PoolError, PoolErrorCode
from ai_gateway_pool.jobs.affinity import RedisJobAffinityStore
from ai_gateway_pool.jobs.contracts import AffinityStoreUnavailable, JobStatus
from ai_gateway_pool.nodes.models import Environment
from tests.redis.conftest import REDIS_URL, client, requires_container, requires_redis

pytestmark = [pytest.mark.redis_real, requires_redis]
STG, PRD = Environment.STAGING, Environment.PRODUCTION


async def test_register_lookup_and_stored_value_is_metadata_only(
    store: RedisJobAffinityStore, redis_client: aioredis.Redis, prefix: str
) -> None:
    item = await store.register_job(STG, "job-1", "node-a", 3600)
    assert item.gateway_job_id == "job-1" and item.status is JobStatus.ACTIVE
    found = await store.lookup_job(STG, "job-1")
    assert found is not None and found.node_id == "node-a"
    raw = await redis_client.get(f"{prefix}:staging:asr-job:job-1")
    assert set(json.loads(raw)) == {
        "environment",
        "job_id",
        "node_id",
        "status",
        "created_at",
        "expires_at",
        "gateway_job_id",
    }
    ttl = await redis_client.ttl(f"{prefix}:staging:asr-job:job-1")
    assert 3500 < ttl <= 3600


async def test_environment_isolation(store: RedisJobAffinityStore) -> None:
    await store.register_job(STG, "same", "stg-node", 600)
    assert await store.lookup_job(PRD, "same") is None
    await store.register_job(PRD, "same", "prd-node", 600)
    assert (await store.lookup_job(STG, "same")).node_id == "stg-node"  # type: ignore[union-attr]
    assert (await store.lookup_job(PRD, "same")).node_id == "prd-node"  # type: ignore[union-attr]


async def test_value_tampered_to_other_environment_is_ignored(
    store: RedisJobAffinityStore, redis_client: aioredis.Redis, prefix: str
) -> None:
    await store.register_job(STG, "job-x", "n", 600)
    key = f"{prefix}:staging:asr-job:job-x"
    doc = json.loads(await redis_client.get(key))
    doc["environment"] = "production"
    await redis_client.set(key, json.dumps(doc))
    assert await store.lookup_job(STG, "job-x") is None  # defence in depth


async def test_conflict_never_overwrites_and_idempotent_rebind(
    store: RedisJobAffinityStore,
) -> None:
    await store.register_job(STG, "job-1", "node-a", 600)
    with pytest.raises(PoolError) as exc:
        await store.register_job(STG, "job-1", "node-b", 600)
    assert exc.value.code is PoolErrorCode.JOB_AFFINITY_CONFLICT
    assert (await store.lookup_job(STG, "job-1")).node_id == "node-a"  # type: ignore[union-attr]
    await store.register_job(STG, "job-1", "node-a", 600)


async def test_concurrent_registrations_single_winner(store: RedisJobAffinityStore) -> None:
    results = await asyncio.gather(
        *(store.register_job(STG, "race", f"node-{i}", 600) for i in range(50)),
        return_exceptions=True,
    )
    winners = [r for r in results if not isinstance(r, BaseException)]
    losers = [r for r in results if isinstance(r, BaseException)]
    assert len({w.node_id for w in winners}) == 1
    assert all(
        isinstance(e, PoolError) and e.code is PoolErrorCode.JOB_AFFINITY_CONFLICT for e in losers
    )


def _register_in_process(args: tuple[str, str, int, object]) -> str:
    url, prefix, i, barrier = args

    async def go() -> str:
        c = aioredis.from_url(url)
        s = RedisJobAffinityStore(c, key_prefix=prefix)
        barrier.wait()  # type: ignore[attr-defined]
        try:
            await s.register_job(STG, "proc-race", f"node-{i}", 600)
            return f"won:node-{i}"
        except PoolError as exc:
            return exc.code.value
        finally:
            await c.aclose()

    return asyncio.run(go())


def test_registration_race_between_os_processes(prefix: str) -> None:
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        barrier = manager.Barrier(6)
        with ctx.Pool(6) as pool:
            jobs = [(REDIS_URL, prefix, i, barrier) for i in range(6)]
            out = pool.map(_register_in_process, jobs)
    assert sum(1 for o in out if o.startswith("won:")) == 1
    assert out.count("JOB_AFFINITY_CONFLICT") == 5

    async def check() -> str | None:
        c = client()
        item = await RedisJobAffinityStore(c, key_prefix=prefix).lookup_job(STG, "proc-race")
        await c.delete(f"{prefix}:staging:asr-job:proc-race")
        await c.aclose()
        return item.node_id if item else None

    winner = next(o for o in out if o.startswith("won:")).removeprefix("won:")
    assert asyncio.run(check()) == winner


async def test_persistence_across_new_client(prefix: str, store: RedisJobAffinityStore) -> None:
    await store.register_job(STG, "job-p", "node-a", 600)
    fresh = client()  # a "restarted" router: new connection, same Redis
    try:
        item = await RedisJobAffinityStore(fresh, key_prefix=prefix).lookup_job(STG, "job-p")
        assert item is not None and item.node_id == "node-a"
    finally:
        await fresh.aclose()


async def test_completion_lua_updates_status_and_ttl(
    store: RedisJobAffinityStore, redis_client: aioredis.Redis, prefix: str
) -> None:
    await store.register_job(STG, "job-c", "node-a", 86400)
    done = await store.mark_job_completed(STG, "job-c", 120)
    assert done is not None and done.status is JobStatus.COMPLETED and done.node_id == "node-a"
    assert 100 < await redis_client.ttl(f"{prefix}:staging:asr-job:job-c") <= 120
    assert await store.mark_job_completed(STG, "missing", 120) is None


async def test_compare_and_delete_lua(store: RedisJobAffinityStore) -> None:
    await store.register_job(STG, "job-d", "node-a", 600)
    assert await store.forget_job(STG, "job-d", node_id="node-b") is False  # bound elsewhere
    assert await store.lookup_job(STG, "job-d") is not None
    assert await store.forget_job(STG, "job-d", node_id="node-a") is True
    assert await store.lookup_job(STG, "job-d") is None
    assert await store.forget_job(STG, "job-d") is False


async def test_real_expiration(store: RedisJobAffinityStore) -> None:
    await store.register_job(STG, "job-e", "node-a", 1)
    assert await store.lookup_job(STG, "job-e") is not None
    await asyncio.sleep(1.6)
    assert await store.lookup_job(STG, "job-e") is None
    # After expiry the id can be bound again (e.g. reused by the Gateway later).
    await store.register_job(STG, "job-e", "node-b", 60)


async def test_unreachable_redis_fails_closed_fast(prefix: str) -> None:
    dead = aioredis.from_url(
        "redis://127.0.0.1:1/0", socket_connect_timeout=0.5, socket_timeout=0.5
    )
    s = RedisJobAffinityStore(dead, key_prefix=prefix)
    t0 = time.perf_counter()
    with pytest.raises(AffinityStoreUnavailable):
        await s.ping()
    with pytest.raises(AffinityStoreUnavailable):
        await s.register_job(STG, "job-u", "n", 60)
    with pytest.raises(AffinityStoreUnavailable):
        await s.lookup_job(STG, "job-u")
    assert time.perf_counter() - t0 < 5
    await dead.aclose()


async def test_wrong_password_fails_closed(prefix: str) -> None:
    assert REDIS_URL
    if "@" not in REDIS_URL:
        pytest.skip("qualification Redis has no password")
    bad = REDIS_URL.split("://", 1)[0] + "://:wrong@" + REDIS_URL.split("@", 1)[1]
    c = aioredis.from_url(bad, socket_connect_timeout=1, socket_timeout=1)
    with pytest.raises(AffinityStoreUnavailable):
        await RedisJobAffinityStore(c, key_prefix=prefix).ping()
    await c.aclose()


@requires_container
async def test_outage_and_reconnection_keep_mappings(store: RedisJobAffinityStore) -> None:
    from tests.redis.conftest import pause_redis

    await store.register_job(STG, "job-o", "node-a", 600)
    with pause_redis():
        with pytest.raises(AffinityStoreUnavailable):
            await store.ping()
        with pytest.raises(AffinityStoreUnavailable):
            await store.lookup_job(STG, "job-o")
    for _ in range(20):  # the client reconnects by itself
        try:
            await store.ping()
            break
        except AffinityStoreUnavailable:
            await asyncio.sleep(0.25)
    item = await store.lookup_job(STG, "job-o")
    assert item is not None and item.node_id == "node-a"  # nothing lost
