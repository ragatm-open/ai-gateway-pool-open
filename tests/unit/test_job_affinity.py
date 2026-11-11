"""JobAffinityStore guarantees, run against every implementation."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ai_gateway_pool.errors import PoolError, PoolErrorCode
from ai_gateway_pool.jobs.affinity import InMemoryJobAffinityStore, RedisJobAffinityStore
from ai_gateway_pool.jobs.contracts import AffinityStoreUnavailable, JobStatus, affinity_key
from ai_gateway_pool.jobs.sqlite_store import SqliteJobAffinityStore
from ai_gateway_pool.nodes.models import Environment

STG, PRD = Environment.STAGING, Environment.PRODUCTION


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class FakeAsyncRedis:
    """In-process stand-in for redis.asyncio implementing SET NX EX / GET / DELETE."""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self._data: dict[str, tuple[str, datetime | None]] = {}

    def _alive(self, name: str) -> bool:
        item = self._data.get(name)
        if item and item[1] is not None and item[1] <= self._clock():
            del self._data[name]
        return name in self._data

    async def set(self, name: str, value: str, *, ex: int | None = None, nx: bool = False) -> Any:
        if nx and self._alive(name):
            return None
        exp = self._clock() + timedelta(seconds=ex) if ex else None
        self._data[name] = (value, exp)
        return True

    async def get(self, name: str) -> Any:
        return self._data[name][0].encode() if self._alive(name) else None

    async def delete(self, *names: str) -> int:
        return sum(1 for n in names if self._data.pop(n, None) is not None)

    async def eval(self, script: str, numkeys: int, *args: Any) -> Any:
        """Python emulation of the two Lua scripts (the real Lua is qualified
        against a real Redis in tests/redis/)."""

        from ai_gateway_pool.jobs.affinity import LUA_COMPARE_AND_DELETE, LUA_COMPLETE

        key, argv = args[0], args[numkeys:]
        if not self._alive(key):
            return 0 if script == LUA_COMPARE_AND_DELETE else None
        value, _ = self._data[key]
        doc = json.loads(value)
        if script == LUA_COMPARE_AND_DELETE:
            if argv[0] and doc["node_id"] != argv[0]:
                return -1
            del self._data[key]
            return 1
        assert script == LUA_COMPLETE
        doc["status"], doc["expires_at"] = argv[0], argv[1]
        out = json.dumps(doc)
        self._data[key] = (out, self._clock() + timedelta(seconds=int(argv[2])))
        return out.encode()


@pytest.fixture(params=["memory", "redis", "sqlite"])
def store_and_clock(request: pytest.FixtureRequest, tmp_path: Any) -> tuple[Any, FakeClock]:
    clock = FakeClock()
    if request.param == "memory":
        return InMemoryJobAffinityStore(clock), clock
    if request.param == "sqlite":
        return SqliteJobAffinityStore(tmp_path / "a.sqlite3", clock), clock
    return RedisJobAffinityStore(FakeAsyncRedis(clock), clock), clock


async def test_register_and_lookup(store_and_clock: tuple[Any, FakeClock]) -> None:
    store, _ = store_and_clock
    item = await store.register_job(STG, "job-1", "gw-staging", 3600)
    assert item.status is JobStatus.ACTIVE
    found = await store.lookup_job(STG, "job-1")
    assert found is not None and found.node_id == "gw-staging"


async def test_environment_isolation(store_and_clock: tuple[Any, FakeClock]) -> None:
    store, _ = store_and_clock
    await store.register_job(STG, "job-1", "gw-staging", 3600)
    assert await store.lookup_job(PRD, "job-1") is None
    # Same upstream id in production is a different, independent mapping.
    await store.register_job(PRD, "job-1", "gw-production", 3600)
    assert (await store.lookup_job(STG, "job-1")).node_id == "gw-staging"  # type: ignore[union-attr]
    assert (await store.lookup_job(PRD, "job-1")).node_id == "gw-production"  # type: ignore[union-attr]


async def test_collision_is_refused_not_overwritten(store_and_clock: tuple[Any, FakeClock]) -> None:
    store, _ = store_and_clock
    await store.register_job(STG, "job-1", "node-a", 3600)
    with pytest.raises(PoolError) as exc:
        await store.register_job(STG, "job-1", "node-b", 3600)
    assert exc.value.code is PoolErrorCode.JOB_AFFINITY_CONFLICT
    assert (await store.lookup_job(STG, "job-1")).node_id == "node-a"  # type: ignore[union-attr]
    # Idempotent re-registration on the same node is fine.
    await store.register_job(STG, "job-1", "node-a", 3600)


async def test_missing_mapping_is_explicit(store_and_clock: tuple[Any, FakeClock]) -> None:
    store, _ = store_and_clock
    assert await store.lookup_job(STG, "never-seen") is None
    assert await store.mark_job_completed(STG, "never-seen", 60) is None
    assert await store.forget_job(STG, "never-seen") is False


async def test_lifetime_completion_and_expiry(store_and_clock: tuple[Any, FakeClock]) -> None:
    store, clock = store_and_clock
    await store.register_job(STG, "job-1", "n", 600)
    clock.now += timedelta(seconds=500)
    done = await store.mark_job_completed(STG, "job-1", 600)
    assert done is not None and done.status is JobStatus.COMPLETED
    clock.now += timedelta(seconds=500)  # 1000 s after creation, 500 s after completion
    assert await store.lookup_job(STG, "job-1") is not None
    clock.now += timedelta(seconds=200)
    assert await store.lookup_job(STG, "job-1") is None


async def test_controlled_deletion(store_and_clock: tuple[Any, FakeClock]) -> None:
    store, _ = store_and_clock
    await store.register_job(STG, "job-1", "n", 600)
    assert await store.forget_job(STG, "job-1") is True
    assert await store.lookup_job(STG, "job-1") is None
    await store.register_job(STG, "job-2", "n", 600)
    assert await store.expire_job(STG, "job-2") is True


async def test_restart_survival_only_with_durable_backend() -> None:
    clock = FakeClock()
    redis = FakeAsyncRedis(clock)
    await RedisJobAffinityStore(redis, clock).register_job(STG, "job-1", "n", 600)
    restarted = RedisJobAffinityStore(redis, clock)  # new router process, same Redis
    assert (await restarted.lookup_job(STG, "job-1")).node_id == "n"  # type: ignore[union-attr]
    assert RedisJobAffinityStore.durable is True
    assert InMemoryJobAffinityStore.durable is False


async def test_sqlite_survives_new_store_instance(tmp_path: Any) -> None:
    path = tmp_path / "durable.sqlite3"
    await SqliteJobAffinityStore(path).register_job(STG, "job-1", "n", 600)
    reopened = SqliteJobAffinityStore(path)  # new instance, same file = restart
    assert (await reopened.lookup_job(STG, "job-1")).node_id == "n"  # type: ignore[union-attr]
    assert SqliteJobAffinityStore.durable is True


async def test_sqlite_unavailable_raises_explicit_error(tmp_path: Any) -> None:
    store = SqliteJobAffinityStore(tmp_path / "x.sqlite3")
    store._path = str(tmp_path / "missing-dir" / "deeper" / "x.sqlite3")  # unreachable file
    with pytest.raises(AffinityStoreUnavailable):
        await store.ping()
    with pytest.raises(AffinityStoreUnavailable):
        await store.lookup_job(STG, "job-1")


async def test_concurrent_registration_single_winner(tmp_path: Any) -> None:
    import asyncio

    store = SqliteJobAffinityStore(tmp_path / "c.sqlite3")
    results = await asyncio.gather(
        *(store.register_job(STG, "same", f"node-{i}", 600) for i in range(8)),
        return_exceptions=True,
    )
    winners = {r.node_id for r in results if not isinstance(r, BaseException)}
    assert len(winners) == 1
    assert all(
        isinstance(r, PoolError) and r.code is PoolErrorCode.JOB_AFFINITY_CONFLICT
        for r in results
        if isinstance(r, BaseException)
    )


def test_key_namespace_and_job_id_validation() -> None:
    assert affinity_key(STG, "abc-123") == "aigp:staging:asr-job:abc-123"
    assert affinity_key(STG, "x") != affinity_key(PRD, "x")
    for bad in ("", "a b", "x" * 257, ".", "..", "a/b", "a%2Fb"):
        with pytest.raises(ValueError):
            affinity_key(STG, bad)
