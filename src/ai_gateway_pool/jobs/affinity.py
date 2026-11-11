"""JobAffinityStore implementations.

* :class:`InMemoryJobAffinityStore` — NOT durable; tests and single-node dev.
* :class:`RedisJobAffinityStore` — adapter over any asyncio Redis client exposing
  ``set(name, value, ex=, nx=)``, ``get``, ``delete`` (e.g. ``redis.asyncio``).
  The ``redis`` package is intentionally NOT a dependency of POOL-1: no job is
  routed yet and no Redis instance is provisioned. The adapter is tested against
  an in-process fake implementing the same three calls.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from ai_gateway_pool.errors import PoolError, PoolErrorCode
from ai_gateway_pool.jobs.contracts import (
    AffinityStoreUnavailable,
    JobAffinity,
    JobStatus,
    affinity_key,
)
from ai_gateway_pool.nodes.models import Environment

Clock = Callable[[], datetime]

try:  # redis-py is a runtime dependency of the Redis backend only
    from redis.exceptions import AuthenticationError as _RedisAuthError
    from redis.exceptions import ConnectionError as _RedisConnectionError
    from redis.exceptions import MaxConnectionsError as _RedisMaxConnections
except ImportError:  # pragma: no cover - redis always installed in the image
    _RedisConnectionError = _RedisAuthError = _RedisMaxConnections = ()  # type: ignore[assignment,misc]


def _is_stale_connection(exc: BaseException) -> bool:
    """A connection error that a NEW connection can cure (POOL-PROD-2).

    Typical case: ``Connection closed by server.`` on a pooled connection after a
    Redis restart, or the reconnect itself being refused. Deliberately excluded:
    timeouts (ambiguous: the command may still run), authentication/authorization
    failures and pool exhaustion (a new connection changes nothing).
    """

    return isinstance(exc, _RedisConnectionError) and not isinstance(
        exc, (_RedisAuthError, _RedisMaxConnections)
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _conflict(job_id: str, existing: str, new: str) -> PoolError:
    return PoolError(
        PoolErrorCode.JOB_AFFINITY_CONFLICT,
        f"job {job_id!r} already bound to node {existing}, refusing to rebind to {new}",
    )


class InMemoryJobAffinityStore:
    durable = False

    async def ping(self) -> None:
        return None

    def __init__(self, clock: Clock = _utcnow) -> None:
        self._clock = clock
        self._data: dict[str, JobAffinity] = {}

    def _live(self, key: str) -> JobAffinity | None:
        item = self._data.get(key)
        if item is not None and item.expires_at <= self._clock():
            del self._data[key]
            return None
        return item

    async def register_job(
        self, environment: Environment, job_id: str, node_id: str, ttl_seconds: int
    ) -> JobAffinity:
        key = affinity_key(environment, job_id)
        existing = self._live(key)
        if existing is not None:
            if existing.node_id != node_id:
                raise _conflict(job_id, existing.node_id, node_id)
            return existing
        now = self._clock()
        item = JobAffinity(
            environment=environment,
            job_id=job_id,
            node_id=node_id,
            status=JobStatus.ACTIVE,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        self._data[key] = item
        return item

    async def lookup_job(self, environment: Environment, job_id: str) -> JobAffinity | None:
        return self._live(affinity_key(environment, job_id))

    async def mark_job_completed(
        self, environment: Environment, job_id: str, ttl_seconds: int
    ) -> JobAffinity | None:
        key = affinity_key(environment, job_id)
        item = self._live(key)
        if item is None:
            return None
        updated = item.model_copy(
            update={
                "status": JobStatus.COMPLETED,
                "expires_at": self._clock() + timedelta(seconds=ttl_seconds),
            }
        )
        self._data[key] = updated
        return updated

    async def forget_job(
        self, environment: Environment, job_id: str, node_id: str | None = None
    ) -> bool:
        key = affinity_key(environment, job_id)
        item = self._data.get(key)
        if item is None or (node_id is not None and item.node_id != node_id):
            return False
        del self._data[key]
        return True

    async def expire_job(self, environment: Environment, job_id: str) -> bool:
        return await self.forget_job(environment, job_id)


class AsyncRedisLike(Protocol):
    async def set(
        self, name: str, value: str, *, ex: int | None = None, nx: bool = False
    ) -> Any: ...

    async def get(self, name: str) -> Any: ...

    async def delete(self, *names: str) -> int: ...

    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any: ...


# Compare-and-delete: delete only if the stored node_id matches ARGV[1] ('' = any).
# Returns 1 deleted, 0 absent, -1 bound to another node (kept).
LUA_COMPARE_AND_DELETE = """
local v = redis.call('GET', KEYS[1])
if not v then return 0 end
if ARGV[1] ~= '' and cjson.decode(v)['node_id'] ~= ARGV[1] then return -1 end
return redis.call('DEL', KEYS[1])
"""

# Atomic completion: rewrite status/expires_at and TTL of an existing mapping.
# ARGV: status, expires_at ISO, ttl seconds. Returns the new value or false.
LUA_COMPLETE = """
local v = redis.call('GET', KEYS[1])
if not v then return false end
local d = cjson.decode(v)
d['status'] = ARGV[1]
d['expires_at'] = ARGV[2]
local out = cjson.encode(d)
redis.call('SET', KEYS[1], out, 'EX', tonumber(ARGV[3]))
return out
"""


class RedisJobAffinityStore:
    """Durable, shared job -> node store on a REAL Redis (qualified in POOL-3).

    * ``register_job``: ``SET key value NX EX ttl`` — atomic across every router
      process: two processes can never bind one (environment, job_id) to two nodes.
    * ``forget_job`` / ``mark_job_completed``: server-side Lua scripts (atomic).
    * Values are technical metadata only (environment, ids, node, status, times):
      never request bodies, audio, transcripts or credentials.
    * Every Redis/network error becomes ``AffinityStoreUnavailable`` (fail closed);
      nothing falls back to memory or SQLite.
    """

    durable = True

    def __init__(
        self, client: AsyncRedisLike, clock: Clock = _utcnow, key_prefix: str = "aigp"
    ) -> None:
        self._redis = client
        self._clock = clock
        self._prefix = key_prefix
        self.reconnects = 0  # stale-connection recoveries (one extra command each)
        affinity_key(Environment.STAGING, "probe", key_prefix)  # validate prefix early

    def _key(self, environment: Environment, job_id: str) -> str:
        return affinity_key(environment, job_id, self._prefix)

    async def _call(self, op: Callable[[], Awaitable[Any]], *, reconnect: bool) -> Any:
        """Run one Redis command; errors become ``AffinityStoreUnavailable`` (fail closed).

        ``reconnect=True`` is passed ONLY for commands whose second execution is
        harmless (reads; atomic, idempotent Lua scripts). Then, on a stale
        connection error, idle pooled connections are invalidated and the command
        is sent ONCE more. Never on a timeout, never in a loop, never for SET NX
        (see ``register_job``: read and reconcile instead).
        """

        try:
            return await op()
        except PoolError:
            raise
        except Exception as exc:  # redis.exceptions.*, OSError, TimeoutError
            if not (reconnect and _is_stale_connection(exc)):
                raise AffinityStoreUnavailable(f"redis unavailable: {type(exc).__name__}") from exc
        await self._invalidate_idle_connections()
        try:
            return await op()
        except PoolError:
            raise
        except Exception as exc:
            raise AffinityStoreUnavailable(
                f"redis unavailable after reconnect: {type(exc).__name__}"
            ) from exc

    async def _invalidate_idle_connections(self) -> None:
        """After a server-side close every IDLE pooled connection is dead too:
        drop them so the single new attempt opens a fresh one. In-use connections
        belong to concurrent requests, which handle their own error."""

        self.reconnects += 1
        pool = getattr(self._redis, "connection_pool", None)
        if pool is not None:
            with contextlib.suppress(Exception):
                await pool.disconnect(inuse_connections=False)

    async def ping(self) -> None:
        await self._call(lambda: self._redis.get(f"{self._prefix}:ping"), reconnect=True)

    @staticmethod
    def _encode(item: JobAffinity) -> str:
        return item.model_dump_json()

    @staticmethod
    def _decode(raw: Any) -> JobAffinity | None:
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return JobAffinity.model_validate(json.loads(raw))

    async def register_job(
        self, environment: Environment, job_id: str, node_id: str, ttl_seconds: int
    ) -> JobAffinity:
        key = self._key(environment, job_id)
        now = self._clock()
        item = JobAffinity(
            environment=environment,
            job_id=job_id,
            node_id=node_id,
            status=JobStatus.ACTIVE,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        value = self._encode(item)
        set_attempts = 0
        for _ in range(2):  # second pass only if the key expired between SET and GET
            set_attempts += 1
            try:
                stored = await self._redis.set(key, value, ex=ttl_seconds, nx=True)
            except PoolError:
                raise
            except Exception as exc:
                # The reply was lost: the SET may or may not have been applied.
                # Never re-send blindly. Only a stale connection (not a timeout) is
                # reconciled: invalidate, READ the key, decide.
                if set_attempts > 1 or not _is_stale_connection(exc):
                    raise AffinityStoreUnavailable(
                        f"redis unavailable: {type(exc).__name__}"
                    ) from exc
                await self._invalidate_idle_connections()
                existing = self._decode(
                    await self._call(lambda: self._redis.get(key), reconnect=False)
                )
                if existing is None:
                    continue  # not applied: one more SET NX (still atomic, still NX)
                if existing.node_id == node_id and existing.environment == environment:
                    return existing  # applied (ours) or an identical binding
                raise _conflict(job_id, existing.node_id, node_id) from exc
            if stored:
                return item
            existing = self._decode(await self._call(lambda: self._redis.get(key), reconnect=True))
            if existing is not None:
                if existing.node_id == node_id and existing.environment == environment:
                    return existing
                raise _conflict(job_id, existing.node_id, node_id)
        raise AffinityStoreUnavailable("redis: could not register mapping")

    async def lookup_job(self, environment: Environment, job_id: str) -> JobAffinity | None:
        key = self._key(environment, job_id)
        item = self._decode(await self._call(lambda: self._redis.get(key), reconnect=True))
        if item is not None and item.environment != environment:  # defence in depth
            return None
        return item

    async def mark_job_completed(
        self, environment: Environment, job_id: str, ttl_seconds: int
    ) -> JobAffinity | None:
        expires = self._clock() + timedelta(seconds=ttl_seconds)
        key = self._key(environment, job_id)
        # Atomic script, fixed arguments: a second run rewrites the same values.
        out = await self._call(
            lambda: self._redis.eval(
                LUA_COMPLETE,
                1,
                key,
                JobStatus.COMPLETED.value,
                expires.isoformat(),
                str(ttl_seconds),
            ),
            reconnect=True,
        )
        return self._decode(out) if out else None

    async def forget_job(
        self, environment: Environment, job_id: str, node_id: str | None = None
    ) -> bool:
        key = self._key(environment, job_id)
        # Atomic compare-and-delete: running it twice leaves the same state and
        # keeps the identity condition (only a mapping bound to node_id is removed).
        # After a lost reply the second run may answer 0 although the first one
        # deleted: the return value is informational (the proxy ignores it).
        res = await self._call(
            lambda: self._redis.eval(LUA_COMPARE_AND_DELETE, 1, key, node_id or ""),
            reconnect=True,
        )
        return int(res) == 1

    async def expire_job(self, environment: Environment, job_id: str) -> bool:
        return await self.forget_job(environment, job_id)
