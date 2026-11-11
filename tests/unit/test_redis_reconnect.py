"""POOL-PROD-2: bounded Redis reconnection — operation-by-operation safety (no Redis).

A scripted fake client raises chosen redis-py exceptions on chosen calls, so
every ambiguous case is exercised deterministically, including "the write was
applied but its reply was lost".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from redis.exceptions import AuthenticationError, MaxConnectionsError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from ai_gateway_pool.errors import PoolError, PoolErrorCode
from ai_gateway_pool.jobs.affinity import RedisJobAffinityStore
from ai_gateway_pool.jobs.contracts import AffinityStoreUnavailable
from ai_gateway_pool.nodes.models import Environment

PRD = Environment.PRODUCTION
STALE = RedisConnectionError("Connection closed by server.")


class Pool:
    def __init__(self) -> None:
        self.disconnects: list[bool] = []

    async def disconnect(self, inuse_connections: bool = True) -> None:
        self.disconnects.append(inuse_connections)


class Scripted:
    """Minimal redis.asyncio-like client. ``plan[op]`` = list of actions per call:
    None (behave), an exception (raise BEFORE applying), or ("after", exc)
    (APPLY, then raise: reply lost)."""

    def __init__(self, plan: dict[str, list[Any]] | None = None) -> None:
        self.data: dict[str, str] = {}
        self.plan = plan or {}
        self.calls: dict[str, int] = {}
        self.connection_pool = Pool()

    def _step(self, op: str) -> Any:
        self.calls[op] = self.calls.get(op, 0) + 1
        actions = self.plan.get(op, [])
        return actions.pop(0) if actions else None

    async def _run(self, op: str, apply: Callable[[], Any]) -> Any:
        action = self._step(op)
        if isinstance(action, BaseException):
            raise action
        result = apply()
        if isinstance(action, tuple):
            raise action[1]
        return result

    async def get(self, name: str) -> Any:
        return await self._run("get", lambda: self.data.get(name))

    async def set(self, name: str, value: str, *, ex: int | None = None, nx: bool = False) -> Any:
        def apply() -> Any:
            if nx and name in self.data:
                return None
            self.data[name] = value
            return True

        return await self._run("set", apply)

    async def delete(self, *names: str) -> int:
        return await self._run(
            "delete", lambda: sum(self.data.pop(n, None) is not None for n in names)
        )

    async def eval(self, script: str, numkeys: int, *args: Any) -> Any:
        key, node = args[0], args[1]

        def apply() -> Any:
            import json

            v = self.data.get(key)
            if v is None:
                return 0
            if node and json.loads(v)["node_id"] != node:
                return -1
            del self.data[key]
            return 1

        return await self._run("eval", apply)


def store(plan: dict[str, list[Any]] | None = None) -> tuple[RedisJobAffinityStore, Scripted]:
    fake = Scripted(plan)
    return RedisJobAffinityStore(fake, key_prefix="aigp"), fake  # type: ignore[arg-type]


# --------------------------------------------------------------------------- reads


async def test_normal_operation_no_reconnect() -> None:
    s, fake = store()
    await s.register_job(PRD, "j", "n", 60)
    assert (await s.lookup_job(PRD, "j")).node_id == "n"  # type: ignore[union-attr]
    assert s.reconnects == 0 and fake.connection_pool.disconnects == []


async def test_stale_read_reconnects_once_and_invalidates_idle_connections() -> None:
    s, fake = store({"get": [STALE]})
    await s.ping()
    assert fake.calls["get"] == 2 and s.reconnects == 1
    assert fake.connection_pool.disconnects == [False]  # idle ones only, never in-use


async def test_read_still_failing_after_reconnect_fails_closed_no_loop() -> None:
    s, fake = store({"get": [STALE, STALE, STALE]})
    with pytest.raises(AffinityStoreUnavailable, match="after reconnect"):
        await s.lookup_job(PRD, "j")
    assert fake.calls["get"] == 2  # exactly one reconnection attempt


@pytest.mark.parametrize(
    "exc",
    [
        RedisTimeoutError("Timeout reading from socket"),
        AuthenticationError("invalid username-password pair"),
        MaxConnectionsError("Too many connections"),
        OSError("raw"),
    ],
)
async def test_non_reconnectable_errors_are_not_retried(exc: BaseException) -> None:
    s, fake = store({"get": [exc]})
    with pytest.raises(AffinityStoreUnavailable):
        await s.lookup_job(PRD, "j")
    assert fake.calls["get"] == 1 and s.reconnects == 0


# --------------------------------------------------------------------------- SET NX


async def test_set_nx_lost_reply_after_apply_is_reconciled_without_second_set() -> None:
    s, fake = store({"set": [("after", STALE)]})
    item = await s.register_job(PRD, "j", "node-a", 60)
    assert item.node_id == "node-a"
    assert fake.calls["set"] == 1  # never re-sent: the read showed our mapping
    assert len(fake.data) == 1


async def test_set_nx_failed_before_apply_is_sent_once_more_after_read() -> None:
    s, fake = store({"set": [STALE]})
    await s.register_job(PRD, "j", "node-a", 60)
    assert fake.calls["set"] == 2 and fake.calls["get"] == 1
    assert len(fake.data) == 1


async def test_set_nx_ambiguous_then_other_node_is_a_conflict() -> None:
    s, fake = store()
    await s.register_job(PRD, "j", "node-b", 60)  # existing binding to ANOTHER node
    fake.plan["set"] = [STALE]
    fake.calls.clear()
    with pytest.raises(PoolError) as e:
        await s.register_job(PRD, "j", "node-a", 60)
    assert e.value.code is PoolErrorCode.JOB_AFFINITY_CONFLICT
    assert fake.calls["set"] == 1  # reconciliation decided, no blind SET


async def test_set_nx_ambiguous_and_reconciliation_read_fails_closed() -> None:
    s, fake = store({"set": [STALE], "get": [STALE]})
    with pytest.raises(AffinityStoreUnavailable):
        await s.register_job(PRD, "j", "node-a", 60)
    assert fake.calls["set"] == 1 and fake.calls["get"] == 1  # one reconnection only


async def test_second_set_failure_is_not_retried_again() -> None:
    s, fake = store({"set": [STALE, STALE]})
    with pytest.raises(AffinityStoreUnavailable):
        await s.register_job(PRD, "j", "node-a", 60)
    assert fake.calls["set"] == 2


async def test_set_nx_timeout_is_never_retried_nor_reconciled() -> None:
    s, fake = store({"set": [("after", RedisTimeoutError("Timeout reading from socket"))]})
    with pytest.raises(AffinityStoreUnavailable, match="TimeoutError"):
        await s.register_job(PRD, "j", "node-a", 60)
    assert fake.calls["set"] == 1 and fake.calls.get("get", 0) == 0
    # the write may exist: fail closed; the proxy compensates the job on its node


# --------------------------------------------------------------------------- scripts


async def test_compare_and_delete_lost_reply_is_idempotent() -> None:
    s, fake = store({"eval": [("after", STALE)]})
    await s.register_job(PRD, "j", "node-a", 60)
    await s.forget_job(PRD, "j", node_id="node-a")  # result unused by the proxy
    assert fake.data == {} and fake.calls["eval"] == 2


async def test_compare_and_delete_keeps_identity_condition_on_retry() -> None:
    s, fake = store({"eval": [STALE]})
    await s.register_job(PRD, "j", "node-a", 60)
    assert await s.forget_job(PRD, "j", node_id="node-b") is False
    assert len(fake.data) == 1


async def test_job_ids_kept_verbatim() -> None:
    s, fake = store({"set": [("after", STALE)]})
    jid = "9f1c2a3b-4d5e-6f70-8192-a3b4c5d6e7f8"
    item = await s.register_job(PRD, jid, "node-a", 60)
    assert item.job_id == jid and list(fake.data) == [f"aigp:production:asr-job:{jid}"]
