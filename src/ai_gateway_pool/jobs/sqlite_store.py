"""Durable SQLite JobAffinityStore — development and qualification ONLY.

Survives a router restart (same file, new process / new app instance). It is
refused in an approved staging/production configuration (schema rule): a
deployment uses Redis so that several router replicas share the mappings.

Concurrency: one connection per call inside a worker thread, ``BEGIN IMMEDIATE``
for read-modify-write, WAL journal. Plenty for one local router process.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar

from ai_gateway_pool.errors import PoolError, PoolErrorCode
from ai_gateway_pool.jobs.contracts import (
    AffinityStoreUnavailable,
    JobAffinity,
    JobStatus,
    affinity_key,
)
from ai_gateway_pool.nodes.models import Environment

T = TypeVar("T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_affinity (
    key         TEXT PRIMARY KEY,
    environment TEXT NOT NULL,
    job_id      TEXT NOT NULL,
    node_id     TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
)
"""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SqliteJobAffinityStore:
    durable = True

    def __init__(self, path: str | Path, clock: Callable[[], datetime] = _utcnow) -> None:
        self._path = str(path)
        self._clock = clock
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._run_sync(self._init)

    # ------------------------------------------------------------------ plumbing

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(_SCHEMA)

    def _run_sync(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        try:
            conn = self._connect()
        except sqlite3.Error as exc:
            raise AffinityStoreUnavailable(f"sqlite unavailable: {type(exc).__name__}") from exc
        try:
            return fn(conn)
        except sqlite3.Error as exc:
            raise AffinityStoreUnavailable(f"sqlite error: {type(exc).__name__}") from exc
        finally:
            conn.close()

    async def _run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        return await asyncio.to_thread(self._run_sync, fn)

    @staticmethod
    def _row(row: sqlite3.Row | None) -> JobAffinity | None:
        if row is None:
            return None
        return JobAffinity(
            environment=Environment(row["environment"]),
            job_id=row["job_id"],
            node_id=row["node_id"],
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
        )

    def _live(self, conn: sqlite3.Connection, key: str) -> JobAffinity | None:
        row = conn.execute("SELECT * FROM job_affinity WHERE key = ?", (key,)).fetchone()
        item = self._row(row)
        if item is not None and item.expires_at <= self._clock():
            conn.execute("DELETE FROM job_affinity WHERE key = ?", (key,))
            return None
        return item

    # ------------------------------------------------------------------ contract

    async def ping(self) -> None:
        await self._run(lambda c: c.execute("SELECT 1").fetchone())

    async def register_job(
        self, environment: Environment, job_id: str, node_id: str, ttl_seconds: int
    ) -> JobAffinity:
        key = affinity_key(environment, job_id)

        def tx(conn: sqlite3.Connection) -> JobAffinity:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self._live(conn, key)
                if existing is not None:
                    conn.execute("COMMIT")
                    if existing.node_id != node_id:
                        raise PoolError(
                            PoolErrorCode.JOB_AFFINITY_CONFLICT,
                            f"job {job_id!r} already bound to node {existing.node_id}, "
                            f"refusing to rebind to {node_id}",
                        )
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
                conn.execute(
                    "INSERT INTO job_affinity VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        key,
                        environment.value,
                        job_id,
                        node_id,
                        item.status.value,
                        item.created_at.isoformat(),
                        item.expires_at.isoformat(),
                    ),
                )
                conn.execute("COMMIT")
                return item
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

        return await self._run(tx)

    async def lookup_job(self, environment: Environment, job_id: str) -> JobAffinity | None:
        key = affinity_key(environment, job_id)
        item = await self._run(lambda c: self._live(c, key))
        if item is not None and item.environment != environment:  # defence in depth
            return None
        return item

    async def mark_job_completed(
        self, environment: Environment, job_id: str, ttl_seconds: int
    ) -> JobAffinity | None:
        key = affinity_key(environment, job_id)

        def tx(conn: sqlite3.Connection) -> JobAffinity | None:
            item = self._live(conn, key)
            if item is None:
                return None
            expires = self._clock() + timedelta(seconds=ttl_seconds)
            conn.execute(
                "UPDATE job_affinity SET status = ?, expires_at = ? WHERE key = ?",
                (JobStatus.COMPLETED.value, expires.isoformat(), key),
            )
            return item.model_copy(update={"status": JobStatus.COMPLETED, "expires_at": expires})

        return await self._run(tx)

    async def forget_job(
        self, environment: Environment, job_id: str, node_id: str | None = None
    ) -> bool:
        key = affinity_key(environment, job_id)
        if node_id is None:
            return await self._run(
                lambda c: c.execute("DELETE FROM job_affinity WHERE key = ?", (key,)).rowcount > 0
            )
        return await self._run(
            lambda c: (
                c.execute(
                    "DELETE FROM job_affinity WHERE key = ? AND node_id = ?", (key, node_id)
                ).rowcount
                > 0
            )
        )

    async def expire_job(self, environment: Environment, job_id: str) -> bool:
        return await self.forget_job(environment, job_id)
