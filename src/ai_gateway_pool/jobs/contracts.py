"""Contract of the job -> node affinity store (ASR async jobs).

Problem: ``POST /v1/audio/transcriptions`` creates a job on node A; the later
``GET``/``DELETE /v1/audio/transcriptions/{job_id}`` MUST reach node A.

Guarantees every implementation must provide (tested in
``tests/unit/test_job_affinity.py`` against all implementations):

* **namespace isolation** — keys are scoped by environment; a staging lookup can
  never read a production mapping, even for an identical ``job_id``;
* **no silent collision** — ``register_job`` refuses to overwrite an existing
  live mapping of the same ``(environment, job_id)`` bound to another node
  (``JOB_AFFINITY_CONFLICT``); job ids are NOT assumed globally unique across
  nodes;
* **missing mapping is explicit** — ``lookup_job`` returns ``None``; the relay
  must answer ``JOB_AFFINITY_UNKNOWN`` and must NOT broadcast the request to
  every node nor guess;
* **lifetime** — a mapping lives at least ``ttl_seconds`` (refreshed on
  completion) so it outlives the job's polling window;
* **controlled deletion** — ``forget_job`` removes it after a successful
  upstream DELETE, ``expire_job`` is the TTL path;
* **restart survival** — required for production multi-node; only the Redis
  adapter provides it. The in-memory store is for tests / single-node dev and
  declares ``durable = False``.

The Gateway's ``job_id`` format is never changed nor parsed by the pool.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from ai_gateway_pool.errors import PoolError, PoolErrorCode
from ai_gateway_pool.nodes.models import Environment


class JobStatus(StrEnum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"


class JobAffinity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: Environment
    # Public id the client uses. Strategy B (POOL-3): the native Gateway id, kept
    # verbatim — no translation, so the client contract is unchanged.
    job_id: str
    node_id: str
    status: JobStatus
    created_at: datetime
    expires_at: datetime
    # Id of the job ON the node (== job_id under strategy B; explicit so that a
    # future translated public id cannot be confused with the native one).
    gateway_job_id: str = ""

    @model_validator(mode="after")
    def _default_gateway_id(self) -> JobAffinity:
        if not self.gateway_job_id:
            object.__setattr__(self, "gateway_job_id", self.job_id)
        return self


class JobAffinityStore(Protocol):
    durable: bool

    async def ping(self) -> None:
        """Raise ``AffinityStoreUnavailable`` if the store cannot be used right now."""
        ...

    async def register_job(
        self, environment: Environment, job_id: str, node_id: str, ttl_seconds: int
    ) -> JobAffinity: ...

    async def lookup_job(self, environment: Environment, job_id: str) -> JobAffinity | None: ...

    async def mark_job_completed(
        self, environment: Environment, job_id: str, ttl_seconds: int
    ) -> JobAffinity | None: ...

    async def forget_job(
        self, environment: Environment, job_id: str, node_id: str | None = None
    ) -> bool:
        """Delete the mapping. With ``node_id``, only if it still points to that
        node (compare-and-delete: never erase a mapping re-bound elsewhere)."""
        ...

    async def expire_job(self, environment: Environment, job_id: str) -> bool: ...


_JOB_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")


def valid_job_id(job_id: str) -> bool:
    """Opaque Gateway job id: safe token, never ``.``/``..`` (no path tricks)."""

    return bool(_JOB_ID_RE.match(job_id)) and job_id not in {".", ".."}


def affinity_key(environment: Environment, job_id: str, prefix: str = "aigp") -> str:
    """Namespaced key ``<prefix>:<environment>:asr-job:<job_id>``.

    The job id is kept verbatim (validated, not parsed). The environment is part
    of the key AND of the stored value (checked on read): a staging lookup can
    never resolve a production mapping.
    """

    if not valid_job_id(job_id):
        raise ValueError("job_id must be a safe token of at most 256 characters")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", prefix):
        raise ValueError("invalid key prefix")
    return f"{prefix}:{environment.value}:asr-job:{job_id}"


class AffinityStoreUnavailable(PoolError):
    def __init__(self, message: str) -> None:
        super().__init__(PoolErrorCode.AFFINITY_STORE_UNAVAILABLE, message)
