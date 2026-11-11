"""Transport contracts: timeouts, retry/idempotency rules, header forwarding.

This module defines *what* the future HTTP relay (POOL-2) must respect. It is
pure and side-effect free so that the rules can be tested before any request is
actually forwarded.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum

import httpx
from pydantic import BaseModel, ConfigDict, Field


class TrafficClass(StrEnum):
    """Each class gets its own timeout profile. Health probes never share the
    budget of inference requests, and long generations / audio uploads are never
    cut by a short global timeout."""

    HEALTH_PROBE = "health_probe"
    INFERENCE = "inference"  # synchronous JSON inference (generate, embeddings...)
    STREAMING = "streaming"  # NDJSON response streams
    UPLOAD = "upload"  # large multipart uploads (ASR audio)
    JOB_POLL = "job_poll"  # GET / DELETE on async jobs


class TimeoutProfile(BaseModel):
    """Seconds. ``None`` means "no limit" for that phase (httpx semantics).

    ``total_budget`` bounds the time until the upstream response HEADERS arrive
    (body streaming is then governed by ``read``, an inter-chunk timeout). It is
    ``None`` for inference/streaming/upload so that long jobs are not killed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    connect: float = Field(gt=0, le=60)
    read: float | None = Field(default=None, gt=0)
    write: float | None = Field(default=None, gt=0)
    pool: float = Field(gt=0, le=120)
    total_budget: float | None = Field(default=None, gt=0)

    def to_httpx(self) -> httpx.Timeout:
        return httpx.Timeout(connect=self.connect, read=self.read, write=self.write, pool=self.pool)


DEFAULT_TIMEOUTS: dict[TrafficClass, TimeoutProfile] = {
    TrafficClass.HEALTH_PROBE: TimeoutProfile(connect=2, read=3, write=3, pool=2, total_budget=5),
    # Generations can be long: RAGatm Open allows up to 300 s on generate and the
    # Gateway Open relays Ollama with a 300 s budget. Keep headroom above every client.
    TrafficClass.INFERENCE: TimeoutProfile(connect=5, read=600, write=60, pool=10),
    # Inter-chunk read timeout; no total budget for a stream.
    TrafficClass.STREAMING: TimeoutProfile(connect=5, read=300, write=60, pool=10),
    # Audio upload: generous write, no total budget.
    TrafficClass.UPLOAD: TimeoutProfile(connect=5, read=120, write=900, pool=10),
    # Long-poll: the Gateway accepts wait_ms up to 30 000 ms, keep read above it.
    TrafficClass.JOB_POLL: TimeoutProfile(connect=3, read=45, write=10, pool=5, total_budget=60),
}


# --------------------------------------------------------------------------- retries

IDEMPOTENT_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})


class RetryDecision(StrEnum):
    RETRY_SAME_NODE = "RETRY_SAME_NODE"
    NO_RETRY = "NO_RETRY"


def retry_decision(method: str, *, connection_established: bool, attempt: int) -> RetryDecision:
    """No blind retry of inference POSTs.

    A POST may have started GPU work on the Gateway even if the client saw a
    timeout, so a POST is never retried once the connection was established.
    Only idempotent reads are retried, at most once, and only on the same node
    (job affinity!), and only when no response byte was received.
    ``DELETE`` is never retried: a lost response may hide a completed deletion.
    """

    if attempt >= 1:
        return RetryDecision.NO_RETRY
    if method.upper() in IDEMPOTENT_METHODS and not connection_established:
        # Only transport failures BEFORE any response byte (connect error, stale
        # keep-alive connection closed by the peer) are retried, once, same node.
        return RetryDecision.RETRY_SAME_NODE
    return RetryDecision.NO_RETRY


# --------------------------------------------------------------------------- headers

# RFC 9110 §7.6.1 hop-by-hop headers. Never relayed in either direction.
HOP_BY_HOP_HEADERS: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

# Request side: `host` is recomputed for the upstream. `content-length` IS kept:
# the relay streams the body with the client's declared length, so the upstream
# receives Content-Length and never Transfer-Encoding as well (AI Gateway refuses
# a request carrying both).
REQUEST_STRIPPED_HEADERS: frozenset[str] = HOP_BY_HOP_HEADERS | {"host"}

# Response side: framing comes from the upstream (raw bytes relayed unchanged, so
# Content-Length / Content-Encoding stay consistent). `server` is not relayed.
RESPONSE_STRIPPED_HEADERS: frozenset[str] = HOP_BY_HOP_HEADERS | {"server"}

# Headers a client could use to try to steer the upstream destination. The pool
# refuses them outright (the destination only ever comes from the registry).
UPSTREAM_OVERRIDE_HEADERS: frozenset[str] = frozenset(
    {
        "x-upstream-url",
        "x-upstream-host",
        "x-forwarded-host",
        "x-original-url",
        "x-rewrite-url",
        "x-target-url",
        "x-gateway-url",
        "forwarded",
    }
)

POOL_HEADER_PREFIX = "x-aigp-"

Headers = Iterable[tuple[str, str]]


def contains_upstream_override(headers: Iterable[str]) -> bool:
    return any(h.lower() in UPSTREAM_OVERRIDE_HEADERS for h in headers)


def _connection_tokens(headers: list[tuple[str, str]]) -> set[str]:
    """Header names listed in `Connection:` are hop-by-hop too (RFC 9110 §7.6.1)."""

    tokens: set[str] = set()
    for name, value in headers:
        if name.lower() == "connection":
            tokens |= {t.strip().lower() for t in value.split(",") if t.strip()}
    return tokens


def forwardable_request_headers(
    headers: Headers | Mapping[str, str], *, admin_token: str | None = None
) -> list[tuple[str, str]]:
    """Headers to forward upstream (repeated headers preserved, order kept).

    Removed: hop-by-hop, ``Host``, destination-steering headers, pool-internal
    ``X-AIGP-*`` headers, and an ``Authorization`` carrying the pool's OWN admin
    token (pool credentials never reach a Gateway). Any other ``Authorization``
    is relayed unchanged (it belongs to the client/Gateway contract) and never logged.
    """

    items = list(headers.items()) if isinstance(headers, Mapping) else list(headers)
    extra = _connection_tokens(items)
    out: list[tuple[str, str]] = []
    for name, value in items:
        lname = name.lower()
        if lname in REQUEST_STRIPPED_HEADERS or lname in extra:
            continue
        if lname in UPSTREAM_OVERRIDE_HEADERS or lname.startswith(POOL_HEADER_PREFIX):
            continue
        if lname == "authorization" and admin_token and _is_token(value, admin_token):
            continue
        out.append((name, value))
    return out


def _is_token(header_value: str, token: str) -> bool:
    import hmac

    supplied = header_value[7:] if header_value.lower().startswith("bearer ") else header_value
    return hmac.compare_digest(supplied.strip().encode(), token.encode())


def forwardable_response_headers(headers: Headers | Mapping[str, str]) -> list[tuple[str, str]]:
    """Upstream response headers relayed to the client.

    ``X-AIGP-*`` is the pool's own namespace (e.g. ``X-AIGP-Error`` marks a
    POOL-originated error): an upstream can never forge it (AIGP-HARDEN-1).
    """

    items = list(headers.items()) if isinstance(headers, Mapping) else list(headers)
    extra = _connection_tokens(items)
    return [
        (k, v)
        for k, v in items
        if k.lower() not in RESPONSE_STRIPPED_HEADERS | extra
        and not k.lower().startswith(POOL_HEADER_PREFIX)
    ]
