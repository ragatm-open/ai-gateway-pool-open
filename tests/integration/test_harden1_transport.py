"""AIGP-HARDEN-1: breaker trial release through the real relay (AIGP-NEW-01),
privileged node pinning without failover (AIGP-NEW-03), authoritative request id
(AIGP-NEW-04), upstream X-AIGP-* never relayed (AIGP-NEW-05), bounded log
fields (AIGP-NEW-02).

httpx.MockTransport scripted per test (no socket, no backend); the breaker
clock is injected, so nothing depends on real delays.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from ai_gateway_pool.config.schema import PoolConfig
from ai_gateway_pool.config.settings import Settings
from ai_gateway_pool.main import create_app
from ai_gateway_pool.observability.logging import MAX_FIELD_CHARS, JsonFormatter
from tests.conftest import ADMIN_TOKEN, node, raw_config

AUTH = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
EMB = {"model": "legacy-768", "texts": ["x"]}


class Clock:
    def __init__(self) -> None:
        self.t = 5000.0

    def __call__(self) -> float:
        return self.t


class Upstream:
    """Scripted upstream: `mode[host]` decides what each call does."""

    def __init__(self) -> None:
        self.mode: dict[str, str] = {}
        self.calls: list[httpx.Request] = []
        self.response_headers: dict[str, str] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        mode = self.mode.get(request.url.host, "ok")
        if request.url.path == "/health/runtime":
            return httpx.Response(200, json={"status": "ok"})
        if mode == "connect":
            raise httpx.ConnectError("refused", request=request)
        if mode == "pool_timeout":
            raise httpx.PoolTimeout("no connection", request=request)
        if mode == "local_protocol":
            raise httpx.LocalProtocolError("body shorter than Content-Length", request=request)
        if mode == "unexpected":
            raise RuntimeError("unexpected bug")
        if mode == "cancelled":
            raise asyncio.CancelledError()
        return httpx.Response(
            200,
            json={"node": request.url.host, "rid": request.headers.get("x-request-id")},
            headers=self.response_headers,
        )


def _client(nodes: list[dict[str, Any]], **extra: Any) -> tuple[TestClient, Upstream, Clock]:
    cfg = PoolConfig.model_validate(
        raw_config(
            nodes=nodes,
            circuit_breaker={"failure_threshold": 1, "open_seconds": 1, "max_open_seconds": 4},
            **extra,
        )
    )
    settings = Settings(
        environment=cfg.environment,
        config_file=Path("/nonexistent/in-memory.yaml"),
        admin_token=ADMIN_TOKEN,
        log_level="WARNING",
    )
    c = TestClient(create_app(settings, cfg), raise_server_exceptions=False)
    up = Upstream()
    ctx = c.app.state.pool  # type: ignore[attr-defined]
    ctx.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(up.handler))
    clock = Clock()
    ctx.registry.availability._clock = clock
    return c, up, clock


@pytest.fixture
def single() -> Iterator[tuple[TestClient, Upstream, Clock]]:
    yield _client([node("gw-staging", url="http://gw-a.invalid:8200")])


def _open_breaker(c: TestClient, up: Upstream, clock: Clock) -> None:
    up.mode["gw-a.invalid"] = "connect"
    r = c.post("/v1/embeddings", json=EMB)
    assert r.status_code == 503 and r.headers["x-aigp-error"] == "UPSTREAM_UNAVAILABLE"
    r = c.post("/v1/embeddings", json=EMB)
    assert r.status_code == 503 and "CIRCUIT_OPEN" in r.json()["error"]["message"]
    clock.t += 1.01  # open period over: HALF_OPEN, one trial allowed


def _breaker(c: TestClient) -> str:
    nodes = c.get("/internal/nodes", headers=AUTH).json()["nodes"]
    return str(nodes[0]["availability_this_process"]["breaker_state"])


@pytest.mark.parametrize(
    ("mode", "status", "code"),
    [
        ("pool_timeout", 504, "UPSTREAM_TIMEOUT"),
        ("local_protocol", 400, "INVALID_REQUEST_FRAMING"),
        ("unexpected", 500, None),
        ("cancelled", 500, None),
    ],
)
def test_neutral_trial_outcome_never_leaves_the_node_excluded(
    single: tuple[TestClient, Upstream, Clock], mode: str, status: int, code: str | None
) -> None:
    """The F4 reproduction of the assessment, for every outcome that records no
    verdict: the next request is admitted (was: CIRCUIT_OPEN until restart)."""

    c, up, clock = single
    _open_breaker(c, up, clock)
    up.mode["gw-a.invalid"] = mode
    r = c.post("/v1/embeddings", json=EMB)  # this request IS the HALF_OPEN trial
    assert r.status_code == status
    if code:
        assert r.headers["x-aigp-error"] == code
    assert _breaker(c) == "HALF_OPEN"
    up.mode["gw-a.invalid"] = "ok"
    r = c.post("/v1/embeddings", json=EMB)  # no clock advance needed
    assert r.status_code == 200, r.text
    assert _breaker(c) == "CLOSED"


def test_failed_trial_reopens_then_recovers(single: tuple[TestClient, Upstream, Clock]) -> None:
    c, up, clock = single
    _open_breaker(c, up, clock)
    r = c.post("/v1/embeddings", json=EMB)  # trial fails (still "connect")
    assert r.status_code == 503 and _breaker(c) == "OPEN"
    up.mode["gw-a.invalid"] = "ok"
    clock.t += 1.01
    assert _breaker(c) == "OPEN"  # period doubled (2 s)
    clock.t += 1.0
    assert c.post("/v1/embeddings", json=EMB).status_code == 200
    assert _breaker(c) == "CLOSED"


def test_probe_recovers_a_half_open_node_without_client_traffic(
    single: tuple[TestClient, Upstream, Clock],
) -> None:
    c, up, clock = single
    _open_breaker(c, up, clock)
    assert _breaker(c) == "HALF_OPEN"
    r = c.post("/internal/nodes/gw-staging/probe", headers=AUTH)
    assert r.status_code == 200
    assert _breaker(c) == "CLOSED"


def test_post_never_replayed_after_neutral_trial(
    single: tuple[TestClient, Upstream, Clock],
) -> None:
    c, up, clock = single
    _open_breaker(c, up, clock)
    up.mode["gw-a.invalid"] = "local_protocol"
    before = len(up.calls)
    c.post("/v1/embeddings", json=EMB)
    assert len(up.calls) - before == 1  # one attempt, no retry of the POST


# ------------------------------------------------------------------ pinning


def _two_nodes() -> tuple[TestClient, Upstream, Clock]:
    a = node("node-a", url="http://gw-a.invalid:8200", host="host-a")
    b = node("node-b", url="http://gw-b.invalid:8200", host="host-b")
    return _client(
        [a, b],
        routing={"policy": "WEIGHTED_ROUND_ROBIN"},
        failover={"enabled": True, "max_extra_nodes": 1},
    )


def test_unpinned_request_fails_over_before_dispatch() -> None:
    c, up, _ = _two_nodes()
    up.mode["gw-a.invalid"] = "connect"
    hosts = {c.post("/v1/embeddings", json=EMB).json()["node"] for _ in range(4)}
    assert hosts == {"gw-b.invalid"}  # control: failover is active


def test_pinned_request_is_never_failed_over() -> None:
    c, up, _ = _two_nodes()
    up.mode["gw-a.invalid"] = "connect"
    before = [r.url.host for r in up.calls]
    r = c.post("/v1/embeddings", json=EMB, headers={"X-AIGP-Target-Node": "node-a", **AUTH})
    assert r.status_code == 503 and r.headers["x-aigp-error"] == "UPSTREAM_UNAVAILABLE"
    after = [r.url.host for r in up.calls][len(before) :]
    assert after == ["gw-a.invalid"]  # node-b never contacted


def test_pinned_request_reaches_the_pinned_node_without_the_admin_token() -> None:
    c, up, _ = _two_nodes()
    for _ in range(3):
        r = c.post("/v1/embeddings", json=EMB, headers={"X-AIGP-Target-Node": "node-b", **AUTH})
        assert r.status_code == 200 and r.json()["node"] == "gw-b.invalid"
    assert all("authorization" not in {k.lower() for k in r.headers} for r in up.calls)
    assert all(r.headers.get("x-aigp-target-node") is None for r in up.calls)


# ------------------------------------------------------------------ request id


@pytest.mark.parametrize(
    "supplied", ["has spaces", "x" * 129, "semi;colon", "slash/char", "ok-but\ttab"]
)
def test_invalid_request_id_is_never_relayed_raw(
    single: tuple[TestClient, Upstream, Clock], supplied: str
) -> None:
    c, up, _ = single
    r = c.post("/v1/embeddings", json=EMB, headers={"X-Request-ID": supplied})
    assert r.status_code == 200
    rid = r.headers["x-request-id"]
    assert rid != supplied
    assert r.json()["rid"] == rid  # upstream got exactly the pool's id
    assert [v for k, v in up.calls[-1].headers.multi_items() if k == "x-request-id"] == [rid]


def test_valid_request_id_is_kept_end_to_end(single: tuple[TestClient, Upstream, Clock]) -> None:
    c, _, _ = single
    r = c.post("/v1/embeddings", json=EMB, headers={"X-Request-ID": "int3c-Q01-1"})
    assert r.headers["x-request-id"] == "int3c-Q01-1" and r.json()["rid"] == "int3c-Q01-1"


# ------------------------------------------------------------------ upstream X-AIGP-*


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_upstream_cannot_forge_pool_headers_nor_error_category(
    single: tuple[TestClient, Upstream, Clock],
) -> None:
    c, up, _ = single
    up.response_headers = {
        "X-AIGP-Error": "UPSTREAM_OVERRIDE_DENIED",
        "X-AIGP-Affinity": "forged",
        "Retry-After": "7",
        "Content-Type": "application/json",
        "X-Gateway-Build": "0ba5e0ba",  # identifiant de build SYNTHÉTIQUE
    }
    collector = _Collect()
    logger = logging.getLogger("ai_gateway_pool")
    logger.addHandler(collector)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        r = c.post("/v1/embeddings", json=EMB)
    finally:
        logger.removeHandler(collector)
        logger.setLevel(old_level)
    assert r.status_code == 200
    names = {k.lower() for k in r.headers}
    assert not any(n.startswith("x-aigp-") for n in names)
    assert r.headers["retry-after"] == "7" and r.headers["x-gateway-build"] == "0ba5e0ba"
    assert r.headers["content-type"] == "application/json"
    access = [
        rec
        for rec in collector.records
        if getattr(rec, "fields", {}).get("event") == "http_request"
    ]
    assert access and access[-1].fields["error_category"] is None  # type: ignore[attr-defined]


def test_pool_errors_still_carry_x_aigp_error(single: tuple[TestClient, Upstream, Clock]) -> None:
    c, _, _ = single
    r = c.post("/v1/embeddings", json={"model": "gpt-imaginary"})
    assert r.headers["x-aigp-error"] == "MODEL_NOT_AVAILABLE"


# ------------------------------------------------------------------ bounded logs


def test_long_model_is_a_typed_400_not_a_500(single: tuple[TestClient, Upstream, Clock]) -> None:
    c, up, _ = single
    before = len(up.calls)
    for size in (201, 1000, 200_000):
        r = c.post(
            "/v1/embeddings",
            content=json.dumps({"model": "m" * size}),
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400 and r.headers["x-aigp-error"] == "INVALID_REQUEST_BODY"
        assert len(r.content) < 1024  # never echoed
    assert len(up.calls) == before


def test_every_logged_string_field_is_bounded() -> None:
    rec = logging.LogRecord("ai_gateway_pool", logging.INFO, __file__, 1, "request", None, None)
    rec.fields = {"model_id": "m" * 200_000, "route": "r" * 300, "request_id": "ok"}  # type: ignore[attr-defined]
    line = JsonFormatter().format(rec)
    assert len(line) < 4 * MAX_FIELD_CHARS + 400
    out = json.loads(line)
    assert out["request_id"] == "ok"
    assert out["model_id"].startswith("m" * MAX_FIELD_CHARS) and "more]" in out["model_id"]
