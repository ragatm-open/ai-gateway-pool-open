"""Relay qualification of EVERY route of the live AI Gateway OpenAPI snapshot.

For each route the same request is sent directly to the fake Gateway and through
the pool; the client-observable contract must match (status, body bytes,
content type, Retry-After) and the upstream must receive the same method, path,
query and body. Result level: MOCK_CONTRACT_QUALIFIED (not a live proof).

Set ``AIGP_MATRIX_OUT=<file>`` to dump the per-route results (used by the audit).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_gateway_pool.gateway.contracts import ENDPOINTS, GatewayEndpoint, ResponseKind
from tests.conftest import gateway_fixture
from tests.integration.live_servers import FakeState, ServerThread

Fake = tuple[FakeState, ServerThread]
RESULTS: dict[str, dict[str, Any]] = {}

REQUEST_BODIES: dict[str, dict[str, Any]] = {
    "/v1/generate": {"model": "llama3.1:8b", "prompt": "synthetic"},
    "/v1/generate/stream": {"model": "llama-chat", "prompt": "synthetic"},
    "/v1/chat": {"model": "llama-chat", "messages": [{"role": "user", "content": "hi"}]},
    "/v1/chat/stream": {"model": "llama-chat", "messages": [{"role": "user", "content": "hi"}]},
    "/v1/embeddings": {"model": "legacy-768", "task": "query", "texts": ["synthetic"]},
    "/v1/rerank": {"model": "rerank-default", "query": "q", "documents": ["a", "b"]},
    "/v1/vision/analyze": {
        "model": "vision-large",
        "schema_version": "1",
        "task": "x",
        "image": {"data": "AAAA", "mime_type": "image/png"},
    },
    "/v1/vision/structured-extract": {
        "schema_version": "1",
        "task": "chart",
        "image": {"data": "AAAA", "mime_type": "image/png"},
        "request_id": "synthetic-1",
    },
}
MULTIPART = {
    "request": (None, b'{"model":"asr-whisper-turbo","mode":"async"}', "application/json"),
    "file": ("audio", b"synthetic", "application/octet-stream"),
}


def _id(ep: GatewayEndpoint) -> str:
    return f"{ep.method} {ep.path}"


def _call(http: httpx.Client, base: str, ep: GatewayEndpoint, job: str | None) -> httpx.Response:
    path = ep.path.replace("{job_id}", job or "none")
    query = "?format=json&wait_ms=0" if ep.method == "GET" and job else ""
    if ep.path == "/v1/audio/transcriptions":
        return http.post(base + path, files=MULTIPART, headers={"X-Fake-Job-Id": job or "j"})
    if ep.method == "POST":
        return http.post(base + path, json=REQUEST_BODIES[ep.path])
    return http.request(ep.method, base + path + query)


def test_matrix_covers_the_open_contract_routes() -> None:
    from ai_gateway_pool.gateway.contracts import POOL_LOCAL_ROUTES

    live = {tuple(r) for r in gateway_fixture("openapi-routes.json")["routes"]}
    assert {(e.method, e.path) for e in ENDPOINTS} | set(POOL_LOCAL_ROUTES) == live
    assert len(live) == 20 and len(ENDPOINTS) == 19


@pytest.mark.parametrize(
    "method,path",
    [("POST", "/v1/vision/structured-extract"), ("GET", "/health/structured-extract")],
)
def test_excluded_route_is_refused_before_dispatch(
    fake: Fake, pool: str, http: httpx.Client, method: str, path: str
) -> None:
    """A route outside gateway-api/1.0 is refused by the pool (404 ROUTE_NOT_IN_CONTRACT)
    even though the fake upstream would serve it: nothing reaches the node."""
    state, _ = fake
    n0 = len(state.received)
    r = http.request(method, pool + path, json={"x": 1} if method == "POST" else None)
    assert r.status_code == 404
    assert r.headers.get("x-aigp-error") == "ROUTE_NOT_IN_CONTRACT"
    assert len(state.received) == n0


@pytest.mark.parametrize("ep", ENDPOINTS, ids=_id)
def test_route_relayed_like_direct(
    fake: Fake, pool: str, http: httpx.Client, ep: GatewayEndpoint
) -> None:
    state, srv = fake
    job: str | None = None
    if "{job_id}" in ep.path:
        # Create the job through the pool (affinity), mirror it for the direct call.
        job = f"m-{ep.method.lower()}"
        created = _call(
            http, pool, next(e for e in ENDPOINTS if e.path == "/v1/audio/transcriptions"), job
        )
        assert created.status_code == 202
    n0 = len(state.received)
    proxied = _call(http, pool, ep, job)
    via_pool = state.received[n0:]
    if ep.method == "DELETE" and job:
        state.jobs[job] = {}  # recreate for the direct comparison call
    elif ep.path == "/v1/audio/transcriptions":
        job = None
    n1 = len(state.received)
    direct = _call(http, srv.url, ep, job if "{job_id}" in ep.path else None)
    direct_rec = state.received[n1:]

    checks = {
        "status": proxied.status_code == direct.status_code,
        "body": proxied.content == direct.content,
        "content_type": proxied.headers.get("content-type") == direct.headers.get("content-type"),
        "upstream_method_path_query": [(r.method, r.raw_path, r.query) for r in via_pool]
        == [(r.method, r.raw_path, r.query) for r in direct_rec],
        "upstream_body": [r.body_sha256 for r in via_pool] == [r.body_sha256 for r in direct_rec]
        if ep.path != "/v1/audio/transcriptions"  # multipart boundary is random per call
        else [r.body_len > 0 for r in via_pool] == [True],
        "served_by_node": proxied.headers.get("x-fake-node") == "fake-staging",
    }
    if ep.response is ResponseKind.NDJSON:
        checks["ndjson_lines"] = proxied.text.count("\n") == direct.text.count("\n") >= 2
    RESULTS[_id(ep)] = {
        "capability": ep.capability.value if ep.capability else None,
        "status": proxied.status_code,
        **checks,
    }
    assert all(checks.values()), checks


@pytest.mark.parametrize("ep", [e for e in ENDPOINTS if e.method == "POST"], ids=_id)
def test_route_error_and_retry_after_relayed(
    fake: Fake, pool: str, http: httpx.Client, ep: GatewayEndpoint
) -> None:
    headers = {
        "X-Fake-Status": "503",
        "X-Fake-Code": "resource_saturated",
        "X-Fake-Retry-After": "3",
    }
    if ep.path == "/v1/audio/transcriptions":
        r = http.post(pool + ep.path, files=MULTIPART, headers=headers)
    else:
        r = http.post(pool + ep.path, json=REQUEST_BODIES[ep.path], headers=headers)
    assert r.status_code == 503 and r.headers["retry-after"] == "3"
    assert r.json()["detail"]["code"] == "resource_saturated"
    RESULTS.setdefault(_id(ep), {})["error_503_retry_after"] = True


def teardown_module() -> None:
    out = os.environ.get("AIGP_MATRIX_OUT")
    if out:
        Path(out).write_text(json.dumps(RESULTS, indent=1, sort_keys=True))
