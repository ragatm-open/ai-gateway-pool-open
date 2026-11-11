"""Preparatory contract tests.

These verify that the pool's contract MAP matches the AI Gateway Open OpenAPI
snapshot ``gateway-api/1.0`` (``tests/fixtures/gateway/openapi-routes.json``,
20 routes: 19 relayed + ``GET /health/live`` answered by the pool itself) and
records the right transport traits. They are marked ``contract_stub``: passing
them means "contract mapped", NOT "contract relayed compatibly" — that is POOL-2.
"""

from __future__ import annotations

import pytest

from ai_gateway_pool.gateway.contracts import (
    CONTRACT_VERSION,
    ENDPOINTS,
    EXCLUDED_ROUTES,
    POOL_LOCAL_ROUTES,
    BodyKind,
    ProxyStatus,
    ResponseKind,
    match_endpoint,
)
from ai_gateway_pool.nodes.models import Capability
from ai_gateway_pool.transport.contracts import TrafficClass
from tests.conftest import gateway_fixture

pytestmark = pytest.mark.contract_stub


def test_catalogue_matches_open_contract_snapshot() -> None:
    snapshot = gateway_fixture("openapi-routes.json")
    assert snapshot["contract_version"] == CONTRACT_VERSION == "gateway-api/1.0"
    live = {tuple(r) for r in snapshot["routes"]}
    mapped = {(e.method, e.path) for e in ENDPOINTS}
    assert len(live) == 20 and len(mapped) == 19
    assert mapped | set(POOL_LOCAL_ROUTES) == live
    # Excluded historical routes are neither in the snapshot nor dispatchable.
    for method, path in EXCLUDED_ROUTES:
        assert (method, path) not in live
        assert match_endpoint(method, path) is None


def test_every_route_is_relayed() -> None:
    # POOL-1 asserted NOT_IMPLEMENTED; POOL-2 relays every catalogued route (relay
    # behaviour is proven by test_route_relay_matrix.py, not by this flag).
    assert {e.proxy_status for e in ENDPOINTS} == {ProxyStatus.RELAYED}


def test_inferential_health_routes_flagged() -> None:
    for path in ("/health/ready", "/health/embeddings", "/health/rerank"):
        ep = match_endpoint("GET", path)
        assert ep is not None and ep.gpu_compute, path
    runtime = match_endpoint("GET", "/health/runtime")
    assert runtime is not None and not runtime.gpu_compute


def _ep(method: str, path: str):  # type: ignore[no-untyped-def]
    ep = match_endpoint(method, path)
    assert ep is not None, (method, path)
    return ep


def test_generate_contract() -> None:
    ep = _ep("POST", "/v1/generate")
    assert (ep.capability, ep.body, ep.response) == (
        Capability.GENERATE,
        BodyKind.JSON,
        ResponseKind.JSON,
    )


def test_generate_stream_contract() -> None:
    ep = _ep("POST", "/v1/generate/stream")
    assert ep.response is ResponseKind.NDJSON and ep.traffic is TrafficClass.STREAMING


def test_chat_contract() -> None:
    ep = _ep("POST", "/v1/chat")
    assert ep.capability is Capability.CHAT and ep.response is ResponseKind.JSON


def test_chat_stream_contract() -> None:
    ep = _ep("POST", "/v1/chat/stream")
    assert ep.response is ResponseKind.NDJSON and ep.traffic is TrafficClass.STREAMING


def test_embeddings_contract() -> None:
    ep = _ep("POST", "/v1/embeddings")
    assert ep.capability is Capability.EMBEDDINGS and ep.gpu_compute


def test_rerank_contract() -> None:
    assert _ep("POST", "/v1/rerank").capability is Capability.RERANK


def test_vision_contract() -> None:
    assert _ep("POST", "/v1/vision/analyze").capability is Capability.VISION
    assert not _ep("GET", "/v1/vision/capacity").gpu_compute


def test_structured_extract_is_not_dispatchable() -> None:
    """Open edition: StructX has no route; the pool must never dispatch it to a node."""
    assert match_endpoint("POST", "/v1/vision/structured-extract") is None
    assert match_endpoint("GET", "/health/structured-extract") is None
    assert Capability.STRUCTURED_EXTRACT not in {e.capability for e in ENDPOINTS}


def test_asr_job_contract() -> None:
    submit = _ep("POST", "/v1/audio/transcriptions")
    assert submit.body is BodyKind.MULTIPART and submit.traffic is TrafficClass.UPLOAD
    poll = _ep("GET", "/v1/audio/transcriptions/abc-123")
    delete = _ep("DELETE", "/v1/audio/transcriptions/abc-123")
    assert all(e.async_job for e in (submit, poll, delete))
    assert poll.response is ResponseKind.JOB_RESULT
    assert match_endpoint("GET", "/v1/audio/transcriptions/a/b") is None
    assert match_endpoint("PUT", "/v1/audio/transcriptions/abc") is None
