"""Bounded discovery and streaming primitives against fake nodes."""

from __future__ import annotations

import httpx
import pytest

from ai_gateway_pool.errors import PoolErrorCode, RoutingError
from ai_gateway_pool.nodes.discovery import probe_node
from ai_gateway_pool.nodes.models import Capability, Environment, HealthState, ModelRequirement
from ai_gateway_pool.nodes.registry import NodeRegistry
from ai_gateway_pool.routing.contracts import RoutingRequest
from ai_gateway_pool.routing.selector import SingleNodeSelector
from ai_gateway_pool.transport.contracts import DEFAULT_TIMEOUTS, TrafficClass
from ai_gateway_pool.transport.streaming import (
    UpstreamStreamError,
    relay_request_body,
    relay_response_body,
)
from tests.conftest import build_config
from tests.integration.fake_gateway import NDJSON_LINES, FakeGateway, NdjsonStream

NODE = "gw-staging"
PROBE = DEFAULT_TIMEOUTS[TrafficClass.HEALTH_PROBE]


async def _probe(scenario: str, environment: str = "staging") -> tuple[NodeRegistry, FakeGateway]:
    reg = NodeRegistry.from_config(build_config())
    fake = FakeGateway(scenario, environment)
    async with fake.client() as client:
        await probe_node(reg, NODE, client, probe_path="/health/runtime", timeouts=PROBE)
    return reg, fake


def _embed_request() -> RoutingRequest:
    return RoutingRequest(
        environment=Environment.STAGING,
        service="embeddings",
        capability=Capability.EMBEDDINGS,
        model=ModelRequirement(model_id="legacy-768"),
    )


async def test_service_available() -> None:
    reg, fake = await _probe("available")
    state = reg.state(NODE)
    assert state.health_state is HealthState.HEALTHY
    assert state.last_checked_at is not None
    assert state.observed_model_ids and "legacy-768" in state.observed_model_ids
    assert fake.calls == [("GET", "/health/runtime"), ("GET", "/v1/models")]
    assert SingleNodeSelector().select(_embed_request(), reg).node_id == NODE


@pytest.mark.parametrize(
    ("scenario", "error"),
    [
        ("unavailable", "probe_connect_error"),
        ("timeout", "probe_timeout"),
        ("http_503", "probe_http_503"),
        ("invalid_response", "runtime_payload_invalid"),
    ],
)
async def test_failure_scenarios_mark_node_unavailable(scenario: str, error: str) -> None:
    reg, fake = await _probe(scenario)
    state = reg.state(NODE)
    assert state.health_state is HealthState.UNAVAILABLE
    assert state.last_error == error
    assert fake.calls == [("GET", "/health/runtime")]
    with pytest.raises(RoutingError) as exc:
        SingleNodeSelector().select(_embed_request(), reg)
    assert exc.value.code is PoolErrorCode.NODE_UNAVAILABLE


async def test_wrong_environment_behind_url_is_refused() -> None:
    """A staging node whose URL actually answers 'production' is never used."""

    reg, _ = await _probe("available", environment="production")
    assert reg.state(NODE).health_state is HealthState.UNAVAILABLE
    assert reg.state(NODE).last_error == "environment_mismatch"


async def test_model_missing_on_node() -> None:
    reg, _ = await _probe("model_missing")
    with pytest.raises(RoutingError) as exc:
        SingleNodeSelector().select(_embed_request(), reg)
    assert exc.value.code is PoolErrorCode.MODEL_NOT_AVAILABLE


async def test_inventory_down_degrades_without_restricting_models() -> None:
    reg, _ = await _probe("inventory_down")
    state = reg.state(NODE)
    assert state.health_state is HealthState.DEGRADED
    assert state.observed_model_ids is None
    assert SingleNodeSelector().select(_embed_request(), reg).node_id == NODE


async def test_probe_never_touches_inferential_endpoints() -> None:
    for scenario in ("available", "inventory_down", "model_missing"):
        _, fake = await _probe(scenario)
        paths = {p for _, p in fake.calls}
        assert paths <= {"/health/runtime", "/v1/models"}


async def test_asr_unknown_job_fixture() -> None:
    fake = FakeGateway()
    async with fake.client() as client:
        r = await client.get("http://gw.invalid/v1/audio/transcriptions/never-seen")
    assert r.status_code == 404 and r.json()["detail"]["code"] == "job_not_found"


# --------------------------------------------------------------------- streaming


async def _stream_response(interrupt: bool) -> tuple[httpx.Response, NdjsonStream]:
    stream = NdjsonStream(interrupt=interrupt)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "application/x-ndjson"}, stream=stream)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    req = client.build_request("POST", "http://gw.invalid/v1/generate/stream", json={})
    return await client.send(req, stream=True), stream


async def test_ndjson_relayed_verbatim_not_aggregated() -> None:
    response, stream = await _stream_response(interrupt=False)
    chunks = [c async for c in relay_response_body(response)]
    assert b"".join(chunks) == b"".join(NDJSON_LINES)
    assert len(chunks) > 1  # chunk by chunk, not one aggregated document
    assert stream.closed


async def test_interrupted_stream_raises_and_closes() -> None:
    response, stream = await _stream_response(interrupt=True)
    received: list[bytes] = []
    with pytest.raises(UpstreamStreamError):
        async for chunk in relay_response_body(response):
            received.append(chunk)
    assert b"done" not in b"".join(received)  # no fabricated terminal event
    assert stream.closed


async def test_client_disconnect_closes_upstream() -> None:
    response, stream = await _stream_response(interrupt=False)
    gen = relay_response_body(response)
    first = await gen.__anext__()
    assert first == NDJSON_LINES[0]
    await gen.aclose()  # downstream client went away
    assert stream.closed
    assert stream.pulled < len(NDJSON_LINES)  # backpressure: nothing read ahead


async def test_request_body_streamed_without_buffering() -> None:
    pulled: list[int] = []

    async def source():  # type: ignore[no-untyped-def]
        for i in range(3):
            pulled.append(i)
            yield b"x" * 1024

    gen = relay_request_body(source())
    await gen.__anext__()
    assert pulled == [0]  # only one chunk materialised
