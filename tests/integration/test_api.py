"""Local API (brief §20 'API locale' and 'Sécurité')."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from ai_gateway_pool.config.schema import PoolConfig
from tests.conftest import ADMIN_TOKEN, build_config, make_client, node
from tests.integration.fake_gateway import FakeGateway

AUTH = {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def test_health_live(client: TestClient) -> None:
    r = client.get("/health/live")
    assert r.status_code == 200 and r.json() == {"status": "live"}
    assert r.headers["X-Request-ID"]


def test_health_ready(client: TestClient) -> None:
    # POOL-2: pool readiness moved to /pool/health/ready; /health/ready is the
    # AI Gateway's own route and is relayed (see test_contract_routes_relayed).
    r = client.get("/pool/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["environment"] == "staging"
    assert body["proxy"] == "single_node"


def test_public_health_reveals_no_topology(client: TestClient) -> None:
    for path in ("/health/live", "/pool/health/live", "/pool/health/ready"):
        text = client.get(path).text
        for leak in ("http://", "gw-staging", "gpu-host-a", "8200", "legacy-768"):
            assert leak not in text, (path, leak)


def test_no_public_api_docs(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_internal_node_view(client: TestClient) -> None:
    r = client.get("/internal/nodes", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    [n] = body["nodes"]
    assert n["node_id"] == "gw-staging"
    assert n["physical_host_id"] == "gpu-host-a"
    assert n["health_state"] == "UNKNOWN"
    assert body["physical_hosts"] == [
        {
            "physical_host_id": "gpu-host-a",
            "node_ids": ["gw-staging"],
            "shared_gpu_between_nodes": False,
        }
    ]


def test_internal_capabilities_view(client: TestClient) -> None:
    r = client.get("/internal/capabilities", headers=AUTH)
    assert r.status_code == 200
    caps = r.json()["nodes"][0]["capabilities"]
    # Declared, not yet observed: never reported AVAILABLE.
    assert set(caps.values()) == {"DECLARED"}
    assert r.json()["nodes"][0]["observed_model_ids"] is None


def test_no_secret_exposure(client: TestClient) -> None:
    for path in ("/internal/nodes", "/internal/capabilities", "/health/ready"):
        assert ADMIN_TOKEN not in client.get(path, headers=AUTH).text


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": ADMIN_TOKEN},
        {"Authorization": "Basic " + ADMIN_TOKEN},
    ],
)
def test_unauthorized_internal_access_denied(client: TestClient, headers: dict[str, str]) -> None:
    for path in ("/internal/nodes", "/internal/capabilities"):
        r = client.get(path, headers=headers)
        assert r.status_code == 401
        assert "gpu-host-a" not in r.text
    assert client.post("/internal/nodes/gw-staging/probe", headers=headers).status_code == 401


def test_internal_routes_absent_without_token(staging_config: PoolConfig) -> None:
    c = make_client(staging_config, admin_token=None)
    assert c.get("/internal/nodes", headers=AUTH).status_code == 404


@pytest.mark.parametrize(
    "header", ["X-Upstream-URL", "X-Forwarded-Host", "Forwarded", "X-Gateway-URL"]
)
def test_client_supplied_upstream_url_denied(staging_config: PoolConfig, header: str) -> None:
    fake = FakeGateway()
    c = make_client(staging_config)
    c.app.state.pool.client_factory = fake.client
    for method, path in (("POST", "/v1/generate"), ("GET", "/health/ready")):
        r = c.request(method, path, headers={header: "http://evil.example:80"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "UPSTREAM_OVERRIDE_DENIED"
    assert fake.calls == []


def test_absolute_url_in_path_is_not_a_destination(client: TestClient) -> None:
    r = client.get("/v1/http://evil.example/steal")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "ROUTE_NOT_IN_CONTRACT"


def test_invalid_node_destination_denied(client: TestClient) -> None:
    # Even an admin-authenticated caller can only pin a CONFIGURED node.
    for target in ("http://evil.example", "unknown-node", "gw-production"):
        r = client.post("/v1/embeddings", headers={"X-AIGP-Target-Node": target, **AUTH}, json={})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_NODE_DESTINATION"


def test_target_node_requires_admin_credentials(client: TestClient) -> None:
    """AIGP-HARDEN-1 (AIGP-NEW-03): node pinning is privileged."""

    for headers in (
        {},
        {"Authorization": "Bearer not-the-admin-token"},
        {"Authorization": f"Basic {ADMIN_TOKEN}"},
    ):
        for target in ("gw-staging", "http://evil.example", "unknown-node"):
            r = client.post(
                "/v1/embeddings", headers={"X-AIGP-Target-Node": target, **headers}, json={}
            )
            assert r.status_code == 403
            assert r.json()["error"]["code"] == "TARGET_NODE_FORBIDDEN"
            assert r.headers["x-aigp-error"] == "TARGET_NODE_FORBIDDEN"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v1/generate"),
        ("POST", "/v1/generate/stream"),
        ("POST", "/v1/chat"),
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/rerank"),
        ("POST", "/v1/vision/analyze"),
        ("POST", "/v1/vision/structured-extract"),  # excluded from the Open contract
        ("POST", "/v1/audio/transcriptions"),
        ("GET", "/v1/audio/transcriptions/job-1"),
        ("DELETE", "/v1/audio/transcriptions/job-1"),
        ("GET", "/v1/models"),
    ],
)
def test_contract_routes_relayed_to_registry_node(
    staging_config: PoolConfig, method: str, path: str
) -> None:
    """POOL-2: contract routes are relayed (POOL-1 asserted 501 here). ASR job
    routes stay refused when no durable affinity backend is configured."""

    fake = FakeGateway()
    c = make_client(staging_config)
    c.app.state.pool.client_factory = fake.client
    r = c.request(method, path)
    if path == "/v1/vision/structured-extract":
        # Open edition: no such route in gateway-api/1.0 — refused before any dispatch.
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "ROUTE_NOT_IN_CONTRACT"
        assert fake.calls == []
    elif path.startswith("/v1/vision/"):
        # The POOL-1 test node does not declare VISION.
        assert r.json()["error"]["code"] == "CAPABILITY_NOT_AVAILABLE"
        assert fake.calls == []
    elif path.startswith("/v1/audio/transcriptions"):
        assert r.status_code == 501
        assert r.json()["error"]["code"] == "ROUTING_NOT_READY"
        assert fake.calls == []
    else:
        assert fake.calls == [(method, path)]  # reached the configured node, verbatim
        assert "x-aigp-error" not in r.headers  # upstream answer relayed as-is


def test_unknown_v1_route_rejected(client: TestClient) -> None:
    r = client.post("/v1/admin/shutdown")
    assert r.status_code == 404 and r.json()["error"]["code"] == "ROUTE_NOT_IN_CONTRACT"


def test_probe_endpoint_with_fake_node(staging_config: PoolConfig) -> None:
    fake = FakeGateway()
    c = make_client(staging_config)
    c.app.state.pool.client_factory = fake.client
    r = c.post("/internal/nodes/gw-staging/probe", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["health_state"] == "HEALTHY"
    assert fake.calls == [("GET", "/health/runtime"), ("GET", "/v1/models")]
    caps = c.get("/internal/capabilities", headers=AUTH).json()["nodes"][0]["capabilities"]
    assert caps["ASR"] == "AVAILABLE" and caps["EMBEDDINGS"] == "DECLARED"


def test_probe_unknown_node_makes_no_call(staging_config: PoolConfig) -> None:
    fake = FakeGateway()
    c = make_client(staging_config)
    c.app.state.pool.client_factory = fake.client
    r = c.post("/internal/nodes/http:%2F%2Fevil.example/probe", headers=AUTH)
    assert r.status_code in (400, 404)
    assert fake.calls == []


def test_request_id_propagation(client: TestClient) -> None:
    assert (
        client.get("/health/live", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"]
        == "abc-123"
    )
    generated = client.get("/health/live", headers={"X-Request-ID": "bad id\n"}).headers[
        "X-Request-ID"
    ]
    assert generated != "bad id\n" and len(generated) == 32


def test_access_log_has_no_body_or_auth(
    staging_config: PoolConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    c = make_client(staging_config)
    c.post(
        "/v1/embeddings",
        json={"texts": ["CONFIDENTIAL-DOC"]},
        headers={"Authorization": "Bearer sk-secret-xyz123456"},
    )
    out = capsys.readouterr().out
    lines = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    assert any(rec.get("response_status") for rec in lines)
    assert "CONFIDENTIAL-DOC" not in out and "sk-secret" not in out
    logging.getLogger("ai_gateway_pool").handlers.clear()


def test_multi_node_registry_through_api() -> None:
    cfg = build_config(
        nodes=[
            node("a", url="http://a.invalid:1", host="h"),
            node("b", url="http://b.invalid:1", host="h"),
        ],
        routing={"policy": "WEIGHTED_ROUND_ROBIN"},
    )
    body = make_client(cfg).get("/internal/nodes", headers=AUTH).json()
    assert body["physical_hosts"][0]["shared_gpu_between_nodes"] is True


def test_internal_runtime_identity(client: TestClient) -> None:
    """POOL-4: deployment identity, protected like every /internal route."""

    assert client.get("/internal/runtime").status_code == 401
    body = client.get("/internal/runtime", headers=AUTH).json()
    assert body["environment"] == "staging" and body["routing_policy"] == "SINGLE_NODE"
    assert body["workers"] == 1 and body["nodes_enabled"] == ["gw-staging"]
    assert ADMIN_TOKEN not in str(body)
