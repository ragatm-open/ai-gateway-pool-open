"""Security of the relay, model/capability routing and resilience."""

from __future__ import annotations

import io
import json
import logging
import socket
import time
from collections.abc import Callable

import httpx
import pytest

from ai_gateway_pool.observability.logging import JsonFormatter
from tests.conftest import ADMIN_TOKEN
from tests.integration.live_servers import FakeState, ServerThread, fake_gateway_app

Fake = tuple[FakeState, ServerThread]
AUTH = {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def raw_request(pool: str, data: bytes) -> bytes:
    host, port = pool.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=5) as s:
        s.sendall(data)
        chunks = []
        try:
            while chunk := s.recv(65536):
                chunks.append(chunk)
        except TimeoutError:
            pass
    return b"".join(chunks)


# ------------------------------------------------------------------ security


def test_admin_token_not_forwarded(fake: Fake, pool: str, http: httpx.Client) -> None:
    http.get(pool + "/v1/models", headers=AUTH)
    got = {k.lower(): v for k, v in fake[0].received[-1].headers}
    assert "authorization" not in got
    # A client credential that is NOT the pool admin token is relayed unchanged.
    http.get(pool + "/v1/models", headers={"Authorization": "Bearer client-gw-token"})
    got = {k.lower(): v for k, v in fake[0].received[-1].headers}
    assert got["authorization"] == "Bearer client-gw-token"


def test_authorization_and_bodies_not_logged(fake: Fake, pool: str, http: httpx.Client) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("ai_gateway_pool")
    level = logger.level
    logger.setLevel(logging.INFO)  # the harness runs the pool at WARNING
    logger.addHandler(handler)
    try:
        http.post(
            pool + "/v1/embeddings",
            json={"model": "legacy-768", "texts": ["CONFIDENTIAL-PROMPT-TEXT"]},
            headers={
                "Authorization": "Bearer sk-live-SECRET123456789",
                "Cookie": "session=COOKIE-SECRET",
                "X-API-Key": "APIKEY-SECRET",
            },
        )
        http.post(pool + "/v1/generate/stream", json={"prompt": "PROMPT-SECRET"})
        time.sleep(0.2)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    out = stream.getvalue()
    lines = [json.loads(x) for x in out.splitlines()]
    relayed = [x for x in lines if x.get("selected_node_id") == "fake-staging"]
    assert relayed and relayed[0]["model_id"] == "legacy-768"
    assert any(x.get("streaming") is True for x in relayed)
    for secret in (
        "SECRET123",
        "COOKIE-SECRET",
        "APIKEY-SECRET",
        "CONFIDENTIAL-PROMPT",
        "PROMPT-SECRET",
        "t0",
        "synthetic",
    ):
        assert secret not in out, secret


@pytest.mark.parametrize("header", ["X-Upstream-URL", "X-Forwarded-Host", "Forwarded"])
def test_unregistered_upstream_denied(
    fake: Fake, pool: str, http: httpx.Client, header: str
) -> None:
    r = http.post(pool + "/v1/generate", json={}, headers={header: "http://169.254.169.254"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "UPSTREAM_OVERRIDE_DENIED"
    for target in ("http://169.254.169.254", "fake-production", "unknown"):
        # unauthenticated pinning: refused before any destination check (AIGP-NEW-03)
        r = http.post(pool + "/v1/generate", json={}, headers={"X-AIGP-Target-Node": target})
        assert r.status_code == 403 and r.json()["error"]["code"] == "TARGET_NODE_FORBIDDEN"
        # admin-authenticated: still only a configured node of this environment
        r = http.post(
            pool + "/v1/generate", json={}, headers={"X-AIGP-Target-Node": target, **AUTH}
        )
        assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_NODE_DESTINATION"
    assert fake[0].received == []


def test_external_url_in_request_target_denied(fake: Fake, pool: str) -> None:
    resp = raw_request(
        pool,
        b"GET http://evil.example/v1/models HTTP/1.1\r\n"
        b"Host: evil.example\r\nConnection: close\r\n\r\n",
    )
    status = int(resp.split(b" ", 2)[1])
    assert status in (400, 404)
    assert fake[0].received == []


@pytest.mark.parametrize(
    "path",
    [
        "/v1/audio/transcriptions/%2e%2e",
        "/v1/audio/transcriptions/..%2Fhealth",
        "/v1/%6dodels",
        "/v1/audio/transcriptions/a%5cb",
        "/v1//models",
        "/v1/models/../generate",
    ],
)
def test_invalid_path_denied(fake: Fake, pool: str, path: str) -> None:
    resp = raw_request(
        pool, f"GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".encode()
    )
    status = int(resp.split(b" ", 2)[1])
    assert status in (400, 404), (path, status)
    assert fake[0].received == []


def test_host_header_not_trusted(fake: Fake, pool: str, http: httpx.Client) -> None:
    http.get(pool + "/v1/models", headers={"Host": "evil.example"})
    got = {k.lower(): v for k, v in fake[0].received[-1].headers}
    assert got["host"] == fake[1].url.removeprefix("http://")


def test_contradictory_framing_rejected(fake: Fake, pool: str) -> None:
    resp = raw_request(
        pool,
        b"POST /v1/generate HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n"
        b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n0\r\n\r\n",
    )
    assert int(resp.split(b" ", 2)[1]) == 400
    assert b"INVALID_REQUEST_FRAMING" in resp
    assert fake[0].received == []


def test_no_public_admin_access(fake: Fake, pool: str, http: httpx.Client) -> None:
    for path in ("/internal/nodes", "/internal/capabilities", "/internal/metrics"):
        assert http.get(pool + path).status_code == 401
    assert http.post(pool + "/internal/nodes/fake-staging/probe").status_code == 401
    from ai_gateway_pool.config.settings import Settings

    assert Settings.model_fields["bind_host"].default == "127.0.0.1"


# ------------------------------------------------------------------ models


def test_exact_model_selection_and_body_unchanged(
    fake: Fake, pool: str, http: httpx.Client
) -> None:
    for path, model in [
        ("/v1/embeddings", "legacy-768"),
        ("/v1/generate", "llama3.1:8b"),
        ("/v1/chat", "llama-chat"),
        ("/v1/rerank", "rerank-default"),
    ]:
        raw = json.dumps({"model": model, "x": 1}).encode()
        r = http.post(pool + path, content=raw, headers={"Content-Type": "application/json"})
        assert r.status_code == 200, (path, r.text)
        import hashlib

        assert r.json()["echo"]["body_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    ("path", "model"),
    [
        ("/v1/embeddings", "fr-en-retrieval"),  # 768-d like legacy-768, still incompatible
        ("/v1/embeddings", "bge-m3:latest"),
        ("/v1/generate", "gpt-imaginary"),
        ("/v1/chat", "legacy-768"),
    ],  # embedding model on a chat route
)
def test_incompatible_or_missing_model_rejected(
    fake: Fake, pool: str, http: httpx.Client, path: str, model: str
) -> None:
    r = http.post(pool + path, json={"model": model, "texts": ["x"]})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "MODEL_NOT_AVAILABLE"
    assert fake[0].received == []  # never sent to a node "hoping it works"


def test_request_without_model_routed_by_capability(
    fake: Fake, pool: str, http: httpx.Client
) -> None:
    assert http.post(pool + "/v1/embeddings", json={"texts": ["x"]}).status_code == 200


def _probe(http: httpx.Client, pool: str) -> dict[str, object]:
    return http.post(pool + "/internal/nodes/fake-staging/probe", headers=AUTH).json()


def test_unavailable_node_rejected_but_diagnostics_relayed(
    make_pool: Callable[..., ServerThread], http: httpx.Client
) -> None:
    state = FakeState(node="fake-staging", environment="production")  # mis-wired port!
    with ServerThread(fake_gateway_app(state)) as srv:
        pool = make_pool(srv.url).url
        view = _probe(http, pool)
        assert (
            view["health_state"] == "UNAVAILABLE" and view["last_error"] == "environment_mismatch"
        )
        assert view["reachability"] == "REACHABLE"
        r = http.post(pool + "/v1/generate", json={"prompt": "x"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "NODE_UNAVAILABLE"
        assert http.get(pool + "/pool/health/ready").status_code == 503
        # Diagnostics still reach the node so the operator sees its real answer.
        assert http.get(pool + "/health/runtime").json()["environment"] == "production"


def test_cross_environment_rejected(
    fake: Fake, make_pool: Callable[..., ServerThread], http: httpx.Client
) -> None:
    # A production pool whose node answers "staging": refused after probe, and no
    # other node exists to fall back to — the pool reports a production outage.
    pool = make_pool(fake[1].url, env="production", backend="none").url
    r = http.post(pool + "/internal/nodes/fake-production/probe", headers=AUTH)
    assert r.json()["last_error"] == "environment_mismatch"
    g = http.post(pool + "/v1/generate", json={"prompt": "x"})
    assert g.status_code == 503 and g.json()["error"]["code"] == "NODE_UNAVAILABLE"


def test_observed_inventory_used_after_probe(fake: Fake, pool: str, http: httpx.Client) -> None:
    view = _probe(http, pool)
    # `0ba5e0ba…` : identifiant de build SYNTHÉTIQUE porté par la fixture.
    assert view["health_state"] == "HEALTHY" and view["build_git_sha"].startswith("0ba5e0ba")
    caps = http.get(pool + "/internal/capabilities", headers=AUTH).json()["nodes"][0]
    assert "legacy-768" in caps["observed_model_ids"]
    probe_paths = {x.path for x in fake[0].received}
    assert probe_paths == {"/health/runtime", "/v1/models"}  # never /health/ready


def test_metrics_exposed_after_traffic(fake: Fake, pool: str, http: httpx.Client) -> None:
    http.get(pool + "/v1/models")
    http.post(pool + "/v1/generate/stream", json={})
    http.post(pool + "/v1/embeddings", json={"model": "nope"})
    text = http.get(pool + "/internal/metrics", headers=AUTH).text
    assert 'aigp_proxy_requests_total{environment="staging",service="models"' in text
    assert "aigp_proxy_streaming_responses_total" in text
    assert (
        'aigp_proxy_model_unavailable_total{environment="staging",service="embeddings"} 1' in text
    )
    assert 'aigp_proxy_active_requests{environment="staging",node_id="fake-staging"} 0' in text
