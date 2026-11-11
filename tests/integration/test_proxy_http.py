"""HTTP transparency of the relay (real pool server -> real fake Gateway server)."""

from __future__ import annotations

import json

import httpx
import pytest

from tests.integration.live_servers import FakeState, ServerThread

Fake = tuple[FakeState, ServerThread]


def test_http_get_proxy(fake: Fake, pool: str, http: httpx.Client) -> None:
    r = http.get(pool + "/v1/models")
    assert r.status_code == 200
    assert r.headers["x-fake-node"] == "fake-staging"
    assert r.json()["provider"] == "ollama"
    assert fake[0].received[-1].method == "GET"


def test_http_post_json_proxy_body_verbatim(fake: Fake, pool: str, http: httpx.Client) -> None:
    raw = b'{"model": "legacy-768", "task":"query",  "texts": ["synthetic"]}'  # odd spacing kept
    r = http.post(
        pool + "/v1/embeddings", content=raw, headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 200
    echo = r.json()["echo"]
    import hashlib

    assert echo["body_sha256"] == hashlib.sha256(raw).hexdigest()
    assert echo["method"] == "POST" and echo["path"] == "/v1/embeddings"


def test_query_parameters_preserved_raw(fake: Fake, pool: str, http: httpx.Client) -> None:
    fake[0].jobs["job-q"] = {}
    # Register affinity by creating the job through the pool first.
    files = {"request": (None, b"{}", "application/json"), "file": ("audio", b"abc")}
    created = http.post(
        pool + "/v1/audio/transcriptions", files=files, headers={"X-Fake-Job-Id": "job-q"}
    )
    assert created.status_code == 202
    raw_query = "format=srt&wait_ms=0&tag=a&tag=b&x=%C3%A9t%C3%A9"
    r = http.get(f"{pool}/v1/audio/transcriptions/job-q?{raw_query}")
    assert r.status_code == 200
    assert fake[0].received[-1].query == raw_query  # repeated values + encoding untouched


def test_request_and_response_content_type_preserved(
    fake: Fake, pool: str, http: httpx.Client
) -> None:
    r = http.post(
        pool + "/v1/generate",
        content=b"{}",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert r.json()["echo"]["content_type"] == "application/json; charset=utf-8"
    s = http.post(pool + "/v1/generate/stream", json={"prompt": "x"})
    assert s.headers["content-type"].startswith("application/x-ndjson")


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 422, 429, 500, 502, 503, 504])
def test_status_code_and_error_body_preserved(
    fake: Fake, pool: str, http: httpx.Client, status: int
) -> None:
    r = http.post(
        pool + "/v1/chat",
        json={"model": "llama-chat", "messages": []},
        headers={"X-Fake-Status": str(status), "X-Fake-Code": "some_code"},
    )
    assert r.status_code == status
    assert r.json() == {
        "detail": {"status": "error", "code": "some_code", "provider": "ai_gateway"}
    }
    assert "x-aigp-error" not in r.headers  # a Gateway error is not a pool error


def test_resource_saturated_and_retry_after_preserved(
    fake: Fake, pool: str, http: httpx.Client
) -> None:
    r = http.post(
        pool + "/v1/vision/analyze",
        json={"schema_version": "1"},
        headers={
            "X-Fake-Status": "503",
            "X-Fake-Code": "resource_saturated",
            "X-Fake-Retry-After": "7",
        },
    )
    assert r.status_code == 503
    assert r.headers["retry-after"] == "7"
    assert r.json()["detail"]["code"] == "resource_saturated"


def test_unknown_route_rejected(fake: Fake, pool: str, http: httpx.Client) -> None:
    before = len(fake[0].received)
    for method, path in [
        ("POST", "/v1/admin/shutdown"),
        ("GET", "/v1/generate"),
        ("PUT", "/v1/models"),
        ("GET", "/health/gpu"),
    ]:
        r = http.request(method, pool + path)
        assert r.status_code == 404, path
        assert r.json()["error"]["code"] == "ROUTE_NOT_IN_CONTRACT"
    assert len(fake[0].received) == before


def test_upstream_unavailable_handled(make_pool, http: httpx.Client) -> None:  # type: ignore[no-untyped-def]
    from tests.integration.live_servers import free_port

    pool = make_pool(f"http://127.0.0.1:{free_port()}").url  # nothing listens there
    r = http.post(pool + "/v1/generate", json={"prompt": "x"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "UPSTREAM_UNAVAILABLE"
    assert r.headers["x-aigp-error"] == "UPSTREAM_UNAVAILABLE"
    # The pool's own routes stay usable while the Gateway is down.
    assert http.get(pool + "/health/live").status_code == 200
    assert (
        http.get(
            pool + "/internal/nodes",
            headers={"Authorization": "Bearer test-admin-token-not-a-secret-0123456789"},
        ).status_code
        == 200
    )


def test_upstream_timeout_handled(fake: Fake, make_pool, http: httpx.Client) -> None:  # type: ignore[no-untyped-def]
    pool = make_pool(
        fake[1].url,
        timeouts={"inference": {"connect": 2, "read": 0.5, "write": 5, "pool": 2}},
    ).url
    r = http.post(pool + "/v1/generate", json={"prompt": "x"}, headers={"X-Fake-Delay": "2"})
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "UPSTREAM_TIMEOUT"
    assert "NOT retried" in r.json()["error"]["message"]
    gen_calls = [x for x in fake[0].received if x.path == "/v1/generate"]
    assert len(gen_calls) == 1  # no blind retry of an inference POST


def test_hop_by_hop_not_forwarded_and_request_id(fake: Fake, pool: str, http: httpx.Client) -> None:
    http.post(
        pool + "/v1/rerank",
        json={"query": "q", "documents": ["a"]},
        headers={
            "Connection": "keep-alive, X-Secret-Hop",
            "X-Secret-Hop": "1",
            "Keep-Alive": "timeout=5",
            "X-Request-ID": "req-123",
            "Accept": "application/json",
        },
    )
    got = {k.lower(): v for k, v in fake[0].received[-1].headers}
    assert "x-secret-hop" not in got and "keep-alive" not in got
    assert got["x-request-id"] == "req-123"
    assert got["accept"] == "application/json"
    assert got["host"].startswith("127.0.0.1")  # recomputed for the upstream


def test_large_json_response_streamed(fake: Fake, pool: str, http: httpx.Client) -> None:
    size = 32 * 1024 * 1024
    with http.stream(
        "GET", pool + "/health/vision", headers={"X-Fake-Response-Bytes": str(size)}
    ) as r:
        assert r.headers["content-length"] == str(size)
        total = sum(len(c) for c in r.iter_raw())
    assert total == size


def test_get_retried_once_on_connection_failure_never_post() -> None:
    """Unit-level check of the relay's retry rule through a scripted transport."""

    import asyncio
    from pathlib import Path

    from ai_gateway_pool.config.settings import Settings
    from ai_gateway_pool.main import create_app
    from tests.conftest import build_config

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if len([c for c in calls if c == request.method]) == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"ok": True})

    cfg = build_config()
    app = create_app(Settings(environment="staging", config_file=Path("/x")), cfg)
    app.state.pool.client_factory = lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )

    async def go() -> tuple[int, int]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://pool") as c:
            g = await c.get("/v1/models")
            p = await c.post(
                "/v1/generate",
                content=json.dumps({"prompt": "x"}),
                headers={"Content-Type": "application/json"},
            )
            return g.status_code, p.status_code

    g, p = asyncio.run(go())
    assert g == 200 and calls.count("GET") == 2  # retried once
    assert p == 503 and calls.count("POST") == 1  # never retried
