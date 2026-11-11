"""AIGP-HARDEN-1 / AIGP-01: model governance is fail-closed whatever the HTTP framing.

The 31-case reproduction of the POST-DEEP-REVIEW assessment, on the 6 JSON
routes that carry a model, against a REAL fake-Gateway HTTP server (never a real
backend). Invariant: no model outside the registry is ever relayed; a body whose
model cannot be established is refused (typed pool error), never treated as
"no model".
"""

from __future__ import annotations

import gzip
import hashlib
import json
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_gateway_pool.config.schema import PoolConfig
from tests.conftest import raw_config
from tests.integration.conftest import full_node, pool_server
from tests.integration.live_servers import FakeState, ServerThread, fake_gateway_app

LIMIT = 4096
UNKNOWN = "gpt-imaginary"
# route -> a model the test registry declares for that route's capability
ROUTES: dict[str, str] = {
    "/v1/embeddings": "legacy-768",
    "/v1/generate": "llama3.1:8b",
    "/v1/generate/stream": "llama3.1:8b",
    "/v1/chat": "llama-chat",
    "/v1/rerank": "rerank-default",
    "/v1/vision/analyze": "vision-large",
}
JSON = {"Content-Type": "application/json"}

Fake = tuple[FakeState, ServerThread]


def _config(
    nodes: list[dict[str, Any]], policy: str = "SINGLE_NODE", **transport: int
) -> PoolConfig:
    return PoolConfig.model_validate(
        raw_config(
            nodes=nodes,
            routing={"policy": policy},
            transport={"json_inspect_max_bytes": LIMIT, "json_inspect_budget_bytes": 2 * LIMIT}
            | transport,
            job_affinity={
                "backend": "sqlite",
                "sqlite_path": str(Path(tempfile.mkdtemp()) / "a.sqlite3"),
            },
        )
    )


@pytest.fixture(scope="module")
def single() -> Iterator[tuple[FakeState, str]]:
    state = FakeState(node="fake-staging")
    with (
        ServerThread(fake_gateway_app(state)) as gw,
        pool_server(_config([full_node(gw.url)])) as p,
    ):
        yield state, p.url


def _raw(url: str, head: bytes, body: bytes) -> tuple[int, dict[str, str]]:
    host, port = url.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=10) as s:
        s.sendall(head + body)
        data = b""
        while chunk := s.recv(65536):
            data += chunk
    lines = data.split(b"\r\n\r\n", 1)[0].split(b"\r\n")
    status = int(lines[0].split(b" ")[1])
    hdrs = {}
    for line in lines[1:]:
        k, _, v = line.partition(b":")
        hdrs[k.strip().lower().decode()] = v.strip().decode()
    return status, hdrs


def _chunked(b: bytes) -> bytes:
    return f"{len(b):x}\r\n".encode() + b + b"\r\n0\r\n\r\n"


def _pad(obj: dict[str, Any], size: int) -> bytes:
    base = json.dumps({**obj, "pad": ""}).encode()
    out = json.dumps({**obj, "pad": "x" * max(0, size - len(base))}).encode()
    assert len(out) == size
    return out


def _body(model: Any = ..., **extra: Any) -> bytes:
    obj: dict[str, Any] = {"prompt": "x", "texts": ["x"], **extra}
    if model is not ...:
        obj["model"] = model
    return json.dumps(obj).encode()


Sender = Callable[[str, str, bytes, dict[str, str]], tuple[int, str | None]]


def _cl(url: str, path: str, body: bytes, headers: dict[str, str]) -> tuple[int, str | None]:
    with httpx.Client(timeout=10, trust_env=False) as c:
        r = c.post(url + path, content=body, headers={**JSON, **headers})
    return r.status_code, r.headers.get("x-aigp-error")


def _stream(url: str, path: str, body: bytes, headers: dict[str, str]) -> tuple[int, str | None]:
    with httpx.Client(timeout=10, trust_env=False) as c:  # generator => chunked, no CL
        r = c.post(
            url + path,
            content=iter([body[: len(body) // 2], body[len(body) // 2 :]]),
            headers={**JSON, **headers},
        )
    return r.status_code, r.headers.get("x-aigp-error")


# (case id, body builder(known_model) -> bytes, extra headers, expected status, expected code)
CASES: list[tuple[str, Callable[[str], bytes], dict[str, str], int, str | None]] = [
    ("known", lambda k: _body(k), {}, 200, None),
    ("unknown", lambda k: _body(UNKNOWN), {}, 404, "MODEL_NOT_AVAILABLE"),
    (
        "limit-1 unknown",
        lambda k: _pad({"model": UNKNOWN}, LIMIT - 1),
        {},
        404,
        "MODEL_NOT_AVAILABLE",
    ),
    ("limit known", lambda k: _pad({"model": k}, LIMIT), {}, 200, None),
    (
        "limit+1 unknown",
        lambda k: _pad({"model": UNKNOWN}, LIMIT + 1),
        {},
        413,
        "REQUEST_BODY_TOO_LARGE",
    ),
    ("limit+1 known", lambda k: _pad({"model": k}, LIMIT + 1), {}, 413, "REQUEST_BODY_TOO_LARGE"),
    (
        "64x limit",
        lambda k: _pad({"model": UNKNOWN}, LIMIT * 64),
        {},
        413,
        "REQUEST_BODY_TOO_LARGE",
    ),
    ("model absent (implicit)", lambda k: _body(), {}, 200, None),
    ("model int", lambda k: _body(123), {}, 400, "INVALID_REQUEST_BODY"),
    ("model list", lambda k: _body([UNKNOWN]), {}, 400, "INVALID_REQUEST_BODY"),
    ("model null", lambda k: _body(None), {}, 400, "INVALID_REQUEST_BODY"),
    ("model object", lambda k: _body({"id": UNKNOWN}), {}, 400, "INVALID_REQUEST_BODY"),
    ("model empty", lambda k: _body(""), {}, 400, "INVALID_REQUEST_BODY"),
    ("model 201 chars", lambda k: _body("m" * 201), {}, 400, "INVALID_REQUEST_BODY"),
    ("model 200 chars unknown", lambda k: _body("m" * 200), {}, 404, "MODEL_NOT_AVAILABLE"),
    (
        "malformed",
        lambda k: b'{"model": "' + UNKNOWN.encode() + b'", "texts": [',
        {},
        400,
        "INVALID_REQUEST_BODY",
    ),
    ("not utf-8", lambda k: b'{"model": "\xff\xfe"}', {}, 400, "INVALID_REQUEST_BODY"),
    (
        "dup known->unknown",
        lambda k: b'{"model":"' + k.encode() + b'","model":"' + UNKNOWN.encode() + b'"}',
        {},
        400,
        "INVALID_REQUEST_BODY",
    ),
    (
        "dup unknown->known",
        lambda k: b'{"model":"' + UNKNOWN.encode() + b'","model":"' + k.encode() + b'"}',
        {},
        400,
        "INVALID_REQUEST_BODY",
    ),
    (
        "dup same value",
        lambda k: b'{"model":"' + k.encode() + b'","model":"' + k.encode() + b'"}',
        {},
        400,
        "INVALID_REQUEST_BODY",
    ),
    (
        "top-level array",
        lambda k: json.dumps([{"model": UNKNOWN}]).encode(),
        {},
        400,
        "INVALID_REQUEST_BODY",
    ),
    ("top-level string", lambda k: b'"' + UNKNOWN.encode() + b'"', {}, 400, "INVALID_REQUEST_BODY"),
    (
        "text/plain unknown",
        lambda k: _body(UNKNOWN),
        {"Content-Type": "text/plain"},
        404,
        "MODEL_NOT_AVAILABLE",
    ),
    (
        "octet-stream content-type unknown",
        lambda k: _body(UNKNOWN),
        {"Content-Type": "application/octet-stream"},
        404,
        "MODEL_NOT_AVAILABLE",
    ),
    (
        "utf-16 unknown",
        lambda k: json.dumps({"model": UNKNOWN}).encode("utf-16"),
        {},
        404,
        "MODEL_NOT_AVAILABLE",
    ),
    ("bom unknown", lambda k: b"\xef\xbb\xbf" + _body(UNKNOWN), {}, 404, "MODEL_NOT_AVAILABLE"),
    (
        "escaped key unknown",
        lambda k: b'{"\\u006dodel":"' + UNKNOWN.encode() + b'"}',
        {},
        404,
        "MODEL_NOT_AVAILABLE",
    ),
    (
        "gzip encoding",
        lambda k: gzip.compress(_body(UNKNOWN)),
        {"Content-Encoding": "gzip"},
        415,
        "UNSUPPORTED_CONTENT_ENCODING",
    ),
    (
        "gzip encoding known",
        lambda k: gzip.compress(_body(k)),
        {"Content-Encoding": "gzip"},
        415,
        "UNSUPPORTED_CONTENT_ENCODING",
    ),
    ("identity encoding known", lambda k: _body(k), {"Content-Encoding": "identity"}, 200, None),
    ("nested model ignored", lambda k: _body(k, options={"model": UNKNOWN}), {}, 200, None),
]


@pytest.mark.parametrize("framing", ["content-length", "chunked"])
@pytest.mark.parametrize("path", list(ROUTES))
@pytest.mark.parametrize(
    ("case", "build", "headers", "status", "code"), CASES, ids=[c[0] for c in CASES]
)
def test_model_governance_matrix(
    single: tuple[FakeState, str],
    framing: str,
    path: str,
    case: str,
    build: Callable[[str], bytes],
    headers: dict[str, str],
    status: int,
    code: str | None,
) -> None:
    state, url = single
    body = build(ROUTES[path])
    before = len(state.received)
    send: Sender = _cl if framing == "content-length" else _stream
    got_status, got_code = send(url, path, body, headers)
    relayed = state.received[before:]
    assert (got_status, got_code) == (status, code), case
    if status == 200:
        assert len(relayed) == 1
        # byte-for-byte relay, whatever the client framing
        assert relayed[0].body_sha256 == hashlib.sha256(body).hexdigest()
    else:
        assert relayed == [], f"{case}: refused request reached the backend"


@pytest.mark.parametrize("path", list(ROUTES))
def test_raw_chunked_unknown_model_refused(single: tuple[FakeState, str], path: str) -> None:
    state, url = single
    before = len(state.received)
    body = _body(UNKNOWN)
    status, hdrs = _raw(
        url,
        f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n".encode(),
        _chunked(body),
    )
    assert status == 404 and hdrs.get("x-aigp-error") == "MODEL_NOT_AVAILABLE"
    assert len(state.received) == before


def test_contradictory_or_duplicated_framing_refused(single: tuple[FakeState, str]) -> None:
    state, url = single
    before = len(state.received)
    body = _body(UNKNOWN)
    status, hdrs = _raw(
        url,
        b"POST /v1/embeddings HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
        _chunked(body),
    )
    assert status == 400 and hdrs.get("x-aigp-error") == "INVALID_REQUEST_FRAMING"
    status, _ = _raw(
        url,
        b"POST /v1/embeddings HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\nContent-Length: 5\r\n".encode()
        + b"Connection: close\r\n\r\n",
        body,
    )
    assert status == 400
    assert len(state.received) == before


def test_error_bodies_never_echo_the_request(single: tuple[FakeState, str]) -> None:
    _, url = single
    secret = "TOP-SECRET-PROMPT-" + "z" * 50
    with httpx.Client(timeout=10, trust_env=False) as c:
        for body in (
            b'{"model": 1, "prompt": "' + secret.encode() + b'"}',
            b'{"prompt": "' + secret.encode() + b'", "model": [',
            _body(UNKNOWN, prompt=secret),
        ):
            r = c.post(url + "/v1/generate", content=body, headers=JSON)
            assert r.status_code in (400, 404)
            assert set(r.json()) == {"error"} and r.headers["x-aigp-error"]
            assert secret not in r.text


def test_multipart_asr_upload_still_streamed_unchanged(single: tuple[FakeState, str]) -> None:
    """Multipart is not JSON: streamed as before (model inside a part)."""

    state, url = single
    files = {
        "request": (None, b'{"model":"asr-whisper-turbo"}', "application/json"),
        "file": ("a", b"\x00" * (LIMIT * 8), "application/octet-stream"),  # > JSON limit
    }
    with httpx.Client(timeout=10, trust_env=False) as c:
        r = c.post(url + "/v1/audio/transcriptions", files=files)
    assert r.status_code == 202
    assert state.received[-1].body_len > LIMIT * 8


# ------------------------------------------------------------------ multi-node


@pytest.mark.parametrize("policy", ["WEIGHTED_ROUND_ROBIN", "LEAST_INFLIGHT"])
def test_multinode_routes_on_the_inspected_model_whatever_the_framing(policy: str) -> None:
    """Same defaults, different model sets: `extra-embed` exists ONLY on node-b.
    Before the fix a chunked body alternated to node-a (which lacks the model)."""

    sa, sb = FakeState(node="node-a"), FakeState(node="node-b")
    with ServerThread(fake_gateway_app(sa)) as ga, ServerThread(fake_gateway_app(sb)) as gb:
        na = full_node(ga.url, "node-a")
        nb = full_node(gb.url, "node-b")
        nb["physical_host_id"] = "other-host"
        nb["models"] = [
            *nb["models"],
            {"model_id": "extra-embed", "capabilities": ["EMBEDDINGS"], "dimension": 768},
        ]
        with pool_server(_config([na, nb], policy=policy)) as pool:
            body = _body("extra-embed")
            for send in (_cl, _stream):
                for _ in range(4):
                    status, code = send(pool.url, "/v1/embeddings", body, {})
                    assert (status, code) == (200, None)
            assert _posts(sa) == 0 and _posts(sb) == 8
            for send in (_cl, _stream):
                assert send(pool.url, "/v1/embeddings", _body(UNKNOWN), {}) == (
                    404,
                    "MODEL_NOT_AVAILABLE",
                )
            assert _posts(sa) == 0 and _posts(sb) == 8


def _posts(state: FakeState) -> int:
    return sum(1 for r in state.received if r.method == "POST")


def test_multinode_digest_mismatch_refused_for_chunked_bodies() -> None:
    """Same model id, different artifacts on two nodes: refused even when chunked."""

    sa, sb = FakeState(node="node-a"), FakeState(node="node-b")
    sb.digest_overrides = {"legacy-768": "sha256:" + "b" * 64}
    with ServerThread(fake_gateway_app(sa)) as ga, ServerThread(fake_gateway_app(sb)) as gb:
        na, nb = full_node(ga.url, "node-a"), full_node(gb.url, "node-b")
        nb["physical_host_id"] = "other-host"
        with pool_server(_config([na, nb], policy="WEIGHTED_ROUND_ROBIN")) as pool:
            auth = {"Authorization": "Bearer " + _admin_token()}
            with httpx.Client(timeout=10, trust_env=False) as c:
                for nid in ("node-a", "node-b"):
                    c.post(f"{pool.url}/internal/nodes/{nid}/probe", headers=auth)
            for send in (_cl, _stream):
                status, code = send(pool.url, "/v1/embeddings", _body("legacy-768"), {})
                assert (status, code) == (404, "MODEL_NOT_AVAILABLE")
            assert _posts(sa) == 0 and _posts(sb) == 0


def _admin_token() -> str:
    from tests.conftest import ADMIN_TOKEN

    return ADMIN_TOKEN


# ------------------------------------------------------------------ memory budget


def test_inspection_budget_is_bounded_and_released() -> None:
    """Budget = LIMIT. A chunked body of LIMIT-10 bytes held mid-upload uses it:
    the next JSON request gets 503 INSPECTION_BUDGET_EXHAUSTED (never relayed
    uninspected); once the held request ends, the budget is released."""

    state = FakeState(node="fake-staging")
    with ServerThread(fake_gateway_app(state)) as gw:
        cfg = _config([full_node(gw.url)], json_inspect_budget_bytes=LIMIT)
        with pool_server(cfg) as pool:
            host, port = pool.url.removeprefix("http://").split(":")
            held = _pad({"model": "legacy-768"}, LIMIT - 10)
            s = socket.create_connection((host, int(port)), timeout=10)
            try:
                s.sendall(
                    b"POST /v1/embeddings HTTP/1.1\r\nHost: x\r\nContent-Type: application/json"
                    b"\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                    + f"{len(held):x}\r\n".encode()
                    + held
                    + b"\r\n"
                )  # body NOT terminated: the pool holds LIMIT-10 bytes of budget
                time.sleep(0.3)
                small = _body("legacy-768")
                assert _cl(pool.url, "/v1/embeddings", small, {}) == (
                    503,
                    "INSPECTION_BUDGET_EXHAUSTED",
                )
                s.sendall(b"0\r\n\r\n")  # finish the held request
                data = b""
                while chunk := s.recv(65536):
                    data += chunk
                assert data.startswith(b"HTTP/1.1 200")
            finally:
                s.close()
            assert _cl(pool.url, "/v1/embeddings", _body("legacy-768"), {}) == (200, None)


def test_budget_must_hold_one_maximum_body() -> None:
    with pytest.raises(ValueError, match="json_inspect_budget_bytes"):
        _config([full_node("http://127.0.0.1:1")], json_inspect_budget_bytes=LIMIT - 1)


def test_concurrent_small_requests_do_not_leak_budget(single: tuple[FakeState, str]) -> None:
    _, url = single
    errors: list[Any] = []

    def worker() -> None:
        for _ in range(10):
            got = _stream(url, "/v1/embeddings", _body("legacy-768"), {})
            if got != (200, None):
                errors.append(got)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    # every reservation was released: a near-limit body is still accepted
    assert _cl(url, "/v1/embeddings", _pad({"model": "legacy-768"}, LIMIT), {}) == (200, None)
