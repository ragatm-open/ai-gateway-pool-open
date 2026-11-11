"""Real HTTP servers on 127.0.0.1 ephemeral ports for proxy tests.

* :func:`fake_gateway_app` — an AI Gateway stand-in implementing the routes of
  the Open contract snapshot PLUS the two historical routes excluded from the Open
  edition (to prove the pool never dispatches them), recording what it receives, consuming bodies
  progressively, and able to simulate errors, delays, streams and interruptions
  through ``X-Fake-*`` request headers (relayed end-to-end by the pool).
* :class:`ServerThread` — runs any ASGI app under uvicorn in a background thread.

No GPU, no real Gateway: these are mocks; passing tests are MOCK_QUALIFIED only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from tests.conftest import gateway_fixture


@dataclass
class Received:
    method: str
    path: str
    raw_path: str
    query: str
    headers: list[tuple[str, str]]
    body_len: int
    body_sha256: str
    chunks: int
    max_chunk: int
    first_byte_at: float | None
    complete: bool


@dataclass
class FakeState:
    node: str
    environment: str = "staging"
    received: list[Received] = field(default_factory=list)
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    job_seq: int = 0
    stream_closed: list[str] = field(default_factory=list)
    read_delay: float = 0.0  # seconds per received chunk (slow consumer)
    # POOL-3 per-node variations
    digest_overrides: dict[str, str] = field(default_factory=dict)  # model id -> digest
    llm_caps_overrides: dict[str, list[str]] = field(default_factory=dict)
    drop_models: set[str] = field(default_factory=set)  # removed from /v1/models


async def _consume(request: Request, state: FakeState) -> tuple[Received, bytes]:
    h = hashlib.sha256()
    size = chunks = max_chunk = 0
    first: float | None = None
    head = b""
    complete = True
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            if first is None:
                first = time.perf_counter()
            chunks += 1
            size += len(chunk)
            max_chunk = max(max_chunk, len(chunk))
            h.update(chunk)
            if len(head) < 65536:
                head += chunk[: 65536 - len(head)]
            if state.read_delay:
                await asyncio.sleep(state.read_delay)
    except Exception:
        complete = False
    rec = Received(
        method=request.method,
        path=request.url.path,
        raw_path=request.scope.get("raw_path", b"").decode("latin-1"),
        query=request.scope.get("query_string", b"").decode("latin-1"),
        headers=[(k.decode("latin-1"), v.decode("latin-1")) for k, v in request.scope["headers"]],
        body_len=size,
        body_sha256=h.hexdigest(),
        chunks=chunks,
        max_chunk=max_chunk,
        first_byte_at=first,
        complete=complete,
    )
    state.received.append(rec)
    return rec, head


def _hdr(request: Request, name: str) -> str | None:
    return request.headers.get(name)


def fake_gateway_app(state: FakeState) -> Starlette:
    node_headers = {"X-Fake-Node": state.node}

    def scripted(request: Request) -> Response | None:
        status = _hdr(request, "x-fake-status")
        if status is None:
            return None
        code = _hdr(request, "x-fake-code") or "fake_error"
        headers = dict(node_headers)
        if ra := _hdr(request, "x-fake-retry-after"):
            headers["Retry-After"] = ra
        return JSONResponse(
            {"detail": {"status": "error", "code": code, "provider": "ai_gateway"}},
            status_code=int(status),
            headers=headers,
        )

    async def echo(request: Request) -> Response:
        rec, _ = await _consume(request, state)
        if delay := _hdr(request, "x-fake-delay"):
            await asyncio.sleep(float(delay))
        if (resp := scripted(request)) is not None:
            return resp
        if size := _hdr(request, "x-fake-response-bytes"):
            return await big_response(int(size))
        body = {
            "echo": {
                "method": rec.method,
                "path": rec.path,
                "query": rec.query,
                "body_len": rec.body_len,
                "body_sha256": rec.body_sha256,
                "content_type": request.headers.get("content-type"),
            },
            "node": state.node,
        }
        return JSONResponse(body, headers=node_headers)

    async def big_response(size: int) -> Response:
        block = b"x" * 65536

        async def gen() -> AsyncIterator[bytes]:
            sent = 0
            while sent < size:
                n = min(len(block), size - sent)
                sent += n
                yield block[:n]

        return StreamingResponse(
            gen(),
            media_type="application/octet-stream",
            headers={**node_headers, "Content-Length": str(size)},
        )

    async def runtime(request: Request) -> Response:
        await _consume(request, state)
        if (resp := scripted(request)) is not None:
            return resp
        data = gateway_fixture("health-runtime.staging.json")
        data["environment"] = state.environment
        for m in data["llm_models"]:
            if m["model_logical"] in state.llm_caps_overrides:
                m["capabilities"] = state.llm_caps_overrides[m["model_logical"]]
        return JSONResponse(data, headers=node_headers)

    async def models(request: Request) -> Response:
        await _consume(request, state)
        if (resp := scripted(request)) is not None:
            return resp
        data = gateway_fixture("v1-models.json")
        data["models"] = [
            m
            for m in data["models"]
            if m.get("name") not in state.drop_models
            and m.get("model_logical") not in state.drop_models
        ]
        for m in data["models"]:
            for key in ("name", "model_logical"):
                if m.get(key) in state.digest_overrides:
                    m["digest"] = state.digest_overrides[m[key]]
        return JSONResponse(data, headers=node_headers)

    async def ndjson(request: Request) -> Response:
        await _consume(request, state)
        if (resp := scripted(request)) is not None:
            return resp
        n = int(_hdr(request, "x-fake-lines") or 5)
        first_gap = float(_hdr(request, "x-fake-first-gap") or 0)
        gap = float(_hdr(request, "x-fake-chunk-delay") or 0)
        interrupt = _hdr(request, "x-fake-interrupt-after")
        tag = _hdr(request, "x-fake-tag") or "stream"

        async def gen() -> AsyncIterator[bytes]:
            try:
                for i in range(n):
                    if interrupt is not None and i == int(interrupt):
                        raise RuntimeError("simulated upstream crash")
                    yield (json.dumps({"type": "delta", "delta": f"t{i}"}) + "\n").encode()
                    await asyncio.sleep(first_gap if i == 0 else gap)
                yield b'{"type":"done","model":"llama-chat","tokens":' + str(n).encode() + b"}\n"
            finally:
                state.stream_closed.append(tag)

        return StreamingResponse(gen(), media_type="application/x-ndjson", headers=node_headers)

    async def asr_submit(request: Request) -> Response:
        rec, head = await _consume(request, state)
        if delay := _hdr(request, "x-fake-delay"):
            await asyncio.sleep(float(delay))
        if (resp := scripted(request)) is not None:
            return resp
        ctype = request.headers.get("content-type", "")
        i_req, i_file = head.find(b'name="request"'), head.find(b'name="file"')
        if not ctype.startswith("multipart/form-data") or not (0 <= i_req < i_file):
            return JSONResponse(
                {"detail": "multipart/form-data attendu (parties `request` puis `file`)"},
                status_code=400,
                headers=node_headers,
            )
        if "content-length" in request.headers and "transfer-encoding" in request.headers:
            return JSONResponse({"detail": "Invalid HTTP request received."}, status_code=400)
        state.job_seq += 1
        job_id = _hdr(request, "x-fake-job-id") or f"job-{state.job_seq}"
        state.jobs[job_id] = {"bytes": rec.body_len, "sha256": rec.body_sha256}
        return JSONResponse(
            {"job_id": job_id, "status": "queued"}, status_code=202, headers=node_headers
        )

    async def asr_job(request: Request) -> Response:
        await _consume(request, state)
        if (resp := scripted(request)) is not None:
            return resp
        job_id = request.path_params["job_id"]
        if job_id not in state.jobs:
            return JSONResponse(
                {"detail": {"code": "job_not_found"}}, status_code=404, headers=node_headers
            )
        if request.method == "DELETE":
            del state.jobs[job_id]
            return JSONResponse({"job_id": job_id, "deleted": True}, headers=node_headers)
        return JSONResponse(
            {
                "transcription_id": job_id,
                "job_id": job_id,
                "text": "synthetic transcript",
                "segments": [],
                "query": request.scope["query_string"].decode(),
            },
            headers=node_headers,
        )

    routes = [
        Route("/health", echo, methods=["GET"]),
        Route("/health/runtime", runtime, methods=["GET"]),
        Route("/health/ready", echo, methods=["GET"]),
        Route("/health/embeddings", echo, methods=["GET"]),
        Route("/health/rerank", echo, methods=["GET"]),
        Route("/health/vision", echo, methods=["GET"]),
        Route("/health/asr", echo, methods=["GET"]),
        Route("/health/structured-extract", echo, methods=["GET"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/v1/vision/capacity", echo, methods=["GET"]),
        Route("/v1/generate", echo, methods=["POST"]),
        Route("/v1/generate/stream", ndjson, methods=["POST"]),
        Route("/v1/chat", echo, methods=["POST"]),
        Route("/v1/chat/stream", ndjson, methods=["POST"]),
        Route("/v1/embeddings", echo, methods=["POST"]),
        Route("/v1/rerank", echo, methods=["POST"]),
        Route("/v1/vision/analyze", echo, methods=["POST"]),
        Route("/v1/vision/structured-extract", echo, methods=["POST"]),
        Route("/v1/audio/transcriptions", asr_submit, methods=["POST"]),
        Route("/v1/audio/transcriptions/{job_id}", asr_job, methods=["GET", "DELETE"]),
    ]
    return Starlette(routes=routes)


class DropServer:
    """TCP server that reads the request (headers + up to 64 KiB of body), records
    it, then closes the connection WITHOUT answering: "request reached the node,
    outcome unknown" (the case where a POST must never be replayed)."""

    def __init__(self) -> None:
        self.port = free_port()
        self.requests = 0
        self._stop = threading.Event()
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", self.port))
        self._sock.listen(16)
        self._sock.settimeout(0.2)
        self._t = threading.Thread(target=self._run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            with conn:
                conn.settimeout(2)
                data = b""
                try:
                    while b"\r\n\r\n" not in data:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        data += chunk
                    conn.recv(65536)
                except OSError:
                    pass
                if data:
                    self.requests += 1

    def __enter__(self) -> DropServer:
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._t.join()
        self._sock.close()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ServerThread:
    """uvicorn in a daemon thread; ``url`` is ready once the context is entered."""

    def __init__(self, app: Any, port: int | None = None) -> None:
        self.port = port or free_port()
        self.config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            access_log=False,
            lifespan="on",
            timeout_keep_alive=5,
        )
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> ServerThread:
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline or not self.thread.is_alive():
                raise RuntimeError("server did not start")
            time.sleep(0.01)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


if __name__ == "__main__":  # pragma: no cover - used by scripts/bench_proxy.py
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--node", default="fake-staging")
    parser.add_argument("--environment", default="staging")
    a = parser.parse_args()
    uvicorn.run(
        fake_gateway_app(FakeState(node=a.node, environment=a.environment)),
        host="127.0.0.1",
        port=a.port,
        log_level="warning",
        access_log=False,
    )
