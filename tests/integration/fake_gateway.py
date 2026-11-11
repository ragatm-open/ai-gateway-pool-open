"""Fake AI Gateway nodes built on httpx.MockTransport (no socket, no GPU).

Scenarios (brief §20 'Tests d'intégration'): available, unavailable, timeout,
503, missing model, interrupted stream, invalid response, unknown ASR job.
Every request received is recorded so tests can assert what was (not) called.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from typing import Any

import httpx

from tests.conftest import gateway_fixture


class FakeGateway:
    def __init__(self, scenario: str = "available", environment: str = "staging") -> None:
        self.scenario = scenario
        self.environment = environment
        self.calls: list[tuple[str, str]] = []

    def runtime(self) -> dict[str, Any]:
        data = copy.deepcopy(gateway_fixture("health-runtime.staging.json"))
        data["environment"] = self.environment
        return data

    def models(self) -> dict[str, Any]:
        data = copy.deepcopy(gateway_fixture("v1-models.json"))
        if self.scenario == "model_missing":
            data["models"] = [
                m
                for m in data["models"]
                if m.get("model_logical") != "legacy-768"
                and m.get("name") != "nomic-embed-text:latest"
            ]
        return data

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        path, sc = request.url.path, self.scenario
        if sc == "unavailable":
            raise httpx.ConnectError("connection refused", request=request)
        if sc == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if sc == "http_503":
            return httpx.Response(
                503, json={"detail": {"code": "resource_saturated"}}, headers={"Retry-After": "5"}
            )
        if sc == "invalid_response" and path == "/health/runtime":
            return httpx.Response(200, content=b"<html>not json</html>")
        if path == "/health/runtime":
            return httpx.Response(200, json=self.runtime())
        if path == "/v1/models":
            if sc == "inventory_down":
                return httpx.Response(500)
            return httpx.Response(200, json=self.models())
        if path.startswith("/v1/audio/transcriptions/"):
            return httpx.Response(404, json={"detail": {"code": "job_not_found"}})
        if path == "/v1/generate/stream":
            return httpx.Response(
                200,
                headers={"Content-Type": "application/x-ndjson"},
                stream=NdjsonStream(interrupt=sc == "stream_interrupted"),
            )
        return httpx.Response(404)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


NDJSON_LINES = [
    b'{"type":"delta","delta":"Bon"}\n',
    b'{"type":"delta","delta":"jour"}\n',
    b"\n",  # keep-alive blank line, must be relayed untouched
    b'{"type":"done","model":"llama-chat","tokens":2}\n',
]


class NdjsonStream(httpx.AsyncByteStream):
    def __init__(self, interrupt: bool = False) -> None:
        self.interrupt = interrupt
        self.closed = False
        self.pulled = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for i, line in enumerate(NDJSON_LINES):
            if self.interrupt and i == 2:
                raise httpx.ReadError("upstream reset")
            self.pulled += 1
            yield line

    async def aclose(self) -> None:
        self.closed = True
