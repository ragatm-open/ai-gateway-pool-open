"""Streaming relay primitives (prepared for POOL-2, not wired to any route yet).

Rules these primitives encode:

* bytes are relayed **verbatim** — NDJSON lines are never parsed, merged,
  re-serialized or aggregated into one JSON document;
* nothing is buffered beyond one chunk — request bodies (large multipart audio)
  and response bodies are iterated, never ``read()`` in full;
* backpressure is natural: the next upstream chunk is only pulled once the
  downstream consumer asked for it (async generator semantics);
* a client disconnect (``GeneratorExit`` / cancellation) closes the upstream
  response, which closes the upstream connection.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from enum import StrEnum

import httpx


class StreamKind(StrEnum):
    JSON = "json"  # single JSON document, still relayed as bytes
    NDJSON = "ndjson"  # application/x-ndjson, one JSON object per line
    MULTIPART_UPLOAD = "multipart_upload"  # request-side streaming


class UpstreamStreamError(Exception):
    """The upstream stream broke mid-way. Already-sent bytes cannot be recalled:
    the relay must close the downstream stream (no fabricated terminal event)."""


async def relay_response_body(response: httpx.Response) -> AsyncIterator[bytes]:
    """Yield the upstream body chunk by chunk, unchanged, then close it."""

    try:
        async for chunk in response.aiter_raw():
            if chunk:
                yield chunk
    except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout) as exc:
        raise UpstreamStreamError(type(exc).__name__) from exc
    finally:
        await response.aclose()


async def relay_request_body(source: AsyncIterable[bytes]) -> AsyncIterator[bytes]:
    """Pass-through of an incoming request body to httpx ``content=`` (no buffering)."""

    async for chunk in source:
        if chunk:
            yield chunk
