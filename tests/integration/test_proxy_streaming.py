"""Request and response streaming through the real pool server."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tests.integration.live_servers import FakeState, ServerThread

Fake = tuple[FakeState, ServerThread]
MiB = 1024 * 1024


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("no VmRSS")


class RssSampler:
    """Peak RSS of THIS process (client + pool + fake all live here) while running."""

    def __init__(self) -> None:
        self.base = _rss_bytes()
        self.peak = self.base
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, _rss_bytes())
            time.sleep(0.005)

    def __enter__(self) -> RssSampler:
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._t.join()

    @property
    def growth(self) -> int:
        return self.peak - self.base


# ------------------------------------------------------------------ responses


def test_first_chunk_delivered_early_and_order_preserved(
    fake: Fake, pool: str, http: httpx.Client
) -> None:
    headers = {"X-Fake-Lines": "4", "X-Fake-First-Gap": "1.0"}
    t0 = time.perf_counter()
    with http.stream(
        "POST", pool + "/v1/generate/stream", json={"prompt": "x"}, headers=headers
    ) as r:
        lines_iter = r.iter_lines()
        first = next(lines_iter)
        t_first = time.perf_counter() - t0
        rest = list(lines_iter)
    total = time.perf_counter() - t0
    assert json.loads(first) == {"type": "delta", "delta": "t0"}
    assert t_first < 0.5, t_first  # generation still running upstream (1 s gap)
    assert total >= 1.0
    events = [json.loads(x) for x in [first, *rest] if x]
    assert [e.get("delta") for e in events[:-1]] == ["t0", "t1", "t2", "t3"]  # order kept
    assert events[-1]["type"] == "done"  # stream end detected, nothing invented after it
    assert len(events) == 5


def test_chat_stream_relayed_unchanged(fake: Fake, pool: str, http: httpx.Client) -> None:
    direct = http.post(fake[1].url + "/v1/chat/stream", json={"model": "llama-chat"}).content
    proxied = http.post(pool + "/v1/chat/stream", json={"model": "llama-chat"}).content
    assert proxied == direct  # byte-identical NDJSON, not aggregated


def test_upstream_disconnect_mid_stream(fake: Fake, pool: str, http: httpx.Client) -> None:
    received: list[str] = []
    with (
        pytest.raises(httpx.HTTPError),  # downstream aborted, not cleanly finished
        http.stream(
            "POST",
            pool + "/v1/generate/stream",
            json={},
            headers={"X-Fake-Interrupt-After": "2", "X-Fake-Lines": "5"},
        ) as r,
    ):
        for line in r.iter_lines():
            received.append(line)
    assert all('"done"' not in line for line in received)  # no fabricated end of stream
    assert any("t1" in line for line in received)


def test_client_disconnect_closes_upstream_stream(
    fake: Fake, pool: str, http: httpx.Client
) -> None:
    state = fake[0]
    headers = {"X-Fake-Lines": "200", "X-Fake-Chunk-Delay": "0.05", "X-Fake-Tag": "cd"}
    with http.stream("POST", pool + "/v1/generate/stream", json={}, headers=headers) as r:
        it = r.iter_lines()
        next(it)
        next(it)
    # Client went away after 2 lines; the upstream generator must be stopped
    # long before its natural end (200 * 50 ms = 10 s).
    deadline = time.time() + 5
    while "cd" not in state.stream_closed and time.time() < deadline:
        time.sleep(0.05)
    assert "cd" in state.stream_closed


def test_no_full_response_buffering(fake: Fake, pool: str, http: httpx.Client) -> None:
    size = 256 * MiB
    with (
        RssSampler() as rss,
        http.stream(
            "GET", pool + "/health/embeddings", headers={"X-Fake-Response-Bytes": str(size)}
        ) as r,
    ):
        total = sum(len(c) for c in r.iter_raw())
    assert total == size
    assert rss.growth < 96 * MiB, rss.growth / MiB


# ------------------------------------------------------------------ requests


def _multipart(total: int, chunk: int = MiB) -> tuple[Iterator[bytes], str, int, str]:
    boundary = "aigpBoundary7MA4YWxkTrZu0gW"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="request"\r\n'
        f"Content-Type: application/json\r\n\r\n"
        f'{{"model":"asr-whisper-turbo","mode":"async"}}\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    length = len(head) + total + len(tail)
    h = hashlib.sha256()
    block = bytes(range(256)) * (chunk // 256)

    def gen() -> Iterator[bytes]:
        h.update(head)
        yield head
        sent = 0
        while sent < total:
            n = min(chunk, total - sent)
            h.update(block[:n])
            yield block[:n]
            sent += n
        h.update(tail)
        yield tail

    gen_iter = gen()
    return gen_iter, f"multipart/form-data; boundary={boundary}", length, h  # type: ignore[return-value]


@pytest.mark.parametrize("size_mib", [1, 300])
def test_large_multipart_streamed_with_bounded_memory(
    fake: Fake, pool: str, http: httpx.Client, size_mib: int
) -> None:
    body, ctype, length, digest = _multipart(size_mib * MiB)
    t_sent: list[float] = []

    def tracked() -> Iterator[bytes]:
        yield from body
        t_sent.append(time.perf_counter())

    with RssSampler() as rss:
        r = http.post(
            pool + "/v1/audio/transcriptions",
            content=tracked(),
            headers={"Content-Type": ctype, "Content-Length": str(length)},
        )
    assert r.status_code == 202, r.text
    rec = [x for x in fake[0].received if x.path == "/v1/audio/transcriptions"][-1]
    assert rec.complete and rec.body_len == length
    assert rec.body_sha256 == digest.hexdigest()  # type: ignore[attr-defined]
    # Content-Length kept, never together with Transfer-Encoding (Gateway contract).
    names = {k.lower() for k, _ in rec.headers}
    assert "content-length" in names and "transfer-encoding" not in names
    assert rec.max_chunk <= 4 * MiB
    if size_mib >= 300:
        # Streaming, not store-and-forward: the upstream got bytes long before the
        # client finished (small bodies fit in socket buffers, hence the size guard).
        assert rec.first_byte_at is not None and rec.first_byte_at < t_sent[0]
        assert rss.growth < 96 * MiB, rss.growth / MiB


def test_chunked_request_forwarded_chunked(fake: Fake, pool: str, http: httpx.Client) -> None:
    body, ctype, length, _ = _multipart(2 * MiB)
    r = http.post(
        pool + "/v1/audio/transcriptions", content=body, headers={"Content-Type": ctype}
    )  # no Content-Length -> chunked
    assert r.status_code == 202
    rec = fake[0].received[-1]
    names = {k.lower() for k, _ in rec.headers}
    assert "transfer-encoding" in names and "content-length" not in names
    assert rec.body_len == length


def test_backpressure_slow_upstream_bounded_memory(
    fake: Fake, pool: str, http: httpx.Client
) -> None:
    fake[0].read_delay = 0.002  # slow consumer: ~ tens of MB/s
    body, ctype, length, _ = _multipart(96 * MiB)
    with RssSampler() as rss:
        r = http.post(
            pool + "/v1/audio/transcriptions",
            content=body,
            headers={"Content-Type": ctype, "Content-Length": str(length)},
        )
    fake[0].read_delay = 0
    assert r.status_code == 202
    # Without backpressure the pool would accumulate most of the 96 MiB.
    assert rss.growth < 64 * MiB, rss.growth / MiB


def test_incomplete_request_body_not_forwarded_as_complete(fake: Fake, pool: str) -> None:
    import socket

    host, port = pool.removeprefix("http://").split(":")
    before = len(fake[0].received)
    with socket.create_connection((host, int(port))) as s:
        s.sendall(
            b"POST /v1/embeddings HTTP/1.1\r\nHost: x\r\nContent-Type: application/json"
            b'\r\nContent-Length: 1000\r\n\r\n{"texts":['
        )
        time.sleep(0.3)
    time.sleep(0.5)
    complete = [x for x in fake[0].received[before:] if x.complete and x.body_len == 1000]
    assert complete == []
