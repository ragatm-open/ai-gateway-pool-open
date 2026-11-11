"""AIGP-HARDEN-1 / AIGP-NEW-01, end to end over real sockets (F4 reproduction).

Since JSON bodies are now fully inspected before dispatch, the remaining way a
client can abort a HALF_OPEN trial mid-request is a streamed (multipart ASR)
upload that it cuts. Before the fix this left the breaker stuck HALF_OPEN with
its trial "in flight" for good (503 CIRCUIT_OPEN until a restart).
One real 1.1 s wait (minimum open period is 1 s).
"""

from __future__ import annotations

import socket
import tempfile
import time
from pathlib import Path

import httpx

from ai_gateway_pool.config.schema import PoolConfig
from tests.conftest import raw_config
from tests.integration.conftest import full_node, pool_server
from tests.integration.live_servers import FakeState, ServerThread, fake_gateway_app

EMB = b'{"model": "legacy-768", "texts": ["x"]}'
H = {"Content-Type": "application/json"}


def test_cut_streamed_upload_as_half_open_trial_does_not_wedge_the_node() -> None:
    state = FakeState(node="fake-staging")
    with ServerThread(fake_gateway_app(state)) as gw:
        data = raw_config(
            nodes=[full_node(gw.url)],
            job_affinity={
                "backend": "sqlite",
                "sqlite_path": str(Path(tempfile.mkdtemp()) / "a.sqlite3"),
            },
            circuit_breaker={"failure_threshold": 1, "open_seconds": 1, "max_open_seconds": 1},
        )
        data["timeouts"] = {"inference": {"connect": 2, "read": 0.5, "write": 5, "pool": 2}}
        with (
            pool_server(PoolConfig.model_validate(data)) as pool,
            httpx.Client(timeout=10, trust_env=False) as http,
        ):
            r = http.post(
                pool.url + "/v1/embeddings", content=EMB, headers={**H, "X-Fake-Delay": "2"}
            )
            assert r.status_code == 504  # response timeout = transport failure -> OPEN
            r = http.post(pool.url + "/v1/embeddings", content=EMB, headers=H)
            assert r.status_code == 503 and "CIRCUIT_OPEN" in r.json()["error"]["message"]
            time.sleep(1.1)  # HALF_OPEN: the next dispatched request is the trial

            host, port = pool.url.removeprefix("http://").split(":")
            boundary = "b0undary"
            head = (
                f'--{boundary}\r\nContent-Disposition: form-data; name="request"\r\n'
                'Content-Type: application/json\r\n\r\n{"model":"asr-whisper-turbo"}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                'filename="a"\r\nContent-Type: application/octet-stream\r\n\r\n'
            ).encode()
            with socket.create_connection((host, int(port)), timeout=5) as s:
                s.sendall(
                    b"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: x\r\n"
                    + f"Content-Type: multipart/form-data; boundary={boundary}\r\n".encode()
                    + b"Content-Length: 100000\r\n\r\n"
                    + head
                    + b"\x00" * 1000
                )
                time.sleep(0.3)  # the trial is dispatched, then the client vanishes
            time.sleep(0.3)

            r = http.post(pool.url + "/v1/embeddings", content=EMB, headers=H)
            assert r.status_code == 200, r.text  # was: 503 CIRCUIT_OPEN for ever
