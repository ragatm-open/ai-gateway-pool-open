"""Several REAL pool processes sharing one REAL Redis (ASR affinity end to end).

Each pool runs as its own OS process (the ``ai-gateway-pool`` entrypoint); the
fake AI Gateway nodes run in this test process and record every call.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from tests.conftest import ADMIN_TOKEN, raw_config
from tests.integration.conftest import full_node
from tests.integration.live_servers import FakeState, ServerThread, fake_gateway_app, free_port
from tests.redis.conftest import REDIS_URL, pause_redis, requires_container, requires_redis

AUTH = {"Authorization": f"Bearer {ADMIN_TOKEN}"}  # X-AIGP-Target-Node is privileged

pytestmark = [pytest.mark.redis_real, requires_redis]
ENTRY = str(Path(sys.executable).parent / "ai-gateway-pool")
FILES = {
    "request": (None, b'{"model":"asr-whisper-turbo","mode":"async"}', "application/json"),
    "file": ("audio", b"synthetic", "application/octet-stream"),
}


class PoolProcess:
    def __init__(self, cfg_path: Path, env_name: str) -> None:
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        env = {
            **os.environ,
            "POOL_ENVIRONMENT": env_name,
            "POOL_CONFIG_FILE": str(cfg_path),
            "POOL_BIND_PORT": str(self.port),
            "POOL_LOG_LEVEL": "WARNING",
            "POOL_ADMIN_TOKEN": ADMIN_TOKEN,
            "POOL_TEST_REDIS_URL": REDIS_URL or "",
        }
        self.proc = subprocess.Popen(
            [ENTRY], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                httpx.get(self.url + "/health/live", timeout=0.5)
                return
            except httpx.HTTPError:
                time.sleep(0.05)
        self.stop()
        raise RuntimeError("pool process did not start")

    def stop(self) -> None:
        self.proc.terminate()
        self.proc.wait(10)


def write_config(tmp: Path, name: str, env: str, nodes: list[dict[str, Any]], prefix: str) -> Path:
    data = raw_config(
        env=env,
        nodes=nodes,
        routing={"policy": "WEIGHTED_ROUND_ROBIN"},
        job_affinity={
            "backend": "redis",
            "redis_url_env": "POOL_TEST_REDIS_URL",
            "redis_key_prefix": prefix,
        },
    )
    path = tmp / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def cluster(prefix: str, tmp_path: Path) -> Iterator[dict[str, Any]]:
    with ExitStack() as stack:
        states = {n: FakeState(node=n) for n in ("a", "b")}
        servers = {
            n: stack.enter_context(ServerThread(fake_gateway_app(s))) for n, s in states.items()
        }
        nodes = [full_node(servers[n].url, n) for n in ("a", "b")]
        cfg = write_config(tmp_path, "stg", "staging", nodes, prefix)
        procs: list[PoolProcess] = []

        def spawn(path: Path = cfg, env: str = "staging") -> PoolProcess:
            p = PoolProcess(path, env)
            procs.append(p)
            return p

        yield {
            "states": states,
            "servers": servers,
            "spawn": spawn,
            "nodes": nodes,
            "tmp": tmp_path,
            "prefix": prefix,
        }
        for p in procs:
            if p.proc.poll() is None:
                p.stop()


def calls(state: FakeState, method: str) -> int:
    return sum(1 for r in state.received if r.method == method)


def test_job_created_on_one_process_resolved_by_another(cluster: dict[str, Any]) -> None:
    p1, p2 = cluster["spawn"](), cluster["spawn"]()
    with httpx.Client(timeout=30) as http:
        created = http.post(p1.url + "/v1/audio/transcriptions", files=FILES)
        assert created.status_code == 202
        job, owner = created.json()["job_id"], created.headers["x-fake-node"]
        other = "b" if owner == "a" else "a"
        for _ in range(3):
            g = http.get(f"{p2.url}/v1/audio/transcriptions/{job}")
            assert g.status_code == 200 and g.headers["x-fake-node"] == owner
        d = http.delete(f"{p2.url}/v1/audio/transcriptions/{job}")
        assert d.status_code == 200 and d.headers["x-fake-node"] == owner
        assert http.get(f"{p1.url}/v1/audio/transcriptions/{job}").status_code == 404
    assert calls(cluster["states"][other], "GET") == calls(cluster["states"][other], "DELETE") == 0


def test_affinity_survives_process_restart(cluster: dict[str, Any]) -> None:
    p1 = cluster["spawn"]()
    with httpx.Client(timeout=30) as http:
        created = http.post(p1.url + "/v1/audio/transcriptions", files=FILES)
        job, owner = created.json()["job_id"], created.headers["x-fake-node"]
        p1.stop()  # router process gone
        p3 = cluster["spawn"]()  # brand new process, same Redis
        g = http.get(f"{p3.url}/v1/audio/transcriptions/{job}")
        assert g.status_code == 200 and g.headers["x-fake-node"] == owner


def test_concurrent_native_id_collision_across_processes(cluster: dict[str, Any]) -> None:
    p1, p2 = cluster["spawn"](), cluster["spawn"]()
    hdr = {"X-Fake-Job-Id": "dup-native"}

    def submit(args: tuple[str, str]) -> httpx.Response:
        url, node = args
        with httpx.Client(timeout=30) as c:
            return c.post(
                url + "/v1/audio/transcriptions",
                files=FILES,
                headers={**hdr, "X-AIGP-Target-Node": node, **AUTH},
            )

    with ThreadPoolExecutor(2) as ex:
        r1, r2 = ex.map(submit, [(p1.url, "a"), (p2.url, "b")])
    statuses = sorted([r1.status_code, r2.status_code])
    assert statuses == [202, 409]
    winner = r1 if r1.status_code == 202 else r2
    loser = r2 if winner is r1 else r1
    owner = winner.headers["x-fake-node"]
    loser_node = "b" if owner == "a" else "a"
    assert loser.json()["error"]["compensation"] == "deleted"
    assert "dup-native" not in cluster["states"][loser_node].jobs  # compensated
    assert "dup-native" in cluster["states"][owner].jobs
    with httpx.Client(timeout=30) as http:
        for p in (p1, p2):
            g = http.get(f"{p.url}/v1/audio/transcriptions/dup-native")
            assert g.status_code == 200 and g.headers["x-fake-node"] == owner


def test_staging_job_not_resolvable_by_production_pool(cluster: dict[str, Any]) -> None:
    stg = cluster["spawn"]()
    prod_state = FakeState(node="prod", environment="production")
    with ServerThread(fake_gateway_app(prod_state)) as prod_srv, httpx.Client(timeout=30) as http:
        job = http.post(stg.url + "/v1/audio/transcriptions", files=FILES).json()["job_id"]
        cfg = write_config(
            cluster["tmp"],
            "prd",
            "production",
            [full_node(prod_srv.url, "prod", env="production")],
            cluster["prefix"],
        )
        prd = cluster["spawn"](cfg, "production")
        r = http.get(f"{prd.url}/v1/audio/transcriptions/{job}")
        assert r.status_code == 404 and r.json()["error"]["code"] == "JOB_AFFINITY_UNKNOWN"
    assert prod_state.received == []


@requires_container
def test_redis_outage_fail_closed_then_recovery(cluster: dict[str, Any]) -> None:
    p1 = cluster["spawn"]()
    with httpx.Client(timeout=30) as http:
        job = http.post(p1.url + "/v1/audio/transcriptions", files=FILES).json()["job_id"]
        before = {n: calls(s, "POST") for n, s in cluster["states"].items()}
        with pause_redis():
            ready = http.get(p1.url + "/pool/health/ready")
            assert ready.status_code == 503
            assert ready.json()["job_affinity_store"] == "unavailable"
            sub = http.post(p1.url + "/v1/audio/transcriptions", files=FILES)
            assert sub.status_code == 503
            assert sub.json()["error"]["code"] == "AFFINITY_STORE_UNAVAILABLE"
            g = http.get(f"{p1.url}/v1/audio/transcriptions/{job}")
            assert g.status_code == 503
        # No job was created while the store was down (no orphan, no SQLite fallback).
        assert {n: calls(s, "POST") for n, s in cluster["states"].items()} == before
        deadline = time.time() + 10
        while http.get(p1.url + "/pool/health/ready").status_code != 200:
            assert time.time() < deadline
            time.sleep(0.25)
        g = http.get(f"{p1.url}/v1/audio/transcriptions/{job}")
        assert g.status_code == 200  # mapping persisted through the outage


def test_redis_required_but_unset_refuses_to_start(tmp_path: Path, prefix: str) -> None:
    data = raw_config(
        job_affinity={
            "backend": "redis",
            "redis_url_env": "POOL_UNSET_REDIS_URL",
            "redis_key_prefix": prefix,
        }
    )
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump(data))
    env = {k: v for k, v in os.environ.items() if k != "POOL_UNSET_REDIS_URL"}
    env |= {
        "POOL_ENVIRONMENT": "staging",
        "POOL_CONFIG_FILE": str(cfg),
        "POOL_BIND_PORT": str(free_port()),
    }
    proc = subprocess.run([ENTRY], env=env, capture_output=True, timeout=20)
    assert proc.returncode != 0
    assert b"POOL_UNSET_REDIS_URL is not set" in proc.stderr + proc.stdout
