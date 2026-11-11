"""ASR job lifecycle and durable job -> node affinity through the real pool."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from tests.integration.conftest import pool_config, pool_server
from tests.integration.live_servers import FakeState, ServerThread, fake_gateway_app

Fake = tuple[FakeState, ServerThread]
FILES = {
    "request": (None, b'{"model":"asr-whisper-turbo","mode":"async"}', "application/json"),
    "file": ("audio", b"\x00\x01synthetic-audio", "application/octet-stream"),
}


def submit(http: httpx.Client, pool: str, **headers: str) -> httpx.Response:
    return http.post(pool + "/v1/audio/transcriptions", files=FILES, headers=headers)


def test_asr_post_get_delete_same_node(fake: Fake, pool: str, http: httpx.Client) -> None:
    r = submit(http, pool)
    assert r.status_code == 202
    job = r.json()["job_id"]
    assert r.json()["status"] == "queued"  # Gateway body and job id unchanged
    g = http.get(f"{pool}/v1/audio/transcriptions/{job}?format=json&wait_ms=0")
    assert g.status_code == 200 and g.json()["transcription_id"] == job
    assert g.headers["x-fake-node"] == "fake-staging"
    d = http.delete(f"{pool}/v1/audio/transcriptions/{job}")
    assert d.status_code == 200 and d.json()["deleted"] is True
    # Mapping forgotten after a confirmed delete.
    again = http.get(f"{pool}/v1/audio/transcriptions/{job}")
    assert again.status_code == 404
    assert again.json()["error"]["code"] == "JOB_AFFINITY_UNKNOWN"


def test_asr_affinity_after_router_restart(fake: Fake, tmp_path: Path, http: httpx.Client) -> None:
    cfg = pool_config(fake[1].url, tmp_path)
    with pool_server(cfg) as first:
        job = submit(http, first.url).json()["job_id"]
    # New process-equivalent: brand new app instance + server, same durable file.
    cfg2 = pool_config(fake[1].url, tmp_path)
    with pool_server(cfg2) as second:
        g = http.get(f"{second.url}/v1/audio/transcriptions/{job}")
        assert g.status_code == 200 and g.json()["job_id"] == job
        assert http.delete(f"{second.url}/v1/audio/transcriptions/{job}").status_code == 200


def test_asr_affinity_routes_to_owner_node_among_several(
    tmp_path: Path, http: httpx.Client
) -> None:
    """Two nodes: the job created on node B is always read/deleted on node B."""

    from ai_gateway_pool.config.schema import PoolConfig
    from tests.conftest import raw_config
    from tests.integration.conftest import full_node

    a, b = FakeState(node="node-a"), FakeState(node="node-b")
    with ServerThread(fake_gateway_app(a)) as sa, ServerThread(fake_gateway_app(b)) as sb:
        data = raw_config(
            nodes=[full_node(sa.url, "node-a"), full_node(sb.url, "node-b")],
            routing={"policy": "WEIGHTED_ROUND_ROBIN"},
            job_affinity={"backend": "sqlite", "sqlite_path": str(tmp_path / "a.sqlite3")},
        )
        with pool_server(PoolConfig.model_validate(data)) as p:
            jobs: dict[str, str] = {}
            for i in range(4):  # WRR alternates nodes; ids differ per node
                r = submit(http, p.url, **{"X-Fake-Job-Id": f"job-{i}"})
                jobs[r.json()["job_id"]] = r.headers["x-fake-node"]
            assert set(jobs.values()) == {"node-a", "node-b"}
            for job, owner in jobs.items():
                for _ in range(3):
                    g = http.get(f"{p.url}/v1/audio/transcriptions/{job}")
                    assert g.status_code == 200 and g.headers["x-fake-node"] == owner
                assert (
                    http.delete(f"{p.url}/v1/audio/transcriptions/{job}").headers["x-fake-node"]
                    == owner
                )


def test_asr_unknown_job_never_guessed(fake: Fake, pool: str, http: httpx.Client) -> None:
    fake[0].jobs["external-direct-job"] = {}  # exists on the node, created outside the pool
    before = len(fake[0].received)
    for method in ("GET", "DELETE"):
        r = http.request(method, f"{pool}/v1/audio/transcriptions/external-direct-job")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "JOB_AFFINITY_UNKNOWN"
    assert (
        len(fake[0].received) == before
    )  # no node contacted, another Gateway client job untouched
    assert "external-direct-job" in fake[0].jobs


def test_asr_job_cancelled_409_preserved_mapping_kept(
    fake: Fake, pool: str, http: httpx.Client
) -> None:
    job = submit(http, pool).json()["job_id"]
    r = http.get(
        f"{pool}/v1/audio/transcriptions/{job}",
        headers={"X-Fake-Status": "409", "X-Fake-Code": "job_cancelled"},
    )
    assert r.status_code == 409 and r.json()["detail"]["code"] == "job_cancelled"
    d = http.delete(f"{pool}/v1/audio/transcriptions/{job}", headers={"X-Fake-Status": "500"})
    assert d.status_code == 500  # failed delete: mapping must survive
    assert http.get(f"{pool}/v1/audio/transcriptions/{job}").status_code == 200


def test_asr_resource_saturated_503_retry_after_no_job_registered(
    fake: Fake, pool: str, http: httpx.Client
) -> None:
    r = submit(
        http,
        pool,
        **{"X-Fake-Status": "503", "X-Fake-Code": "resource_saturated", "X-Fake-Retry-After": "12"},
    )
    assert r.status_code == 503
    assert r.headers["retry-after"] == "12"
    assert r.json()["detail"]["code"] == "resource_saturated"
    posts = [x for x in fake[0].received if x.method == "POST"]
    assert len(posts) == 1  # never duplicated


def test_asr_not_resubmitted_on_ambiguous_failure(
    fake: Fake, make_pool: Callable[..., ServerThread], http: httpx.Client
) -> None:
    pool = make_pool(
        fake[1].url, timeouts={"upload": {"connect": 2, "read": 0.3, "write": 5, "pool": 2}}
    ).url
    r = submit(http, pool, **{"X-Fake-Delay": "1"})
    assert r.status_code == 504 and r.json()["error"]["code"] == "UPSTREAM_TIMEOUT"
    assert len([x for x in fake[0].received if x.method == "POST"]) == 1


def test_asr_routes_refused_without_affinity_backend(
    fake: Fake, make_pool: Callable[..., ServerThread], http: httpx.Client
) -> None:
    pool = make_pool(fake[1].url, backend="none").url
    r = submit(http, pool)
    assert r.status_code == 501 and r.json()["error"]["code"] == "ROUTING_NOT_READY"
    assert fake[0].received == []


def test_affinity_store_unavailable_refuses_before_creating_job(
    fake: Fake, tmp_path: Path, http: httpx.Client
) -> None:
    cfg = pool_config(fake[1].url, tmp_path)
    srv = pool_server(cfg)
    store = srv.config.app.state.pool.job_store  # type: ignore[attr-defined]

    async def broken() -> None:
        from ai_gateway_pool.jobs.contracts import AffinityStoreUnavailable

        raise AffinityStoreUnavailable("simulated outage")

    store.ping = broken
    store.lookup_job = lambda *a, **k: broken()
    with srv:
        r = submit(http, srv.url)
        assert r.status_code == 503 and r.json()["error"]["code"] == "AFFINITY_STORE_UNAVAILABLE"
        g = http.get(f"{srv.url}/v1/audio/transcriptions/job-1")
        assert g.status_code == 503
    assert [x for x in fake[0].received if x.method == "POST"] == []  # no orphan job


def test_affinity_write_failure_after_creation_fails_closed_with_compensation(
    fake: Fake, tmp_path: Path, http: httpx.Client
) -> None:
    """POOL-3 (replaces POOL-2 `..._is_flagged`, which announced the job with
    X-AIGP-Affinity: unregistered): a job whose affinity cannot be persisted is
    NEVER announced; the pool deletes that one job on that one node."""

    cfg = pool_config(fake[1].url, tmp_path)
    srv = pool_server(cfg)
    store = srv.config.app.state.pool.job_store  # type: ignore[attr-defined]

    async def broken(*a: object, **k: object) -> None:
        from ai_gateway_pool.jobs.contracts import AffinityStoreUnavailable

        raise AffinityStoreUnavailable("write failed")

    store.register_job = broken
    with srv:
        r = submit(http, srv.url, **{"X-Fake-Job-Id": "job-orphan"})
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "AFFINITY_STORE_UNAVAILABLE" and err["compensation"] == "deleted"
    assert "job_id" not in r.text and "job-orphan" not in r.text  # not announced
    calls = [(x.method, x.path) for x in fake[0].received]
    assert calls == [
        ("POST", "/v1/audio/transcriptions"),
        ("DELETE", "/v1/audio/transcriptions/job-orphan"),
    ]
    assert "job-orphan" not in fake[0].jobs


@pytest.mark.parametrize("bad", ["..", ".", "a%2Fb", "x y"])
def test_asr_invalid_job_id_denied(fake: Fake, pool: str, http: httpx.Client, bad: str) -> None:
    r = http.get(f"{pool}/v1/audio/transcriptions/{bad}")
    assert r.status_code in (400, 404)
    assert fake[0].received == []
