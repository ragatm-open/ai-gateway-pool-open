"""Multi-node routing, controlled failover and ASR collisions through the real pool.

Several fake Gateways (real HTTP servers) stand for several AI Gateway nodes; each
records exactly what it receives, so tests can prove which node got each call and
that no non-idempotent POST is ever executed twice.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_gateway_pool.config.schema import PoolConfig
from tests.conftest import ADMIN_TOKEN, raw_config
from tests.integration.conftest import full_node, pool_server
from tests.integration.live_servers import (
    DropServer,
    FakeState,
    ServerThread,
    fake_gateway_app,
    free_port,
)

AUTH = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
FILES = {
    "request": (None, b'{"model":"asr-whisper-turbo","mode":"async"}', "application/json"),
    "file": ("audio", b"synthetic", "application/octet-stream"),
}


class Cluster:
    def __init__(self, stack: ExitStack, names: list[str]) -> None:
        self.states = {n: FakeState(node=n) for n in names}
        self.servers = {
            n: stack.enter_context(ServerThread(fake_gateway_app(s)))
            for n, s in self.states.items()
        }

    def url(self, name: str) -> str:
        return self.servers[name].url

    def calls(self, name: str, method: str | None = None, path: str | None = None) -> int:
        return sum(
            1
            for r in self.states[name].received
            if (method is None or r.method == method) and (path is None or r.path == path)
        )


def make_pool(
    tmp_path: Path, nodes: list[dict[str, Any]], policy: str = "WEIGHTED_ROUND_ROBIN", **extra: Any
) -> ServerThread:
    extra.setdefault(
        "job_affinity", {"backend": "sqlite", "sqlite_path": str(tmp_path / "aff.sqlite3")}
    )
    cfg = PoolConfig.model_validate(raw_config(nodes=nodes, routing={"policy": policy}, **extra))
    return pool_server(cfg)


@pytest.fixture
def stack() -> Iterator[ExitStack]:
    with ExitStack() as s:
        yield s


@pytest.fixture
def http() -> Iterator[httpx.Client]:
    with httpx.Client(timeout=30, trust_env=False) as c:
        yield c


def emb(http: httpx.Client, pool: str, model: str = "legacy-768", **h: str) -> httpx.Response:
    return http.post(pool + "/v1/embeddings", json={"model": model, "texts": ["x"]}, headers=h)


# ------------------------------------------------------------------ distribution


def test_wrr_distribution_follows_weights(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["n1", "n2", "n3"])
    nodes = [
        full_node(c.url("n1"), "n1"),
        full_node(c.url("n2"), "n2"),
        full_node(c.url("n3"), "n3"),
    ]
    nodes[0]["weight"], nodes[1]["weight"], nodes[2]["weight"] = 200, 100, 100
    pool = stack.enter_context(make_pool(tmp_path, nodes))
    served = Counter(emb(http, pool.url).headers["x-fake-node"] for _ in range(40))
    assert served == {"n1": 20, "n2": 10, "n3": 10}


def test_least_inflight_routes_to_idle_node(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["n1", "n2"])
    pool = stack.enter_context(
        make_pool(
            tmp_path,
            [full_node(c.url("n1"), "n1"), full_node(c.url("n2"), "n2")],
            policy="LEAST_INFLIGHT",
        )
    )
    with http.stream(
        "POST",
        pool.url + "/v1/generate/stream",
        json={"model": "llama-chat"},
        headers={"X-Fake-Lines": "50", "X-Fake-Chunk-Delay": "0.05"},
    ) as r:
        first = r.headers["x-fake-node"]
        # Keep a reference to the line iterator: a discarded generator is closed
        # by the GC, which closes the stream (and the pool rightly ends the relay).
        lines = r.iter_lines()
        next(lines)
        # n-first holds one open stream: the next requests go to the other node.
        others = {emb(http, pool.url).headers["x-fake-node"] for _ in range(3)}
    assert others == {"n2" if first == "n1" else "n1"}


def test_model_present_on_one_node_only(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["n1", "n2"])
    n2 = full_node(c.url("n2"), "n2")
    n2["models"] = [
        *n2["models"],
        {"model_id": "fr-en-retrieval", "capabilities": ["EMBEDDINGS"], "dimension": 768},
    ]
    pool = stack.enter_context(make_pool(tmp_path, [full_node(c.url("n1"), "n1"), n2]))
    assert {emb(http, pool.url, "fr-en-retrieval").headers["x-fake-node"] for _ in range(6)} == {
        "n2"
    }


def test_disabled_and_incompatible_nodes_never_receive(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["ok", "off", "nochat"])
    off = full_node(c.url("off"), "off") | {"enabled": False}
    nochat = full_node(c.url("nochat"), "nochat") | {
        "capabilities": ["EMBEDDINGS"],
        "models": [{"model_id": "legacy-768", "capabilities": ["EMBEDDINGS"]}],
        "default_models": {},
    }
    pool = stack.enter_context(make_pool(tmp_path, [full_node(c.url("ok"), "ok"), off, nochat]))
    for _ in range(4):
        http.post(pool.url + "/v1/generate", json={"model": "llama-chat", "prompt": "x"})
    assert c.calls("off") == 0 and c.calls("nochat") == 0 and c.calls("ok") == 4


def test_observed_digest_conflict_between_nodes_refused(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["n1", "n2"])
    c.states["n2"].digest_overrides = {
        "legacy-768": "ffffdifferent",
        "nomic-embed-text:latest": "ffffdifferent",
    }
    pool = stack.enter_context(
        make_pool(tmp_path, [full_node(c.url("n1"), "n1"), full_node(c.url("n2"), "n2")])
    )
    for nid in ("n1", "n2"):
        http.post(f"{pool.url}/internal/nodes/{nid}/probe", headers=AUTH)
    r = emb(http, pool.url)
    assert r.status_code == 404 and "different artifacts" in r.json()["error"]["message"]
    assert c.calls("n1", "POST") == c.calls("n2", "POST") == 0


def test_routing_explain_endpoint(stack: ExitStack, tmp_path: Path, http: httpx.Client) -> None:
    c = Cluster(stack, ["n1", "n2"])
    n2 = full_node(c.url("n2"), "n2") | {"enabled": False}
    pool = stack.enter_context(make_pool(tmp_path, [full_node(c.url("n1"), "n1"), n2]))
    body = http.post(
        pool.url + "/internal/routing/explain",
        headers=AUTH,
        json={"path": "/v1/embeddings", "model": "legacy-768"},
    ).json()
    assert body["candidates"] == ["n1"] and body["excluded"] == {"n2": "DISABLED"}
    assert body["failover_allowed_for_route"] is True
    assert (
        http.post(
            pool.url + "/internal/routing/explain", json={"path": "/v1/embeddings"}
        ).status_code
        == 401
    )
    assert c.calls("n1") == 0  # explain never sends anything


# ------------------------------------------------------------------ failover


def test_connect_refused_fails_over_before_dispatch(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["alive"])
    dead = f"http://127.0.0.1:{free_port()}"  # nothing listens: connection refused
    nodes = [full_node(dead, "dead"), full_node(c.url("alive"), "alive")]
    pool = stack.enter_context(make_pool(tmp_path, nodes))
    for _ in range(8):
        r = http.post(pool.url + "/v1/generate", json={"model": "llama-chat", "prompt": "x"})
        assert r.status_code == 200 and r.headers["x-fake-node"] == "alive"
    assert c.calls("alive", "POST", "/v1/generate") == 8  # exactly once each
    metrics = http.get(pool.url + "/internal/metrics", headers=AUTH).text
    # WRR picks "dead" every other request: 3 refused connects open its breaker,
    # after which it is excluded BEFORE any connection attempt (no more failovers).
    assert (
        'aigp_pool_failover_attempts_total{environment="staging",service="generate"} 3' in metrics
    )
    assert 'aigp_pool_failover_success_total{environment="staging",service="generate"} 3' in metrics
    view = {
        n["node_id"]: n
        for n in http.get(pool.url + "/internal/nodes", headers=AUTH).json()["nodes"]
    }
    assert view["dead"]["availability_this_process"]["breaker_state"] == "OPEN"


def test_post_lost_after_send_never_replayed(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["alive"])
    drop = stack.enter_context(DropServer())
    pool = stack.enter_context(
        make_pool(tmp_path, [full_node(drop.url, "drop"), full_node(c.url("alive"), "alive")])
    )
    r = http.post(pool.url + "/v1/generate", json={"model": "llama-chat", "prompt": "x"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "UPSTREAM_UNAVAILABLE"
    assert "NOT retried" in r.json()["error"]["message"]
    assert drop.requests == 1 and c.calls("alive", "POST") == 0  # never executed twice


def test_asr_submit_lost_after_send_not_resubmitted(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["alive"])
    drop = stack.enter_context(DropServer())
    pool = stack.enter_context(
        make_pool(tmp_path, [full_node(drop.url, "drop"), full_node(c.url("alive"), "alive")])
    )
    r = http.post(pool.url + "/v1/audio/transcriptions", files=FILES)
    assert r.status_code == 503
    assert drop.requests == 1 and c.calls("alive") == 0


def test_timeout_after_send_not_failed_over(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["slow", "alive"])
    pool = stack.enter_context(
        make_pool(
            tmp_path,
            [full_node(c.url("slow"), "slow"), full_node(c.url("alive"), "alive")],
            timeouts={"inference": {"connect": 2, "read": 0.4, "write": 5, "pool": 2}},
        )
    )
    r = http.post(
        pool.url + "/v1/generate",
        json={"model": "llama-chat", "prompt": "x"},
        headers={"X-Fake-Delay": "1.5"},
    )
    assert r.status_code == 504
    assert c.calls("slow", "POST") + c.calls("alive", "POST") == 1


def test_503_relayed_not_retried_elsewhere_then_soft_excluded(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["n1", "n2"])
    pool = stack.enter_context(
        make_pool(tmp_path, [full_node(c.url("n1"), "n1"), full_node(c.url("n2"), "n2")])
    )
    r = emb(
        http,
        pool.url,
        **{"X-Fake-Status": "503", "X-Fake-Code": "resource_saturated", "X-Fake-Retry-After": "5"},
    )
    first = r.headers["x-fake-node"]
    other = "n2" if first == "n1" else "n1"
    assert r.status_code == 503 and r.headers["retry-after"] == "5"
    assert r.json()["detail"]["code"] == "resource_saturated"
    assert c.calls(other, "POST") == 0  # the saturated answer is NOT replayed elsewhere
    # Next requests avoid the saturated node while an alternative exists.
    assert {emb(http, pool.url).headers["x-fake-node"] for _ in range(4)} == {other}


def test_stream_interruption_never_replayed(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["n1", "n2"])
    pool = stack.enter_context(
        make_pool(tmp_path, [full_node(c.url("n1"), "n1"), full_node(c.url("n2"), "n2")])
    )
    lines: list[str] = []
    with pytest.raises(httpx.HTTPError):  # noqa: SIM117
        with http.stream(
            "POST",
            pool.url + "/v1/generate/stream",
            json={"model": "llama-chat"},
            headers={"X-Fake-Interrupt-After": "2", "X-Fake-Lines": "5"},
        ) as r:
            lines.extend(r.iter_lines())
    total = sum(c.calls(n, "POST", "/v1/generate/stream") for n in ("n1", "n2"))
    assert total == 1 and all('"done"' not in x for x in lines)


def test_no_alternative_gives_stable_error(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    dead1, dead2 = (f"http://127.0.0.1:{free_port()}" for _ in range(2))
    pool = stack.enter_context(
        make_pool(tmp_path, [full_node(dead1, "d1"), full_node(dead2, "d2")])
    )
    r = emb(http, pool.url)
    assert r.status_code == 503 and r.json()["error"]["code"] == "UPSTREAM_UNAVAILABLE"


def test_alternative_without_model_is_never_used(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["other"])
    dead = f"http://127.0.0.1:{free_port()}"
    with_fr = full_node(dead, "dead")
    with_fr["models"] = [
        *with_fr["models"],
        {"model_id": "fr-en-retrieval", "capabilities": ["EMBEDDINGS"]},
    ]
    pool = stack.enter_context(make_pool(tmp_path, [with_fr, full_node(c.url("other"), "other")]))
    r = emb(http, pool.url, "fr-en-retrieval")
    assert r.status_code == 503 and c.calls("other") == 0


def test_diagnostics_not_failed_over(stack: ExitStack, tmp_path: Path, http: httpx.Client) -> None:
    c = Cluster(stack, ["alive"])
    dead = f"http://127.0.0.1:{free_port()}"
    pool = stack.enter_context(
        make_pool(
            tmp_path,
            [full_node(dead, "dead"), full_node(c.url("alive"), "alive")],
            policy="WEIGHTED_ROUND_ROBIN",
        )
    )
    r = http.get(pool.url + "/health/runtime")  # first candidate = dead
    assert r.status_code == 503 and c.calls("alive") == 0  # never another node's identity


# ------------------------------------------------------------------ ASR multi-node


def test_asr_native_id_collision_fails_closed(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["a", "b"])
    pool = stack.enter_context(
        make_pool(tmp_path, [full_node(c.url("a"), "a"), full_node(c.url("b"), "b")])
    )
    hdr = {"X-Fake-Job-Id": "same-native-id"}
    first = http.post(pool.url + "/v1/audio/transcriptions", files=FILES, headers=hdr)
    owner = first.headers["x-fake-node"]
    other = "b" if owner == "a" else "a"
    second = http.post(pool.url + "/v1/audio/transcriptions", files=FILES, headers=hdr)
    assert first.status_code == 202
    assert second.status_code == 409
    err = second.json()["error"]
    assert err["code"] == "JOB_AFFINITY_CONFLICT" and err["compensation"] == "deleted"
    assert c.calls(other, "DELETE") == 1 and "same-native-id" not in c.states[other].jobs
    assert "same-native-id" in c.states[owner].jobs  # the owner's job is untouched
    for _ in range(3):
        g = http.get(pool.url + "/v1/audio/transcriptions/same-native-id")
        assert g.status_code == 200 and g.headers["x-fake-node"] == owner
    assert c.calls(other, "GET") == 0


def test_asr_connect_failure_fails_over_and_binds_new_node(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["alive"])
    dead = f"http://127.0.0.1:{free_port()}"
    pool = stack.enter_context(
        make_pool(tmp_path, [full_node(dead, "dead"), full_node(c.url("alive"), "alive")])
    )
    r = http.post(pool.url + "/v1/audio/transcriptions", files=FILES)
    assert r.status_code == 202 and r.headers["x-fake-node"] == "alive"
    job = r.json()["job_id"]
    assert http.get(f"{pool.url}/v1/audio/transcriptions/{job}").headers["x-fake-node"] == "alive"


def test_asr_job_of_removed_node_not_rerouted(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    c = Cluster(stack, ["a", "b"])
    nodes = [full_node(c.url("a"), "a"), full_node(c.url("b"), "b")]
    with make_pool(tmp_path, nodes, policy="LEAST_INFLIGHT") as p1:
        job = http.post(p1.url + "/v1/audio/transcriptions", files=FILES).json()["job_id"]
    owner = "a"  # LEAST_INFLIGHT, all idle -> config order
    remaining = [n for n in nodes if n["node_id"] != owner]
    with make_pool(tmp_path, remaining, policy="LEAST_INFLIGHT") as p2:
        g = http.get(f"{p2.url}/v1/audio/transcriptions/{job}")
    assert g.status_code == 503 and g.json()["error"]["code"] == "NODE_UNAVAILABLE"
    assert c.calls("b", "GET") == 0


def test_asr_owner_node_down_no_exploration(
    stack: ExitStack, tmp_path: Path, http: httpx.Client
) -> None:
    state_a = FakeState(node="a")
    srv_a = ServerThread(fake_gateway_app(state_a)).__enter__()
    c = Cluster(stack, ["b"])
    nodes = [full_node(srv_a.url, "a"), full_node(c.url("b"), "b")]
    pool = stack.enter_context(make_pool(tmp_path, nodes, policy="LEAST_INFLIGHT"))
    job = http.post(pool.url + "/v1/audio/transcriptions", files=FILES).json()["job_id"]
    srv_a.__exit__(None, None, None)  # owner node goes down
    g = http.get(f"{pool.url}/v1/audio/transcriptions/{job}")
    assert g.status_code == 503 and c.calls("b", "GET") == 0


@pytest.mark.parametrize(
    ("exc", "fails_over"),
    [
        (httpx.ConnectTimeout, True),
        (httpx.PoolTimeout, True),
        (httpx.ReadTimeout, False),
        (httpx.RemoteProtocolError, False),
        (httpx.WriteError, False),
    ],
)
def test_failover_classification_by_transport_error(exc: type[Exception], fails_over: bool) -> None:
    """Deterministic transport: node `a` raises `exc`, node `b` answers."""

    import asyncio

    from ai_gateway_pool.config.settings import Settings
    from ai_gateway_pool.main import create_app

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "a.invalid":
            raise exc("simulated", request=request)
        return httpx.Response(200, json={"node": "b"})

    nodes = [full_node("http://a.invalid:1", "a"), full_node("http://b.invalid:1", "b")]
    cfg = PoolConfig.model_validate(raw_config(nodes=nodes, routing={"policy": "LEAST_INFLIGHT"}))
    app = create_app(Settings(environment="staging", config_file=Path("/x")), cfg)
    app.state.pool.client_factory = lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )

    async def go() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://pool"
        ) as c:
            return await c.post("/v1/generate", json={"model": "llama-chat", "prompt": "x"})

    r = asyncio.run(go())
    if fails_over:
        assert r.status_code == 200 and calls == ["a.invalid", "b.invalid"]
    else:
        assert r.status_code in (503, 504) and calls == ["a.invalid"]  # never replayed
