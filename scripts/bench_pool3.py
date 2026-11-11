#!/usr/bin/env python3
"""POOL-3 performance comparison on FAKE Gateways (no GPU, loopback).

    .venv/bin/python scripts/bench_pool3.py --pool2-src <POOL-2 worktree>/src --out perf.csv

Targets, all measured in the same session:
  DIRECT_GATEWAY                 client -> fake node
  POOL_2_EQUIVALENT_SINGLE_NODE  client -> POOL-2 code (révision de référence) -> fake node
  POOL_3_SINGLE_NODE             client -> POOL-3 (SINGLE_NODE) -> fake node
  POOL_3_MULTI_NODE              client -> POOL-3 (WEIGHTED_ROUND_ROBIN) -> 3 fake nodes

Each fake node and each pool is a separate OS process. Budgets are fixed BELOW,
before any measurement, and are never adjusted after observation.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tests.integration.live_servers import free_port  # noqa: E402

MiB = 1024 * 1024

# ------------------------------------------------------------------ budgets (a priori)
BUDGET = {
    # POOL-3 single node vs POOL-2 equivalent (multi-node logic must not regress)
    "p3s_vs_p2_small_p50_ms": 0.30,
    "p3s_vs_p2_small_p95_ms": 0.50,
    "p3s_vs_p2_ttfc_p50_ms": 0.50,
    "p3s_vs_p2_rps16_ratio_min": 0.90,
    # POOL-3 multi-node (3 nodes) vs POOL-3 single node
    "p3m_vs_p3s_small_p50_ms": 0.30,
    "p3m_vs_p3s_rps16_ratio_min": 0.90,
    # in-process costs
    "routing_decision_p95_us": 100.0,
    "redis_op_p95_ms": 2.0,
    # every target
    "error_rate_max": 0.0,
}

POOL_YAML = """schema_version: 1
status: example
environment: staging
routing: {{policy: {policy}}}
job_affinity: {{backend: none}}
nodes:
{nodes}
"""
NODE_YAML = """  - node_id: {nid}
    physical_host_id: {host}
    display_name: {nid}
    environment: staging
    gateway_instance: staging
    base_url: {url}
    capabilities: [GENERATE, EMBEDDINGS]
    default_models: {{GENERATE: llama-chat, EMBEDDINGS: legacy-768}}
    models:
      - {{model_id: legacy-768, capabilities: [EMBEDDINGS], dimension: 768}}
      - {{model_id: llama-chat, capabilities: [GENERATE]}}
"""
POOL2_NODE_YAML = """  - node_id: {nid}
    physical_host_id: {host}
    display_name: {nid}
    environment: staging
    gateway_instance: staging
    base_url: {url}
    capabilities: [GENERATE, EMBEDDINGS]
    models:
      - {{model_id: legacy-768, capabilities: [EMBEDDINGS], dimension: 768}}
      - {{model_id: llama-chat, capabilities: [GENERATE]}}
"""


def wait(url: str) -> None:
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            httpx.get(url, timeout=0.5)
            return
        except httpx.HTTPError:
            time.sleep(0.05)
    raise RuntimeError(f"not up: {url}")


def pct(v: list[float], p: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def hwm(pid: int) -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return 0.0


def start_fake(port: int, node: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.integration.live_servers",
            "--port",
            str(port),
            "--node",
            node,
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def start_pool(cfg: Path, port: int, src: Path | None) -> subprocess.Popen[bytes]:
    env = {
        **os.environ,
        "POOL_ENVIRONMENT": "staging",
        "POOL_CONFIG_FILE": str(cfg),
        "POOL_BIND_PORT": str(port),
        "POOL_LOG_LEVEL": "WARNING",
    }
    if src is not None:
        env["PYTHONPATH"] = str(src)
    code = "from ai_gateway_pool.main import run; run()"
    return subprocess.Popen([sys.executable, "-c", code], env=env)


def measure(base: str, label: str) -> dict[str, float]:
    client = httpx.Client(timeout=60, trust_env=False)

    def small(c: httpx.Client) -> None:
        c.post(
            base + "/v1/embeddings", json={"model": "legacy-768", "texts": ["s"]}
        ).raise_for_status()

    for _ in range(50):
        small(client)
    lat = []
    for _ in range(400):
        t0 = time.perf_counter()
        small(client)
        lat.append((time.perf_counter() - t0) * 1000)
    ttfc, ttfb = [], []
    for _ in range(30):
        t0 = time.perf_counter()
        with client.stream(
            "POST",
            base + "/v1/generate/stream",
            json={"model": "llama-chat"},
            headers={"X-Fake-Lines": "3", "X-Fake-First-Gap": "0.05"},
        ) as r:
            ttfb.append((time.perf_counter() - t0) * 1000)
            it = r.iter_raw()
            next(it)
            ttfc.append((time.perf_counter() - t0) * 1000)
            for _ in it:
                pass
    out = {
        "small_p50_ms": pct(lat, 50),
        "small_p95_ms": pct(lat, 95),
        "ttfb_p50_ms": pct(ttfb, 50),
        "ttfc_p50_ms": pct(ttfc, 50),
    }
    for conc in (1, 4, 16):
        errors = 0

        def worker(_: int) -> int:
            bad = 0
            with httpx.Client(timeout=60, trust_env=False) as c:
                for _ in range(40):
                    try:
                        small(c)
                    except httpx.HTTPError:
                        bad += 1
            return bad

        t0 = time.perf_counter()
        with ThreadPoolExecutor(conc) as ex:
            errors = sum(ex.map(worker, range(conc)))
        total = conc * 40
        out[f"rps_c{conc}"] = total / (time.perf_counter() - t0)
        out[f"error_rate_c{conc}"] = errors / total
    client.close()
    print(f"{label:<32} " + " ".join(f"{k}={v:.3f}" for k, v in out.items()), flush=True)
    return out


def routing_decision_us() -> float:
    sys.path.insert(0, str(ROOT / "src"))
    from tests.conftest import build_config, node

    from ai_gateway_pool.nodes.models import Capability, Environment, ModelRequirement
    from ai_gateway_pool.nodes.registry import NodeRegistry
    from ai_gateway_pool.routing.contracts import RoutingRequest
    from ai_gateway_pool.routing.selector import WeightedRoundRobinSelector

    nodes = [node(f"n{i}", url=f"http://n{i}.invalid:1", host=f"h{i}") for i in range(3)]
    reg = NodeRegistry.from_config(
        build_config(nodes=nodes, routing={"policy": "WEIGHTED_ROUND_ROBIN"})
    )
    sel = WeightedRoundRobinSelector()
    rq = RoutingRequest(
        environment=Environment.STAGING,
        service="embeddings",
        capability=Capability.EMBEDDINGS,
        model=ModelRequirement(model_id="legacy-768"),
    )
    samples = []
    for _ in range(20000):
        t0 = time.perf_counter_ns()
        sel.select(rq, reg)
        samples.append((time.perf_counter_ns() - t0) / 1000)
    return pct(samples[2000:], 95)


def redis_ops_ms(url: str | None) -> dict[str, float] | None:
    if not url:
        return None
    import redis.asyncio as aioredis

    from ai_gateway_pool.jobs.affinity import RedisJobAffinityStore
    from ai_gateway_pool.nodes.models import Environment

    async def go() -> dict[str, float]:
        c = aioredis.from_url(url)
        s = RedisJobAffinityStore(c, key_prefix="benchp3")
        res: dict[str, list[float]] = {"register": [], "lookup": [], "forget": []}
        for i in range(500):
            for op, coro in (
                ("register", s.register_job(Environment.STAGING, f"j{i}", "n", 60)),
                ("lookup", s.lookup_job(Environment.STAGING, f"j{i}")),
                ("forget", s.forget_job(Environment.STAGING, f"j{i}", node_id="n")),
            ):
                t0 = time.perf_counter()
                await coro
                res[op].append((time.perf_counter() - t0) * 1000)
        await c.aclose()
        return {f"redis_{k}_p95_ms": pct(v, 95) for k, v in res.items()} | {
            f"redis_{k}_p50_ms": pct(v, 50) for k, v in res.items()
        }

    return asyncio.run(go())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool2-src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="aigp-p3bench-"))
    procs: list[subprocess.Popen[bytes]] = []
    results: dict[str, dict[str, float]] = {}
    peaks: dict[str, float] = {}
    try:
        fports = [free_port() for _ in range(3)]
        for i, p in enumerate(fports):
            procs.append(start_fake(p, f"n{i + 1}"))
        urls = [f"http://127.0.0.1:{p}" for p in fports]
        for u in urls:
            wait(u + "/health/runtime")

        def pool_cfg(name: str, policy: str, n: int, tmpl: str) -> Path:
            nodes = "".join(
                tmpl.format(nid=f"n{i + 1}", host=f"h{i + 1}", url=urls[i]) for i in range(n)
            )
            path = tmp / f"{name}.yaml"
            path.write_text(POOL_YAML.format(policy=policy, nodes=nodes))
            return path

        targets: list[tuple[str, Callable[[], tuple[str, subprocess.Popen[bytes] | None]]]] = []

        def mk(name: str, cfg: Path, src: Path | None) -> Callable[[], Any]:
            def f() -> tuple[str, subprocess.Popen[bytes]]:
                port = free_port()
                proc = start_pool(cfg, port, src)
                procs.append(proc)
                wait(f"http://127.0.0.1:{port}/health/live")
                return f"http://127.0.0.1:{port}", proc

            return f

        targets = [
            ("DIRECT_GATEWAY", lambda: (urls[0], None)),
            (
                "POOL_2_EQUIVALENT_SINGLE_NODE",
                mk("p2", pool_cfg("p2", "SINGLE_NODE", 1, POOL2_NODE_YAML), args.pool2_src),
            ),
            ("POOL_3_SINGLE_NODE", mk("p3s", pool_cfg("p3s", "SINGLE_NODE", 1, NODE_YAML), None)),
            (
                "POOL_3_MULTI_NODE",
                mk("p3m", pool_cfg("p3m", "WEIGHTED_ROUND_ROBIN", 3, NODE_YAML), None),
            ),
        ]
        for name, factory in targets:
            base, proc = factory()
            results[name] = measure(base, name)
            if proc is not None:
                peaks[name] = hwm(proc.pid)
                proc.terminate()
                proc.wait(10)
        decision = routing_decision_us()
        rops = redis_ops_ms(os.environ.get("AIGP_TEST_REDIS_URL"))
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
                p.wait(10)

    rows: list[dict[str, Any]] = []
    for name, vals in results.items():
        for metric, v in vals.items():
            rows.append(
                {
                    "target": name,
                    "metric": metric,
                    "value": round(v, 4),
                    "budget": "",
                    "verdict": "",
                }
            )
        if name in peaks:
            rows.append(
                {
                    "target": name,
                    "metric": "peak_rss_mib",
                    "value": round(peaks[name], 1),
                    "budget": "",
                    "verdict": "",
                }
            )
    d, p2, p3s, p3m = (
        results[k]
        for k in (
            "DIRECT_GATEWAY",
            "POOL_2_EQUIVALENT_SINGLE_NODE",
            "POOL_3_SINGLE_NODE",
            "POOL_3_MULTI_NODE",
        )
    )

    def check(name: str, value: float, budget_key: str, ok: bool) -> None:
        rows.append(
            {
                "target": "COMPARISON",
                "metric": name,
                "value": round(value, 4),
                "budget": BUDGET[budget_key],
                "verdict": "PASS" if ok else "FAIL",
            }
        )

    for k, bk in (
        ("small_p50_ms", "p3s_vs_p2_small_p50_ms"),
        ("small_p95_ms", "p3s_vs_p2_small_p95_ms"),
        ("ttfc_p50_ms", "p3s_vs_p2_ttfc_p50_ms"),
    ):
        delta = p3s[k] - p2[k]
        check(f"pool3_single_minus_pool2_{k}", delta, bk, delta <= BUDGET[bk])
    r = p3s["rps_c16"] / p2["rps_c16"]
    check(
        "pool3_single_over_pool2_rps_c16",
        r,
        "p3s_vs_p2_rps16_ratio_min",
        r >= BUDGET["p3s_vs_p2_rps16_ratio_min"],
    )
    delta = p3m["small_p50_ms"] - p3s["small_p50_ms"]
    check(
        "pool3_multi_minus_single_small_p50_ms",
        delta,
        "p3m_vs_p3s_small_p50_ms",
        delta <= BUDGET["p3m_vs_p3s_small_p50_ms"],
    )
    r = p3m["rps_c16"] / p3s["rps_c16"]
    check(
        "pool3_multi_over_single_rps_c16",
        r,
        "p3m_vs_p3s_rps16_ratio_min",
        r >= BUDGET["p3m_vs_p3s_rps16_ratio_min"],
    )
    check(
        "routing_decision_p95_us",
        decision,
        "routing_decision_p95_us",
        decision <= BUDGET["routing_decision_p95_us"],
    )
    if rops:
        for k, v in rops.items():
            if k.endswith("_p95_ms"):
                check(k, v, "redis_op_p95_ms", v <= BUDGET["redis_op_p95_ms"])
            else:
                rows.append(
                    {
                        "target": "REDIS",
                        "metric": k,
                        "value": round(v, 4),
                        "budget": "",
                        "verdict": "",
                    }
                )
    errs = max(v for t in results.values() for k, v in t.items() if k.startswith("error_rate"))
    check("max_error_rate", errs, "error_rate_max", errs <= BUDGET["error_rate_max"])
    for name in ("POOL_2_EQUIVALENT_SINGLE_NODE", "POOL_3_SINGLE_NODE", "POOL_3_MULTI_NODE"):
        rows.append(
            {
                "target": "DEBT",
                "metric": f"{name}_rps_c16_over_direct",
                "value": round(results[name]["rps_c16"] / d["rps_c16"], 4),
                "budget": "0.5 (POOL-2 budget, known debt)",
                "verdict": "INFO",
            }
        )
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "metric", "value", "budget", "verdict"])
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        if row["target"] in ("COMPARISON", "DEBT"):
            print(f"{row['metric']:<44} {row['value']:>10} budget={row['budget']} {row['verdict']}")
    return 0 if all(r["verdict"] != "FAIL" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
