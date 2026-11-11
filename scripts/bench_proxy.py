#!/usr/bin/env python3
"""Direct vs pool benchmark on a FAKE Gateway (no GPU, loopback, reproducible).

    .venv/bin/python scripts/bench_proxy.py --out direct-vs-proxy-performance.csv

Starts two processes on 127.0.0.1 ephemeral ports: the fake Gateway and the
real pool (``ai-gateway-pool`` entrypoint) in front of it, then measures the same
scenarios directly and through the pool. The pool's memory is read from
/proc/<pid>/status of the pool process (VmRSS sampled, VmHWM peak).

Budgets are fixed HERE, before measuring (see docs/performance.md).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tests.integration.live_servers import free_port  # noqa: E402

MB = 1_000_000
MiB = 1024 * 1024

# ---------------------------------------------------------------- budgets (a priori)
BUDGET = {
    "small_json_p50_overhead_ms": 2.0,
    "small_json_p95_overhead_ms": 5.0,
    "medium_json_p50_overhead_ms": 5.0,
    "stream_first_chunk_p50_overhead_ms": 5.0,
    # >= 1 GbE line rate (~117 MB/s) so the pool is never the LAN bottleneck.
    "upload_throughput_min_MBps": 150.0,
    "download_throughput_min_MBps": 150.0,
    "pool_peak_rss_max_MiB": 160.0,
    "pool_rss_growth_600MB_upload_max_MiB": 64.0,
    "concurrency_error_rate_max": 0.0,
    "concurrency_throughput_ratio_min": 0.5,
}

POOL_YAML = """schema_version: 1
status: example
environment: staging
routing: {{policy: SINGLE_NODE}}
job_affinity: {{backend: sqlite, sqlite_path: {sqlite}}}
nodes:
  - node_id: fake-staging
    physical_host_id: bench
    display_name: bench fake
    environment: staging
    gateway_instance: staging
    base_url: {url}
    capabilities: [GENERATE, CHAT, EMBEDDINGS, RERANK, VISION, STRUCTURED_EXTRACT, ASR]
    models:
      - {{model_id: legacy-768, capabilities: [EMBEDDINGS], dimension: 768}}
      - {{model_id: llama-chat, capabilities: [GENERATE, CHAT]}}
      - {{model_id: asr-whisper-turbo, capabilities: [ASR]}}
"""


def proc_status(pid: int, field: str) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
    return 0


def wait_http(url: str) -> None:
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            httpx.get(url, timeout=0.5)
            return
        except httpx.HTTPError:
            time.sleep(0.05)
    raise RuntimeError(f"not up: {url}")


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


class RssSampler:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.base = proc_status(pid, "VmRSS")
        self.peak = self.base
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, proc_status(self.pid, "VmRSS"))
            time.sleep(0.01)

    def __enter__(self) -> RssSampler:
        self._t.start()
        return self

    def __exit__(self, *a: object) -> None:
        self._stop.set()
        self._t.join()


def multipart(total: int, chunk: int = MiB) -> tuple[Callable[[], Iterator[bytes]], str, int]:
    boundary = "benchBoundary0123456789"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="request"\r\n'
        f'Content-Type: application/json\r\n\r\n{{"model":"asr-whisper-turbo"}}\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    block = b"\xab" * chunk

    def gen() -> Iterator[bytes]:
        yield head
        sent = 0
        while sent < total:
            n = min(chunk, total - sent)
            yield block[:n]
            sent += n
        yield tail

    return gen, f"multipart/form-data; boundary={boundary}", len(head) + total + len(tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--small-n", type=int, default=500)
    ap.add_argument("--upload-bytes", type=int, default=600_000_000)
    ap.add_argument("--download-bytes", type=int, default=512 * MiB)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="aigp-bench-"))
    fport, pport = free_port(), free_port()
    fake_url, pool_url = f"http://127.0.0.1:{fport}", f"http://127.0.0.1:{pport}"
    (tmp / "pool.yaml").write_text(POOL_YAML.format(url=fake_url, sqlite=tmp / "a.sqlite3"))
    env = {
        **os.environ,
        "POOL_ENVIRONMENT": "staging",
        "POOL_CONFIG_FILE": str(tmp / "pool.yaml"),
        "POOL_BIND_PORT": str(pport),
        "POOL_LOG_LEVEL": "WARNING",
        "PYTHONPATH": str(ROOT),
    }
    py = sys.executable
    fake = subprocess.Popen(
        [py, "-m", "tests.integration.live_servers", "--port", str(fport)], cwd=ROOT, env=env
    )
    pool = subprocess.Popen([str(Path(py).parent / "ai-gateway-pool")], cwd=ROOT, env=env)
    rows: list[dict[str, object]] = []
    try:
        wait_http(fake_url + "/health/runtime")
        wait_http(pool_url + "/health/live")
        pool_base_rss = proc_status(pool.pid, "VmRSS")
        client = httpx.Client(timeout=120, trust_env=False)

        def latency(base: str, fn: Callable[[httpx.Client, str], None], n: int) -> list[float]:
            for _ in range(20):  # warm-up
                fn(client, base)
            out = []
            for _ in range(n):
                t0 = time.perf_counter()
                fn(client, base)
                out.append((time.perf_counter() - t0) * 1000)
            return out

        def record(
            scenario: str,
            metric: str,
            direct: float,
            via: float,
            unit: str,
            budget_key: str | None,
            ok: bool | None,
            note: str = "",
        ) -> None:
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "direct": round(direct, 3),
                    "via_pool": round(via, 3),
                    "overhead": round(via - direct, 3),
                    "unit": unit,
                    "budget": BUDGET.get(budget_key, "") if budget_key else "",
                    "verdict": "" if ok is None else ("PASS" if ok else "FAIL"),
                    "note": note,
                }
            )

        # 1. small JSON (embeddings-like POST, ~0.3 KB answer)
        def small(c: httpx.Client, b: str) -> None:
            r = c.post(b + "/v1/embeddings", json={"model": "legacy-768", "texts": ["s"]})
            r.raise_for_status()

        d, v = latency(fake_url, small, args.small_n), latency(pool_url, small, args.small_n)
        for p, key in ((50, "small_json_p50_overhead_ms"), (95, "small_json_p95_overhead_ms")):
            dp, vp = pct(d, p), pct(v, p)
            record("small_json_post", f"latency_p{p}", dp, vp, "ms", key, vp - dp <= BUDGET[key])

        # 2. medium JSON answer (256 KiB)
        def medium(c: httpx.Client, b: str) -> None:
            c.get(
                b + "/health/vision", headers={"X-Fake-Response-Bytes": str(256 * 1024)}
            ).raise_for_status()

        d, v = latency(fake_url, medium, 200), latency(pool_url, medium, 200)
        key = "medium_json_p50_overhead_ms"
        record(
            "medium_response_256KiB",
            "latency_p50",
            pct(d, 50),
            pct(v, 50),
            "ms",
            key,
            pct(v, 50) - pct(d, 50) <= BUDGET[key],
        )
        record("medium_response_256KiB", "latency_p95", pct(d, 95), pct(v, 95), "ms", None, None)

        # 3. streaming: time to first chunk / time to first byte (first line then 200 ms gap)
        def ttfc(base: str) -> tuple[list[float], list[float]]:
            first, total = [], []
            for _ in range(30):
                t0 = time.perf_counter()
                with client.stream(
                    "POST",
                    base + "/v1/generate/stream",
                    json={},
                    headers={"X-Fake-Lines": "3", "X-Fake-First-Gap": "0.2"},
                ) as r:
                    it = r.iter_raw()
                    next(it)
                    first.append((time.perf_counter() - t0) * 1000)
                    for _ in it:
                        pass
                total.append((time.perf_counter() - t0) * 1000)
            return first, total

        (df, dt), (vf, vt) = ttfc(fake_url), ttfc(pool_url)
        key = "stream_first_chunk_p50_overhead_ms"
        record(
            "ndjson_stream",
            "time_to_first_chunk_p50",
            pct(df, 50),
            pct(vf, 50),
            "ms",
            key,
            pct(vf, 50) - pct(df, 50) <= BUDGET[key],
        )
        record(
            "ndjson_stream", "time_to_first_chunk_p95", pct(df, 95), pct(vf, 95), "ms", None, None
        )
        record(
            "ndjson_stream",
            "total_transfer_p50",
            pct(dt, 50),
            pct(vt, 50),
            "ms",
            None,
            None,
            "includes the simulated 200 ms generation gap",
        )

        # 4. large multipart upload (600 000 000 bytes, ASR-shaped, simulated)
        def upload(base: str) -> float:
            gen, ctype, length = multipart(args.upload_bytes)
            t0 = time.perf_counter()
            r = client.post(
                base + "/v1/audio/transcriptions",
                content=gen(),
                headers={"Content-Type": ctype, "Content-Length": str(length)},
            )
            r.raise_for_status()
            return length / (time.perf_counter() - t0) / MB

        du = upload(fake_url)
        with RssSampler(pool.pid) as s:
            vu = upload(pool_url)
        growth = (s.peak - s.base) / MiB
        record(
            "multipart_upload_600MB",
            "throughput",
            du,
            vu,
            "MB/s",
            "upload_throughput_min_MBps",
            vu >= BUDGET["upload_throughput_min_MBps"],
            "simulated payload, not a real ASR job",
        )
        record(
            "multipart_upload_600MB",
            "pool_rss_growth",
            0,
            growth,
            "MiB",
            "pool_rss_growth_600MB_upload_max_MiB",
            growth <= BUDGET["pool_rss_growth_600MB_upload_max_MiB"],
        )

        # 5. long response download
        def download(base: str) -> float:
            t0 = time.perf_counter()
            with client.stream(
                "GET",
                base + "/health/embeddings",
                headers={"X-Fake-Response-Bytes": str(args.download_bytes)},
            ) as r:
                n = sum(len(c) for c in r.iter_raw())
            assert n == args.download_bytes
            return n / (time.perf_counter() - t0) / MB

        dd, vd = download(fake_url), download(pool_url)
        record(
            "long_response_512MiB",
            "throughput",
            dd,
            vd,
            "MB/s",
            "download_throughput_min_MBps",
            vd >= BUDGET["download_throughput_min_MBps"],
        )

        # 6. moderate concurrency: 16 workers x 50 small requests
        def burst(base: str) -> tuple[float, float]:
            errors = 0

            def worker(_: int) -> int:
                bad = 0
                with httpx.Client(timeout=30, trust_env=False) as c:
                    for _ in range(50):
                        try:
                            small(c, base)
                        except httpx.HTTPError:
                            bad += 1
                return bad

            t0 = time.perf_counter()
            with ThreadPoolExecutor(16) as ex:
                errors = sum(ex.map(worker, range(16)))
            return 800 / (time.perf_counter() - t0), errors / 800

        (drps, derr), (vrps, verr) = burst(fake_url), burst(pool_url)
        record(
            "concurrency_16x50",
            "throughput",
            drps,
            vrps,
            "req/s",
            "concurrency_throughput_ratio_min",
            vrps / drps >= BUDGET["concurrency_throughput_ratio_min"],
            f"ratio={vrps / drps:.2f}",
        )
        record(
            "concurrency_16x50",
            "http_error_rate",
            derr,
            verr,
            "ratio",
            "concurrency_error_rate_max",
            verr <= BUDGET["concurrency_error_rate_max"],
        )

        peak = proc_status(pool.pid, "VmHWM") / MiB
        record(
            "pool_process",
            "peak_rss_VmHWM",
            0,
            peak,
            "MiB",
            "pool_peak_rss_max_MiB",
            peak <= BUDGET["pool_peak_rss_max_MiB"],
            f"baseline {pool_base_rss / MiB:.1f} MiB",
        )
        client.close()
    finally:
        for p in (pool, fake):
            p.terminate()
            p.wait(10)

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            f"{r['scenario']:<26} {r['metric']:<26} direct={r['direct']:>10} "
            f"pool={r['via_pool']:>10} {r['unit']:<6} {r['verdict']}"
        )
    return 0 if all(r["verdict"] != "FAIL" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
