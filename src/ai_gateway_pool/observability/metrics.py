"""In-process metrics (no external system), rendered in Prometheus text format
on the protected ``GET /internal/metrics`` route.

Label sets are bounded: no prompt, document, user, job id or URL can become a
label. No GPU/VRAM metric is defined: the Gateway exposes no reliable one and
the pool must not invent them.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum


class MetricType(StrEnum):
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"


ALLOWED_LABELS: frozenset[str] = frozenset(
    {"environment", "service", "node_id", "status_class", "error_category"}
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    type: MetricType
    labels: tuple[str, ...]
    help: str


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "aigp_proxy_requests_total",
        MetricType.COUNTER,
        ("environment", "service", "node_id", "status_class"),
        "Requests relayed to an upstream node (any final status)",
    ),
    MetricSpec(
        "aigp_proxy_errors_total",
        MetricType.COUNTER,
        ("environment", "service", "error_category"),
        "Pool-originated errors (routing, transport, affinity)",
    ),
    MetricSpec(
        "aigp_proxy_request_duration_seconds",
        MetricType.HISTOGRAM,
        ("environment", "service", "node_id"),
        "Time until upstream response headers were relayed",
    ),
    MetricSpec(
        "aigp_proxy_active_requests",
        MetricType.GAUGE,
        ("environment", "node_id"),
        "In-flight relayed requests (including open streams)",
    ),
    MetricSpec(
        "aigp_proxy_streaming_responses_total",
        MetricType.COUNTER,
        ("environment", "service", "node_id"),
        "NDJSON streaming responses relayed",
    ),
    MetricSpec(
        "aigp_proxy_upstream_unavailable_total",
        MetricType.COUNTER,
        ("environment", "node_id"),
        "Upstream connection failures or timeouts",
    ),
    MetricSpec(
        "aigp_proxy_model_unavailable_total",
        MetricType.COUNTER,
        ("environment", "service"),
        "Requests refused because no node serves the requested model",
    ),
    MetricSpec(
        "aigp_proxy_asr_affinity_lookup_failures_total",
        MetricType.COUNTER,
        ("environment", "error_category"),
        "ASR GET/DELETE without a usable job -> node mapping",
    ),
    MetricSpec(
        "aigp_proxy_stream_interruptions_total",
        MetricType.COUNTER,
        ("environment", "service", "node_id"),
        "Upstream bodies broken mid-way or abandoned by the client",
    ),
)

# POOL-3 multi-node routing / failover / affinity (bounded labels only).
METRICS = (
    *METRICS,
    MetricSpec(
        "aigp_pool_routing_decisions_total",
        MetricType.COUNTER,
        ("environment", "service", "node_id"),
        "Routing decisions (node chosen before dispatch)",
    ),
    MetricSpec(
        "aigp_pool_routing_rejections_total",
        MetricType.COUNTER,
        ("environment", "service", "error_category"),
        "Requests refused by routing (no eligible node, model, capability...)",
    ),
    MetricSpec(
        "aigp_pool_node_selection_total",
        MetricType.COUNTER,
        ("environment", "service", "node_id"),
        "Requests that obtained an HTTP answer from the node",
    ),
    MetricSpec(
        "aigp_pool_node_unavailable_total",
        MetricType.COUNTER,
        ("environment", "node_id"),
        "Circuit breaker transitions to OPEN (this process)",
    ),
    MetricSpec(
        "aigp_pool_failover_attempts_total",
        MetricType.COUNTER,
        ("environment", "service"),
        "Pre-dispatch failovers attempted (nothing had been sent)",
    ),
    MetricSpec(
        "aigp_pool_failover_success_total",
        MetricType.COUNTER,
        ("environment", "service"),
        "Failovers that obtained an answer from another node",
    ),
    MetricSpec(
        "aigp_pool_failover_rejected_total",
        MetricType.COUNTER,
        ("environment", "service", "error_category"),
        "Failovers refused (route not eligible, no compatible alternative)",
    ),
    MetricSpec(
        "aigp_pool_asr_affinity_conflicts_total",
        MetricType.COUNTER,
        ("environment",),
        "ASR job ids already bound to another node (collision, fail closed)",
    ),
    MetricSpec(
        "aigp_pool_asr_affinity_store_errors_total",
        MetricType.COUNTER,
        ("environment",),
        "Affinity store unavailable on an ASR operation",
    ),
)

_SPECS = {m.name: m for m in METRICS}
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600)

LabelKey = tuple[tuple[str, str], ...]


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, dict[LabelKey, float]] = defaultdict(dict)
        self._hist: dict[str, dict[LabelKey, list[float]]] = defaultdict(dict)

    @staticmethod
    def _key(name: str, labels: dict[str, str]) -> LabelKey:
        spec = _SPECS[name]
        if set(labels) != set(spec.labels):
            raise ValueError(f"{name}: labels {sorted(labels)} != {sorted(spec.labels)}")
        return tuple((k, str(labels[k])) for k in spec.labels)

    def add(self, name: str, amount: float, labels: dict[str, str]) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._values[name][key] = self._values[name].get(key, 0) + amount

    def inc(self, name: str, /, **labels: str) -> None:
        self.add(name, 1, labels)

    def dec(self, name: str, /, **labels: str) -> None:
        self.add(name, -1, labels)

    def observe(self, name: str, value: float, /, **labels: str) -> None:
        key = self._key(name, labels)
        with self._lock:
            counts = self._hist[name].setdefault(key, [0.0] * (len(_BUCKETS) + 2))
            for i, bound in enumerate(_BUCKETS):
                if value <= bound:
                    counts[i] += 1
            counts[-2] += 1  # count
            counts[-1] += value  # sum

    def value(self, name: str, /, **labels: str) -> float:
        return self._values[name].get(self._key(name, labels), 0)

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for spec in METRICS:
                lines.append(f"# HELP {spec.name} {spec.help}")
                lines.append(f"# TYPE {spec.name} {spec.type.value}")
                if spec.type is MetricType.HISTOGRAM:
                    for key, counts in self._hist[spec.name].items():
                        base = ",".join(f'{k}="{v}"' for k, v in key)
                        for i, bound in enumerate(_BUCKETS):
                            lines.append(f'{spec.name}_bucket{{{base},le="{bound}"}} {counts[i]:g}')
                        lines.append(f'{spec.name}_bucket{{{base},le="+Inf"}} {counts[-2]:g}')
                        lines.append(f"{spec.name}_count{{{base}}} {counts[-2]:g}")
                        lines.append(f"{spec.name}_sum{{{base}}} {counts[-1]:.6f}")
                else:
                    for key, val in self._values[spec.name].items():
                        base = ",".join(f'{k}="{v}"' for k, v in key)
                        lines.append(f"{spec.name}{{{base}}} {val:g}")
        return "\n".join(lines) + "\n"
