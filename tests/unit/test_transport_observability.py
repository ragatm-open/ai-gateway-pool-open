"""Transport rules, gateway payload interpretation, logging hygiene, metrics catalogue."""

from __future__ import annotations

import io
import json
import logging

import pytest

from ai_gateway_pool.gateway.health import (
    FORBIDDEN_PROBE_PATHS,
    SAFE_PROBE_PATHS,
    interpret_models,
    interpret_runtime,
)
from ai_gateway_pool.nodes.models import Capability, CapabilityState, Environment, HealthState
from ai_gateway_pool.observability.logging import JsonFormatter, log_event, scrub
from ai_gateway_pool.observability.metrics import ALLOWED_LABELS, METRICS
from ai_gateway_pool.transport.contracts import (
    DEFAULT_TIMEOUTS,
    RetryDecision,
    TrafficClass,
    contains_upstream_override,
    forwardable_request_headers,
    forwardable_response_headers,
    retry_decision,
)
from tests.conftest import gateway_fixture

# --------------------------------------------------------------------- transport


def test_no_blind_retry_of_inference_posts() -> None:
    for established in (True, False):
        assert retry_decision("POST", connection_established=established, attempt=0) is (
            RetryDecision.NO_RETRY
        )
    # POOL-2: a GET is retried only if no response byte was received.
    assert retry_decision("GET", connection_established=False, attempt=0) is (
        RetryDecision.RETRY_SAME_NODE
    )
    assert retry_decision("GET", connection_established=True, attempt=0) is (RetryDecision.NO_RETRY)
    assert retry_decision("GET", connection_established=True, attempt=1) is RetryDecision.NO_RETRY
    assert retry_decision("DELETE", connection_established=True, attempt=0) is (
        RetryDecision.NO_RETRY
    )


def test_timeout_profiles_are_distinct() -> None:
    probe = DEFAULT_TIMEOUTS[TrafficClass.HEALTH_PROBE]
    infer = DEFAULT_TIMEOUTS[TrafficClass.INFERENCE]
    assert probe.total_budget is not None and probe.total_budget <= 10
    # Long generations / uploads / streams are never cut by a short global budget.
    for cls in (TrafficClass.INFERENCE, TrafficClass.STREAMING, TrafficClass.UPLOAD):
        assert DEFAULT_TIMEOUTS[cls].total_budget is None
    assert infer.read is not None and infer.read >= 300  # another Gateway client generate = 300 s
    assert DEFAULT_TIMEOUTS[TrafficClass.UPLOAD].write >= 600  # type: ignore[operator]
    httpx_timeout = infer.to_httpx()
    assert httpx_timeout.connect == infer.connect


def test_header_forwarding_rules() -> None:
    incoming = {
        "Content-Type": "application/json",
        "Authorization": "Bearer abc",
        "Connection": "keep-alive",
        "Host": "pool.internal",
        "Content-Length": "12",
        "X-Upstream-URL": "http://evil.example",
        "X-AIGP-Target-Node": "gateway-staging",
        "Accept": "application/x-ndjson",
    }
    out = forwardable_request_headers(incoming)
    # POOL-2: Content-Length is now KEPT (body relayed with the declared length,
    # never with Transfer-Encoding as well); results are ordered (name, value) pairs.
    assert out == [
        ("Content-Type", "application/json"),
        ("Authorization", "Bearer abc"),
        ("Content-Length", "12"),
        ("Accept", "application/x-ndjson"),
    ]
    # The pool's own admin token is never forwarded to a Gateway.
    assert forwardable_request_headers(incoming, admin_token="abc") == [
        ("Content-Type", "application/json"),
        ("Content-Length", "12"),
        ("Accept", "application/x-ndjson"),
    ]
    assert forwardable_response_headers(
        [
            ("Retry-After", "5"),
            ("Transfer-Encoding", "chunked"),
            ("Set-Cookie", "a"),
            ("Set-Cookie", "b"),
        ]
    ) == [("Retry-After", "5"), ("Set-Cookie", "a"), ("Set-Cookie", "b")]


@pytest.mark.parametrize(
    "header", ["X-Upstream-URL", "x-forwarded-host", "Forwarded", "X-Target-Url"]
)
def test_upstream_override_headers_detected(header: str) -> None:
    assert contains_upstream_override([header])
    assert not contains_upstream_override(["Content-Type", "X-Request-ID"])


# --------------------------------------------------------------------- gateway health


def test_inferential_endpoints_are_never_probe_paths() -> None:
    assert "/health/ready" in FORBIDDEN_PROBE_PATHS
    assert not (SAFE_PROBE_PATHS & FORBIDDEN_PROBE_PATHS)


def test_interpret_runtime_fixture() -> None:
    obs = interpret_runtime(gateway_fixture("health-runtime.staging.json"), Environment.STAGING)
    assert obs.health is HealthState.HEALTHY
    # `0ba5e0ba…` : identifiant de build SYNTHÉTIQUE porté par la fixture.
    assert obs.build_git_sha and obs.build_git_sha.startswith("0ba5e0ba")
    assert obs.capability_states[Capability.ASR] is CapabilityState.AVAILABLE
    assert obs.capability_states[Capability.RERANK] is CapabilityState.AVAILABLE
    # Capabilities without an explicit signal are NOT claimed.
    assert Capability.EMBEDDINGS not in obs.capability_states
    assert {"llama-chat", "rerank-default", "asr-whisper-turbo"} <= obs.model_ids


def test_runtime_environment_mismatch_marks_unavailable() -> None:
    obs = interpret_runtime(gateway_fixture("health-runtime.staging.json"), Environment.PRODUCTION)
    assert obs.health is HealthState.UNAVAILABLE
    assert obs.error == "environment_mismatch"


@pytest.mark.parametrize("payload", [None, [], "ok", {"environment": None}])
def test_runtime_invalid_payload(payload: object) -> None:
    assert interpret_runtime(payload, Environment.STAGING).health is HealthState.UNAVAILABLE


def test_interpret_models_fixture() -> None:
    ids = interpret_models(gateway_fixture("v1-models.json"))
    assert ids is not None
    assert {"legacy-768", "nomic-embed-text:latest", "fr-en-retrieval", "llama-chat"} <= ids
    assert interpret_models({"models": "nope"}) is None


# --------------------------------------------------------------------- logging


def _capture() -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    logger = logging.getLogger("test.pool")
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger, stream


def test_secret_logging_absent() -> None:
    logger, stream = _capture()
    log_event(
        logger,
        logging.INFO,
        "upstream said Authorization: Bearer sk-live-abcdef1234567890 api_key=zzz",
        request_id="r1",
        prompt="Tell me the secret plan",  # not an allowed field
        body='{"texts":["confidential"]}',  # not an allowed field
        headers={"Authorization": "Bearer t"},  # not an allowed field
        selected_node_id="gateway-staging",
    )
    line = stream.getvalue()
    record = json.loads(line)
    assert record["request_id"] == "r1"
    assert record["selected_node_id"] == "gateway-staging"
    for leaked in ("sk-live", "zzz", "secret plan", "confidential", "Bearer t", "prompt", "body"):
        assert leaked not in line


def test_exception_message_not_logged() -> None:
    logger, stream = _capture()
    try:
        raise ValueError("payload fragment: patient record 42")
    except ValueError:
        logger.exception("failure")
    line = stream.getvalue()
    assert "patient record" not in line
    assert json.loads(line)["exception"] == "ValueError"


def test_scrub_patterns() -> None:
    assert "abc" not in scrub("Bearer abc.def")
    assert "hunter2" not in scrub("password=hunter2")


# --------------------------------------------------------------------- metrics


def test_metric_labels_are_bounded() -> None:
    names = [m.name for m in METRICS]
    assert len(names) == len(set(names))
    for metric in METRICS:
        assert set(metric.labels) <= ALLOWED_LABELS, metric.name
        assert not any(w in metric.name for w in ("vram", "gpu_memory"))


def test_metrics_registry_render_and_label_guard() -> None:
    from ai_gateway_pool.observability.metrics import MetricsRegistry

    reg = MetricsRegistry()
    reg.inc(
        "aigp_proxy_requests_total",
        environment="staging",
        service="chat",
        node_id="n",
        status_class="2xx",
    )
    reg.observe(
        "aigp_proxy_request_duration_seconds",
        0.02,
        environment="staging",
        service="chat",
        node_id="n",
    )
    text = reg.render()
    assert 'aigp_proxy_requests_total{environment="staging",service="chat",node_id="n",' in text
    assert "aigp_proxy_request_duration_seconds_count" in text
    with pytest.raises(ValueError):
        reg.inc("aigp_proxy_requests_total", environment="staging", prompt="leak")
