"""POOL-3 multi-node routing, independent of HTTP."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from ai_gateway_pool.errors import PoolErrorCode, RoutingError
from ai_gateway_pool.gateway.contracts import ENDPOINTS
from ai_gateway_pool.nodes.availability import AvailabilityTracker, BreakerSettings, BreakerState
from ai_gateway_pool.nodes.models import Capability, Environment, HealthState, ModelRequirement
from ai_gateway_pool.nodes.registry import NodeRegistry
from ai_gateway_pool.routing.contracts import RoutingRequest
from ai_gateway_pool.routing.policies import RoutingPolicyName
from ai_gateway_pool.routing.selector import (
    LeastInflightSelector,
    WeightedRoundRobinSelector,
    build_selector,
    evaluate,
)
from ai_gateway_pool.transport.proxy import failover_allowed
from tests.conftest import build_config, node

STG = Environment.STAGING
EMB = Capability.EMBEDDINGS


def n(nid: str, host: str = "h1", **kw: Any) -> dict[str, Any]:
    return node(nid, url=f"http://{nid}.invalid:8200", host=host, **kw)


def reg(*nodes: dict[str, Any], policy: str = "WEIGHTED_ROUND_ROBIN", **kw: Any) -> NodeRegistry:
    return NodeRegistry.from_config(
        build_config(nodes=list(nodes), routing={"policy": policy}, **kw)
    )


def rq(cap: Capability = EMB, model: str | None = "legacy-768", **kw: Any) -> RoutingRequest:
    return RoutingRequest(
        environment=STG,
        service=cap.value.lower(),
        capability=cap,
        model=ModelRequirement(model_id=model) if model else None,
        **kw,
    )


def observe(
    r: NodeRegistry,
    nid: str,
    *,
    ids: set[str],
    caps: dict[str, set[str]] | None = None,
    digests: dict[str, str] | None = None,
    at: datetime | None = None,
) -> None:
    r.update_state(
        nid,
        health=HealthState.HEALTHY,
        observed_model_ids=frozenset(ids),
        observed_model_caps={k: frozenset(v) for k, v in (caps or {}).items()},
        observed_digests=digests or {},
        checked_at=at,
    )


# --------------------------------------------------------------------- registry


def test_registry_three_nodes_two_physical_hosts() -> None:
    r = reg(n("a", "gpu-host-a"), n("b", "gpu-host-a"), n("c", "gpu2"))
    assert r.physical_hosts() == {"gpu-host-a": ["a", "b"], "gpu2": ["c"]}


@pytest.mark.parametrize("weight", [-1, 1001])
def test_invalid_weight_rejected(weight: int) -> None:
    with pytest.raises(ValidationError):
        build_config(nodes=[n("a", weight=weight)])


def test_invalid_max_inflight_rejected() -> None:
    with pytest.raises(ValidationError):
        build_config(nodes=[n("a", max_inflight=0)])


def test_upstream_allowlist_enforced() -> None:
    build_config(
        nodes=[node("a", url="http://203.0.113.95:8200")], upstream_allowlist=["203.0.113.0/24"]
    )
    with pytest.raises(ValidationError, match="not in upstream_allowlist"):
        build_config(
            nodes=[node("a", url="http://198.51.100.5:8200")], upstream_allowlist=["203.0.113.95"]
        )


def test_approved_config_requires_allowlist_and_durable_affinity() -> None:
    with pytest.raises(ValidationError, match="upstream_allowlist"):
        build_config(status="approved")
    with pytest.raises(ValidationError, match="approved"):
        build_config(
            status="approved",
            upstream_allowlist=["gw-staging.invalid"],
            job_affinity={"backend": "sqlite", "sqlite_path": "/tmp/x"},
        )
    build_config(
        status="approved",
        upstream_allowlist=["gw-staging.invalid"],
        job_affinity={"backend": "redis", "redis_url_env": "POOL_REDIS_URL"},
    )


def test_embedding_identity_must_agree_across_nodes() -> None:
    def emb_node(nid: str, rev: str) -> dict[str, Any]:
        return n(
            nid,
            capabilities=["EMBEDDINGS"],
            models=[
                {
                    "model_id": "legacy-768",
                    "capabilities": ["EMBEDDINGS"],
                    "dimension": 768,
                    "revision": rev,
                }
            ],
        )

    with pytest.raises(ValidationError, match="revision differs"):
        build_config(
            nodes=[emb_node("a", "0a109f42"), emb_node("b", "ffffffff")],
            routing={"policy": "WEIGHTED_ROUND_ROBIN"},
        )


def test_default_model_must_be_declared() -> None:
    with pytest.raises(ValidationError, match="not declared"):
        build_config(nodes=[n("a", default_models={"EMBEDDINGS": "bge-m3"})])


def test_unknown_policy_and_unimplemented_rejected() -> None:
    with pytest.raises(ValidationError):
        build_config(routing={"policy": "RANDOM"})
    with pytest.raises(ValidationError, match="ROUTING_NOT_READY"):
        build_config(routing={"policy": "LEAST_BUSY"})


# --------------------------------------------------------------------- policies


def test_wrr_weights_shape_proportions() -> None:
    r = reg(n("a", weight=300), n("b", weight=100))
    sel = WeightedRoundRobinSelector()
    counts = Counter(sel.select(rq(), r).node_id for _ in range(400))
    assert counts == {"a": 300, "b": 100}


def test_wrr_skips_ineligible_nodes() -> None:
    r = reg(
        n("a"),
        n("b", enabled=False),
        n(
            "c",
            capabilities=["CHAT"],
            models=[{"model_id": "oss-reasoning", "capabilities": ["CHAT"]}],
        ),
    )
    sel = WeightedRoundRobinSelector()
    assert {sel.select(rq(), r).node_id for _ in range(10)} == {"a"}


def test_least_inflight_prefers_idle_physical_host() -> None:
    r = reg(n("a1", "gpu-host-a"), n("a2", "gpu-host-a"), n("b1", "gpu2"), policy="LEAST_INFLIGHT")
    sel = LeastInflightSelector()
    r.availability.begin("a1")  # one request on gpu-host-a (through a1)
    # a2 is idle but shares gpu-host-a's GPU: gpu2 wins.
    assert sel.select(rq(), r).node_id == "b1"
    r.availability.begin("b1")
    r.availability.begin("b1")
    assert sel.select(rq(), r).node_id == "a2"  # gpu-host-a 1 < gpu2 2, a2 idle < a1
    for _ in range(2):
        r.availability.end("b1")
    r.availability.end("a1")
    assert sel.select(rq(), r).node_id == "a1"  # all idle: weight tie -> config order


def test_least_inflight_concurrent_spread() -> None:
    r = reg(n("a", "h1"), n("b", "h2"), n("c", "h3"), policy="LEAST_INFLIGHT")
    sel = LeastInflightSelector()
    picks = []
    for _ in range(6):
        d = sel.select(rq(), r)
        r.availability.begin(d.node_id)
        picks.append(d.node_id)
    assert Counter(picks) == {"a": 2, "b": 2, "c": 2}


def test_single_node_backward_compatible() -> None:
    r = reg(n("a"), policy="SINGLE_NODE")
    d = build_selector(RoutingPolicyName.SINGLE_NODE).select(rq(), r)
    assert d.node_id == "a" and d.candidates == ("a",)


# --------------------------------------------------------------------- capabilities / models


def test_observed_capability_overrides_declaration_llama_chat() -> None:
    """Regression POOL-2: llama-chat publishes completion, not chat."""

    r = reg(
        n(
            "a",
            capabilities=["GENERATE", "CHAT"],
            models=[{"model_id": "llama-chat", "capabilities": ["GENERATE", "CHAT"]}],
        )
    )
    observe(r, "a", ids={"llama-chat"}, caps={"llama-chat": {"completion", "tools"}})
    assert evaluate(rq(Capability.GENERATE, "llama-chat"), r).candidates[0].node_id == "a"
    with pytest.raises(RoutingError) as exc:
        evaluate(rq(Capability.CHAT, "llama-chat"), r)
    assert exc.value.code is PoolErrorCode.MODEL_NOT_AVAILABLE
    assert "CAPABILITY_NOT_OBSERVED" in exc.value.message


def test_model_only_on_some_nodes_is_routed_there() -> None:
    models_b = [{"model_id": "fr-en-retrieval", "capabilities": ["EMBEDDINGS"], "dimension": 768}]
    r = reg(n("a"), n("b", models=models_b))
    ev = evaluate(rq(model="fr-en-retrieval"), r)
    assert [c.node_id for c in ev.candidates] == ["b"]
    assert ev.excluded == {"a": "MODEL_NOT_DECLARED"}


def test_same_dimension_other_profile_never_substituted() -> None:
    models_b = [{"model_id": "fr-en-retrieval", "capabilities": ["EMBEDDINGS"], "dimension": 768}]
    r = reg(n("b", models=models_b))
    with pytest.raises(RoutingError) as exc:
        evaluate(rq(model="legacy-768"), r)
    assert exc.value.code is PoolErrorCode.MODEL_NOT_AVAILABLE


def test_declared_revision_vs_observed_digest() -> None:
    models = [{"model_id": "legacy-768", "capabilities": ["EMBEDDINGS"], "revision": "0a109f42"}]
    r = reg(n("a", models=models), n("b", models=models))
    observe(r, "a", ids={"legacy-768"}, digests={"legacy-768": "0a109f42deadbeef"})
    observe(r, "b", ids={"legacy-768"}, digests={"legacy-768": "ffff0000"})
    ev = evaluate(rq(), r)
    assert [c.node_id for c in ev.candidates] == ["a"]
    assert ev.excluded["b"] == "MODEL_IDENTITY_MISMATCH"


def test_same_model_id_different_artifacts_refused() -> None:
    r = reg(n("a"), n("b"))
    observe(r, "a", ids={"legacy-768"}, digests={"legacy-768": "aaaa"})
    observe(r, "b", ids={"legacy-768"}, digests={"legacy-768": "bbbb"})
    with pytest.raises(RoutingError) as exc:
        evaluate(rq(), r)
    assert exc.value.code is PoolErrorCode.MODEL_NOT_AVAILABLE
    assert "different artifacts" in exc.value.message


def test_model_observed_missing_excluded() -> None:
    r = reg(n("a"), n("b"))
    observe(r, "a", ids={"llama-chat"})
    ev = evaluate(rq(), r)
    assert ev.excluded == {"a": "MODEL_NOT_OBSERVED"} and ev.candidates[0].node_id == "b"


def test_stale_inventory_restricts_never_widens() -> None:
    r = reg(n("a"))
    old = datetime.now(UTC) - timedelta(hours=2)
    observe(r, "a", ids={"llama-chat"}, at=old)
    assert r.inventory_freshness("a") == "STALE"
    with pytest.raises(RoutingError):
        evaluate(rq(), r)  # legacy-768 declared but absent from the (stale) observation
    observe(r, "a", ids={"legacy-768"}, at=old)
    ev = evaluate(rq(), r)
    assert ev.inventory == {"a": "STALE"}


def test_freshness_states() -> None:
    r = reg(n("a"))
    assert r.inventory_freshness("a") == "NEVER_OBSERVED"
    observe(r, "a", ids={"legacy-768"})
    assert r.inventory_freshness("a") == "KNOWN"


# --------------------------------------------------------------------- implicit model


def test_implicit_model_single_candidate_allowed() -> None:
    r = reg(n("a"))
    assert evaluate(rq(model=None), r).candidates[0].node_id == "a"


def test_implicit_model_needs_identical_defaults() -> None:
    r = reg(
        n("a", default_models={"EMBEDDINGS": "legacy-768"}),
        n("b", default_models={"EMBEDDINGS": "legacy-768"}),
    )
    ev = evaluate(rq(model=None), r)
    assert ev.implicit_model == "legacy-768" and len(ev.candidates) == 2
    models_b = [
        {"model_id": "legacy-768", "capabilities": ["EMBEDDINGS"]},
        {"model_id": "fr-en-retrieval", "capabilities": ["EMBEDDINGS"]},
    ]
    r2 = reg(
        n("a", default_models={"EMBEDDINGS": "legacy-768"}),
        n("b", models=models_b, default_models={"EMBEDDINGS": "fr-en-retrieval"}),
    )
    with pytest.raises(RoutingError) as exc:
        evaluate(rq(model=None), r2)
    assert exc.value.code is PoolErrorCode.MODEL_REQUIRED
    with pytest.raises(RoutingError) as exc:
        evaluate(rq(model=None), reg(n("a"), n("b")))  # defaults unknown
    assert exc.value.code is PoolErrorCode.MODEL_REQUIRED


def test_uninspectable_asr_model_needs_identical_model_sets() -> None:
    extra = [
        {"model_id": "asr-other", "capabilities": ["ASR"]},
        {"model_id": "asr-whisper-turbo", "capabilities": ["ASR"]},
    ]
    d = {"ASR": "asr-whisper-turbo"}
    r = reg(n("a", default_models=d), n("b", models=extra, default_models=d))
    with pytest.raises(RoutingError) as exc:
        evaluate(rq(Capability.ASR, None, model_uninspectable=True), r)
    assert exc.value.code is PoolErrorCode.MODEL_REQUIRED


# --------------------------------------------------------------------- availability


def test_breaker_open_half_open_backoff() -> None:
    clock = [0.0]
    t = AvailabilityTracker(
        BreakerSettings(failure_threshold=2, open_seconds=10, max_open_seconds=30),
        clock=lambda: clock[0],
    )
    t.record_transport_failure("a", "connect_error")
    assert t.breaker_state("a") is BreakerState.CLOSED
    t.record_transport_failure("a", "connect_error")
    assert t.breaker_state("a") is BreakerState.OPEN and not t.admits("a")
    clock[0] = 10.5
    assert t.breaker_state("a") is BreakerState.HALF_OPEN and t.admits("a")
    t.begin("a")
    assert not t.admits("a")  # exactly one trial
    t.end("a")
    t.record_transport_failure("a", "connect_error")  # trial failed -> OPEN, doubled
    assert t.breaker_state("a") is BreakerState.OPEN
    clock[0] = 10.5 + 19
    assert t.breaker_state("a") is BreakerState.OPEN
    clock[0] = 10.5 + 21
    assert t.breaker_state("a") is BreakerState.HALF_OPEN
    t.record_success("a")
    assert t.breaker_state("a") is BreakerState.CLOSED


def test_open_breaker_excludes_node_and_reports_reason() -> None:
    r = reg(n("a"), n("b"))
    for _ in range(3):
        r.availability.record_transport_failure("a", "connect_error")
    ev = evaluate(rq(), r)
    assert ev.excluded == {"a": "CIRCUIT_OPEN"} and [c.node_id for c in ev.candidates] == ["b"]
    for _ in range(3):
        r.availability.record_transport_failure("b", "connect_error")
    with pytest.raises(RoutingError) as exc:
        evaluate(rq(), r)
    assert exc.value.code is PoolErrorCode.NODE_UNAVAILABLE


def test_saturation_is_soft_exclusion() -> None:
    r = reg(n("a"), n("b"))
    r.availability.record_saturation("a", "embeddings", 5)
    ev = evaluate(rq(), r)
    assert [c.node_id for c in ev.candidates] == ["b"] and ev.excluded == {"a": "SATURATED"}
    r.availability.record_saturation("b", "embeddings", 5)
    assert len(evaluate(rq(), r).candidates) == 2  # nobody better: the Gateway decides


def test_4xx_and_503_do_not_open_breaker() -> None:
    t = AvailabilityTracker(BreakerSettings(failure_threshold=1))
    t.record_saturation("a", "chat", 3)
    t.record_success("a")  # any HTTP answer, e.g. 400 or 503 resource_saturated
    assert t.breaker_state("a") is BreakerState.CLOSED


def test_max_inflight_cap() -> None:
    r = reg(n("a", max_inflight=1), n("b"))
    r.availability.begin("a")
    assert evaluate(rq(), r).excluded == {"a": "AT_CAPACITY"}


def test_exclude_nodes_for_failover_reselection() -> None:
    r = reg(n("a"), n("b"))
    ev = evaluate(rq(exclude_nodes=frozenset({"a"})), r)
    assert ev.excluded == {"a": "ALREADY_TRIED"}


def test_no_cross_environment_even_with_failover() -> None:
    r = reg(n("a"))
    with pytest.raises(RoutingError) as exc:
        evaluate(
            RoutingRequest(
                environment=Environment.PRODUCTION, service="embeddings", capability=EMB
            ),
            r,
        )
    assert exc.value.code is PoolErrorCode.NO_ELIGIBLE_NODE


# --------------------------------------------------------------------- failover matrix


def test_failover_route_classes() -> None:
    allowed = {(e.method, e.path) for e in ENDPOINTS if failover_allowed(e)}
    assert allowed == {
        ("POST", p)
        for p in (
            "/v1/generate",
            "/v1/generate/stream",
            "/v1/chat",
            "/v1/chat/stream",
            "/v1/embeddings",
            "/v1/rerank",
            "/v1/vision/analyze",
            "/v1/audio/transcriptions",
        )
    }


@pytest.mark.parametrize(
    ("field", "a", "b"),
    [("dimension", 768, 1024), ("pipeline_id", "legacy-768@0a109f42", "legacy-768@ffffffff")],
)
def test_embedding_dimension_and_pipeline_must_agree_across_nodes(
    field: str, a: object, b: object
) -> None:
    def emb_node(nid: str, value: object) -> dict[str, Any]:
        return n(
            nid,
            capabilities=["EMBEDDINGS"],
            models=[{"model_id": "legacy-768", "capabilities": ["EMBEDDINGS"], field: value}],
        )

    with pytest.raises(ValidationError, match=f"{field} differs"):
        build_config(
            nodes=[emb_node("a", a), emb_node("b", b)], routing={"policy": "WEIGHTED_ROUND_ROBIN"}
        )


def test_allowlist_host_port_pins_the_port() -> None:
    """POOL-4: a staging pool allowed 203.0.113.95:8200 can never target :8100."""

    build_config(
        nodes=[node("a", url="http://203.0.113.95:8200")], upstream_allowlist=["203.0.113.95:8200"]
    )
    with pytest.raises(ValidationError, match=r"203\.0\.113\.95:8100 is not in upstream_allowlist"):
        build_config(
            nodes=[node("a", url="http://203.0.113.95:8100")],
            upstream_allowlist=["203.0.113.95:8200"],
        )
    with pytest.raises(ValidationError, match="invalid port"):
        build_config(
            nodes=[node("a", url="http://203.0.113.95:8200")],
            upstream_allowlist=["203.0.113.95:99999"],
        )
    build_config(
        nodes=[node("a", url="http://gw.invalid:8200")], upstream_allowlist=["gw.invalid:8200"]
    )


def test_untagged_ollama_reference_matches_latest_only() -> None:
    """POOL-4 live finding: RAGatm staging sends `nomic-embed-text` (untagged); the
    Gateway resolves it to `nomic-embed-text:latest` (same digest and vectors)."""

    models = [
        {
            "model_id": "nomic-embed-text:latest",
            "capabilities": ["EMBEDDINGS"],
            "dimension": 768,
            "revision": "0a109f42",
        }
    ]
    r = reg(n("a", models=models))
    observe(
        r,
        "a",
        ids={"nomic-embed-text:latest", "legacy-768"},
        caps={"nomic-embed-text:latest": {"embedding"}},
        digests={"nomic-embed-text:latest": "0a109f422b47e3a3"},
    )
    assert evaluate(rq(model="nomic-embed-text"), r).candidates[0].node_id == "a"
    assert evaluate(rq(model="nomic-embed-text:latest"), r).candidates[0].node_id == "a"
    for other in ("nomic-embed-text:v1.5", "nomic-embed", "legacy-768"):
        with pytest.raises(RoutingError) as exc:
            evaluate(rq(model=other), r)  # legacy-768 is not declared on this node
        assert exc.value.code is PoolErrorCode.MODEL_NOT_AVAILABLE


def test_untagged_reference_still_checks_identity() -> None:
    models = [
        {
            "model_id": "nomic-embed-text:latest",
            "capabilities": ["EMBEDDINGS"],
            "revision": "0a109f42",
        }
    ]
    r = reg(n("a", models=models))
    observe(r, "a", ids={"nomic-embed-text:latest"}, digests={"nomic-embed-text:latest": "ffff"})
    with pytest.raises(RoutingError) as exc:
        evaluate(rq(model="nomic-embed-text"), r)
    assert "MODEL_IDENTITY_MISMATCH" in exc.value.message


def test_canonical_model_id() -> None:
    from ai_gateway_pool.routing.policies import canonical_model_id

    assert canonical_model_id("nomic-embed-text") == "nomic-embed-text:latest"
    assert canonical_model_id("llama3.1:8b") == "llama3.1:8b"
    assert canonical_model_id("registry.local:5000/org/m") == "registry.local:5000/org/m:latest"
