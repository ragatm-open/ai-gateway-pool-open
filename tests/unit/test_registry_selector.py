"""Registry and selection (brief §20 'Registre' and 'Sélection')."""

from __future__ import annotations

import pytest

from ai_gateway_pool.errors import PoolError, PoolErrorCode, RoutingError
from ai_gateway_pool.nodes.models import (
    Capability,
    CapabilityState,
    Environment,
    HealthState,
    ModelRequirement,
)
from ai_gateway_pool.nodes.registry import NodeRegistry
from ai_gateway_pool.routing.contracts import RoutingRequest
from ai_gateway_pool.routing.policies import RoutingPolicyName
from ai_gateway_pool.routing.selector import (
    SingleNodeSelector,
    WeightedRoundRobinSelector,
    build_selector,
)
from tests.conftest import build_config, node

STG = Environment.STAGING


def req(
    cap: Capability = Capability.EMBEDDINGS,
    model: str | None = "legacy-768",
    env: Environment = STG,
    **kw: object,
) -> RoutingRequest:
    return RoutingRequest(
        environment=env,
        service=cap.value.lower(),
        capability=cap,
        model=ModelRequirement(model_id=model, **kw) if model else None,  # type: ignore[arg-type]
    )


def registry(**kw: object) -> NodeRegistry:
    return NodeRegistry.from_config(build_config(**kw))  # type: ignore[arg-type]


def multi_registry(*nodes: dict[str, object]) -> NodeRegistry:
    return registry(nodes=list(nodes), routing={"policy": "WEIGHTED_ROUND_ROBIN"})


# --------------------------------------------------------------------- registry


def test_single_node_registry() -> None:
    reg = registry()
    assert [n.node_id for n in reg.all()] == ["gw-staging"]
    assert reg.environment is STG
    assert reg.state("gw-staging").health_state is HealthState.UNKNOWN


def test_registry_supports_multiple_nodes_and_physical_hosts() -> None:
    reg = multi_registry(
        node("a", url="http://a.invalid:1", host="h1"),
        node("b", url="http://b.invalid:1", host="h1"),
        node("c", url="http://c.invalid:1", host="h2"),
    )
    assert reg.physical_hosts() == {"h1": ["a", "b"], "h2": ["c"]}


def test_registry_rejects_duplicates_and_foreign_env_directly() -> None:
    cfg = build_config()
    n = cfg.nodes[0]
    with pytest.raises(ValueError, match="duplicate"):
        NodeRegistry(STG, [n, n])
    with pytest.raises(ValueError, match="cannot join"):
        NodeRegistry(Environment.PRODUCTION, [n])


def test_disabled_node_excluded() -> None:
    reg = multi_registry(
        node("a", url="http://a.invalid:1", enabled=False),
        node("b", url="http://b.invalid:1"),
    )
    assert [n.node_id for n in reg.enabled()] == ["b"]
    decision = WeightedRoundRobinSelector().select(req(), reg)
    assert decision.node_id == "b"
    assert decision.candidates == ("b",)
    with pytest.raises(PoolError) as exc:
        reg.resolve_destination("a")
    assert exc.value.code is PoolErrorCode.INVALID_NODE_DESTINATION


def test_unknown_node_destination_denied() -> None:
    with pytest.raises(PoolError) as exc:
        registry().resolve_destination("http://evil.example")
    assert exc.value.code is PoolErrorCode.INVALID_NODE_DESTINATION


def test_declared_capability_is_not_reported_available() -> None:
    reg = registry()
    assert reg.capability_view("gw-staging", Capability.ASR) is CapabilityState.DECLARED
    assert reg.capability_view("gw-staging", Capability.VISION) is CapabilityState.UNKNOWN


def test_unknown_capability_safe_failure() -> None:
    with pytest.raises(RoutingError) as exc:
        SingleNodeSelector().select(req(Capability.VISION, None), registry())
    assert exc.value.code is PoolErrorCode.CAPABILITY_NOT_AVAILABLE


def test_no_compatible_model_safe_failure() -> None:
    with pytest.raises(RoutingError) as exc:
        SingleNodeSelector().select(req(model="bge-m3"), registry())
    assert exc.value.code is PoolErrorCode.MODEL_NOT_AVAILABLE


def test_no_compatible_node_safe_failure() -> None:
    reg = registry()
    reg.update_state("gw-staging", health=HealthState.UNAVAILABLE, error="probe_timeout")
    with pytest.raises(RoutingError) as exc:
        SingleNodeSelector().select(req(), reg)
    assert exc.value.code is PoolErrorCode.NODE_UNAVAILABLE


# --------------------------------------------------------------------- selection


def test_single_node_selection() -> None:
    decision = SingleNodeSelector().select(req(), registry())
    assert decision.node_id == "gw-staging"
    assert decision.physical_host_id == "gpu-host-a"
    assert decision.policy is RoutingPolicyName.SINGLE_NODE


def test_target_environment_isolation() -> None:
    with pytest.raises(RoutingError) as exc:
        SingleNodeSelector().select(req(env=Environment.PRODUCTION), registry())
    assert exc.value.code is PoolErrorCode.NO_ELIGIBLE_NODE
    assert "cross-environment" in exc.value.message


def test_model_compatibility_required_same_dimension_is_not_enough() -> None:
    """Critical for RAGatm: fr-en-retrieval and legacy-768 are both 768-d."""

    reg = registry()
    with pytest.raises(RoutingError) as exc:
        SingleNodeSelector().select(req(model="fr-en-retrieval", dimension=768), reg)
    assert exc.value.code is PoolErrorCode.MODEL_NOT_AVAILABLE
    # Exact identity + matching dimension passes; a wrong pinned dimension fails.
    assert SingleNodeSelector().select(req(dimension=768), reg).node_id == "gw-staging"
    with pytest.raises(RoutingError):
        SingleNodeSelector().select(req(dimension=1024), reg)


def test_embedding_revision_pin_enforced() -> None:
    n = node(
        models=[
            {
                "model_id": "legacy-768",
                "capabilities": ["EMBEDDINGS"],
                "revision": "0a109f42",
                "dimension": 768,
            }
        ],
        capabilities=["EMBEDDINGS"],
    )
    reg = registry(nodes=[n])
    assert SingleNodeSelector().select(req(revision="0a109f42"), reg)
    with pytest.raises(RoutingError) as exc:
        SingleNodeSelector().select(req(revision="ffffffff"), reg)
    assert exc.value.code is PoolErrorCode.MODEL_NOT_AVAILABLE


def test_model_must_serve_requested_capability() -> None:
    with pytest.raises(RoutingError) as exc:
        SingleNodeSelector().select(req(Capability.CHAT, "legacy-768"), registry())
    assert exc.value.code is PoolErrorCode.MODEL_NOT_AVAILABLE


def test_observed_inventory_overrides_declaration() -> None:
    reg = registry()
    reg.update_state(
        "gw-staging", health=HealthState.HEALTHY, observed_model_ids=frozenset({"llama-chat"})
    )
    with pytest.raises(RoutingError) as exc:
        SingleNodeSelector().select(req(), reg)  # legacy-768 declared but not observed
    assert exc.value.code is PoolErrorCode.MODEL_NOT_AVAILABLE


def test_unavailable_node_excluded() -> None:
    reg = multi_registry(node("a", url="http://a.invalid:1"), node("b", url="http://b.invalid:1"))
    reg.update_state("a", health=HealthState.UNAVAILABLE)
    sel = WeightedRoundRobinSelector()
    assert {sel.select(req(), reg).node_id for _ in range(5)} == {"b"}


def test_capability_observed_unavailable_excluded() -> None:
    reg = registry()
    reg.update_state(
        "gw-staging",
        health=HealthState.HEALTHY,
        capability_states={Capability.ASR: CapabilityState.UNAVAILABLE},
    )
    with pytest.raises(RoutingError) as exc:
        SingleNodeSelector().select(req(Capability.ASR, "asr-whisper-turbo"), reg)
    assert exc.value.code is PoolErrorCode.CAPABILITY_NOT_AVAILABLE


def test_no_cross_env_fallback_when_production_is_down() -> None:
    prod = registry(env="production", nodes=[node("gw-prod", env="production")])
    prod.update_state("gw-prod", health=HealthState.UNAVAILABLE)
    with pytest.raises(RoutingError) as exc:
        SingleNodeSelector().select(req(env=Environment.PRODUCTION), prod)
    # Reported as a PRODUCTION unavailability, never redirected.
    assert exc.value.code is PoolErrorCode.NODE_UNAVAILABLE
    assert "gw-prod" in exc.value.message


def test_targeted_node_by_id_only() -> None:
    reg = multi_registry(node("a", url="http://a.invalid:1"), node("b", url="http://b.invalid:1"))
    sel = WeightedRoundRobinSelector()
    pinned = req().model_copy(update={"target_node_id": "b"})
    assert {sel.select(pinned, reg).node_id for _ in range(4)} == {"b"}
    with pytest.raises(RoutingError) as exc:
        sel.select(req().model_copy(update={"target_node_id": "http://evil.example"}), reg)
    assert exc.value.code is PoolErrorCode.INVALID_NODE_DESTINATION
    # A pinned node that is down is NOT replaced by another one.
    reg.update_state("b", health=HealthState.UNAVAILABLE)
    with pytest.raises(RoutingError) as exc:
        sel.select(pinned, reg)
    assert exc.value.code is PoolErrorCode.NODE_UNAVAILABLE


def test_weighted_round_robin_is_deterministic_and_weighted() -> None:
    reg = multi_registry(
        node("a", url="http://a.invalid:1", weight=300),
        node("b", url="http://b.invalid:1", weight=100),
        node("z", url="http://z.invalid:1", weight=0),
    )
    sel = WeightedRoundRobinSelector()
    picks = [sel.select(req(), reg).node_id for _ in range(8)]
    assert picks == ["a", "a", "b", "a", "a", "a", "b", "a"]
    assert "z" not in picks


@pytest.mark.parametrize(
    "policy",
    [
        RoutingPolicyName.LEAST_BUSY,
        RoutingPolicyName.CAPABILITY_AWARE,
        RoutingPolicyName.TARGETED_NODE,
    ],
)
def test_unimplemented_selectors_not_ready(policy: RoutingPolicyName) -> None:
    with pytest.raises(RoutingError) as exc:
        build_selector(policy)
    assert exc.value.code is PoolErrorCode.ROUTING_NOT_READY
