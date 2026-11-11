"""Node selection, independent of HTTP — explainable and deterministic.

Each node of the request's environment is evaluated in a fixed order; the first
failing check becomes its exclusion reason:

1. ``DISABLED`` / ``ALREADY_TRIED`` (failover re-selection);
2. ``CAPABILITY_NOT_DECLARED``;
3. model: ``MODEL_NOT_DECLARED`` (strict identity, :func:`models_compatible`),
   ``MODEL_NOT_OBSERVED`` (absent from the node's observed inventory, even if
   stale), ``CAPABILITY_NOT_OBSERVED`` (the node publishes the model WITHOUT the
   capability, e.g. ``llama-chat`` has no ``chat``), ``MODEL_IDENTITY_MISMATCH``
   (observed digest differs from the declared revision);
4. ``NODE_UNAVAILABLE`` (last probe), ``CAPABILITY_UNAVAILABLE`` (probe);
5. ``CIRCUIT_OPEN`` (process-local breaker), ``AT_CAPACITY`` (``max_inflight``).

Then, across the remaining candidates:

* the same model id must be the same artifact on every candidate (observed
  digests): otherwise the request is refused (``MODEL_NOT_AVAILABLE``) — never
  split between two vector spaces / weight files;
* implicit model (no ``model`` in the body) with several candidates: every
  candidate must publish the SAME default for the capability, else
  ``MODEL_REQUIRED`` — the pool never changes the meaning of a request;
* ``SATURATED`` (recent ``503`` + ``Retry-After``) is a SOFT exclusion: dropped
  only if a non-saturated candidate remains.

There is no fallback outside this set and never across environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ai_gateway_pool.errors import PoolError, PoolErrorCode, RoutingError
from ai_gateway_pool.nodes.models import (
    Capability,
    CapabilityState,
    HealthState,
    ModelDescriptor,
    NodeDescriptor,
)
from ai_gateway_pool.nodes.registry import NodeRegistry
from ai_gateway_pool.routing.contracts import RoutingDecision, RoutingRequest
from ai_gateway_pool.routing.policies import (
    RoutingPolicyName,
    canonical_model_id,
    models_compatible,
)

# Pool capability -> raw capability name published by the Gateway.
RAW_CAPABILITY: dict[Capability, str] = {
    Capability.GENERATE: "completion",
    Capability.CHAT: "chat",
    Capability.VISION: "vision",
    Capability.EMBEDDINGS: "embedding",
    Capability.RERANK: "rerank",
    Capability.ASR: "asr",
}

_STAGE = {
    "DISABLED": 0,
    "ALREADY_TRIED": 0,
    "CAPABILITY_NOT_DECLARED": 1,
    "MODEL_NOT_DECLARED": 2,
    "MODEL_NOT_OBSERVED": 2,
    "CAPABILITY_NOT_OBSERVED": 2,
    "MODEL_IDENTITY_MISMATCH": 2,
    "NODE_UNAVAILABLE": 3,
    "CAPABILITY_UNAVAILABLE": 3,
    "CIRCUIT_OPEN": 4,
    "AT_CAPACITY": 4,
}


@dataclass
class Evaluation:
    candidates: list[NodeDescriptor]
    excluded: dict[str, str] = field(default_factory=dict)
    inventory: dict[str, str] = field(default_factory=dict)
    implicit_model: str | None = None


def _declared(node: NodeDescriptor, model_id: str, cap: Capability) -> ModelDescriptor | None:
    want = canonical_model_id(model_id)
    for m in node.models:
        if canonical_model_id(m.model_id) == want and cap in m.capabilities:
            return m
    return None


def _observed_get[T](mapping: dict[str, T], model_id: str) -> T | None:
    if model_id in mapping:
        return mapping[model_id]
    want = canonical_model_id(model_id)
    for key, value in mapping.items():
        if canonical_model_id(key) == want:
            return value
    return None


def _node_reason(
    node: NodeDescriptor, request: RoutingRequest, registry: NodeRegistry
) -> str | None:
    if not node.enabled:
        return "DISABLED"
    if node.node_id in request.exclude_nodes:
        return "ALREADY_TRIED"
    cap = request.capability
    if cap is None:
        return None  # diagnostics: see evaluate()
    if cap not in node.capabilities:
        return "CAPABILITY_NOT_DECLARED"
    state = registry.state(node.node_id)
    if request.model is not None:
        if not any(models_compatible(request.model, cap, m) for m in node.models):
            return "MODEL_NOT_DECLARED"
        mid = request.model.model_id
        if state.observed_model_ids is not None:
            canon = canonical_model_id(mid)
            if mid not in state.observed_model_ids and not any(
                canonical_model_id(o) == canon for o in state.observed_model_ids
            ):
                return "MODEL_NOT_OBSERVED"
            raw = RAW_CAPABILITY.get(cap)
            observed_caps = _observed_get(state.observed_model_caps, mid)
            if raw and observed_caps is not None and raw not in observed_caps:
                return "CAPABILITY_NOT_OBSERVED"
            declared = _declared(node, mid, cap)
            digest = _observed_get(state.observed_digests, mid)
            if (
                declared is not None
                and declared.revision
                and digest
                and not digest.startswith(declared.revision)
            ):
                return "MODEL_IDENTITY_MISMATCH"
    if state.health_state is HealthState.UNAVAILABLE:
        return "NODE_UNAVAILABLE"
    if registry.capability_view(node.node_id, cap) is CapabilityState.UNAVAILABLE:
        return "CAPABILITY_UNAVAILABLE"
    if not registry.availability.admits(node.node_id):
        return "CIRCUIT_OPEN"
    if node.max_inflight is not None and (
        registry.availability.in_flight(node.node_id) >= node.max_inflight
    ):
        return "AT_CAPACITY"
    return None


def _raise_for(request: RoutingRequest, reasons: dict[str, str]) -> None:
    cap = request.capability
    values = set(reasons.values())
    stage = max((_STAGE.get(r, 0) for r in values), default=0)
    if stage <= 1 and "CAPABILITY_NOT_DECLARED" in values:
        raise RoutingError(
            PoolErrorCode.CAPABILITY_NOT_AVAILABLE, f"no node declares capability {cap}"
        )
    if stage <= 1:
        raise RoutingError(PoolErrorCode.NO_ELIGIBLE_NODE, "no enabled node in this pool")
    if stage == 2:
        model_id = request.model.model_id if request.model else "?"
        detail = ", ".join(f"{n}={r}" for n, r in sorted(reasons.items()))
        raise RoutingError(
            PoolErrorCode.MODEL_NOT_AVAILABLE,
            f"no node serves model {model_id!r} for {cap} (model substitution is forbidden; "
            f"{detail})",
        )
    at_stage3 = {n: r for n, r in reasons.items() if _STAGE.get(r) == 3}
    if stage == 3 and at_stage3 and all(r == "CAPABILITY_UNAVAILABLE" for r in at_stage3.values()):
        raise RoutingError(
            PoolErrorCode.CAPABILITY_NOT_AVAILABLE,
            f"capability {cap} is observed unavailable on every compatible node",
        )
    compatible = [n for n, r in reasons.items() if _STAGE.get(r, 0) >= 3]
    raise RoutingError(
        PoolErrorCode.NODE_UNAVAILABLE,
        "every compatible node is currently unavailable: "
        + ", ".join(f"{n} ({reasons[n]})" for n in sorted(compatible)),
    )


def evaluate(request: RoutingRequest, registry: NodeRegistry) -> Evaluation:
    """Candidates + exclusion reasons, or the most specific RoutingError."""

    if request.environment != registry.environment:
        raise RoutingError(
            PoolErrorCode.NO_ELIGIBLE_NODE,
            f"request environment {request.environment} cannot be served by the "
            f"{registry.environment} pool (no cross-environment routing)",
        )

    pool = registry.all()
    if request.target_node_id is not None:
        try:
            target = registry.get(request.target_node_id)
        except PoolError as exc:
            raise RoutingError(exc.code, exc.message) from None
        if not target.enabled:
            raise RoutingError(
                PoolErrorCode.INVALID_NODE_DESTINATION, f"node {target.node_id} is disabled"
            )
        pool = [target]

    reasons: dict[str, str] = {}
    candidates: list[NodeDescriptor] = []
    for node in pool:
        reason = _node_reason(node, request, registry)
        if reason is None:
            candidates.append(node)
        else:
            reasons[node.node_id] = reason
    inventory = {n.node_id: registry.inventory_freshness(n.node_id) for n in candidates}

    if request.capability is None:
        # Diagnostics are relayed even to a node observed UNAVAILABLE: the client
        # gets the node's real answer (or an explicit UPSTREAM_UNAVAILABLE).
        if not candidates:
            raise RoutingError(PoolErrorCode.NO_ELIGIBLE_NODE, "no enabled node in this pool")
        return Evaluation(candidates, reasons, inventory)

    if not candidates:
        _raise_for(request, reasons)

    cap = request.capability
    implicit: str | None = None
    if request.model is not None:
        digests = {
            _observed_get(registry.state(n.node_id).observed_digests, request.model.model_id)
            for n in candidates
        } - {None}
        if len(digests) > 1:
            raise RoutingError(
                PoolErrorCode.MODEL_NOT_AVAILABLE,
                f"model {request.model.model_id!r} resolves to different artifacts on the "
                f"candidate nodes {sorted(n.node_id for n in candidates)}: refused rather than "
                "split between incompatible models",
            )
    elif len(candidates) > 1:
        if request.model_uninspectable:
            sets = {
                frozenset(m.model_id for m in n.models if cap in m.capabilities) for n in candidates
            }
            if len(sets) > 1:
                raise RoutingError(
                    PoolErrorCode.MODEL_REQUIRED,
                    f"the {cap} model is not inspectable and candidate nodes serve different "
                    "model sets: restrict the environment to one node for this capability",
                )
        defaults = {
            n.default_models.get(cap) or registry.state(n.node_id).observed_defaults.get(cap)
            for n in candidates
        }
        if None in defaults or len(defaults) != 1:
            raise RoutingError(
                PoolErrorCode.MODEL_REQUIRED,
                f"no explicit model and the {len(candidates)} candidate nodes do not publish one "
                f"identical default {cap} model: send an explicit `model`",
            )
        implicit = next(iter(defaults))

    # Soft exclusion of nodes that recently answered 503 + Retry-After.
    service = request.service
    fresh = [n for n in candidates if not registry.availability.saturated(n.node_id, service)]
    if fresh and len(fresh) < len(candidates):
        for n in candidates:
            if n not in fresh:
                reasons[n.node_id] = "SATURATED"
        candidates = fresh
    return Evaluation(candidates, reasons, inventory, implicit)


def eligible_nodes(request: RoutingRequest, registry: NodeRegistry) -> list[NodeDescriptor]:
    """Backward-compatible helper (POOL-1/2 API): eligible nodes or RoutingError."""

    return evaluate(request, registry).candidates


class NodeSelector(Protocol):
    policy: RoutingPolicyName

    def select(self, request: RoutingRequest, registry: NodeRegistry) -> RoutingDecision: ...


def _decision(node: NodeDescriptor, policy: RoutingPolicyName, ev: Evaluation) -> RoutingDecision:
    return RoutingDecision(
        node_id=node.node_id,
        physical_host_id=node.physical_host_id,
        policy=policy,
        candidates=tuple(c.node_id for c in ev.candidates),
        excluded=tuple(sorted(ev.excluded.items())),
        inventory=tuple(sorted(ev.inventory.items())),
        implicit_model=ev.implicit_model,
    )


class SingleNodeSelector:
    """Reference / rollback policy: exactly one enabled node; use it if eligible."""

    policy = RoutingPolicyName.SINGLE_NODE

    def select(self, request: RoutingRequest, registry: NodeRegistry) -> RoutingDecision:
        if len(registry.enabled()) != 1:
            raise RoutingError(
                PoolErrorCode.CONFIGURATION_INVALID,
                "SINGLE_NODE policy requires exactly one enabled node",
            )
        ev = evaluate(request, registry)
        return _decision(ev.candidates[0], self.policy, ev)


class WeightedRoundRobinSelector:
    """Deterministic smooth weighted round-robin (nginx algorithm) over candidates.

    Weights shape the proportions among ELIGIBLE nodes only; this says nothing
    about GPU availability, and two nodes on one physical host are weighted
    independently (use LEAST_INFLIGHT to account for a shared host).
    """

    policy = RoutingPolicyName.WEIGHTED_ROUND_ROBIN

    def __init__(self) -> None:
        self._current: dict[str, int] = {}

    def select(self, request: RoutingRequest, registry: NodeRegistry) -> RoutingDecision:
        ev = evaluate(request, registry)
        candidates = [n for n in ev.candidates if n.weight > 0]
        if not candidates:
            raise RoutingError(PoolErrorCode.NO_ELIGIBLE_NODE, "all eligible nodes have weight 0")
        total = sum(n.weight for n in candidates)
        for n in candidates:
            self._current[n.node_id] = self._current.get(n.node_id, 0) + n.weight
        best = max(candidates, key=lambda n: (self._current[n.node_id], -candidates.index(n)))
        self._current[best.node_id] -= total
        ev.candidates = candidates
        return _decision(best, self.policy, ev)


class LeastInflightSelector:
    """Fewest requests in flight FROM THIS PROCESS, aggregated per physical host.

    Ordering key (lowest wins): in-flight on the node's physical host (several
    logical nodes share one GPU), then on the node, then higher weight, then
    configuration order. Counters are process-local: with several router
    processes each one balances its own traffic; this is not a global or GPU
    measurement.
    """

    policy = RoutingPolicyName.LEAST_INFLIGHT

    def select(self, request: RoutingRequest, registry: NodeRegistry) -> RoutingDecision:
        ev = evaluate(request, registry)
        candidates = [n for n in ev.candidates if n.weight > 0] or ev.candidates
        avail = registry.availability
        host_load: dict[str, int] = {}
        for n in registry.all():
            host_load[n.physical_host_id] = host_load.get(n.physical_host_id, 0) + (
                avail.in_flight(n.node_id)
            )
        order = {n.node_id: i for i, n in enumerate(registry.all())}
        best = min(
            candidates,
            key=lambda n: (
                host_load[n.physical_host_id],
                avail.in_flight(n.node_id),
                -n.weight,
                order[n.node_id],
            ),
        )
        ev.candidates = candidates
        return _decision(best, self.policy, ev)


def build_selector(policy: RoutingPolicyName) -> NodeSelector:
    if policy is RoutingPolicyName.SINGLE_NODE:
        return SingleNodeSelector()
    if policy is RoutingPolicyName.WEIGHTED_ROUND_ROBIN:
        return WeightedRoundRobinSelector()
    if policy is RoutingPolicyName.LEAST_INFLIGHT:
        return LeastInflightSelector()
    # TARGETED_NODE is a per-request option (target_node_id) honored by every
    # selector. LEAST_BUSY would need GPU load telemetry the Gateway does not
    # publish. CAPABILITY_AWARE is not a separate policy: capability / model /
    # identity eligibility is applied by EVERY policy (evaluate()).
    raise RoutingError(PoolErrorCode.ROUTING_NOT_READY, f"policy {policy} is not implemented")
