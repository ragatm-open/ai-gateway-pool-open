"""Routing policy vocabulary and model-substitution rules.

Only the policies listed in :data:`IMPLEMENTED_POLICIES` are executable in this
version. The others are *named* so that configuration and documentation can
refer to them, but a configuration selecting them is rejected at startup with
``ROUTING_NOT_READY`` instead of silently degrading.
"""

from __future__ import annotations

from enum import StrEnum

from ai_gateway_pool.nodes.models import Capability, ModelDescriptor, ModelRequirement


class RoutingPolicyName(StrEnum):
    SINGLE_NODE = "SINGLE_NODE"
    TARGETED_NODE = "TARGETED_NODE"
    WEIGHTED_ROUND_ROBIN = "WEIGHTED_ROUND_ROBIN"
    # In-flight requests of THIS router process, aggregated per physical host.
    LEAST_INFLIGHT = "LEAST_INFLIGHT"
    # Would need GPU load telemetry the Gateway does not expose: not implemented.
    LEAST_BUSY = "LEAST_BUSY"
    CAPABILITY_AWARE = "CAPABILITY_AWARE"


IMPLEMENTED_POLICIES: frozenset[RoutingPolicyName] = frozenset(
    {
        RoutingPolicyName.SINGLE_NODE,
        RoutingPolicyName.WEIGHTED_ROUND_ROBIN,
        RoutingPolicyName.LEAST_INFLIGHT,
    }
)


class ModelSubstitution(StrEnum):
    """Whether a request for model X may be served by model Y.

    Only ``FORBIDDEN`` exists for now: the requested ``model_id`` must be served
    exactly. The enum exists so that a future, explicit and audited alias table
    can be introduced without changing the call sites — never by default.
    """

    FORBIDDEN = "FORBIDDEN"


def canonical_model_id(model_id: str) -> str:
    """Ollama model REFERENCE normalisation: an untagged name means ``:latest``.

    ``nomic-embed-text`` and ``nomic-embed-text:latest`` are the same reference
    (verified live on AI Gateway staging, POOL-4: same model_logical, digest,
    pipeline_id and identical vectors). This is NOT an alias table: a different
    tag (``:v1.5``) never matches ``:latest``, and the request body is forwarded
    unchanged. Governed logical ids (``legacy-768``) are normalised the same way
    on both sides, so exact matching between them is preserved.
    """

    name = model_id.rsplit("/", 1)[-1]
    return model_id if ":" in name else f"{model_id}:latest"


def models_compatible(
    requested: ModelRequirement, capability: Capability, candidate: ModelDescriptor
) -> bool:
    """Strict identity check used for every capability, and critical for embeddings.

    * identifiers must be equal (no alias, no "same family" shortcut);
    * the candidate must serve the requested capability;
    * for embeddings, a pinned revision or dimension must match — but an equal
      dimension NEVER makes two different models compatible: a different
      ``model_id`` is rejected first.
    """

    if canonical_model_id(requested.model_id) != canonical_model_id(candidate.model_id):
        return False
    if capability not in candidate.capabilities:
        return False
    if capability is Capability.EMBEDDINGS:
        if requested.revision and requested.revision != candidate.revision:
            return False
        if requested.dimension and requested.dimension != candidate.dimension:
            return False
    return True
