"""Transport-independent routing request/decision types."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ai_gateway_pool.nodes.models import Capability, Environment, ModelRequirement
from ai_gateway_pool.routing.policies import RoutingPolicyName


class RoutingRequest(BaseModel):
    """What a request needs. Built by the pool from the matched Gateway route.

    ``target_node_id`` lets a client pin a *configured* node by id (never a URL).
    ``model`` is the requested model identity; for embeddings it may pin the
    revision/dimension the client's vector space was built with.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: Environment
    service: str
    # None = diagnostic / inventory route (/health, /health/runtime, /v1/models):
    # any enabled node of the environment may answer, whatever its observed state.
    capability: Capability | None
    model: ModelRequirement | None = None
    target_node_id: str | None = None
    # The body carries a model the pool cannot read (ASR multipart): candidates
    # must then serve identical model sets for the capability.
    model_uninspectable: bool = False
    # Nodes already tried for THIS request (controlled failover re-selection).
    exclude_nodes: frozenset[str] = frozenset()


class RoutingDecision(BaseModel):
    """Explainable decision: who was chosen, who was eligible, who was excluded and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    physical_host_id: str
    policy: RoutingPolicyName
    candidates: tuple[str, ...]
    excluded: tuple[tuple[str, str], ...] = ()
    inventory: tuple[tuple[str, str], ...] = ()  # node -> KNOWN | STALE | NEVER_OBSERVED
    implicit_model: str | None = None
