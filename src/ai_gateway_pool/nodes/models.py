"""Typed model of an AI Gateway node.

Two layers are deliberately separated:

* :class:`NodeDescriptor` — *static*, comes from configuration, immutable.
  What an operator **declared** (URL, environment, declared models...).
* :class:`NodeState` — *runtime*, observed by probes. What has actually been
  **seen** (health, capability availability, observed model identities).

A declared capability is never treated as available until a probe observed it.

``physical_host_id`` and ``node_id`` are distinct: ``gpu-host-a-production`` and
``gpu-host-a-staging`` would be two logical nodes on ONE physical host that share
the same GPU. They must never be counted as two independent GPUs.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Identifier = Annotated[
    str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$", strip_whitespace=False)
]


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Capability(StrEnum):
    GENERATE = "GENERATE"
    CHAT = "CHAT"
    EMBEDDINGS = "EMBEDDINGS"
    RERANK = "RERANK"
    VISION = "VISION"
    # Historical capability with NO route in gateway-api/1.0 (Open edition): kept in
    # the vocabulary for error messages and tests, refused in node declarations.
    STRUCTURED_EXTRACT = "STRUCTURED_EXTRACT"
    ASR = "ASR"


class CapabilityState(StrEnum):
    DECLARED = "DECLARED"  # configured, not yet observed
    AVAILABLE = "AVAILABLE"  # observed available by a probe
    UNAVAILABLE = "UNAVAILABLE"  # observed unavailable
    UNKNOWN = "UNKNOWN"  # probe could not tell


class HealthState(StrEnum):
    UNKNOWN = "UNKNOWN"
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


class Reachability(StrEnum):
    """Network-level view, distinct from availability: a node can be REACHABLE
    (it answered HTTP) yet UNAVAILABLE (wrong environment, invalid payload)."""

    UNKNOWN = "UNKNOWN"
    REACHABLE = "REACHABLE"
    UNREACHABLE = "UNREACHABLE"


class ModelDescriptor(BaseModel):
    """Identity of a model as exposed by the AI Gateway.

    ``model_id`` is the exact identifier the Gateway accepts (e.g. the Ollama tag).
    For embeddings, ``revision`` + ``dimension`` pin what a vector space depends on (model
    identity + revision + dimension). Two embedding models are NEVER
    interchangeable because they share a dimension.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    capabilities: frozenset[Capability] = Field(min_length=1)
    revision: str | None = Field(
        default=None, description="Digest / revision if known (e.g. Ollama digest)."
    )
    dimension: int | None = Field(default=None, gt=0, description="Embeddings only.")
    pipeline_id: str | None = Field(
        default=None,
        description="Embeddings only: encoding pipeline identity (e.g. legacy-768@0a109f42).",
    )


class ModelRequirement(BaseModel):
    """The model identity a request needs (exact id, optionally pinned revision/dimension)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    revision: str | None = None
    dimension: int | None = Field(default=None, gt=0)


class NodeDescriptor(BaseModel):
    """Static description of one AI Gateway instance (one logical node)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: Identifier
    physical_host_id: Identifier
    display_name: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    environment: Environment
    gateway_instance: Environment = Field(
        description="Which AI Gateway instance this is. Must equal `environment`."
    )
    base_url: str
    enabled: bool = True
    weight: Annotated[int, Field(ge=0, le=1000)] = 100
    capabilities: frozenset[Capability] = frozenset()
    models: tuple[ModelDescriptor, ...] = ()
    # Model the Gateway uses when a request carries no `model`, per capability.
    # Needed to route implicit-model requests between several nodes (POOL-3).
    default_models: dict[Capability, str] = Field(default_factory=dict)
    # Optional per-node cap on requests in flight FROM THIS ROUTER PROCESS.
    max_inflight: Annotated[int, Field(ge=1, le=10000)] | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    def declares_model(self, model_id: str) -> bool:
        return any(m.model_id == model_id for m in self.models)


class NodeState(BaseModel):
    """Runtime, observed state of one node. Starts fully UNKNOWN."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    health_state: HealthState = HealthState.UNKNOWN
    reachability: Reachability = Reachability.UNKNOWN
    build_git_sha: str | None = None
    last_checked_at: datetime | None = None
    last_error: str | None = None
    capability_states: dict[Capability, CapabilityState] = Field(default_factory=dict)
    observed_model_ids: frozenset[str] | None = None  # None = never observed
    # model id -> raw Gateway capability names ("completion", "chat", "vision",
    # "embedding"...), exactly as published by /health/runtime and /v1/models.
    observed_model_caps: dict[str, frozenset[str]] = Field(default_factory=dict)
    # model id -> artifact digest published by /v1/models (concrete and logical ids).
    observed_digests: dict[str, str] = Field(default_factory=dict)
    # capability -> default model published by the Gateway (rerank, asr).
    observed_defaults: dict[Capability, str] = Field(default_factory=dict)
    observed_at: datetime | None = None
    in_flight: int = 0

    def capability_state(self, capability: Capability, declared: bool) -> CapabilityState:
        if capability in self.capability_states:
            return self.capability_states[capability]
        return CapabilityState.DECLARED if declared else CapabilityState.UNKNOWN
