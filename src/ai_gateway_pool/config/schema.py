"""YAML configuration schema of one pool environment.

One file == one environment. A file never describes staging and production at
the same time, which makes a cross-environment node structurally visible and
rejectable. Validation is strict: an invalid file is refused, it is NEVER
completed with production defaults.
"""

from __future__ import annotations

import ipaddress
import re
from enum import StrEnum
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from ai_gateway_pool.gateway.contracts import CONTRACT_CAPABILITIES
from ai_gateway_pool.gateway.health import SAFE_PROBE_PATHS
from ai_gateway_pool.nodes.models import Capability, Environment, NodeDescriptor
from ai_gateway_pool.routing.policies import (
    IMPLEMENTED_POLICIES,
    ModelSubstitution,
    RoutingPolicyName,
)
from ai_gateway_pool.transport.contracts import DEFAULT_TIMEOUTS, TimeoutProfile, TrafficClass


class ConfigStatus(StrEnum):
    EXAMPLE = "example"  # template, refused when POOL_REQUIRE_APPROVED_CONFIG=1
    APPROVED = "approved"  # reviewed for a target deployment


def validate_base_url(url: str) -> str:
    """Accept only ``http(s)://host[:port]`` — no credentials, path, query or fragment."""

    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise ValueError(f"base_url scheme must be http or https: {url!r}")
    if not parts.hostname:
        raise ValueError(f"base_url has no host: {url!r}")
    if parts.username or parts.password:
        raise ValueError("base_url must not embed credentials")
    if parts.path not in {"", "/"} or parts.query or parts.fragment:
        raise ValueError(f"base_url must not carry a path, query or fragment: {url!r}")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError(f"base_url has an invalid port: {url!r}") from exc
    if port == 0:
        raise ValueError(f"base_url has an invalid port: {url!r}")
    return f"{parts.scheme}://{parts.netloc}"


class NodeConfig(NodeDescriptor):
    @field_validator("base_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        return validate_base_url(value)

    @model_validator(mode="after")
    def _instance_matches_environment(self) -> NodeConfig:
        outside = self.capabilities - CONTRACT_CAPABILITIES
        if outside:
            raise ValueError(
                f"node {self.node_id}: capabilities {sorted(outside)} have no route in the "
                "Open Gateway contract (gateway-api/1.0) and cannot be declared"
            )
        if self.gateway_instance != self.environment:
            raise ValueError(
                f"node {self.node_id}: gateway_instance={self.gateway_instance} differs from "
                f"environment={self.environment} (cross-environment wiring is forbidden)"
            )
        ids = [m.model_id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError(f"node {self.node_id}: a model_id is declared twice")
        for model in self.models:
            extra = model.capabilities - self.capabilities
            if extra:
                raise ValueError(
                    f"node {self.node_id}: model {model.model_id} declares capabilities "
                    f"{sorted(extra)} that the node does not declare"
                )
        return self


class RoutingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: RoutingPolicyName
    model_substitution: ModelSubstitution = ModelSubstitution.FORBIDDEN

    @field_validator("policy")
    @classmethod
    def _implemented(cls, value: RoutingPolicyName) -> RoutingPolicyName:
        if value not in IMPLEMENTED_POLICIES:
            raise ValueError(
                f"ROUTING_NOT_READY: policy {value} is defined but not implemented in this "
                f"version (implemented: {sorted(IMPLEMENTED_POLICIES)})"
            )
        return value


class HealthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe_path: str = "/health/runtime"
    # Periodic probing only ever calls the probe path + /v1/models (both
    # non-inferential). Off by default; startup probing is a separate switch.
    probe_on_startup: bool = False
    background_probing: bool = False
    probe_interval_seconds: Annotated[float, Field(ge=10, le=3600)] = 30
    # Observed inventories older than this are STALE: they keep RESTRICTING routing
    # (last observed set) but never widen it back to the declared list.
    capability_ttl_seconds: Annotated[float, Field(ge=10, le=86400)] = 300

    @field_validator("probe_path")
    @classmethod
    def _safe_probe(cls, value: str) -> str:
        if value not in SAFE_PROBE_PATHS:
            raise ValueError(
                f"probe_path {value!r} is not a known non-inferential endpoint "
                f"(allowed: {sorted(SAFE_PROBE_PATHS)})"
            )
        return value


class JobAffinityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # "none": ASR job routes are refused (no affinity => no async job).
    # "memory": NOT durable, development environment only.
    # "sqlite": durable, local file — development and qualification only.
    # "redis": durable, target backend for deployments (POOL-3 qualification).
    backend: Literal["none", "memory", "sqlite", "redis"] = "none"
    sqlite_path: str | None = None
    # NAME of the env var holding the Redis URL — never the URL itself in YAML.
    redis_url_env: Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")] | None = (
        None
    )
    redis_key_prefix: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")] = (
        "aigp"
    )
    redis_timeout_seconds: Annotated[float, Field(gt=0, le=10)] = 2.0
    ttl_seconds: Annotated[int, Field(ge=60, le=7 * 86400)] = 86400

    @model_validator(mode="after")
    def _backend_settings(self) -> JobAffinityConfig:
        if self.backend == "redis" and not self.redis_url_env:
            raise ValueError("job_affinity.backend=redis requires redis_url_env")
        if self.backend == "sqlite" and not self.sqlite_path:
            raise ValueError("job_affinity.backend=sqlite requires sqlite_path")
        return self


class CircuitBreakerConfig(BaseModel):
    """Process-local breaker on TRANSPORT failures only (see nodes/availability.py)."""

    model_config = ConfigDict(extra="forbid")

    failure_threshold: Annotated[int, Field(ge=1, le=100)] = 3
    open_seconds: Annotated[float, Field(ge=1, le=600)] = 15
    max_open_seconds: Annotated[float, Field(ge=1, le=3600)] = 300
    saturation_max_seconds: Annotated[float, Field(ge=0, le=300)] = 30

    @model_validator(mode="after")
    def _bounds(self) -> CircuitBreakerConfig:
        if self.max_open_seconds < self.open_seconds:
            raise ValueError("circuit_breaker.max_open_seconds must be >= open_seconds")
        return self


class FailoverConfig(BaseModel):
    """Controlled failover: only when the request provably never left the pool
    (connection refused / connect timeout / no pool connection), to another
    compatible node of the SAME environment. Never after bytes were sent."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_extra_nodes: Annotated[int, Field(ge=0, le=5)] = 1


class TransportConfig(BaseModel):
    """Outbound HTTP client limits (one shared connection pool per process)."""

    model_config = ConfigDict(extra="forbid")

    max_connections: Annotated[int, Field(ge=1, le=1000)] = 100
    max_keepalive_connections: Annotated[int, Field(ge=0, le=1000)] = 20
    # Below the 5 s keep-alive of uvicorn (AI Gateway): never reuse a socket the
    # server is about to close.
    keepalive_expiry_seconds: Annotated[float, Field(gt=0, le=300)] = 4
    # AIGP-HARDEN-1: every JSON body of a model-bearing route is read (bounded,
    # whatever its framing: Content-Length or chunked) to validate the requested
    # `model` BEFORE routing, then forwarded byte-for-byte. A larger body is
    # refused (413), never relayed uninspected. AI Gateway vision images are
    # <= ~5.6 MB of base64 (RAGatm caps them there too).
    json_inspect_max_bytes: Annotated[int, Field(ge=0, le=64 * 1024 * 1024)] = 16 * 1024 * 1024
    # Aggregate bytes of JSON bodies held for inspection by in-flight requests of
    # this process; beyond it new JSON requests get 503 INSPECTION_BUDGET_EXHAUSTED.
    json_inspect_budget_bytes: Annotated[int, Field(ge=0, le=1024 * 1024 * 1024)] = 96 * 1024 * 1024
    # Bytes per chunk when relaying (memory bound per in-flight request).
    stream_chunk_bytes: Annotated[int, Field(ge=4096, le=4 * 1024 * 1024)] = 64 * 1024

    @model_validator(mode="after")
    def _budget_holds_one_max_body(self) -> TransportConfig:
        if self.json_inspect_budget_bytes < self.json_inspect_max_bytes:
            raise ValueError(
                "transport.json_inspect_budget_bytes must be >= json_inspect_max_bytes "
                "(otherwise a maximum-size JSON body could never be inspected)"
            )
        return self


class TimeoutsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    health_probe: TimeoutProfile = DEFAULT_TIMEOUTS[TrafficClass.HEALTH_PROBE]
    inference: TimeoutProfile = DEFAULT_TIMEOUTS[TrafficClass.INFERENCE]
    streaming: TimeoutProfile = DEFAULT_TIMEOUTS[TrafficClass.STREAMING]
    upload: TimeoutProfile = DEFAULT_TIMEOUTS[TrafficClass.UPLOAD]
    job_poll: TimeoutProfile = DEFAULT_TIMEOUTS[TrafficClass.JOB_POLL]

    def for_class(self, traffic: TrafficClass) -> TimeoutProfile:
        profile: TimeoutProfile = getattr(self, traffic.value)
        return profile


class PoolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    status: ConfigStatus
    environment: Environment
    routing: RoutingConfig
    nodes: list[NodeConfig] = Field(min_length=1)
    health: HealthConfig = HealthConfig()
    timeouts: TimeoutsConfig = TimeoutsConfig()
    job_affinity: JobAffinityConfig = JobAffinityConfig()
    transport: TransportConfig = TransportConfig()
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()
    failover: FailoverConfig = FailoverConfig()
    # Hosts (exact names / IPs) or CIDR networks every node URL must belong to.
    # Mandatory in an approved staging/production configuration.
    upstream_allowlist: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coherence(self) -> PoolConfig:
        seen: set[str] = set()
        urls: set[str] = set()
        for node in self.nodes:
            if node.node_id in seen:
                raise ValueError(f"duplicate node_id: {node.node_id}")
            seen.add(node.node_id)
            if node.environment != self.environment:
                raise ValueError(
                    f"node {node.node_id} belongs to {node.environment} but this pool is "
                    f"{self.environment}: cross-environment nodes are forbidden"
                )
            if node.base_url in urls:
                raise ValueError(f"base_url declared twice: node {node.node_id}")
            urls.add(node.base_url)

        self._check_allowlist()
        self._check_model_identities()

        backend = self.job_affinity.backend
        if backend == "memory" and self.environment is not Environment.DEVELOPMENT:
            raise ValueError("job_affinity.backend=memory is not durable: development only")
        if (
            backend == "sqlite"
            and self.environment is not Environment.DEVELOPMENT
            and self.status is ConfigStatus.APPROVED
        ):
            raise ValueError(
                "job_affinity.backend=sqlite is for development/qualification only and can "
                "never be part of an approved staging/production configuration"
            )

        if self.status is ConfigStatus.APPROVED and self.environment is not Environment.DEVELOPMENT:
            if not self.upstream_allowlist:
                raise ValueError("an approved staging/production config needs upstream_allowlist")
            if self.job_affinity.backend not in {"redis", "none"}:
                raise ValueError(
                    "an approved staging/production config needs job_affinity.backend "
                    "redis (durable, shared) or none (ASR disabled)"
                )

        enabled = [n for n in self.nodes if n.enabled]
        if not enabled:
            raise ValueError("no enabled node: the pool would have no authorized destination")
        if self.routing.policy is RoutingPolicyName.SINGLE_NODE and len(enabled) != 1:
            raise ValueError(
                f"routing.policy=SINGLE_NODE requires exactly one enabled node, got {len(enabled)}"
            )
        return self

    def _check_allowlist(self) -> None:
        """Every node URL must match an entry: exact hostname, IP, CIDR, or
        ``host:port`` / ``ip:port`` (POOL-4: pins the PORT too, e.g. a staging
        pool allowed ``gateway-staging:8200`` can never point at ``:8100``)."""

        if not self.upstream_allowlist:
            return
        Net = ipaddress.IPv4Network | ipaddress.IPv6Network
        rules: list[tuple[Net | str, int | None]] = []
        for entry in self.upstream_allowlist:
            port: int | None = None
            target = entry.strip()
            m = re.fullmatch(r"([^:\[\]]+):(\d{1,5})", target)
            if m:
                target, port = m.group(1), int(m.group(2))
                if not 0 < port < 65536:
                    raise ValueError(f"upstream_allowlist: invalid port in {entry!r}")
            try:
                rules.append((ipaddress.ip_network(target, strict=False), port))
            except ValueError:
                rules.append((target.lower(), port))  # exact hostname
        for node in self.nodes:
            parts = urlsplit(node.base_url)
            host = (parts.hostname or "").lower()
            node_port = parts.port or (443 if parts.scheme == "https" else 80)
            try:
                ip = ipaddress.ip_address(host)
            except ValueError:
                ip = None
            ok = any(
                (port is None or port == node_port)
                and (
                    (isinstance(n, str) and n == host)
                    or (ip is not None and not isinstance(n, str) and ip in n)
                )
                for n, port in rules
            )
            if not ok:
                raise ValueError(
                    f"node {node.node_id}: upstream {host}:{node_port} is not in upstream_allowlist"
                )

    def _check_model_identities(self) -> None:
        """The same embedding model id on several nodes must be the same artifact:
        otherwise two requests for "legacy-768" could land in different vector
        spaces. Pinned identity fields must agree (revision, dimension, pipeline)."""

        seen: dict[str, tuple[str, tuple[str | None, int | None, str | None]]] = {}
        for node in self.nodes:
            for m in node.models:
                if Capability.EMBEDDINGS not in m.capabilities:
                    continue
                ident = (m.revision, m.dimension, m.pipeline_id)
                if m.model_id in seen:
                    other_node, other = seen[m.model_id]
                    for a, b, field_name in zip(
                        ident, other, ("revision", "dimension", "pipeline_id"), strict=True
                    ):
                        if a is not None and b is not None and a != b:
                            raise ValueError(
                                f"embedding model {m.model_id!r}: {field_name} differs between "
                                f"{other_node} and {node.node_id} (incompatible vector spaces)"
                            )
                else:
                    seen[m.model_id] = (node.node_id, ident)
            for cap, default in node.default_models.items():
                if cap not in node.capabilities:
                    raise ValueError(
                        f"node {node.node_id}: default model for undeclared capability {cap}"
                    )
                if not any(m.model_id == default and cap in m.capabilities for m in node.models):
                    raise ValueError(
                        f"node {node.node_id}: default {cap} model {default!r} is not declared "
                        "with that capability"
                    )


def check_environment_pair(staging: PoolConfig, production: PoolConfig) -> None:
    """Cross-file guard: staging and production must never share an upstream."""

    if staging.environment is not Environment.STAGING:
        raise ValueError("first config is not a staging config")
    if production.environment is not Environment.PRODUCTION:
        raise ValueError("second config is not a production config")
    shared = {n.base_url for n in staging.nodes} & {n.base_url for n in production.nodes}
    if shared:
        raise ValueError(f"staging and production share upstream(s): {sorted(shared)}")
    shared_ids = {n.node_id for n in staging.nodes} & {n.node_id for n in production.nodes}
    if shared_ids:
        raise ValueError(f"staging and production share node_id(s): {sorted(shared_ids)}")
