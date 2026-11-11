"""Node registry: the ONLY source of upstream destinations.

The registry is built from a validated :class:`PoolConfig` and is scoped to a
single environment. Callers can only ever reference a node by ``node_id``; there
is no API that accepts a URL from the outside, so the pool cannot be turned into
an arbitrary HTTP proxy.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from ai_gateway_pool.config.schema import PoolConfig
from ai_gateway_pool.errors import PoolError, PoolErrorCode
from ai_gateway_pool.nodes.availability import AvailabilityTracker, BreakerSettings
from ai_gateway_pool.nodes.models import (
    Capability,
    CapabilityState,
    Environment,
    HealthState,
    NodeDescriptor,
    NodeState,
    Reachability,
)


class NodeRegistry:
    def __init__(
        self,
        environment: Environment,
        nodes: Iterable[NodeDescriptor],
        availability: AvailabilityTracker | None = None,
        capability_ttl_seconds: float = 300.0,
    ) -> None:
        self._environment = environment
        # Process-local breaker / saturation / in-flight signals (see availability.py).
        self.availability = availability or AvailabilityTracker()
        self.capability_ttl_seconds = capability_ttl_seconds
        self._nodes: dict[str, NodeDescriptor] = {}
        self._states: dict[str, NodeState] = {}
        for node in nodes:
            if node.node_id in self._nodes:
                raise ValueError(f"duplicate node_id: {node.node_id}")
            if node.environment != environment:
                raise ValueError(
                    f"node {node.node_id} ({node.environment}) cannot join a {environment} registry"
                )
            self._nodes[node.node_id] = node
            self._states[node.node_id] = NodeState(node_id=node.node_id)

    @classmethod
    def from_config(cls, config: PoolConfig) -> NodeRegistry:
        cb = config.circuit_breaker
        return cls(
            config.environment,
            config.nodes,
            AvailabilityTracker(
                BreakerSettings(
                    failure_threshold=cb.failure_threshold,
                    open_seconds=cb.open_seconds,
                    max_open_seconds=cb.max_open_seconds,
                    saturation_max_seconds=cb.saturation_max_seconds,
                )
            ),
            capability_ttl_seconds=config.health.capability_ttl_seconds,
        )

    def inventory_freshness(self, node_id: str) -> str:
        """NEVER_OBSERVED | KNOWN | STALE (observation older than the TTL).

        A stale inventory still RESTRICTS routing (last observed set); it never
        widens it back to the declared list.
        """

        state = self.state(node_id)
        if state.observed_at is None or state.observed_model_ids is None:
            return "NEVER_OBSERVED"
        age = (datetime.now(UTC) - state.observed_at).total_seconds()
        return "KNOWN" if age <= self.capability_ttl_seconds else "STALE"

    # ------------------------------------------------------------------ queries

    @property
    def environment(self) -> Environment:
        return self._environment

    def all(self) -> list[NodeDescriptor]:
        return list(self._nodes.values())

    def enabled(self) -> list[NodeDescriptor]:
        return [n for n in self._nodes.values() if n.enabled]

    def get(self, node_id: str) -> NodeDescriptor:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise PoolError(
                PoolErrorCode.INVALID_NODE_DESTINATION, f"unknown node_id {node_id!r}"
            ) from None

    def state(self, node_id: str) -> NodeState:
        self.get(node_id)
        return self._states[node_id]

    def physical_hosts(self) -> dict[str, list[str]]:
        """physical_host_id -> logical node ids. Several nodes on one host share its GPU."""

        hosts: dict[str, list[str]] = {}
        for node in self._nodes.values():
            hosts.setdefault(node.physical_host_id, []).append(node.node_id)
        return hosts

    def resolve_destination(self, node_id: str) -> str:
        """The base URL of a configured, ENABLED node. The only way to get a URL."""

        node = self.get(node_id)
        if not node.enabled:
            raise PoolError(PoolErrorCode.INVALID_NODE_DESTINATION, f"node {node_id} is disabled")
        return node.base_url

    def capability_view(self, node_id: str, capability: Capability) -> CapabilityState:
        node = self.get(node_id)
        state = self._states[node_id]
        if state.health_state is HealthState.UNAVAILABLE:
            return CapabilityState.UNAVAILABLE
        return state.capability_state(capability, declared=capability in node.capabilities)

    # ------------------------------------------------------------------ updates

    def update_state(
        self,
        node_id: str,
        *,
        health: HealthState,
        capability_states: dict[Capability, CapabilityState] | None = None,
        observed_model_ids: frozenset[str] | None = None,
        error: str | None = None,
        checked_at: datetime | None = None,
        reachability: Reachability | None = None,
        build_git_sha: str | None = None,
        observed_model_caps: dict[str, frozenset[str]] | None = None,
        observed_digests: dict[str, str] | None = None,
        observed_defaults: dict[Capability, str] | None = None,
    ) -> NodeState:
        current = self.state(node_id)
        new = current.model_copy(
            update={
                "health_state": health,
                "last_checked_at": checked_at or datetime.now(UTC),
                "last_error": error,
                "reachability": reachability or current.reachability,
                "observed_model_caps": observed_model_caps
                if observed_model_caps is not None
                else current.observed_model_caps,
                "observed_digests": observed_digests
                if observed_digests is not None
                else current.observed_digests,
                "observed_defaults": observed_defaults
                if observed_defaults is not None
                else current.observed_defaults,
                "observed_at": (checked_at or datetime.now(UTC))
                if observed_model_ids is not None
                else current.observed_at,
                "build_git_sha": build_git_sha or current.build_git_sha,
                "capability_states": (
                    capability_states
                    if capability_states is not None
                    else current.capability_states
                ),
                "observed_model_ids": (
                    observed_model_ids
                    if observed_model_ids is not None
                    else current.observed_model_ids
                ),
            }
        )
        self._states[node_id] = new
        return new
