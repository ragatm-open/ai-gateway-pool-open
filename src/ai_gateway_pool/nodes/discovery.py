"""Bounded, on-demand node discovery.

One call = at most two GET requests per node (``/health/runtime`` then
``/v1/models``), both non-inferential, with the HEALTH_PROBE timeout profile.
Triggered by ``POST /internal/nodes/{node_id}/probe``, optionally once at
startup and/or periodically (``health.background_probing``). ``/health/ready``
— which computes an embedding — is never used.
"""

from __future__ import annotations

import httpx

from ai_gateway_pool.gateway.health import INVENTORY_PATH, interpret_inventory, interpret_runtime
from ai_gateway_pool.nodes.models import HealthState, NodeState, Reachability
from ai_gateway_pool.nodes.registry import NodeRegistry
from ai_gateway_pool.transport.contracts import TimeoutProfile


async def probe_node(
    registry: NodeRegistry,
    node_id: str,
    client: httpx.AsyncClient,
    *,
    probe_path: str,
    timeouts: TimeoutProfile,
) -> NodeState:
    base_url = registry.resolve_destination(node_id)  # configured nodes only
    node = registry.get(node_id)
    timeout = timeouts.to_httpx()
    try:
        resp = await client.get(base_url + probe_path, timeout=timeout)
    except httpx.TimeoutException:
        return registry.update_state(
            node_id,
            health=HealthState.UNAVAILABLE,
            error="probe_timeout",
            reachability=Reachability.UNREACHABLE,
        )
    except httpx.HTTPError:
        return registry.update_state(
            node_id,
            health=HealthState.UNAVAILABLE,
            error="probe_connect_error",
            reachability=Reachability.UNREACHABLE,
        )
    # Any HTTP answer proves transport reachability (the breaker only counts
    # transport failures): a HALF_OPEN breaker can recover without client traffic.
    registry.availability.record_probe_success(node_id)
    if resp.status_code != 200:
        return registry.update_state(
            node_id,
            health=HealthState.UNAVAILABLE,
            error=f"probe_http_{resp.status_code}",
            reachability=Reachability.REACHABLE,
        )
    try:
        payload = resp.json()
    except ValueError:
        payload = None
    obs = interpret_runtime(payload, node.environment)
    if obs.health is not HealthState.HEALTHY:
        return registry.update_state(
            node_id, health=obs.health, error=obs.error, reachability=Reachability.REACHABLE
        )

    observed = set(obs.model_ids)
    health = HealthState.HEALTHY
    error: str | None = None
    try:
        inv = await client.get(base_url + INVENTORY_PATH, timeout=timeout)
        inventory = interpret_inventory(inv.json()) if inv.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        inventory = None
    if inventory is None:
        # Runtime is fine but the inventory is unknown: do not restrict models.
        health, error = HealthState.DEGRADED, "inventory_unavailable"
        observed_ids: frozenset[str] | None = None
    else:
        observed_ids = frozenset(observed | inventory.model_ids)
    return registry.update_state(
        node_id,
        health=health,
        capability_states=dict(obs.capability_states),
        observed_model_ids=observed_ids,
        error=error,
        reachability=Reachability.REACHABLE,
        build_git_sha=obs.build_git_sha,
        # Runtime capabilities of governed ids take precedence over /v1/models.
        observed_model_caps=(
            {**inventory.model_caps, **obs.model_caps} if inventory is not None else None
        ),
        observed_digests=inventory.digests if inventory is not None else None,
        observed_defaults=dict(obs.defaults),
    )


async def probe_all(
    registry: NodeRegistry,
    client: httpx.AsyncClient,
    *,
    probe_path: str,
    timeouts: TimeoutProfile,
) -> None:
    for node in registry.enabled():
        await probe_node(registry, node.node_id, client, probe_path=probe_path, timeouts=timeouts)
