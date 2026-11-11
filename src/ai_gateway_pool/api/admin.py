"""Internal administrative routes (``/internal/*``) — never public.

Two layers of protection:

* network: the router is only attached to internal Docker networks / loopback
  (see ``docs/security.md``); nginx never exposes it;
* application: a bearer token (``POOL_ADMIN_TOKEN``) compared in constant time.
  When no token is configured the routes do not exist (404).
"""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from ai_gateway_pool.api.context import PoolContext, get_context
from ai_gateway_pool.errors import PoolError, PoolErrorCode
from ai_gateway_pool.gateway.contracts import BodyKind, match_endpoint
from ai_gateway_pool.nodes.discovery import probe_node
from ai_gateway_pool.nodes.models import Capability, ModelRequirement
from ai_gateway_pool.routing.contracts import RoutingRequest
from ai_gateway_pool.routing.selector import evaluate
from ai_gateway_pool.transport.contracts import TrafficClass
from ai_gateway_pool.transport.proxy import failover_allowed

router = APIRouter(prefix="/internal", tags=["pool-internal"])


def require_admin(
    authorization: str | None = Header(default=None),
    ctx: PoolContext = Depends(get_context),  # noqa: B008
) -> PoolContext:
    token = ctx.settings.admin_token
    if token is None:
        raise HTTPException(status_code=404, detail="Not Found")
    expected = token.get_secret_value()
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:]
    if not expected or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="unauthorized")
    return ctx


def _node_view(ctx: PoolContext, node_id: str) -> dict[str, Any]:
    node = ctx.registry.get(node_id)
    state = ctx.registry.state(node_id)
    return {
        "node_id": node.node_id,
        "physical_host_id": node.physical_host_id,
        "display_name": node.display_name,
        "environment": node.environment.value,
        "gateway_instance": node.gateway_instance.value,
        "base_url": node.base_url,
        "enabled": node.enabled,
        "weight": node.weight,
        "health_state": state.health_state.value,
        "reachability": state.reachability.value,
        "build_git_sha": state.build_git_sha,
        "inventory": ctx.registry.inventory_freshness(node_id),
        "max_inflight": node.max_inflight,
        "availability_this_process": ctx.registry.availability.snapshot(node_id),
        "last_checked_at": state.last_checked_at.isoformat() if state.last_checked_at else None,
        "last_error": state.last_error,
        "metadata": dict(node.metadata),
    }


@router.get("/nodes")
async def nodes(ctx: PoolContext = Depends(require_admin)) -> dict[str, Any]:  # noqa: B008
    hosts = ctx.registry.physical_hosts()
    return {
        "environment": ctx.registry.environment.value,
        "routing_policy": ctx.config.routing.policy.value,
        "nodes": [_node_view(ctx, n.node_id) for n in ctx.registry.all()],
        "physical_hosts": [
            {"physical_host_id": h, "node_ids": ids, "shared_gpu_between_nodes": len(ids) > 1}
            for h, ids in hosts.items()
        ],
    }


@router.get("/capabilities")
async def capabilities(ctx: PoolContext = Depends(require_admin)) -> dict[str, Any]:  # noqa: B008
    out = []
    for node in ctx.registry.all():
        state = ctx.registry.state(node.node_id)
        out.append(
            {
                "node_id": node.node_id,
                "capabilities": {
                    cap.value: ctx.registry.capability_view(node.node_id, cap).value
                    for cap in Capability
                    if cap in node.capabilities or cap in state.capability_states
                },
                "declared_models": [m.model_dump(exclude_none=True) for m in node.models],
                "observed_model_ids": (
                    sorted(state.observed_model_ids)
                    if state.observed_model_ids is not None
                    else None
                ),
            }
        )
    return {"environment": ctx.registry.environment.value, "nodes": out}


@router.post("/nodes/{node_id}/probe")
async def probe(node_id: str, ctx: PoolContext = Depends(require_admin)) -> dict[str, Any]:  # noqa: B008
    """Bounded on-demand probe (2 non-inferential GETs). No background loop."""

    ctx.registry.get(node_id)  # unknown id -> INVALID_NODE_DESTINATION, no outbound call
    await probe_node(
        ctx.registry,
        node_id,
        ctx.client,
        probe_path=ctx.config.health.probe_path,
        timeouts=ctx.config.timeouts.health_probe,
    )
    return _node_view(ctx, node_id)


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics(ctx: PoolContext = Depends(require_admin)) -> PlainTextResponse:  # noqa: B008
    return PlainTextResponse(ctx.metrics.render(), media_type="text/plain; version=0.0.4")


class ExplainRequest(BaseModel):
    method: str = "POST"
    path: str
    model: str | None = None
    target_node_id: str | None = None


@router.post("/routing/explain")
async def explain(
    req: ExplainRequest,
    ctx: PoolContext = Depends(require_admin),  # noqa: B008
) -> dict[str, Any]:
    """Explain the routing of a hypothetical request WITHOUT sending anything and
    without advancing any policy state (no user payload involved)."""

    endpoint = match_endpoint(req.method, req.path)
    if endpoint is None:
        raise PoolError(PoolErrorCode.ROUTE_NOT_IN_CONTRACT, "route not in the AI Gateway contract")
    diagnostic = endpoint.traffic is TrafficClass.HEALTH_PROBE
    rr = RoutingRequest(
        environment=ctx.config.environment,
        service=endpoint.service,
        capability=None if diagnostic else endpoint.capability,
        model=ModelRequirement(model_id=req.model) if req.model and not diagnostic else None,
        target_node_id=req.target_node_id,
        model_uninspectable=endpoint.body is BodyKind.MULTIPART,
    )
    try:
        ev = evaluate(rr, ctx.registry)
    except PoolError as exc:
        return {"eligible": False, "error": exc.code.value, "message": exc.message}
    return {
        "eligible": True,
        "policy": ctx.config.routing.policy.value,
        "candidates": [n.node_id for n in ev.candidates],
        "excluded": ev.excluded,
        "inventory": ev.inventory,
        "implicit_model": ev.implicit_model,
        "failover_allowed_for_route": failover_allowed(endpoint),
    }


@router.get("/runtime")
async def runtime_identity(ctx: PoolContext = Depends(require_admin)) -> dict[str, Any]:  # noqa: B008
    """Identity of THIS running pool process (deployment verification)."""

    import os
    from pathlib import Path

    from ai_gateway_pool import __version__

    store_status: str | None = None
    if ctx.job_store is not None:
        try:
            await ctx.job_store.ping()
            store_status = "ok"
        except PoolError:
            store_status = "unavailable"
    rev_file = Path("/BUILD_REVISION")
    return {
        "version": __version__,
        "build_sha": os.environ.get("APP_GIT_SHA")
        or (rev_file.read_text().strip() if rev_file.exists() else None),
        "environment": ctx.config.environment.value,
        "config_status": ctx.config.status.value,
        "routing_policy": ctx.config.routing.policy.value,
        "workers": 1,  # uvicorn.run(): single process by construction
        "nodes_enabled": [n.node_id for n in ctx.registry.enabled()],
        "job_affinity_backend": ctx.config.job_affinity.backend,
        "job_affinity_store": store_status,
    }
