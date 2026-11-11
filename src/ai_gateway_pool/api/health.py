"""The POOL's own health routes (never relayed).

* ``GET /health/live`` — pool liveness (AI Gateway has no ``/health/live``).
* ``GET /pool/health/live`` / ``GET /pool/health/ready`` — pool namespace.

``/health/ready`` and every other ``/health/*`` route of the AI Gateway contract
are RELAYED to the node (POOL-2): the pool must not shadow a Gateway route. Pool
readiness therefore lives under ``/pool/health/ready``.

They reveal no URL, IP or model. Readiness = valid configuration, at least one
enabled node, and no enabled node known UNAVAILABLE by its last probe.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ai_gateway_pool.api.context import PoolContext, get_context
from ai_gateway_pool.errors import PoolError
from ai_gateway_pool.nodes.models import HealthState

router = APIRouter(tags=["pool-health"])

PROXY_STATUS = "single_node"


@router.get("/health/live")
@router.get("/pool/health/live")
async def live() -> dict[str, str]:
    return {"status": "live"}


@router.get("/pool/health/ready")
async def ready(ctx: PoolContext = Depends(get_context)) -> JSONResponse:  # noqa: B008
    enabled = ctx.registry.enabled()
    states = [ctx.registry.state(n.node_id).health_state for n in enabled]
    unavailable = sum(1 for s in states if s is HealthState.UNAVAILABLE)
    usable = len(enabled) - unavailable
    store_ok: bool | None = None
    if ctx.job_store is not None:
        # A durable store the environment requires is part of readiness.
        try:
            await ctx.job_store.ping()
            store_ok = True
        except PoolError:
            store_ok = False
    ready_ok = usable > 0 and store_ok is not False
    body: dict[str, Any] = {
        "status": "ready" if ready_ok else "degraded",
        "environment": ctx.config.environment.value,
        "nodes_enabled": len(enabled),
        "nodes_unavailable": unavailable,
        "routing_policy": ctx.config.routing.policy.value,
        "proxy": PROXY_STATUS if len(enabled) == 1 else "multi_node",
        "job_affinity": ctx.config.job_affinity.backend,
        "job_affinity_store": None if store_ok is None else ("ok" if store_ok else "unavailable"),
    }
    return JSONResponse(body, status_code=200 if ready_ok else 503)
