"""Application factory and entrypoint.

POOL-2 scope: single-node transparent relay of the 21 AI Gateway routes
(``/health*`` diagnostics, ``/v1/*``), plus the pool's own routes
(``/health/live``, ``/pool/health/*``, protected ``/internal/*``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ai_gateway_pool import __version__
from ai_gateway_pool.api import admin, health
from ai_gateway_pool.api.context import PoolContext
from ai_gateway_pool.config.schema import PoolConfig
from ai_gateway_pool.config.settings import Settings, load_for_settings, load_settings
from ai_gateway_pool.errors import PoolError, PoolErrorCode
from ai_gateway_pool.gateway.contracts import match_endpoint
from ai_gateway_pool.jobs.factory import build_job_store
from ai_gateway_pool.nodes.discovery import probe_all
from ai_gateway_pool.nodes.registry import NodeRegistry
from ai_gateway_pool.observability.logging import configure_logging, log_event
from ai_gateway_pool.routing.selector import build_selector
from ai_gateway_pool.transport.contracts import contains_upstream_override
from ai_gateway_pool.transport.proxy import (
    GatewayRelay,
    RelayInfo,
    pool_error_response,
)

_ALLOWED_RID = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _request_id(headers: list[tuple[bytes, bytes]]) -> str:
    for k, v in headers:
        if k.lower() == b"x-request-id":
            supplied = v.decode("latin-1")
            if 0 < len(supplied) <= 128 and set(supplied) <= _ALLOWED_RID:
                return supplied
            break
    return uuid.uuid4().hex


def error_response(code: PoolErrorCode, message: str, request_id: str, status: int) -> JSONResponse:
    del status  # derived from the code (kept for backwards compatibility)
    return pool_error_response(code, message, request_id)


class RequestContextMiddleware:
    """Pure ASGI middleware (no body buffering, streaming- and disconnect-safe):
    request id, refusal of destination-steering headers, one access log line."""

    def __init__(self, app: ASGIApp, ctx: PoolContext) -> None:
        self.app = app
        self.ctx = ctx

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        raw_headers: list[tuple[bytes, bytes]] = list(scope.get("headers", []))
        rid = _request_id(raw_headers)
        scope.setdefault("state", {})["request_id"] = rid
        start = time.perf_counter()
        status_holder: dict[str, Any] = {"status": 0, "bytes": 0, "error": None}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", rid.encode()))
                for k, v in headers:
                    if k.lower() == b"x-aigp-error":
                        status_holder["error"] = v.decode("latin-1")
                message["headers"] = headers
            elif message["type"] == "http.response.body":
                status_holder["bytes"] += len(message.get("body", b""))
            await send(message)

        try:
            names = [k.decode("latin-1") for k, _ in raw_headers]
            lowered = [n.lower() for n in names]
            if "content-length" in lowered and "transfer-encoding" in lowered:
                # RFC 9112 §6.1: a message with both is a smuggling vector; refuse it.
                resp = pool_error_response(
                    PoolErrorCode.INVALID_REQUEST_FRAMING,
                    "Content-Length and Transfer-Encoding are mutually exclusive",
                    rid,
                )
                await resp(scope, receive, send_wrapper)
            elif contains_upstream_override(names):
                resp = pool_error_response(
                    PoolErrorCode.UPSTREAM_OVERRIDE_DENIED,
                    "upstream destination headers are not accepted",
                    rid,
                )
                await resp(scope, receive, send_wrapper)
            else:
                await self.app(scope, receive, send_wrapper)
        finally:
            info: RelayInfo | None = scope["state"].get("relay")
            route = scope.get("route")
            log_event(
                self.ctx.logger,
                logging.INFO,
                "request",
                event="http_request",
                request_id=rid,
                environment=self.ctx.config.environment.value,
                method=scope.get("method"),
                route=getattr(route, "path", "unmatched") if info is None else info.service,
                response_status=status_holder["status"],
                http_status=status_holder["status"],
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                error_category=status_holder["error"],
                selected_node_id=info.node_id if info else None,
                capability=info.capability if info else None,
                model_id=info.model_id if info else None,
                upstream_latency_ms=info.upstream_latency_ms if info else None,
                streaming=info.streaming if info else None,
                job_affinity=info.job_affinity if info else None,
                response_bytes=status_holder["bytes"],
                physical_host_id=info.physical_host_id if info else None,
                routing_policy=info.routing_policy if info else None,
                candidate_count=info.candidate_count if info else None,
                routing_decision=info.routing_decision if info else None,
                failover_attempted=info.failover_attempted if info else None,
                failover_allowed=info.failover_allowed if info else None,
                failover_result=info.failover_result if info else None,
                affinity_backend=self.ctx.config.job_affinity.backend if info else None,
            )


def create_app(settings: Settings | None = None, config: PoolConfig | None = None) -> FastAPI:
    settings = settings or load_settings()
    config = config or load_for_settings(settings)
    logger = configure_logging(settings.log_level)
    registry = NodeRegistry.from_config(config)
    ctx = PoolContext(
        settings=settings,
        config=config,
        registry=registry,
        selector=build_selector(config.routing.policy),
        logger=logger,
        job_store=build_job_store(config),
    )
    relay = GatewayRelay(ctx)

    async def probe_loop() -> None:
        while True:
            await asyncio.sleep(config.health.probe_interval_seconds)
            with contextlib.suppress(Exception):
                await probe_all(
                    registry,
                    ctx.client,
                    probe_path=config.health.probe_path,
                    timeouts=config.timeouts.health_probe,
                )

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task: asyncio.Task[None] | None = None
        if config.health.probe_on_startup:
            with contextlib.suppress(Exception):
                await probe_all(
                    registry,
                    ctx.client,
                    probe_path=config.health.probe_path,
                    timeouts=config.timeouts.health_probe,
                )
        if config.health.background_probing:
            task = asyncio.create_task(probe_loop())
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await ctx.aclose()

    app = FastAPI(
        title="AI Gateway Pool",
        version=__version__,
        docs_url=None,  # no public API docs
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.pool = ctx
    app.include_router(health.router)
    app.include_router(admin.router)
    app.add_middleware(RequestContextMiddleware, ctx=ctx)

    @app.exception_handler(PoolError)
    async def pool_error(request: Request, exc: PoolError) -> JSONResponse:
        rid = getattr(request.state, "request_id", uuid.uuid4().hex)
        return pool_error_response(exc.code, exc.message, rid)

    async def gateway_route(request: Request) -> Response:
        endpoint = match_endpoint(request.method, request.url.path)
        if endpoint is None:
            return pool_error_response(
                PoolErrorCode.ROUTE_NOT_IN_CONTRACT,
                "route not in the AI Gateway contract",
                request.state.request_id,
            )
        return await relay.handle(request, endpoint)

    methods = ["GET", "POST", "DELETE", "PUT", "PATCH", "HEAD", "OPTIONS"]
    app.add_api_route("/v1/{path:path}", gateway_route, methods=methods, include_in_schema=False)
    app.add_api_route("/health", gateway_route, methods=methods, include_in_schema=False)
    app.add_api_route(
        "/health/{path:path}", gateway_route, methods=methods, include_in_schema=False
    )

    log_event(
        logger,
        logging.INFO,
        "pool started",
        event="startup",
        environment=config.environment.value,
        routing_policy=config.routing.policy.value,
    )
    return app


def run() -> None:  # pragma: no cover - exercised by scripts/check-local.sh
    import uvicorn

    settings = load_settings()
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.bind_host,
        port=settings.bind_port,
        log_config=None,
        access_log=False,
        proxy_headers=False,
        server_header=False,
        date_header=False,  # the upstream Date header is relayed instead
        timeout_graceful_shutdown=30,
    )
