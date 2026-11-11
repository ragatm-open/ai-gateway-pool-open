"""Per-application context shared by the API routers and the relay."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
from fastapi import Request

from ai_gateway_pool.config.schema import PoolConfig
from ai_gateway_pool.config.settings import Settings
from ai_gateway_pool.jobs.contracts import JobAffinityStore
from ai_gateway_pool.nodes.registry import NodeRegistry
from ai_gateway_pool.observability.metrics import MetricsRegistry
from ai_gateway_pool.routing.selector import NodeSelector


def default_client_factory(config: PoolConfig) -> Callable[[], httpx.AsyncClient]:
    """One shared outbound client per process.

    ``trust_env=False``: environment proxies are ignored. ``follow_redirects=False``:
    an upstream redirect can never move the destination off the registry.
    """

    t = config.transport

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(
                max_connections=t.max_connections,
                max_keepalive_connections=t.max_keepalive_connections,
                keepalive_expiry=t.keepalive_expiry_seconds,
            ),
            timeout=config.timeouts.inference.to_httpx(),
        )

    return factory


@dataclass
class PoolContext:
    settings: Settings
    config: PoolConfig
    registry: NodeRegistry
    selector: NodeSelector
    logger: logging.Logger
    job_store: JobAffinityStore | None = None
    metrics: MetricsRegistry = field(default_factory=MetricsRegistry)
    # Outbound client factory (tests inject a fake transport).
    client_factory: Callable[[], httpx.AsyncClient] | None = None
    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            factory = self.client_factory or default_client_factory(self.config)
            self._client = factory()
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def admin_token(self) -> str | None:
        token = self.settings.admin_token
        return token.get_secret_value() if token is not None else None


def get_context(request: Request) -> PoolContext:
    ctx: PoolContext = request.app.state.pool
    return ctx
