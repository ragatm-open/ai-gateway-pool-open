"""Single-node transparent relay of the AI Gateway contract.

Flow for one request::

    match route (catalogue) -> validate path -> plan body (stream | bounded JSON
    inspection for `model`) -> build RoutingRequest -> selector (registry) ->
    [ASR GET/DELETE: affinity lookup -> pinned node] -> httpx stream send ->
    relay status + headers + raw body bytes -> close upstream -> metrics/log

Invariants:

* the destination is ALWAYS ``registry.resolve_destination(selected_node)``;
* method, raw path and raw query string are forwarded verbatim; the body is
  forwarded byte-for-byte (streamed, or the exact bytes read for inspection);
* upstream statuses, error bodies and ``Retry-After`` are relayed unchanged; the
  pool only answers by itself for routing / transport / affinity failures, with
  an ``{"error": {...}}`` body and an ``X-AIGP-Error`` header (a Gateway error
  body is ``{"detail": ...}`` and never carries ``X-AIGP-Error``);
* no automatic retry of a POST; a GET is retried once, same node, only if no
  response byte was received;
* response bodies are streamed as they arrive (NDJSON never aggregated), and
  the upstream response is closed on success, error, timeout or disconnect.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import Request
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, Response, StreamingResponse

from ai_gateway_pool.api.context import PoolContext
from ai_gateway_pool.errors import PoolError, PoolErrorCode
from ai_gateway_pool.gateway.contracts import BodyKind, GatewayEndpoint, ResponseKind
from ai_gateway_pool.jobs.contracts import AffinityStoreUnavailable, valid_job_id
from ai_gateway_pool.nodes.models import ModelRequirement
from ai_gateway_pool.observability.logging import log_event
from ai_gateway_pool.routing.contracts import RoutingDecision, RoutingRequest
from ai_gateway_pool.transport.contracts import (
    TimeoutProfile,
    TrafficClass,
    forwardable_request_headers,
    forwardable_response_headers,
)

REQUEST_ID_HEADER = "X-Request-ID"
TARGET_NODE_HEADER = "x-aigp-target-node"
AFFINITY_HEADER = "X-AIGP-Affinity"
ASR_JOBS_PREFIX = "/v1/audio/transcriptions/"
_SMALL_BODY_LIMIT = 1024 * 1024  # ASR submit / delete answers are small JSON


@dataclass
class RelayInfo:
    """Per-request facts for the access log (identifiers and numbers only)."""

    service: str | None = None
    capability: str | None = None
    model_id: str | None = None
    node_id: str | None = None
    upstream_latency_ms: float | None = None
    streaming: bool = False
    job_affinity: str | None = None
    physical_host_id: str | None = None
    routing_policy: str | None = None
    candidate_count: int | None = None
    routing_decision: str | None = None
    failover_attempted: bool = False
    failover_allowed: bool | None = None
    failover_result: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def pool_error_response(
    code: PoolErrorCode, message: str, request_id: str, extra: dict[str, Any] | None = None
) -> JSONResponse:
    status = PoolError(code, message).http_status
    error: dict[str, Any] = {"code": code.value, "message": message, "request_id": request_id}
    if extra:
        error |= extra
    return JSONResponse(
        status_code=status,
        content={"error": error},
        headers={REQUEST_ID_HEADER: request_id, "X-AIGP-Error": code.value},
    )


async def _raw_chunks(resp: httpx.Response) -> AsyncIterator[bytes]:
    """Raw upstream bytes as they arrive. A response whose body was already loaded
    (in-memory transports only) is yielded as one chunk."""

    if resp.is_stream_consumed:
        yield resp.content
        return
    async for chunk in resp.aiter_raw():
        yield chunk


def _raw_path(request: Request) -> str:
    raw = request.scope.get("raw_path")
    if isinstance(raw, bytes | bytearray):
        return bytes(raw).decode("latin-1")
    return str(request.scope["path"])


def validate_raw_path(raw_path: str) -> str | None:
    """Return a reason if the path is ambiguous / unsafe to forward, else None.

    AI Gateway paths contain no percent-encoding, no empty or dot segments and no
    backslash: anything else could be normalised differently by the upstream.
    """

    if not raw_path.startswith("/"):
        return "path must be origin-form (absolute URLs are not accepted)"
    if "%" in raw_path or "\\" in raw_path:
        return "encoded or backslash characters are not accepted in the path"
    if "//" in raw_path:
        return "empty path segment"
    segments = raw_path.split("/")[1:]
    if any(seg in {".", ".."} for seg in segments):
        return "dot segment in path"
    return None


class GatewayRelay:
    def __init__(self, ctx: PoolContext) -> None:
        self.ctx = ctx
        self.env = ctx.config.environment.value
        # Bytes of JSON bodies currently held for inspection (single event loop:
        # plain integer arithmetic, no await between check and update).
        self._inspect_in_use = 0

    # ------------------------------------------------------------------ helpers

    def _fail(self, info: RelayInfo, code: PoolErrorCode, message: str, rid: str) -> JSONResponse:
        self.ctx.metrics.inc(
            "aigp_proxy_errors_total",
            environment=self.env,
            service=info.service or "unknown",
            error_category=code.value,
        )
        if code is PoolErrorCode.MODEL_NOT_AVAILABLE:
            self.ctx.metrics.inc(
                "aigp_proxy_model_unavailable_total",
                environment=self.env,
                service=info.service or "unknown",
            )
        return pool_error_response(code, message, rid)

    def _reserve(self, nbytes: int) -> None:
        """Account ``nbytes`` against the process-wide JSON inspection budget."""

        budget = self.ctx.config.transport.json_inspect_budget_bytes
        if self._inspect_in_use + nbytes > budget:
            raise PoolError(
                PoolErrorCode.INSPECTION_BUDGET_EXHAUSTED,
                "too many large JSON bodies are being inspected right now; retry later",
            )
        self._inspect_in_use += nbytes

    async def _plan_body(
        self, request: Request, endpoint: GatewayEndpoint, reservation: list[int]
    ) -> tuple[bytes | AsyncIterator[bytes] | None, str | None]:
        """Return (content to forward, validated requested model id or None).

        AIGP-HARDEN-1 invariant: a JSON body (every model-bearing route) is NEVER
        relayed before its ``model`` was established, whatever its HTTP framing.
        "Could not inspect" is a refusal, never "no model". Bytes held for the
        inspection are counted in ``reservation[0]`` (released by the caller).
        """

        headers = request.headers
        has_body = "content-length" in headers or "transfer-encoding" in headers
        if endpoint.body is not BodyKind.JSON:
            # Multipart ASR uploads (model inside a part) and body-less routes:
            # streamed unchanged, as before.
            return (request.stream() if has_body else None), None
        if not has_body:
            return None, None
        encoding = headers.get("content-encoding", "").strip().lower()
        if encoding not in ("", "identity"):
            raise PoolError(
                PoolErrorCode.UNSUPPORTED_CONTENT_ENCODING,
                "Content-Encoding is not accepted on JSON routes: the model must be inspectable",
            )
        limit = self.ctx.config.transport.json_inspect_max_bytes
        declared = headers.get("content-length")
        if declared is not None:
            if not declared.isdigit():
                raise PoolError(PoolErrorCode.INVALID_REQUEST_FRAMING, "invalid Content-Length")
            if int(declared) > limit:
                raise PoolError(
                    PoolErrorCode.REQUEST_BODY_TOO_LARGE,
                    f"JSON body larger than the inspection limit ({limit} bytes)",
                )
            self._reserve(int(declared))
            reservation[0] += int(declared)
        buf = bytearray()
        async for chunk in request.stream():
            if not chunk:
                continue
            if len(buf) + len(chunk) > limit:
                raise PoolError(
                    PoolErrorCode.REQUEST_BODY_TOO_LARGE,
                    f"JSON body larger than the inspection limit ({limit} bytes)",
                )
            if declared is None:
                self._reserve(len(chunk))
                reservation[0] += len(chunk)
            buf += chunk
        body = bytes(buf)
        del buf
        if not body:
            # An empty body cannot carry a model: same relay as before (the
            # Gateway answers 422 itself); nothing to govern.
            return body, None
        return body, _validated_model(body)

    def _routing_request(
        self, request: Request, endpoint: GatewayEndpoint, model: str | None, target: str | None
    ) -> RoutingRequest:
        diagnostic = endpoint.traffic is TrafficClass.HEALTH_PROBE
        return RoutingRequest(
            environment=self.ctx.config.environment,
            service=endpoint.service,
            capability=None if diagnostic else endpoint.capability,
            model=ModelRequirement(model_id=model) if model and not diagnostic else None,
            target_node_id=target,
            model_uninspectable=endpoint.body is BodyKind.MULTIPART,
        )

    # ------------------------------------------------------------------ entry

    async def handle(self, request: Request, endpoint: GatewayEndpoint) -> Response:
        rid: str = request.state.request_id
        info = RelayInfo(
            service=endpoint.service,
            capability=endpoint.capability.value if endpoint.capability else None,
        )
        request.state.relay = info
        raw_path = _raw_path(request)
        reason = validate_raw_path(raw_path)
        if reason:
            return self._fail(info, PoolErrorCode.INVALID_REQUEST_PATH, reason, rid)

        is_job_route = endpoint.async_job and endpoint.method in {"GET", "DELETE"}
        job_id = raw_path[len(ASR_JOBS_PREFIX) :] if is_job_route else None
        if job_id is not None and not valid_job_id(job_id):
            return self._fail(info, PoolErrorCode.INVALID_REQUEST_PATH, "invalid job id", rid)

        if endpoint.async_job and self.ctx.job_store is None:
            return self._fail(
                info,
                PoolErrorCode.ROUTING_NOT_READY,
                "ASR job routes need a durable job affinity backend (job_affinity.backend)",
                rid,
            )

        target = request.headers.get(TARGET_NODE_HEADER)
        if target is not None and not self._admin_authorized(request):
            # Privileged node selection: only an admin-authenticated caller may pin
            # a node (the token itself is never forwarded upstream).
            return self._fail(
                info,
                PoolErrorCode.TARGET_NODE_FORBIDDEN,
                f"{TARGET_NODE_HEADER} requires the pool admin credentials",
                rid,
            )
        reservation = [0]
        try:
            return await self._handle_routed(
                request, endpoint, info, raw_path, job_id, target, reservation, rid
            )
        finally:
            self._inspect_in_use -= reservation[0]

    def _admin_authorized(self, request: Request) -> bool:
        token = self.ctx.admin_token
        if not token:
            return False
        supplied = request.headers.get("authorization", "")
        if not supplied.lower().startswith("bearer "):
            return False
        return hmac.compare_digest(supplied[7:].strip().encode(), token.encode())

    async def _handle_routed(
        self,
        request: Request,
        endpoint: GatewayEndpoint,
        info: RelayInfo,
        raw_path: str,
        job_id: str | None,
        target: str | None,
        reservation: list[int],
        rid: str,
    ) -> Response:
        pinned = target is not None or job_id is not None
        try:
            if job_id is not None:
                target = await self._affinity_target(job_id, info)
            elif endpoint.async_job:
                assert self.ctx.job_store is not None  # noqa: S101 - checked above
                await self.ctx.job_store.ping()  # never create a job we cannot track
            content, model = await self._plan_body(request, endpoint, reservation)
            info.model_id = model
            decision = self.ctx.selector.select(
                self._routing_request(request, endpoint, model, target), self.ctx.registry
            )
        except PoolError as exc:
            if job_id is not None:
                self.ctx.metrics.inc(
                    "aigp_proxy_asr_affinity_lookup_failures_total",
                    environment=self.env,
                    error_category=exc.code.value,
                )
                if isinstance(exc, AffinityStoreUnavailable):
                    self.ctx.metrics.inc(
                        "aigp_pool_asr_affinity_store_errors_total", environment=self.env
                    )
            elif not isinstance(exc, AffinityStoreUnavailable):
                self.ctx.metrics.inc(
                    "aigp_pool_routing_rejections_total",
                    environment=self.env,
                    service=endpoint.service,
                    error_category=exc.code.value,
                )
            return self._fail(info, exc.code, exc.message, rid)
        except ClientDisconnect:
            return Response(status_code=499)

        info.node_id = decision.node_id
        info.physical_host_id = decision.physical_host_id
        info.routing_policy = decision.policy.value
        info.candidate_count = len(decision.candidates)
        info.routing_decision = "pinned_by_affinity" if job_id is not None else "selected"
        self.ctx.metrics.inc(
            "aigp_pool_routing_decisions_total",
            environment=self.env,
            service=endpoint.service,
            node_id=decision.node_id,
        )
        return await self._forward(
            request, endpoint, decision, content, raw_path, job_id, info, pinned=pinned
        )

    async def _affinity_target(self, job_id: str, info: RelayInfo) -> str:
        store = self.ctx.job_store
        assert store is not None  # noqa: S101 - checked by caller
        mapping = await store.lookup_job(self.ctx.config.environment, job_id)
        if mapping is None:
            info.job_affinity = "unknown"
            raise PoolError(
                PoolErrorCode.JOB_AFFINITY_UNKNOWN,
                "no job -> node mapping in this environment (jobs created outside the pool, "
                "expired or from another environment are not reachable through the pool)",
            )
        info.job_affinity = "hit"
        try:
            self.ctx.registry.get(mapping.node_id)
        except PoolError:
            # The owning node left the registry: never guess another one; keep
            # the mapping (the node may come back) and answer explicitly.
            info.job_affinity = "owner_not_configured"
            raise PoolError(
                PoolErrorCode.NODE_UNAVAILABLE,
                "the job belongs to a node that is no longer configured; it is not "
                "re-routed to another node",
            ) from None
        return mapping.node_id

    # ------------------------------------------------------------------ forward

    async def _forward(
        self,
        request: Request,
        endpoint: GatewayEndpoint,
        decision: RoutingDecision,
        content: bytes | AsyncIterator[bytes] | None,
        raw_path: str,
        job_id: str | None,
        info: RelayInfo,
        *,
        pinned: bool = False,
    ) -> Response:
        """Send to the selected node; controlled failover only if the request
        provably never left the pool (see docs/failover.md) and was not pinned to
        that node (X-AIGP-Target-Node or ASR job affinity)."""

        body = (
            TrackedBody(content) if content is not None and not isinstance(content, bytes) else None
        )
        payload: bytes | TrackedBody | None = body if body is not None else content  # type: ignore[assignment]
        tried: list[str] = [decision.node_id]
        max_extra = (
            self.ctx.config.failover.max_extra_nodes if self.ctx.config.failover.enabled else 0
        )
        while True:
            outcome = await self._send_once(request, endpoint, decision, payload, raw_path, info)
            if isinstance(outcome, httpx.Response):
                if len(tried) > 1:
                    info.failover_result = "success"
                    self.ctx.metrics.inc(
                        "aigp_pool_failover_success_total",
                        environment=self.env,
                        service=endpoint.service,
                    )
                return await self._relay(request, endpoint, decision, outcome, job_id, info)
            response, not_sent = outcome
            replayable = body is None or not body.started
            allowed = (
                not_sent
                and replayable
                and not pinned
                and failover_allowed(endpoint)
                and len(tried) - 1 < max_extra
            )
            info.failover_allowed = allowed
            if not not_sent or not allowed:
                if not_sent and len(tried) - 1 < max_extra and failover_allowed(endpoint) is False:
                    self.ctx.metrics.inc(
                        "aigp_pool_failover_rejected_total",
                        environment=self.env,
                        service=endpoint.service,
                        error_category="NOT_FAILOVER_ELIGIBLE_ROUTE",
                    )
                if len(tried) > 1:
                    info.failover_result = "exhausted"
                return response
            # Pre-dispatch failure: pick ANOTHER compatible node of the same environment.
            info.failover_attempted = True
            self.ctx.metrics.inc(
                "aigp_pool_failover_attempts_total", environment=self.env, service=endpoint.service
            )
            try:
                rr = self._routing_request(request, endpoint, info.model_id, None)
                decision = self.ctx.selector.select(
                    rr.model_copy(update={"exclude_nodes": frozenset(tried)}), self.ctx.registry
                )
            except PoolError as exc:
                info.failover_result = "no_alternative"
                self.ctx.metrics.inc(
                    "aigp_pool_failover_rejected_total",
                    environment=self.env,
                    service=endpoint.service,
                    error_category=exc.code.value,
                )
                return response
            tried.append(decision.node_id)
            info.node_id = decision.node_id
            info.physical_host_id = decision.physical_host_id

    async def _send_once(
        self,
        request: Request,
        endpoint: GatewayEndpoint,
        decision: RoutingDecision,
        payload: bytes | TrackedBody | None,
        raw_path: str,
        info: RelayInfo,
    ) -> httpx.Response | tuple[Response, bool]:
        """Return the upstream response, or (pool error response, not_sent)."""

        rid: str = request.state.request_id
        node = decision.node_id
        base = self.ctx.registry.resolve_destination(node)
        query = request.scope.get("query_string", b"")
        url = base + raw_path + (("?" + query.decode("latin-1")) if query else "")
        # The pool is authoritative for the correlation id: the upstream always
        # receives the id the pool validated (or generated), never a raw client
        # value the pool rejected.
        headers = [
            (k, v)
            for k, v in forwardable_request_headers(
                request.headers.items(), admin_token=self.ctx.admin_token
            )
            if k.lower() != "x-request-id"
        ]
        headers.append((REQUEST_ID_HEADER, rid))
        profile = self.ctx.config.timeouts.for_class(endpoint.traffic)
        upstream_req = self.ctx.client.build_request(
            request.method,
            httpx.URL(url),
            headers=headers,
            content=payload,
            timeout=profile.to_httpx(),
        )
        labels = {"environment": self.env, "node_id": node}
        avail = self.ctx.registry.availability
        start = time.perf_counter()
        # AIGP-HARDEN-1 (AIGP-NEW-01): `trial` is True when this request is the
        # single HALF_OPEN trial of the breaker. EVERY way out of this method that
        # does not reach a verdict (success / transport failure) releases it, so a
        # neutral outcome (client gone, malformed client body, local pool timeout,
        # cancellation, unexpected error) can never leave the node excluded.
        trial = avail.begin(node)
        settled = False

        def fail(
            code: PoolErrorCode, msg: str, *, not_sent: bool, transport: str | None
        ) -> tuple[Response, bool]:
            nonlocal settled
            settled = True
            if transport is not None:
                state = avail.record_transport_failure(node, transport)
                self.ctx.metrics.inc("aigp_proxy_upstream_unavailable_total", **labels)
                if state.value == "OPEN":
                    self.ctx.metrics.inc("aigp_pool_node_unavailable_total", **labels)
            avail.end(node, trial=trial)  # releases a still-pending neutral trial
            return self._fail(info, code, msg, rid), not_sent

        try:
            resp = await self._send_loop(request, payload, upstream_req, profile, endpoint, fail)
        except ClientDisconnect:
            settled = True
            avail.end(node, trial=trial)  # client gone: neutral, never a node failure
            return Response(status_code=499), False
        except BaseException:
            if not settled:  # cancellation / unexpected error: release, then propagate
                settled = True
                avail.end(node, trial=trial)
            raise
        if not isinstance(resp, httpx.Response):
            return resp
        return self._after_send(resp, node, endpoint, info, start)

    async def _send_loop(
        self,
        request: Request,
        payload: bytes | TrackedBody | None,
        upstream_req: httpx.Request,
        profile: TimeoutProfile,
        endpoint: GatewayEndpoint,
        fail: Callable[..., tuple[Response, bool]],
    ) -> httpx.Response | tuple[Response, bool]:
        attempts = 0
        while True:
            try:
                async with asyncio.timeout(profile.total_budget):
                    return await self.ctx.client.send(upstream_req, stream=True)
            except httpx.ConnectTimeout:
                return fail(
                    PoolErrorCode.UPSTREAM_UNAVAILABLE,
                    "connect timeout to the node (nothing was sent)",
                    not_sent=True,
                    transport="connect_timeout",
                )
            except httpx.ConnectError:
                if request.method in {"GET", "HEAD"} and payload is None and attempts == 0:
                    attempts += 1
                    continue  # idempotent read, no byte received: once, same node
                return fail(
                    PoolErrorCode.UPSTREAM_UNAVAILABLE,
                    "the node is unreachable (connection refused or failed; nothing was sent)",
                    not_sent=True,
                    transport="connect_error",
                )
            except httpx.RemoteProtocolError:
                if request.method in {"GET", "HEAD"} and payload is None and attempts == 0:
                    attempts += 1
                    continue
                return fail(
                    PoolErrorCode.UPSTREAM_UNAVAILABLE,
                    "connection to the node was lost; the request may have been processed "
                    "and is NOT retried",
                    not_sent=False,
                    transport="connection_lost",
                )
            except httpx.PoolTimeout:
                return fail(
                    PoolErrorCode.UPSTREAM_TIMEOUT,
                    "no free upstream connection (nothing was sent)",
                    not_sent=True,
                    transport=None,
                )
            except (httpx.TimeoutException, TimeoutError):
                return fail(
                    PoolErrorCode.UPSTREAM_TIMEOUT,
                    f"no response from the node within the {endpoint.traffic.value} budget; "
                    "the request may still be running upstream and is NOT retried",
                    not_sent=False,
                    transport="response_timeout",
                )
            except (httpx.WriteError, httpx.ReadError) as exc:
                return fail(
                    PoolErrorCode.UPSTREAM_UNAVAILABLE,
                    f"transport error with the node ({type(exc).__name__}); not retried",
                    not_sent=False,
                    transport=type(exc).__name__,
                )
            except httpx.LocalProtocolError:
                # The client body did not match its own framing (e.g. shorter than
                # its Content-Length): nothing complete was forwarded.
                return fail(
                    PoolErrorCode.INVALID_REQUEST_FRAMING,
                    "request body does not match its declared framing",
                    not_sent=False,
                    transport=None,
                )
            except httpx.HTTPError as exc:
                return fail(
                    PoolErrorCode.UPSTREAM_UNAVAILABLE,
                    f"transport error with the node ({type(exc).__name__}); not retried",
                    not_sent=False,
                    transport=type(exc).__name__,
                )

    def _after_send(
        self,
        resp: httpx.Response,
        node: str,
        endpoint: GatewayEndpoint,
        info: RelayInfo,
        start: float,
    ) -> httpx.Response:
        avail = self.ctx.registry.availability
        avail.record_success(node)  # any HTTP answer proves the node is reachable
        if resp.status_code == 503 and "retry-after" in resp.headers:
            avail.record_saturation(node, endpoint.service, _retry_after(resp.headers))
        info.upstream_latency_ms = round((time.perf_counter() - start) * 1000, 2)
        self.ctx.metrics.observe(
            "aigp_proxy_request_duration_seconds",
            time.perf_counter() - start,
            environment=self.env,
            service=endpoint.service,
            node_id=node,
        )
        self.ctx.metrics.inc(
            "aigp_proxy_requests_total",
            environment=self.env,
            service=endpoint.service,
            node_id=node,
            status_class=f"{resp.status_code // 100}xx",
        )
        self.ctx.metrics.inc(
            "aigp_pool_node_selection_total",
            environment=self.env,
            service=endpoint.service,
            node_id=node,
        )
        return resp

    async def _relay(
        self,
        request: Request,
        endpoint: GatewayEndpoint,
        decision: RoutingDecision,
        resp: httpx.Response,
        job_id: str | None,
        info: RelayInfo,
    ) -> Response:
        out_headers = forwardable_response_headers(resp.headers.multi_items())
        if endpoint.async_job and request.method in {"POST", "DELETE"}:
            try:
                return await self._finish_asr(request, resp, out_headers, decision, job_id, info)
            finally:
                self.ctx.registry.availability.end(decision.node_id)

        streaming = endpoint.response is ResponseKind.NDJSON
        info.streaming = streaming
        if streaming:
            self.ctx.metrics.inc(
                "aigp_proxy_streaming_responses_total",
                environment=self.env,
                service=endpoint.service,
                node_id=decision.node_id,
            )
        return self._stream_back(resp, out_headers, endpoint, decision)

    def _stream_back(
        self,
        resp: httpx.Response,
        headers: list[tuple[str, str]],
        endpoint: GatewayEndpoint,
        decision: RoutingDecision,
    ) -> Response:
        metrics, env, logger = self.ctx.metrics, self.env, self.ctx.logger
        avail = self.ctx.registry.availability
        gauge = {"environment": env, "node_id": decision.node_id}
        metrics.inc("aigp_proxy_active_requests", **gauge)

        async def body() -> AsyncIterator[bytes]:
            completed = False
            try:
                async for chunk in _raw_chunks(resp):
                    if chunk:
                        yield chunk
                completed = True
            except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout) as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "upstream stream interrupted",
                    event="stream_interrupted",
                    environment=env,
                    selected_node_id=decision.node_id,
                    service=endpoint.service,
                    error_category=type(exc).__name__,
                )
                # Abort the downstream connection: no fabricated end of stream and
                # NEVER a replay on another node once bytes reached the client.
                raise
            finally:
                if not completed:
                    metrics.inc(
                        "aigp_proxy_stream_interruptions_total",
                        environment=env,
                        service=endpoint.service,
                        node_id=decision.node_id,
                    )
                metrics.dec("aigp_proxy_active_requests", **gauge)
                avail.end(decision.node_id)
                with contextlib.suppress(Exception):
                    await resp.aclose()

        response = StreamingResponse(body(), status_code=resp.status_code)
        response.raw_headers = [
            (k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers
        ]
        return response

    async def _read_small(self, resp: httpx.Response) -> bytes:
        chunks: list[bytes] = []
        size = 0
        try:
            async for chunk in _raw_chunks(resp):
                size += len(chunk)
                if size > _SMALL_BODY_LIMIT:
                    raise PoolError(
                        PoolErrorCode.POOL_ROUTING_ERROR, "unexpectedly large ASR control answer"
                    )
                chunks.append(chunk)
        finally:
            await resp.aclose()
        return b"".join(chunks)

    async def _compensate(self, node_id: str, gateway_job_id: str) -> str:
        """Targeted compensation: DELETE the ONE job just created on the ONE node
        that created it. Never a search on other nodes, never a bulk delete."""

        try:
            base = self.ctx.registry.resolve_destination(node_id)
            r = await self.ctx.client.delete(
                f"{base}{ASR_JOBS_PREFIX}{gateway_job_id}",
                timeout=self.ctx.config.timeouts.job_poll.to_httpx(),
            )
        except (httpx.HTTPError, PoolError):
            return "failed"
        return "deleted" if 200 <= r.status_code < 300 or r.status_code == 404 else "failed"

    async def _finish_asr(
        self,
        request: Request,
        resp: httpx.Response,
        headers: list[tuple[str, str]],
        decision: RoutingDecision,
        job_id: str | None,
        info: RelayInfo,
    ) -> Response:
        """ASR submit (persist affinity BEFORE announcing the job) or delete."""

        store = self.ctx.job_store
        assert store is not None  # noqa: S101 - async routes require a store
        env = self.ctx.config.environment
        rid: str = request.state.request_id
        body = await self._read_small(resp)

        if request.method == "POST" and 200 <= resp.status_code < 300:
            created = _job_id_from(body, resp.headers.get("content-encoding"))
            if created is None:
                info.job_affinity = "no_job_id"
            else:
                try:
                    await store.register_job(
                        env, created, decision.node_id, self.ctx.config.job_affinity.ttl_seconds
                    )
                    info.job_affinity = "registered"
                except PoolError as exc:
                    # The job exists on the node but cannot be made durably
                    # resolvable (store down, or id already bound to ANOTHER node):
                    # fail closed, compensate on that node only, never announce it.
                    conflict = exc.code is PoolErrorCode.JOB_AFFINITY_CONFLICT
                    info.job_affinity = "conflict" if conflict else "store_unavailable"
                    self.ctx.metrics.inc(
                        "aigp_pool_asr_affinity_conflicts_total"
                        if conflict
                        else "aigp_pool_asr_affinity_store_errors_total",
                        environment=self.env,
                    )
                    outcome = await self._compensate(decision.node_id, created)
                    log_event(
                        self.ctx.logger,
                        logging.ERROR,
                        "asr job not announced: affinity not persisted",
                        event="asr_affinity_failed",
                        request_id=rid,
                        environment=self.env,
                        selected_node_id=decision.node_id,
                        job_affinity=info.job_affinity,
                        error_category=f"compensation_{outcome}",
                        job_ref=hashlib.sha256(created.encode()).hexdigest()[:12],
                    )
                    extra: dict[str, Any] = {"compensation": outcome}
                    if outcome == "failed":
                        # Orphan on that node until the Gateway purges it: give the
                        # creator what it needs to diagnose, nothing more.
                        extra |= {"node_id": decision.node_id, "gateway_job_id": created}
                    return pool_error_response(
                        exc.code,
                        "the Gateway accepted the job but its node affinity could not be "
                        "persisted; the job is not announced"
                        + (" (id collision with another node)" if conflict else ""),
                        rid,
                        extra=extra,
                    )

        if request.method == "DELETE" and job_id is not None:
            # Forget only once the Gateway confirmed the job is gone (deleted or
            # already absent/expired), and only if still bound to THIS node.
            if 200 <= resp.status_code < 300 or resp.status_code == 404:
                with contextlib.suppress(AffinityStoreUnavailable):
                    await store.forget_job(env, job_id, node_id=decision.node_id)
                info.job_affinity = "forgotten"
            else:
                info.job_affinity = "kept"

        response = Response(content=body, status_code=resp.status_code)
        response.raw_headers = [
            (k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers
        ]
        return response


class TrackedBody:
    """Async request body wrapper that records whether any byte was pulled.

    Used to PROVE, before a failover, that nothing of the body left the pool."""

    def __init__(self, source: AsyncIterator[bytes]) -> None:
        self._source = source
        self.started = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._source:
            self.started = True
            yield chunk


MODEL_ID_MAX_CHARS = 200  # == ModelRequirement / ModelDescriptor bound


class _JsonObject(dict[str, Any]):
    """dict that remembers whether the key ``model`` was given more than once."""

    duplicate_model = False


def _object_pairs(pairs: list[tuple[str, Any]]) -> _JsonObject:
    obj = _JsonObject()
    for key, value in pairs:
        if key == "model" and key in obj:
            obj.duplicate_model = True
        obj[key] = value
    return obj


def _validated_model(body: bytes) -> str | None:
    """The requested model of a JSON request body, or None if absent (implicit).

    Strict and fail-closed (AIGP-HARDEN-1): anything that does not let the pool
    establish the model unambiguously is refused before routing. The error
    messages never echo the body.
    """

    def invalid(why: str) -> PoolError:
        return PoolError(PoolErrorCode.INVALID_REQUEST_BODY, why)

    try:
        parsed = json.loads(body, object_pairs_hook=_object_pairs)
    except (ValueError, RecursionError):
        raise invalid("request body is not valid JSON") from None
    if not isinstance(parsed, dict):
        raise invalid("request body must be a JSON object")
    if isinstance(parsed, _JsonObject) and parsed.duplicate_model:
        raise invalid("`model` is given more than once")
    if "model" not in parsed:
        return None
    model = parsed["model"]
    if not isinstance(model, str):
        raise invalid("`model` must be a string")
    if not 1 <= len(model) <= MODEL_ID_MAX_CHARS:
        raise invalid(f"`model` must be 1 to {MODEL_ID_MAX_CHARS} characters long")
    return model


def failover_allowed(endpoint: GatewayEndpoint) -> bool:
    """Route-level failover rule (pre-dispatch failures only).

    * diagnostics / inventory GETs: NO — the answer describes ONE node;
    * ASR job GET/DELETE: NO — pinned to the node owning the job;
    * POST inference and ASR submit: YES, but only when nothing was sent.
    """

    if endpoint.traffic is TrafficClass.HEALTH_PROBE:
        return False
    return not (endpoint.async_job and endpoint.method in {"GET", "DELETE"})


def _retry_after(headers: httpx.Headers) -> float | None:
    raw = headers.get("retry-after", "")
    try:
        return float(raw)
    except ValueError:
        return None  # HTTP-date form: hint of 1 s, bounded by the breaker settings


def _job_id_from(body: bytes, content_encoding: str | None) -> str | None:
    if content_encoding and content_encoding.lower() != "identity":
        return None
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    for key in ("job_id", "transcription_id"):
        value = parsed.get(key)
        if isinstance(value, str) and valid_job_id(value):
            return value
    return None
