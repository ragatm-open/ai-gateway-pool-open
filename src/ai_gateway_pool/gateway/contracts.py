"""Catalogue of the AI Gateway Open HTTP contract — ``gateway-api/1.0``.

Source: the OpenAPI snapshot ``gateway-api/1.0.yaml`` published by AI Gateway Open
(19 routes, status CANDIDATE_INTEGRATION_VERIFIED) and the live
API of an AI Gateway Open node during the qualification. The two routes
of the historical private contract that are NOT part of the Open edition
(``POST /v1/vision/structured-extract``, ``GET /health/structured-extract``) are
deliberately absent: a request for them is refused with ``ROUTE_NOT_IN_CONTRACT``
(404) and is never dispatched to a node.

``GET /health/live`` exists on both sides: the pool answers it itself (pool
liveness, see ``api/health.py``) and never relays it — a liveness probe must
describe the process that received it.

``proxy_status`` is ``RELAYED`` for every route since POOL-2. It states that the
relay handles the route, NOT its qualification level (see
docs/contract-qualification.md: mock vs live).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from ai_gateway_pool.nodes.models import Capability
from ai_gateway_pool.transport.contracts import TrafficClass


class BodyKind(StrEnum):
    NONE = "none"
    JSON = "json"
    MULTIPART = "multipart"  # parts `request` (JSON) THEN `file`, order enforced


class ResponseKind(StrEnum):
    JSON = "json"
    NDJSON = "ndjson"
    JOB_RESULT = "job_result"  # json | text | srt | vtt depending on ?format=


class ProxyStatus(StrEnum):
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    RELAYED = "RELAYED"  # POOL-2: relayed by transport/proxy.py


@dataclass(frozen=True)
class GatewayEndpoint:
    method: str
    path: str  # template, `{job_id}` placeholders
    service: str
    capability: Capability | None  # None = diagnostic / inventory route
    body: BodyKind
    response: ResponseKind
    traffic: TrafficClass
    gpu_compute: bool  # True = calling it triggers real inference
    async_job: bool = False
    proxy_status: ProxyStatus = ProxyStatus.RELAYED

    @property
    def pattern(self) -> re.Pattern[str]:
        return re.compile("^" + re.sub(r"\{[a-z_]+\}", r"[^/]+", self.path) + "$")


_H = TrafficClass.HEALTH_PROBE
_I = TrafficClass.INFERENCE
_S = TrafficClass.STREAMING
_J = TrafficClass.JOB_POLL

ENDPOINTS: tuple[GatewayEndpoint, ...] = (
    # --- health / diagnostics -------------------------------------------------
    GatewayEndpoint("GET", "/health", "health", None, BodyKind.NONE, ResponseKind.JSON, _H, False),
    GatewayEndpoint(
        "GET", "/health/runtime", "health", None, BodyKind.NONE, ResponseKind.JSON, _H, False
    ),
    # /health/ready really computes an embedding (60 s cache, under a lock).
    GatewayEndpoint(
        "GET",
        "/health/ready",
        "health",
        Capability.EMBEDDINGS,
        BodyKind.NONE,
        ResponseKind.JSON,
        _H,
        True,
    ),
    GatewayEndpoint(
        "GET",
        "/health/embeddings",
        "health",
        Capability.EMBEDDINGS,
        BodyKind.NONE,
        ResponseKind.JSON,
        _H,
        True,
    ),
    GatewayEndpoint(
        "GET",
        "/health/rerank",
        "health",
        Capability.RERANK,
        BodyKind.NONE,
        ResponseKind.JSON,
        _H,
        True,
    ),
    GatewayEndpoint(
        "GET",
        "/health/vision",
        "health",
        Capability.VISION,
        BodyKind.NONE,
        ResponseKind.JSON,
        _H,
        False,
    ),
    GatewayEndpoint(
        "GET", "/health/asr", "health", Capability.ASR, BodyKind.NONE, ResponseKind.JSON, _H, False
    ),
    # --- inventory --------------------------------------------------------------
    GatewayEndpoint(
        "GET", "/v1/models", "models", None, BodyKind.NONE, ResponseKind.JSON, _H, False
    ),
    GatewayEndpoint(
        "GET",
        "/v1/vision/capacity",
        "vision",
        Capability.VISION,
        BodyKind.NONE,
        ResponseKind.JSON,
        _H,
        False,
    ),
    # --- inference ----------------------------------------------------------------
    GatewayEndpoint(
        "POST",
        "/v1/generate",
        "generate",
        Capability.GENERATE,
        BodyKind.JSON,
        ResponseKind.JSON,
        _I,
        True,
    ),
    GatewayEndpoint(
        "POST",
        "/v1/generate/stream",
        "generate",
        Capability.GENERATE,
        BodyKind.JSON,
        ResponseKind.NDJSON,
        _S,
        True,
    ),
    GatewayEndpoint(
        "POST", "/v1/chat", "chat", Capability.CHAT, BodyKind.JSON, ResponseKind.JSON, _I, True
    ),
    GatewayEndpoint(
        "POST",
        "/v1/chat/stream",
        "chat",
        Capability.CHAT,
        BodyKind.JSON,
        ResponseKind.NDJSON,
        _S,
        True,
    ),
    GatewayEndpoint(
        "POST",
        "/v1/embeddings",
        "embeddings",
        Capability.EMBEDDINGS,
        BodyKind.JSON,
        ResponseKind.JSON,
        _I,
        True,
    ),
    GatewayEndpoint(
        "POST",
        "/v1/rerank",
        "rerank",
        Capability.RERANK,
        BodyKind.JSON,
        ResponseKind.JSON,
        _I,
        True,
    ),
    GatewayEndpoint(
        "POST",
        "/v1/vision/analyze",
        "vision",
        Capability.VISION,
        BodyKind.JSON,
        ResponseKind.JSON,
        _I,
        True,
    ),
    # --- ASR async jobs ---------------------------------------------------------
    GatewayEndpoint(
        "POST",
        "/v1/audio/transcriptions",
        "asr",
        Capability.ASR,
        BodyKind.MULTIPART,
        ResponseKind.JSON,
        TrafficClass.UPLOAD,
        True,
        async_job=True,
    ),
    GatewayEndpoint(
        "GET",
        "/v1/audio/transcriptions/{job_id}",
        "asr",
        Capability.ASR,
        BodyKind.NONE,
        ResponseKind.JOB_RESULT,
        _J,
        False,
        async_job=True,
    ),
    GatewayEndpoint(
        "DELETE",
        "/v1/audio/transcriptions/{job_id}",
        "asr",
        Capability.ASR,
        BodyKind.NONE,
        ResponseKind.JSON,
        _J,
        False,
        async_job=True,
    ),
)


#: Version of the Gateway contract this catalogue mirrors.
CONTRACT_VERSION = "gateway-api/1.0"

#: Routes of gateway-api/1.0 answered by the POOL ITSELF (never relayed): liveness
#: describes the process that received the probe.
POOL_LOCAL_ROUTES: tuple[tuple[str, str], ...] = (("GET", "/health/live"),)

#: Routes of the historical private contract that the Open edition does not serve.
#: Kept as DATA (not as endpoints) so that tests can prove they are refused.
EXCLUDED_ROUTES: tuple[tuple[str, str], ...] = (
    ("GET", "/health/structured-extract"),
    ("POST", "/v1/vision/structured-extract"),
)

#: Capabilities a node may declare: every capability that has at least one route.
CONTRACT_CAPABILITIES: frozenset[Capability] = frozenset(
    e.capability for e in ENDPOINTS if e.capability is not None
)


def match_endpoint(method: str, path: str) -> GatewayEndpoint | None:
    method = method.upper()
    for endpoint in ENDPOINTS:
        if endpoint.method == method and endpoint.pattern.match(path):
            return endpoint
    return None


def paths_for(method: str) -> set[str]:
    return {e.path for e in ENDPOINTS if e.method == method.upper()}
