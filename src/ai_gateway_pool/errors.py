"""Explicit, typed error vocabulary of the pool.

Every refusal the pool emits carries one of these codes so that clients and
operators can tell *why* no upstream was chosen. A refusal never pretends that an
unknown destination is available.
"""

from __future__ import annotations

from enum import StrEnum


class PoolErrorCode(StrEnum):
    NO_ELIGIBLE_NODE = "NO_ELIGIBLE_NODE"
    NODE_UNAVAILABLE = "NODE_UNAVAILABLE"
    MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
    MODEL_REQUIRED = "MODEL_REQUIRED"
    CAPABILITY_NOT_AVAILABLE = "CAPABILITY_NOT_AVAILABLE"
    ROUTING_NOT_READY = "ROUTING_NOT_READY"
    CONFIGURATION_INVALID = "CONFIGURATION_INVALID"
    # Security refusals.
    INVALID_NODE_DESTINATION = "INVALID_NODE_DESTINATION"
    UPSTREAM_OVERRIDE_DENIED = "UPSTREAM_OVERRIDE_DENIED"
    ROUTE_NOT_IN_CONTRACT = "ROUTE_NOT_IN_CONTRACT"
    INVALID_REQUEST_PATH = "INVALID_REQUEST_PATH"
    INVALID_REQUEST_FRAMING = "INVALID_REQUEST_FRAMING"
    # AIGP-HARDEN-1: JSON bodies of model-bearing routes are always inspected.
    INVALID_REQUEST_BODY = "INVALID_REQUEST_BODY"
    REQUEST_BODY_TOO_LARGE = "REQUEST_BODY_TOO_LARGE"
    UNSUPPORTED_CONTENT_ENCODING = "UNSUPPORTED_CONTENT_ENCODING"
    INSPECTION_BUDGET_EXHAUSTED = "INSPECTION_BUDGET_EXHAUSTED"
    TARGET_NODE_FORBIDDEN = "TARGET_NODE_FORBIDDEN"
    # Transport (pool <-> upstream). Distinct from errors RETURNED by the Gateway,
    # which are relayed unchanged.
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    POOL_ROUTING_ERROR = "POOL_ROUTING_ERROR"
    AFFINITY_STORE_UNAVAILABLE = "AFFINITY_STORE_UNAVAILABLE"
    # Job affinity.
    JOB_AFFINITY_UNKNOWN = "JOB_AFFINITY_UNKNOWN"
    JOB_AFFINITY_CONFLICT = "JOB_AFFINITY_CONFLICT"


# HTTP status the pool's own API uses for each code (pool-originated errors only;
# upstream AI Gateway statuses will be relayed unchanged in POOL-2).
HTTP_STATUS: dict[PoolErrorCode, int] = {
    PoolErrorCode.NO_ELIGIBLE_NODE: 503,
    PoolErrorCode.NODE_UNAVAILABLE: 503,
    PoolErrorCode.MODEL_NOT_AVAILABLE: 404,
    PoolErrorCode.MODEL_REQUIRED: 400,
    PoolErrorCode.CAPABILITY_NOT_AVAILABLE: 503,
    PoolErrorCode.ROUTING_NOT_READY: 501,
    PoolErrorCode.CONFIGURATION_INVALID: 500,
    PoolErrorCode.INVALID_NODE_DESTINATION: 400,
    PoolErrorCode.UPSTREAM_OVERRIDE_DENIED: 400,
    PoolErrorCode.ROUTE_NOT_IN_CONTRACT: 404,
    PoolErrorCode.INVALID_REQUEST_PATH: 400,
    PoolErrorCode.INVALID_REQUEST_FRAMING: 400,
    PoolErrorCode.INVALID_REQUEST_BODY: 400,
    PoolErrorCode.REQUEST_BODY_TOO_LARGE: 413,
    PoolErrorCode.UNSUPPORTED_CONTENT_ENCODING: 415,
    PoolErrorCode.INSPECTION_BUDGET_EXHAUSTED: 503,
    PoolErrorCode.TARGET_NODE_FORBIDDEN: 403,
    PoolErrorCode.UPSTREAM_UNAVAILABLE: 503,
    PoolErrorCode.UPSTREAM_TIMEOUT: 504,
    PoolErrorCode.POOL_ROUTING_ERROR: 500,
    PoolErrorCode.AFFINITY_STORE_UNAVAILABLE: 503,
    PoolErrorCode.JOB_AFFINITY_UNKNOWN: 404,
    PoolErrorCode.JOB_AFFINITY_CONFLICT: 409,
}


class PoolError(Exception):
    """Base class of every pool-originated, client-reportable error."""

    def __init__(self, code: PoolErrorCode, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message

    @property
    def http_status(self) -> int:
        return HTTP_STATUS[self.code]


class ConfigurationInvalidError(PoolError):
    def __init__(self, message: str) -> None:
        super().__init__(PoolErrorCode.CONFIGURATION_INVALID, message)


class RoutingError(PoolError):
    """Raised by selectors when no node may serve a request."""
