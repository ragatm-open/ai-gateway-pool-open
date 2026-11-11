"""Structured JSON logging with a strict field allowlist.

Only the fields in :data:`ALLOWED_FIELDS` ever reach the output. Request bodies,
prompts, transcripts, images, multipart payloads and headers are not loggable
fields at all, so they cannot leak by accident. Messages are additionally
scrubbed of bearer tokens / API keys as defence in depth.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from typing import Any

ALLOWED_FIELDS: frozenset[str] = frozenset(
    {
        "request_id",
        "environment",
        "selected_node_id",
        "service",
        "model_identifier",
        "routing_policy",
        "response_status",
        "latency_ms",
        "error_category",
        "method",
        "route",
        "event",
        # POOL-2 relay fields (identifiers and numbers only, never content).
        "capability",
        "model_id",
        "http_status",
        "upstream_latency_ms",
        "streaming",
        "response_bytes",
        "job_affinity",
        # POOL-3 multi-node fields (identifiers, enums and counts only).
        "physical_host_id",
        "candidate_count",
        "routing_decision",
        "failover_attempted",
        "failover_allowed",
        "failover_result",
        "affinity_backend",
        "node_health",
        "job_ref",  # sha256 prefix of a job id, never the id itself in error logs
    }
)

_SECRET_PATTERNS = [
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b(authorization|api[_-]?key|x-api-key|token|password|secret)\b\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
]
REDACTED = "[REDACTED]"

LOGGER_NAME = "ai_gateway_pool"


MAX_FIELD_CHARS = 256  # every logged string field is bounded (client-derived values)


def _bounded(text: str) -> str:
    if len(text) <= MAX_FIELD_CHARS:
        return text
    return text[:MAX_FIELD_CHARS] + f"...[{len(text) - MAX_FIELD_CHARS} more]"


def scrub(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": scrub(record.getMessage()),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                if key in ALLOWED_FIELDS:
                    payload[key] = scrub(_bounded(value)) if isinstance(value, str) else value
        if record.exc_info and record.exc_info[0] is not None:
            # Exception TYPE only: exception messages may embed payload fragments.
            payload["exception"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    logger.log(level, message, extra={"fields": fields})
