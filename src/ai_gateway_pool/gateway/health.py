"""Interpretation of AI Gateway diagnostic payloads (pure functions).

Only non-inferential endpoints may be probed by the pool:

* ``/health/runtime`` — "no upstream call, callable in a loop at no cost";
* ``/v1/models`` — model inventory read.

``/health/ready``, ``/health/embeddings`` and ``/health/rerank`` run real model
computations and are NEVER used as probes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from ai_gateway_pool.nodes.models import Capability, CapabilityState, Environment, HealthState

SAFE_PROBE_PATHS: frozenset[str] = frozenset({"/health/runtime"})
INVENTORY_PATH = "/v1/models"
FORBIDDEN_PROBE_PATHS: frozenset[str] = frozenset(
    {"/health/ready", "/health/embeddings", "/health/rerank"}
)

# The Gateway names environments like the pool does.
_GATEWAY_ENV = {"staging": Environment.STAGING, "production": Environment.PRODUCTION}


class RuntimeObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    health: HealthState
    error: str | None = None
    reported_environment: str | None = None
    build_git_sha: str | None = None
    capability_states: dict[Capability, CapabilityState] = {}
    model_ids: frozenset[str] = frozenset()
    # model id -> raw capability names published by the Gateway
    model_caps: dict[str, frozenset[str]] = {}
    defaults: dict[Capability, str] = {}


# gateway-api/1.0 `capabilities` section names -> pool capabilities.
_CAPABILITY_SECTIONS: dict[Capability, str] = {
    Capability.GENERATE: "generation",
    Capability.CHAT: "chat",
    Capability.EMBEDDINGS: "embeddings",
    Capability.RERANK: "rerank",
    Capability.ASR: "asr",
    Capability.VISION: "vision",
}
_STATE_MAP: dict[str, CapabilityState] = {
    "ENABLED": CapabilityState.AVAILABLE,
    "DISABLED": CapabilityState.UNAVAILABLE,
    "UNAVAILABLE": CapabilityState.UNAVAILABLE,
    "UNKNOWN": CapabilityState.UNKNOWN,
}


def _capability_state(capabilities: Any, section: str) -> CapabilityState | None:
    if not isinstance(capabilities, dict):
        return None
    entry = capabilities.get(section)
    if not isinstance(entry, dict) or not isinstance(entry.get("state"), str):
        return None
    return _STATE_MAP.get(entry["state"].upper())


def _enabled_state(section: Any) -> CapabilityState:
    if not isinstance(section, dict) or "enabled" not in section:
        return CapabilityState.UNKNOWN
    return CapabilityState.AVAILABLE if section["enabled"] is True else CapabilityState.UNAVAILABLE


def _logicals(items: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("model_logical"), str):
                out.add(item["model_logical"])
    return out


def interpret_runtime(payload: Any, expected: Environment) -> RuntimeObservation:
    """Map ``/health/runtime`` to an observation.

    The environment self-reported by the Gateway MUST equal the node's
    environment: a production node answering ``staging`` (mis-wired port) is
    marked UNAVAILABLE, never used.

    Only ASR and RERANK publish an explicit ``enabled`` flag; every other
    capability stays UNKNOWN (declared ≠ available).
    """

    if not isinstance(payload, dict):
        return RuntimeObservation(health=HealthState.UNAVAILABLE, error="runtime_payload_invalid")
    reported = payload.get("environment")
    if _GATEWAY_ENV.get(str(reported)) != expected:
        return RuntimeObservation(
            health=HealthState.UNAVAILABLE,
            error="environment_mismatch",
            reported_environment=str(reported) if reported is not None else None,
        )
    build = payload.get("build")
    sha = build.get("git_sha") if isinstance(build, dict) else None
    caps = {
        Capability.ASR: _enabled_state(payload.get("asr")),
        Capability.RERANK: _enabled_state(payload.get("rerank")),
    }
    # AI Gateway Open (gateway-api/1.0) also publishes `capabilities.<name>.state`
    # (ENABLED / DISABLED / UNAVAILABLE / UNKNOWN) for every capability. When present
    # it is authoritative: a DISABLED or UNAVAILABLE capability is never routed to.
    for cap, section in _CAPABILITY_SECTIONS.items():
        state = _capability_state(payload.get("capabilities"), section)
        if state is not None:
            caps[cap] = state
    models = _logicals(payload.get("llm_models"))
    model_caps: dict[str, frozenset[str]] = {}
    llm = payload.get("llm_models")
    if isinstance(llm, list):
        for item in llm:
            if isinstance(item, dict) and isinstance(item.get("model_logical"), str):
                raw = item.get("capabilities")
                model_caps[item["model_logical"]] = (
                    frozenset(c for c in raw if isinstance(c, str))
                    if isinstance(raw, list)
                    else frozenset()
                )
    defaults: dict[Capability, str] = {}
    for section, cap, raw_cap in (
        ("asr", Capability.ASR, "asr"),
        ("rerank", Capability.RERANK, "rerank"),
    ):
        sec = payload.get(section)
        if isinstance(sec, dict):
            logicals = _logicals(sec.get("models"))
            models |= logicals
            for m in logicals:
                model_caps[m] = frozenset({raw_cap})
            if isinstance(sec.get("default_model"), str):
                defaults[cap] = sec["default_model"]
    return RuntimeObservation(
        health=HealthState.HEALTHY,
        reported_environment=str(reported),
        build_git_sha=sha if isinstance(sha, str) else None,
        capability_states=caps,
        model_ids=frozenset(models),
        model_caps=model_caps,
        defaults=defaults,
    )


def interpret_models(payload: Any) -> frozenset[str] | None:
    """Model identities from ``/v1/models``: concrete ``name`` and governed ``model_logical``."""

    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return None
    ids: set[str] = set()
    for item in payload["models"]:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model_logical"):
            if isinstance(item.get(key), str) and item[key]:
                ids.add(item[key])
    return frozenset(ids)


class InventoryObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_ids: frozenset[str]
    model_caps: dict[str, frozenset[str]]
    digests: dict[str, str]


def interpret_inventory(payload: Any) -> InventoryObservation | None:
    """Richer reading of ``/v1/models``: ids, raw capabilities and artifact digests.

    The digest of a concrete entry is also attached to its governed
    ``model_logical`` id (they are the same artifact, as published by the Gateway).
    """

    ids = interpret_models(payload)
    if ids is None:
        return None
    caps: dict[str, frozenset[str]] = {}
    digests: dict[str, str] = {}
    for item in payload["models"]:
        if not isinstance(item, dict):
            continue
        raw = item.get("capabilities")
        c = (
            frozenset(x for x in raw if isinstance(x, str))
            if isinstance(raw, list)
            else frozenset()
        )
        for key in ("name", "model_logical"):
            mid = item.get(key)
            if isinstance(mid, str) and mid:
                caps[mid] = c
                if isinstance(item.get("digest"), str) and item["digest"]:
                    digests[mid] = item["digest"]
    return InventoryObservation(model_ids=ids, model_caps=caps, digests=digests)
