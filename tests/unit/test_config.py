"""Configuration validation (brief §20 'Configuration')."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_gateway_pool.config import cli
from ai_gateway_pool.config.schema import PoolConfig, check_environment_pair
from ai_gateway_pool.config.settings import (
    Settings,
    load_for_settings,
    load_pool_config,
    load_settings,
)
from ai_gateway_pool.errors import ConfigurationInvalidError
from tests.conftest import ROOT, build_config, node, raw_config, write_yaml

# Open edition example configurations: one real single-node Compact profile
# (production label, as published by AI Gateway Open) and one FICTIONAL multi-node example.
COMPACT = ROOT / "config" / "compact.example.yaml"
MULTINODE = ROOT / "config" / "multinode.example.yaml"
LOCAL = ROOT / "config" / "local.example.yaml"
# Backwards-compatible aliases for the pair checks below.
PRODUCTION = COMPACT
STAGING = MULTINODE


def test_valid_compact_config() -> None:
    cfg = load_pool_config(COMPACT)
    assert cfg.environment == "production"
    [n] = cfg.nodes
    assert (n.node_id, n.physical_host_id, n.base_url) == (
        "compact-gateway",
        "compact-host",
        "http://ai-gateway:8100",
    )
    # Compact profile: generation, embeddings and rerank only (ASR / vision / chat are DISABLED).
    assert {c.value for c in n.capabilities} == {"GENERATE", "EMBEDDINGS", "RERANK"}
    assert n.declares_model("compact-generation-8") and n.declares_model("fr-en-retrieval")
    # The 12 GB candidate is not installed in Compact: never declared, never a fallback.
    assert not n.declares_model("compact-generation")
    assert cfg.upstream_allowlist == ["ai-gateway:8100"]
    assert cfg.job_affinity.backend == "none"


def test_valid_multinode_example_is_fictional() -> None:
    cfg = load_pool_config(MULTINODE)
    assert cfg.environment == "staging"
    assert [n.node_id for n in cfg.nodes] == ["gateway-a", "gateway-b"]
    assert (
        cfg.job_affinity.backend == "redis" and cfg.job_affinity.redis_url_env == "AIGP_REDIS_URL"
    )
    text = MULTINODE.read_text()
    assert "EXAMPLE_ONLY" in text and "MULTINODE_NOT_HARDWARE_QUALIFIED" in text


def test_valid_local_config() -> None:
    cfg = load_pool_config(LOCAL)
    assert cfg.environment == "development" and cfg.job_affinity.backend == "sqlite"


def test_staging_and_production_are_isolated() -> None:
    check_environment_pair(load_pool_config(STAGING), load_pool_config(PRODUCTION))


def test_structured_extract_capability_cannot_be_declared() -> None:
    """The Open contract has no StructX route: a node declaring it is refused."""
    with pytest.raises(ValidationError, match="no route in the Open Gateway contract"):
        build_config(nodes=[node(capabilities=["GENERATE", "STRUCTURED_EXTRACT"])])


def test_pair_sharing_an_upstream_is_rejected() -> None:
    stg = build_config(env="staging", nodes=[node(url="http://gw.invalid:8100")])
    prd = build_config(
        env="production",
        nodes=[node("gw-prod", env="production", url="http://gw.invalid:8100")],
    )
    with pytest.raises(ValueError, match="share upstream"):
        check_environment_pair(stg, prd)


def test_missing_environment_rejected() -> None:
    data = raw_config()
    del data["environment"]
    with pytest.raises(ValidationError):
        PoolConfig.model_validate(data)


def test_missing_environment_setting_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POOL_ENVIRONMENT", raising=False)
    monkeypatch.setenv("POOL_CONFIG_FILE", str(STAGING))
    with pytest.raises(ConfigurationInvalidError, match="POOL_ENVIRONMENT"):
        load_settings()


def test_duplicate_node_id_rejected() -> None:
    nodes = [node("a", url="http://x.invalid:1"), node("a", url="http://y.invalid:2")]
    with pytest.raises(ValidationError, match="duplicate node_id"):
        build_config(nodes=nodes, routing={"policy": "WEIGHTED_ROUND_ROBIN"})


@pytest.mark.parametrize(
    "url",
    [
        "ftp://gw.invalid:8200",
        "gw.invalid:8200",
        "http://",
        "http://user:pw@gw.invalid:8200",
        "http://gw.invalid:8200/v1",
        "http://gw.invalid:8200?x=1",
        "http://gw.invalid:99999",
        "http://gw.invalid:0",
    ],
)
def test_invalid_node_url_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        build_config(nodes=[node(url=url)])


def test_trailing_slash_normalized() -> None:
    cfg = build_config(nodes=[node(url="http://gw.invalid:8200/")])
    assert cfg.nodes[0].base_url == "http://gw.invalid:8200"


def test_cross_environment_node_rejected() -> None:
    with pytest.raises(ValidationError, match="cross-environment"):
        build_config(env="production", nodes=[node(env="staging")])


def test_gateway_instance_mismatch_rejected() -> None:
    bad = node(env="production", gateway_instance="staging")
    with pytest.raises(ValidationError, match="cross-environment"):
        build_config(env="production", nodes=[bad])


@pytest.mark.parametrize("policy", ["RANDOM", "round_robin", ""])
def test_unsupported_routing_policy_rejected(policy: str) -> None:
    with pytest.raises(ValidationError):
        build_config(routing={"policy": policy})


@pytest.mark.parametrize("policy", ["LEAST_BUSY", "CAPABILITY_AWARE", "TARGETED_NODE"])
def test_defined_but_unimplemented_policy_rejected(policy: str) -> None:
    with pytest.raises(ValidationError, match="ROUTING_NOT_READY"):
        build_config(routing={"policy": policy})


def test_missing_required_node_safe_failure() -> None:
    with pytest.raises(ValidationError):
        build_config(nodes=[])
    with pytest.raises(ValidationError, match="no enabled node"):
        build_config(nodes=[node(enabled=False)])


def test_single_node_policy_requires_exactly_one_enabled() -> None:
    nodes = [node("a", url="http://a.invalid:1"), node("b", url="http://b.invalid:2")]
    with pytest.raises(ValidationError, match="exactly one enabled node"):
        build_config(nodes=nodes)


def test_inferential_probe_path_rejected() -> None:
    with pytest.raises(ValidationError, match="non-inferential"):
        build_config(health={"probe_path": "/health/ready"})


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        build_config(nodes=[node(api_url="http://x.invalid")])


def test_model_capability_must_be_declared_by_node() -> None:
    bad = node(
        capabilities=["CHAT"],
        models=[{"model_id": "legacy-768", "capabilities": ["EMBEDDINGS"]}],
    )
    with pytest.raises(ValidationError, match="does not declare"):
        build_config(nodes=[bad])


def test_settings_environment_must_match_file(tmp_path: Path) -> None:
    settings = Settings(environment="production", config_file=STAGING)
    with pytest.raises(ConfigurationInvalidError, match="declares environment=staging"):
        load_for_settings(settings)
    settings = Settings(environment="staging", config_file=COMPACT)
    with pytest.raises(ConfigurationInvalidError, match="declares environment=production"):
        load_for_settings(settings)


def test_example_config_refused_when_approval_required() -> None:
    settings = Settings(environment="staging", config_file=STAGING, require_approved_config=True)
    with pytest.raises(ConfigurationInvalidError, match="approved"):
        load_for_settings(settings)


def test_invalid_yaml_and_missing_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("nodes: [unterminated")
    with pytest.raises(ConfigurationInvalidError, match="not valid YAML"):
        load_pool_config(bad)
    with pytest.raises(ConfigurationInvalidError, match="not found"):
        load_pool_config(tmp_path / "absent.yaml")


def test_cli_validate_exit_codes(tmp_path: Path) -> None:
    good = write_yaml(tmp_path, raw_config(), "good.yaml")
    bad = write_yaml(
        tmp_path, raw_config(env="production", nodes=[node(env="staging")]), "bad.yaml"
    )
    assert cli.main(["validate", str(good)]) == 0
    assert cli.main(["validate", str(good), str(bad)]) == 1
    assert cli.main(["check-pair", str(STAGING), str(PRODUCTION)]) == 0


def test_affinity_backend_environment_rules() -> None:
    # memory is not durable: development only.
    with pytest.raises(ValidationError, match="development only"):
        build_config(job_affinity={"backend": "memory"})
    # sqlite: allowed for local qualification (status example), never approved.
    build_config(job_affinity={"backend": "sqlite", "sqlite_path": "/tmp/x.sqlite3"})
    with pytest.raises(ValidationError, match="never be part of an approved"):
        build_config(status="approved", job_affinity={"backend": "sqlite", "sqlite_path": "/tmp/x"})
    with pytest.raises(ValidationError, match="requires sqlite_path"):
        build_config(job_affinity={"backend": "sqlite"})


def test_transport_and_probe_settings_bounded() -> None:
    cfg = build_config()
    assert cfg.transport.keepalive_expiry_seconds < 5  # below uvicorn keep-alive of the Gateway
    assert cfg.health.background_probing is False
    with pytest.raises(ValidationError):
        build_config(health={"probe_interval_seconds": 1})
    with pytest.raises(ValidationError, match="non-inferential"):
        build_config(health={"probe_path": "/health/ready", "background_probing": True})


@pytest.mark.parametrize("path", [COMPACT])
def test_example_capabilities_match_published_runtime(path: Path) -> None:
    """Logical ids must not declare a capability the Gateway does not publish
    (regression: a chat-only model was once wrongly declared GENERATE; the Gateway
    answers 400). Checked against the LIVE Compact runtime snapshot."""

    from tests.conftest import gateway_fixture

    published = {
        m["model_logical"]: set(m["capabilities"])
        for m in gateway_fixture("health-runtime.compact.json")["llm_models"]
    }
    mapping = {"GENERATE": "completion", "CHAT": "chat", "VISION": "vision"}
    [node] = load_pool_config(path).nodes
    for model in node.models:
        if model.model_id in published:
            for cap in model.capabilities:
                assert mapping[cap.value] in published[model.model_id], (model.model_id, cap)
        elif ":" in model.model_id:  # concrete Ollama name
            assert "CHAT" not in {c.value for c in model.capabilities}, model.model_id


@pytest.mark.parametrize("path", [COMPACT, MULTINODE])
def test_deployment_examples_valid_once_approved(path: Path) -> None:
    """The example configs are statically validatable as `approved` (allowlist +
    durable or disabled affinity) — validation only, no deployment."""

    import yaml

    data = yaml.safe_load(path.read_text())
    data["status"] = "approved"
    cfg = PoolConfig.model_validate(data)
    assert cfg.job_affinity.backend in {"redis", "none"} and cfg.upstream_allowlist


def test_example_configs_carry_no_private_destination() -> None:
    """Open edition: only Compose service names / documentation ranges, never a private host."""
    import re

    for path in (COMPACT, MULTINODE, LOCAL):
        text = path.read_text()
        assert not re.search(r"192\.168\.|\b10\.\d+\.\d+\.\d+|\.lan\b|saint", text, re.I), path
