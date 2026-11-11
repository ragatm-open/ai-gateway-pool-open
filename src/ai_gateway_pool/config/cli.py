"""Offline configuration validator: ``ai-gateway-pool-config validate FILE...``.

Never contacts any node. Exit code 0 = all files valid, 1 = at least one invalid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_gateway_pool.config.schema import check_environment_pair
from ai_gateway_pool.config.settings import load_pool_config
from ai_gateway_pool.errors import ConfigurationInvalidError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai-gateway-pool-config")
    sub = parser.add_subparsers(dest="command", required=True)
    val = sub.add_parser("validate", help="validate one or more pool YAML files")
    val.add_argument("files", nargs="+", type=Path)
    pair = sub.add_parser("check-pair", help="verify staging/production isolation")
    pair.add_argument("staging", type=Path)
    pair.add_argument("production", type=Path)
    args = parser.parse_args(argv)

    if args.command == "validate":
        failures = 0
        for file in args.files:
            try:
                cfg = load_pool_config(file)
            except ConfigurationInvalidError as exc:
                failures += 1
                print(f"INVALID {file}: {exc.message}", file=sys.stderr)
                continue
            enabled = sum(1 for n in cfg.nodes if n.enabled)
            print(
                f"VALID   {file}: environment={cfg.environment} status={cfg.status} "
                f"policy={cfg.routing.policy} nodes={len(cfg.nodes)} enabled={enabled}"
            )
        return 1 if failures else 0

    try:
        check_environment_pair(load_pool_config(args.staging), load_pool_config(args.production))
    except (ConfigurationInvalidError, ValueError) as exc:
        print(f"PAIR INVALID: {exc}", file=sys.stderr)
        return 1
    print("PAIR VALID: staging and production share no upstream and no node_id")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
