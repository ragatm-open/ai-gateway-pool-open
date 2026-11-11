#!/usr/bin/env bash
# Offline validation of every versioned pool config (Open edition). Never contacts any node.
# Fails (non-zero) on the first problem.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

"$PY" -m ai_gateway_pool.config.cli validate config/*.yaml
# The fictional multi-node (staging) example and the Compact (production) profile must never
# share an upstream or a node id.
"$PY" -m ai_gateway_pool.config.cli check-pair config/multinode.example.yaml config/compact.example.yaml

# Example configs must stay non-deployable.
for f in config/*.example.yaml; do
  grep -q '^status: example$' "$f" || { echo "FAIL: $f must keep 'status: example'" >&2; exit 1; }
done
# The single-node profile targets the Compose service name of AI Gateway Open, nothing else.
grep -q 'base_url: http://ai-gateway:8100$' config/compact.example.yaml \
  || { echo "FAIL: compact example must target http://ai-gateway:8100" >&2; exit 1; }
# The multi-node example is illustrative only and must say so.
grep -q 'MULTINODE_NOT_HARDWARE_QUALIFIED' config/multinode.example.yaml \
  || { echo "FAIL: multinode example must carry the EXAMPLE_ONLY marker" >&2; exit 1; }
# No private host, LAN address or personal path anywhere in the examples.
if grep -nEi '192\.168\.|(^|[^0-9])10\.[0-9]+\.[0-9]+\.[0-9]+|\.lan\b|saint|/home/' config/*.yaml; then
  echo "FAIL: private destination or path in a config example" >&2; exit 1
fi
echo "CONFIG_VALIDATION=PASS"
