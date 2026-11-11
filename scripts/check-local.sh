#!/usr/bin/env bash
# Full local quality gate (developer workstation or CI). Never deploys, never contacts a real
# Gateway. Fails explicitly on the first error.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
VENV="${VENV:-.venv}"
PY="$VENV/bin/python"
[[ -x "$PY" ]] || { echo "Missing $VENV. Run: python3.12 -m venv .venv && .venv/bin/pip install --require-hashes -r requirements-dev.lock && .venv/bin/pip install --no-deps -e ." >&2; exit 1; }

step() { echo; echo "==> $*"; }

step "git structure"
if git rev-parse --git-dir >/dev/null 2>&1; then
  [[ "$(git rev-parse --show-toplevel)" == "$PWD" ]] || { echo "FAIL: nested repository"; exit 1; }
  # Must never be wired to the AI Gateway or RAGatm repositories.
  if git remote -v | grep -E '/(ai-gateway|ragatm)(\.git)?( |$)'; then
    echo "FAIL: remote points to another project's repository"; exit 1
  fi
  echo "GIT_STRUCTURE=PASS"
fi

step "lint (ruff check + format --check)"
"$VENV/bin/ruff" check src tests
"$VENV/bin/ruff" format --check src tests
step "types (mypy strict on critical modules)"
"$VENV/bin/mypy"
step "config validation"
PYTHON="$PY" ./scripts/validate-config.sh
step "secret checks"
./scripts/check-secrets.sh
step "real Redis for affinity tests"
if [[ -z "${AIGP_TEST_REDIS_URL:-}" ]]; then
  if command -v docker >/dev/null && docker image inspect redis:7.2-alpine >/dev/null 2>&1; then
    RPW=$(python3 -c "import secrets;print(secrets.token_hex(16))")
    RNAME="aigp-check-redis-$$"
    docker run -d --rm --name "$RNAME" -p 127.0.0.1::6379 redis:7.2-alpine \
      redis-server --requirepass "$RPW" --save "" --appendonly no >/dev/null
    trap 'docker rm -f "$RNAME" >/dev/null 2>&1 || true' EXIT
    RPORT=$(docker port "$RNAME" 6379/tcp | head -1 | cut -d: -f2)
    for _ in $(seq 1 50); do docker exec "$RNAME" redis-cli -a "$RPW" --no-auth-warning ping >/dev/null 2>&1 && break; sleep 0.1; done
    export AIGP_TEST_REDIS_URL="redis://:$RPW@127.0.0.1:$RPORT/0" AIGP_TEST_REDIS_CONTAINER="$RNAME"
    echo "isolated Redis $RNAME on 127.0.0.1:$RPORT (removed on exit)"
  else
    echo "WARNING: no Redis available: real Redis tests will be SKIPPED (REDIS_REAL_TESTS=NOT_RUN)"
  fi
fi

step "tests"
"$VENV/bin/pytest" -q -rs
step "lock consistency"
"$VENV/bin/pip" install --dry-run --require-hashes --no-deps -r requirements.lock -q >/dev/null
echo "DEPENDENCIES_LOCKED=PASS"

step "local start (loopback, ephemeral port)"
PORT="${SMOKE_PORT:-18081}"
SQLITE_DIR=$(mktemp -d)
sed "s#var/job-affinity.sqlite3#$SQLITE_DIR/a.sqlite3#" config/local.example.yaml > "$SQLITE_DIR/local.yaml"
POOL_ENVIRONMENT=development POOL_CONFIG_FILE="$SQLITE_DIR/local.yaml" POOL_BIND_PORT="$PORT" \
  "$VENV/bin/ai-gateway-pool" >/tmp/aigp-smoke-$$.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true; rm -rf /tmp/aigp-smoke-$$.log "$SQLITE_DIR"; [[ -n "${RNAME:-}" ]] && docker rm -f "$RNAME" >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 50); do
  curl -fsS "http://127.0.0.1:$PORT/health/live" >/dev/null 2>&1 && break
  sleep 0.1
done
curl -fsS "http://127.0.0.1:$PORT/health/live" | grep -q '"live"'
curl -fsS "http://127.0.0.1:$PORT/pool/health/ready" | grep -q '"ready"'
# config/local.example.yaml points at 127.0.0.1:18299 where nothing listens: the relay must
# answer an explicit pool error, never a fake success.
hdr=$(curl -s -o /dev/null -D - -X POST -H 'Content-Type: application/json' -d '{}' \
  "http://127.0.0.1:$PORT/v1/generate" | tr -d '\r')
echo "$hdr" | grep -q '^HTTP/1.1 503' || { echo "FAIL: expected 503 from an unreachable node"; exit 1; }
echo "$hdr" | grep -qi '^x-aigp-error: UPSTREAM_UNAVAILABLE' || { echo "FAIL: missing UPSTREAM_UNAVAILABLE"; exit 1; }
echo "APPLICATION_STARTS_LOCALLY=PASS HEALTH_ENDPOINTS=PASS"

step "package build"
"$VENV/bin/python" -m build --wheel --no-isolation --outdir "/tmp/aigp-dist-$$" . >/dev/null
ls "/tmp/aigp-dist-$$"/*.whl && rm -rf "/tmp/aigp-dist-$$"
echo "PACKAGE_BUILD=PASS"

echo; echo "CHECK_LOCAL=PASS"
