#!/usr/bin/env bash
# Basic secret hygiene on tracked (or all) files. Not a replacement for a dedicated scanner
# (the OPEN programme replays gitleaks/trufflehog before any publication); fails on obvious leaks.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if git rev-parse --git-dir >/dev/null 2>&1; then
  mapfile -t FILES < <(git ls-files --cached --others --exclude-standard)
else
  mapfile -t FILES < <(find . -type f -not -path './.venv/*' -not -path './.git/*' -not -path '*/__pycache__/*' | sed 's|^\./||')
fi

fail=0
for f in "${FILES[@]}"; do
  case "$f" in
    .env|*/.env|.env.*) [[ "$f" == ".env.example" || "$f" == */.env.example ]] || { echo "FAIL: env file tracked: $f"; fail=1; } ;;
    *.pem|*.key|*id_rsa*|*id_ed25519*) echo "FAIL: key material tracked: $f"; fail=1 ;;
  esac
done

PATTERNS='-----BEGIN [A-Z ]*PRIVATE KEY-----|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}'
# Test fixtures intentionally contain fake tokens to prove redaction (triaged in the OPEN allowlists).
if hits=$(printf '%s\0' "${FILES[@]}" | xargs -0 grep -nIE "$PATTERNS" -- 2>/dev/null | grep -v '^tests/' ); then
  echo "FAIL: possible secrets:"; echo "$hits"; fail=1
fi
# Non-empty assignments of secret-looking variables in example files.
mapfile -t EXAMPLES < <(find . -name '.env.example' -not -path './.venv/*' | sed 's|^\./||')
if grep -nE '^(POOL_ADMIN_TOKEN|.*PASSWORD|.*SECRET|.*API_KEY|AIGP_REDIS_URL)=.+' "${EXAMPLES[@]}"; then
  echo "FAIL: example env files must not carry values for secrets"; fail=1
fi
[[ $fail -eq 0 ]] && echo "SECURITY_BASIC_CHECKS=PASS"
exit $fail
