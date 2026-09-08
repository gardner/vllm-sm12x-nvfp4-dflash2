#!/usr/bin/env bash
# Check local artifacts and optionally apply patches to a fresh upstream tree.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
sha256sum --check SHA256SUMS
bash -n build.sh start.sh stop.sh status.sh verify.sh scripts/prepare-source.sh scripts/check-release.sh
python3 -m py_compile sidecar.py
python3 scripts/check-config.py
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  docker compose --env-file .env.example -f compose.yaml config --quiet
  docker compose --env-file .env.example -f compose.yaml --profile vision config --quiet
else
  echo "Docker Compose unavailable; render checks skipped."
fi
if [[ "${CHECK_PATCH_APPLY:-0}" == 1 ]]; then
  work="$(mktemp -d /tmp/vllm-dflash2-check.XXXXXX)"
  trap 'rm -rf "$work"' EXIT
  ./scripts/prepare-source.sh "$work/vllm"
  python3 -m compileall -q "$work/vllm/vllm" "$work/vllm/tests"
  VLLM_SOURCE_DIR="$work/vllm" python3 -m pytest -q tests/test_patched_source.py
fi
echo "release integrity: PASS (GPU validation is separate)"
