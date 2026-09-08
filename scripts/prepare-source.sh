#!/usr/bin/env bash
# Prepare a clean, pinned upstream checkout with the maintained patches.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/release.env"
[[ $# == 1 ]] || { echo "usage: $0 NEW_SOURCE_DIRECTORY" >&2; exit 2; }
source_dir="$1"
[[ ! -e "$source_dir" ]] || { echo "error: destination already exists: $source_dir" >&2; exit 2; }
(cd "$ROOT" && sha256sum --check SHA256SUMS)
git clone --depth 1 --branch "$VLLM_VERSION" \
  https://github.com/vllm-project/vllm.git "$source_dir"
actual_commit="$(git -C "$source_dir" rev-parse HEAD)"
[[ "$actual_commit" == "$VLLM_COMMIT" ]] || {
  echo "error: $VLLM_VERSION resolves to $actual_commit; expected $VLLM_COMMIT" >&2
  exit 1
}
while IFS= read -r patch_name; do
  echo "Applying $patch_name"
  git -C "$source_dir" apply --index "$ROOT/patches/$patch_name"
done < "$ROOT/patches/series"
git -C "$source_dir" diff --cached --check
echo "Prepared vLLM $VLLM_VERSION at $source_dir"
