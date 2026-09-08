#!/usr/bin/env bash
# Build the pinned vLLM release with SM12x NVFP4 and fused M-RoPE support.
# Usage: ./build.sh | SPARK=1 ./build.sh | GHCR_OWNER=owner ./build.sh --push
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
source "$ROOT/release.env"
[[ $# == 0 || ( $# == 1 && "$1" == --push ) ]] || {
  echo "usage: $0 [--push]" >&2; exit 2;
}
for cmd in docker git sha256sum; do
  command -v "$cmd" >/dev/null || { echo "error: required command not found: $cmd" >&2; exit 1; }
done
if [[ "${1:-}" == --push ]]; then
  : "${GHCR_OWNER:?set GHCR_OWNER to your GitHub user or organization}"
  : "${GHCR_PAT:?set GHCR_PAT to a token with write:packages}"
fi
if [[ -n "${VISION_MROPE:-}" ]]; then
  echo "error: VISION_MROPE was removed; fused M-RoPE is included in every build" >&2
  exit 2
fi
if [[ "${SPARK:-0}" == 1 ]]; then
  BUILD_PLATFORM=linux/arm64
  TORCH_ARCH_LIST=12.1
else
  BUILD_PLATFORM=linux/amd64
  TORCH_ARCH_LIST=12.0
fi
LOCAL_IMG="${LOCAL_IMG:-$DEFAULT_IMAGE}"
BUILD_MAX_JOBS="${MAX_JOBS:-3}"
BUILD_NVCC_THREADS="${NVCC_THREADS:-2}"
for value in "$BUILD_MAX_JOBS" "$BUILD_NVCC_THREADS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "error: build concurrency must be positive integers" >&2; exit 2; }
done
WORK="$(mktemp -d /tmp/vllm-dflash2-build.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
"$ROOT/scripts/prepare-source.sh" "$WORK/vllm"
manifest_sha="$(sha256sum "$ROOT/SHA256SUMS" | cut -d' ' -f1)"
echo "Building $LOCAL_IMG for $BUILD_PLATFORM (SM${TORCH_ARCH_LIST})"
docker build \
  --platform "$BUILD_PLATFORM" \
  --target vllm-openai-nonroot \
  -f "$WORK/vllm/docker/Dockerfile" \
  --build-arg CUDA_VERSION="$CUDA_VERSION" \
  --build-arg FLASHINFER_VERSION="$FLASHINFER_VERSION" \
  --build-arg torch_cuda_arch_list="$TORCH_ARCH_LIST" \
  --build-arg max_jobs="$BUILD_MAX_JOBS" \
  --build-arg nvcc_threads="$BUILD_NVCC_THREADS" \
  --label "org.opencontainers.image.source=$SOURCE_URL" \
  --label "org.opencontainers.image.version=$RELEASE_TAG" \
  --label "org.opencontainers.image.licenses=Apache-2.0" \
  --label "ai.bickford.vllm.upstream-revision=$VLLM_COMMIT" \
  --label "ai.bickford.vllm.manifest-sha256=$manifest_sha" \
  -t "$LOCAL_IMG" "$WORK/vllm"

echo "Built $LOCAL_IMG. Set IMAGE=$LOCAL_IMG in .env, then run ./start.sh."
echo "GPU validation: ./verify.sh --full; ./verify.sh --vision (with the sidecar running)."
if [[ "${1:-}" == --push ]]; then
  PUSH_IMG="ghcr.io/${GHCR_OWNER}/vllm-sm12x-nvfp4-dflash2:${RELEASE_TAG}"
  docker tag "$LOCAL_IMG" "$PUSH_IMG"
  echo "$GHCR_PAT" | docker login ghcr.io --username "$GHCR_OWNER" --password-stdin
  docker push "$PUSH_IMG"
  echo "Pushed $PUSH_IMG. Record the registry digest before using it as a release pin."
fi
