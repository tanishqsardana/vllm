#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   IMAGE_REPO=vllm-combined IMAGE_TAG=phase2 ./scripts/build_combined_image.sh
#   IMAGE_REPO=docker.io/<user>/<repo> IMAGE_TAG=phase2 PUSH=1 ./scripts/build_combined_image.sh

IMAGE_REPO="${IMAGE_REPO:-vllm-combined}"
IMAGE_TAG="${IMAGE_TAG:-phase2}"
IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"
PUSH="${PUSH:-0}"
TARGET_PLATFORM="${TARGET_PLATFORM:-linux/amd64}"

BUILD_SHA="${BUILD_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo dev)}"
BUILD_TIME="${BUILD_TIME:-$(date -u +"%Y-%m-%dT%H:%M:%SZ")}" 

if ! docker buildx version >/dev/null 2>&1; then
  echo "docker buildx is required but not available." >&2
  exit 1
fi

echo "Building combined image: ${IMAGE}"
echo "Target platform: ${TARGET_PLATFORM}"

if [[ "${PUSH}" == "1" ]]; then
  docker buildx build \
    --platform "${TARGET_PLATFORM}" \
    -f services/combined/Dockerfile \
    --build-arg BUILD_SHA="${BUILD_SHA}" \
    --build-arg BUILD_TIME="${BUILD_TIME}" \
    -t "${IMAGE}" \
    --push \
    .
else
  docker buildx build \
    --platform "${TARGET_PLATFORM}" \
    -f services/combined/Dockerfile \
    --build-arg BUILD_SHA="${BUILD_SHA}" \
    --build-arg BUILD_TIME="${BUILD_TIME}" \
    -t "${IMAGE}" \
    --load \
    .
fi

echo "Build complete: ${IMAGE}"
echo "BUILD_SHA=${BUILD_SHA}"
echo "BUILD_TIME=${BUILD_TIME}"
