#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   IMAGE_REPO=ghcr.io/acme/vllm-engine IMAGE_TAG=phase1 ./scripts/build_image.sh
#   IMAGE_REPO=ghcr.io/acme/vllm-engine IMAGE_TAG=phase1 PUSH=1 ./scripts/build_image.sh

IMAGE_REPO="${IMAGE_REPO:-vllm-engine}"
IMAGE_TAG="${IMAGE_TAG:-phase1}"
IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"
PUSH="${PUSH:-0}"

BUILD_SHA="${BUILD_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo dev)}"
BUILD_TIME="${BUILD_TIME:-$(date -u +"%Y-%m-%dT%H:%M:%SZ")}"

echo "Building image: ${IMAGE}"
docker build \
  -f services/engine/Dockerfile \
  --build-arg BUILD_SHA="${BUILD_SHA}" \
  --build-arg BUILD_TIME="${BUILD_TIME}" \
  -t "${IMAGE}" \
  .

echo "Build complete: ${IMAGE}"
echo "BUILD_SHA=${BUILD_SHA}"
echo "BUILD_TIME=${BUILD_TIME}"

if [[ "${PUSH}" == "1" ]]; then
  echo "Pushing image: ${IMAGE}"
  docker push "${IMAGE}"
fi
