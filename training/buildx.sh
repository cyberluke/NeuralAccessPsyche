#!/usr/bin/env bash
set -euo pipefail
: "${IMAGE_NAMESPACE:?set IMAGE_NAMESPACE to a private Docker Hub namespace}"
: "${SOURCE_COMMIT:?set SOURCE_COMMIT to an immutable git SHA}"
IMAGE="${IMAGE_NAMESPACE}/nram-training:qwen3-14b-lora-${SOURCE_COMMIT}"
docker buildx build --platform linux/amd64 --sbom=true --provenance=true --push --tag "$IMAGE" --label "org.opencontainers.image.revision=${SOURCE_COMMIT}" -f training/Dockerfile .
docker buildx imagetools inspect "$IMAGE" --format '{{json .Manifest}}' > "artifacts/training/${SOURCE_COMMIT}.manifest.json"
