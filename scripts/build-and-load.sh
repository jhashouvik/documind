#!/usr/bin/env bash
# Build a service image on the VM and copy it into every kind node.
#   scripts/build-and-load.sh knowledge-service 1.0.0
#   scripts/build-and-load.sh agent-service 1.1.0
# kind nodes have their own container image store, separate from the VM's
# Docker, so a freshly built image must be "kind load"-ed before pods can use it.
set -euo pipefail
SERVICE="${1:?usage: $0 <knowledge-service|agent-service> [tag]}"
TAG="${2:-1.0.0}"
CLUSTER="${KIND_CLUSTER:-learn}"
IMAGE="documind/${SERVICE}:${TAG}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ">> running unit tests inside Docker (test stage)"
docker build --target test -t "documind/${SERVICE}:test" "${ROOT}/services/${SERVICE}"

echo ">> building ${IMAGE}"
docker build --build-arg APP_VERSION="${TAG}" -t "${IMAGE}" "${ROOT}/services/${SERVICE}"
docker image ls "documind/${SERVICE}"

echo ">> loading ${IMAGE} into kind cluster '${CLUSTER}'"
kind load docker-image "${IMAGE}" --name "${CLUSTER}"
echo ">> done. Pods can now use image: ${IMAGE} (imagePullPolicy: IfNotPresent)"
