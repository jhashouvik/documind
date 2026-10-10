#!/usr/bin/env bash
# Create the kind cluster the Makefile expects ("learn"), ready for DocuMind:
#   scripts/kind-up.sh
#
#   * 1 control-plane + 2 workers  -> topology spread puts agent pods on different nodes
#   * host port 8080 -> ingress     -> http://documind.localtest.me:8080 (UI)
#                                      http://kb.localtest.me:8080       (knowledge-service)
#     (*.localtest.me resolves to 127.0.0.1 - no /etc/hosts edits needed)
#   * ingress-nginx                 -> the Ingress objects in k8s/
#   * metrics-server                -> `kubectl top` and the CPU-based HPA. Without it
#                                      the HPA shows <unknown> and never scales.
# Skip this if you already have the cluster from the tutorial.
set -euo pipefail
CLUSTER="${KIND_CLUSTER:-learn}"
INGRESS_NGINX="controller-v1.12.1"
METRICS_SERVER="v0.7.2"

if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  echo ">> kind cluster '$CLUSTER' already exists"
else
  echo ">> creating kind cluster '$CLUSTER'"
  cat <<EOF | kind create cluster --name "$CLUSTER" --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "ingress-ready=true"
    extraPortMappings:
      - {containerPort: 80, hostPort: 8080, protocol: TCP}
  - role: worker
  - role: worker
EOF
fi

echo ">> ingress-nginx ${INGRESS_NGINX}"
kubectl apply -f "https://raw.githubusercontent.com/kubernetes/ingress-nginx/${INGRESS_NGINX}/deploy/static/provider/kind/deploy.yaml"

echo ">> metrics-server ${METRICS_SERVER} (kind kubelets use self-signed certs: --kubelet-insecure-tls)"
kubectl apply -f "https://github.com/kubernetes-sigs/metrics-server/releases/download/${METRICS_SERVER}/components.yaml"
kubectl -n kube-system patch deployment metrics-server --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' 2>/dev/null || true

echo ">> waiting for ingress-nginx and metrics-server"
kubectl -n ingress-nginx wait --for=condition=ready pod -l app.kubernetes.io/component=controller --timeout=180s
kubectl -n kube-system rollout status deployment/metrics-server --timeout=180s
kubectl get nodes -o wide
echo ">> ready. Next: make secrets && make deploy-all   (see README)"
