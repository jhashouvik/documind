# DocuMind - an agentic RAG platform on Kubernetes

DocuMind answers questions about your documents with citations. It is built as
two Python microservices. The repository shows the whole path from code to a
production-style Kubernetes release: plain manifests, then GitOps with Argo CD,
blue-green and canary releases with Argo Rollouts, node autoscaling with Karpenter
(simulated with KWOK), and the three pillars of observability.

```
 browser ──► ingress-nginx ──► agent-service (Service 2) ──► OpenRouter chat models
                                   │  tool calls                 (tool calling, streaming)
                                   ▼
                          knowledge-service (Service 1) ──► OpenRouter embeddings API
                                   │
                                   └──► Qdrant (one collection per embedding model)
 agent-service ──► Redis (conversation memory, rate limits)
```

| Service | What it does | Port |
|---|---|---|
| **knowledge-service** | Upload PDF/MD/TXT → parse → chunk (size/overlap per upload) → embed with an OpenRouter embedding model → store in Qdrant; semantic search API | 8001 |
| **agent-service** | Tool-calling agent (search, list documents, calculator) over OpenRouter; streams answers (SSE) with citations; Redis memory; rate limiting; web UI with a Settings panel | 8002 |

## What you can configure in the UI (left panel, Settings tab)

| Group | Setting | Validated against |
|---|---|---|
| Ingestion | embedding model, chunk size, chunk overlap | `EMBEDDING_MODELS`, `MIN/MAX_CHUNK_SIZE` |
| Retrieval | top-k, similarity threshold | `MAX_SEARCH_TOP_K`, 0 - 0.95 |
| Generation | LLM model, temperature, max tokens, max agent steps, prompt version | `LLM_MODELS`, `MAX_TOKENS_LIMIT`, `MAX_STEPS_LIMIT` |

Every request carries its settings; the servers only accept values inside the
allow-lists and limits configured in the ConfigMaps.

## Highlights

- Agent loop with parallel tool calls, streamed tool-call reassembly, step limit, graceful degradation
- Remote embeddings (OpenRouter `/api/v1/embeddings`) with batching, retries on 429/5xx, clear 401/402 errors; a separate Qdrant collection per embedding model
- Kubernetes: probes, resources, HPA, PDB, topology spread, non-root + read-only root FS, NetworkPolicies, Secrets
- GitOps (Argo CD), blue-green with an automated self-test gate and canary with a Prometheus analysis (Argo Rollouts), Karpenter NodePool (KWOK), KEDA event-driven autoscaling, Istio ambient service mesh (mTLS, identity policies)
- Observability: Prometheus metrics (tokens, time to first token, embeddings, tool calls), Fluent Bit → Loki logs, OpenTelemetry → Jaeger traces with GenAI semantic conventions, Grafana over all three
- 84 unit/API tests (fake LLM, fake Redis, in-memory Qdrant, mocked OpenRouter); retrieval evaluation (hit@k, MRR); CI with manifest validation and Trivy

## Layout

```
services/knowledge-service/   Service 1 (FastAPI) + Dockerfile + tests + release self-test
services/agent-service/       Service 2 (FastAPI) + web UI + Dockerfile + tests
k8s/00..50                    namespace, Qdrant, Service 1, Redis, Service 2, NetworkPolicies
k8s/60-canary                 manual canary with ingress-nginx weights
k8s/70-observability          ServiceMonitor, alert rules, Grafana dashboard
k8s/observability             OTel Collector + Jaeger manifests, Helm values for Prometheus/Grafana, Loki, Fluent Bit
k8s/gitops                    what Argo CD deploys: Rollouts (blue-green + canary), analysis templates
k8s/argocd                    Argo CD Application(s) + UI Ingress
k8s/karpenter                 NodePool + KWOKNodeClass + inflate demo
k8s/keda                      KEDA ScaledObject (component) + in-cluster load test (Day 13)
k8s/istio                     Istio ambient mesh: policies, waypoint, HTTPRoute (component, Day 14)
scripts/                      build-and-load, ingest, smoke test, traffic, load test, retrieval eval
sample-docs/                  fictional "Nimbus Re" documents + eval/golden.jsonl
docker-compose.yml            the whole stack locally (optional Jaeger profile)
```

## Quick start (kind cluster "learn" with ingress-nginx)

```bash
read -rsp "OpenRouter key: " OPENROUTER_API_KEY; echo
kubectl create namespace documind
kubectl create secret generic documind-llm -n documind --from-literal=OPENROUTER_API_KEY="$OPENROUTER_API_KEY"
make build-knowledge && make deploy-data && make deploy-knowledge && make ingest
make build-agent && make deploy-agent && make smoke
# open http://documind.localtest.me:8080 through your SSH tunnel
```

The day-by-day tutorial (what, why, commands, expected output, troubleshooting)
is the PDF that accompanies this repository.

Sample documents describe a fictional company (Nimbus Re) and are for training only.
