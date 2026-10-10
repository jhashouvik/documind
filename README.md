# DocuMind - an agentic RAG platform on Kubernetes

DocuMind answers questions about your documents with citations. It is built as
two Python microservices; the agent is a LangGraph multi-agent system (supervisor,
corrective-RAG researcher, parallel planner, analyst, librarian with human approval,
writer and fact checker). The repository shows the whole path from code to a
production-style Kubernetes release: plain manifests, then GitOps with Argo CD,
blue-green and canary releases with Argo Rollouts, node autoscaling with Karpenter
(simulated with KWOK), and the three pillars of observability.

```
 browser ──► ingress-nginx ──► agent-service (Service 2) ──► OpenRouter chat models
                                   │  LangGraph agents           (structured output, streaming)
                                   ▼
                          knowledge-service (Service 1) ──► OpenRouter embeddings API
                                   │
                                   └──► Qdrant (one collection per embedding model)
 agent-service ──► Postgres (LangGraph checkpoints + long-term memory)
               └──► Redis (rate limits)
```

| Service | What it does | Port |
|---|---|---|
| **knowledge-service** | Upload PDF/MD/TXT → parse → chunk (size/overlap per upload) → embed with an OpenRouter embedding model → store in Qdrant; semantic search API | 8001 |
| **agent-service** | LangGraph + LangChain multi-agent system over OpenRouter: supervisor routing, corrective RAG, parallel planning (Send), self-RAG fact check, human-in-the-loop, Pydantic-validated decisions; Postgres checkpoints (stateless pods); streams answers (SSE) with citations; web UI | 8002 |

## What you can configure in the UI (left panel, Settings tab)

| Group | Setting | Validated against |
|---|---|---|
| Ingestion | embedding model, chunk size, chunk overlap | `EMBEDDING_MODELS`, `MIN/MAX_CHUNK_SIZE` |
| Retrieval | top-k, similarity threshold | `MAX_SEARCH_TOP_K`, 0 - 0.95 |
| Generation | LLM model, temperature, max tokens, max agent steps, prompt version | `LLM_MODELS`, `MAX_TOKENS_LIMIT`, `MAX_STEPS_LIMIT` |

Every request carries its settings; the servers only accept values inside the
allow-lists and limits configured in the ConfigMaps.

## Highlights

- LangGraph multi-agent graph: supervisor (Command routing), corrective RAG subgraph, plan-and-execute with parallel Send, self-RAG fact check, create_agent workers with middleware, human-in-the-loop (interrupt/resume), Postgres checkpointer + long-term store, time travel; Pydantic-validated, self-correcting LLM decisions
- Remote embeddings (OpenRouter `/api/v1/embeddings`) with batching, retries on 429/5xx, clear 401/402 errors; a separate Qdrant collection per embedding model
- Kubernetes: stateless agent pods with external state (Postgres StatefulSet), startup/readiness/liveness probes, graceful shutdown sized for long agent turns, HPA, PDB, topology spread, non-root + read-only root FS, NetworkPolicies, Secrets
- GitOps (Argo CD), blue-green with an automated self-test gate and canary with a Prometheus analysis (Argo Rollouts), Karpenter NodePool (KWOK), KEDA event-driven autoscaling, Istio ambient service mesh (mTLS, identity policies)
- Observability: Prometheus metrics (tokens, time to first token, embeddings, tool calls), Fluent Bit → Loki logs, OpenTelemetry → Jaeger traces with GenAI semantic conventions, Grafana over all three
- 116 unit/API tests (38 knowledge-service, 78 agent-service: scripted fake models per agent role, fake Redis, in-memory Qdrant and checkpointer, mocked OpenRouter); retrieval evaluation (hit@k, MRR); CI with manifest validation and Trivy

## Layout

```
services/knowledge-service/   Service 1 (FastAPI) + Dockerfile + tests + release self-test
services/agent-service/       Service 2 (FastAPI) + web UI + Dockerfile + tests
k8s/00..50                    namespace, Qdrant, Service 1, Redis, Postgres, Service 2, NetworkPolicies
k8s/60-canary                 manual canary with ingress-nginx weights
k8s/70-observability          ServiceMonitor, alert rules, Grafana dashboard
k8s/observability             OTel Collector + Jaeger manifests, Helm values for Prometheus/Grafana, Loki, Fluent Bit
k8s/gitops                    what Argo CD deploys: Rollouts (blue-green + canary), analysis templates
k8s/argocd                    Argo CD Application(s) + UI Ingress
k8s/karpenter                 NodePool + KWOKNodeClass + inflate demo
k8s/keda                      KEDA ScaledObject (component) + in-cluster load test (Day 13)
k8s/istio                     Istio ambient mesh: policies, waypoint, HTTPRoute (component, Day 14)
scripts/                      kind-up, secrets, build-and-load, ingest, smoke test, stateless +
                              failover demos, traffic, load test, retrieval eval
sample-docs/                  fictional "Nimbus Re" documents + eval/golden.jsonl
docker-compose.yml            the whole stack locally (optional Jaeger profile)
```

## Quick start (kind cluster "learn" with ingress-nginx)

```bash
make kind-up                 # skip if you already have the cluster (3 nodes, ingress-nginx, metrics-server)
make secrets                 # documind-llm (asks for your OpenRouter key) + documind-postgres (random)
make build-knowledge build-agent
make deploy-all              # Qdrant, knowledge-service, sample docs, Redis, Postgres, agent-service, NetworkPolicies
make smoke
# open http://documind.localtest.me:8080 (through your SSH tunnel if the cluster runs on a VM)
make demo-stateless          # one conversation, several pods
make demo-failover           # a paused approval survives replacing every agent pod
```

How the agent works and how it scales on Kubernetes: `documind_details.md`
(sections 10-12).

The day-by-day tutorial (what, why, commands, expected output, troubleshooting)
is the PDF that accompanies this repository.

Sample documents describe a fictional company (Nimbus Re) and are for training only.
