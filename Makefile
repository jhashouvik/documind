# Shortcuts for the tutorial. Run `make help` to list them.
CLUSTER ?= learn
NS      ?= documind
KB_URL  ?= http://kb.localtest.me
AGENT_URL ?= http://documind.localtest.me
AGENT_VERSION ?= 2.0.0
PYTHON  ?= $(shell python3 -c "" >/dev/null 2>&1 && echo python3 || echo python)

.PHONY: help test validate kind-up secrets build-knowledge build-agent build-knowledge-v2 build-agent-canary \
        deploy-data deploy-knowledge deploy-postgres deploy-agent deploy-netpol deploy-all deploy-canary \
        deploy-observability tracing ingest smoke eval demo-stateless demo-failover graph psql \
        watch-scale loadtest-search loadtest-agent loadtest-keda traffic refresh status rollouts \
        logs-agent logs-knowledge logs-postgres clean

help:            ## list targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-22s %s\n", $$1, $$2}'

test:            ## run both services' unit tests inside Docker
	docker build --target test -t documind/knowledge-service:test services/knowledge-service
	docker build --target test -t documind/agent-service:test services/agent-service

validate:        ## render every kustomization (catches YAML/kustomize mistakes before kubectl)
	@for d in k8s k8s/60-canary k8s/70-observability k8s/observability k8s/gitops; do \
	  printf '%-24s' "$$d"; kubectl kustomize $$d | grep -c '^kind:' | sed 's/$$/ objects/'; done

# ---------------------------------------------------------------- cluster + images
kind-up:         ## create the kind cluster "learn" (3 nodes, ingress-nginx, metrics-server)
	KIND_CLUSTER=$(CLUSTER) bash scripts/kind-up.sh

secrets:         ## create the documind-llm + documind-postgres Secrets (asks for the key)
	NS=$(NS) bash scripts/create-secrets.sh

build-knowledge: ## build + kind-load knowledge-service 1.0.0
	bash scripts/build-and-load.sh knowledge-service 1.0.0

build-agent:     ## build + kind-load agent-service 2.0.0 (LangGraph multi-agent)
	bash scripts/build-and-load.sh agent-service $(AGENT_VERSION)

build-knowledge-v2: ## build + kind-load knowledge-service 1.1.0 (blue-green demo)
	bash scripts/build-and-load.sh knowledge-service 1.1.0

build-agent-canary: ## build + kind-load agent-service 2.1.0 (canary demo)
	bash scripts/build-and-load.sh agent-service 2.1.0

# ---------------------------------------------------------------- deploy (plain manifests)
deploy-data:     ## namespace + Qdrant
	kubectl apply -k k8s/00-namespace
	kubectl apply -k k8s/10-qdrant
	kubectl -n $(NS) rollout status statefulset/qdrant --timeout=180s

deploy-knowledge: ## Service 1
	kubectl apply -k k8s/20-knowledge-service
	kubectl -n $(NS) rollout status deployment/knowledge-service --timeout=300s

deploy-postgres: ## Postgres: the agent's memory (needs the documind-postgres Secret)
	kubectl apply -k k8s/35-postgres
	kubectl -n $(NS) rollout status statefulset/postgres --timeout=180s

deploy-agent:    ## Redis + Postgres + Service 2 (needs both Secrets: make secrets)
	kubectl apply -k k8s/30-redis
	$(MAKE) deploy-postgres
	kubectl apply -k k8s/40-agent-service
	kubectl -n $(NS) rollout status deployment/agent-service --timeout=240s

deploy-netpol:   ## zero-trust NetworkPolicies
	kubectl apply -k k8s/50-network-policies

deploy-all:      ## everything above in order, then load the sample documents
	$(MAKE) deploy-data deploy-knowledge ingest deploy-agent deploy-netpol
	$(MAKE) status

deploy-canary:   ## agent-service 2.1.0 manual canary at 20% (Day 5)
	kubectl apply -k k8s/60-canary

deploy-observability: ## ServiceMonitor + alerts + Grafana dashboard (needs kube-prometheus-stack)
	kubectl apply -k k8s/70-observability

tracing:         ## OTel Collector + Jaeger (Day 10)
	kubectl apply -k k8s/observability
	kubectl -n observability rollout status deployment/jaeger --timeout=180s
	kubectl -n observability rollout status deployment/otel-collector --timeout=180s

# ---------------------------------------------------------------- use + learn
ingest:          ## upload sample-docs/ to knowledge-service
	KB_URL=$(KB_URL) bash scripts/ingest-samples.sh

smoke:           ## end-to-end smoke test (2 questions)
	KB_URL=$(KB_URL) AGENT_URL=$(AGENT_URL) bash scripts/smoke-test.sh

eval:            ## retrieval quality: hit@5 and MRR (free)
	$(PYTHON) scripts/eval_retrieval.py --url $(KB_URL)

demo-stateless:  ## one conversation answered by different pods (memory lives in Postgres)
	AGENT_URL=$(AGENT_URL) bash scripts/stateless-demo.sh

demo-failover:   ## pause for approval, replace ALL agent pods, resume (rejects the delete)
	AGENT_URL=$(AGENT_URL) NS=$(NS) bash scripts/hitl-failover-demo.sh

graph:           ## print the agent graph (Mermaid) - paste into https://mermaid.live
	@curl -fsS $(AGENT_URL)/v1/graph | $(PYTHON) -c 'import json,sys; print(json.load(sys.stdin)["mermaid"])'

psql:            ## open psql in Postgres (try: \dt   select thread_id, count(*) from checkpoints group by 1;)
	kubectl -n $(NS) exec -it postgres-0 -- psql -U documind -d documind

watch-scale:     ## watch the autoscaler and the agent pods (Ctrl+C to stop)
	kubectl -n $(NS) get hpa,pods -l 'app.kubernetes.io/name in (agent-service)' -w

loadtest-search: ## 3 minutes of search load to trigger the knowledge-service HPA
	$(PYTHON) scripts/loadtest.py --mode search --url $(KB_URL) --concurrency 8 --duration 180

loadtest-agent:  ## 6 in-cluster pods chat for 3 minutes (see the CPU HPA barely move; COSTS tokens)
	kubectl create -f k8s/keda/loadtest-job.yaml

loadtest-keda: loadtest-agent ## same load test, for the KEDA chapter (Day 13)

traffic:         ## steady chat traffic for canary analysis (Ctrl+C to stop; 1 question / 3 s)
	AGENT_URL=$(AGENT_URL) bash scripts/traffic.sh

refresh:         ## ask Argo CD to poll Git now (instead of waiting up to 3 minutes)
	kubectl -n argocd annotate application documind argocd.argoproj.io/refresh=normal --overwrite

rollouts:        ## Argo Rollouts status of both services
	kubectl argo rollouts -n $(NS) list rollouts
	kubectl argo rollouts -n $(NS) get rollout agent-service || true

status:          ## everything in the namespace
	kubectl -n $(NS) get pods,svc,ingress,hpa,pdb,pvc,statefulset -o wide

logs-agent:      ## follow agent-service logs (all pods)
	kubectl -n $(NS) logs -f -l app.kubernetes.io/name=agent-service --max-log-requests 6 --prefix

logs-knowledge:  ## follow knowledge-service logs
	kubectl -n $(NS) logs -f -l app.kubernetes.io/name=knowledge-service --prefix

logs-postgres:   ## follow Postgres logs
	kubectl -n $(NS) logs -f statefulset/postgres

clean:           ## delete DocuMind from the cluster (keeps Docker images; DELETES the PVCs = documents + memory)
	kubectl delete namespace $(NS) --ignore-not-found
