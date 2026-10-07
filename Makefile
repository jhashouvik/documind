# Shortcuts for the tutorial. Run `make help` to list them.
CLUSTER ?= learn
NS      ?= documind
KB_URL  ?= http://kb.localtest.me
AGENT_URL ?= http://documind.localtest.me

.PHONY: help test validate build-knowledge build-agent build-knowledge-v2 build-agent-canary \
        deploy-data deploy-knowledge deploy-agent deploy-netpol deploy-canary deploy-observability \
        tracing ingest smoke eval loadtest-search loadtest-keda traffic refresh status rollouts logs-agent logs-knowledge clean

help:            ## list targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-22s %s\n", $$1, $$2}'

test:            ## run both services' unit tests inside Docker
	docker build --target test -t documind/knowledge-service:test services/knowledge-service
	docker build --target test -t documind/agent-service:test services/agent-service

validate:        ## render every kustomization (catches YAML/kustomize mistakes before kubectl)
	@for d in k8s k8s/60-canary k8s/70-observability k8s/observability k8s/gitops; do \
	  printf '%-24s' "$$d"; kubectl kustomize $$d | grep -c '^kind:' | sed 's/$$/ objects/'; done

build-knowledge: ## build + kind-load knowledge-service 1.0.0
	scripts/build-and-load.sh knowledge-service 1.0.0

build-agent:     ## build + kind-load agent-service 1.0.0
	scripts/build-and-load.sh agent-service 1.0.0

build-knowledge-v2: ## build + kind-load knowledge-service 1.1.0 (blue-green demo)
	scripts/build-and-load.sh knowledge-service 1.1.0

build-agent-canary: ## build + kind-load agent-service 1.1.0 (canary demo)
	scripts/build-and-load.sh agent-service 1.1.0

deploy-data:     ## namespace + Qdrant
	kubectl apply -k k8s/00-namespace
	kubectl apply -k k8s/10-qdrant
	kubectl -n $(NS) rollout status statefulset/qdrant --timeout=180s

deploy-knowledge: ## Service 1
	kubectl apply -k k8s/20-knowledge-service
	kubectl -n $(NS) rollout status deployment/knowledge-service --timeout=300s

deploy-agent:    ## Redis + Service 2 (needs the documind-llm Secret)
	kubectl apply -k k8s/30-redis
	kubectl apply -k k8s/40-agent-service
	kubectl -n $(NS) rollout status deployment/agent-service --timeout=180s

deploy-netpol:   ## zero-trust NetworkPolicies
	kubectl apply -k k8s/50-network-policies

deploy-canary:   ## agent-service 1.1.0 manual canary at 20% (Day 5)
	kubectl apply -k k8s/60-canary

deploy-observability: ## ServiceMonitor + Grafana dashboard (needs kube-prometheus-stack)
	kubectl apply -k k8s/70-observability

tracing:         ## OTel Collector + Jaeger (Day 10)
	kubectl apply -k k8s/observability
	kubectl -n observability rollout status deployment/jaeger --timeout=180s
	kubectl -n observability rollout status deployment/otel-collector --timeout=180s

ingest:          ## upload sample-docs/ to knowledge-service
	KB_URL=$(KB_URL) scripts/ingest-samples.sh

smoke:           ## end-to-end smoke test (1 LLM call)
	KB_URL=$(KB_URL) AGENT_URL=$(AGENT_URL) scripts/smoke-test.sh

eval:            ## retrieval quality: hit@5 and MRR (free)
	python3 scripts/eval_retrieval.py --url $(KB_URL)

loadtest-search: ## 3 minutes of search load to trigger the knowledge-service HPA
	python3 scripts/loadtest.py --mode search --url $(KB_URL) --concurrency 8 --duration 180

loadtest-keda:   ## Day 13: 6 in-cluster pods chat for 3 minutes (costs ~0.2-0.3 USD)
	kubectl create -f k8s/keda/loadtest-job.yaml

traffic:         ## steady chat traffic for canary analysis (Ctrl+C to stop; ~1 LLM call / 3 s)
	AGENT_URL=$(AGENT_URL) scripts/traffic.sh

refresh:         ## ask Argo CD to poll Git now (instead of waiting up to 3 minutes)
	kubectl -n argocd annotate application documind argocd.argoproj.io/refresh=normal --overwrite

rollouts:        ## Argo Rollouts status of both services
	kubectl argo rollouts -n $(NS) list rollouts
	kubectl argo rollouts -n $(NS) get rollout agent-service || true

status:          ## everything in the namespace
	kubectl -n $(NS) get pods,svc,ingress,hpa,pdb,pvc -o wide

logs-agent:      ## follow agent-service logs (all pods)
	kubectl -n $(NS) logs -f -l app.kubernetes.io/name=agent-service --max-log-requests 6 --prefix

logs-knowledge:  ## follow knowledge-service logs
	kubectl -n $(NS) logs -f -l app.kubernetes.io/name=knowledge-service --prefix

clean:           ## delete DocuMind from the cluster (keeps Docker images)
	kubectl delete namespace $(NS) --ignore-not-found
