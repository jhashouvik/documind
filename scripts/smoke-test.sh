#!/usr/bin/env bash
# End-to-end smoke test of a deployed DocuMind. Exit code 0 = healthy.
#   scripts/smoke-test.sh            # uses 2 questions = ~10 LLM calls (a few cents at most)
set -uo pipefail
KB="${KB_URL:-http://kb.localtest.me}"
AG="${AGENT_URL:-http://documind.localtest.me}"
fail=0
check() {  # name, command
  if out=$(eval "$2" 2>&1); then printf '  PASS  %s\n' "$1"; else printf '  FAIL  %s\n        %s\n' "$1" "${out:0:300}"; fail=1; fi
}
echo "knowledge-service (${KB})"
check "healthz"          "curl -fsS ${KB}/healthz"
check "readyz"           "curl -fsS ${KB}/readyz"
check "info"             "curl -fsS ${KB}/v1/info"
check "options"          "curl -fsS ${KB}/v1/options | grep -q embedding_models"
check "search returns"   "curl -fsS -X POST ${KB}/v1/search -H 'Content-Type: application/json' -d '{\"query\":\"sprinkler\"}' | grep -q results"
check "metrics"          "curl -fsS ${KB}/metrics | grep -q documind_search_seconds"
echo "agent-service (${AG})"
check "healthz"          "curl -fsS ${AG}/healthz"
check "readyz"           "curl -fsS ${AG}/readyz"
check "not degraded"     "curl -fsS ${AG}/readyz | grep -q '\"degraded\":\[\]'"           # Redis + Postgres reachable
check "langgraph+postgres" "curl -fsS ${AG}/v1/info | grep -q '\"persistence\":\"postgres\"'"
check "agent graph"      "curl -fsS ${AG}/v1/graph | grep -q supervisor"
check "event schema"     "curl -fsS ${AG}/v1/events/schema | grep -q discriminator"
check "UI served"        "curl -fsS ${AG}/ | grep -q DocuMind"
check "UI options"       "curl -fsS ${AG}/v1/options | grep -q llm_models"
check "BFF lists docs"   "curl -fsS ${AG}/api/knowledge/v1/documents"
check "chat answers"     "curl -fsS --max-time 120 -X POST ${AG}/v1/chat -H 'Content-Type: application/json' -d '{\"message\":\"What TIV must be referred to the chief underwriter?\"}' | grep -q '\"status\":\"done\"'"
check "streaming"        "curl -fsSN -X POST ${AG}/v1/chat/stream -H 'Content-Type: application/json' -d '{\"message\":\"hello\"}' --max-time 120 | grep -q 'event: done'"
exit $fail
