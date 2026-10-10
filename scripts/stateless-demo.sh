#!/usr/bin/env bash
# Prove that agent-service pods are STATELESS: one conversation, several pods.
#   scripts/stateless-demo.sh                      # via the Ingress
#   AGENT_URL=http://localhost:8002 scripts/stateless-demo.sh
#
# Each turn prints the pod that answered (X-Served-By header). With 2+ replicas
# the turns land on different pods, yet the follow-up questions still know the
# conversation: the history is in the LangGraph checkpointer (Postgres), keyed
# by session_id = thread_id - not in any pod's memory.
# Cost: 3 questions (a few cents at most).
# python3 on Linux/macOS, python on Windows (where "python3" may be a Store stub that does not run)
PY="$(for p in python3 python; do "$p" -c "" >/dev/null 2>&1 && { echo "$p"; break; }; done)"
set -euo pipefail
AG="${AGENT_URL:-http://documind.localtest.me}"
SESSION="demo-$(date +%s)"
QUESTIONS=(
  "What total insured value must be referred to the chief underwriter?"
  "And what is 15% of that amount?"
  "Summarise our conversation in one sentence."
)
echo "session: $SESSION   (agent pods: $(kubectl -n documind get pods -l app.kubernetes.io/name=agent-service --no-headers 2>/dev/null | wc -l | tr -d ' '))"
for q in "${QUESTIONS[@]}"; do
  # two requests per turn would show load balancing even better, but each costs LLM calls
  resp=$(curl -sS -D /tmp/documind-headers.$$ --max-time 120 -X POST "$AG/v1/chat" \
           -H 'Content-Type: application/json' \
           -d "{\"message\": \"$q\", \"session_id\": \"$SESSION\"}")
  pod=$(grep -i '^x-served-by:' /tmp/documind-headers.$$ | tr -d '\r' | cut -d' ' -f2)
  printf '\n[%s] Q: %s\n' "${pod:-?}" "$q"
  printf '%s' "$resp" | "$PY" -c 'import json,sys; b=json.load(sys.stdin); print("   A:", (b.get("answer") or b)[:300].replace("\n"," "))'
done
rm -f /tmp/documind-headers.$$
echo
echo "history stored for $SESSION (from Postgres, served by whichever pod answers now):"
curl -sS "$AG/v1/sessions/$SESSION" | "$PY" -c 'import json,sys; [print("  ", m["role"], ":", m["content"][:80].replace("\n"," ")) for m in json.load(sys.stdin)["messages"]]'
