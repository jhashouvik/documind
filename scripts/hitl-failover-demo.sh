#!/usr/bin/env bash
# Human-in-the-loop survives losing EVERY agent pod.
#   scripts/hitl-failover-demo.sh            # the deletion is REJECTED at the end (safe)
#   DECISION=approve scripts/hitl-failover-demo.sh    # really deletes the document!
#
#  1. ask the agents to delete a document -> the librarian pauses (interrupt) and
#     the HTTP response is {"status": "interrupted"}; the paused graph state is
#     now a checkpoint in Postgres
#  2. delete ALL agent-service pods (kubectl rollout restart) and wait for new ones
#  3. resume the SAME session on a brand-new pod: the graph continues from its
#     checkpoint exactly where it stopped
# python3 on Linux/macOS, python on Windows (where "python3" may be a Store stub that does not run)
PY="$(for p in python3 python; do "$p" -c "" >/dev/null 2>&1 && { echo "$p"; break; }; done)"
set -euo pipefail
AG="${AGENT_URL:-http://documind.localtest.me}"
NS="${NS:-documind}"
DECISION="${DECISION:-reject}"
SESSION="hitl-$(date +%s)"

echo "== 1. ask for a deletion (session $SESSION)"
curl -sS --max-time 120 -X POST "$AG/v1/chat" -H 'Content-Type: application/json' \
  -d "{\"message\": \"Please delete the claims handling SOP document from the knowledge base\", \"session_id\": \"$SESSION\"}" \
  | "$PY" -c '
import json, sys
b = json.load(sys.stdin)
print("   status:", b.get("status"))
for a in (b.get("interrupt") or {}).get("actions", []):
    print("   waiting for approval:", a["name"], a["args"])
if b.get("status") != "interrupted":
    print("   (no pause: the agents did not try to delete anything - is the document indexed?)"); sys.exit(1)'

echo "== 2. replace every agent-service pod"
kubectl -n "$NS" get pods -l app.kubernetes.io/name=agent-service -o name | sed 's/^/   before: /'
kubectl -n "$NS" rollout restart deployment/agent-service
kubectl -n "$NS" rollout status deployment/agent-service --timeout=180s
kubectl -n "$NS" get pods -l app.kubernetes.io/name=agent-service --field-selector=status.phase=Running -o name | sed 's/^/   after:  /'

echo "== 3. resume on a new pod with decision=$DECISION"
curl -sS --max-time 120 -D /tmp/documind-headers.$$ -X POST "$AG/v1/chat/resume" -H 'Content-Type: application/json' \
  -d "{\"session_id\": \"$SESSION\", \"decision\": \"$DECISION\"}" \
  | "$PY" -c 'import json,sys; b=json.load(sys.stdin); print("   status:", b.get("status")); print("   answer:", (b.get("answer") or b)[:300])'
echo "   answered by pod: $(grep -i '^x-served-by:' /tmp/documind-headers.$$ | tr -d '\r' | cut -d' ' -f2)"
rm -f /tmp/documind-headers.$$
