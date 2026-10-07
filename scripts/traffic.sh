#!/usr/bin/env bash
# Steady chat traffic so a canary analysis has something to measure.
#   scripts/traffic.sh                 # until Ctrl+C
#   COUNT=40 scripts/traffic.sh        # 40 requests
# Prints: HTTP status, the version that answered (X-App-Version header), seconds.
# Cost: every request is a real LLM call (gpt-4o-mini: roughly $0.0005 each).
# The sleep keeps us under the 20 requests/minute rate limit (else: 429s).
set -uo pipefail
AG="${AGENT_URL:-http://documind.localtest.me}"
COUNT="${COUNT:-0}"
SLEEP="${SLEEP:-3.2}"
QUESTIONS=(
  "What total insured value must be referred to the chief underwriter?"
  "What is the maximum line Nimbus writes on one risk?"
  "Which warehouses need a full sprinkler system?"
  "How fast must a new claim be acknowledged?"
  "Who approves claim payments above the adjuster's authority?"
)
i=0
while :; do
  q="${QUESTIONS[$((i % ${#QUESTIONS[@]}))]}"
  out=$(curl -s -o /dev/null --max-time 60 -X POST "${AG}/v1/chat" \
          -H 'Content-Type: application/json' -d "{\"message\": \"${q}\"}" \
          -w '%{http_code} %header{x-app-version} %{time_total}')
  printf '%3d  %s\n' "$i" "$out"
  i=$((i + 1))
  [ "$COUNT" -gt 0 ] && [ "$i" -ge "$COUNT" ] && break
  sleep "$SLEEP"
done
