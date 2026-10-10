#!/usr/bin/env bash
# Upload every file in sample-docs/ to knowledge-service.
#   scripts/ingest-samples.sh                         # via the Ingress (default)
#   KB_URL=http://localhost:8001 scripts/ingest-samples.sh   # via port-forward / compose
# python3 on Linux/macOS, python on Windows (where "python3" may be a Store stub that does not run)
PY="$(for p in python3 python; do "$p" -c "" >/dev/null 2>&1 && { echo "$p"; break; }; done)"
set -euo pipefail
KB_URL="${KB_URL:-http://kb.localtest.me}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for f in "${ROOT}"/sample-docs/*.md; do
  [ -e "$f" ] || continue
  printf '%-48s ' "$(basename "$f")"
  curl -sS -X POST "${KB_URL}/v1/documents" -F "file=@${f}" \
    | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status"), d.get("chunks"), "chunks", d.get("took_ms"), "ms", d.get("embedding_model")) if "status" in d else print(d)'
done
