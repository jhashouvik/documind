#!/usr/bin/env bash
# Create the two Secrets DocuMind needs, without writing them to any file.
#   scripts/create-secrets.sh                      # asks for the OpenRouter key
#   OPENROUTER_API_KEY=sk-or-... scripts/create-secrets.sh
#
#   documind-llm       OPENROUTER_API_KEY  -> knowledge-service (embeddings) + agent-service (chat)
#   documind-postgres  POSTGRES_PASSWORD   -> postgres StatefulSet + agent-service (DATABASE_URL)
#
# Idempotent: an existing Secret is left alone (re-run with FORCE=1 to replace).
# The Postgres password is random and only ever stored in the cluster; changing it
# later means changing it inside Postgres too (ALTER USER), so keep the Secret.
set -euo pipefail
NS="${NS:-documind}"
kubectl get namespace "$NS" >/dev/null 2>&1 || kubectl create namespace "$NS"

exists() { kubectl -n "$NS" get secret "$1" >/dev/null 2>&1; }

if exists documind-llm && [ "${FORCE:-0}" != 1 ]; then
  echo "documind-llm       exists (FORCE=1 to replace)"
else
  if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    read -rsp "OpenRouter API key: " OPENROUTER_API_KEY; echo
  fi
  kubectl -n "$NS" create secret generic documind-llm \
    --from-literal=OPENROUTER_API_KEY="$OPENROUTER_API_KEY" --dry-run=client -o yaml | kubectl apply -f -
  echo "documind-llm       created"
fi

if exists documind-postgres; then
  echo "documind-postgres  exists (kept: the database was initialised with this password)"
else
  password="$(openssl rand -hex 16 2>/dev/null || head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  kubectl -n "$NS" create secret generic documind-postgres --from-literal=POSTGRES_PASSWORD="$password"
  echo "documind-postgres  created (random password)"
fi
