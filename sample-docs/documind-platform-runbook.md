# DocuMind Platform - On-call Runbook

Owner: Platform Engineering. Applies to the DocuMind services running in the
`documind` Kubernetes namespace (knowledge-service, agent-service, Qdrant, Redis).

## Service level objectives

DocuMind targets 99.5 percent monthly availability for the chat endpoint,
measured as the share of successful (non-5xx) requests. This leaves an error
budget of about 3 hours 36 minutes per 30-day month. The p95 time to first
token should stay below 3 seconds.

## Incident priorities and response times

- P1 - DocuMind is down or answers are wrong for everyone: acknowledge within
  15 minutes, update stakeholders every 30 minutes.
- P2 - degraded (slow answers, uploads failing, one service unhealthy):
  acknowledge within 1 hour.
- P3 - minor issue with a workaround: next business day.

A written postmortem is required for every P1 within 5 business days. Postmortems
are blameless and must list at least one preventive action with an owner.

## On-call rotation

On-call rotates weekly. Handover happens every Monday at 10:00 IST with a short
review of open incidents and recent deployments.

## Common procedures

### Roll back a bad release

Run `kubectl rollout undo deployment/agent-service -n documind` and confirm with
`kubectl rollout status`. For a canary, set the canary weight to 0 first.

### OpenRouter errors

HTTP 401 means the API key in the documind-llm secret is wrong. HTTP 402 means
the OpenRouter account is out of credits. HTTP 429 means the rate limit was hit;
configure LLM_FALLBACK_MODELS to fail over to another model.

### Knowledge base empty after a restart

Check that the Qdrant pod still has its PersistentVolumeClaim
(`storage-qdrant-0`). Never delete that PVC unless you intend to wipe all
indexed documents.

## Security

The OpenRouter API key is rotated every 90 days. Keys are never committed to
Git; they are created as Kubernetes Secrets with kubectl. Access to the
production namespace requires the platform-admin group.
