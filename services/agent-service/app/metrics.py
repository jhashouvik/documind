"""Prometheus metrics for agent-service (GET /metrics).

The LLM-specific metrics are what make an AI service operable: latency to the
first token, tokens (= cost) per model, and which tools the agent uses and how
often they fail.
"""
from prometheus_client import Counter, Gauge, Histogram, Info

APP_INFO = Info("documind_agent_build", "Build information")

HTTP_REQUESTS = Counter("documind_http_requests_total", "HTTP requests",
                        ["route", "method", "status"])
HTTP_LATENCY = Histogram("documind_http_request_duration_seconds", "HTTP latency (to first byte)",
                         ["route", "method"],
                         buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))

CHAT_REQUESTS = Counter("documind_chat_requests_total", "Chat turns", ["status"])
CHAT_DURATION = Histogram("documind_chat_duration_seconds", "Full agent turn duration",
                          buckets=(0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60, 120))
INFLIGHT = Gauge("documind_chat_inflight", "Chat turns currently being processed")
AGENT_STEPS = Histogram("documind_agent_steps", "LLM round-trips per chat turn",
                        buckets=(1, 2, 3, 4, 5, 6, 8, 10))

LLM_REQUESTS = Counter("documind_llm_requests_total", "LLM API calls", ["model", "status"])
LLM_LATENCY = Histogram("documind_llm_request_seconds", "LLM call duration", ["model"],
                        buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60))
LLM_TTFT = Histogram("documind_llm_time_to_first_token_seconds", "Time to first streamed token",
                     ["model"], buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13))
LLM_TOKENS = Counter("documind_llm_tokens_total", "Tokens consumed", ["model", "type"])

TOOL_CALLS = Counter("documind_tool_calls_total", "Tool executions", ["tool", "status"])
TOOL_LATENCY = Histogram("documind_tool_seconds", "Tool execution time", ["tool"],
                         buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
RATE_LIMITED = Counter("documind_rate_limited_total", "Requests rejected by the rate limiter")
