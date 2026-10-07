"""Distributed tracing with OpenTelemetry (the third pillar of observability).

Enabled only when OTEL_EXPORTER_OTLP_ENDPOINT is set (in Kubernetes it points
to the OpenTelemetry Collector, e.g. http://otel-collector.observability:4318).

What gets traced automatically
  * every incoming HTTP request (FastAPI instrumentation)
  * every outgoing HTTP call made with httpx (OpenRouter, knowledge-service),
    which also injects the W3C `traceparent` header - that is how a trace
    started in agent-service continues inside knowledge-service
What we add by hand
  * GenAI spans for embedding and chat calls with the OpenTelemetry GenAI
    semantic conventions (gen_ai.request.model, gen_ai.usage.input_tokens...)
"""
import logging
import os
from contextlib import contextmanager

from opentelemetry import trace

log = logging.getLogger(__name__)
_tracer = trace.get_tracer("documind")


def setup_tracing(app, service_name: str, version: str) -> bool:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({
        "service.name": service_name,
        "service.version": version,
        "service.namespace": "documind",
        "deployment.environment": os.getenv("DEPLOYMENT_ENVIRONMENT", "kind"),
    })
    provider = TracerProvider(resource=resource)
    # BatchSpanProcessor: spans are buffered and sent in the background, so a
    # slow or missing collector never slows down user requests.
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # reads the OTEL_* env vars
    trace.set_tracer_provider(provider)
    # exclude_spans: skip the per-chunk ASGI "http send/receive" spans (noise, esp. for SSE)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics,static",
                                       exclude_spans=["receive", "send"])
    HTTPXClientInstrumentor().instrument()
    log.info("tracing enabled", extra={"extra_fields": {"endpoint": endpoint}})
    return True


@contextmanager
def span(name: str, **attributes):
    """with span("embeddings openai/text-embedding-3-small", **{"gen_ai.request.model": m}) as s: ..."""
    with _tracer.start_as_current_span(name) as s:
        for k, v in attributes.items():
            if v is not None:
                s.set_attribute(k, v)
        yield s


def current_trace_id() -> str:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else "-"
