"""Structured JSON logging with a per-request correlation id.

Each HTTP request gets an id (taken from the incoming X-Request-ID header, or
generated). It is stored in a ContextVar so every log line written while
handling that request carries it, and it is echoed back in the response.
agent-service forwards the same header, so one user question can be followed
across both services with:  kubectl logs ... | grep <request_id>
"""
import json
import logging
import sys
import time
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def _trace_id() -> str:
    """OpenTelemetry trace id of the current request ('-' when tracing is off).
    Grafana uses it to jump from a log line in Loki to the trace in Jaeger."""
    try:
        from opentelemetry import trace
        ctx = trace.get_current_span().get_span_context()
        return format(ctx.trace_id, "032x") if ctx.is_valid else "-"
    except Exception:  # noqa: BLE001
        return "-"


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "trace_id": _trace_id(),
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(service: str, level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # uvicorn's own loggers go through the same JSON handler
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.propagate = True
    logging.getLogger("uvicorn.access").disabled = True  # we log requests ourselves
    for noisy in ("httpx", "httpx2", "opentelemetry"):
        logging.getLogger(noisy).setLevel("WARNING")


def log_extra(**fields) -> dict:
    """Usage: logger.info("indexed", extra=log_extra(doc_id=..., chunks=...))"""
    return {"extra_fields": fields}
