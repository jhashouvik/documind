"""Prometheus metrics exposed at GET /metrics.

Naming follows Prometheus conventions: <namespace>_<what>_<unit>, counters end
in _total. Labels are kept LOW-cardinality: we label HTTP metrics with the
route *template* (/v1/documents/{doc_id}) and never with raw paths, doc ids or
user input, otherwise every new document would create a new time series.
"""
from prometheus_client import Counter, Gauge, Histogram, Info

APP_INFO = Info("documind_knowledge_build", "Build information")

HTTP_REQUESTS = Counter(
    "documind_http_requests_total", "HTTP requests", ["route", "method", "status"]
)
HTTP_LATENCY = Histogram(
    "documind_http_request_duration_seconds",
    "HTTP request latency",
    ["route", "method"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)

INGEST_DOCUMENTS = Counter(
    "documind_ingest_documents_total", "Documents ingested", ["status"]
)
INGEST_CHUNKS = Counter("documind_ingest_chunks_total", "Chunks written to the vector store")
CHUNKS_PER_DOC = Histogram(
    "documind_chunks_per_document", "Chunks produced per document",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
)
EMBED_LATENCY = Histogram(
    "documind_embed_seconds", "Time spent getting embeddings from the API", ["kind", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
EMBED_TOKENS = Counter("documind_embed_tokens_total", "Tokens sent to the embedding API", ["model"])
EMBED_ERRORS = Counter("documind_embed_errors_total", "Failed embedding requests", ["status"])
SEARCH_LATENCY = Histogram(
    "documind_search_seconds", "End-to-end semantic search latency",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
SEARCH_RESULTS = Histogram(
    "documind_search_results", "Results returned per search", buckets=(0, 1, 2, 3, 5, 8, 13, 20)
)
SEARCH_TOP_SCORE = Histogram(
    "documind_search_top_score", "Similarity score of the best hit",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
READY = Gauge("documind_knowledge_ready", "1 when the service is ready to serve traffic")
