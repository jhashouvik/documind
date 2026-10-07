"""The custom GenAI spans are created with the right attributes."""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(_provider)


def test_ingest_and_search_emit_spans(client):
    exporter.clear()
    client.post("/v1/documents", files={"file": ("a.md", b"Sprinklers are required in large warehouses.")})
    client.post("/v1/search", json={"query": "sprinklers"})
    names = [s.name for s in exporter.get_finished_spans()]
    assert "parse document" in names and "chunk document" in names
    assert "qdrant upsert" in names and "qdrant search" in names
    search = next(s for s in exporter.get_finished_spans() if s.name == "qdrant search")
    assert search.attributes["db.system"] == "qdrant" and search.attributes["documind.results"] >= 1
