"""One chat turn produces one trace: invoke_agent > agent_node <name> spans."""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.graph.state import Grades, Grounding, RouteDecision

exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(_provider)


def test_agent_turn_spans(make_client):
    exporter.clear()
    c, *_ = make_client({"supervisor": [RouteDecision(next="researcher", instruction="TIV"),
                                        RouteDecision(next="writer")],
                         "grader": [Grades(relevant=[1])], "writer": ["ok [S1]"],
                         "grounding": [Grounding(grounded=True)]})
    c.post("/v1/chat", json={"message": "q"})
    spans = exporter.get_finished_spans()
    turn = next(s for s in spans if s.name == "invoke_agent documind")
    nodes = [s for s in spans if s.name.startswith("agent_node ")]
    names = [s.name.removeprefix("agent_node ") for s in nodes]
    for expected in ("start_turn", "supervisor", "researcher", "writer", "grounding_check", "finalize"):
        assert expected in names
    assert names.count("supervisor") == 2
    assert all(s.context.trace_id == turn.context.trace_id for s in nodes)    # one trace per turn
    assert turn.attributes["gen_ai.usage.input_tokens"] == 100
    assert turn.attributes["documind.steps"] == 2
