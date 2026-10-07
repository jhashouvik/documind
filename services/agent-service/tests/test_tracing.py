"""One chat turn produces the GenAI span tree: invoke_agent > chat / execute_tool."""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(_provider)


def test_agent_turn_spans(make_client):
    exporter.clear()
    c, *_ = make_client([("tools", [("search_knowledge_base", {"query": "TIV"})]), ("text", "ok [S1]")])
    c.post("/v1/chat", json={"message": "q"})
    spans = {s.name: s for s in exporter.get_finished_spans()}
    turn = spans["invoke_agent documind"]
    chat = [s for s in exporter.get_finished_spans() if s.name == "chat openai/gpt-4o-mini"]
    tool = spans["execute_tool search_knowledge_base"]
    assert len(chat) == 2
    assert all(s.parent.span_id == turn.context.span_id for s in chat)
    assert chat[0].attributes["gen_ai.usage.input_tokens"] == 100
    assert chat[0].attributes["documind.tool_calls"] == 1
    assert tool.attributes["documind.tool.status"] == "ok"
    assert turn.attributes["gen_ai.usage.input_tokens"] == 200 and turn.attributes["documind.steps"] == 2
