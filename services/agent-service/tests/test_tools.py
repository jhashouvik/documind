import pytest

from app.tools import CitationRegistry, safe_eval, tool_schemas


@pytest.mark.parametrize("expr,expected", [
    ("2 + 3 * 4", 14), ("(250000000 * 0.15) / 12", 3125000.0), ("2**10", 1024),
    ("-5 + 2", -3), ("40,000,000 * 0.02", 800000.0), ("7 // 2", 3), ("7 % 4", 3),
])
def test_safe_eval(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('id')", "open('/etc/passwd')", "a + 1", "[1,2,3]",
    "2 ** 100000", "(lambda: 1)()",
])
def test_safe_eval_rejects_code(expr):
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(expr)


def test_citation_ids_are_stable():
    reg = CitationRegistry()
    hit = {"doc_id": "a", "chunk_index": 1, "filename": "f", "page": 1, "score": 0.9, "text": "t"}
    other = {**hit, "chunk_index": 2}
    assert reg.register(hit) == "S1"
    assert reg.register(other) == "S2"
    assert reg.register(hit) == "S1"          # same chunk again -> same id


def test_tool_schemas_are_openai_shaped():
    schemas = {s["function"]["name"]: s for s in tool_schemas()}
    assert set(schemas) == {"search_knowledge_base", "list_documents", "calculator"}
    search = schemas["search_knowledge_base"]["function"]["parameters"]
    assert search["type"] == "object" and "query" in search["properties"]
    assert search["required"] == ["query"]
    assert "title" not in search
    assert schemas["list_documents"]["function"]["parameters"]["properties"] == {}
