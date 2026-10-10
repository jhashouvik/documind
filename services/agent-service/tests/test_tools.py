import pytest

from app.graph.common import numbered_sources
from app.graph.state import RESET, add_findings, merge_sources
from app.graph.tools import calculator, delete_document, list_documents, safe_eval


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


def test_tool_schemas_hide_the_injected_runtime():
    """The LLM sees only the real arguments; `runtime` is injected by LangGraph."""
    def props(t):
        return set(t.tool_call_schema.model_json_schema().get("properties", {}))
    assert props(calculator) == {"expression"}
    assert props(delete_document) == {"doc_id", "filename"}
    assert props(list_documents) == set()
    assert "Arithmetic" in calculator.description or "arithmetic" in calculator.description


def _p(key, score):
    doc, idx = key.split(":")
    return {"key": key, "doc_id": doc, "filename": f"{doc}.md", "page": 1, "chunk_index": int(idx),
            "score": score, "text": "t" * 10, "query": "q"}


def test_merge_sources_reducer_unions_parallel_branches():
    a = merge_sources({}, {"d1:0": _p("d1:0", 0.5)})
    b = merge_sources(a, {"d1:0": _p("d1:0", 0.9), "d2:1": _p("d2:1", 0.4)})
    assert set(b) == {"d1:0", "d2:1"} and b["d1:0"]["score"] == 0.9     # higher score wins
    assert merge_sources(b, RESET) == {}


def test_add_findings_reducer():
    assert add_findings([{"agent": "a"}], [{"agent": "b"}]) == [{"agent": "a"}, {"agent": "b"}]
    assert add_findings([{"agent": "a"}], RESET) == []


def test_numbered_sources_are_ranked_and_bounded():
    srcs = {k: _p(k, s) for k, s in [("d1:0", 0.3), ("d2:0", 0.9), ("d3:0", 0.6)]}
    out = numbered_sources(srcs, max_n=2, max_chars=1000)
    assert [(s["source_id"], s["doc_id"]) for s in out] == [("S1", "d2"), ("S2", "d3")]
    assert len(numbered_sources(srcs, max_n=10, max_chars=15)) == 2      # text budget


def test_numbered_sources_strip_text_repeated_by_a_neighbour_chunk():
    base = {"doc_id": "d", "filename": "d.md", "page": 1, "query": "q"}
    first = {**base, "key": "d:14", "chunk_index": 14, "score": 0.9,
             "text": "Section 3.3.2 says the insurance ceases. Section 3.3.3 says payment is a discharge."}
    second = {**base, "key": "d:15", "chunk_index": 15, "score": 0.8,
              "text": "Section 3.3.3 says payment is a discharge. 3.4 No maturity or survival benefits are "
                      "payable under the Policy. 4.1 The Death Benefit is payable on proof of death."}
    out = numbered_sources({"a": first, "b": second}, max_n=10, max_chars=10_000)
    assert out[1]["text"].startswith("3.4 No maturity")          # the shared sentence is gone
    dup = {**second, "text": "Section 3.3.3 says payment is a discharge. Short tail."}
    assert len(numbered_sources({"a": first, "b": dup}, max_n=10, max_chars=10_000)) == 1


def test_each_bullet_needs_its_own_citation():
    from app.graph.nodes import uncited_paragraphs
    answer = ("The surrender benefit is described as follows:\n\n"
              "- A Member may request surrender at any time during the Period of Coverage.\n"
              "- The Surrender Value is calculated using the formula:\n"
              "  Surrender Value = 70% of Premium paid x (unexpired / total months).\n"
              "- Payment of the Surrender Value is a full discharge of liability [S1].\n\n"
              "The documents do not mention any surrender charges.")
    out = uncited_paragraphs(answer)
    assert len(out) == 2                                   # bullets 1 and 2 (with its formula line)
    assert out[1].startswith("- The Surrender Value") and "70% of Premium" in out[1]
    assert uncited_paragraphs("A cited paragraph that is long enough to be checked [S2].") == []


def test_indented_blocks_belong_to_their_bullet():
    """Real answer from the stack: the formula is an indented block under bullet 1."""
    from app.graph.nodes import uncited_paragraphs
    answer = ("The surrender benefit under the policy is described as follows:\n\n"
              "- A Member may request the surrender of their Insurance at any time during the Period "
              "of Coverage. The insurer shall pay the Surrender Value based on the formula:\n\n"
              "  Surrender Value = 70% of Premium paid x (Unexpired Period of Coverage in months / Total "
              "Period of Coverage in months)\n\n"
              "  ^ Ignoring fraction of a month [S1]\n\n"
              "- Upon receipt of a valid surrender request the insurance shall cease [S1].")
    assert uncited_paragraphs(answer) == []
