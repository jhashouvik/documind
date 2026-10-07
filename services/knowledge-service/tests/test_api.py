import io

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

GUIDE = """# Property underwriting guidelines

## Referral limits
Any risk with a total insured value above USD 250 million must be referred to the
chief underwriter before a quote is issued.

## Sprinklers
Warehouses larger than 5,000 square metres require a full sprinkler system.
"""

CLAIMS = """# Claims handling
First notice of loss must be acknowledged within 24 hours.
Claims above USD 1 million are assigned to a senior adjuster.
"""


def _pdf_bytes(lines_per_page: list[list[str]]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for lines in lines_per_page:
        y = 800
        for line in lines:
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200


def test_request_id_is_echoed(client):
    r = client.get("/v1/documents", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


def test_upload_markdown_then_search(client):
    r = client.post("/v1/documents",
                    files={"file": ("underwriting-guidelines.md", GUIDE.encode(), "text/markdown")})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "indexed" and body["chunks"] >= 1

    client.post("/v1/documents", files={"file": ("claims.md", CLAIMS.encode(), "text/markdown")})

    r = client.post("/v1/search", json={"query": "sprinkler system warehouses", "top_k": 3})
    assert r.status_code == 200
    hits = r.json()["results"]
    assert hits[0]["filename"] == "underwriting-guidelines.md"
    assert "sprinkler" in hits[0]["text"].lower()
    assert hits[0]["score"] >= hits[-1]["score"]


def test_duplicate_upload_is_idempotent(client):
    f = {"file": ("claims.md", CLAIMS.encode(), "text/markdown")}
    first = client.post("/v1/documents", files=f).json()
    r = client.post("/v1/documents", files=f)
    assert r.status_code == 200
    assert r.json()["status"] == "duplicate"
    assert r.json()["doc_id"] == first["doc_id"]
    assert len(client.get("/v1/documents").json()) == 1


def test_replace_reindexes(client):
    f = {"file": ("claims.md", CLAIMS.encode(), "text/markdown")}
    client.post("/v1/documents", files=f)
    r = client.post("/v1/documents?replace=true", files=f)
    assert r.status_code == 201 and r.json()["status"] == "reindexed"
    assert client.get("/v1/info").json()["collections"][0]["points"] == r.json()["chunks"]


def test_pdf_pages_are_tracked(client):
    pdf = _pdf_bytes([["Page one talks about flood zones."],
                      ["Page two talks about earthquake deductibles."]])
    r = client.post("/v1/documents", files={"file": ("cat-guide.pdf", pdf, "application/pdf")})
    assert r.status_code == 201, r.text
    assert r.json()["pages"] == 2
    hits = client.post("/v1/search", json={"query": "earthquake deductibles"}).json()["results"]
    assert hits[0]["page"] == 2


def test_filter_by_document(client):
    a = client.post("/v1/documents", files={"file": ("a.md", GUIDE.encode())}).json()["doc_id"]
    client.post("/v1/documents", files={"file": ("b.md", CLAIMS.encode())})
    hits = client.post("/v1/search", json={"query": "claims adjuster", "doc_ids": [a]}).json()
    assert {h["doc_id"] for h in hits["results"]} <= {a}


def test_delete_document(client):
    doc_id = client.post("/v1/documents", files={"file": ("claims.md", CLAIMS.encode())}).json()["doc_id"]
    assert client.get(f"/v1/documents/{doc_id}").status_code == 200
    assert client.delete(f"/v1/documents/{doc_id}").status_code == 204
    assert client.get(f"/v1/documents/{doc_id}").status_code == 404
    assert client.delete(f"/v1/documents/{doc_id}").status_code == 404
    assert client.get("/v1/info").json()["collections"][0]["points"] == 0


def test_text_endpoint(client):
    r = client.post("/v1/documents/text", json={"title": "notes", "text": "Renewal date is 1 April."})
    assert r.status_code == 201
    assert r.json()["filename"] == "notes.md"


def test_rejects_unsupported_type(client):
    r = client.post("/v1/documents", files={"file": ("image.png", b"\x89PNG....")})
    assert r.status_code == 415


def test_rejects_empty_document(client):
    r = client.post("/v1/documents", files={"file": ("empty.txt", b"   \n  ")})
    assert r.status_code == 422


def test_rejects_too_large(client):
    r = client.post("/v1/documents", files={"file": ("big.txt", b"a" * (1024 * 1024 + 10))})
    assert r.status_code == 413


def test_search_validation(client):
    assert client.post("/v1/search", json={"query": ""}).status_code == 422
    assert client.post("/v1/search", json={"query": "x", "top_k": 500}).status_code == 422


def test_metrics_exposed(client):
    client.post("/v1/search", json={"query": "anything"})
    text = client.get("/metrics").text
    assert "documind_search_seconds" in text
    assert 'route="/v1/search"' in text


def test_options(client):
    o = client.get("/v1/options").json()
    assert o["embedding_models"] == ["test/model-a", "test/model-b"]
    assert o["default_embedding_model"] == "test/model-a"
    assert o["chunk_size"]["default"] == 300 and o["chunk_size"]["min"] == 200


def test_custom_chunking_changes_chunk_count_and_reindexes(client):
    text = ("Sentence about referral limits and sprinklers. " * 60).encode()
    f = {"file": ("long.md", text, "text/markdown")}
    a = client.post("/v1/documents?chunk_size=1000&chunk_overlap=0", files=f).json()
    b = client.post("/v1/documents?chunk_size=250&chunk_overlap=50", files=f)
    assert b.status_code == 201 and b.json()["status"] == "reindexed"
    assert b.json()["chunks"] > a["chunks"]
    doc = client.get("/v1/documents").json()[0]
    assert doc["chunk_size"] == 250 and doc["chunk_overlap"] == 50
    # same settings again -> duplicate
    c = client.post("/v1/documents?chunk_size=250&chunk_overlap=50", files=f)
    assert c.status_code == 200 and c.json()["status"] == "duplicate"


def test_chunking_limits_enforced(client):
    f = {"file": ("x.md", b"hello world", "text/markdown")}
    assert client.post("/v1/documents?chunk_size=50", files=f).status_code == 422
    assert client.post("/v1/documents?chunk_size=400&chunk_overlap=300", files=f).status_code == 422


def test_each_embedding_model_is_a_separate_knowledge_base(client):
    client.post("/v1/documents?embedding_model=test/model-b", files={"file": ("b.md", CLAIMS.encode())})
    assert client.get("/v1/documents").json() == []                       # default model a: empty
    assert len(client.get("/v1/documents?embedding_model=test/model-b").json()) == 1
    hits = client.post("/v1/search", json={"query": "claims adjuster",
                                           "embedding_model": "test/model-b"}).json()
    assert hits["embedding_model"] == "test/model-b" and hits["results"]
    names = [c["collection"] for c in client.get("/v1/info").json()["collections"]]
    assert "test__test_model_b" in names


def test_unknown_embedding_model_rejected(client):
    r = client.post("/v1/search", json={"query": "x", "embedding_model": "evil/expensive-model"})
    assert r.status_code == 422
