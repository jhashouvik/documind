"""The release-gate script works against a live app (served by TestClient)."""

import app.selftest as st


def test_selftest_passes_on_healthy_service(client, monkeypatch, capsys):
    client.post("/v1/documents", files={"file": ("nimbus-property-underwriting-guidelines.md",
                                                 b"Any risk above USD 250 million must be referred to "
                                                 b"the chief underwriter before a quote is issued.")})

    def fake_call(url, body=None, timeout=30):
        path = url.split("http://svc", 1)[1]
        r = client.post(path, json=body) if body is not None else client.get(path)
        return r.status_code, (r.json() if r.content else {})
    monkeypatch.setattr(st, "call", fake_call)
    monkeypatch.setattr("sys.argv", ["selftest", "--url", "http://svc"])
    assert st.main() == 0
    assert "PROMOTE" in capsys.readouterr().out


def test_selftest_fails_when_no_results(client, monkeypatch):
    def fake_call(url, body=None, timeout=30):
        path = url.split("http://svc", 1)[1]
        r = client.post(path, json=body) if body is not None else client.get(path)
        return r.status_code, (r.json() if r.content else {})
    monkeypatch.setattr(st, "call", fake_call)
    monkeypatch.setattr("sys.argv", ["selftest", "--url", "http://svc", "--min-results", "1"])
    assert st.main() == 1
