"""Release gate for a NEW knowledge-service version, run as a Kubernetes Job by
an Argo Rollouts AnalysisTemplate before a blue-green switch (see
k8s/gitops/knowledge-service/analysis-selftest.yaml). Exit code 0 = promote, non-zero = abort.

    python -m app.selftest --url http://knowledge-service-preview:8001 --min-hit-rate 0.8

Checks, in order:
  1. /readyz answers 200                       (the new pods are healthy)
  2. /v1/options lists the default embedding model
  3. a search returns at least --min-results passages
  4. optional: hit@5 on the golden questions >= --min-hit-rate (retrieval quality)
Only the standard library is used.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path


def call(url: str, body: dict | None = None, timeout: float = 30) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--query", default="When must a risk be referred to the chief underwriter?")
    ap.add_argument("--min-results", type=int, default=1)
    ap.add_argument("--min-hit-rate", type=float, default=0.0, help="0 disables the golden-set check")
    args = ap.parse_args()
    ok = True

    status, _ = call(f"{args.url}/readyz")
    print(f"[{'PASS' if status == 200 else 'FAIL'}] readyz -> {status}")
    ok &= status == 200

    status, opts = call(f"{args.url}/v1/options")
    good = status == 200 and opts.get("default_embedding_model") in opts.get("embedding_models", [])
    print(f"[{'PASS' if good else 'FAIL'}] options -> default model {opts.get('default_embedding_model')}")
    ok &= good

    status, res = call(f"{args.url}/v1/search", {"query": args.query, "top_k": 5})
    n = len(res.get("results", []))
    good = status == 200 and n >= args.min_results
    print(f"[{'PASS' if good else 'FAIL'}] search -> HTTP {status}, {n} results")
    ok &= good

    if args.min_hit_rate > 0:
        rows = [json.loads(line) for line in (Path(__file__).parent / "golden.jsonl").read_text().splitlines()
                if line.strip()]
        hits = 0
        for row in rows:
            _, res = call(f"{args.url}/v1/search", {"query": row["question"], "top_k": 5, "score_threshold": 0})
            hits += row["expected_file"] in [r["filename"] for r in res.get("results", [])]
        rate = hits / len(rows)
        good = rate >= args.min_hit_rate
        print(f"[{'PASS' if good else 'FAIL'}] golden hit@5 = {rate:.2f} (need {args.min_hit_rate})")
        ok &= good

    print("RESULT:", "PROMOTE" if ok else "ABORT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
