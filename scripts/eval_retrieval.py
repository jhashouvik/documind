#!/usr/bin/env python3
"""Measure retrieval quality of knowledge-service against a golden set.

    python3 scripts/eval_retrieval.py --url http://kb.localtest.me --k 5
    python3 scripts/eval_retrieval.py --url http://kb.localtest.me --answers \
        --agent-url http://documind.localtest.me      # also grade LLM answers (costs tokens)

Retrieval metrics (no LLM, free):
  hit@k  share of questions whose expected document appears in the top k results
  MRR    mean reciprocal rank: 1/rank of the first correct result (1.0 = always first)
Answer metric (optional):
  keyword accuracy: share of answers that contain the expected fact
Only uses the Python standard library, so it runs anywhere.
"""
import argparse
import json
import sys
import time
import urllib.request


def post(url: str, body: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://kb.localtest.me", help="knowledge-service base URL")
    ap.add_argument("--golden", default="eval/golden.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--answers", action="store_true", help="also ask agent-service (uses tokens)")
    ap.add_argument("--agent-url", default="http://documind.localtest.me")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.golden) if line.strip()]
    hits, rr, kw_ok = 0, 0.0, 0
    print(f"{'#':>2}  {'rank':>4}  {'score':>5}  question")
    for i, row in enumerate(rows, 1):
        res = post(f"{args.url}/v1/search", {"query": row["question"], "top_k": args.k,
                                               "score_threshold": 0})["results"]
        files = [r["filename"] for r in res]
        rank = files.index(row["expected_file"]) + 1 if row["expected_file"] in files else None
        if rank:
            hits += 1
            rr += 1 / rank
        top = res[0]["score"] if res else 0
        line = f"{i:>2}  {rank or '-':>4}  {top:5.3f}  {row['question']}"
        if args.answers:
            ans = post(f"{args.agent_url}/v1/chat", {"message": row["question"]})["answer"]
            ok = all(k.lower() in ans.lower() for k in row["keywords"])
            kw_ok += ok
            line += f"   answer {'OK' if ok else 'MISSING ' + str(row['keywords'])}"
            time.sleep(3.2)     # stay under the default rate limit (20/min)
        print(line)
    n = len(rows)
    print(f"\nhit@{args.k} = {hits / n:.2f}   MRR = {rr / n:.2f}   ({n} questions)")
    if args.answers:
        print(f"answer keyword accuracy = {kw_ok / n:.2f}")
    return 0 if hits / n >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
