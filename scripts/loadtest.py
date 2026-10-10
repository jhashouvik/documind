#!/usr/bin/env python3
"""Tiny async load generator (standard library + threads, no installs needed).

    # drive knowledge-service CPU to watch the HPA scale it (free, no LLM):
    python3 scripts/loadtest.py --mode search --url http://kb.localtest.me --concurrency 8 --duration 180

    # a few chat requests end-to-end (COSTS TOKENS - keep it small: every request is a
    # full agent turn of 5-10 LLM calls, and 20 requests/minute per client IP are allowed):
    python3 scripts/loadtest.py --mode chat --url http://documind.localtest.me --concurrency 2 --duration 30
"""
import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request

QUESTIONS = [
    "referral threshold for total insured value", "sprinkler requirements for warehouses",
    "earthquake zone 4 deductible", "flood sublimit in a floodplain", "risk survey validity",
    "first notice of loss acknowledgement", "senior adjuster payment authority",
    "catastrophe notification to retrocession partners", "P1 incident response time",
    "API key rotation policy", "loss-free discount", "maximum line size per risk",
]


def worker(args, stop, latencies, errors, lock):
    while not stop.is_set():
        q = random.choice(QUESTIONS)
        if args.mode == "search":
            url, body = f"{args.url}/v1/search", {"query": q, "top_k": 5}
        else:
            url, body = f"{args.url}/v1/chat", {"message": f"Briefly: {q}?"}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                r.read()
            with lock:
                latencies.append(time.perf_counter() - t0)
        except (urllib.error.URLError, TimeoutError) as exc:
            with lock:
                errors.append(str(getattr(exc, "code", exc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["search", "chat"], default="search")
    ap.add_argument("--url", default="http://kb.localtest.me")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--duration", type=int, default=60)
    args = ap.parse_args()

    stop, lock = threading.Event(), threading.Lock()
    latencies: list[float] = []
    errors: list[str] = []
    threads = [threading.Thread(target=worker, args=(args, stop, latencies, errors, lock), daemon=True)
               for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    start = time.time()
    try:
        while time.time() - start < args.duration:
            time.sleep(10)
            with lock:
                n = len(latencies)
                p95 = statistics.quantiles(latencies, n=20)[-1] if n > 20 else float("nan")
            print(f"{int(time.time() - start):>4}s  ok={n:<6} errors={len(errors):<4} "
                  f"rps={n / (time.time() - start):6.1f}  p95={p95 * 1000:7.0f} ms", flush=True)
    except KeyboardInterrupt:
        pass
    stop.set()
    if errors:
        print("sample errors:", sorted(set(errors))[:5])


if __name__ == "__main__":
    main()
