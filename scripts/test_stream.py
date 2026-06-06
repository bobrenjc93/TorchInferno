#!/usr/bin/env python3
"""Streaming load test: continuous Poisson arrivals at a target QPS, mirroring
inference-bench's pattern (a stream of requests with bounded concurrency) rather
than a single synchronous burst.

This is the tool needed to LOCALLY validate streaming-sensitive optimizations
(chunked prefill, KV-token-bounded admission) whose benefit only shows under
continuous arrivals — a burst harness (test_ttft.py) cannot see it.

Usage:
  python scripts/test_stream.py --n 256 --qps 32 --prompt-tokens 640 --max-tokens 32
  python scripts/test_stream.py --n 512 --qps 64 --prompt-tokens 16 --max-tokens 256  # long_output-ish
"""

import argparse
import asyncio
import json
import random
import time

import aiohttp

URL = "http://localhost:8321/v1/chat/completions"


def make_prompt(prompt_tokens: int, i: int) -> str:
    filler = " ".join(f"w{j}" for j in range(max(1, prompt_tokens // 2)))
    return f"req{i} {filler}\nReply briefly."


async def one(session, i, prompt_tokens, max_tokens, temperature):
    body = {
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "messages": [{"role": "user", "content": make_prompt(prompt_tokens, i)}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    n = 0
    try:
        async with session.post(URL, json=body) as resp:
            async for line in resp.content:
                t = line.decode("utf-8", "ignore").strip()
                if not t.startswith("data: "):
                    continue
                d = t[6:]
                if d == "[DONE]":
                    break
                try:
                    c = json.loads(d).get("choices", [{}])[0].get("delta", {}).get("content", "")
                except json.JSONDecodeError:
                    continue
                if c and ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                if c:
                    n += 1
    except Exception as e:
        return {"ttft": 0.0, "tpot": 0.0, "n": 0, "err": repr(e)}
    total = (time.perf_counter() - t0) * 1000
    tpot = (total - (ttft or 0)) / max(1, n - 1) if n > 1 else 0.0
    return {"ttft": ttft or 0.0, "tpot": tpot, "n": n, "err": None}


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--qps", type=float, default=0.0,
                    help="open-loop Poisson arrival rate; 0 disables (use --concurrency)")
    ap.add_argument("--concurrency", type=int, default=64,
                    help="closed-loop: max in-flight requests (mirrors inference-bench)")
    ap.add_argument("--prompt-tokens", type=int, default=640)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()

    async with aiohttp.ClientSession() as s:
        await one(s, -1, a.prompt_tokens, 8, a.temperature)  # warm
        t0 = time.perf_counter()
        if a.qps > 0:
            # Open-loop: Poisson arrivals at target QPS (can overload the server).
            tasks = []
            for i in range(a.n):
                tasks.append(asyncio.create_task(
                    one(s, i, a.prompt_tokens, a.max_tokens, a.temperature)))
                await asyncio.sleep(random.expovariate(a.qps))
            res = await asyncio.gather(*tasks)
        else:
            # Closed-loop: keep `concurrency` requests in flight, fire as slots
            # free (self-throttling, matches inference-bench's "N concurrent").
            sem = asyncio.Semaphore(a.concurrency)

            async def guarded(i):
                async with sem:
                    return await one(s, i, a.prompt_tokens, a.max_tokens, a.temperature)

            res = await asyncio.gather(*[guarded(i) for i in range(a.n)])
        wall = time.perf_counter() - t0

    ok = [r for r in res if r["n"] > 0]
    errs = [r for r in res if r["err"]]
    ttfts = [r["ttft"] for r in ok]
    tpots = [r["tpot"] for r in ok if r["tpot"] > 0]
    toks = sum(r["n"] for r in ok)
    mode = f"qps={a.qps}" if a.qps > 0 else f"concurrency={a.concurrency}"
    print(f"n={a.n} {mode} prompt~{a.prompt_tokens}tok max_tokens={a.max_tokens} ok={len(ok)} err={len(errs)}")
    print(f"TTFT ms  p50={pct(ttfts,50):.0f}  p90={pct(ttfts,90):.0f}  p99={pct(ttfts,99):.0f}")
    print(f"TPOT ms  p50={pct(tpots,50):.0f}  p90={pct(tpots,90):.0f}")
    print(f"wall={wall:.1f}s  agg_throughput={toks/wall:.0f} tok/s")
    if errs:
        print("first error:", errs[0]["err"])


if __name__ == "__main__":
    asyncio.run(main())
