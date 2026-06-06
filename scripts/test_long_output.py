#!/usr/bin/env python3
"""Reproduce long_output scenario: many concurrent tiny-prompt requests.

Measures TTFT distribution to isolate scheduling/queuing overhead from
prefill compute (tiny prompts prefill in ~20ms, so any TTFT above that is
batcher/scheduler overhead).
"""

import asyncio
import json
import sys
import time

import aiohttp

URL = "http://localhost:8321/v1/chat/completions"


def body(i: int, max_tokens: int) -> dict:
    return {
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "messages": [{"role": "user", "content": f"Count from 1. (req {i})"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }


async def one(session, i, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    n = 0
    async with session.post(URL, json=body(i, max_tokens)) as resp:
        async for line in resp.content:
            t = line.decode().strip()
            if not t.startswith("data: "):
                continue
            d = t[6:]
            if d == "[DONE]":
                break
            try:
                obj = json.loads(d)
                c = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if c and ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                if c:
                    n += 1
            except json.JSONDecodeError:
                pass
    return {"ttft": ttft or 0.0, "n": n, "total": (time.perf_counter() - t0) * 1000}


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    async with aiohttp.ClientSession() as s:
        # warm
        await one(s, -1, 8)
        t0 = time.perf_counter()
        res = await asyncio.gather(*[one(s, i, max_tokens) for i in range(n)])
        wall = (time.perf_counter() - t0) * 1000
    ttfts = sorted(r["ttft"] for r in res)
    toks = sum(r["n"] for r in res)
    print(f"n={n} max_tokens={max_tokens}")
    print(f"TTFT  p50={ttfts[len(ttfts)//2]:.0f}  p10={ttfts[len(ttfts)//10]:.0f}  "
          f"min={ttfts[0]:.0f}  max={ttfts[-1]:.0f}  (ms)")
    print(f"Wall={wall:.0f}ms  total_tokens={toks}  agg_throughput={toks/(wall/1000):.0f} tok/s")


if __name__ == "__main__":
    asyncio.run(main())
