#!/usr/bin/env python3
"""Verify shared-prefix reuse (self_consistency pattern).

Sends the SAME long prompt N times concurrently, twice in a row. With prefix
reuse working, the second wave should have much lower TTFT than the first
(the shared prompt's KV is prefilled once and reused).
"""

import asyncio
import json
import sys
import time

import aiohttp

URL = "http://localhost:8321/v1/chat/completions"
# A long shared prompt (~400 tokens) so prefill cost is meaningful.
SHARED = "You are a careful mathematician. " + " ".join(
    f"Fact {i}: number {i} is {'even' if i % 2 == 0 else 'odd'}." for i in range(180)
)


def body() -> dict:
    return {
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "messages": [{"role": "user", "content": SHARED + "\nWhat is 2+2?"}],
        "max_tokens": 16,
        "temperature": 0.7,
        "stream": True,
    }


async def one(session):
    t0 = time.perf_counter()
    ttft = None
    async with session.post(URL, json=body()) as resp:
        async for line in resp.content:
            t = line.decode().strip()
            if not t.startswith("data: "):
                continue
            d = t[6:]
            if d == "[DONE]":
                break
            try:
                c = json.loads(d).get("choices", [{}])[0].get("delta", {}).get("content", "")
                if c and ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
            except json.JSONDecodeError:
                pass
    return ttft or 0.0


async def wave(session, n, label):
    t0 = time.perf_counter()
    ttfts = sorted(await asyncio.gather(*[one(session) for _ in range(n)]))
    wall = (time.perf_counter() - t0) * 1000
    print(f"{label}: n={n} TTFT p50={ttfts[len(ttfts)//2]:.0f} min={ttfts[0]:.0f} "
          f"max={ttfts[-1]:.0f} wall={wall:.0f}ms")
    return ttfts[len(ttfts)//2]


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    async with aiohttp.ClientSession() as s:
        await one(s)  # warm
        w1 = await wave(s, n, "wave1 (cold shared prefix)")
        await asyncio.sleep(0.5)
        w2 = await wave(s, n, "wave2 (warm shared prefix)")
        print(f"\nwave2/wave1 TTFT ratio: {w2/max(w1,1):.2f} (lower = reuse working)")


if __name__ == "__main__":
    asyncio.run(main())
