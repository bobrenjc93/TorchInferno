#!/usr/bin/env python3
"""Send concurrent requests and measure TTFT, matching inference-bench patterns."""

import asyncio
import json
import time
import sys
import aiohttp


SERVER_URL = "http://localhost:8321/v1/chat/completions"
NUM_REQUESTS = 16
PROMPT_TOKENS = 640  # typical few_shot/multi_turn length
MAX_TOKENS = 32


def make_request_body(req_id: int) -> dict:
    # Build a prompt that's roughly PROMPT_TOKENS tokens
    filler = " ".join(f"word{i}" for i in range(PROMPT_TOKENS // 2))
    return {
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Request {req_id}. {filler}\n\nWrite a short poem."},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
    }


async def send_request(session: aiohttp.ClientSession, req_id: int) -> dict:
    body = make_request_body(req_id)
    t_start = time.perf_counter()
    ttft = None
    token_count = 0

    async with session.post(SERVER_URL, json=body) as resp:
        async for line in resp.content:
            text = line.decode("utf-8").strip()
            if not text.startswith("data: "):
                continue
            data_str = text[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content and ttft is None:
                    ttft = (time.perf_counter() - t_start) * 1000
                if content:
                    token_count += 1
            except json.JSONDecodeError:
                continue

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000
    tpot = (total_ms - (ttft or 0)) / max(1, token_count - 1) if token_count > 1 else 0

    return {
        "req_id": req_id,
        "ttft_ms": round(ttft or 0, 1),
        "total_ms": round(total_ms, 1),
        "tokens": token_count,
        "tpot_ms": round(tpot, 1),
    }


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_REQUESTS
    print(f"Sending {n} concurrent requests with ~{PROMPT_TOKENS}-token prompts...")

    async with aiohttp.ClientSession() as session:
        # First, send 1 warmup request
        print("Warmup request...", flush=True)
        warmup = await send_request(session, -1)
        print(f"  warmup: TTFT={warmup['ttft_ms']}ms total={warmup['total_ms']}ms tokens={warmup['tokens']}")

        # Now send N concurrent requests
        print(f"\nSending {n} concurrent requests...", flush=True)
        t0 = time.perf_counter()
        tasks = [send_request(session, i) for i in range(n)]
        results = await asyncio.gather(*tasks)
        wall = (time.perf_counter() - t0) * 1000

        ttfts = sorted(r["ttft_ms"] for r in results)
        tpots = sorted(r["tpot_ms"] for r in results if r["tpot_ms"] > 0)
        total_tokens = sum(r["tokens"] for r in results)

        print(f"\n{'='*60}")
        print(f"Results: {n} concurrent requests")
        print(f"{'='*60}")
        print(f"TTFT  min={ttfts[0]:.0f}ms  median={ttfts[len(ttfts)//2]:.0f}ms  max={ttfts[-1]:.0f}ms")
        if tpots:
            print(f"TPOT  min={tpots[0]:.0f}ms  median={tpots[len(tpots)//2]:.0f}ms  max={tpots[-1]:.0f}ms")
        print(f"Wall time: {wall:.0f}ms")
        print(f"Total tokens: {total_tokens}")
        print(f"Throughput: {total_tokens / (wall / 1000):.1f} tok/s")
        print(f"\nPer-request details:")
        for r in sorted(results, key=lambda x: x["ttft_ms"]):
            print(f"  req={r['req_id']:2d}  TTFT={r['ttft_ms']:7.1f}ms  total={r['total_ms']:7.1f}ms  tokens={r['tokens']}  TPOT={r['tpot_ms']:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
