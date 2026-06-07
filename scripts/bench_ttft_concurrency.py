#!/usr/bin/env python3
"""Parameterizable concurrent-load TTFT/TPOT probe against the running server.

Reproduces the inference-bench load shapes so we can A/B serving levers (e.g. the
admission cap) on the real 70B without the full harness. All knobs are env vars so
the same script measures few_shot-like (short prompt, short out, 64 conc) and
long_output-like (short prompt, LONG out, 64 conc) and multi_turn-like (long
prompt, 125 conc) shapes by only changing the environment between runs.

  N=64 MAX_TOKENS=256 PROMPT_TOKENS=200 python scripts/bench_ttft_concurrency.py
"""
import asyncio
import json
import os
import time

import aiohttp

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8321/v1/chat/completions")
MODEL = os.environ.get("MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
N = int(os.environ.get("N", "64"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", "200"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))


def make_body(req_id: int) -> dict:
    filler = " ".join(f"word{i}" for i in range(max(1, PROMPT_TOKENS // 2)))
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Request {req_id}. {filler}\n\nWrite a long detailed story."},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
    }


async def send(session, req_id, t_release):
    # Stagger nothing: all fire together to mimic a concurrency burst.
    body = make_body(req_id)
    t0 = time.perf_counter()
    ttft = None
    n_tok = 0
    async with session.post(SERVER_URL, json=body) as resp:
        async for line in resp.content:
            text = line.decode("utf-8").strip()
            if not text.startswith("data: "):
                continue
            ds = text[6:]
            if ds == "[DONE]":
                break
            try:
                data = json.loads(ds)
                content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            except json.JSONDecodeError:
                continue
            if content and ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            if content:
                n_tok += 1
    total = (time.perf_counter() - t0) * 1000
    tpot = (total - (ttft or 0)) / max(1, n_tok - 1) if n_tok > 1 else 0.0
    return {"id": req_id, "ttft": ttft or 0.0, "total": total, "tok": n_tok, "tpot": tpot}


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


async def main():
    print(f"N={N} MAX_TOKENS={MAX_TOKENS} PROMPT_TOKENS={PROMPT_TOKENS} temp={TEMPERATURE}", flush=True)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        w = await send(s, -1, 0)
        print(f"warmup: TTFT={w['ttft']:.0f}ms total={w['total']:.0f}ms tok={w['tok']}", flush=True)
        t0 = time.perf_counter()
        res = await asyncio.gather(*[send(s, i, t0) for i in range(N)])
        wall = (time.perf_counter() - t0) * 1000
    ttfts = [r["ttft"] for r in res]
    tpots = [r["tpot"] for r in res if r["tpot"] > 0]
    toks = sum(r["tok"] for r in res)
    print(f"\n=== {N} concurrent ===", flush=True)
    print(f"TTFT  p50={pct(ttfts,0.5):.0f}  p90={pct(ttfts,0.9):.0f}  max={max(ttfts):.0f} ms", flush=True)
    print(f"TPOT  p50={pct(tpots,0.5):.0f}  p90={pct(tpots,0.9):.0f} ms", flush=True)
    print(f"E2E   p50={pct([r['total'] for r in res],0.5):.0f} ms", flush=True)
    print(f"wall={wall:.0f}ms  tokens={toks}  throughput={toks/(wall/1000):.1f} tok/s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
