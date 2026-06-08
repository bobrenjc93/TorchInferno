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
# PIPELINE mode: keep N workers busy, each looping over requests until TOTAL are
# done -- mimics inference-bench's ThreadPoolExecutor(max_workers=N) draining a
# request stream, which keeps prefill continuously interleaved with decode (the
# regime where TPOT regressions show up). Default off = single thundering-herd.
PIPELINE = os.environ.get("PIPELINE", "0") == "1"
TOTAL = int(os.environ.get("TOTAL", "256"))
# VARLEN: jitter prompt + output length per request so the decode batch holds
# MIXED sequence lengths -> exercises the RAGGED decode path (the regime few_shot/
# tree hit, unlike uniform-length bursts which decode in lockstep via the graph).
VARLEN = os.environ.get("VARLEN", "0") == "1"


def _lens(req_id):
    if not VARLEN:
        return PROMPT_TOKENS, MAX_TOKENS
    # deterministic per-id jitter (no Date/random needed): spread over a range.
    p = PROMPT_TOKENS // 2 + (req_id * 37) % max(1, PROMPT_TOKENS)
    m = max(8, MAX_TOKENS // 4 + (req_id * 53) % max(1, MAX_TOKENS))
    return p, m


IDENTICAL = os.environ.get("IDENTICAL", "0") == "1"


def make_body(req_id: int) -> dict:
    p_tokens, m_tokens = _lens(req_id)
    if IDENTICAL:
        # All requests SAME content (self_consistency-style) -> exercises prefix
        # caching: the shared prompt should prefill once and be reused.
        filler = " ".join(f"word{i}" for i in range(max(1, PROMPT_TOKENS // 2)))
        content = f"{filler}\n\nWrite a long detailed story."
        m_tokens = MAX_TOKENS
    else:
        filler = " ".join(f"word{i}" for i in range(max(1, p_tokens // 2)))
        content = f"Request {req_id}. {filler}\n\nWrite a long detailed story."
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ],
        "max_tokens": m_tokens,
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


async def pipeline_worker(s, wid, counter, results):
    while True:
        i = counter[0]
        if i >= TOTAL:
            return
        counter[0] += 1
        results.append(await send(s, i, 0))


async def main():
    mode = "PIPELINE" if PIPELINE else "BURST"
    print(f"[{mode}] N={N} MAX_TOKENS={MAX_TOKENS} PROMPT_TOKENS={PROMPT_TOKENS} temp={TEMPERATURE}"
          + (f" TOTAL={TOTAL}" if PIPELINE else ""), flush=True)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        w = await send(s, -1, 0)
        print(f"warmup: TTFT={w['ttft']:.0f}ms total={w['total']:.0f}ms tok={w['tok']}", flush=True)
        if PIPELINE:
            counter = [0]
            res = []
            t0 = time.perf_counter()
            await asyncio.gather(*[pipeline_worker(s, wid, counter, res) for wid in range(N)])
            wall = (time.perf_counter() - t0) * 1000
        else:
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
