#!/usr/bin/env python3
"""Simulate multi_turn: 16 concurrent conversations, 4 turns each."""
import openai, time, concurrent.futures, statistics, sys

client = openai.OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")

def conversation(conv_id):
    ttfts = []
    for turn in range(4):
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            messages=[{"role": "user", "content": f"Conv {conv_id} turn {turn}: what is {conv_id*10+turn}+1?"}],
            max_tokens=10, temperature=0.0, stream=True)
        tokens = []; ttft = None
        for c in r:
            d = c.choices[0].delta if c.choices else None
            if d and d.content:
                if ttft is None: ttft = (time.perf_counter() - t0) * 1000
                tokens.append(d.content)
        ttfts.append(ttft or 0)
    return conv_id, ttfts

# Warmup
conversation(0)

t0 = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as p:
    futs = [p.submit(conversation, i) for i in range(1, 17)]
    results = [f.result() for f in concurrent.futures.as_completed(futs)]
wall = (time.perf_counter() - t0) * 1000

all_ttfts = [t for _, ttfts in results for t in ttfts]
print(f"16 conversations × 4 turns = {len(all_ttfts)} requests:")
print(f"  TTFT: p50={sorted(all_ttfts)[len(all_ttfts)//2]:.0f}ms p90={sorted(all_ttfts)[int(len(all_ttfts)*0.9)]:.0f}ms")
print(f"  Wall: {wall:.0f}ms")
print(f"  Avg per-turn: {statistics.mean(all_ttfts):.0f}ms")
