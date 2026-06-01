#!/usr/bin/env python3
import subprocess, time, os, sys, concurrent.futures, statistics
os.chdir("/data/users/bobren/d/TorchInferno")
subprocess.run(["pkill", "-9", "-f", "torchinferno"], capture_output=True)
time.sleep(3)
proc = subprocess.Popen(
    ["python", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "8",
     "-m", "torchinferno.openai_server", "--model", "meta-llama/Meta-Llama-3.1-70B-Instruct",
     "--tensor-parallel-size", "8", "--port", "8001", "--trust-remote-code"],
    stdout=subprocess.DEVNULL, stderr=open("srv.err", "w"),
    env={**os.environ, "PYTHONPATH": "src"})
import urllib.request
for i in range(90):
    time.sleep(10)
    try:
        if b"ok" in urllib.request.urlopen("http://localhost:8001/health", timeout=2).read():
            print(f"Ready {(i+1)*10}s"); break
    except: pass
else: print("TIMEOUT"); sys.exit(1)

# Check persistent engine
with open("srv.err") as f:
    err = f.read()
if "PERSISTENT" in err:
    print("PERSISTENT ENGINE: " + [l for l in err.split("\n") if "PERSISTENT" in l][0])

import openai
client = openai.OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")
def do(i):
    num = str(12345 + i * 7919)
    t0 = time.perf_counter()
    r = client.chat.completions.create(model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        messages=[{"role": "system", "content": "Calculator. Respond with only the number."},
                  {"role": "user", "content": f"1 * {num} ="}],
        max_tokens=len(num)//3+16, temperature=0.0, stream=True)
    tokens=[]; ttft=None
    for c in r:
        d=c.choices[0].delta if c.choices else None
        if d and d.content:
            if ttft is None: ttft=(time.perf_counter()-t0)*1000
            tokens.append(d.content)
    e2e=(time.perf_counter()-t0)*1000
    return ttft or 0, len(tokens), e2e, len(tokens)/(e2e/1000) if e2e>0 and tokens else 0
do(0)  # warmup
for label, N in [("64 concurrent", 64), ("200 concurrent", 200)]:
    t0=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(N, 64)) as p:
        futs=[p.submit(do,i) for i in range(1,N+1)]
        results=[f.result() for f in concurrent.futures.as_completed(futs)]
    wall=(time.perf_counter()-t0)*1000
    ttfts=sorted([r[0] for r in results])
    tps_list=[r[3] for r in results if r[3]>0]
    total_tok=sum(r[1] for r in results)
    mid=len(ttfts)//2
    p90=int(len(ttfts)*0.9)
    print(f"\n{label}:")
    print(f"  TTFT: p50={ttfts[mid]:.0f}ms p90={ttfts[p90]:.0f}ms")
    print(f"  Per-req tput: {statistics.median(tps_list):.1f} tps")
    print(f"  Sys tput: {total_tok/(wall/1000):.0f} tok/s")
subprocess.run(["pkill", "-9", "-f", "torchinferno"], capture_output=True)
