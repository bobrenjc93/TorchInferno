#!/usr/bin/env python3
"""End-to-end paged serving loop == dense: the engine-logic capstone.

generate_paged_continuous composes every validated paged piece -- page-aware
admission, per-request prefill, graphed variable-batch decode, free-on-completion --
into the continuous-batching loop the OpenAI online batcher will wrap. This runs
MORE requests than max_active (so admission/completion/re-admission all fire) with
VARIABLE max_new_tokens, and asserts each request's greedy tokens match an
independent DENSE per-request greedy reference.

  torchrun ... --nproc-per-node 8 scripts/validate_paged_continuous.py
"""
import os

import torch
import torch.distributed as dist

from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged_serving import PagedEngine, generate_paged, generate_paged_continuous


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def main():
    import flashinfer  # noqa: F401

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")

    model = Llama3TensorParallelForCausalLM.from_pretrained(dtype="bfloat16").eval()
    log(rank, "model loaded")

    # 6 requests, max_active=3 -> admission + completion + re-admission; varied max_new.
    base_prompts = [
        [1, 5, 9, 13, 2, 6], [3, 7, 11, 4, 8, 10], [2, 14, 1, 9, 5, 7],
        [4, 8, 12, 6, 10, 2], [1, 3, 5, 7, 9, 11], [6, 2, 8, 4, 10, 12],
    ]
    max_news = [3, 7, 4, 9, 5, 6]
    requests = list(zip(base_prompts, max_news))

    with torch.inference_mode():
        got_eager = generate_paged_continuous(model, requests, page_size=16, max_active=3, use_graph=False)
        got = generate_paged_continuous(model, requests, page_size=16, max_active=3, use_graph=True)

        # dense per-request greedy reference
        ref = []
        for prompt, max_new in requests:
            seq = list(prompt)
            gen = []
            for _ in range(max_new):
                c = model.allocate_cache(1, max_seq_len=64, cache_backend="dense")
                logits, _ = model.forward(torch.tensor([seq], dtype=torch.long, device=dev), cache=c, use_cache=True)
                t = int(logits[0, -1, :].argmax())
                gen.append(t)
                seq.append(t)
            ref.append(gen)

        # ISOLATION 1: generate_paged batched on the 6 prompts, uniform max_new=4 (sanity)
        gp = generate_paged(model, base_prompts, max_new_tokens=4, page_size=16, use_graph=False)
        gp_ref = []
        for prompt in base_prompts:
            seq = list(prompt); g = []
            for _ in range(4):
                c = model.allocate_cache(1, max_seq_len=64, cache_backend="dense")
                lg, _ = model.forward(torch.tensor([seq], dtype=torch.long, device=dev), cache=c, use_cache=True)
                t = int(lg[0, -1, :].argmax()); g.append(t); seq.append(t)
            gp_ref.append(g)
        log(rank, f"[isolation] generate_paged(batched) == dense: {gp == gp_ref}")

        # ISOLATION 2: continuous with ONE request, max_active=1 (per-request prefill, single decode)
        one = generate_paged_continuous(model, [requests[1]], page_size=16, max_active=1, use_graph=False)
        log(rank, f"[isolation] continuous(1 req, max_active=1) == dense: {one == [ref[1]]}")
        if rank == 0 and one != [ref[1]]:
            print(f"    one={one[0]}\n    ref={ref[1]}", flush=True)

        # ISOLATION 3: incremental PagedEngine (submit/step) == batch generate_paged_continuous
        eng = PagedEngine(model, page_size=16, max_active=3, max_seq=64, use_graph=True)
        for i, (prompt, max_new) in enumerate(requests):
            eng.submit(i, prompt, max_new)
        eng_out: list[list[int]] = [[] for _ in requests]
        guard = 0
        while eng.has_work() and guard < 10000:
            for ext_id, tok, _fin in eng.step():
                eng_out[ext_id].append(tok)
            guard += 1
        log(rank, f"[isolation] PagedEngine(submit/step) == generate_paged_continuous: {eng_out == got}")
        if rank == 0 and eng_out != got:
            for i, (e, g) in enumerate(zip(eng_out, got)):
                if e != g:
                    print(f"    req {i}: engine={e}\n            batch ={g}", flush=True)

    log(rank, f"[continuous] eager-decode  == dense: {got_eager == ref}")
    log(rank, f"[continuous] graphed-decode == dense: {got == ref}")
    log(rank, f"[continuous] eager == graphed: {got_eager == got}")
    ok = got == ref
    if not ok and rank == 0:
        for i, (g, r) in enumerate(zip(got, ref)):
            mark = "OK" if g == r else "DIFF"
            print(f"  req {i} [{mark}] max_new={max_news[i]}\n    got={g}\n    ref={r}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
