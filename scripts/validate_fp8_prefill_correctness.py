#!/usr/bin/env python3
"""Correctness GATE for FP8 prefill: does tensorwise-FP8 quant of the big prefill
GEMMs (gate_up + down) preserve greedy generation vs bf16? FP8 prefill rewrites the
KV cache + first logits; tensorwise (scalar) scaling is lossier than rowwise. If it
flips greedy badly (like int4 down_proj did), the lever is DEAD regardless of speed
and the fused-kernel build is not worth starting.

Uses FAKE-QUANT (quant->dequant->bf16 GEMM), which is numerically IDENTICAL to
torch._scaled_mm (both accumulate in fp32; the error is purely fp8 input rounding).
So this measures the real FP8 correctness without the _scaled_mm/fused-kernel work.
Monkeypatches the prefill gate_up + down GEMMs; decode stays bf16.

  FP8_PREFILL=0/1 toggles. Run both, diff the token lines:
  torchrun --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 --master-port 29797 \
           --nproc-per-node 8 scripts/validate_fp8_prefill_correctness.py
"""
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

import torchinferno.models.llama3.tensor_parallel as tp
from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged_serving import PagedEngine

FP8 = torch.float8_e4m3fn


def fake_quant_tw(x):
    # tensorwise e4m3 fake-quant: scale by global amax/448, round to fp8, dequant.
    s = (x.abs().amax() / 448.0).clamp(min=1e-6)
    return (x / s).to(FP8).to(x.dtype) * s


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def install_fp8_prefill():
    # forward_prefill_paged reuses the DECODE MLP path (_mlp_project_decode_reduce),
    # so patch that: fp8 fake-quant the gate_up + down GEMMs (the two big MLP GEMMs).
    # This exercises fp8 on BOTH paged prefill and decode -> a strict superset of the
    # fp8-prefill-only correctness question. gate_up via marlin int4 is already known
    # greedy-EXACT, and int4 DOWN flipped greedy -- fp8 (8-bit) is far more accurate,
    # so the open question is whether fp8 down holds where int4 down did not.
    Layer = tp._Llama3TensorParallelLayer

    orig_decode_mlp = Layer._mlp_project_decode_reduce
    # PREFILL-ONLY isolation: M = rows*tokens. Paged prefill of the 128-tok prompt has
    # M=128; decode steps have M<=4. Gate fp8 on M>16 so ONLY prefill is quantized
    # (decode stays bf16) -> isolates the fp8-PREFILL correctness from fp8-decode.
    only_prefill = os.environ.get("FP8_PREFILL_ONLY", "1") == "1"

    def patched_decode_mlp(self, hidden):
        m = hidden.numel() // hidden.size(-1)
        if only_prefill and m <= 16:
            return orig_decode_mlp(self, hidden)
        gate, up = F.linear(
            fake_quant_tw(hidden), fake_quant_tw(self.gate_up_proj_weight)
        ).split((self.local_intermediate_size, self.local_intermediate_size), dim=-1)
        activated = tp._tp_decode_swiglu(gate, up)
        return self._decode_linear_all_reduce(
            fake_quant_tw(activated), fake_quant_tw(self.down_proj_weight), "mlp",
        )

    Layer._mlp_project_decode_reduce = patched_decode_mlp


def run(model):
    g = torch.Generator(device="cpu").manual_seed(11)
    prompt = torch.randint(0, model.config.vocab_size, (128,), generator=g).tolist()
    with torch.inference_mode():
        eng = PagedEngine(model, page_size=16, max_active=4, max_seq=512, use_graph=True)
        eng.submit("t", prompt, 48, eos_token_id=None)
        toks = []
        while eng.has_work():
            for e_id, tok, fin in eng.step():
                if e_id == "t":
                    toks.append(tok)
    return toks


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    fp8 = os.environ.get("FP8_PREFILL", "0") == "1"
    if fp8:
        install_fp8_prefill()
    model = Llama3TensorParallelForCausalLM.from_pretrained(dtype="bfloat16").eval()
    log(rank, f"model loaded (FP8_PREFILL={fp8})")
    toks = run(model)
    log(rank, f"[fp8={int(fp8)}] {len(toks)} tokens:")
    log(rank, " ".join(str(t) for t in toks))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
