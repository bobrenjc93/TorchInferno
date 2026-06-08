#!/usr/bin/env python3
"""VALIDATED: prompt-lookup (n-gram) speculative decoding is correct + ~4x fewer decode
forwards on long_output's ECHO output. long_output asks "1 * <big_num> =" and the answer
is big_num's digits -- which are IN THE PROMPT. So proposing the next tokens by matching
the recent context against the prompt (no draft model) + verifying in one forward accepts
~3.3 tokens/step. Verification preserves GREEDY-EXACT output (OUTPUT-MATCH=True).

Result (real 70B TP8, 46-tok echo prompt, 48 gen): baseline 48 forwards vs spec 12 = 4.0x
fewer, output IDENTICAL. KEY correctness fix: after accepting J of K drafts, set the KV
cache length back to pos+J (rejected-draft KV beyond the length is never read).

=> potential long_output TPOT flip (22.9 -> ~6-8ms vs vllm 14.7) + E2E + throughput. A
GENERAL optimization (helps any echo/repetition-heavy output), greedy-exact. Remaining
work = serving-engine integration (dense continuous batcher decode path: eager K-token
forward + verify + per-request variable acceptance/batching + TP -- deterministic per-rank
so no COW-style collective divergence).

  torchrun --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 --master-port 29810 \
           --nproc-per-node 8 scripts/validate_prompt_lookup_specdecode.py
"""
import math, os, time, torch, torch.distributed as dist
from transformers import AutoTokenizer
from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged import LayeredPagedKVCache
from torchinferno.runtime.paged_serving import _plan_prefill, _greedy_tokens
PAGE=16
def log(r,*a):
    if r==0: print(*a,flush=True)
def main():
    import flashinfer
    dist.init_process_group("nccl"); r=dist.get_rank()
    loc=int(os.environ.get("LOCAL_RANK",r)); torch.cuda.set_device(loc); dev=torch.device(f"cuda:{loc}")
    m=Llama3TensorParallelForCausalLM.from_pretrained(dtype="bfloat16").eval()
    nqo=m.layers[0].local_attention_heads; nkv=m.layers[0].local_key_value_heads; hd=m.config.head_dim
    tok=AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-70B-Instruct')
    bignum="".join(str((i*7+3)%10) for i in range(80))
    ids=tok(f"Q: What is 1 * {bignum}? Reply with just the number.\nA: ")['input_ids']
    NGEN=48; MAXP=math.ceil((len(ids)+NGEN+16)/PAGE)+8

    def fwd(cache,rid,toks,start):
        # forward toks at positions [start, start+len); set length; return per-pos argmax
        cache.reserve(rid,start+len(toks)); cache._sequences[rid].length=start+len(toks)
        pw=_plan_prefill(flashinfer,cache,[rid],[len(toks)],nqo,nkv,hd,PAGE)
        bt=cache.block_table([rid],max_pages=MAXP)
        out=m.forward_prefill_paged(torch.tensor([toks],device=dev),cache,request_ids=[rid],prefill_wrapper=pw,block_table=bt,start_position=start)
        return [int(x) for x in _greedy_tokens(m,out[0].float())]

    with torch.inference_mode():
        # BASELINE greedy
        c1=LayeredPagedKVCache(num_layers=len(m.layers),num_pages=MAXP+4,page_size=PAGE,num_key_value_heads=nkv,head_dim=hd,device=dev,dtype=m.dtype)
        preds=fwd(c1,"b",list(ids),0); base=[preds[-1]]; fb=1
        while len(base)<NGEN:
            p=fwd(c1,"b",[base[-1]],len(ids)+len(base)-1); base.append(p[-1]); fb+=1
        # SPEC prompt-lookup with correct length mgmt
        c2=LayeredPagedKVCache(num_layers=len(m.layers),num_pages=MAXP+4,page_size=PAGE,num_key_value_heads=nkv,head_dim=hd,device=dev,dtype=m.dtype)
        K=8; NG=3
        preds=fwd(c2,"s",list(ids),0); spec=[preds[-1]]; fs=1; acc=0
        while len(spec)<NGEN:
            full=list(ids)+spec
            prop=[]; last=tuple(full[-NG:])
            for i in range(len(full)-NG-1,-1,-1):
                if tuple(full[i:i+NG])==last: prop=full[i+NG:i+NG+K]; break
            inp=[spec[-1]]+prop
            pos=len(ids)+len(spec)-1
            p=fwd(c2,"s",inp,pos); fs+=1   # length now pos+len(inp); will be truncated below
            emit=[p[0]]
            for i in range(len(prop)):
                if prop[i]==p[i]: emit.append(p[i+1])
                else: break
            emit=emit[:NGEN-len(spec)]
            spec.extend(emit); acc+=len(emit)-1
            # CORRECT KV LENGTH: only positions [pos .. pos+len(emit)] are valid (inputs were real tokens)
            c2._sequences["s"].length=pos+len(emit)
        ok=base[:NGEN]==spec[:NGEN]
        log(r,f"prompt {len(ids)} tok, gen {NGEN}")
        log(r,f"BASELINE forwards={fb}")
        log(r,f"SPEC forwards={fs}  accepted={acc}  avg_accept/step={acc/max(1,fs-1):.1f}  OUTPUT-MATCH={ok}  forward-speedup={fb/fs:.2f}x")
        if not ok:
            d=next((i for i in range(NGEN) if base[i]!=spec[i]),NGEN)
            log(r,f"  first diff at {d}: base={base[d:d+4]} spec={spec[d:d+4]}")
    dist.destroy_process_group()
main()
