import os, time, torch, torch.distributed as dist
from transformers import AutoTokenizer
from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged_serving import PagedEngine
def log(r,*a):
    if r==0: print(*a,flush=True)
def run(model, prompts, NGEN):
    eng=PagedEngine(model,page_size=16,max_active=8,max_seq=1024,use_graph=False)
    for i,p in enumerate(prompts): eng.submit(f"r{i}",p,NGEN,eos_token_id=None)
    out={f"r{i}":[] for i in range(len(prompts))}
    steps=0
    while eng.has_work():
        for eid,tok,fin in eng.step(): out[eid].append(tok)
        steps+=1
    return out, steps
def main():
    dist.init_process_group("nccl"); r=dist.get_rank()
    loc=int(os.environ.get("LOCAL_RANK",r)); torch.cuda.set_device(loc)
    m=Llama3TensorParallelForCausalLM.from_pretrained(dtype="bfloat16").eval()
    tok=AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-70B-Instruct')
    NGEN=40
    prompts=[]
    for s in range(4):
        bn="".join(str((i*7+s*3+1)%10) for i in range(60))
        prompts.append(tok(f"Q: What is 1 * {bn}? Reply with just the number.\nA: ")['input_ids'])
    with torch.inference_mode():
        os.environ["TORCHINFERNO_PAGED_SPEC_DECODE"]="0"
        base,sb=run(m,prompts,NGEN)
        os.environ["TORCHINFERNO_PAGED_SPEC_DECODE"]="1"
        spec,ss=run(m,prompts,NGEN)
    allok=all(base[k][:NGEN]==spec[k][:NGEN] for k in base)
    log(r,f"ENGINE SPEC: base_steps={sb} spec_steps={ss} step-speedup={sb/max(1,ss):.2f}x  OUTPUT-MATCH(all {len(prompts)} reqs)={allok}")
    for k in list(base)[:2]:
        log(r,f"  {k}: match={base[k][:NGEN]==spec[k][:NGEN]} base[:6]={base[k][:6]} spec[:6]={spec[k][:6]}")
    dist.destroy_process_group()
main()
