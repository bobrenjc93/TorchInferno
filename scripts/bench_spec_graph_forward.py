import math, os, torch, torch.distributed as dist
from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.runtime.paged import LayeredPagedKVCache
from torchinferno.runtime.paged_serving import _plan_prefill, _plan_decode
PAGE=16
def log(r,*a):
    if r==0: print(*a,flush=True)
def gbench(fn,it=50):
    try:
        fn(); torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
        torch.cuda.synchronize()
        for _ in range(10): g.replay()
        torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it): g.replay()
        e.record();torch.cuda.synchronize()
        return s.elapsed_time(e)/it, True
    except Exception as ex:
        return str(ex)[:80], False
def main():
    import flashinfer
    dist.init_process_group("nccl"); r=dist.get_rank()
    loc=int(os.environ.get("LOCAL_RANK",r)); torch.cuda.set_device(loc); dev=torch.device(f"cuda:{loc}")
    m=Llama3TensorParallelForCausalLM.from_pretrained(dtype="bfloat16").eval()
    nqo=m.layers[0].local_attention_heads; nkv=m.layers[0].local_key_value_heads; hd=m.config.head_dim
    N=16; K=8; CTX=300
    cache=LayeredPagedKVCache(num_layers=len(m.layers),num_pages=N*math.ceil((CTX+K+8)/PAGE)+16,page_size=PAGE,num_key_value_heads=nkv,head_dim=hd,device=dev,dtype=m.dtype)
    g=torch.Generator(device="cpu").manual_seed(1)
    rids=[f"r{i}" for i in range(N)]
    with torch.inference_mode():
        for rid in rids:
            cache.reserve(rid,CTX); cache._sequences[rid].length=CTX
        # warm KV with a prefill
        pw0=_plan_prefill(flashinfer,cache,rids,[CTX]*N,nqo,nkv,hd,PAGE)
        bt=cache.block_table(rids)
        m.forward_prefill_paged(torch.randint(0,m.config.vocab_size,(N,CTX),generator=g).to(dev),cache,request_ids=rids,prefill_wrapper=pw0,block_table=bt)
        # DECODE (1 token) graphed
        dtok=torch.zeros(N,1,dtype=torch.long,device=dev)
        dpos=torch.full((N,),CTX,dtype=torch.long,device=dev)
        dw=_plan_decode(flashinfer,cache,rids,nqo,nkv,hd,PAGE)
        td,okd=gbench(lambda: m.forward_decode_paged(dtok,cache,request_ids=rids,positions=dpos,decode_wrapper=dw))
        # SPEC (1+K tokens) graphed -- forward_prefill_paged at per-row start
        for rid in rids:
            cache.reserve(rid,CTX+1+K); cache._sequences[rid].length=CTX+1+K
        stok=torch.zeros(N,1+K,dtype=torch.long,device=dev)
        sstart=torch.full((N,),CTX,dtype=torch.long,device=dev)
        bt2=cache.block_table(rids)
        pw=_plan_prefill(flashinfer,cache,rids,[1+K]*N,nqo,nkv,hd,PAGE)
        ts,oks=gbench(lambda: m.forward_prefill_paged(stok,cache,request_ids=rids,prefill_wrapper=pw,block_table=bt2,start_position=sstart))
        log(r,f"N={N} K={K} CTX={CTX}")
        log(r,f"  graphed DECODE(1tok)  = {td if okd else 'FAIL:'+str(td)}")
        log(r,f"  graphed SPEC(1+{K}tok) = {ts if oks else 'FAIL:'+str(ts)}")
        if okd and oks:
            log(r,f"  spec/decode per-step ratio = {ts/td:.2f}x  (if < 3.28 -> spec WINS wall-time at 3.28x fewer steps)")
    dist.destroy_process_group()
main()
