import os, torch, torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
dist.init_process_group("nccl")
rank=dist.get_rank(); local=int(os.environ.get("LOCAL_RANK",rank))
torch.cuda.set_device(local); dev=torch.device(f"cuda:{local}")
gn=dist.group.WORLD.group_name
H=8192
def t(fn,it=200,wu=50):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); dist.barrier()
    import time; s=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-s)/it*1e6
for M in (48,):
    x=torch.randn(M,H,device=dev,dtype=torch.bfloat16)
    nccl=t(lambda: dist.all_reduce(x))
    buf=symm_mem.empty(M,H,device=dev,dtype=torch.bfloat16); symm_mem.rendezvous(buf,gn); buf.copy_(x)
    sm=t(lambda: torch.ops.symm_mem.multimem_all_reduce_(buf,"sum",gn))
    if rank==0:
        print(f"[48,8192] bf16 single allreduce: nccl={nccl:.1f}us symm={sm:.1f}us")
        print(f"  x160 (decode step allreduces): nccl={nccl*160/1000:.2f}ms symm={sm*160/1000:.2f}ms")
        print(f"  (decode step total ~21ms; if symm*160 is a big fraction -> allreduce IS the bottleneck)")
dist.destroy_process_group()
