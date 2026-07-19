import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank=dist.get_rank(); local=int(os.environ.get("LOCAL_RANK",rank))
torch.cuda.set_device(local); dev=torch.device(f"cuda:{local}")
H=8192
def t(fn,it=300,wu=50):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); dist.barrier(); s=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-s)/it*1e6
# symm-mem multimem baseline vs vLLM custom AR.
import torch.distributed._symmetric_memory as symm_mem
gn=dist.group.WORLD.group_name
car=None; err=None
try:
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    gloo=dist.new_group(backend="gloo")
    car=CustomAllreduce(group=gloo, device=dev)
    if car.disabled:
        err="CustomAllreduce.disabled=True"
except Exception as e:
    err=repr(e)[:160]
for M in (16, 32, 48, 64, 128, 256, 512, 1024):
    buf=symm_mem.empty(M,H,device=dev,dtype=torch.bfloat16); symm_mem.rendezvous(buf,gn)
    inp=torch.randn(M,H,device=dev,dtype=torch.bfloat16); buf.copy_(inp)
    mm=t(lambda: torch.ops.symm_mem.multimem_all_reduce_(buf,"sum",gn))
    car_us=None
    if car is not None and not car.disabled:
        out=car.custom_all_reduce(inp)
        if out is not None:
            car_us=t(lambda: car.custom_all_reduce(inp))
    if rank==0:
        line=f"[{M},8192] multimem={mm:.1f}us"
        if car_us is not None:
            line += f" custom_ar={car_us:.1f}us speedup={mm/car_us:.2f}x"
        else:
            line += f" custom_ar unavailable: {err or 'size not supported'}"
        print(line, flush=True)
dist.destroy_process_group()
