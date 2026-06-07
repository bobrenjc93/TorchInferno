import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank=dist.get_rank(); local=int(os.environ.get("LOCAL_RANK",rank))
torch.cuda.set_device(local); dev=torch.device(f"cuda:{local}")
H=8192; M=48
def t(fn,it=300,wu=50):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); dist.barrier(); s=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-s)/it*1e6
# symm-mem multimem baseline
import torch.distributed._symmetric_memory as symm_mem
gn=dist.group.WORLD.group_name
buf=symm_mem.empty(M,H,device=dev,dtype=torch.bfloat16); symm_mem.rendezvous(buf,gn)
x=torch.randn(M,H,device=dev,dtype=torch.bfloat16); buf.copy_(x)
mm=t(lambda: torch.ops.symm_mem.multimem_all_reduce_(buf,"sum",gn))
# vllm custom AR
car_us=None; err=None
try:
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    gloo=dist.new_group(backend="gloo")
    car=CustomAllreduce(group=gloo, device=dev)
    if car.disabled:
        err="CustomAllreduce.disabled=True"
    else:
        inp=torch.randn(M,H,device=dev,dtype=torch.bfloat16)
        out=car.custom_all_reduce(inp)
        if out is None: err="custom_all_reduce returned None (size not supported)"
        else: car_us=t(lambda: car.custom_all_reduce(inp))
except Exception as e:
    err=repr(e)[:160]
if rank==0:
    print(f"[48,8192] symm-mem multimem = {mm:.1f}us")
    if car_us is not None: print(f"[48,8192] vllm custom_ar    = {car_us:.1f}us  speedup vs multimem={mm/car_us:.2f}x")
    else: print(f"vllm custom_ar unavailable: {err}")
dist.destroy_process_group()
