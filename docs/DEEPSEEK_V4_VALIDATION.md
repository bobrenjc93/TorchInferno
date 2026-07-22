# DeepSeek V4 Validation

This note records the initial `DeepSeek-V4-Flash` bring-up on one 8xH100
96 GB node. It is evidence for the implementation, not a portable performance
claim.

## Configuration

- Checkpoint: public `deepseek-ai/DeepSeek-V4-Flash`, 46 safetensor shards,
  159,609,485,896 bytes.
- TorchInferno, SGLang, and vLLM used PyTorch 2.11.0+cu130 on the same idle
  node.
- All three used all eight GPUs. TorchInferno and SGLang used their TP/EP
  layouts with Marlin MXFP4 experts. vLLM used TP8 without expert parallelism,
  its `deepseek_v4_fp8` linear/attention path, and its Marlin MXFP4 MoE backend.
- CUDA graphs and prefix/radix reuse were disabled for the comparison.
- Requests used greedy sampling, 128 input tokens, 16 output tokens, and eight
  concurrent OpenAI chat completions. Each reported usage was checked rather
  than inferred from the input text.
- Five iterations followed warmup. The table reports the median batch wall
  time and aggregate completion-token throughput.

SGLang was launched from commit `a375e9f3da` with
`--tp-size 8 --moe-runner-backend marlin --disable-radix-cache
--disable-cuda-graph --max-running-requests 8`. TorchInferno loaded kernels
prepared by `scripts/prepare_deepseek_v4_kernels.py`; a fresh serving process
also passed with `CUDA_HOME=/does/not/exist`, confirming that runtime did not
compile them.

vLLM used its official CUDA 13 wheel
`0.20.2rc1.dev107+g2a16ece2d` with tensor parallelism eight,
`--max-model-len 256 --max-num-seqs 8 --kv-cache-dtype fp8 --enforce-eager`,
and prefix caching disabled. `VLLM_USE_FLASHINFER_SAMPLER=0` selected vLLM's
PyTorch-native greedy sampler after the wheel's FlashInfer sampler failed
during warmup with an invalid CUDA resource handle. This changes sampling,
not the V4 model kernels.

## Results

| Engine/path | Median wall time | Completion throughput |
| --- | ---: | ---: |
| TorchInferno OpenAI streaming | 2.105 s | 60.8 token/s |
| SGLang OpenAI streaming | 2.023 s | 63.3 token/s |
| SGLang OpenAI non-streaming | 2.031 s | 63.0 token/s |
| vLLM OpenAI streaming | 1.825 s | 70.1 token/s |
| TorchInferno direct TP8 batch diagnostic | 1.952 s | 65.6 token/s |

The comparable TorchInferno streaming result is 3.9% behind SGLang and 13.3%
behind vLLM. The direct diagnostic does not include HTTP or scheduler work and
is included only to localize the remaining gap outside the model kernels.

The five vLLM batch wall times were 1.765, 9.927, 1.800, 10.082, and 1.825
seconds. The two latency spikes contained in that measured set coincide with
vLLM compiling `mhc_pre_big_fuse_tilelang` inside request processing on all
eight workers. The table reports the specified five-run median without
deleting or replacing those samples. TorchInferno's prepared-kernel runtime
showed no corresponding compilation.

## Disaggregated Path

The TP4 prefill plus TP4 decode configuration produced the same generated
tokens as monolithic TP4 for batch sizes one and two, including a 129-token
prompt that crosses C4 and C128 compressor boundaries. A warm batch-one KV
handoff moved 30,695,424 bytes in 6.7 ms end to end, including about 3.2 ms of
NCCL point-to-point transfer. The current implementation is synchronous, so
this validates role isolation and live heterogeneous-KV transfer; it does not
claim an eight-GPU throughput gain.
