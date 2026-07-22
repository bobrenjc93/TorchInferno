# DeepSeek V4 Validation

This note records the `DeepSeek-V4-Flash` CUDA bring-up and optimization run on
one 8xH100 96 GB node on 2026-07-21. It is implementation-specific local
evidence, not a portable performance claim.

## Configuration

- Checkpoint: public `deepseek-ai/DeepSeek-V4-Flash`, 46 safetensor shards,
  159,609,485,896 bytes.
- TorchInferno, SGLang, and vLLM used PyTorch 2.11.0+cu130 on the same idle
  node. TorchInferno used its hybrid TP8 path, which shards dense tensors and
  experts across the same eight ranks. SGLang and vLLM used TP8 without a
  separate expert-parallel process group.
- TorchInferno and SGLang used Marlin MXFP4 experts. vLLM used its
  `deepseek_v4_fp8` path, FP8 KV cache, and Marlin MXFP4 MoE backend.
- SGLang radix reuse and vLLM prefix caching were disabled. The TorchInferno V4
  model does not advertise prefix-cache support. No V4 CUDA graph was active.
- Requests used greedy sampling, 128 reported input tokens, 16 output tokens,
  and eight concurrent OpenAI chat completions. Every response's token counts
  were checked.
- The final TorchInferno and vLLM runs used cyclically rotated, distinct prompts
  within and across batches. TorchInferno's fallback chat formatter needs 61
  words for 128 tokens; vLLM's V4 formatter needs 62 for the same reported
  length. This keeps token counts equal without enabling prompt reuse.

TorchInferno loaded TileLang and Marlin artifacts prepared offline for TP4 and
TP8. Its serving, 4K-context smoke, and disaggregated runs all passed with
`CUDA_HOME=/does/not/exist`, so no request or model-loading path compiled CUDA.
The TileLang definitions hash was
`4451234d60b693a68be6cde9ee012fb79d4c2d11fa644e50c8e32874757f3a0e`.
The decode kernels accept the selected 64-value frequency row, so their artifact
keys do not contain maximum context length.

The relevant launch settings were:

```bash
# TorchInferno
TORCHINFERNO_V4_KERNEL_ARTIFACT_DIR=/path/to/artifacts/tilelang \
TVM_FFI_CACHE_DIR=/path/to/artifacts/marlin CUDA_HOME=/does/not/exist \
PYTHONPATH=src python -m torchinferno.openai_server \
  --model /path/to/DeepSeek-V4-Flash --model-kind deepseek-v4 \
  --tensor-parallel-size 8 --max-model-len 256 --max-batch-size 8

# SGLang commit a375e9f3da
python -m sglang.launch_server --model-path /path/to/DeepSeek-V4-Flash \
  --tp-size 8 --moe-runner-backend marlin --disable-radix-cache \
  --disable-cuda-graph --max-running-requests 8

# vLLM 0.20.2rc1.dev107+g2a16ece2d
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve /path/to/DeepSeek-V4-Flash \
  --tensor-parallel-size 8 --max-model-len 256 --max-num-seqs 8 \
  --kv-cache-dtype fp8 --enforce-eager --no-enable-prefix-caching
```

`VLLM_USE_FLASHINFER_SAMPLER=0` selected vLLM's PyTorch-native sampler because
its FlashInfer sampler failed startup warmup with an invalid CUDA resource
handle. This changes sampling, not V4 model kernels.

## OpenAI Results

| Engine/path | Samples | Median wall time | Completion throughput |
| --- | ---: | ---: | ---: |
| TorchInferno before this optimization | 5 | 2.105 s | 60.8 token/s |
| TorchInferno final, distinct prompts | 7 | 1.742 s | 73.47 token/s |
| vLLM final, distinct prompts | 7 | 1.963 s | 65.22 token/s |
| vLLM earlier repeated-prompt run | 5 | 1.825 s | 70.14 token/s |
| SGLang streaming, radix disabled | 5 | 2.023 s | 63.26 token/s |
| SGLang non-streaming, radix disabled | 5 | 2.031 s | 63.04 token/s |

On the distinct-prompt run, TorchInferno is 12.7% faster than vLLM. It is 16.1%
faster than the recorded SGLang streaming control and 4.8% faster than the
stronger earlier vLLM repeated-prompt result. The original SGLang request
payloads were not retained, so this note does not claim they were distinct.
Streaming to non-streaming to streaming transitions were run before the final
TorchInferno measurement and each returned the requested 16 tokens.

The seven TorchInferno wall times were 1.759, 1.741, 1.693, 1.742, 1.799,
1.734, and 1.771 seconds. The seven distinct-prompt vLLM times were 11.548,
1.963, 2.033, 1.816, 1.982, 1.783, and 1.797 seconds. The 11.548-second sample
contains request-time Triton compilation and `mhc_pre_big_fuse_tilelang`
compilation on all ranks. It remains in the measured set.

The earlier repeated-prompt vLLM times were 1.765, 9.927, 1.800, 10.082, and
1.825 seconds. Its two long samples also coincide with request-time TileLang
compilation. The specified median is reported without deleting or replacing
any sample.

## CUDA Profile

An apples-to-apples TP8, batch-8, 128-token prompt, 16-step profile compared the
same production sampler and distinct input rows with all new fusions enabled or
disabled through the environment switches recorded in profile metadata.

| Profile measure | Unfused | Fused | Reduction |
| --- | ---: | ---: | ---: |
| `deepseek_v4_decode_step` CUDA time per step | 215.282 ms | 168.710 ms | 21.6% |
| Profiler self CUDA total | 524.527 ms | 423.709 ms | 19.2% |
| `aten::copy_` calls | 26,268 | 12,828 | 51.2% |
| `aten::mm` calls | 4,416 | 3,040 | 31.2% |

The changes are general decode-phase operations: precompiled RMSNorm, fused Q
self-RMSNorm plus RoPE, forward and inverse RoPE, fused mHC pre-projection plus
square reduction, fused mHC post-mapping, and greedy sampling that gathers one
candidate per vocabulary shard instead of full logits. Each fusion has an
environment opt-out and falls back to the readable torch path. There is no
benchmark-specific prompt or request-length fingerprint; `seqlen == 1` is the
generic one-token decode specialization. No fusion uses a dataset, benchmark
name, or token value.

## Numerical Validation

The final prepared artifacts were compared against the torch reference on H100:

- Fused Q normalization plus RoPE, forward RoPE, and inverse RoPE were exact for
  the tested tensors.
- mHC post had maximum absolute error 0.0078125 and mean absolute error below
  8e-8.
- mHC pre projection had maximum relative error 4.86e-5; its square reduction
  had maximum relative error 1.73e-4.
- A real layer-0 fused-versus-reference mHC output had maximum absolute error
  0.03125 and mean absolute error 2.79e-5.
- Fixed-input full-model decode selected the same top-1 tokens `[5, 144]`.

The Marlin MXFP4 path is not bitwise deterministic because of atomic reduction
order. In a 32-position teacher-forced check, reference-versus-reference and
fused-versus-reference each agreed on top-1 at 25 of 32 positions. All 32
cross-mode greedy tokens and all 32 reference-repeat greedy tokens matched. A
fused repeat differed once at a top-1 margin of 0.1166. This is evidence of the
existing near-tie nondeterminism, not a systematic fused-kernel regression.

## Disaggregated Path

The final TP4 prefill plus TP4 decode run used batch two, distinct rows, a
129-token prompt crossing the C4 and C128 compressor boundaries, and two decode
tokens. It generated tails ending in `[271, 5]` and `[235, 146]`.

The warm heterogeneous-KV handoff moved 61,390,848 bytes in 5.897 ms end to
end, including 2.786 ms of NCCL point-to-point transfer. The measured generation
call took 0.253 s. The implementation is synchronous, so this validates role
isolation, TP4 artifact coverage, and live CUDA KV transfer; it does not claim
an eight-GPU throughput gain over monolithic TP8.

## Integrity Review

An independent adversarial agent reviewed the final diff and profile artifacts.
It found no prompt or request fingerprint, benchmark branch, expected-answer or
logits cache, output cache, harness detection, or runtime compiler invocation.
The `seqlen == 1` branches are ordinary decode-phase specialization. Retained
row-owned KV allocations are reset before reuse and synchronized before graph
release. Profile metadata records source state, device, CUDA and torch versions,
artifact inventories, kernel-definition hash, prompt stride, and effective
fusion switches.

The fused mHC layout is adapted from vLLM's Apache-2.0 implementation and the
precompiled RMSNorm operator is supplied by SGLang; both are recorded in
`THIRD_PARTY_NOTICES.md`. No benchmark-shaped behavior is part of the runtime.
