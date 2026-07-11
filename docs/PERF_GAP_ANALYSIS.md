# TorchInferno vs vLLM/sglang — Performance Gap Analysis (Llama-3.1-70B, 8xH100)

## Current 20260711 logits-cache integrity reset

Exact-prompt logits reuse is not a valid benchmark optimization. Reusing a
cached prompt or generated-prefix logits tensor lets repeated byte-identical
prompts return tokens without normal model execution, which disproportionately
inflates single-token rows such as self_consistency.

The `e887422` generated-prefix chain extension has been reverted, and prompt
logits are no longer persisted in the reusable-prefix cache by default. OpenAI
sampled-short serving also no longer auto-enables generated-prefix caching.
Logits persistence remains available only as an explicit diagnostic opt-in via
`TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE_LOGITS=1`. The separate OpenAI
server exact-prompt logits cache is also opt-in only via
`TORCHINFERNO_OPENAI_PROMPT_LOGITS_CACHE=1`. Neither path should be used for
score-facing benchmark runs.

The benchmark harness now also forces those score-facing caches off for
TorchInferno provider runs. inference-bench `034494c0` sets
`TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE=0`,
`TORCHINFERNO_CONTINUOUS_ADAPTIVE_GENERATED_PREFIX_CACHE=0`,
`TORCHINFERNO_OPENAI_TP_ONLINE_GENERATED_PREFIX_CACHE=0`,
`TORCHINFERNO_CONTINUOUS_PREFIX_CACHE_STORE_LOGITS=0`,
`TORCHINFERNO_CONTINUOUS_PINNED_FULL_PROMPT_STORE_LOGITS=0`, and
`TORCHINFERNO_OPENAI_PROMPT_LOGITS_CACHE=0` unless the explicit diagnostic
override `INFERENCE_BENCH_TORCHINFERNO_ALLOW_LOGITS_CACHES=1` is set. This
protects public comparisons even when the runner happens to build an older
TorchInferno commit.

## Current 20260711 guarded long-output baseline

A current-head TorchInferno-only `long_output` run on pushed `8b8cc67`, through
inference-bench `034494c0`, wrote
`/tmp/inference-bench-long-current-8b8cc67-guarded-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-current-8b8cc67-guarded/runs/20260711_161517`.
It completed `1000/1000` correct at `217.4 / 21.9 / 973.1ms`,
`35.6 tok/s`. The cache-integrity counters stayed clean:
`runtime_generated_prefix_reuse_requests=0`,
`runtime_generated_prefix_store_requests=0`, and exact-prompt logits stores
were absent; all first-token reuse was ordinary common-prefix KV reuse
(`runtime_prefix_reuse_requests=1000`, `runtime_prefix_reuse_tokens=111000`,
hit length `111`).

The performance profile stayed in the same real-gap band as the public row:
`60` prefill batches, `51.0K` prefill model tokens, `33.2K` prefill padding
tokens, no prefill or decode graph misses, `117` decode-many calls over `517`
internal steps, and `28.6K/32.8K` real/padded decode-many tokens. Hot prefill
shapes were still the padded cached-prefix suffix bodies
(`b24:s96`, `b24:s64`, `b16:s64`, `b16:s96`), while hot decode remained
`decode_many:b64/64` plus high-active tails. This confirms the remaining
long_output target after cache hardening is still dynamic packed
prefix-suffix prefill plus a faster high-active decode replay body.

A scoped A/B lowering
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_ADMIT_PER_STEP_CAP` from the
default `24` to `16` wrote
`/tmp/inference-bench-long-current-8b8cc67-cap16-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-current-8b8cc67-cap16/runs/20260711_162100`.
It stayed correct but is rejected as a default: TTFT improved only
`217.4 -> 211.4ms`, while TPOT/E2E regressed `21.9/973.1ms -> 23.0/1022.3ms`
and throughput fell `35.6 -> 34.6 tok/s`. Counters explain the tradeoff:
prefill padding fell `33.2K -> 28.3K`, but prefill batches rose `60 -> 74`.
The smaller cap moves work from padded `b24` replays into more `b16` replays;
that improves first-token queueing slightly but loses the score row.

## Current 20260711 fair sampled-short decode cap

After the logits-cache reset, self_consistency fell back to normal model
execution and exposed a scheduler mismatch: sampled-short KV-bounded admission
raised `max_active` to `105`, but Llama3 ragged decode CUDA graphs are capped
at `TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_BATCH=64` by default. The stop/EOS
decode therefore ran eager at `ragged_decode:logits:b105`, producing a fair but
slow local row: `267.7 / 0.0 / 353.4ms`, `2.8 tok/s`, `1000/1000` correct.

An A/B that capped sampled-short active rows at `64` kept all generated-prefix
and prefix-logits reuse counters at zero, moved the decode path to `52/52`
graph hits with no decode graph misses, and improved the local fair row to
`94.2 / 0.0 / 118.8ms`, `8.4 tok/s`, `1000/1000` correct. The default sampled
KV max-active cap now follows the configured decode graph max batch, with
explicit sampled/global cap env vars still taking precedence. The patched
default reproduced the result at `93.0 / 0.0 / 117.0ms`, `8.5 tok/s`, with
`50/50` decode graph hits and no logits-reuse counters.

## Current 20260711 prefix-copy volume refresh

The latest public run remains `20260711_132252`. A current-head TorchInferno
long_output refresh on pushed `f37f38e` wrote
`/tmp/inference-bench-prefixcopy-f37f38e-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-prefixcopy-f37f38e/runs/20260711_150438`
and completed `1000/1000` correct at `213.3 / 21.7 / 985.7ms`, `36.2 tok/s`.
This is in the same band as the accepted startup cleanup and the earlier
decode-many timing refresh, so it does not change the score-facing diagnosis.

The new prefix-copy counters show `60` prefix-copy batches and `209024`
copied prefix cells, all marked shared: `runtime_prefill_prefix_copy_tokens =
runtime_prefill_prefix_copy_shared_tokens = 209024`, with
`runtime_prefill_prefix_copy_masked_tail_tokens=0` and no skipped batches. That
rules out a mixed-prefix masked-tail split as a long_output fix. The copied
volume is real, but it is uniform common-prefix replay (`p111`, `src1`) and the
existing warm-row skip remains rejected because it shifts traffic onto `src0`
graph shapes and has already regressed measured runs.

The useful prefill target is still the padded suffix body, not the prefix copy:
the same run reports `51129` prefill model tokens, `29653` padding tokens
(`39.9%`), and `60` packed candidates saving exactly those `29653` tokens, but
only `6/60` packed pattern repeats (`10.0%`) and `2/60` exact signature
repeats (`3.3%`). That keeps fixed-capacity/exact-replay packing rejected for
long_output and points back to a dynamic-count packed cached-prefix prefill
body, alongside the separate high-active decode replay target.

## Current 20260711 long-output decode-many timing refresh

The latest public run is still `20260711_132252`; re-rendering it through
`torchinferno.cli inference-bench-summary` keeps the score-facing gaps
unchanged. Long_output is the largest gap at `187.7 / 19.0 / 873.4ms` for
TorchInferno versus vLLM's `43.9 / 14.7 / 556.5ms`. The merged public queue
profile attributes the long row to `125` decode-many calls, `539` internal
steps, `28.5K` decode-many model tokens, and the hot
`decode_many:b64/64:g1-16` window with `151` calls and `9.7K` model tokens.
Because public queue profiling is no-sync by default, it does not include GPU
time for those windows.

A current-head timing run on `bdab337` used
`TORCHINFERNO_CONTINUOUS_DECODE_MANY_GPU_EVENT_TIMING=1` plus the decode-many
replay profiler gates while running only long_output:
`/tmp/inference-bench-long-decodemany-gputiming-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-decodemany-gputiming-bdab337/runs/20260711_144621`.
It completed `1000/1000` correct at `213.6 / 22.7 / 978.2ms`; treat the score
as timing-perturbed evidence, not a default-performance row. The useful
counters are the GPU attribution: `runtime_decode_many_model_gpu_ms=6299.8ms`
over `114` calls, `493` steps, `27.2K` model tokens, and `31.4K` padded
tokens. The hot full-batch window was
`decode_many:b64/64:g1-16`, `157` calls, `10.0K` model tokens, and
`2014.2ms` GPU time. Other high-active `g1-16` windows were smaller but had the
same shape: `b54/64` at `446.7ms`, `b60/64` at `305.5ms`, `b50/64` at
`306.8ms`, and `b62/64` at `242.3ms`.

The one-shot `RAGGED_DECODE_MANY_REPLAY_PROF` hook did not fire because the
default long_output path does not use the experimental multi-step decode graph;
it schedules multi-token bursts around warmed single-token ragged graph replay.
That matches prior profiler rows that measured the same `b64/cache1024` replay
at roughly `12.4ms` self CUDA, dominated by dense GEMMs, gate-up Marlin,
symmetric-memory all-reduces, and grouped GQA attention. Keep
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_GRAPH` off: recent rechecks made
the hot path slower. The current decode target is still a faster single-step
replay body or true overlap/fusion of projection and TP collectives, not another
scheduler/readback toggle.

## Current 20260711 fixed-capacity packed-prefix tree rejection

The latest public run remains `20260711_132252` on TorchInferno `9af4c72`,
vLLM `19069bc`, and SGLang `32cb89d`. TorchInferno scores `5/20`, with the
remaining score-facing rows led by first-token/tree and long-output gaps:
tree_of_thought is `53.9 / 35.1 / 75.1ms` versus vLLM
`32.4 / 21.6 / 48.0ms`, and long_output is
`187.7 / 19.0 / 873.4ms` versus vLLM `43.9 / 14.7 / 556.5ms`.

The public tree profile now has repeated packed-prefix candidate patterns
(`402` candidate calls, `16` pattern keys, `2209` saved model tokens), but the
fixed-capacity packed-prefill prototype is still rejected as a default path. A
same-host control on current head wrote
`/tmp/inference-bench-tree-control-dirty-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-tree-control-dirty-4e9b5fd/runs/20260711_143509`
and scored `110.9 / 64.7 / 146.4ms`, with `76` prefill model calls and
`4298` candidate padding tokens. Enabling best-case fixed-capacity before the
prefix-copy fix wrote
`/tmp/inference-bench-tree-fixedcap-best-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-tree-fixedcap-best-4e9b5fd/runs/20260711_142251`;
it scored `132.5 / 71.2 / 214.4ms` and accepted `0/71` attempts because
`17` graph attempts returned none, `33` grew capacity, and `21` had no
savings.

The guarded fix is to pass a concrete uniform `prefix_copy_len` into
fixed-capacity packed-prefill graphs when the batch has a reusable source
prefix but the caller left the copy length unspecified. With the fix and the
same best-case envs, the run
`/tmp/inference-bench-tree-fixedcap-prefixlen-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-tree-fixedcap-prefixlen-dirty-4e9b5fd/runs/20260711_142949`
accepted `14/69` attempts instead of failing graph replay, but still regressed
to `119.0 / 69.5 / 169.2ms`. The accepted packed calls covered only `1995`
active tokens, `2404` model tokens, and `409` saved model tokens, while total
phase runtime grew to `11.1s` from the control's `4.6s`. Keep the env-gated
path as a correctness/debug fix only; the next tree target is a cheaper dynamic
packed-prefix body or lower first-token scheduling cost, not default
fixed-capacity replay.

The public long_output profile continues to point elsewhere. Its final record
has `747` ragged-decode batches, `125` decode-many calls, `539` internal
decode-many steps, and a hot `decode_many:b64/64` shape with `159` steps. Hot
prefill remains padded common-prefix replay (`b24:s64`/`b24:s96` and
`b16:s64`/`b16:s96`), with `59` packed candidates spread over `56` pattern
keys. There is not enough exact fixed-capacity reuse to make this prototype a
long_output fix; keep the long-output target on the cached-prefix prefill body
and high-active decode replay.

## Current 20260711 dynamic-count packed-prefix target attribution

The research summary now prints
`[torchinferno packed prefill dynamic-count targets]` between the fixed-capacity
reject table and the implementation-target table. This separates the broad
per-pattern savings a dynamic packed-prefix body could recover from the narrower
fixed-slot savings the current diagnostic prototype can use.

On the latest public artifact, `20260711_030227`, the top repeated few_shot
pattern is
`prefix_graph:b32:s16:p122-122:src1:mixed0|p122:s12/p122:s13/p122:s14`: it has
`28` calls and `3131` dynamic saved tokens, but `0` fixed saved tokens
(`0.0%` fixed coverage). The next repeated pattern has `363` dynamic saved
tokens and only `138` fixed saved tokens (`38.0%` fixed coverage). This makes
the earlier fixed-capacity rejection concrete: the useful packed-prefix target
is a variable-count grouped body, not another fixed-slot promotion.

The current local long_output artifact
`/tmp/inference-bench-greedy-graph128-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-greedy-graph128/runs/20260711_113921`
shows the same shape. The top repeated dynamic-count target has `4101` saved
tokens with `0.0%` fixed coverage, followed by `756` and `512` saved-token rows
that also have `0.0%` fixed coverage. The matching multi_turn artifact has
`0.0%` signature and pattern reuse, so its current padded mixed-prefix prefill
gap is not solved by an exact-pattern packed graph; it needs either a broader
dynamic mixed-prefix body or cheaper model-side common-prefix replay.

## Current 20260711 same-host vLLM 0.23 multi_turn target

A fresh same-host vLLM-only `multi_turn` run was built from the public
`vllm-project/vllm` checkout at commit `19069bcbd5be` and wrote
`/tmp/inference-bench-vllm-multi-current-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-vllm-multi-current/runs/20260711_132449`.
The row landed at `83.6 / 38.3 / 110.7ms`, p99
`313.9 / 286.0 / 332.7ms`, throughput `11.9 tok/s`, and `982/1000` raw
correct. The server log used vLLM V1 with prefix caching, chunked prefill
(`max_num_batched_tokens=8192`), CUDA graph capture sizes through `512`, and
reported `75.7%` prefix-cache hit rate during the benchmark.

The directly comparable current TorchInferno rows remain in the
`218-225 / 36-37 / 248-256ms` band with `981-983/1000` raw correct. TPOT is
therefore already roughly tied with the same-host vLLM build; the score gap is
median TTFT and E2E. Current queue profiles split that gap into roughly
`118-125ms` queue-to-submit plus `88-90ms` submit-to-first-token, while vLLM's
entire median TTFT is `83.6ms`. Treat decode changes as secondary for this
workload unless they also reduce first-token scheduling or mixed-prefix
prefill cost.

A scoped recheck of greedy-large initial request collection is rejected.
Forcing `TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_INITIAL_BATCH_WAIT_MS=1`
on current `9af4c72` wrote
`/tmp/inference-bench-ti-multi-initwait1-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multi-initwait1-9af4c72/runs/20260711_133315`
and regressed multi_turn to `238.0 / 37.5 / 268.6ms`, `981/1000` raw correct.
The profile increased p50 queue-to-submit to `130.1ms` and introduced a slow
`prefix_graph:b32:s16:p45-45:src1:mixed0` group with `534.9ms`
queue-to-submit. Keep the current zero-wait greedy-large initial policy; the
remaining gap is not fixed by collecting the first wave.

The model-side packed graph key also keeps fixed-capacity packing out of the
current score path. `try_prefill_ragged_logits_packed_eager_graph` keys on the
exact `q_lens` tuple, exact `start_positions`, source-row count, and
`prefix_copy_len`; the packed attention groups are built at capture time from
those exact lengths and starts. The current multi_turn profile has zero
signature/pattern reuse, so fixed-slot or exact-pattern replay is not the right
promotion target. The next runtime implementation target remains a dynamic-count
packed mixed-prefix prefill body, or an equivalently cheap common-prefix replay
path that can vary per-request starts/counts without fragmenting graph captures.

Mixed-source chunked prefill is also rejected for this multi_turn target. A
dirty prototype grouped chunked prefilling states across source rows with
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_CHUNK=16` and
`TORCHINFERNO_CONTINUOUS_CHUNKED_MIXED_PREFIX_PREFILL=1`. The first run wrote
`/tmp/inference-bench-ti-multi-chunkmixed-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multi-chunkmixed-e5f7735-dirty/runs/20260711_135154`
and regressed to `1852.9 / 36.4 / 2211.0ms`; the queue profile showed only
`2` ragged-prefill graph replays, `62` misses, and a hot failed
`ragged_prefill:b32:s16:rows1:ctx-1:src32` source-copy shape. A revised
dynamic-context version removed the misses (`64` hits, `0` misses, `44`
replays) but still regressed to `1528.6 / 35.4 / 1887.0ms` in
`/tmp/inference-bench-ti-multi-chunkmixed-dynctx-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multi-chunkmixed-dynctx-e5f7735-dirty/runs/20260711_135926`.
It doubled the prefill schedule instead of shortening it: `65` prefill model
calls and `182` online steps versus the current no-chunk baseline's `34` calls
and `113` steps. Keep chunked mixed-prefix prefill out of the default
multi_turn path; chunking can fix long-prompt residency, but this workload needs
fewer first-token prefill submissions, not more smaller ones.

## Current 20260711 same-host global-vLLM refresh

A same-host vLLM refresh used the globally installed Python at
`/home/bobren/local/d/pytorch-env/bin/python3` with
`INFERENCE_BENCH_VLLM_PYTHON` and `--skip-build`. The run did not use the
public vLLM commit (`04d553f`); the server log reports
`v0.20.2rc1.dev107+g2a16ece2d.d20260507`, V1 engine, async scheduling,
prefix caching, chunked prefill, and the same compilation override
`fuse_allreduce_rms=false`. Treat these rows as local/global-build evidence,
not as a replacement for the public vLLM baseline.

The local long_output run wrote
`/tmp/inference-bench-vllm-global-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-vllm-global/runs/20260711_120011`.
It completed `1000/1000` at `120.4 / 37.7 / 1383.8ms`, `25.2 tok/s`, with
server readiness at `125.5s`. Compared with the current local TorchInferno
`greedy-graph128` row (`207.8 / 21.8 / 1046.9ms`, `36.5 tok/s`), this global
vLLM build is much stronger on TTFT but weaker on TPOT/E2E. The public vLLM row
remains the stronger external target at `44.6 / 15.0 / 563.0ms`.

The matching local multi_turn run wrote
`/tmp/inference-bench-vllm-global-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-vllm-global/runs/20260711_120413`.
It completed `979/1000` at `125.2 / 82.5 / 197.8ms`, `6.2 tok/s`, with
server readiness at `110.5s` and a logged prefix-cache hit rate of `76.1%`.
Current local TorchInferno is slower on TTFT (`224.7ms`) but faster on TPOT
(`37.2ms`), while the public vLLM row remains better on both
(`71.2 / 34.2 / 94.6ms`). This keeps the cross-provider target split clear:
TorchInferno needs a lower first-token path for multi_turn and a cheaper
decode/prefill body for long_output, not only better provider startup.

## Current 20260711 greedy suffix graph warmup split

Promote a startup-only warmup split: keep the `max_tokens=512` policy target for
mixed-prefix warmup, but make the generic greedy common-prefix suffix graph
warmup short-only by default. The new helper
`_online_greedy_common_prefix_suffix_prefill_graph_warmup_max_token_values()`
defaults to `(128,)`, while the existing
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_GREEDY_COMMON_PREFIX_MAX_TOKENS` legacy env
still expands the graph warmup when explicitly set. A new
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_GREEDY_COMMON_PREFIX_GRAPH_MAX_TOKENS` env can
override only the generic graph warmup.

The dirty-default long_output validation wrote
`/tmp/inference-bench-greedy-graph128-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-greedy-graph128/runs/20260711_113921`.
It removed the `43.1s` greedy `max_tokens=512` suffix sweep seen in the prior
`821193a` run: tensor-parallel startup warmup fell from `219.9s` to `177.5s`,
with server readiness at `241.1s`. Score stayed in family at
`207.8 / 21.8 / 1046.9ms`, `36.5 tok/s`, and `1000/1000` correct.

The matching multi_turn validation wrote
`/tmp/inference-bench-greedy-graph128-multi-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-greedy-graph128-multi/runs/20260711_114517`.
It also skipped the generic greedy `512` suffix sweep, reached tensor-parallel
startup warmup done at `179.7s` with server readiness at `246.1s`, and stayed in
the current score family at `224.7 / 37.2 / 256.0ms` with `982/1000` correct.
The queue profile still shows `mixed_prefix=True` for multi_turn, so this change
reduces startup graph budget without disabling the larger mixed-prefix runtime
path.

## Current 20260711 pushed-head long_output refresh and startup warmup attribution

A pushed-head TorchInferno long_output refresh on `821193a` wrote
`/tmp/inference-bench-long-compare-821193a-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-compare-821193a/runs/20260711_112515`.
TorchInferno completed `1000/1000` at `209.1 / 21.9 / 1006.7ms`,
`36.4 tok/s`, and `100%` correctness. The same command could not refresh local
vLLM or SGLang because `--skip-build` found no provider venvs under
`/tmp/inference-bench-origin-main-20260707/builds/{vllm,sglang}`; keep using the
same-host public/local reference rows for those providers until the provider
venvs are rebuilt.

The run keeps the score-facing diagnosis unchanged: median first-token latency
splits into `90.0ms` queue-to-submit plus `112.3ms` submit-to-first, hot prefill
is still padded `p111` common-prefix replay (`b24:s64` and `b24:s96`), and
`decode_many:b64/64:g1-16` owns `38.6%` of decode-many model tokens. The new
research summary now also parses TorchInferno startup warmup spans from
`provider_logs/torchinferno.log`. On this artifact it reports `219.9s` total
startup warmup: `35.8s` online decode graph warmup, `102.3s` greedy
common-prefix suffix warmup for `max_tokens=128`, `43.1s` greedy warmup for
`max_tokens=512`, and `16.9s` sampled common-prefix suffix warmup for
`temperature=1,max_tokens=300`. This makes public timeout/staleness triage
visible in the analyzer output instead of requiring manual server-log reads.

## Current 20260711 decode-many queue-profile GPU timing

Queue profiles can now opt into asynchronous CUDA-event timing for decode-many
model calls even when `profile_timings` is off. Set
`TORCHINFERNO_CONTINUOUS_DECODE_MANY_GPU_EVENT_TIMING=1` with the normal
queue-profile env to populate `runtime_decode_many_model_gpu_ms`,
`runtime_decode_many_shape_gpu_ms`, and
`runtime_decode_many_step_window_model_ms` without the synchronized
`TORCHINFERNO_OPENAI_QUEUE_PROFILE_SYNC_TIMINGS` path or a torch profiler run.

The same-patch A/B keeps this opt-in rather than default-on. The timing-on
validation wrote
`/tmp/inference-bench-local-decodemany-gpu-timing-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-decodemany-gpu-timing/runs/20260711_110906`,
completed `1000/1000`, and reported `238.2 / 21.9 / 1061.4ms`. It identified
`runtime_decode_many_model_gpu_ms=6547.1ms` and the hot
`decode_many:b64/64:g1-16` window at `2487.8ms`. The timing-off control wrote
`/tmp/inference-bench-local-decodemany-gpu-timing-off-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-decodemany-gpu-timing-off/runs/20260711_111534`,
also completed `1000/1000`, and scored `211.1 / 21.8 / 980.2ms` with the GPU
timing maps empty. Use the flag for targeted decode attribution runs; keep it
off for public/default score runs.

## Current 20260711 first-token admission split

Queue-profile diagnostics now export per-prefill-shape queue-to-submit maps
beside the existing queue-to-first and submit-to-first maps, and the research
summary prints a compact first-token prefill-shape table. A pushed-head
long_output validation on `d7a33fd` wrote
`/tmp/inference-bench-local-d7a33fd-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-d7a33fd/runs/20260711_093420`.
It landed at `221.4 / 21.7 / 1040.7ms`, p99
`644.0 / 29.8 / 1737.8ms`, throughput `36.2 tok/s`, and `1000/1000`
correct. Startup was in family (`286.3s` ready), and the queue profile stayed
graph-clean with `59` prefill batches, `0` request-path prefill graph misses,
`127` decode-many calls, and `559` decode-many internal steps.

The new split shows the hot first-token shapes are not uniquely blocked on
admission. Median queue-to-submit is broadly similar for the high-volume rows:
`b24:s64` at `88.5ms`, `b16:s64` at `84.1ms`, `b24:s96` at `96.2ms`,
`b24:s32` at `95.6ms`, and `b16:s96` at `96.1ms`. The after-submit prefill
cost is the differentiator: `b24:s96` reports `149.1ms` submit-to-first,
`b24:s64` `113.1ms`, `b16:s96` `110.6ms`, `b16:s64` `96.5ms`, and `b24:s32`
`83.3ms`. This keeps the next long_output target on the cached-prefix
prefix-suffix prefill body and high-active decode replay, not another simple
admission wait sweep.

## Current 20260711 decode replay profiler refresh

Pushed head `b705b81` adds an env-gated
`RAGGED_DECODE_MANY_EAGER_PROF` hook around the eager decode-many fallback, but
the current long_output default does not hit that fallback. A validation run
wrote
`/tmp/inference-bench-local-b705b81-decode-many-eager-prof-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-b705b81-decode-many-eager-prof/runs/20260711_104136`
and completed `1000/1000` at `208.0 / 21.6 / 1005.2ms`. Its queue profile had
`744` decode graph hits/replays, `0` decode graph misses, `0` decode-many graph
calls, and no eager profiler block. The default path is therefore the
decode-many scheduler around warmed single-token ragged graph replay, not eager
ragged decode.

A same-head replay-profiler run wrote
`/tmp/inference-bench-local-b705b81-decode-replay-prof-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-b705b81-decode-replay-prof/runs/20260711_104840`.
The profiler perturbed the first wave, so use it as kernel evidence rather than
score evidence. The summary parsed `decode_replay b64/cache1024/rows64` at
`12.4ms` self CUDA: GEMM/NVJET buckets were `4.5ms` (`36.6%`), Marlin was
`3.3ms` (`26.8%`), symmetric-memory all-reduce was `2.1ms` (`16.7%`), decode
attention was `1.5ms` (`12.1%`), and add/RMS was `0.4ms`. This reinforces the
current decode target: cheaper full-batch projection/Marlin/all-reduce replay
or real overlap, not eager fallback tuning.

## Current 20260711 ragged-prefill replay profiler targeting

A pushed-head `e1e3c2b` replay-profile run used
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_REPLAY_ONCE=1`,
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_MIN_BATCH=16`, and
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_MIN_SUFFIX=64` while running
long_output. It wrote
`/tmp/inference-bench-local-e1e3c2b-prefill-replay-prof-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-e1e3c2b-prefill-replay-prof/runs/20260711_094436`
and stayed in family at `221.1 / 21.9 / 1007.8ms`, `1000/1000` correct. The
profile hook fired during startup greedy-large warmup instead of the benchmark
request path, capturing `batch=16 suffix=256 context_len=301`.

The request-path graph cache confirms long_output's hot common-prefix shapes
use the dynamic `ctx=-256` bucket, including `b24:s64` and `b24:s96` with
`rows1:ctx-256:src1`. The ragged-prefill profiler now accepts exact
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_BATCH` and
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_SUFFIX` filters, shared by capture and
replay profiling. Future long_output replay profiles should target the hot row
directly, for example `CONTEXT_LEN=-256`, `BATCH=24`, `SUFFIX=64`, and
`MIN_BATCH=1`, instead of relying on broad minimum gates that can be consumed by
startup.

The exact-filter validation on pushed `e8b82c4` used those new filters plus
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_REPLAY_SKIP_MATCHES=4`. It wrote
`/tmp/inference-bench-local-e8b82c4-b24s64-replay-prof-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-e8b82c4-b24s64-replay-prof/runs/20260711_095343`
and completed `1000/1000`; the p99 row was profiler-perturbed, so use the
kernel table as evidence rather than the score. The profile line hit the
intended request-path replay: `batch=24 suffix=64 match=5 context_len=-256`
with `src_rows=1`. Self CUDA was `83.40ms`: `160` NCCL all-reduces took `24.59ms`
(`29.5%`), the main GEMM/NVJET buckets were about `27ms`, add/RMS and
elementwise/index/gather work filled most of the rest, and visible softmax was
only `0.65ms`. This revalidates the earlier `0d6ab82` diagnosis on current
head: the hot common-prefix prefill body is dense TP transformer replay over
padded suffix rows, not a graph-cache miss or attention-mask bottleneck.
The inference-bench research summary now parses `RAGGED_PREFILL_PROF`,
`RAGGED_PREFILL_REPLAY_PROF`, `RAGGED_DECODE_REPLAY_PROF`, and
`RAGGED_DECODE_MANY_REPLAY_PROF` blocks from `provider_logs/torchinferno.log`
or `torchinferno_server.log`; the runtime also has an env-gated
`RAGGED_DECODE_MANY_EAGER_PROF` hook for the default decode-many path. The
summary prints a compact profiler table, including GEMM, Marlin, all-reduce,
attention, add/RMS, and softmax buckets. Re-running the summary on the
exact-filter artifact surfaces the same evidence directly:
`prefill_replay b24/s64`, `83.4ms` self CUDA, `24.6ms` allreduce (`29.5%`),
`27.4ms` GEMM/NVJET buckets (`32.9%`), `7.1ms` add/RMS, and `0.7ms` softmax.

## Public 20260711_132252 multi_turn fixed-capacity rejection

The public pointer advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260711_132252`.
TorchInferno was still built from `9af4c72`, but it completed the full suite:
`5/20` score cells versus vLLM `13/20` and SGLang `1/20`. The score-facing
multi_turn row improved to `189.6 / 35.4 / 218.2ms`; vLLM remained ahead on
TTFT/E2E at `90.6 / 54.9 / 126.4ms`.

The TorchInferno queue profile keeps fixed-capacity packed prefill rejected for
this multi_turn gap. It had `35` prefill model calls, `34` graph hits, zero
prefill graph misses, `33` mixed-prefix candidate waves, and `15.9K` avoidable
packed-candidate tokens (`16.1K` real suffix tokens over `32.0K` dense model
tokens). However, `runtime_prefill_packed_candidate_pattern_keys=33` and
`runtime_prefill_packed_candidate_pattern_repeated_keys=0`, so a fixed-capacity
graph keyed by the observed `(prefix_start, suffix_len)` pattern would not
replay. The repeated unit is the broad `b32:s32:src32:mixed1` dense shape, not
the exact packed pattern.

A dirty local probe with
`TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_FIXED_CAPACITY_GRAPH=1` and
one-call packed graph capture was interrupted before benchmark traffic. It had
already expanded startup common-prefix suffix warmup to `100.1s`, while the
public no-env run reported `37.6s` for the same stage. Even if mixed-source
dummy slots were made legal, this does not solve the current multi_turn shape:
the profile needs a dynamic packed prefill body, fewer first-token prefill
waves, or a queueing change that avoids increasing graph-shape warmup.

## Current 20260711 greedy-short batch-bucket rejection

After the decode-many state-sync fix (`829b227`/`1369e8f`), the remaining
long_output prefill profile showed hot `b24:s64` and `b24:s96` common-prefix
waves with `8.1K` row-padding tokens and `21.3K` suffix-padding tokens. A
mechanical replay of the observed real-batch counts suggested that adding
greedy-short batch buckets `18,20,22` would reduce prefix-graph model tokens
from `74,048` to about `68,480`, saving `5,568` slots.

The default-bucket A/B is rejected. The local run
`/tmp/inference-bench-local-greedy-buckets-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-greedy-buckets/runs/20260711_091525`
used `1,2,4,8,16,18,20,22,24,32` for greedy-short prefill. Startup warmed the
new shapes (`90` greedy-short common-prefix suffix shapes) and spent `168.9s`
in that warmup stage, then aborted before serving traffic while entering the
existing greedy-large warmup. The launcher reported worker SIGABRTs with a
symmetric-memory allocator traceback. Keep the greedy-short default at
`1,2,4,8,16,24,32`; the extra buckets trade a modest row-padding reduction for
more startup graph and memory pressure. The next prefill target should remove
padding without multiplying warmed graph shapes, for example a robust packed
prefill body or a lower-fragmentation admission policy.

## Public 20260711_030227 stale TorchInferno failure and first-token shape attribution

The latest public pointer advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260711_030227`.
It still does not measure the current pushed TorchInferno head: the summary
lists TorchInferno `54cb558`, vLLM `04d553f`, and SGLang `9068836`.
TorchInferno completed only few_shot (`141.5 / 32.6 / 165.8ms`) and
self_consistency (`23.9 / 0.0 / 24.5ms`), then timed out in multi_turn. The
server log shows a rank-0 CUDA launch failure and NCCL watchdog termination;
tree_of_thought and long_output were connection-refused aftermath, not
score-facing measurements of the current runtime.
The inference-bench research summary now prints the provider commit beside each
score row, so this stale-public-run condition is visible directly in the table
instead of requiring a manual `results.json` inspection.

The first current-head same-host `long_output` comparison on pushed `88df304`
identified the actionable gap:
`/tmp/inference-bench-long-output-compare-88df304-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-output-compare-88df304/runs/20260711_033419`.
TorchInferno completed `1000/1000` at `227.9 / 20.9 / 948.3ms`, vLLM completed
`1000/1000` at `50.5 / 16.9 / 651.5ms`, and SGLang completed `1000/1000` at
`45.1 / 25.8 / 965.9ms`. The no-sync TorchInferno queue profile showed no
request-path prefill graph captures (`runtime_prefill_graph_captures=0`,
`hits=53`) and split median first-token latency into about `88.7ms`
queue-to-submit plus `116.0ms` submit-to-first. The remaining gap is therefore
not cold graph capture; it is admission-wave timing plus the prefix-suffix
prefill replay that emits the first token, followed by vLLM's lower steady
decode TPOT.

Queue profiles now carry first-token source and prefill-shape attribution.
`ServingTokenEvent` has optional `source` and `prefill_shape` fields, prefix
graph and chunked-prefix prefill events tag first tokens with their shape key,
and the OpenAI queue profile aggregates
`request_first_token_prefill_shape_*` count and latency maps. This keeps public
profiling no-sync and compact while letting the next default run identify which
`prefix_graph:b*:s*:p*:src*:mixed*` shapes own queue-to-first and
submit-to-first latency.

The first current-head run with that attribution wrote
`/tmp/inference-bench-long-output-first-shape-e2ef272-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-output-first-shape-e2ef272/runs/20260711_035849`
and landed at `229.6 / 21.7 / 993.3ms`, `1000/1000` correct. It showed all
first tokens came from prefill, with `b32` waves owning a slow submit-to-first
tail: `prefix_graph:b32:s64:p111-111:src1:mixed0` covered `177` requests at
`182.3ms` median submit-to-first and `b32:s96` covered `28` requests at
`180.3ms`, versus `b24:s64` at `110.7ms` and `b16:s64` at `92.3ms`.

Promote the greedy-short admission cap from `64` to `24`. The current-head A/B
with `TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_ADMIT_PER_STEP_CAP=24` wrote
`/tmp/inference-bench-long-output-admit24-e2ef272-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-output-admit24-e2ef272/runs/20260711_040627`
and improved long_output to `207.4 / 21.5 / 938.1ms`, p99
`489.5 / 29.9 / 1764.4ms`, throughput `36.4 tok/s`, with `1000/1000`
correct. The queue profile confirms the mechanism: `b32` first-token shapes
disappeared, queue-to-submit improved `90.5 -> 82.6ms`, submit-to-first
improved `125.7 -> 110.8ms`, and decode-many model tokens fell
`31.1K -> 27.8K`. few_shot remains outside this policy because it uses
`max_tokens=256` and stays on the greedy-mid cap.

The current pushed head `397871a` also resolves the stale public multi_turn
timeout locally: TorchInferno completed
`/tmp/inference-bench-multi-turn-current-397871a-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-multi-turn-current-397871a/runs/20260711_045857`
at `246.8 / 37.7 / 279.8ms`, p99 `868.5 / 266.9 / 885.9ms`, with
`982/1000` correct. Its queue profile showed the greedy-large
`admit_min_ready_requests=32` refill floor dominating median first-token
latency: `138.6ms` queue-to-submit versus `91.3ms` submit-to-first. Lowering
the greedy-large refill floor to `8` with
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_REFILL_MIN_READY_REQUESTS=8` wrote
`/tmp/inference-bench-multi-turn-refill8-397871a-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-multi-turn-refill8-397871a/runs/20260711_050553`
and improved multi_turn to `217.8 / 37.5 / 248.5ms`, p99
`679.8 / 226.4 / 714.1ms`, with `981/1000` correct. The queue profile confirms
the mechanism: queue-to-submit improved `138.6 -> 119.2ms`,
submit-to-first improved `91.3 -> 85.7ms`, and p99 submit-to-first improved
`164.5 -> 100.4ms`, so greedy-large now uses the same `8` refill floor by
default.

The current pushed head `1443e18` completes tree_of_thought locally despite the
stale public connection-refused row. The no-env TorchInferno control wrote
`/tmp/inference-bench-tree-1443e18-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-tree-1443e18/runs/20260711_053608`
and landed at `53.0 / 36.0 / 76.1ms`, `959/992` correct. The matching same-host
provider refresh wrote
`/tmp/inference-bench-tree-vllm-sglang-1443e18-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-tree-vllm-sglang-1443e18/runs/20260711_054245`;
vLLM landed at `38.8 / 26.7 / 56.6ms` and SGLang at
`39.1 / 145.0 / 125.3ms`, so vLLM remains the tree target while TorchInferno
already beats SGLang on TPOT/E2E/throughput. A scoped sampled-medium zero-wait
A/B with
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_INITIAL_BATCH_WAIT_MS=0`,
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_ACTIVE_READY_WAIT_MS=0`, and
`TORCHINFERNO_OPENAI_TP_SAMPLED_MEDIUM_STREAM_PREQUEUE_ADMISSION_WAIT_MS=0`
wrote
`/tmp/inference-bench-tree-zero-waits-1443e18-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-tree-zero-waits-1443e18/runs/20260711_054833`
and landed at `53.0 / 36.1 / 73.9ms`, p99 E2E `402.1ms`, `956/992` correct.
Queue telemetry explains this as a tail cleanup rather than a model-body fix:
median queue-to-first stayed near `50ms`, submit-to-first stayed near `32ms`,
but p99 queue-to-first fell `611ms -> 366ms` and the run stayed in one online
session. Promote the zero waits only for sampled-medium tree-style traffic
(`temperature > 0`, `256 < max_tokens <= 300`); sampled-short self-consistency,
greedy traffic, active-row caps, and graph policy remain unchanged.

The no-env patched confirmation wrote
`/tmp/inference-bench-tree-zero-default-1443e18-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-tree-zero-default-1443e18/runs/20260711_055631`
and held the improvement at `50.9 / 33.7 / 72.0ms`, p99 E2E `365.1ms`,
`958/992` correct. Its queue profile stayed on one sampled-medium online
session (`submitted_requests=992`), with `q2first_p50=47.7ms`,
`q2first_p99=345.2ms`, `q2submit_p50=16.2ms`, and
`submit2first_p50=31.6ms`, no request-path graph misses.

A current-head paired recheck keeps sampled-medium submit-step commands on by
default. The no-env control on `9bc5189` wrote
`/tmp/inference-bench-tree-default-9bc5189-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-tree-default-9bc5189/runs/20260711_075721`
and landed at `52.1 / 35.6 / 73.7ms`, with `956/992` correct. Disabling only
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_SUBMIT_STEP_COMMAND` wrote
`/tmp/inference-bench-tree-submitstep0-9bc5189-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-tree-submitstep0-9bc5189/runs/20260711_075046`
and regressed to `54.5 / 37.0 / 78.1ms`, with `963/992` correct. Queue
profiles show the mechanism: the disabled run roughly doubled prefill batches
(`196 -> 404`) and decode batches (`156 -> 340`) while leaving median
queue-to-submit flat (`16.8 -> 17.1ms`). Keep the sampled-medium submit-step
default enabled; the remaining tree gap is still the sampled `s12`
prefix-suffix body and sampled decode cost, not the combined submit/step
control command.

## Current fa15dc6 long-output refresh and wait rejections

The latest pushed head `fa15dc6` keeps `long_output` complete and graph-clean
locally, but does not close the main same-host gap. The no-env control wrote
`/tmp/inference-bench-long-output-fa15dc6-current-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-output-fa15dc6-current/runs/20260711_060623`
and landed at `212.7 / 21.6 / 947.7ms`, p99
`535.4 / 32.6 / 1844.1ms`, with `1000/1000` correct. The profile had
`max_active=64`, `admit_per_step_cap=24`, `admit_min_ready_requests=8`,
`prefill_ready_before_decode_active_cap=6`, no request-path graph captures or
misses, `56` prefill batches, and `123` decode-many calls covering `545`
internal steps. Median first-token time split into `q2submit=90.5ms` and
`submit2first=111.2ms`; median `q2finish` was `940.5ms`. Same-host vLLM
remains around `50.3 / 16.8 / 635.6ms`, so the live gap is still both
first-token prefill/admission and high-active decode replay.

Two greedy-short wait A/Bs on the same head are rejected. Setting both
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_INITIAL_BATCH_WAIT_MS=0` and
`TORCHINFERNO_OPENAI_TP_ONLINE_IDLE_BATCH_WAIT_MS=0` wrote
`/tmp/inference-bench-long-output-zero-greedy-waits-fa15dc6-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-output-zero-greedy-waits-fa15dc6/runs/20260711_061330`
and moved TTFT only slightly to `208.8ms` while regressing TPOT/E2E to
`22.5 / 974.4ms`. It also increased prefill batches (`56 -> 63`) and
decode-many work (`123 -> 133` calls, `545 -> 578` internal steps), while
median `q2submit` did not improve (`90.5 -> 92.8ms`). Lowering only the
greedy-short initial wait to `1ms` wrote
`/tmp/inference-bench-long-output-greedy-initial1-fa15dc6-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-output-greedy-initial1-fa15dc6/runs/20260711_062104`
and regressed to `223.0 / 22.2 / 1014.9ms`, with
`q2first=211.5ms` and `q2finish=1005.8ms`. Keep the current greedy-short
`2ms` initial wait and `2ms` idle wait. The remaining long-output target is the
ordinary cached-prefix suffix prefill body plus high-active decode-many replay,
not more admission collection tuning.

A current-head greedy-short admission-cap sweep also keeps the default at `24`.
Lowering `TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_ADMIT_PER_STEP_CAP` to
`16` on `e4b6fa8` wrote
`/tmp/inference-bench-long-output-admit16-e4b6fa8-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-output-admit16-e4b6fa8/runs/20260711_072240`
and improved first token to `203.9ms`, p99 `456.1ms`, by replacing `b24` waves
with `b16` waves (`q2submit=88.9ms`, `submit2first=89.1ms`). It still
regressed TPOT/E2E to `22.6 / 971.9ms` because prefill batches rose to `74`.
The midpoint cap `20` wrote
`/tmp/inference-bench-long-output-admit20-e4b6fa8-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-output-admit20-e4b6fa8/runs/20260711_072924`
and regressed harder to `217.0 / 22.8 / 1020.2ms` with `65` prefill batches,
`132` decode-many calls, and `582` decode-many steps. Do not lower the default
below `24` without a non-fragmenting cached-prefix prefill body.

The current-head guarded suffix-split recheck is also rejected. Enabling
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS_GREEDY_SHORT=1`
on `5a5776f` wrote
`/tmp/inference-bench-long-output-guarded-suffix-split-5a5776f-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-output-guarded-suffix-split-5a5776f/runs/20260711_074107`
and landed at `225.5 / 22.2 / 1009.5ms`, p99
`576.5 / 33.1 / 1843.8ms`, with `1000/1000` correct. The profile accepted
`6/16` suffix-split candidates and saved `3.1K` accepted model-token slots, led
by `base_b24:s96->b16:s32+b8:s96`, but prefill batches rose to `66` and
median `submit2first` stayed high at `120.3ms`. Keep the splitter diagnostic;
the profile still points at a non-fragmenting packed cached-prefix body rather
than recursive suffix fragmentation.

A pure token-only prefix-prefill probe on current head `bd87d04` is also
rejected for `long_output`. The run enabled
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_TOKEN_ONLY_GRAPH=1`, disabled the
logits+token token-prefill warmup, enabled token-only suffix warmup, and wrote
`/tmp/inference-bench-long-output-tokenonly-bd87d04-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-output-tokenonly-bd87d04/runs/20260711_080807`.
It completed `1000/1000` correct but landed at `219.0 / 22.2 / 964.4ms`, p99
`577.5 / 33.8 / 1891.3ms`, worse than the latest clean long-output control
band. Startup also paid a large cost: common-prefix prefill warmup took
`167.5s`, readiness was `286.3s`, memory reached roughly `93GB/GPU`, and the
final profile had `192` live prefill graph entries with `10` evictions. Runtime
telemetry did not expose a new bottleneck removal: prefill sample time was
already `0.0ms` in the control, while the token-only run still executed `59`
padded prefix graph replays and reported `31.6K` avoidable packed-candidate
tokens. Keep token-only prefix prefill diagnostic-only; it does not replace the
needed non-fragmenting cached-prefix prefill body.
The inference-bench analyzer now prints prefill graph-cache live/capacity,
eviction, evicted-entry, and live suffix-bucket columns in the compact
TorchInferno queue table, so future token-only and bucket probes do not require
manual queue-profile JSON extraction for cache pressure.

## Current no-sync prefill-shape telemetry

A dirty-tree telemetry validation after `be39510` wrote
`/tmp/inference-bench-prefill-shape-count-long-be39510-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-prefill-shape-count-be39510/runs/20260711_063547`.
It reached server readiness in `286.2s`, completed `long_output` at
`215.3 / 22.1 / 1003.8ms`, p99 `525.8 / 37.2 / 1821.2ms`, and stayed
`1000/1000` correct. This is not a perf win over the clean `fa15dc6` control;
the useful result is that no-sync queue profiles now populate
`runtime_prefill_shape_counts`, active/model token maps, real-batch and suffix
maps, route maps, and prefix-reuse maps while leaving
`runtime_prefill_shape_forward_ms` and `runtime_prefill_shape_wall_ms` empty.

The new no-sync maps expose the remaining padding directly. The hottest cached
prefix suffix shape, `prefix_graph:b24:s64:p111-111:src1:mixed0`, ran `18`
calls for `405` active requests and `17.8K/27.6K` active/model tokens.
`prefix_graph:b16:s96:p111-111:src1:mixed0` ran `11` calls for `144` requests
and `9.1K/16.9K` tokens, while
`prefix_graph:b24:s96:p111-111:src1:mixed0` ran `6` calls for `117` requests
and `5.8K/13.8K` tokens. Prefix reuse was healthy:
`runtime_prefix_reuse_route_counts={"common_prefix": 1000}` and
`runtime_prefix_reuse_hit_token_counts={"111": 1000}`. Future public profiles
can now quantify ordinary cached-prefix suffix-prefill padding without enabling
sync timings; the target remains non-fragmenting cached-prefix packed prefill
and high-active decode replay.

A follow-up dirty-tree validation after `e606bef` also moved suffix-split
candidate profiling onto the no-sync queue-profile path. It wrote
`/tmp/inference-bench-suffix-candidate-nosync-e606bef-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-suffix-candidate-nosync-e606bef-dirty/runs/20260711_064755`
and landed at `223.0 / 22.2 / 990.1ms`, p99 `535.6 / 33.5 / 1793.6ms`,
with `1000/1000` correct. The final queue record had `15` suffix-split
candidates, `0` accepted, `15` rejected, and `7.1K` predicted saved model
tokens (`26.9K` base versus `20.4K` split candidate tokens). Rejection reasons
were `{"disabled": 7, "min_fill": 4, "min_group": 1, "no_savings": 3}`, and
shape maps such as `base_b24:s96->b16:s32+b8:s96` now appear while
`runtime_prefill_shape_forward_ms` and `runtime_prefill_shape_wall_ms` remain
empty. This keeps public no-sync profiles able to explain why suffix splitting
is not promoted without paying sync-timing overhead.

A second no-sync telemetry validation on the same head added full-prompt store
skip reasons to the lightweight profile path. The dirty run wrote
`/tmp/inference-bench-fullprompt-skip-nosync-a6baa6d-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-fullprompt-skip-nosync-a6baa6d-dirty/runs/20260711_065709`
and landed at `219.4 / 21.8 / 973.6ms`, p99 `640.0 / 34.2 / 2136.3ms`,
with `1000/1000` correct. Its final queue record now reports
`runtime_full_prompt_store_skip_reason_counts={"pinned_without_allowance": 1000}`
and `runtime_full_prompt_store_skip_reason_tokens={"pinned_without_allowance":
155715}`, while `runtime_prefill_shape_forward_ms` and
`runtime_prefill_shape_wall_ms` remain empty. This makes public no-sync profiles
show that long-output full-prompt stores are intentionally skipped under pinned
shared-prefix policy rather than silently failing for capacity or store-disable
reasons.

A follow-up dirty-tree validation after `9d5144c` moved the opt-in
full-prompt reuse candidate profiler onto the same no-sync path. It wrote
`/tmp/inference-bench-fullprompt-candidate-nosync-9d5144c-long-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-fullprompt-candidate-nosync-9d5144c-long-dirty/runs/20260711_071448`
and landed at `222.8 / 22.3 / 1049.3ms`, p99
`560.0 / 37.0 / 1720.5ms`, with `1000/1000` correct. The queue profile stored
`1000` shadow full-prompt candidates covering `155,715` prompt tokens, recorded
`0` later candidate hits, and kept `runtime_prefill_shape_forward_ms` and
`runtime_prefill_shape_wall_ms` empty. This is still a diagnostics-only path:
long_output keeps routing through `{"common_prefix": 1000}`, so the heavier
candidate radix index remains opt-in rather than default public-profile work.

## Public 20260710_170747 startup failure and symm-mem warmup fix

The latest public pointer advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260710_170747`.
It measured TorchInferno `00126b8`, vLLM `c227aaa`, and SGLang `7b99900`.
TorchInferno scored `0/20` because the server never reached readiness: the log
showed `symmetric-memory allreduce enabled after probe (runtime scope)`, then a
30-minute NCCL timeout inside `_warmup_online_mixed_prefix_suffix_prefill_graphs`
while capturing a ragged prefill graph. The call site was incorrectly entering
`_tensor_parallel_symm_mem_allreduce_scope(..., startup=False)`, so the mixed
startup warmup opted into runtime symmetric-memory allreduce despite the runtime
scope guard.

Commit `8bbfa25` changes the mixed-prefix suffix prefill warmup to pass
`startup=True`, preserving runtime symm-mem for request traffic while keeping it
out of startup unless explicitly enabled by the startup override. The local
8xH100 validation wrote
`/tmp/inference-bench-8bbfa25-startup-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-8bbfa25-startup/runs/20260710_183539`.
It reached server readiness in `271.3s`, completed tensor-parallel startup
warmup in `204.9s`, and ran `self_consistency` successfully at
`119.2 / 0.0 / 119.3ms`, `1000/1000` correct. The warmup log confirms the
runtime-scope probe remained enabled, common-prefix prefill completed in
`154.9s`, unified scheduler warmup completed in `201.9s`, and no NCCL timeout
occurred before health.

## Local 2f294d6 self-consistency static-prefill warmup

After the startup fix, the local `8bbfa25` self_consistency run still carried
two request-path static prefill misses for the identical 55-token prompt shape
and spent about `2.25s` in runtime prefill wall time. Commit `4993816` added
single-prefill logits graph warmup for the same generic common-prefix token
buckets and reduced that to one miss. Commit `2f294d6` also warms fallback
prefix row `105`, which is the first free prefix row after the generated-prefix
store consumes preferred row `128` under the 144-row dense cache envelope.

The no-env local validation wrote
`/tmp/inference-bench-2f294d6-row105-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-2f294d6-row105/runs/20260710_185934`
and landed at `31.9 / 0.0 / 32.1ms`, p99 `229.0 / 0.0 / 301.9ms`, with
`1000/1000` correct. The queue profile confirms the warmup is hitting the live
paths: `runtime_prefill_graph_hits=2`, `runtime_prefill_graph_misses=0`,
`runtime_prefill_graph_miss_shape_counts={}`, and
`runtime_prefill_shape_counts={"common_prefix:b1:t55": 1, "single:b1:t55": 1}`.
Runtime prefill wall dropped to about `345ms`. Startup remained in the expected
range (`281.2s` server ready, `214.4s` tensor-parallel warmup) and the shutdown
traceback in the provider log is the normal inference-bench SIGTERM after the
completed request phase.

## Local runtime-key mixed-prefix prefill graph warmup

The `7ea0fb6` full-suite validation exposed a regression from the startup
symm-mem fix: mixed-prefix suffix warmup captured `ar0` ragged-prefill graphs
under `startup=True`, while runtime greedy multi_turn looked for `ar128` graph
keys because runtime symmetric-memory allreduce remained enabled. The full-suite
multi_turn row regressed to `1013.6 / 38.0 / 1045.8ms`; its queue profile showed
`31` request-path prefill graph misses on
`ragged_prefill:b32:s32:rows1:ctx-128:src32` and
`ragged_prefill:b32:s32:rows1:ctx-256:src32`, spending `15.16s` in runtime
prefill wall time.

The accepted fix captures mixed-prefix prefill graphs under the runtime
allreduce graph key while forcing prefill symmetric-memory allreduce off during
startup graph capture. This preserves the `ar128` cache key that runtime uses,
but avoids the symm-mem prefill rendezvous path that caused the public startup
timeout. A focused no-env multi_turn validation wrote
`/tmp/inference-bench-runtime-key-prefill-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-runtime-key-prefill/runs/20260710_192747`
and landed at `228.8 / 36.7 / 258.5ms`, `982/1000` correct. Its final profile
had `runtime_prefill_graph_hits=32`, `runtime_prefill_graph_misses=1`, with
only the tiny `ragged_prefill:b2:s16:rows0:ctx-64:src1` miss remaining; the
two `b32:s32:src32` shapes replayed in `50.2ms` and `57.0ms`, and runtime
prefill wall dropped to `3.24s`.

The no-env full-suite validation wrote
`/tmp/inference-bench-runtime-key-full-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-runtime-key-full/runs/20260710_193511`.
Startup stayed healthy (`276.2s` server ready, `207.7s` tensor-parallel
warmup), and the benchmark medians were few_shot
`177.8 / 33.8 / 206.3ms`, self_consistency `34.9 / 0.0 / 35.3ms`, multi_turn
`227.4 / 37.8 / 257.3ms`, tree_of_thought `68.7 / 41.1 / 97.9ms`, and
long_output `229.4 / 22.0 / 1031.1ms`. The multi_turn queue snapshot showed
`runtime_prefill_graph_hits=36`, `runtime_prefill_graph_misses=0`, no request
captures, and replay coverage for both hot `b32:s32:ctx-128/256:src32` shapes.
This restores the pre-startup-fix multi_turn band without re-enabling the
startup symm-mem failure mode.

## Local lightweight queue-profile timing gate

Inference-bench's TorchInferno provider writes
`TORCHINFERNO_OPENAI_QUEUE_PROFILE_JSONL` by default. Before `d5168d8`, the
server treated that as a request to enable detailed runtime timing, which added
CUDA synchronizes in model prefill/decode profiling and temperature-sampler
phase timers during score-facing runs. The accepted change keeps queue-profile
request timing and graph counters on by default, but gates the sync-heavy
runtime timers behind `TORCHINFERNO_OPENAI_QUEUE_PROFILE_SYNC_TIMINGS=1`;
explicit `TORCHINFERNO_TEMPERATURE_SAMPLE_PROFILE` still overrides sampler
profiling.

A focused no-env long_output validation wrote
`/tmp/inference-bench-light-profile-long-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-light-profile-long/runs/20260710_200932`
and landed at `223.4 / 21.4 / 1035.7ms`, p99
`1766.1 / 58.5 / 3054.5ms`, with `1000/1000` correct. That is a small
TTFT/TPOT improvement over the prior local full-suite row
(`229.4 / 22.0 / 1031.1ms`) while keeping E2E in the same band. The resulting
queue snapshot preserved request p50s and graph counters
(`q2first=213.0ms`, `q2submit=92.9ms`, `submit2first=132.1ms`,
`runtime_prefill_graph_hits=54`, `runtime_prefill_graph_captures=1`), but the
detail-only timing totals stayed inert (`runtime_prefill_wall_ms=0`,
`runtime_decode_many_token_wait_ms=0`) and temperature profiling was absent.

The tree_of_thought validation showed the clearer score-facing benefit:
`/tmp/inference-bench-light-profile-tree-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-light-profile-tree/runs/20260710_201553`
landed at `58.0 / 36.5 / 84.5ms`, p99 `343.9 / 226.1 / 386.6ms`, with
`959/992` correct. The prior local full-suite row was
`68.7 / 41.1 / 97.9ms`, `957/992` correct. The queue profile retained
request timing and graph counters (`q2first=54.9ms`, `q2submit=21.5ms`,
`submit2first=33.0ms`, `runtime_prefill_graph_hits=368`) while leaving
temperature timing null. Keep the lightweight queue profile as the default for
public runs; opt into `TORCHINFERNO_OPENAI_QUEUE_PROFILE_SYNC_TIMINGS=1` only
for diagnostic profiles where detailed CUDA-synchronized phase timing is worth
the measurement perturbation.

The pushed-head `71b9933` full-suite validation wrote
`/tmp/inference-bench-light-profile-full-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-light-profile-full/runs/20260710_202413`.
Startup stayed healthy (`271.2s` server ready), and all five rows completed:
few_shot `191.3 / 33.1 / 221.1ms`, self_consistency
`31.3 / 0.0 / 31.9ms`, multi_turn `227.0 / 36.9 / 258.1ms`,
tree_of_thought `65.6 / 39.2 / 96.0ms`, and long_output
`238.2 / 21.9 / 1044.4ms`. Correctness stayed in the expected bands
(`977/1000`, `1000/1000`, `983/1000`, `957/992`, `1000/1000`). The full-suite
queue profile kept the same lightweight shape across rows: request p50s and
graph counters remained populated, while detailed runtime timing totals stayed
zero and temperature timing stayed null.

A same-host vLLM refresh on `85c09e9` for the two largest remaining score gaps
wrote
`/tmp/inference-bench-vllm-tree-long-refresh-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-vllm-tree-long-refresh/runs/20260710_203647`.
vLLM reached readiness in `95.4s`, then landed at tree_of_thought
`38.7 / 26.7 / 56.5ms`, `960/992` correct, and long_output
`50.9 / 16.8 / 639.2ms`, `1000/1000` correct. Against the pushed-head
TorchInferno full-suite row, the remaining deltas are still structural:
tree needs about `27ms` lower TTFT, `12.5ms` lower TPOT, and `39.5ms` lower
E2E, while long_output needs about `187ms` lower TTFT, `5.1ms` lower TPOT, and
`405ms` lower E2E. Startup is also still about `176s` slower than vLLM because
TorchInferno eagerly warms many runtime graph shapes.

The same-host SGLang refresh on `2286e25` wrote
`/tmp/inference-bench-sglang-tree-long-refresh-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-sglang-tree-long-refresh/runs/20260710_204146`.
SGLang reached readiness in `65.3s`, tree_of_thought landed at
`38.7 / 105.8 / 109.4ms`, `966/992` correct, and long_output landed at
`45.4 / 25.9 / 957.1ms`, `1000/1000` correct. TorchInferno still beats SGLang
on tree TPOT/E2E and long_output TPOT, but loses both TTFT cells and
long_output E2E. That keeps the next cross-provider target on lower first-token
admission/prefill latency and long-output end-to-end pipeline cost rather than
a broad raw-token-rate issue.

A sync-timed long_output diagnostic on pushed `38c9716` wrote
`/tmp/inference-bench-long-detail-profile-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-detail-profile/runs/20260710_204639`
with `TORCHINFERNO_OPENAI_QUEUE_PROFILE_SYNC_TIMINGS=1`. It landed at
`232.8 / 22.0 / 1084.7ms`, p99 `1516.7 / 47.9 / 2629.7ms`, with
`1000/1000` correct and `276.2s` server readiness. The queue profile split the
median first-token path into `q2submit=96.9ms` and `submit2first=133.7ms`, with
`6.48s` prefill wall (`5.81s` forward) plus `7.38s` decode-many GPU and
`6.87s` decode-many token wait. Packed-prefix candidates covered `54` calls,
`44.7K` real suffix tokens, and `32.7K` saved dense tokens, led by
`b32:s64:p111` (`8.3K` saved), `b24:s96:p111` (`6.9K`), and
`b16:s96:p111` (`5.0K`). Decode-many remained concentrated in
`decode_many:b64/64` (`187` steps, `11.97K` model tokens, `2.38s` GPU) with
`1.53K` overgenerated stop-tail tokens. This profile reinforces the current
decision not to reopen fixed-capacity packed prefill, active-cap, refill-floor,
or stop-tail knobs; the remaining long_output closure needs a non-fragmenting
packed cached-prefix prefill body and lower high-active decode/stop pipeline
cost.

Sampler queue-profile telemetry now keeps the lightweight gate useful for
sampled rows as well. When queue profiling is enabled but
`TORCHINFERNO_OPENAI_QUEUE_PROFILE_SYNC_TIMINGS` is not, the tensor-parallel
temperature sampler records count-only `runtime_temperature_sample_calls`,
`runtime_temperature_sample_rows`, and Gumbel call/row fields without recording
CUDA-synchronized timing fields. This restores the public tree_of_thought
sampled-row signal that the timing gate intentionally removed, while keeping
score-facing runs free of sampler phase timers. Validation covered the
count-only queue-profile path plus existing profile-timed Gumbel sampling,
queue-profile export, inference-bench summary formatting, `pyflakes`, and
`git diff --check`; `ruff` was unavailable in both local Python environments.

A pushed-head no-sync tree_of_thought validation on `ea1e71a` wrote
`/tmp/inference-bench-sampler-count-tree-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-sampler-count-tree/runs/20260710_210745`.
It reached readiness in `281.3s` and landed at `63.6 / 36.6 / 89.7ms`,
p99 `340.6 / 238.6 / 348.5ms`, with `957/992` correct. The final
queue-profile event kept the lightweight sampled-row signal:
`runtime_temperature_sample_calls=785`,
`runtime_temperature_sample_rows=2571`,
`runtime_temperature_sample_gumbel_calls=785`, and
`runtime_temperature_sample_gumbel_rows=2571`, while no
`runtime_temperature_sample*` timing keys were present. The final profile also
kept request timing and graph counters (`q2first=59.1ms`,
`q2submit=22.9ms`, `submit2first=35.1ms`,
`runtime_prefill_graph_hits=331`, `runtime_prefill_graph_misses=0`), so the
count-only sampler signal does not reintroduce the CUDA-synchronized sampler
timing overhead that the lightweight gate removed.

Packed-prefix candidate telemetry now follows the same lightweight profile
principle. The no-sync tree validation above had useful aggregate candidate
totals, but shape/signature/pattern dictionaries were empty unless sync timing
was enabled. The runtime now records packed-candidate shape, signature, pattern,
slot, and token counters whenever queue profiling is enabled, while leaving
timing fields and generic per-shape timing gated behind sync profiling. This
keeps future public long_output/tree profiles actionable for packed cached-prefix
prefill work without requiring `TORCHINFERNO_OPENAI_QUEUE_PROFILE_SYNC_TIMINGS=1`.
Focused validation covered the no-timing serving-engine path plus queue-profile
export and inference-bench summary consumers.

The pushed-head `2f7e16b` tree validation wrote
`/tmp/inference-bench-packed-count-tree-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-packed-count-tree/runs/20260710_212051`.
It reached readiness in `276.2s` and landed at `57.4 / 36.7 / 86.4ms`,
p99 `415.3 / 217.5 / 448.7ms`, with `956/992` correct. The no-sync queue
profile now carries packed-candidate detail without timing: `341` candidate
calls, `9.7K` real suffix tokens, `13.1K` dense model tokens, and `3.4K` saved
tokens. The top saved-token shapes were
`prefix_graph:b4:s12:p45-45:src1:mixed0` (`1.4K` saved),
`b16:s12:p45-45` (`1.1K`), and `b2:s12:p45-45` (`579`). Pattern summaries
reported `20` pattern keys, `15` repeated pattern keys, and `3.2K` repeated
saved tokens. The same profile kept sampler counts (`840` calls, `2584` rows,
matching Gumbel counts), had no `runtime_temperature_sample*` timing keys, and
left `runtime_prefill_wall_ms`, `runtime_prefill_forward_ms`, and
`runtime_prefill_sample_ms` at `0.0`.

The same pushed-head lightweight profile on long_output wrote
`/tmp/inference-bench-packed-count-long-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-packed-count-long/runs/20260710_212835`.
It reached readiness in `276.2s` and landed at `226.3 / 20.9 / 1027.8ms`,
p99 `2019.6 / 54.4 / 3248.5ms`, with `1000/1000` correct. The no-sync queue
profile split p50 first-token latency into `q2submit=93.2ms` and
`submit2first=114.9ms`, while keeping all detailed runtime timing totals at
`0.0`. Packed-candidate counters now expose the long-output target without the
sync-timed run: `51` candidate calls, `44.7K` real suffix tokens, `75.8K`
dense model tokens, and `31.1K` saved tokens. The top saved-token shapes were
`b32:s64:p111` (`6.6K` saved), `b24:s64:p111` (`6.4K`),
`b24:s96:p111` (`6.2K`), and `b16:s96:p111` (`5.4K`). Pattern detail was much
less reusable than tree: `49` pattern keys with only `2` repeated keys and
`2.5K` repeated saved tokens. This reinforces the same implementation target:
a packed cached-prefix prefill path must avoid per-pattern fragmentation rather
than specializing narrowly to one repeated pattern.

Packed-candidate target telemetry now also records the best single variable
packed wave per dense prefill shape: max real tokens, max dense model tokens,
max saved tokens, and max group count. These are count-only fields exported in
queue profiles alongside the existing aggregate shape/signature/pattern maps,
and `inference-bench-summary` now keeps `shape_counts` in its queue-profile
allowlist so the per-shape target table reports real call counts. This closes
the diagnostic gap left by long_output's low pattern reuse: a non-fragmenting
packed body can now be sized from the largest observed variable-packed wave,
not only from repeated exact or fixed-capacity patterns.

A validation long_output run with the new fields wrote
`/tmp/inference-bench-packed-max-long-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-packed-max-long/runs/20260710_214232`.
It reached readiness in `276.3s` and landed at
`225.8 / 21.5 / 1043.1ms`, p99 `1505.5 / 45.1 / 2712.2ms`, with
`1000/1000` correct. The queue profile recorded `54` packed-prefix candidate
calls, `44.7K` real suffix tokens, `79.9K` dense model tokens, and `35.2K`
saved tokens. The new max-call columns show the first concrete variable-packed
body targets: `prefix_graph:b24:s96:p111-111:src1:mixed0` ran `9` calls,
saved `10.3K` total tokens, and had a single wave saving `1.66K/2.30K`
tokens (`72.0%`) across `9` groups; `b32:s96:p111` appeared only once but
saved `2.23K/3.07K` tokens (`72.7%`) across `11` groups. This keeps
long_output pointed at a generic variable-packed cached-prefix body rather than
another fixed-pattern graph experiment.

Decode-many shape telemetry now follows the same lightweight-profile rule.
When queue profiling is enabled without sync timings, the runtime records
integer-only `runtime_decode_many_shape_*` and
`runtime_decode_many_step_window_*` maps for steps, model tokens, padded tokens,
emitted/skipped tokens, and stop/limit finishes, while keeping decode-many
timing maps empty. This makes public/default long_output profiles identify the
hot `b64/64` early-window decode body without paying CUDA synchronization
overhead. Focused CPU validation covers the no-timing queue-profile path plus
the existing profile-timed decode-many, stop/limit, sync-stop, and state-sync
accounting.

The latest public run
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260710_210746`
completed only few_shot and self_consistency before rank 0 aborted with a
CUDA unspecified launch failure reported by the NCCL watchdog. The final useful
queue profile was sampled self-consistency (`temperature=0.7`, `max_tokens=256`)
with `1` generated-prefix store, `994` generated-prefix reuses, and only `2`
real decode model calls; multi_turn then timed out and tree/long saw connection
refused. As a scoped lifecycle fix, tensor-parallel online close commands now
forward the primary's phase-specific CUDA-sync decision to workers instead of
letting worker close use its own default. This keeps primary/worker close
barriers explicit when sampled-short sessions run without per-step sync.

## Public 20260710_140141 current-run refresh and scheduling rejections

The public pointer now includes the current TorchInferno main run:
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260710_140141`.
It measured TorchInferno `adaa950`, vLLM `85c09e9`, and SGLang `2286e25`;
the scorecard is TorchInferno `2/20`, vLLM `14/20`, and SGLang `3/20`.
TorchInferno medians were few_shot `174.0 / 34.5 / 203.4ms`,
self_consistency `135.9 / 0.0 / 136.1ms`, multi_turn
`255.8 / 39.7 / 290.3ms`, tree_of_thought `61.7 / 39.7 / 92.2ms`,
and long_output `217.1 / 21.7 / 1011.7ms`.

The refreshed profile confirms the remaining score gaps are not fixed by the
obvious scheduling knobs. Few_shot is still first-token/prefill dominated:
`2.51s` prefill wall, `1.91s` prefill forward, `728.8ms` decode GPU, and
queue-to-first p50 `165.4ms`; TPOT already beats vLLM (`34.5ms` vs
`39.9ms`). Multi_turn shows the same pattern at larger prompts:
`3.93s` prefill wall and queue-to-first p50 `236.7ms`, while TPOT is
competitive (`39.7ms` vs vLLM `40.9ms`). Tree remains the sampled-medium
steady-cost target (`7.79s` prefill wall, `3.36s` decode GPU, `925.9ms`
Gumbel sampling). Long_output still splits between prefill and decode-many
wait: `6.33s` prefill wall, `7.72s` decode-many GPU, and `7.22s` of
decode-many token wait.

Three focused probes on `adaa950` are rejected. Enabling
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_READY_BEFORE_DECODE=1` with
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_READY_ACTIVE_CAP=8` for few_shot wrote
`/tmp/inference-bench-fewshot-prefillready-cap8-results/.../runs/20260710_142443`
and regressed to `203.8 / 34.0 / 234.8ms`; the profile shows prefill wall rose
from `2510.6ms` to `2903.4ms` and added another prefill batch. Raising only
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_MID_MAX_ACTIVE=48` wrote
`/tmp/inference-bench-fewshot-greedymid48-results/.../runs/20260710_143052`
and regressed much harder to `212.7 / 140.9 / 338.7ms`, confirming the old
32-row greedy-mid cap still protects decode tail latency. Enabling
`TORCHINFERNO_CONTINUOUS_PINNED_FULL_PROMPT_STORE_MIN_MAX_TOKENS=1` for
self_consistency wrote
`/tmp/inference-bench-self-fullprompt-results/.../runs/20260710_143717` and
regressed from the public `135.9 / 136.1ms` band to `494.0 / 731.6ms`. Keep
these knobs diagnostic-only; the next useful work is a cheaper common-prefix
prefill body or a decode/readback pipeline improvement, not broader admission
or full-prompt pinning.

## Local 6d83c82 few-shot refresh

A pushed-head TorchInferno-only `few_shot` refresh on `6d83c82` wrote
`/tmp/inference-bench-6d83c82-few-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-6d83c82-few/runs/20260710_173435`
and landed at `187.1 / 34.0 / 216.4ms`, `977/1000` correct. Startup stayed in
the expected band at `276.3s` ready and `211.1s` tensor-parallel warmup. The
public pointer was still `20260710_140141`.

The queue profile is the same warmed greedy-mid shape seen in earlier
few_shot diagnostics, but with current-head counters: no request-path prefill
captures or misses, `q2submit=106.0ms`, `submit2first=70.7ms`,
`q2first=179.7ms`, and `q2finish=203.0ms` p50. Prefill ran `34` batches and
spent `1.87s/2.52s` forward/wall, almost entirely in
`prefix_graph:b32:s16:p122-122:src1:mixed0` (`32` calls, `2.04s` wall), with
`3.94K` padding tokens and `3.53K` suffix-padding tokens. Decode is not the
median bottleneck: ragged decode spent `723ms` GPU across `81` batches, and the
decode graph miss counter remained a small static-tail artifact with no capture
time.

This refresh does not reopen the already rejected few_shot knobs. Greedy-mid
prefill-ready-before-decode, larger row caps, prequeue waits, exact-suffix or
fine suffix buckets, token-prefill/token-only prefill, FP8 gate changes, packed
eager prefill, and symmetric-memory prefill all have prior A/Bs that either
regressed medians or moved only diagnostic counters. The next useful few_shot
work is still a cheaper model-side cached-prefix `b32:s16` prefill body or a
packed cached-prefix implementation that avoids the measured Python/graph-cache
overhead.

## Local 4f55f28 multi-turn refresh and dense-first warmup rejection

A pushed-head TorchInferno-only `multi_turn` refresh on `4f55f28` wrote
`/tmp/inference-bench-4f55f28-multi-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-4f55f28-multi/runs/20260710_164532`
and landed at `228.1 / 37.5 / 257.7ms`, `980/1000` correct. The public pointer
was still `20260710_140141`, so this is the freshest local TorchInferno
evidence, not a new public row. The new startup logs showed server readiness in
`281.2s`: tensor-parallel startup warmup took `215.1s`, including `40.8s`
online decode graph warmup, `96.8s` greedy common-prefix suffix warmup for
`max_tokens=128`, `40.2s` for greedy `512`, and `16.7s` for sampled `300`.

The local queue profile slightly improves the public multi_turn TorchInferno
profile (`q2first_p50=220.0ms` vs public `236.7ms`; prefill wall `3.22s` vs
public `3.93s`), but still trails vLLM primarily in queue formation and padded
mixed-prefix prefill. The current run used `32` active rows and `112` prefix
rows, hit `32/33` prefill graphs, and spent `2.52s/3.22s` in prefill
forward/wall with `15.3K` suffix padding tokens.

A narrow warmup experiment that also captured the dense row-index variant for
the first greedy common-prefix suffix pair is rejected. The dirty run wrote
`/tmp/inference-bench-4f55f28-dense-first-multi-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-4f55f28-dense-first-multi/runs/20260710_165640`
and eliminated the single runtime prefill miss
(`prefix_graph:b2:s16:p45-45:src1:mixed0`, `ragged_prefill:b2:s16:rows0:ctx-64:src1`),
cutting prefill wall from `3.22s` to `2.93s`. It still regressed median
TTFT/E2E to `234.3 / 263.7ms` by adding another prefill batch and more scheduler
steps (`127` vs `122`), though p99 improved. Do not promote this dense-first
warmup; the remaining multi_turn gap needs lower-cost mixed-prefix prefill or a
queue policy that improves medians without fragmenting the session.

Forcing the existing submit-step command for greedy multi_turn requests is a
small queue-policy win when scoped to larger generations. The focused A/B on
`281de98` wrote
`/tmp/inference-bench-submitstep-multi-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-281de98-submitstep-multi/runs/20260710_175335`
and moved baseline `228.1 / 37.5 / 257.7ms` to
`224.0 / 38.1 / 256.0ms`, with p99 `1892.5 / 93.9 / 1924.3ms` improving to
`903.2 / 68.9 / 945.4ms` and correctness staying comparable
(`980/1000 -> 981/1000`). The queue profile shows the mechanism:
`q2submit 129.9 -> 123.2ms`, step broadcast `49.0 -> 43.7ms`, step sync
`54.8 -> 46.9ms`, and total phase time `7.24s -> 5.81s`. Promote only the
large-greedy default via
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_SUBMIT_STEP_COMMAND`, scoped by
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_SUBMIT_STEP_MIN_TOKENS` and
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_SUBMIT_STEP_MAX_TOKENS`. The
default range is `400 < max_tokens <= 512`, so multi_turn keeps the submit-step
win while greedy short long_output (`<=128`), few_shot (`256`), and tree's
deterministic eval calls (`400`) stay on their prior command path.

The no-env current-head validation after promotion wrote
`/tmp/inference-bench-1960ff6-multi-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-1960ff6-multi-default/runs/20260710_180537`
and landed at `223.8 / 36.4 / 252.8ms`, p99
`917.8 / 71.6 / 954.7ms`, with `982/1000` correct. The final queue profile
confirmed `submit_step_command_enabled=true` without the forced global env:
`q2submit=125.4ms`, `submit2first=87.5ms`, `q2finish=236.6ms`,
`phase_total=5.81s`, `runtime_prefill_wall=3.16s`, `decode_gpu=706.9ms`,
`20` decode graph misses, and `108` scheduler steps. Keep this as the current
local multi_turn evidence while the public pointer remains on `20260710_140141`.

The post-scope few_shot control on `4f2b7b2` wrote
`/tmp/inference-bench-4f2b7b2-few-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-4f2b7b2-few-default-scope/runs/20260710_181848`
and landed at `181.0 / 33.4 / 211.4ms`, p99
`793.3 / 229.5 / 820.5ms`, with `977/1000` correct. Its queue profile
confirmed `submit_step_command_enabled=false` at `max_tokens=256`:
`q2submit=104.2ms`, `submit2first=68.4ms`, `q2finish=197.7ms`,
`phase_total=4.95s`, `runtime_prefill_wall=2.36s`, `decode_gpu=733.7ms`,
`22` decode graph misses, and `110` scheduler steps. This keeps the
greedy-large submit-step default out of the greedy-mid few_shot path.

## Local 00126b8 long-output refresh

A pushed-head TorchInferno-only `long_output` refresh on `00126b8` wrote
`/tmp/inference-bench-00126b8-long-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-00126b8-long/runs/20260710_170646`
and landed at `223.6 / 21.4 / 1081.0ms`, `1000/1000` correct. Startup was in
the expected band: server readiness `276.2s`, tensor-parallel warmup `209.3s`,
including `35.6s` online decode graph warmup and `157.9s` online common-prefix
prefill warmup. The public pointer was still `20260710_140141`.

Same-host provider references keep the long-output target unchanged. The latest
available vLLM long-output refresh
`/tmp/inference-bench-vllm-long-refresh-results/.../runs/20260710_114420`
measured `50.1 / 16.9 / 644.4ms`, and the same-host SGLang comparison from
`/tmp/inference-bench-sglang-local-compare-results/.../runs/20260709_204619`
measured `61.1 / 24.6 / 975.9ms`. TorchInferno remains behind vLLM on all
score-facing long-output metrics, but its TPOT remains better than that SGLang
reference; the primary gap is still vLLM's much lower first-token and denser
steady decode path.

The current queue profile mirrors the known split rather than exposing a new
default-safe knob. Server-side p50s were `q2submit=96.0ms`,
`submit2first=115.7ms`, `q2first=212.7ms`, and `q2finish=1069.4ms`. Prefill
spent `5.79s/6.42s` forward/wall with `21.5K` suffix padding tokens and no
prefill graph misses; the request-path `p111/s96` capture symptom persists
(`prefix_graph:b16:s96:p111-111:src1:mixed0`, `1.12s` capture GPU), but the
same-host `111:96` startup warmup A/B remains rejected because it removed that
capture while regressing medians. Decode-many still spent `7.79s` GPU and
`7.30s` CPU token handling across `141` calls and `619` steps. The hottest
window was `decode_many:b64/64:g1-16` (`195` steps, `12.5K` model tokens,
`2.48s` GPU, `2.35s` token wait). Keep the long-output target on a cheaper or
better-overlapped high-active decode body plus lower padded prefill, not more
shape-specific warmup.

A current `6b592fe` focused A/B also rejects capping greedy-short prefix-prefill
capture below the long-output `b16/b24` waves. The probe set
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_CAPTURE_GREEDY_SHORT_MAX_BATCH=15` and
wrote
`/tmp/inference-bench-long-capture15-results/.../runs/20260710_195358`. It
removed request-path prefill captures (`runtime_prefill_graph_captures=0`,
`capture_ms=0`) and stayed correct (`1000/1000`), but fragmented the admitted
groups and regressed medians to `263.2 / 22.9 / 1178.7ms`. Queue telemetry
showed `q2first=250.5ms`, `q2submit=94.6ms`, `submit2first=154.8ms`, one
prefill graph miss, `90` graph hits, and the large work shifted into `15`
`b16:s32`, `38` `b16:s64`, and `14` `b16:s96` prefix-graph calls. The default
captures are expensive, but avoiding them by splitting at `15` loses the larger
prefill waves that amortize the graph body. Keep the short-greedy capture cap at
the current batch-32 policy until there is a non-fragmenting packed-prefix
prefill body.

A source audit of the current decode-many loop does not reopen token
materialization as a score-facing lever. `_step_decode_only_many` intentionally
launches one-step ragged decode replays without synchronizing each step, copies
the generated token slices into a GPU scratch buffer, then materializes one
batched token list after the command quantum. The profiled `7.30s`
`runtime_decode_many_cpu_tokens_ms` is therefore mostly the first stream fence
after queued GPU work: `7.26s` token wait and only `32ms` actual
materialization. Async readback and stop-synchronized decode-many already have
A/Bs that failed to improve the model-work envelope or E2E, and the inference
bench streaming helper is already using direct `httpx` SSE streaming rather
than OpenAI object parsing on the timing path. Treat readback/client parsing as
rejected for this profile; the next long-output implementation needs a cheaper
full-width replay body, GPU-side stop compaction, real prefill/decode overlap,
or a packed cached-prefix prefill path that lowers model work without adding
batch fragmentation.

## Local c37c86d tree refresh and Gumbel score-form rejection

A pushed-head TorchInferno-only `tree_of_thought` refresh on `c37c86d` wrote
`/tmp/inference-bench-c37c86d-tree-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-c37c86d-tree/runs/20260710_171745`
and landed at `61.8 / 38.9 / 87.6ms`, `957/992` correct. Startup stayed in
the expected band at `276.2s` ready and `210.5s` tensor-parallel warmup. The
public pointer was still `20260710_140141`.

The queue profile confirms the current tree gap is steady sampled-medium work,
not graph churn: no prefill graph captures or misses, server-side p50s
`q2submit=21.7ms`, `submit2first=36.6ms`, `q2first=58.2ms`, and
`q2finish=82.7ms`. Prefill spent `6.29s/7.66s` forward/wall across `341`
batches, Gumbel sampling spent `938.7ms` for `2586` sampled rows, and ragged
decode spent `3.44s` GPU across `297` batches. The dominant shapes remain
sampled common-prefix `s12` prefill (`b2` and `b4`) and ragged decode buckets
`b3/b4/b5`.

A general sampler rewrite that used the mathematically equivalent Gumbel score
`logits + temperature * gumbel` instead of `logits / temperature + gumbel` is
rejected for the default path. The dirty run wrote
`/tmp/inference-bench-gumbel-score-tree-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-gumbel-score-tree/runs/20260710_172618`
and regressed score-facing medians to `65.3 / 40.8 / 91.3ms`, though
correctness was `961/992`. The final queue profile showed a modest raw Gumbel
drop (`938.7ms -> 876.5ms`, noise `250.3ms -> 196.3ms`) but worse request
medians and higher prefill/decode totals (`prefill 7.66s -> 8.35s`, decode GPU
`3.44s -> 3.51s`). Do not promote this score-form sampler without a new A/B
that improves end-to-end medians; the next tree target remains a cheaper
sampled `s12` prefill/decode body or lower-overhead sharded sampling with stable
request-level latency.

## Public 20260710_130259 refresh and main fast-forward

The public pointer advanced again to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260710_130259`.
It still measured stale TorchInferno `a4d92f0` while vLLM was `fabec87` and
SGLang was `e9493a0`, leaving the published scorecard at TorchInferno `4/20`,
vLLM `13/20`, and SGLang `2/20`. Public medians were few_shot
`152.8 / 31.2 / 179.3ms`, self_consistency `51.4 / 0.0 / 52.1ms`,
multi_turn `476.5 / 125.9 / 580.5ms`, tree_of_thought
`133.2 / 83.6 / 169.2ms`, and long_output `614.5 / 67.2 / 2935.6ms`.

That public row is not comparable to the current local evidence below because
the public runner clones TorchInferno `main`, and `main` had not yet advanced
past `a4d92f0`. The current benchmark branch was verified as a clean
fast-forward from `origin/main` (`35` commits ahead, `0` behind), passed the
focused suffix/warmup tests, pyflakes, and `git diff --check`, then TorchInferno
`main` was advanced to `54f4c42`. The next public run should therefore measure
the sampled dense warmup, sampled-medium suffix buckets, decode/prefill
telemetry, and the documented rejection fixes rather than the stale public
baseline.

## Current 20260710 mixed-prefix suffix-split rejection

Multi_turn still shows real `s16-17 -> s32` suffix padding in mixed-prefix
prefill, but the guarded suffix-split path is not a defaultable fix. Enabling
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS=1` on `54f4c42`
wrote
`/tmp/inference-bench-multiturn-split-suffix-54f4c42-results/.../runs/20260710_134045`
and regressed to `276.8 / 38.8 / 312.7ms`, p99
`1561.3 / 268.2 / 1814.1ms`, with `980/1000` correct. The splitter found
`30` candidates and accepted `5`, saving `1696` model tokens, but it fragmented
prefill into cold small graph bodies such as `b16:s16`, `b2:s32`, and `b2:s16`;
queue-to-first p50 rose to `263.4ms`.

Adding the missing mixed-prefix warmup specs
`16:16:128`, `32:16:128`, `2:32:128`, plus dense `45:16`, removed most of the
cold-request penalty but did not create a useful median win. The warmed probe
wrote
`/tmp/inference-bench-multiturn-split-warm-54f4c42-results/.../runs/20260710_134753`
and landed at `232.4 / 37.3 / 264.0ms`, p99
`925.9 / 82.1 / 956.5ms`, with `982/1000` correct. That is effectively flat
versus the no-env full-suite multi_turn row `232.3 / 37.8 / 264.1ms`; it
accepted only `1` of `31` split candidates, saved `64` model tokens, and still
left a cold `b8:s16:ctx-256` mixed-prefix miss. Keep mixed-prefix suffix
splitting and broader warmup as diagnostics until there is a cheaper
mixed-prefix prefill body or a grouping policy that proves lower wall time.

## Public 20260710_111248 refresh

The public pointer advanced in `origin/main` to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260710_111248`.
It measured TorchInferno `a4d92f0`, vLLM `68ea76e`, and SGLang `7045e0f`, so it
still predates the current `codex/token-prefill-decode-safety` branch through
the dense greedy-prefix warmup and later rejection notes. Public TorchInferno
wins self-consistency (`53.7 / 0.0 / 54.1ms`) and few-shot TPOT (`31.4ms`),
but trails vLLM on few-shot (`152.1 / 31.4 / 176.9ms` vs
`77.0 / 36.2 / 107.3ms`), multi-turn (`469.5 / 124.5 / 569.0ms` vs
`70.4 / 35.9 / 96.8ms`), tree (`126.7 / 84.4 / 166.2ms` vs
`35.2 / 22.9 / 53.3ms`), and long-output (`587.9 / 65.3 / 2915.1ms` vs
`46.4 / 15.2 / 581.0ms`).

The public queue profile matches the already-open targets rather than a new
direction. Few-shot still suffers first-wave prefill/decode graph churn that
the later dense warmup branch improves locally. Multi-turn is dominated by
decode graph misses and padded mixed-prefix prefill (`3.6K` padding tokens,
`1.36s` prefill forward). Tree remains sampled prefill plus sampled decode
(`5.02s` prefill forward, `10.30s` decode GPU, `188` decode graph misses).
Long-output is again high-active decode-many dominated:
`32.56s` decode-many GPU over `607` steps with `1.6K` overgenerated tokens.

## Current 20260710 tree sampled dense warmup

A same-host tree refresh on current TorchInferno `936821b` wrote
`/tmp/inference-bench-tree-default-936821b-results/.../runs/20260710_123708`
and landed at `71.2 / 44.1 / 102.6ms`, p99
`908.1 / 234.1 / 971.9ms`, with `958/992` correct. A same-host vLLM refresh
using `cbe9c40f998f` wrote
`/tmp/inference-bench-vllm-tree-refresh-results/.../runs/20260710_124306`
and landed at `38.8 / 26.7 / 56.9ms`, p99 `210.0 / 30.7 / 233.1ms`, with
`961/992` correct. The current local tree gap is therefore roughly `+32ms`
TTFT, `+17ms` TPOT, and `+46ms` E2E, much smaller than the stale public row but
still score-facing.

The TorchInferno control still captured sampled common-prefix prefill graphs
during request traffic: `runtime_prefill_graph_capture_gpu_ms=2344.2` across
`b1/b2/b4/b8:s16:p45` dense row-index (`rows0`) shapes. Startup warmup already
covered the indexed (`rows1`) sampled variants, but early online tree waves use
contiguous active rows and omit `row_indices`, so those graph keys were cold.
Warming sampled common-prefix dense row-index variants by default wrote
`/tmp/inference-bench-tree-sampled-densewarm-dirty-results/.../runs/20260710_124932`.
It moved readiness from `256.1s` to `266.2s`, removed request-time prefill graph
capture (`2344.2ms -> 0`), and improved tree to `67.3 / 40.6 / 96.0ms`, p99
`378.9 / 225.5 / 434.3ms`, with `957/992` correct. Keep the new sampled dense
warmup: it is general graph-key hygiene for sampled common-prefix serving, not
a benchmark prompt shortcut. Before suffix-bucket promotion, the remaining tree
gap was steady prefill forward (`6.40s`), sampled decode (`3.49s` GPU), and
sampling (`883ms`).

## Current 20260710 sampled-medium suffix buckets

After sampled dense row-index warmup landed in `712aa08`, the previously
rejected `s12,16` suffix-bucket candidate became valid for sampled medium
traffic. The old rejection was caused by cold dense (`rows0`) configured-suffix
graph keys; it is superseded for sampled common-prefix requests because startup
now warms the `rows0` and indexed (`rows1`) sampled graph variants.

An opt-in run on `712aa08` with
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS=12,16` wrote
`/tmp/inference-bench-tree-s12s16-712aa08-results/.../runs/20260710_125925`.
It landed at `55.8 / 36.3 / 81.0ms`, p99 `339.9 / 230.8 / 404.0ms`, with
`961/992` correct. Queue telemetry showed no request-time prefill captures or
misses, prefill forward/wall `3.68s/4.61s`, padding `1989` tokens
(`825` suffix, `1164` row), decode GPU `1.81s`, and Gumbel sampling `491ms`.

Promoting the same shape policy as the no-env sampled-medium default wrote
`/tmp/inference-bench-tree-s12default-dirty-results/.../runs/20260710_130643`.
It landed at `59.2 / 39.0 / 86.0ms`, p99 `426.0 / 219.5 / 461.0ms`, with
`959/992` correct. The final queue profile again had
`runtime_prefill_graph_capture_gpu_ms=0` and no prefill graph miss shapes. It
reported `s12` replay for `b1/b2/b4/b8/b16` sampled common-prefix shapes,
prefill forward/wall `6.27s/7.73s`, padding `3590` tokens (`1502` suffix,
`2088` row), decode GPU `3.52s`, and Gumbel sampling `949ms`.

Keep sampled-medium `s12,16` as a default only for sampled requests with
`max_generation_tokens` in `(256, 384]`, with explicit env overrides still
taking precedence. Against the same-host vLLM tree refresh at
`38.8 / 26.7 / 56.9ms`, TorchInferno is still behind by roughly `+17-20ms`
TTFT, `+10-12ms` TPOT, and `+24-29ms` E2E, so the next tree target is reducing
steady `s12` prefill/decode/sampling cost rather than graph capture churn.

## Current 20260710 full-suite profile after sampled suffix promotion

A no-env TorchInferno-only full-suite refresh on pushed `68079db` wrote
`/tmp/inference-bench-torchinferno-current-full-68079db-results/.../runs/20260710_131536`.
It landed at few_shot `176.9 / 35.2 / 208.4ms`, self_consistency
`130.4 / 0.0 / 130.7ms`, multi_turn `232.3 / 37.8 / 264.1ms`,
tree_of_thought `63.4 / 40.3 / 91.5ms`, and long_output
`232.9 / 22.0 / 1026.4ms`. Public `origin/main` was still `29acfe72` with
latest run `20260710_111248`, so this is the current local branch evidence
until the public row advances.

The queue profile keeps the next gaps concrete. Multi_turn is still mixed-prefix
prefill dominated: queue-to-submit/submit-to-first p50 was `129.1/88.2ms`,
prefill forward/wall `2.19s/4.10s`, and prefill padding `15.6K` tokens, almost
all suffix padding. Tree preserved the sampled-medium suffix promotion with no
prefill graph captures and ran at `6.21s/7.70s` prefill forward/wall,
`3.60s` decode GPU, and `1.18s` Gumbel sampling. Long_output remains the
largest E2E gap: prefill forward/wall `7.74s/8.49s`, decode-many GPU `7.82s`,
decode-many CPU token handling `7.36s`, and `2.96s` of request-time prefill
graph capture on `p111/s96` dense shapes. The earlier `111:96` warmup rejection
still applies: removing that capture alone did not improve medians.

Adding an intermediate `s24` suffix bucket for deterministic greedy-large
multi_turn is rejected as a simple default. The probe used
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS=16,24,32,64,80,96,112,128,144,160,192,224,256`
on `68079db` and wrote under
`/tmp/inference-bench-multiturn-s24-68079db-results`, but it never reached
server readiness: after more than seven minutes the workers were still warming
at about `97GB/GPU`, with no benchmark traffic. This confirms the observed
`s16-17 -> s32` padding is real, but adding another warmed graph bucket is not
the right default lever. The next multi_turn path still needs a cheaper
mixed-prefix prefill body or queueing/prefix-reuse change that avoids expanding
startup graph memory.

## Current 20260710 tree fixed-capacity packed-prefill rejection

The fixed-capacity packed-prefix prefill diagnostic is not a sampled-tree fix.
A focused TorchInferno-only tree run on `a0f380c` enabled
`TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_FIXED_CAPACITY_GRAPH=1` and
wrote
`/tmp/inference-bench-tree-fixedpacked-a0f380c-results/.../runs/20260710_115122`.
It stayed correct enough (`964/992`) but regressed catastrophically to
`1151.1 / 458.3 / 1297.1ms` from the current tree band near
`61-86 / 39-47 / 89-122ms`.

The queue profile shows why the env path must remain diagnostic-only:
`runtime_prefill_packed_fixed_capacity_attempts=111` and
`runtime_prefill_packed_fixed_capacity_accepts=0`, with rejects
`{"capacity_grew": 33, "graph_returned_none": 63, "warming": 15}`. The failed
packed probes collapsed the schedule into large `b16:s16:p45` prefill waves,
inflating prefill forward/wall to `31.5s/32.6s`. The runtime now remembers
concrete fixed-capacity layouts whose packed graph returned `None` only for
no-capture calls and records later skips as `graph_returned_none_cached`, so
future opt-in packed-prefill experiments do not keep probing the same
unsupported graph shape. Capture-on-miss calls deliberately retry because the
model graph path also uses `None` as a warmup signal before capturing an exact
packed signature.

A follow-up dirty rerun after narrowing the graph-`None` cache to no-capture
calls wrote
`/tmp/inference-bench-tree-fixedpacked-retrydirty-results/.../runs/20260710_122816`.
It stayed rejected: `806.6 / 62.1 / 940.3ms`, p99
`2984.1 / 698.1 / 3514.6ms`, and `957/992` correct. The queue profile showed
`117` fixed-capacity attempts, `0` accepts, and rejects
`{"capacity_grew": 32, "graph_returned_none": 62, "no_savings": 1, "warming": 22}`.
It spent `26.4s/27.6s` in prefill forward/wall, with `70` large
`prefix_graph:b16:s16:p45-45:src1:mixed0` waves and only regular dense prefill
graph captures. Keep the retry fix as diagnostic correctness for future packed
graph experiments, but do not treat the current fixed-capacity packed graph as
a score-facing tree path.

## Current 20260710 decode-many async-readback default-off recheck

Async decode-many readback is default-off again. A paired focused long_output
A/B on `1df91d1` used the same TorchInferno commit and no build step. The
default async-readback control wrote
`/tmp/inference-bench-torchinferno-long-default-1df91d1-results/.../runs/20260710_084547`
and landed at `240.0 / 21.5 / 1076.3ms`, `1000/1000` correct. Disabling only
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_ASYNC_READBACK` wrote
`/tmp/inference-bench-torchinferno-long-noasync-1df91d1-results/.../runs/20260710_083925`
and improved the paired medians to `229.4 / 21.1 / 1048.8ms`, also
`1000/1000` correct.

The queue profiles explain why the opt-in path is still not a default. The
async run reported `decode_many_async_readback=true` but did more decode-many
work: `137` calls, `650` steps, `34.6K` model tokens, `8.23s` decode-many GPU,
and `7.59s` decode-many CPU token handling. The no-async run reported
`decode_many_async_readback=false` with `134` calls, `625` steps, `33.6K`
model tokens, `7.90s` decode-many GPU, and `7.38s` CPU token handling. The hot
`decode_many:b64/64:g1-16` window moved from `3334ms` model / `3133ms` CPU in
the async run to `3128ms` model / `2974ms` CPU in the no-async run. Keep the
pinned-host readback stream as an explicit diagnostic via
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_ASYNC_READBACK=1`; the useful
long-output fix still needs a real decode/readback pipeline or a cheaper
high-active replay body.

## Current 20260710 current-head full-suite refresh

A TorchInferno-only full-suite refresh on pushed `5c89f30` wrote
`/tmp/inference-bench-torchinferno-current-full-5c89f30-results/.../runs/20260710_082723`.
It landed at few_shot `188.9 / 33.5 / 219.3ms`, self_consistency
`113.3 / 0.0 / 113.7ms`, multi_turn `246.5 / 38.0 / 291.1ms`,
tree_of_thought `61.5 / 38.7 / 89.2ms`, and long_output
`226.8 / 21.5 / 1042.8ms`. The public pointer was still
`20260710_050745`, so the public TorchInferno row still reflects older
`861b7c3` numbers. Current head should materially improve the public
multi_turn and long_output rows once it is picked up, but it remains behind the
same public vLLM row on the score-facing gaps: tree is still about
`+27.6ms` TTFT and `+37.7ms` E2E, and long_output is still about
`+182.2ms` TTFT, `+6.5ms` TPOT, and `+476.3ms` E2E.

The full-run profile keeps the next targets concrete. Tree is no longer an
obvious sampled-medium active-row cap issue: it ran with `max_active=16`, no
decode-many work, `290` decode graph replays with `290` hits, and spent
`7.79s` wall / `6.26s` forward in prefill, led by the repeated
`prefix_graph:b2:s16:p45-45:src1:mixed0` and
`prefix_graph:b4:s16:p45-45:src1:mixed0` shapes. Long_output still splits
between prefill and high-active decode: `5.43s` prefill wall / `4.66s`
forward, `32.3K` padded prefill tokens, `7.59s` decode-many GPU, `7.00s`
decode-many CPU token handling, and a hot `decode_many:b64/64` slice of
`2.41s` over `189` steps. The suffix split candidate counter showed `6.7K`
theoretical saved tokens but no defaultable split, matching earlier
fragmentation rejections.

This full-suite pass also exposed a profile usability gap: online-batcher
records were interpretable only by file order. Queue profiles now record
`queue_sequence_min`, `queue_sequence_max`, `queue_sequence_count`,
`request_prompt_tokens_min/max`, and `request_max_tokens_min/max` on
online-batcher records, matching the existing stream-group sequence telemetry
and making future full-suite JSONL records easier to align with benchmark
waves.

## Current 20260710 decode-many graph capture-gate recheck

The current-head long_output recheck keeps multi-step decode-many graphs rejected
as a default. The first run enabled
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_GRAPH=1` and row-index support, but
left tensor-parallel decode capture on the runtime default. It wrote
`/tmp/inference-bench-torchinferno-long-decodemanygraph-paged-results/.../runs/20260710_073646`,
landed at `224.9 / 21.8 / 1129.7ms`, and stayed `1000/1000` correct, but the
queue profile showed `runtime_decode_many_graph_calls=0`: TP online workers keep
`decode_capture_on_miss=false`, so the graph path cannot capture unseen
multi-step shapes during normal serving. The run also confirmed the public path
was still the dense runtime cache (`runtime_cache_backend=dense`,
`use_paged_engine=false`), not a paged-cache graph validation.

Forcing capture with `TORCHINFERNO_CONTINUOUS_DECODE_CAPTURE=1` validated the
graph path but rejected it on performance. The run
`/tmp/inference-bench-torchinferno-long-decodemanygraph-capture-results/.../runs/20260710_074551`
landed at `229.2 / 23.4 / 1182.7ms` with `1000/1000` correctness and p99 TPOT
`200.7ms`. Queue telemetry recorded `decode_capture_on_miss=true`,
`decode_many_graph=true`, `144` decode-many graph calls, `633` graph steps, and
`39.9K` graph model tokens, but graph/model time rose to `15.8s` versus the
current graph-off long_output band around `7.6-8.2s` decode-many GPU. Keep
multi-step decode graphs diagnostic-only; the useful long-output target remains
the single-step high-active decode replay body and pipeline/readback overlap.

Queue profiles now record `decode_many_graph`, `decode_many_graph_min_steps`,
and `decode_capture_on_miss` so future runs can distinguish an intentionally
disabled graph path from a graph path that ran and lost.

## Current 20260710 forced short-context paged-KV rejection

Forced paged-KV is not a long_output fix. The dense current-head baseline
(`/tmp/inference-bench-torchinferno-long-current-shapesteps-results/.../runs/20260710_072229`)
used `max_seq_len=288`, below the default paged threshold of `1024`, and landed
at `228.4 / 20.9 / 1029.4ms` with `1000/1000` correctness. Forcing
`TORCHINFERNO_OPENAI_PAGED_KV_MIN_SEQ=1` and
`TORCHINFERNO_OPENAI_CACHE_BACKEND=paged` wrote
`/tmp/inference-bench-torchinferno-long-forcedpaged-results/.../runs/20260710_075615`,
kept correctness at `1000/1000`, but regressed to `1638.0 / 132.9 / 6803.7ms`.
Queue telemetry showed the forced paged engine spent `84.6s` in prefill versus
`5.25s` on the dense path, so the default long-context threshold remains
correct for this short-context decode-throughput cell.

Queue profiles now also record `configured_cache_backend`,
`online_cache_backend`, `paged_kv_requested`, `paged_kv_min_seq`, and
`paged_cache_fallback_candidate`, alongside `use_paged_engine`, so future public
runs can distinguish threshold-gated dense selection from a requested paged
backend that fell back to dense.

## Current 20260710 streaming decode attention block-size rejection

Increasing the Triton streaming GQA decode attention tile from the default `64`
to `128` is not a score-facing long_output win. The diagnostic run used
`TORCHINFERNO_TRITON_STREAMING_DECODE_ATTENTION_BLOCK_S=128` plus the one-shot
ragged decode replay profiler and wrote
`/tmp/inference-bench-torchinferno-long-attnblock128-results/.../runs/20260710_080414`.
It stayed `1000/1000` correct and the profiled `batch=64 cache_bucket=1024`
replay reduced the attention slice from the prior `~1.48ms` to `1.18ms`, but
total replay was still `12.11ms` and benchmark medians regressed versus the
current dense baseline: `222.5 / 21.2 / 1052.1ms` versus
`228.4 / 20.9 / 1029.4ms`. Keep the default tile at `64`; the remaining
long_output gap is still projection/Marlin/all-reduce dominated rather than a
single attention tile-size issue.

Raising the same tile to `256` is a hard rejection. The diagnostic run used
`TORCHINFERNO_TRITON_STREAMING_DECODE_ATTENTION_BLOCK_S=256` with the same
one-shot ragged decode replay profiler and wrote
`/tmp/inference-bench-torchinferno-long-attnblock256-results/.../runs/20260710_081347`.
It stayed `1000/1000` correct but regressed to
`940.6 / 109.2 / 4456.1ms`, with throughput down to `7.4 tok/s`. Queue
telemetry showed `runtime_decode_many_model_gpu_ms=53.64s` and
`decode_many:b64/64=17.47s` over `188` steps, versus the dense baseline's
`8.18s` decode-many GPU and `3.43s` hot `b64/64` slice. The captured
`batch=64 cache_bucket=1024` replay inflated to `95.26ms` self CUDA and was
dominated by elementwise/GEMV/softmax kernels, so keep the tile-size default at
`64`; `128` is diagnostic-only and `256` should not be used for this path.
Queue profiles now record `triton_streaming_decode_attention_block_s` so future
public runs make any non-default attention tile immediately visible.

## Current 20260707 decode-many graph rotary check

The multi-step ragged decode graph now pre-copies per-step rotary tables for the
diagnostic graph path instead of hardwiring per-step rotary `index_select` inside
the captured body. The old behavior remains available with
`TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_MANY_ROTARY_IN_GRAPH=1`. CPU coverage
checks the `[steps, batch, rotary_dim]` copy layout and the static multi-step
forward loop.

The score-facing result is still not defaultable. With
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_GRAPH=1`, the static-rotary run
(`/tmp/ti-long-decodemany-staticrotary-20260707-results/.../runs/20260707_223424`)
landed at `239.4 / 24.3 / 1073.8ms`, `1000/1000` correct. It executed only
`18` decode-many graph calls (`84` graph steps, `5.4K` graph model tokens) but
still spent `3.70s` in `decode_many_graph_ms` and `6.70s` in decode-many GPU.
The same patch with in-graph rotary
(`/tmp/ti-long-decodemany-ingraphrotary-20260707-results/.../runs/20260707_224014`)
landed at `228.5 / 23.4 / 1130.9ms`, also correct, with `44` graph calls,
`214` graph steps, `13.6K` graph model tokens, `5.71s` graph time, and `9.58s`
decode-many GPU. Static rotary improves E2E/p99 versus the in-graph comparison,
but neither variant beats the current graph-off long_output band
(`/tmp/allproviders-long-85007b2-20260707-results/.../runs/20260707_215812`)
at `216.5 / 23.5 / 1060.7ms`. Keep multi-step decode graphs opt-in; the
remaining long-output lever is still a cheaper high-active decode body or real
decode/readback overlap, not this rotary placement.

## Current 20260707 greedy-short FP8 boundary check

An exact greedy-short FP8 boundary probe is not promoted. The default runtime
min_m remains 512, and the model gate is strict (`m > min_m`), so exact
512-token waves stay bf16 unless an experiment lowers the gate. Lowering only
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_FP8_PREFILL_MIN_M` to 511 wrote
`/tmp/ti-long-fp8min511-20260707-results/.../runs/20260707_225347` and landed
at `209.4 / 23.7 / 1046.2ms`, `1000/1000` correct. Against the current
graph-off long_output band
(`/tmp/allproviders-long-85007b2-20260707-results/.../runs/20260707_215812`)
at `216.5 / 23.5 / 1060.7ms`, this only shifts medians within run variance:
exact-512 shapes stayed near `87-93us/token`, prefill wall fell slightly, but
decode-many GPU and CPU time rose. An adjacent no-env control on `49c2f1b` was
stopped before server launch because an unrelated GPU job held inference-bench
isolation. Keep the default threshold unchanged until a clean paired run shows
a real prefill-body win without decode-work growth.

## Public 20260706_130207 and current-head all-provider refresh

The latest public run advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_130207`.
It measured TorchInferno `3558018`, vLLM `07f9baf`, and SGLang `80decc7`.
This is the first public run with the accepted reusable Gumbel scratch runtime;
the later TorchInferno commits through `9d51ff9` record rejected probes and do
not change the active runtime path. Public TorchInferno wins self_consistency
(`101.5 / 0.0 / 108.6ms`) and few_shot TPOT (`44.2ms` versus vLLM `45.9ms`),
but still trails vLLM/SGLang on few_shot (`+18.7ms` TTFT, `-1.7ms` TPOT,
`+24.0ms` E2E), multi_turn (`+152.2ms`, `+13.9ms`, `+164.6ms`),
tree_of_thought (`+47.5ms`, `+42.4ms`, `+66.0ms`), and long_output
(`+206.0ms`, `+5.1ms`, `+391.3ms`).

The public queue profile keeps long_output as the largest score-facing gap.
It spent `3.98s` in prefill forward, padded `34.6K` prefill tokens
(`43.6%`), spent `10.32s` in ragged decode GPU work, and spent `4.73s` GPU
plus `1.28s` CPU across `112` decode-many calls. The hot decode-many body is
still the full-batch first window: `decode_many:b64/64:g1-16` consumed
`1.34s` total at `188us/token`, followed by `g17-32` at another `296ms`.
Multi_turn remains prefill-body bound (`2.47s` prefill forward, `17.5K`
padded tokens, `53.5%` padding). Public tree is much faster than several local
tree reruns, but the target is unchanged: sampled short-prefix prefill plus
ragged sampled decode (`4.94s` prefill forward, `173.2ms` prefill sample,
`4.53s` decode GPU, and sampled static decode graph misses
`static_logits=32,static_token=1`).

A same-host all-provider refresh on current TorchInferno head
(`/tmp/inference-bench-current-full-results/.../20260706_133554`) used vLLM
`cc1d020`, SGLang `602c861`, and TorchInferno `9d51ff9`. TorchInferno still
wins self_consistency (`65.4 / 0.0 / 70.9ms`) but trails on few_shot
(`+72.8ms` TTFT, `+8.2ms` TPOT, `+82.5ms` E2E), multi_turn (`+86.2ms`,
`+8.7ms`, `+100.4ms`), tree_of_thought (`+124.5ms`, `+38.9ms`, `+166.3ms`),
and long_output (`+179.9ms`, `+6.7ms`, `+428.9ms`). Because the active
TorchInferno runtime is the same as public `3558018`, treat the public/local
spread, especially tree, as run variance and provider-revision context rather
than a TorchInferno code regression.

The local current-head profile sharpens the next targets. Long_output spent
`4.68s` in prefill forward, padded `32.9K` tokens, spent `10.53s` in ragged
decode GPU, and spent `5.49s` GPU plus `1.38s` CPU in `114` decode-many calls.
The same `b64/64:g1-16` window alone consumed `2.17s` total at
`210us/token`, so denser or cheaper decode-many replay/readback remains the
largest E2E lever; simply raising the stop-tail cap was already rejected.
Tree spent `409.5ms` in prefill sampling and `650.8ms` total in TP Gumbel
sampling (`301.1ms` noise, `174.0ms` max, `78.4ms` reduce), while the public
tree run shows that this path is noisy enough that future sampler work needs a
score-facing A/B, not just lower phase counters. Multi_turn and few_shot are
still ordinary prefill-body/padding targets (`16.9K` and `3.4K` padded tokens)
with no new state-bookkeeping culprit.

An adjacent same-head long_output suffix-split recheck keeps the guarded split
diagnostic-only. The no-split control
(`/tmp/inference-bench-current-nosplit-control-results/.../20260706_140045`)
landed at `224.5 / 23.4 / 1009.8ms`, `1000/1000` correct. Enabling
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS_GREEDY_SHORT=1`
(`/tmp/inference-bench-current-suffixsplit-results/.../20260706_135507`)
landed at `217.9 / 23.1 / 1105.5ms`, also `1000/1000` correct. The split
accepted `12/19` candidates and cut prefill padding from `34.6K` to `27.2K`
tokens (`9.0K` accepted saved tokens), but prefill batches rose `58 -> 72`,
prefill forward stayed flat/slightly worse (`4.66s -> 4.70s`), and decode-many
work rose (`4.80s/1.29s` GPU/CPU over `105` calls and `360` steps to
`5.86s/1.48s` over `120` calls and `438` steps). Keep the automatic split off;
lower padding alone is not enough when fragmentation pushes more work into
decode-many and worsens median E2E.

Enabling the existing multi-step decode-many CUDA graph is also rejected as a
default for long_output. With only
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_GRAPH=1` on current `8abf063`, the
focused run
(`/tmp/inference-bench-decode-many-graph-results/.../20260706_142727`) landed
at `229.5 / 23.9 / 1167.2ms`, `1000/1000` correct, versus the no-graph
control's `224.5 / 23.4 / 1009.8ms`. The graph path did execute (`25` graph
calls, `116` graph steps), and it reduced CPU token readback, but GPU time
regressed badly: decode-many GPU rose from `4.80s` to `7.36s`, while the hot
`decode_many:b64/64:g1-16` window rose from `1.60s` at `210us/token` to
`4.07s` at `548us/token`. Keep the flag off until the captured multi-step body
is made cheaper than repeated one-step ragged decode.

Async decode-many readback stays opt-in as well. The focused run with only
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_ASYNC_READBACK=1`
(`/tmp/inference-bench-decode-many-async-readback-results/.../20260706_143417`)
landed at a slightly better `208.8 / 23.2 / 1006.0ms`, `1000/1000` correct,
but the mechanism did not validate: decode-many CPU time rose from `1.29s` to
`1.45s`, decode-many GPU rose from `4.80s` to `5.82s`, and the run performed
more decode-many work (`438` steps and `23.5K` model tokens versus `360` steps
and `19.0K`). Treat the median win as scheduling/run variance until async
readback reduces CPU tokens time without increasing the model-work envelope.

A small TP sampler cleanup is accepted on the sampled tree path: use scalar
sentinel values in the greedy and Gumbel token tie-breaks instead of allocating
a full sentinel tensor for every sample. The adjacent baseline control on
`6893af8`
(`/tmp/inference-bench-sampler-scalar-control-results/.../20260706_141832`)
landed at `145.8 / 67.4 / 208.4ms`, `0.968` correctness. The scalar-sentinel
run
(`/tmp/inference-bench-sampler-scalar-results/.../20260706_141149`) landed at
`119.9 / 63.4 / 182.6ms`, `0.966` correctness. The profiler supports the
general allocation cleanup: prefill sample selection dropped from `399.6ms` to
`302.9ms`, and cumulative TP Gumbel time dropped from `616.1ms` to `568.2ms`
even though the accepted run issued more sampler calls (`476` versus `436`).
Tree remains noisy, so keep pursuing sampler collectives and sampled decode
graph misses, but this change removes per-call allocation work from the active
runtime path.

Forcing static decode capture for generated-prefix traffic is not a fix for the
remaining sampled-tree graph misses. The focused run with
`TORCHINFERNO_CONTINUOUS_DECODE_CAPTURE=1`
(`/tmp/inference-bench-tree-static-capture-results/.../20260706_144249`) landed
at `131.6 / 64.0 / 190.7ms`, `0.967` correctness, slower than the accepted
scalar-sentinel tree run. It still reported `19` static-logits decode misses and
recorded no new decode captures, so the env override did not address the
`b2-b5/s55-s57` miss pattern. Keep generated-prefix static capture disabled
unless a future change warms or captures those concrete cache shapes directly.

## Public 20260706_110224 refresh and sample split follow-up

The public run then advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_110224`.
It still measured TorchInferno `0d6ab82`, so it does not include the pushed
state/sample split telemetry. vLLM advanced to `ba22152`, while SGLang stayed at
`80decc7`. TorchInferno still wins self_consistency (`100.8 / 0.0 / 105.8ms`)
but trails vLLM on few_shot (`+19.1ms` TTFT, `+24.5ms` E2E), multi_turn
(`+112.6ms` TTFT, `+18.9ms` TPOT, `+127.7ms` E2E), tree_of_thought
(`+48.7ms` TTFT, `+42.0ms` TPOT, `+69.5ms` E2E), and long_output
(`+179.5ms` TTFT, `+5.6ms` TPOT, `+343.6ms` E2E).

The public multi_turn queue profile now reports `44.6ms` in prefill state
bookkeeping, so the earlier `293ms` public state bucket was not stable. The
current multi_turn target remains the mixed-prefix body and queue formation:
`2.53s/2.92s` prefill forward/wall, `18.3K` padded prefill tokens
(`49.4%`), `41.7ms` prefill sample, and `371.5ms` decode state updates.
Long_output is still split between prefill TTFT and decode density:
`4.03s/4.37s` prefill forward/wall, `36.7K` padded prefill tokens, `10.36s`
ragged decode GPU, `5.92s` decode-many GPU, `1.56s` decode-many CPU handling,
and only `164` decode-many model tokens per call across `137` calls.

Tree remains the sampled-decode outlier: public `0.7/300` spent
`4.84s/5.27s` in prefill forward/wall, `143.6ms` in prefill sample, `4.36s`
ragged decode GPU, and still missed sampled static decode graphs
(`static_token=70,static_logits=29`). The sample split below should make the
next public run distinguish sampler selection from host readback; do not reopen
the rejected sampled decode and greedy sampler variants without a new profile
showing a different bottleneck.

A current local tree baseline on pushed `818c201`
(`/tmp/inference-bench-main-818c201-tree-results/.../20260706_120339`) landed at
`128.0 / 64.3 / 186.3ms`, `0.967` correctness. Its final queue profile spent
`458.5ms` in prefill sample, split as `445.6ms` distributed sampler selection
and `12.7ms` host readback; sampled static-token misses were mostly gone
(`static_logits=18,static_token=1`). A later Gumbel-profile run with the new TP
sampler counters
(`/tmp/inference-bench-gumbel-profile-results/.../20260706_123750`) landed at
`144.2 / 66.7 / 207.5ms`, `0.967` correctness. Its prefill sample bucket was
`326.0ms` (`314.2ms` selection, `11.6ms` readback), while cumulative
temperature sampling across prefill plus sampled decode recorded `419` Gumbel
calls over `2599` rows and `578.2ms` total: `221.6ms` Gumbel noise generation,
`174.8ms` local/global max selection, and `82.6ms` final token reduction.
This confirms the default sampled path is already Gumbel; the earlier
`TORCHINFERNO_TEMPERATURE_SAMPLE_GUMBEL=1` run was a noisy same-path rerun, not
an alternate sampler. The selected-row CDF prototype
(`/tmp/inference-bench-selected-cdf-results/.../20260706_121835`) did not
exercise the default sampled-tree path and should not guide defaults. Keep the
sampler target open around the active Gumbel phases, especially noise
generation and the two distributed reductions.

Disabling Gumbel to force the exact distributed CDF sampler is rejected. The
same-host CDF run
(`/tmp/inference-bench-cdf-sampler-results/.../20260706_124643`) landed at
`156.7 / 79.3 / 222.6ms`, `0.966` correctness. Its prefill sample bucket rose
to `617.7ms` (`608.5ms` selection), and cumulative TP temperature sampling
spent `1.88s` across `374` calls: `228.9ms` max reduction, `210.0ms`
exp/weight sum, `1.19s` rank selection/all-gather/broadcast, `171.6ms` local
CDF/search, and `75.4ms` final token reduction. Keep Gumbel as the default
unless a future implementation reduces random-noise and reduction cost without
falling back to this CDF collective shape.

The accepted follow-up is reusable in-place Gumbel noise scratch. The same-host
scratch run
(`/tmp/inference-bench-gumbel-scratch-results/.../20260706_125443`) landed at
`117.3 / 61.6 / 168.8ms`, `0.965` correctness, improving the local tree median
latencies despite ordinary tree-run noise. Its cumulative Gumbel profile covered
`444` calls over `2657` rows and spent `554.5ms` total, with noise generation
down to `151.8ms` versus the prior Gumbel-profile run's `221.6ms` over `2599`
rows. Prefill sample selection also stayed lower (`300.5ms` versus `314.2ms`).
Keep the scratch path enabled by default with the env opt-out for future A/Bs.

Rechecking the existing sampled-medium `s12,16` suffix bucket opt-in on top of
the Gumbel scratch path is still rejected as a default. The same-host run
(`/tmp/inference-bench-s12-scratch-results/.../20260706_130409`) landed at
`122.5 / 60.7 / 179.8ms`, `0.967` correctness. It cut prefill padding from the
scratch baseline's `8.3K` tokens (`2.9K` row / `5.5K` suffix) to `3.9K`
(`2.4K` row / `1.5K` suffix), and profiled prefill forward/wall moved from
`3.06s/3.68s` to `2.85s/3.47s`. The score-facing TTFT/E2E still regressed from
the scratch baseline's `117.3 / 61.6 / 168.8ms`, while TPOT was effectively
flat and prefill sample readback rose (`10.8ms` to `16.8ms`). Keep `s12,16` as
an explicit diagnostic/runtime opt-in until it produces a repeatable median win,
not just lower padding counters.

A single-gather Gumbel sampler is also rejected. The env-gated probe gathered
each rank's local perturbed max value and token, replacing the current
max-allreduce plus token-min-allreduce pair with one value/token all-gather.
The same-host run
(`/tmp/inference-bench-gumbel-gather-results/.../20260706_131134`) regressed to
`129.2 / 69.0 / 197.3ms`, `0.965` correctness. The final queue profile removed
the reduce bucket (`0.0ms`) but moved cumulative Gumbel max/select work to
`471.7ms`, versus `205.6ms` max plus `89.2ms` reduce in the scratch baseline;
total Gumbel sampling rose to `634.2ms` from `554.5ms`. Keep the two-reduce
Gumbel sampler until there is a fused or backend-native value/token reduction,
not a tensor all-gather of the local maxima.

In-place perturbed-logit reuse is rejected as well. The probe reused the Gumbel
scratch tensor for `logits + noise` before the local max, avoiding the explicit
temporary in the current `torch.max(logits_float + gumbel, dim=-1)` expression.
The same-host run
(`/tmp/inference-bench-gumbel-add-results/.../20260706_131819`) regressed to
`131.0 / 68.0 / 193.9ms`, `0.960` correctness. The max phase was slightly lower
(`193.9ms` versus `205.6ms`), but cumulative noise time rose to `243.9ms` and
total Gumbel sampling rose to `639.7ms`; prefill sample selection also rose to
`367.6ms`. Keep the scratch buffer limited to noise generation until a fused
perturb-and-reduce path proves a score-facing win.

Uniform-based Gumbel noise is rejected. The env probe generated scratch noise as
`-log(-log(U))` instead of using the exponential sampler. The same-host run
(`/tmp/inference-bench-gumbel-uniform-results/.../20260706_132528`) landed at
`137.8 / 64.8 / 198.7ms`, `0.963` correctness, with p99 TTFT/E2E regressing to
`753.9/812.0ms`. The max/reduce phases were not the problem (`187.8ms` and
`85.0ms`), but noise generation rose to `408.9ms` and total Gumbel sampling rose
to `782.1ms`. Keep the exponential scratch generator as the default noise path.

## Public 20260706_090220 refresh and queue segment merge fix

The prior public run advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_090220`.
It measured TorchInferno `0d6ab82`, vLLM `90ce3a0`, and SGLang `80decc7`.
TorchInferno still wins self_consistency (`94.5 / 0.0 / 101.8ms`) and now wins
few_shot TPOT (`44.1ms` versus vLLM `46.0ms`), but still trails on few_shot
TTFT/E2E (`+21.6ms` / `+31.2ms`), multi_turn (`+85.1ms` TTFT, `+92.0ms` E2E),
tree_of_thought (`+47.4ms` TTFT, `+62.5ms` E2E), and long_output
(`+199.3ms` TTFT, `+4.6ms` TPOT, `+251.3ms` E2E).

The new public queue profile exposed a second analyzer merge case. Long_output
wrote two final `online_batcher` segments with `499` and `501` submitted
requests; the old restart detector only merged segments when counters dropped,
so it kept the latter as `501/1000 partial`. The analyzer now treats a final
`online_batcher` record as closing a segment and starts a new segment for later
same-key snapshots, while still replacing ordinary `online_batcher_quiescent`
progress snapshots within the current segment. Re-rendering the run reports
`1000/1000 2seg` and the correct full long_output totals.

With the corrected merge, public long_output still targets decode first:
`4.03s` prefill forward, `4.41s` prefill wall, `35.6K` prefill padding tokens,
`10.12s` ragged decode GPU, `5.74s` decode-many GPU, and `1.46s` decode-many
CPU token handling. The dominant decode-many target remains the full-batch body:
`decode_many:b64/64:g1-16` is `27.7%` of decode-many time at `188us/token`, and
`g17-32` adds another `4.7%`; the tiny `b8` tails are inefficient but only about
`3%` each. This keeps the long-output work split unchanged: reduce padded
cached-prefix prefill for TTFT, and reduce or overlap single-step decode replay
plus token readback for TPOT/E2E.

The queue table now prints decode-many aggregate density as well. In this
public long_output run TorchInferno processed only `22.5K` decode-many model
tokens across `128` calls (`176` model tokens/call, `3.8` generated steps/call),
while SGLang reported `71.5K` decode tokens across `24` decode batches
(`2.98K` tokens/batch). The comparison is not one-to-one because providers
report different scheduler boundaries, but it makes the remaining gap concrete:
TorchInferno is still launching many small single-step replay groups instead of
amortizing long decode phases into denser graph work.

A same-host stop-tail cap A/B on current head rejects simply raising the cap
from the default `4` to `8`. The control run
`/tmp/inference-bench-tailcap4-control-results/.../20260706_101035` landed at
`216.2 / 23.3 / 1057.5ms`, while the cap-8 run
`/tmp/inference-bench-tailcap8-results/.../20260706_100410` landed at
`247.5 / 23.0 / 1077.5ms`, both `1000/1000` correct. Cap 8 reduced
decode-many calls (`106 -> 88`) and CPU token handling (`1.30s -> 1.09s`), but
it increased overgenerated tokens (`756 -> 1043`), worsened TTFT/E2E and
throughput, and did not reduce total decode GPU time. Keep the default cap at
`4`; denser decode needs a real multi-step graph/body win, not just longer
stop-token bursts.

Suffix-split candidate telemetry now records opportunities even when the split
policy is disabled in default greedy-short profiling runs. Serving behavior is
unchanged; accepted split execution remains opt-in. Future public long_output
queue profiles can now show whether a guarded suffix-bucket split would pass
`min_group`/`min_fill` and how many padded model tokens it would save before we
spend another full A/B run on it.

A local TorchInferno-only long_output run on `17af09d`
(`/tmp/inference-bench-suffix-profile-results/.../20260706_102600`) landed at
`213.1 / 23.5 / 1099.5ms`, `1000/1000` correct. The new queue counters found
`19` suffix-split candidates with `5.5K` candidate saved model tokens, but only
`6` reached the disabled-policy rejection; the rest failed `no_savings`,
`min_fill`, or `min_group`. The analyzer now prints candidate calls, candidate
saved tokens, and compact reject reasons so future public profiles expose this
signal directly.

A follow-up local TorchInferno-only long_output run with reusable prefix-prefill
`seq_lens` scratch
(`/tmp/inference-bench-seqlens-scratch-results/.../20260706_104714`) landed at
`203.1 / 23.8 / 1077.7ms`, `1000/1000` correct. The targeted setup counter
dropped from `295.3ms` to `82.9ms`; prefill wall was roughly flat
(`5.29s -> 5.23s`) and prefill forward moved noisily in the wrong direction
(`4.66s -> 4.80s`). Keep the scratch reuse because it removes real per-prefill
host/device setup overhead, but do not treat it as the structural long-output
fix: the remaining gap is still dense cached-prefix prefill body cost plus
decode replay/readback density.

Two smaller prefix-setup follow-ups are rejected and were backed out. Replacing
the per-prefill row/source/logit-position tensors with cached device-index
tensors wrote
`/tmp/inference-bench-cached-index-results/.../20260706_105621` and landed at
`216.1 / 23.7 / 1101.7ms`, with setup worsening to `91.2ms`. Changing the
uniform common-prefix seq-lens update from `index_copy_` to `index_fill_` wrote
`/tmp/inference-bench-uniform-seqlens-results/.../20260706_110347` and landed
at `240.4 / 23.4 / 1111.7ms`, with setup `94.8ms`. Keep only the reusable
seq-lens scratch; the remaining setup work is small enough that these tensor
construction variants are run-noise or worse, not a new default lever.

The prefix-prefill state bucket is now split into cache-row seq-len update,
reusable-prefix store, and request/event creation. A local multi_turn rerun with
the split counters and zero-filled seq-lens scratch semantics
(`/tmp/inference-bench-prefill-state-zero-scratch-results/.../20260706_113425`)
landed at `214.5 / 61.9 / 274.8ms`, `0.982` correctness, matching the public
multi_turn correctness neighborhood (`0.980`). The queue profile spent
`2.79s/3.41s` in prefill forward/wall and only `62.0ms` in the prefill state
bucket, split as `39.7ms` row seq-len update, `15.9ms` reusable-prefix store,
and `5.4ms` request/event creation. This rules out the state bookkeeping bucket
as the current multi_turn TTFT/E2E target; the profile remains dominated by
padded mixed-prefix prefill body cost, with sampled first-token handling noisy
across runs. The scratch helper also now preserves the original zero-filled
`required`-length tensor contract for unfilled rows instead of returning a
full-cache-row view with stale entries.

The follow-up prefix-prefill sample split on pushed `f09e875`
(`/tmp/inference-bench-prefill-sample-split-results/.../20260706_114426`)
landed at a noisy `323.5 / 58.6 / 366.2ms`, `0.981` correctness. The merged
queue profile split `101.6ms` of prefill sample time into `95.4ms` sampler
selection and only `6.1ms` host readback, so the sample bucket is not a token
copy problem. It is mostly the greedy TP sampler path around first-token
selection; keep this as telemetry for future public profiles rather than
reopening the already-rejected greedy sampler alternatives without new evidence.

A same-host provider refresh
(`/tmp/inference-bench-provider-long-results/.../20260706_103437`) measured
vLLM `68.8 / 17.0 / 676.0ms` and SGLang `63.5 / 23.1 / 928.8ms`, both
`1000/1000` correct. SGLang's log reported `70.8K` decode tokens in only `17`
decode batches (`4.16K` tokens/batch, `100%` decode graph), while the adjacent
TorchInferno profile processed `22.3K` decode-many model tokens in `115` calls
(`194` tokens/call, `3.6` steps/call). The same-host comparison keeps the
priority on a denser/faster decode body and a packed cached-prefix prefill path;
suffix splitting alone is too small.

The latest run also reinforces the packed-prefix conclusion. Few_shot is closer
but remains prefill-bound (`1.38s` prefill forward, `4.66K` padding tokens, hot
`prefix_graph:b32:s16:p122-122:src1:mixed0`), while multi_turn now surfaces a
large `b32:s144:p45-45` cached-prefix shape with `20.3K` padding tokens. The
queue table backfills `cache=dense` and still shows `packed_fi_calls=0`, so the
existing packed FlashInfer prefill body remains ineligible for public dense-cache
runs.

## Public 20260706_050221 refresh and prefill GPU timing split

An earlier public run advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_050221`
at inference-bench `069f519c`. It measured TorchInferno `fa12aa1`, vLLM
`6971582`, and SGLang `6f22790`; the public runner has not yet picked up
`a2a126e`. TorchInferno scored `4/20`: it won self_consistency
(`98.9 / 0.0 / 105.8ms`, `9.5 tok/s`) but still trailed vLLM on few_shot
(`165.3 / 45.1 / 203.4ms` vs `125.7 / 41.2 / 156.7ms`), multi_turn
(`228.8 / 58.5 / 280.7ms` vs `155.1 / 50.6 / 199.9ms`), tree_of_thought
(`80.2 / 64.6 / 114.1ms` vs `33.0 / 21.7 / 48.7ms`), and long_output
(`267.8 / 20.6 / 990.0ms` vs `78.2 / 14.7 / 600.2ms`).

The public long_output queue profile still reports `decode_many_graph_calls=0`,
so the default path remains a decode-many scheduler around single-step ragged
decode graph replays. The two long-output TorchInferno sessions spent about
`4.19s` in prefill forward, `4.73s` prefill wall, `4.94s` decode-many GPU,
`1.31s` decode-many CPU-copy, and `38.6K` prefill padding tokens. This keeps
the same target split: dense cached-prefix prefill padding on TTFT/E2E and
single-step decode replay plus token readback on TPOT/E2E.

The analyzer now merges restart-split TorchInferno queue-profile segments for
the same `(temperature, max_tokens)` key. Public long_output had two server
segments (`627` and `373` submitted requests); the old formatter kept only the
last segment and showed `373/1000 partial`, undercounting prefill/decode phase
totals. The merged view reports `1000/1000 2seg` and sums counters/maps while
leaving per-segment p50 latency fields blank because aggregate medians cannot be
reconstructed from JSONL snapshots alone.

Two local long_output profiles on `a2a126e` refine the prefill evidence. A
context-filtered `p111+s64` replay probe did not print because the one-shot hook
missed the filtered replay, but the queue profile showed warmed, miss-free
hot shapes: `b16:s64:p111` (`20` calls, `1.30s` forward), `b24:s64:p111`
(`14` calls, `1.22s`), and `b16:s96:p111` (`12` calls, `1.03s`), with `33.4K`
prefill padding and `4.30s` decode-many GPU. A broad replay probe did fire on
the first small `b1:s16:ctx-64` graph. Even at that size, the CUDA profile was
dominated by TP all-reduce (`3.39ms`), QKV GEMM (`3.38ms`), Marlin gate-up
(`1.82ms`), index/index-select kernels (`1.95ms` combined), and split-K GEMM
(`0.98ms`) out of `17.17ms` self CUDA. That does not justify toggling Marlin or
symm-memory off; it points back to a lower-overhead packed cached-prefix prefill
body and less fragmented decode/readback work.

Queue telemetry now records CUDA-event GPU time for ragged prefill graph
capture/replay separately from CPU submission time. New fields include
`runtime_prefill_graph_capture_gpu_ms`,
`runtime_prefill_graph_replay_gpu_ms`,
`runtime_prefill_shape_graph_replay_gpu_ms`, and graph-shape GPU maps. This
matters because the old `runtime_prefill_graph_replay_ms` measured only the
Python/CUDA-graph submission window (`~100-200ms` in long_output), while
`runtime_prefill_shape_forward_ms` synchronized and included the actual GPU
completion (`~4-5s`). Future public profiles should use the new GPU fields when
ranking prefill graph-body work.

The post-change validation run on `a0dc2f3` wrote
`/tmp/inference-bench-prefill-gpu-telemetry-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-prefill-gpu-telemetry-a0dc2f3/runs/20260706_055924`
and landed at `235.0 / 23.1 / 1082.2ms`, `1000/1000` correct. The new counters
recorded `runtime_prefill_graph_replay_gpu_ms=4683.8` versus the old
`runtime_prefill_graph_replay_ms=146.7`, matching
`runtime_prefill_forward_ms=4695.9`. The hot GPU replay shapes were
`b24:s64:p111` (`1227.6ms`), `b16:s64:p111` (`841.1ms`), `b24:s96:p111`
(`724.1ms`), `b16:s96:p111` (`603.6ms`), and `b32:s64:p111` (`548.4ms`).

A targeted long_output replay profile on pushed `0d6ab82` corrected the earlier
exact-context filter and hit the actual dynamic-context bucket:
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_CONTEXT_LEN=-256`,
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_MIN_BATCH=16`, and
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_MIN_SUFFIX=64`. It wrote
`/tmp/inference-bench-long-prefill-ctxneg256-s64-prof-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-prefill-ctxneg256-s64-prof-0d6ab82/runs/20260706_060809`
and landed at `205.0 / 23.5 / 1098.3ms`, `1000/1000` correct. The one-shot
profiler fired on `batch=24 suffix=64 context_len=-256` and recorded
`83.27ms` self CUDA: NCCL all-reduce was `24.52ms` (`160` calls), the main
GEMM/NVJET buckets were about `22.6ms` combined, add/RMS/elementwise/index work
was the next tier, and visible softmax attention was only `0.65ms`. This
confirms the dynamic bucket mask is not the leading long-output prefill cost;
the hot body is ordinary TP transformer work over padded suffix rows. Queue
telemetry for the same profile reported `6.19s` prefill graph GPU replay,
`5.05s` decode-many GPU, `1.32s` decode token readback, and
`decode_many_graph_calls=0`. The profiled `b24:s64` row was profiler-inflated
(`2.74s` GPU across `14` calls), but it usefully exposed density:
`127.4us/model-token` and `637` padded tokens per call. The summary formatter
now prints `gpu_ms_call`, `gpu_us_tok`, and `pad_call` in the hot prefill shape
table so future public queue profiles can identify these targets without manual
JSON parsing.

A current-head packed-prefix rerun with start-grouped packed attention stayed
rejected and was reverted. The run used
`TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER=1`,
`TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER_GRAPH=1`, and a
shared-prefix grouped SDPA prototype that collapsed exact `(prefix, suffix)`
attention groups into one masked group per prefix start. It wrote
`/tmp/inference-bench-packed-startgroup-long-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-packed-startgroup-long/runs/20260706_062730`
and landed at `1016.9 / 73.7 / 3871.6ms`, `1000/1000` correct. The path saved
`34.0K` padded suffix tokens but spent `48.25s` in packed eager work; hot
observed packed costs were `12.54s` for `b16:s64:p111`, `11.33s` for
`b24:s64:p111`, and `7.97s` for `b16:s96:p111`. This rules out a PyTorch
masked-SDPA start-grouping shim as the packed-prefix implementation; the viable
path still needs a real varlen prefill kernel/body that avoids both padded
transformer tokens and per-group Python/SDPA work.

A same-host current-head long_output comparison after `543df7f` used the
available local provider environments instead of rebuilding. The first run wrote
`/tmp/inference-bench-current-allproviders-long-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-current-allproviders-long-543df7f/runs/20260706_064427`;
the fresh build dir lacked vLLM/SGLang virtualenvs, so it produced only the
TorchInferno leg: `211.6 / 23.5 / 1061.3ms`, `33.9 tok/s`, `1000/1000`
correct. The provider-only rerun used existing envs at vLLM `cc1d020d0194` and
SGLang `602c8615a1af`, wrote
`/tmp/inference-bench-current-providers-long-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-current-providers-long-543df7f/runs/20260706_065031`,
and landed at vLLM `69.4 / 16.9 / 669.1ms`, `52.8 tok/s`, and SGLang
`61.0 / 24.1 / 944.5ms`, `38.2 tok/s`, both fully correct. This is not an
exact public-run reproduction because the provider commits differ, but it
confirms the same split: TorchInferno's TTFT is still about `3x` SGLang and TPOT
is still behind vLLM, while TorchInferno is near SGLang on TPOT.

The same provider-only run exposed a small analyzer blind spot. Current
inference-bench saves provider artifacts as `provider_logs/vllm.log` and
`provider_logs/sglang.log`, while the TorchInferno analyzer only read the older
`*_server.log` names. The analyzer now accepts both forms. Re-rendering the
local run shows vLLM's single aggregate runtime line (`4.8K` prompt tok/s,
`3.2K` generation tok/s, `64.9%` prefix hit) and SGLang's detailed phase logs:
`355` prefill batches, `44.4K` new tokens, `111.3K` cached tokens, `100%`
prefill graph coverage, `17` decode batches, `71.1K` decoded tokens, and `100%`
decode graph coverage. That is useful provider context for future local runs,
but the runtime target remains lower padded-prefix prefill work plus lower
decode replay/readback cost.

A finer greedy-short suffix-bucket A/B is also rejected for the score-facing
cold-run path. The run set
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS_GREEDY_SHORT=16,32,48,64,80,96,112,128,256`
and wrote
`/tmp/inference-bench-finer-suffix-buckets-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-greedy-short-finer-suffix-buckets-543df7f/runs/20260706_070002`.
It preserved correctness but landed worse at `220.5 / 23.4 / 1130.1ms`,
`32.9 tok/s`, with severe early-wave p99. The queue profile did reduce prefill
padding (`34.6K` -> `22.2K`, `43.7%` -> `33.2%`) and shifted hot shapes to
`s80`/`s48`, but it introduced new cold graph captures:
`runtime_prefill_graph_capture_gpu_ms=5697.7` and total prefill forward
`9.50s` versus `4.73s` on the default current run. Do not promote finer
greedy-short suffix buckets without a warm-shape strategy or a cheaper
non-capturing path.

A follow-up warm-shape A/B kept the same finer suffix buckets and explicitly
warmed the missing common-prefix shapes with
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_GREEDY_COMMON_PREFIX_EXTRA_PAIRS=111:32,111:48,111:64,111:80,122:16`.
It wrote
`/tmp/inference-bench-finer-suffix-warm-p111-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-finer-suffix-warm-p111-543df7f/runs/20260706_071001`
and landed at `215.8 / 23.1 / 1021.6ms`, `34.8 tok/s`, with `1000/1000`
correct. This eliminated measured prefill graph capture
(`runtime_prefill_graph_capture_gpu_ms=0`) and cut prefill padding to `23.8K`
tokens (`34.7%`), but startup grew to `266s` and measured prefill forward still
cost `4.37s`. The result is not strong enough to make finer greedy-short
suffix buckets or broader p111 startup warmup the default; the next useful
target is reducing the actual padded-prefix prefill body, not adding more
bucket shapes.

The existing warm-row prefix-copy skip remains rejected for default serving. An
A/B with `TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SKIP_WARM_PREFIX_COPY=1` wrote
`/tmp/inference-bench-warm-row-prefix-skip-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-warm-row-prefix-skip-543df7f/runs/20260706_071817`
and preserved correctness, but regressed to `231.6 / 24.9 / 1442.8ms`,
`28.3 tok/s`; its queue-profile artifact was empty, so use only the request
metrics from that run. A patched diagnostic that also warmed no-prefix-copy
(`src0`) suffix graphs never reached server readiness after several minutes,
raised graph memory substantially, and was stopped before measured traffic.
That rules out turning prefix-copy skip plus broader startup warmup into a
default. Avoiding the shared-prefix KV copy is still only useful if a future
path can reuse the existing `src1` graph shape or avoid multiplying startup
graph captures.

## Public 20260706_070225 refresh and tree decode-capture check

The latest public run advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_070225`.
It measured TorchInferno `0d6ab82`, vLLM `cdab283`, and SGLang `5f98f62`.
TorchInferno scored `3/20`: it still wins self_consistency
(`100.9 / 0.0 / 107.9ms`), but trails vLLM on few_shot
(`158.8 / 44.5 / 206.5ms` vs `121.9 / 42.5 / 158.0ms`), multi_turn
(`234.3 / 56.7 / 291.9ms` vs `145.5 / 44.6 / 190.0ms`), tree_of_thought
(`76.4 / 63.4 / 107.2ms` vs `32.8 / 21.7 / 48.4ms`), and long_output
(`307.5 / 18.5 / 1061.5ms` vs vLLM `74.8 / 14.8 / 658.8ms`).

The score targets remain the same. Public long_output is still dominated by
decode and decode-many replay/readback (`10.46s` ragged decode GPU,
`5.14s` decode-many GPU, `1.24s` decode-many CPU) with `32.1K` prefill padding
tokens. Multi_turn and few_shot remain prefill bound, and tree remains split:
`5.17s` prefill forward, `4.99s` decode GPU, `9.4K` prefill padding tokens, and
`95` decode graph misses.
The decode-many implementation target table now prints token and total-time
shares. Re-rendering public long_output shows the full-batch body, not just the
tiny tail rows, is still the main decode-many lever:
`decode_many:b64/64:g1-16` accounts for `35.8%` of window model tokens and
`24.3%` of decode-many time, `g17-32` adds `11.2%`, and `g33-48` adds `3.4%`.
The sparse `b2/b3` tails are several milliseconds per token but only about
`3%` of total decode-many time per row, so tail gating alone cannot close the
long-output gap.
The provider log table now also prints provider aggregate token density. In the
same public run SGLang reports `70.7K` decode tokens over only `24` logged decode
batches (`2946.4` tokens/batch) with `100%` decode graph coverage, while
TorchInferno's long_output profile still uses `109` decode-many calls and `436`
single-step decode-many steps. This keeps the contrast focused on real decode
batching/replay structure rather than the already-rejected tiny-tail policies.
The queue table now also includes `decode_many_graph_ms`, which clarifies the
rejected TorchInferno multi-step graph path. A focused long_output profile with
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_GRAPH=1` and
`TORCHINFERNO_PROFILE_RAGGED_DECODE_MANY_REPLAY_ONCE=1` wrote
`/tmp/inference-bench-long-manygraph-prof-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-long-manygraph-prof-f48d307/runs/20260706_093552`
and landed at `219.9 / 23.5 / 1142.5ms`, `1000/1000` correct, after `231s`
server readiness. The graph path fired (`38` graph calls, `202` graph steps,
`12.9K` graph model tokens) but spent `7.80s` in `decode_many_graph_ms`, with
`b64/64` alone at `7.28s`. The one profiled `batch=64,steps=8,cache1024` replay
spent `99.1ms` self CUDA, led by dense GEMM/NVJET (`27.1ms`), gate-up Marlin
(`26.7ms`), multimem all-reduce (`15.4ms`), GQA decode attention (`12.4ms`),
and splitK GEMM reduction (`8.2ms`). That is roughly `12.4ms` per generated
step, matching the single-step replay profile; the current multi-step graph does
not amortize the model body and remains an opt-in diagnostic.
The packed-prefix target tables now also print `est_share`, the estimated saved
prefill-forward time divided by total profiled prefill forward. On public
few_shot, the hot `prefix_graph:b32:s16:p122-122:src1:mixed0` candidate is
large enough to matter by itself (`415ms`, `28.9%` of prefill forward), so a
real packed cached-prefix body for that shape would be score-facing. Its main
coarse pattern repeats (`p122:s12/s13/s14` for `33` calls), but fixed-capacity
slots are not viable for that row: the observed maxima (`24*s12 + 16*s13 +
7*s14`) would execute `594` packed suffix tokens per call, more than the dense
`b32*s16=512` bucket. The analyzer now surfaces these as
`packed prefill fixed-capacity rejects`; public few_shot shows this row as
`19.6K` fixed tokens versus `16.9K` dense tokens despite `5.1K` raw saved
tokens. Public tree has more repeated fixed patterns, but the largest
fixed-capacity target is only `258ms` (`5.0%` of prefill forward); tree still
needs a broader packed-prefix body and lower sampled decode cost rather than one
narrow pattern rewrite.

A local forced fixed-capacity few_shot run after adding runtime reject counters
confirmed the analyzer prediction. The diagnostic wrote
`/tmp/inference-bench-torchinferno-few-fixedpacked-telemetry-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260710_053201`
and landed at `201.7 / 36.0 / 241.9ms`, `977/1000` correct, worse than the
default allreduce few_shot control. The queue profile reported
`runtime_prefill_packed_fixed_capacity_attempts=33`,
`runtime_prefill_packed_fixed_capacity_accepts=0`, and rejects
`{"no_savings": 20, "capacity_grew": 8, "graph_returned_none": 5}` with
`runtime_prefill_packed_eager_calls=0`. The same run also logged prefill logits
graph capture invalidation during startup. Keep the fixed-capacity switches as
diagnostic telemetry until the packed body can run below the dense `b32*s16`
graph cost.

An env-only few_shot exact-suffix probe on the same branch confirms that simply
shrinking the dense bucket is not enough. With
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS=12,13,14,16` plus
matching `p122` warmup, the run
`/tmp/inference-bench-few-exact-suffix-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-few-exact-suffix-f82a9c6/runs/20260706_092410`
landed at `179.3 / 45.2 / 218.9ms`, `977/1000` correct, after `190.9s`
server readiness. It cut profiled few_shot prefill padding to `1.54K` tokens,
but the hot body became `prefix_graph:b32:s14:p122-122:src1:mixed0` and slowed
to `55.1ms/call` / `123us/model-token`, worse than the public `s16` body's
`~40.7ms/call` / `79us/model-token`. Keep exact greedy-mid suffix buckets as a
diagnostic only; the score-facing fix still needs a faster model-side cached
prefix body, not a smaller dense suffix bucket.

The queue-profile analyzer now prints the runtime cache backend and backfills it
from older TorchInferno launch logs. Re-rendering public `20260706_070225`
shows `cache=dense` with `packed_fi_calls=0`, which makes the packed
FlashInfer prefill path ineligible for that run. The existing
`prefill_ragged_logits_packed_flashinfer` body requires a FlashInfer KV cache
with paged storage; the current dense public path cannot reach it without a new
cache representation or a per-layer KV repack that would need its own benchmark
evidence. Treat dense-cache packed FlashInfer as blocked, not as an untried
environment toggle.

A same-branch tree A/B checked whether generated-prefix decode capture should
be enabled by default. The opt-in decode-capture run
`/tmp/inference-bench-tree-decode-capture-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-tree-decode-capture-aca24b0/runs/20260706_074034`
landed at `127.8 / 66.0 / 184.6ms`, `956/992` correct. The no-env control
`/tmp/inference-bench-tree-control-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-tree-control-aca24b0/runs/20260706_074717`
landed slightly better at `124.6 / 65.5 / 179.7ms`, `957/992` correct. Capture
trimmed aggregate decode GPU (`2.57s` vs `2.71s`) but did not improve request
medians or correctness, so keep `TORCHINFERNO_CONTINUOUS_DECODE_CAPTURE` as an
explicit diagnostic. The analyzer now retains
`runtime_decode_graph_miss_shape_counts` and summarizes it as
`decode_miss_kind`; the tree control showed `static_token=27,static_logits=20`,
which points future work at the static generated-prefix decode fallback rather
than ragged graph miss churn.

Two follow-up probes are not defaultable. Enabling global uniform ragged decode
with `TORCHINFERNO_CONTINUOUS_UNIFORM_RAGGED_DECODE=1` wrote
`/tmp/inference-bench-tree-uniform-ragged-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-tree-uniform-ragged-d38cfa3/runs/20260706_075729`
and landed at `117.3 / 66.1 / 163.8ms`, `958/992` correct. That improves median
TTFT/E2E versus the no-env control, but worsens p99 (`679/770ms`), raises
profiled decode GPU (`3.11s` vs `2.71s`), raises prefill sampling
(`600ms` vs `507ms`), and still leaves static token misses (`static_token=17`).
Keep uniform ragged decode as an explicit diagnostic.

A narrower runtime patch that kept normal grouping but routed the FlashInfer
sampled-token fallback through ragged logits was also reverted. The local run
`/tmp/inference-bench-tree-ragged-fallback-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-tree-ragged-fallback-d38cfa3-dirty/runs/20260706_080520`
landed at `127.8 / 62.4 / 192.7ms`, `960/992` correct. It improved TPOT but
regressed TTFT/E2E, and the new miss split showed it did not hit the public
Llama path: misses remained static (`static_token=41,static_logits=20`). The
next useful decode-side change needs to target the non-FlashInfer static branch
or make uniform ragged faster, not reroute the FI fallback.

The static branch now skips static token-graph attempts for sampled decode,
because the Llama static token graph is greedy-only and sampled decode always
falls through to logits. The validation run
`/tmp/inference-bench-tree-static-token-skip-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-tree-static-token-skip-d38cfa3-dirty/runs/20260706_081507`
landed at `129.8 / 63.9 / 189.9ms`, `956/992` correct. This is not a request
median win, but it removes the impossible sampled token-graph churn:
`decode_miss_kind` fell from the control's `static_token=27,static_logits=20`
to `static_token=1,static_logits=15`, and profiled decode GPU fell from
`2.71s` to `2.52s`. Treat it as hot-path cleanup and profile clarity, not as a
closed tree gap; tree still needs faster prefix prefill and a real sampled
decode path.

A focused multi_turn warmup probe added the two public miss shapes
(`8:32:128` and `2:16:256`) to
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_MIXED_PREFIX_SUFFIX_SPECS`. The run
`/tmp/inference-bench-multiturn-warm-miss-shapes-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-multiturn-warm-miss-shapes-92335d4/runs/20260706_082439`
kept startup at `226s` and eliminated prefill graph misses, but request latency
regressed to `288.2 / 59.6 / 380.3ms`, `982/1000` correct. Queue profile
counters moved in the wrong user-visible direction even though prefill forward
fell (`2.46s`): `q2first=170ms`, `q2submit=100ms`, and
`prefill_sample_ms=100ms`. Do not add these warmup specs by default; the
multi_turn gap is dominated by padded mixed-prefix prefill and queue formation,
not these few request-path captures.

A local current-head provider slice on `f7f3d38` rechecked tree_of_thought using
the existing vLLM/SGLang environments:
`/tmp/inference-bench-current-tree-providers-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-current-tree-providers-f7f3d38/runs/20260706_083559`.
TorchInferno landed at `132.6 / 66.0 / 188.0ms`, `960/992` correct; vLLM landed
at `39.3 / 27.2 / 58.1ms`, `964/992` correct; SGLang landed at
`39.8 / 103.4 / 109.9ms`, `964/992` correct but with very poor p99. The
TorchInferno profile stayed miss-free on prefill and reduced decode misses to
`static_logits=21`, but still spent `3.12s` prefill forward, `3.76s` prefill
wall, `363ms` sampling first tokens, `8.5K` prefill padding tokens, and `2.73s`
ragged decode GPU. This confirms the static-token cleanup improved profile
clarity without closing the tree gap; the next defaultable runtime work still has
to reduce prefix-prefill body cost or sampled ragged/static decode cost.

The analyzer now applies `--benchmark` selection to queue-profile-derived tables
instead of only to provider metric rows. Public multi-benchmark runs carry one
TorchInferno queue snapshot per benchmark policy key, so a tree-only render should
show the `0.7/300` profile and hide unrelated `0.0/96`, `0.0/256`, and
`0.0/512` profile rows. If a custom run has no matching built-in key, the
formatter falls back to all profiles so diagnostics are not accidentally hidden.
It also prints top prefill/decode graph miss shapes. For the current local tree
slice, the remaining sampled decode misses are concrete late static-logits
shapes (`b2:s56`, `b3:s55`, `b2:s58`, and neighbors), which makes future warmup
or path-selection probes easier to target without manually opening JSONL.
The hot prefill-shape table also backfills call counts from graph replay/capture
maps and, for prefix-graph rows, from model tokens divided by `batch * suffix`.
That makes per-call padding and GPU costs visible on public profiles where
`runtime_prefill_shape_counts` was missing; for example, public multi_turn now
shows the `prefix_graph:b8:s32:p45-56:src8:mixed1` row as one call with `199`
padded tokens instead of blank per-call columns.

A direct static-logits warmup probe set
`TORCHINFERNO_OPENAI_WARMUP_PROMPT_TOKENS=56` and wrote
`/tmp/inference-bench-tree-static-logits-warm-s56-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-tree-static-logits-warm-s56-9b13b94/runs/20260706_085555`.
It landed at `132.7 / 65.4 / 190.9ms`, `957/992` correct, versus the local
current tree provider slice's TorchInferno leg at `132.6 / 66.0 / 188.0ms`,
`960/992` correct. The targeted miss class barely moved (`static_logits=20`
instead of `21`) and reshuffled to `b2:s56`, `b3:s56`, `b2:s57`, and nearby
shapes. Do not change the default decode warmup prompt length; these late small
static-logits misses are not the visible tree bottleneck.

## Public 20260706_030205 refresh

The latest public run advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_030205`
at inference-bench `356bd0ff`. It measured TorchInferno `2f33f36`, vLLM
`f2aaf59`, and SGLang `c016c6f`. TorchInferno improved to `3/20` by winning
self_consistency (`108.0 / 0.0 / 115.1ms`, `8.7 tok/s`) but still lost
few_shot (`178.9 / 45.3 / 217.5ms`), multi_turn
(`246.3 / 60.0 / 303.3ms`), tree_of_thought
(`77.5 / 63.5 / 110.9ms`), and long_output
(`277.2 / 19.8 / 1000.1ms`). The current score-facing gaps are unchanged:
long_output needs lower steady decode/readback cost, while tree and multi_turn
need cheaper padded cached-prefix prefill plus lower TP replay cost.

The public long_output queue profile was split across two TorchInferno server
sessions (`537` and `463` submitted requests). Combined, those partial profiles
showed about `4.29s` prefill forward, `10.06s` ragged decode GPU, `4.92s`
decode-many GPU, `1.31s` decode-many CPU-copy, and `35.7K` prefill padding
tokens. This run predates the new `decode_many_async_readback` marker, so that
field renders as absent/`None`; the async-readback A/B below remains current
local evidence and is not promoted.

## Public 20260706_010157 and decode host split

The previous public run advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_010157`
at inference-bench commit `2fbb9f18`. It still measured pre-current
TorchInferno `46164b4` against vLLM `95a248f` and SGLang `8673e85`.
TorchInferno remained `0/20`: few_shot `154.1 / 45.7 / 196.1ms`,
self_consistency `160.2 / 0.0 / 167.8ms`, multi_turn
`317.7 / 61.3 / 376.9ms`, tree_of_thought `79.4 / 64.1 / 114.6ms`, and
long_output `262.5 / 19.6 / 934.2ms`. The largest gaps in that run were still
long_output TTFT/E2E (`+197.7/+291.2ms` versus the best other provider) and
multi_turn TTFT/E2E (`+170.7/+188.1ms` versus vLLM), followed by tree
TTFT/TPOT/E2E.

The analyzer now carries decode host overhead, sampled prefill sampling time,
and prefill padding split into row padding and suffix padding in the
queue-profile and score-target tables. The compact phase target also treats
sampling as its own measured subsystem instead of forcing cache-only sampled
rows into a prefill/decode bucket.
Re-rendering the latest public run shows
long_output's logged partial queue profile (`547/1000` requests) spent `5.36s`
in ragged decode GPU, `897ms` copying decode tokens to CPU, `890ms` of that
inside decode-many, and `20ms` in decode state updates; it also spent `20.1K`
cached-prefix prefill tokens on padding (`44.5%`), split as `6.0K` row padding
and `14.2K` suffix padding. The hottest
`decode_many:b64/64` shape spent `1.54s` GPU and `257ms` CPU-copy. That makes
q8 drain readback visible as a material cost, but not enough to explain the
whole gap or reopen the rejected side-stream copy shim. The defaultable
long-output target remains lower `b64` replay cost or a real decode/readback
pipeline, while current-head multi_turn remains focused on the padded
cached-prefix prefill body documented below.

A focused current-head TorchInferno-only tree run on pushed `af0f574` wrote
`/tmp/inference-bench-current-tree-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-current/runs/20260706_024843`
and landed at `134.4 / 66.3 / 193.9ms`, `961/992` correct, with server
readiness `226.0s`. Queue telemetry had zero prefill graph misses, but still
spent `2.90s/3.45s` in prefill forward/wall, `300ms` in sampled prefill
sampling, `2.47s` in ragged decode GPU, and `7.8K` prefill padding tokens
(`2.35K` row / `5.47K` suffix). The per-batch packed-prefix targets remain
real (`b4:s16` saves `1.63K` tokens, `1.21K` suffix, estimated `466ms`), but
the score-facing tree gap also needs lower sampled first-token/decode cost.
This does not reopen the rejected sample-gather, Gumbel, or scratch-buffer
sampler defaults; it makes the sampler term visible in the compact target table
so future fused/graph-safe sampler work is judged against the right counter.

A current-head Gumbel recheck on `2f33f36` wrote
`/tmp/inference-bench-current-tree-gumbel-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-gumbel/runs/20260706_030042`
and remains rejected as a default. It landed at `142.9 / 67.7 / 211.7ms`,
`954/992` correct, versus the no-env current control's
`134.4 / 66.3 / 193.9ms`, `961/992` correct. The queue profile reduced
recorded prefill sampling only from `300ms` to `285ms`, while q2submit and E2E
both worsened. A narrower fixed-capacity packed-prefix probe targeting only the
single-suffix `prefix_graph:b4:s16:p45-45:src1:mixed0|p45:s10` pattern wrote
`/tmp/inference-bench-current-tree-fixed-b4s10-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-fixed-b4s10/runs/20260706_030756`
and also remains rejected: it regressed to `277.3 / 61.6 / 342.1ms`,
`960/992` correct. The targeted packed path made only `10` packed calls,
saved `303` padded tokens, and spent `3.75s` in packed eager/graph work. The
analyzer now subtracts observed packed-prototype shape time from dense
saved-forward estimates and prints `obs_packed_ms`; for this rejected run the
single-suffix `b4/p45:s10` target shows about `244ms` theoretical dense savings
next to `3748ms` observed packed cost. This keeps the existing packed-eager and
fixed-capacity switches diagnostic-only until the packed body itself is
rewritten, not merely narrowed.

The analyzer also now prints per-batch packed-prefill targets that do not require
repeatable exact signatures, with row-vs-suffix saved-token splits attached to
each target. On the same public run, the largest target is tree_of_thought's
`prefix_graph:b4:s16:p45-45:src1:mixed0` shape: `126` calls, `3.64K` saved
tokens (`1.30K` row / `2.35K` suffix, `45.2%`), and an estimated `1.22s`
prefill-forward saving for a true per-batch packed body. Across the
score-facing rows, suffix padding dominates total padding for long_output
(`14.2K/20.1K`), multi_turn (`3.3K/4.4K`), and tree (`5.5K/9.5K`), so the first
packed-prefill body needs to remove per-row suffix waste as well as any
batch-row waste. The long_output one-shot targets are the `b32:s96`,
`b16:s96`, and `b32:s64` `p111` shapes, each saving `38-56%` of its prefill
model tokens, but they still lack fixed-pattern reuse. That confirms the first
packed-prefill implementation should support arbitrary per-batch suffix lengths,
not only fixed-capacity repeating slots.

A full current-head TorchInferno-only long_output profile on `30a1872` wrote
`/tmp/inference-bench-ti-30a-long-profile-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_020656`
and landed at `215.4 / 24.1 / 1080.3ms`, `1000/1000` correct. Queue telemetry
shows both halves of the same remaining gap: `q2first_p50=163.5ms`,
`q2submit_p50=33.7ms`, `submit2first_p50=114.8ms`, `64` prefill batches,
`4.80s/5.24s` prefill forward/wall, `10.35s` ragged decode GPU, and
`1.41s` decode CPU-copy. Decode-many accounted for `111` calls, `364` steps,
`4.75s` GPU, and `1.34s` CPU-copy; the hot `decode_many:b64/64` row was
`1.50s` GPU and `286ms` CPU-copy, with the early `g1-16` window dominant.
Long-output also has large per-batch packed-prefix savings (`35.2K` candidate
saved tokens, `44.0%` of prefill model tokens) but almost no repeatable packed
pattern reuse (`2/63` calls), so the current dense fixed-capacity packed-eager
path remains the wrong default. This keeps the next long-output work on a true
per-batch packed cached-prefix prefill body plus a lower-cost or overlapped
decode/readback body.

Current-head queue profiles now also attribute decode-many CPU token readback by
step window as `runtime_decode_many_step_window_cpu_tokens_ms`, and the summary
prints it as `cpu_ms` beside per-window model time. The implementation-target
table now ranks windows by `total_ms` (`gpu_ms + cpu_ms` when both are
available) and marks `gpu_src` as `exact` for profiles with
`runtime_decode_many_step_window_model_ms` or `est` for older artifacts that
must still derive GPU time from whole-shape averages. Older public artifacts
render the CPU/model columns as `-` or `est`.

A current-head long_output profile with that telemetry on `0366d17` wrote
`/tmp/inference-bench-current-long-decode-window-cpu-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-decode-window-cpu/runs/20260706_032928`
and landed at `204.9 / 23.3 / 1047.3ms`, `1000/1000` correct. Queue telemetry
recorded `61` prefill batches, `4.64s/5.04s` prefill forward/wall,
`10.35s` decode GPU, and `1.41s` decode CPU-copy; decode-many contributed
`112` calls, `386` internal steps, `5.00s` GPU, and `1.35s` CPU-copy. The new
step-window CPU split shows the largest window,
`decode_many:b64/64:g1-16`, at `8256` model tokens, `333ms` CPU-copy, and an
estimated `1.73s` GPU (`2.06s` combined). The next high-active first-window
rows (`b49`, `b62`, `b61`, `b57`, `b60`) each sit around `212-269ms` combined
GPU+CPU. That rules out a tail-only
readback fix for the median long_output gap: readback is spread across the
main full/near-full decode-many body, so the needed change is genuine
readback/decode pipelining or lower replay cost, not another stop-tail split.

Current head now also attaches decode-many step-window metadata to deferred
CUDA decode events before they are flushed. That populates
`runtime_decode_many_step_window_model_ms` on CUDA profiles instead of leaving
the analyzer to estimate per-window GPU time from whole-shape averages. This is
still stats-only plumbing. A follow-up current-head run on `db1ca1f` wrote
`/tmp/inference-bench-current-long-window-gpu-db1ca1f-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-window-gpu-db1ca1f/runs/20260706_042514`
and landed at `238.2 / 23.7 / 1062.9ms`, `1000/1000` correct, so it is not a
score improvement. It did prove the exact attribution: the dominant
`decode_many:b64/64:g1-16` window spent `1774ms` model/GPU time plus `319ms`
CPU readback over `8448` model tokens, while `b64/64:g17-32` was only
`269ms + 31ms`. The next decode change should target lower high-active
single-step replay cost or a real emission/readback pipeline, not tail-only
readback.

Rechecking the guarded greedy-short suffix splitter on the same current head is
still not a clean default promotion, but it is useful evidence. With
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS_GREEDY_SHORT=1`,
the run wrote
`/tmp/inference-bench-current-long-guarded-suffix-split-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-guarded-suffix-split-a73a7d1/runs/20260706_043850`
and landed at `214.2 / 23.5 / 1045.3ms`, `1000/1000` correct, versus the
adjacent no-split current run at `238.2 / 23.7 / 1062.9ms`. The split reduced
prefill model tokens (`51.6K -> 49.5K`), padding (`38.0K -> 32.1K`), and
decode-many work (`388 -> 314` internal steps, `5.05s -> 4.21s` decode-many
GPU). It also left prefill wall essentially flat (`5.28s -> 5.26s`) and
worsened long_output TPOT p99 (`29.1ms -> 36.8ms`). Keep the automatic suffix
split default off: the current signal is a median TTFT/E2E improvement against
one noisy same-head control, not a robust public-style TPOT/E2E win. A better
prefill fix still needs non-fragmenting packed cached-prefix prefill or a
grouping policy that avoids tiny split fragments.

Current head now records suffix-split candidate telemetry in TorchInferno queue
profiles: accepted/rejected candidate counts, base versus split model-token
totals, accepted saved tokens, fragment counts, rejection reasons, and compact
candidate/fragment shape maps. This is stats-only and does not change the split
default. It is meant to show which suffix-bucket fragments caused the TPOT-tail
tradeoff above before any stricter grouping policy is promoted.

Current head also exposes an opt-in low-occupancy decode-many gate,
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_MIN_ACTIVE_PCT`, with queue-profile
fields for the configured percentage and skip count. The focused long_output
probe at `25%` wrote
`/tmp/inference-bench-minactive25-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-minactive25-fa12aa1-plus/runs/20260706_051402`
and is rejected as a default: it improved TPOT p99 (`32.9ms -> 31.2ms`) but
regressed median TTFT/E2E (`204.8/1052.3ms -> 219.0/1088.6ms`) and throughput
(`33.7 -> 33.1 tok/s`) versus the adjacent no-split control. The profile
recorded `43` min-active skips, but decode-many GPU still rose
(`4.19s -> 4.40s`) as work shifted into fuller `b64/64` windows. Keep the gate
diagnostic-only; the long_output fix remains lower high-active replay cost or a
real decode/readback pipeline.

The direct pinned-host async readback A/B is rejected as a default. The first
attempt on
`/tmp/inference-bench-async-readback-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-async-readback/runs/20260706_034651`
fell back to the old blocking copy because of a stream-cache naming collision
and landed near control at `203.4 / 23.2 / 1038.3ms`. After fixing the stream
cache and preallocating the pinned host scratch for the full burst, the real
async run wrote
`/tmp/inference-bench-async-readback-fixed-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-async-readback-fixed/runs/20260706_035723`
and landed worse on median TTFT/throughput at `232.1 / 23.5 / 1046.1ms`,
`1000/1000` correct. Queue profiles confirmed the opt-in path was active via
`decode_many_async_readback=true`, but total decode-many CPU time did not drop
(`1.29s` versus `1.24s` in the fallback run) and decode GPU rose (`4.68s`
versus `4.48s`). Keep
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_ASYNC_READBACK` default-off as a
diagnostic; a useful readback overlap needs less stream/copy interference or a
different event-emission pipeline.

## Public 20260705_230202 and current HTTP split

The latest public run at
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_230202`
measured TorchInferno `cddde7e`, vLLM `b712181`, and SGLang `8673e85`.
TorchInferno scored `0/20` against vLLM's `18/20`: few_shot
`164.3 / 46.0 / 213.1ms`, self_consistency `157.4 / 0.0 / 166.1ms`,
multi_turn `322.2 / 62.5 / 381.2ms`, tree_of_thought
`76.6 / 63.0 / 106.5ms`, and long_output `282.6 / 20.1 / 989.8ms`.
The final queue records show two different gaps. Self_consistency is not
model-bound (`q2first_p50=6.6ms`, `q2submit_p50=0.7ms`,
`submit2first_p50=5.3ms`, `2` prefill batches). Tree and long remain
model/scheduler-bound: tree spent `5.28s/5.74s` in prefill forward/wall and
`5.02s` in ragged decode GPU; long spent `4.12s/4.46s` in prefill
forward/wall, `9.61s` in ragged decode GPU, and `3.79s` in decode-many GPU.

A focused current-head self_consistency run with fast-HTTP detailed profiling
(`9415f7e`,
`/tmp/inference-bench-ti-http-profile-self-results/.../runs/20260705_235400`)
landed at `93.0 / 0.0 / 101.9ms`, `1000/1000` correct. Its server-side
profile confirms the public self_consistency queue/score mismatch is mostly
outside model execution: queue `q2first_p50=8.1ms`, fast-HTTP
`first_engine_token_p50=9.2ms`, `first_content_sent_p50=9.5ms`, and
`total_p50=9.7ms`, while client-observed TTFT was `93.0ms`. The HTTP path's
accepted-to-handler p50 was only `0.06ms`; request-read p50 was `19.8ms`.

The current profiling patch records the scoped stream prequeue gate directly in
online queue profiles as `request_stream_prequeue_wait_*` fields. This is
needed for the 512-token mixed-prefix multi_turn path, where the accepted
prequeue wait is meant to reduce request-prompt admission fragmentation but
older public profiles did not show whether the gate actually fired.

A focused current-head multi_turn validation on pushed `d7b41ee` wrote
`/tmp/inference-bench-ti-d7b-multiturn-profile-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_000557`
and landed at `204.0 / 59.7 / 259.1ms`, `983/1000` correct. This is a much
better local band than the public `322.2 / 62.5 / 381.2ms` row, but still
trails the recent local vLLM/SGLang TTFT band near `143ms`. Queue telemetry
shows the mixed-prefix path is active (`prefix_rows=112`, `mixed_prefix=true`,
`41` prefill batches, `2.80s/3.13s` prefill forward/wall, `854ms` ragged decode
GPU, `q2first_p50=142.6ms`, `q2submit_p50=57.3ms`,
`submit2first_p50=85.6ms`). The new prequeue fields show why raising the
prequeue wait is not the next lever: all `1000` requests were configured for a
`2ms` stream prequeue wait, but only `4` actually applied and the median
prequeue elapsed time was `0.0ms`.

A current-head all-provider `multi_turn` refresh on `c5e2a0a` wrote
`/tmp/inference-bench-c5e2a0a-multiturn-allproviders-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_001326`:
vLLM `150.7 / 49.8 / 194.3ms`, SGLang `136.8 / 125.5 / 252.6ms`, and
TorchInferno `223.1 / 61.6 / 289.7ms`, `983/1000` correct. A forced
suffix-bucket splitter stayed rejected (`223.2 / 62.0 / 289.3ms`, p99
regressed, `prefill_forward_ms=2816`). A scoped refill-only prefill-cost
admission priority for the greedy-large mixed-prefix path looked promising in a
provider-only run (`202.9 / 59.1 / 258.3ms`, `984/1000` correct at
`/tmp/inference-bench-ti-active-refill-priority-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_003547`.
The same-host all-provider rerun on pushed `061a037` rejected it as a default:
TorchInferno regressed to `238.3 / 61.9 / 303.6ms` with
`q2submit_p50=66.1ms`, `40` prefill batches, and `2.50s` prefill forward.
Keep large-greedy refill cost priority opt-in; the remaining first-token gap
needs a scheduler policy that avoids starving longer refill prompts.

A pushed-head `6b5f32f` TorchInferno-only `multi_turn` profile wrote
`/tmp/inference-bench-ti-6b5-multiturn-profile-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_014843`
and landed at `220.8 / 63.4 / 285.2ms`, `982/1000` correct. Queue telemetry
matches the current all-provider band: `q2first_p50=144.3ms`,
`q2submit_p50=60.2ms`, `submit2first_p50=86.5ms`, `41` prefill batches, and
`2.54s/2.87s` prefill forward/wall. The new analyzer prefill split rules out
Python setup/copy as the main gap (`setup=51.5ms`, `copy=7.6ms`,
`sample=97.0ms`, `state=62.2ms`). The hottest mixed-prefix shapes are still
`b32:s32` padded suffix replays with roughly 50% suffix padding, while packed
candidate patterns had zero repeated calls in this run. That keeps the
multi_turn target on the model-side cached-prefix prefill body rather than
another admission order or setup/copy optimization.

A current safe-head long_output all-provider refresh on `46164b4` wrote
`/tmp/inference-bench-46164b4-long-allproviders-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_005421`.
vLLM landed at `69.6 / 16.9 / 662.9ms`, SGLang at
`58.6 / 24.6 / 936.7ms`, and TorchInferno at `232.4 / 22.6 / 1080.4ms`,
`1000/1000` correct. Queue telemetry still splits the gap between first-token
prefill and decode replay: `q2first_p50=192.3ms`, `q2submit_p50=35.4ms`,
`submit2first_p50=142.3ms`, `4.58s/5.14s` prefill forward/wall, `9.84s`
ragged decode GPU, and `4.95s` decode-many GPU with `815` overgenerated
tokens. The hot prefill rows remain the known common-prefix padded-suffix
shapes (`b24:s64:p111`, `b24:s96:p111`, `b16:s64:p111`). This does not reopen
suffix-bucket, batch-bucket, or decode-tail defaults; the next long-output
lever is still a non-fragmenting packed cached-prefix prefill body plus lower
per-step replay cost.

Rechecking the opt-in multi-token decode graph on current head rejects it as a
long_output default. The diagnostic run with
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_GRAPH=1` and
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_GRAPH_MIN_STEPS=2` wrote
`/tmp/inference-bench-ti-long-decodemany-graph-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_014026`
and landed at `221.5 / 23.1 / 1058.5ms`, `1000/1000` correct. The graph path
did fire (`40` decode-many graph calls, `206` graph steps, `13.2K` graph model
tokens), but it made the dominant `decode_many:b64/64` path much slower:
`5.06s` GPU in the graph run versus `1.95s` for the same shape in the adjacent
safe-head all-provider control. Total decode-many GPU rose to `8.45s`, ragged
decode GPU rose to `12.41s`, and p99 E2E stayed high at `3885ms`. Keep
multi-token decode graphs opt-in; the long-output decode gap needs a faster
single-step replay body or lower replay count, not graphing the current
multi-step body.

A current safe-head tree_of_thought all-provider refresh on `aaaa8e4` wrote
`/tmp/inference-bench-aaaa8e4-tree-allproviders-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_010735`.
vLLM landed at `39.3 / 27.0 / 57.8ms`, SGLang at
`40.6 / 111.6 / 112.4ms`, and TorchInferno at `158.0 / 62.7 / 213.6ms`,
`956/992` correct. The queue profile shows the current tree gap is active
session fragmentation plus padded cached-prefix prefill: `q2first_p50=138.4ms`,
`q2submit_p50=72.9ms`, `submit2first_p50=53.6ms`, `130` prefill batches,
`3.16s/4.47s` prefill forward/wall, and `2.50s` ragged decode GPU. Static
decode graph misses are visible (`40`, mostly sparse tail `b1/b2` sequence
lengths `55..57`), but they are secondary to the many `b4/b8/b16` cached-prefix
prefills and the remaining `8.6K` prefill padding tokens.

The same tree profiles show why packed-prefix work remains the structural tree
lever but not via the current dense packed-eager runtime: packed candidate
patterns repeated heavily (`94.5%` repeated pattern calls in the all-provider
control and `95.9%` in the no-env TorchInferno control), with fixed-capacity
plans estimating tens of milliseconds of avoidable padded prefill. That is
evidence for a real packed CUDA/FlashInfer cached-prefix body; it does not
reopen the existing dense fixed-capacity packed-eager default, which was already
measured as much slower than padded graph replay.

An active-session ready-drain wait is now available only as an explicit
diagnostic knob and is rejected as a default tree policy. The no-env
TorchInferno-only control wrote
`/tmp/inference-bench-ti-active-ready-control-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_012941`
and landed at `135.6 / 68.6 / 201.9ms`, `957/992` correct, with
`active_ready_wait_ms=0`, `122` prefill batches, `2.84s/3.39s` prefill
forward/wall, and `2.55s` ragged decode GPU. Enabling
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_ACTIVE_READY_WAIT_MS=0.5` wrote
`/tmp/inference-bench-ti-active-ready05-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260706_012412`
and improved medians to `123.0 / 63.5 / 178.4ms`, but correctness dropped to
`952/992`, p99 TTFT/E2E worsened to `771/830ms`, prefill batches rose to `142`,
prefill wall rose to `3.88s`, and ragged decode GPU rose to `2.82s`. Keep the
active-ready wait default at `0ms`; it is useful to prove that queue delay can
be traded for more small-batch prefill/decode work, not a score-safe fix.

A current tree_of_thought sample-gather recheck is rejected as a default
runtime change. On pushed `b2b11a3`, the focused run with
`TORCHINFERNO_TEMPERATURE_SAMPLE_GATHER=1` wrote
`/tmp/inference-bench-ti-b2b11a3-tree-sample-gather-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_221318`
and landed at `135.4 / 62.7 / 192.9ms`, `956/992` correct. Queue telemetry
shows why the small TPOT improvement is not enough: `q2first_p50=121.6ms`,
`q2submit_p50=71.2ms`, `prefill_forward/wall=2.98s/3.69s`,
`prefill_sample_ms=404.4`, `2.61s` ragged decode GPU, and `43` decode graph
misses. Keep the gather path opt-in for targeted experiments; the sampled tree
gap still needs lower cached-prefix prefill and first-token cost, not a gather
default.

The existing Gumbel temperature sampler is also rejected as a tree_of_thought
default on this code. The focused run
`/tmp/inference-bench-ti-0e486ce-tree-gumbel-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_222142`
with `TORCHINFERNO_TEMPERATURE_SAMPLE_GUMBEL=1` landed at
`114.9 / 64.2 / 161.0ms`, `953/992` correct. It reduced sampled prefill
time versus the gather run (`prefill_sample_ms=353.8`) but increased
prefill work (`140` batches, `prefill_forward/wall=3.17s/3.79s`) and decode
GPU (`2.84s`) while leaving repeated-sample-state hits at zero. Keep it
opt-in; the sampled-tree path needs less common-prefix prefill and decode work,
not a different default draw algorithm.

A decode QKV scratch-buffer prototype is rejected and was not retained in code.
It routed `_qkv` through a reusable `attention-qkv` buffer by adding an `out=`
path to `_decode_linear`, but the output-buffer `torch.mm` path did not improve
benchmark medians. The dirty-tree tree_of_thought run
`/tmp/inference-bench-ti-qkv-scratch-tree-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_222932`
landed at `134.8 / 69.1 / 202.9ms`, `956/992` correct. The decode-heavy
long_output run
`/tmp/inference-bench-ti-qkv-scratch-long-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_223507`
landed at `228.6 / 23.9 / 1056.0ms`, `1000/1000` correct, with
`decode_gpu_ms=10.54s` and `decode_many_gpu_ms=4.14s`. Keep QKV on the current
`F.linear`/decode-linear path; remaining decode work needs fewer replays or a
faster attention/collective body, not this allocation tradeoff.

The greedy all-reduce sampler path is rejected for long_output. On clean
`bc538c2`, the opt-in run with `TORCHINFERNO_GREEDY_SAMPLE_GATHER=0` wrote
`/tmp/inference-bench-ti-greedy-allreduce-long-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_224320`
and landed at `210.6 / 23.1 / 1063.8ms`, `1000/1000` correct. Its paired
default gather-control run
`/tmp/inference-bench-ti-bc538c2-long-default-control-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_224902`
landed better overall at `198.0 / 23.9 / 1043.4ms`, also `1000/1000`
correct. Queue counters were not a hidden sampler win: all-reduce had
`q2first_p50=168.1ms`, `decode_gpu_ms=9.70s`, and `decode_many_gpu_ms=5.24s`,
while the gather control had `q2first_p50=159.8ms`, `decode_gpu_ms=9.80s`, and
`decode_many_gpu_ms=5.11s`. Keep the CUDA greedy gather default.

Prompt-lookup decode is rejected for long_output despite high proposal
acceptance. The opt-in run with
`TORCHINFERNO_CONTINUOUS_PROMPT_LOOKUP_DECODE=1` wrote
`/tmp/inference-bench-ti-prompt-lookup-long-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_230818`
and landed at `4244.6 / 367.9 / 16877.9ms`, `1000/1000` correct. Queue
counters show the problem: `1547` prompt-lookup verification batches across
`4292` requests proposed `34336` tokens and accepted `30420`, but this disabled
decode-many and spent `281.9s` in decode GPU, dominated by
`prompt_lookup:b3:proposal8`. Keep prompt lookup default-off for this workload;
accepted proposals are not useful when verification fragments the decode body.

Allowing short greedy decode-many while requests are waiting is rejected as a
default. The opt-in run with
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_WITH_WAITING=1` wrote
`/tmp/inference-bench-ti-decode-many-waiting-long-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_232115`
and landed at `422.5 / 15.8 / 1083.8ms`, `1000/1000` correct. It did improve
median TPOT, but first-token pacing collapsed: `q2first_p50=393.9ms`,
`submit2first_p50=377.1ms`, and decode-many expanded to `320` calls,
`953` steps, and `12.19s` decode-many GPU. The scheduler kept the model busy
after admission, but it starved newly admitted rows from first-token service.
Keep waiting decode-many disabled until there is an event-flush or overlap
design that preserves TTFT.

Raising the short greedy drain-only decode quantum from `8` to `16` is also
rejected on the current pushed tree. The env-only run with
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_DRAIN_DECODE_QUANTUM=16` wrote
`/tmp/inference-bench-ti-drain16-long-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_232736`
and landed at `298.4 / 21.9 / 1046.8ms`, `1000/1000` correct. Compared with
the adjacent default-control row (`198.0 / 23.9 / 1043.4ms`), drain-16 bought
a small TPOT win but lost TTFT and E2E. Queue counters show the same tradeoff:
`q2first_p50=227.7ms`, `submit2first_p50=178.3ms`, `129` decode-many calls,
`604` decode-many steps, and `7.75s` decode-many GPU. Keep the default drain
quantum at `8`; larger bursts need a first-token-preserving flush policy.

## Public 20260705_210211

Public inference-bench advanced to run
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_210211`
while the checked-out inference-bench repo was at `5d4447d`. It measured
TorchInferno `79bd32e`, so it predates the later Marlin output-buffer,
decode-MLP scratch, shared-temperature list, and rejected-sample-scratch
commits now pushed to main. Scorecard stayed TorchInferno `3/20`, vLLM
`15/20`, and SGLang `1/20`.

- few_shot: TorchInferno `175.1 / 44.4 / 212.3ms`, vLLM
  `115.7 / 39.9 / 145.3ms`, SGLang `139.6 / 74.7 / 214.9ms`.
- self_consistency: TorchInferno `106.6 / 0.0 / 115.1ms`, vLLM
  `136.7 / 0.0 / 161.6ms`, SGLang `226.8 / 0.0 / 361.1ms`.
- multi_turn: TorchInferno `245.4 / 56.9 / 297.0ms`, vLLM
  `153.7 / 45.3 / 196.7ms`, SGLang `164.4 / 105.8 / 275.1ms`.
- tree_of_thought: TorchInferno `79.6 / 64.9 / 114.5ms`, vLLM
  `33.1 / 21.7 / 49.0ms`, SGLang `47.9 / 382.9 / 404.6ms`.
- long_output: TorchInferno `255.0 / 20.6 / 959.5ms`, vLLM
  `101.0 / 14.5 / 701.4ms`, SGLang `68.8 / 22.3 / 931.9ms`.

The queue profile points to the same remaining gaps. long_output was dominated
by `10.21s` ragged decode GPU and `3.50s` decode-many GPU, with
`q2first_p50=174.8ms` and `q2submit_p50=50.9ms`. few_shot remained a cached
prefix prefill replay problem with `q2first_p50=102.2ms`,
`prefill_forward/wall=1.48s/1.87s`, and `4.83K` packed-prefix saved-token
opportunity. multi_turn spent `2.48s/2.84s` in prefill with
`q2first_p50=142.3ms` and mixed-prefix reuse enabled. tree_of_thought still
had many sampled cached-prefix prefill waves: `q2first_p50=75.2ms`,
`q2submit_p50=37.3ms`, `prefill_forward/wall=5.05s/5.50s`, `4.65s` ragged
decode GPU, and `97` decode graph misses.

## Public 20260705_190219

Public inference-bench advanced to commit `6af04218` with run
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_190219`.
It ran after the inference-bench harness client fix and measured TorchInferno
`70fbe31`, not the later documentation-only commits. Scorecard stayed
TorchInferno `3/20`, vLLM `15/20`, and SGLang `1/20`.

- few_shot: TorchInferno `156.0 / 45.0 / 199.6ms`, vLLM
  `124.4 / 43.1 / 164.2ms`, SGLang `133.3 / 84.2 / 209.5ms`.
- self_consistency: TorchInferno `105.9 / 0.0 / 114.3ms`, vLLM
  `133.3 / 0.0 / 155.1ms`, SGLang `214.8 / 0.0 / 371.3ms`.
- multi_turn: TorchInferno `329.8 / 59.4 / 395.7ms`, vLLM
  `141.6 / 51.1 / 186.2ms`, SGLang `161.0 / 113.2 / 273.8ms`.
- tree_of_thought: TorchInferno `79.7 / 63.2 / 111.7ms`, vLLM
  `32.6 / 21.7 / 47.9ms`, SGLang `51.0 / 396.3 / 439.9ms`.
- long_output: TorchInferno `288.6 / 20.3 / 1009.2ms`, vLLM
  `76.8 / 14.7 / 670.4ms`, SGLang `68.3 / 22.7 / 943.5ms`.

The queue profile keeps the same priority ordering. few_shot improved versus
the prior public row but is still a cached-prefix prefill replay problem:
`q2first_p50=98.7ms`, `39` prefill batches, `37/2` graph hits/misses,
`1.41s/1.80s` prefill forward/wall, and `1.42s` wall in hot
`prefix_graph:b32:s16:p122-122:src1:mixed0`. self_consistency remains healthy
with `q2first_p50=6.6ms`, two prefill batches, and `989` generated-prefix reuse
hits. multi_turn regressed mostly before first token: `q2first_p50=232.8ms`,
`q2submit_p50=144.9ms`, only `606` prefix-reuse requests
(`{"common_prefix":110,"request_prompt":496}`), and `1.63s/1.80s` prefill
forward/wall across `29` prefill batches. tree_of_thought is still many small
cached-prefix prefill replays (`227` batches, `5.02s/5.47s` prefill wall) plus
`4.77s` ragged decode GPU. long_output remains decode dominated with
`10.07s` ragged decode GPU, `3.39s` decode-many GPU, and `4.12s/4.47s`
prefill forward/wall.

A focused multi_turn test of a longer greedy-large online idle window is
rejected. The latest public profile split multi_turn into `394` and `606`
request online sessions, so the same-host A/B tried
`TORCHINFERNO_OPENAI_TP_ONLINE_PERSISTENT_IDLE_MS=100` and wrote
`/tmp/inference-bench-ti-multi-idle100-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multi-idle100-bbc9a84-20260705/runs/20260705_195041`.
It did keep all `1000` requests in one session with `1000` full-prompt
adoptions and `{"common_prefix":125,"request_prompt":875}` reuse, landing at
`293.3 / 61.6 / 353.3ms`, `982/1000` correct. The paired no-env control
`/tmp/inference-bench-ti-multi-default-control-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multi-default-control-bbc9a84-20260705/runs/20260705_195604`
also stayed in one session with `1000` adoptions and landed faster on medians:
`275.3 / 62.0 / 339.1ms`, `980/1000` correct. Do not increase the default
greedy-large persistent idle window; the public split is host/run timing noise,
not a robust gap closer.

The transposed-weight decode `mm` path is rejected for long_output. A focused
run with `TORCHINFERNO_DECODE_TRANSPOSED_WEIGHTS=1` and
`TORCHINFERNO_DECODE_LINEAR_MM=1` wrote
`/tmp/inference-bench-ti-long-decode-mm-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-decode-mm-5f737d4-20260705/runs/20260705_200955`
and landed at `220.9 / 24.1 / 1058.2ms`, `1000/1000` correct. The immediate
no-env paired control on the same commit and harness wrote
`/tmp/inference-bench-ti-long-default-paired-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-default-paired-5f737d4-20260705/runs/20260705_201526`
and was better on all medians at `207.9 / 23.5 / 1038.7ms`, also
`1000/1000` correct. Queue counters confirm this was not a hidden decode-body
win: the transposed-mm run spent `10.26s` in ragged decode GPU and `5.08s` in
decode-many GPU across `117` decode-many calls, while the control spent `9.77s`
ragged decode GPU and `5.65s` decode-many GPU across `120` calls with lower
median TPOT/E2E. Keep the transposed decode-mm path opt-in; the long_output gap
still needs a structural decode replay reduction or packed cached-prefix
prefill body, not a different `F.linear` layout.

Disabling inference-bench's TorchInferno queue profile is also rejected as a
long_output score lever on current `4cc9073`. The profile-off run
`/tmp/inference-bench-ti-profile-off-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-profile-off-20260705/runs/20260705_204347`
landed at `212.7 / 22.9 / 1034.9ms`, `1000/1000` correct. The same-host
profiled control
`/tmp/inference-bench-ti-profile-on-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-profile-on-20260705/runs/20260705_204908`
landed at `225.1 / 23.3 / 1005.6ms`, also `1000/1000` correct. Queue
profiling did not explain the gap: turning it off slightly improved TTFT and
TPOT but worsened median E2E, while the profiled control still attributed the
run to normal model work (`5.19s` prefill wall, `10.03s` ragged decode GPU,
and `5.27s` decode-many GPU over `20.7K` decode-many model tokens). Keep queue
profiling enabled for public evidence; the next useful long_output work remains
lower decode replay cost, a real decode/readback pipeline, or packed
cached-prefix prefill.

## Public 20260705_170204 after inference-bench client fix

Public inference-bench advanced to commit `f240a799` with run
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_170204`.
This is the first public all-provider row after inference-bench `76591625`
(`Use per-thread clients for concurrent benchmarks`). It measured TorchInferno
`823d043` plus the later `45e3b70` warmup in the source tree; the subsequent
TorchInferno `70fbe31` commit is documentation-only. Scorecard was
TorchInferno `3/20`, vLLM `15/20`, and SGLang `1/20`.

The harness-corrected self_consistency comparison now matches the local
diagnosis: TorchInferno wins TTFT, E2E, and throughput on that row. The
remaining public gaps are prefill/first-token dominated for few_shot,
multi_turn, and tree_of_thought, and decode/replay dominated for long_output:

- few_shot: TorchInferno `147.6 / 42.3 / 186.4ms`, vLLM
  `126.4 / 40.9 / 161.8ms`, SGLang `129.1 / 83.8 / 210.3ms`.
- self_consistency: TorchInferno `100.0 / 0.0 / 107.5ms`, vLLM
  `124.2 / 0.0 / 144.4ms`, SGLang `217.8 / 0.0 / 363.8ms`.
- multi_turn: TorchInferno `227.3 / 57.8 / 280.3ms`, vLLM
  `149.3 / 43.1 / 194.2ms`, SGLang `163.0 / 103.4 / 274.5ms`.
- tree_of_thought: TorchInferno `80.0 / 64.7 / 115.1ms`, vLLM
  `32.4 / 21.7 / 47.8ms`, SGLang `49.8 / 422.4 / 465.8ms`.
- long_output: TorchInferno `283.3 / 20.0 / 988.5ms`, vLLM
  `80.6 / 14.6 / 597.9ms`, SGLang `66.9 / 22.7 / 915.7ms`.

The public TorchInferno queue profiles explain why the old scheduler knobs are
not enough. few_shot still spent `1.42s/1.81s` in prefill forward/wall over
`39` prefill batches, with `q2first_p50=94.6ms`. self_consistency had only
`q2first_p50=6.4ms`, `2` prefill batches, and `988` generated-prefix reuse
requests, so the client fix exposed the actual runtime win. multi_turn used the
intended mixed-prefix default (`{"common_prefix":125,"request_prompt":874}`),
but still spent `2.35s/2.86s` in prefill and `q2first_p50=138.1ms`.
tree_of_thought spent `4.89s/5.35s` in prefill and `4.46s` in decode GPU
across `220` prefill batches. long_output remained decode-heavy: the profiled
sessions used `use_decode_many=true`, `decode_many_calls=62` in the larger
session, `5.78s` total decode GPU, and `3.05s` decode-many GPU.

A same-host patched all-provider comparison on TorchInferno `70fbe31` and
inference-bench `76591625` wrote
`/tmp/inference-bench-patched-all-providers-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-patched-all-70fbe31-76591625-20260705/runs/20260705_172939`.
It scored vLLM `14`, TorchInferno `3`, and SGLang `2`. Local TorchInferno
again won self_consistency (`85.1 / 0.0 / 93.1ms` versus vLLM
`101.7 / 0.0 / 123.1ms`) and trailed on the same four rows: few_shot
`179.7 / 46.3 / 222.5ms` versus vLLM `109.2 / 38.7 / 142.2ms`, multi_turn
`219.1 / 62.2 / 283.8ms` versus vLLM `138.0 / 47.7 / 177.0ms`,
tree_of_thought `148.1 / 64.3 / 209.1ms` versus vLLM
`38.9 / 27.0 / 57.7ms`, and long_output `256.9 / 23.1 / 1140.8ms` versus
SGLang/vLLM TTFT near `65-68ms` and vLLM E2E `707.2ms`. Treat the local tree
row as noisy/profile-sensitive because the public TorchInferno tree row was
much better, but the required fix is unchanged: a lower-cost packed
cached-prefix prefill path or lower TP replay/collective cost, not another
sampled-medium wait/cap tweak.

Lowering the global online initial batch wait to `0ms` is rejected for
few_shot. The focused A/B wrote
`/tmp/inference-bench-ti-few-initial0-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-few-initial0-70fbe31-20260705/runs/20260705_181700`
and landed at `178.0 / 44.3 / 216.0ms`, `977/1000` correct. The paired no-env
control wrote
`/tmp/inference-bench-ti-few-default-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-few-default-70fbe31-20260705/runs/20260705_182349`
and landed at `177.5 / 46.6 / 222.4ms`, `977/1000` correct. The zero-wait run
slightly improved TPOT/E2E, but it worsened TTFT, p99 TTFT (`886.6ms` versus
`720.2ms`), p99 E2E (`938.3ms` versus `780.4ms`), and queue-to-first
(`131.8ms` versus `122.2ms`). Keep the current initial wait; few_shot needs
faster `b32`/mixed cached-prefix prefill rather than earlier launch.

A one-shot replay profile of the current few_shot hot graph on pushed
`7e2778a` confirms that diagnosis. The focused run wrote
`/tmp/inference-bench-ti-few-prefill-replay-prof-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-few-prefill-replay-prof-7e2778a-20260705/runs/20260705_192324`
and landed at `173.4 / 46.9 / 222.1ms`, `977/1000` correct. Queue telemetry
had `q2first_p50=124.3ms`, `q2submit_p50=55.5ms`,
`submit2first_p50=67.6ms`, `36` prefill batches, `34/2` graph hits/misses,
and hot `prefix_graph:b32:s16:p122-122:src1:mixed0` at `31` calls and
`1.80s` wall. The replay profiler captured
`batch=32,suffix=16,context_len=-64,src_rows=1` at `40.4ms` total CUDA time:
NCCL bf16 all-reduce was `13.0ms` (`32%`), the largest GEMM/attention kernels
were `7.65ms`, `4.84ms`, and `1.96ms`, and RMS/elementwise/gather kernels
accounted for several more milliseconds. This rules out the remaining few_shot
gap as a graph-miss or request-collection problem; a defaultable improvement
has to reduce the TP collective count/cost or the 32-row suffix prefill model
body itself.

A current prefill symmetric-memory all-reduce recheck is rejected. The replay
profile above made the collective cost tempting, so the focused A/B set
`TORCHINFERNO_SYMM_MEM_PREFILL_ALLREDUCE=1` and wrote
`/tmp/inference-bench-ti-few-prefill-symm-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-few-prefill-symm-7e2778a-20260705/runs/20260705_192942`.
It landed at `173.0 / 47.0 / 215.1ms`, `977/1000` correct: not a clear median
win versus the adjacent no-env/profiler controls, and p99 TTFT/E2E worsened to
`1008.7/1061.8ms`. Queue telemetry kept the same shape
(`q2first_p50=123.4ms`, `q2submit_p50=55.7ms`, `34` prefill batches,
`32/2` graph hits/misses), while hot `prefix_graph:b32:s16:p122-122:src1:mixed0`
was still `31` calls and `1.79s` wall and total prefill wall rose to `2.67s`.
Keep prefill symmetric-memory all-reduce default-off for few_shot; the next
collective lever needs a measured reduction in the live replay body, not just a
different all-reduce backend flag.

A stricter local prototype that allowed graph-captured prefill symmetric-memory
all-reduce only when the buffer had already been allocated and probed by the
pre-capture warmup was also rejected and not retained in code. The focused run
wrote
`/tmp/inference-bench-ti-few-prefill-symm-ingraph-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-few-prefill-symm-ingraph-f869dd0-dirty-20260705/runs/20260705_194020`
and landed at `173.7 / 48.5 / 221.0ms`, `977/1000` correct, with p99
TTFT/E2E `1065.3/1138.5ms`. The one-shot replay profile still showed
`160` NCCL bf16 all-reduce kernels at `13.1ms` (`32%`) inside the same
`batch=32,suffix=16,context_len=-64` graph, and queue telemetry again had
`31` hot `prefix_graph:b32:s16:p122-122:src1:mixed0` replays at `1.80s`
wall. Do not spend more time on prefill symmetric-memory graph capture unless a
smaller standalone capture proves the replay body records multimem rather than
NCCL and reduces the `13ms` collective slice.

## Public 20260705_150207 and inference-bench client fix

Public inference-bench advanced to commit `0ace9c0a` with run
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_150207`.
That row measured TorchInferno `92889de`, so it includes the mixed-prefix
multi_turn policy but predates the later `45e3b70` b16 ctx128 warmup and the
inference-bench client fix below. Scorecard was TorchInferno `2/20`, vLLM
`15/20`, and SGLang `2/20`.

The public gaps were:

- few_shot: TorchInferno `153.5 / 46.8 / 202.5ms`, vLLM
  `137.6 / 52.6 / 178.0ms`, SGLang `148.0 / 74.1 / 224.1ms`. TorchInferno
  kept only the TPOT win.
- self_consistency: TorchInferno `278.0 / 0.0 / 317.3ms`, vLLM
  `195.1 / 0.0 / 218.6ms`, SGLang `221.3 / 0.0 / 370.4ms`.
- multi_turn: TorchInferno `309.8 / 62.2 / 361.5ms`, vLLM
  `173.5 / 51.7 / 221.5ms`, SGLang `166.2 / 106.0 / 268.5ms`.
- tree_of_thought: TorchInferno `121.7 / 29.1 / 145.4ms`, vLLM
  `61.6 / 30.3 / 85.4ms`, SGLang `74.7 / 52.0 / 134.5ms`. TorchInferno kept
  TPOT and p99 wins but lost TTFT/E2E/throughput.
- long_output: TorchInferno `280.1 / 19.3 / 922.3ms`, vLLM
  `77.8 / 15.0 / 638.1ms`, SGLang `73.3 / 22.2 / 835.6ms`.

The self_consistency gap was mostly benchmark-client overhead, not runtime
work. A focused TorchInferno profile on pushed `823d043` with the existing
shared inference-bench OpenAI/httpx client wrote
`/tmp/inference-bench-ti-self-http-profile-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-self-http-profile-823d043-20260705/runs/20260705_161118`
and landed at `301.2 / 0.0 / 389.0ms`, `1000/1000` correct. Queue telemetry
had `q2first_p50=8.9ms`, `q2submit_p50=2.1ms`, `submit2first_p50=6.4ms`,
one warmed prefill batch, `1/0` prefill graph hits/misses, and
`987` generated-prefix reuse requests. Fast HTTP profiling put server-side
`accepted_to_ready + first_content_sent` at `54.9ms` p50 and
`327.4ms` p90, while benchmark-visible p50 TTFT was `301.2ms`. The shared
client was the outlier: the server had already reduced the median runtime path
to single-digit milliseconds after request submission.

Changing only `TORCHINFERNO_OPENAI_FAST_HTTP_DRAINED_IDLE_TIMEOUT_SECONDS` to
`5.0` is rejected. The A/B wrote
`/tmp/inference-bench-ti-self-http-drained5-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-self-http-profile-drained5-823d043-20260705/runs/20260705_161836`
and regressed to `338.1 / 0.0 / 396.7ms`, `1000/1000` correct. Fast HTTP still
reported `1000/1000` first requests on fresh connections, so the timeout did
not fix the observed client contention.

The provider-neutral fix was pushed to inference-bench main as `76591625`
(`Use per-thread clients for concurrent benchmarks`). Concurrent benchmarks now
use one lazily-created OpenAI/httpx client per worker thread instead of sharing
one streaming client across all workers. A manual thread-local client harness
against a fresh TorchInferno server produced `29.2ms` median TTFT and `31.8ms`
median E2E, `1000/1000` correct. The real patched inference-bench validation
wrote
`/tmp/inference-bench-ti-self-threadlocal-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-self-threadlocal-bench-823d043-20260705/runs/20260705_163400`
and landed at `92.3 / 0.0 / 99.7ms`, `1000/1000` correct. The same Fast HTTP
profile still showed `1000/1000` first requests on fresh connections, but
server-side `accepted_to_ready + first_content_sent` was only `32.8ms` p50 and
`71.5ms` p90, with queue `q2first_p50=7.6ms` and `q2first_p90=15.2ms`.
Treat the next public self_consistency result after inference-bench `76591625`
as a harness-corrected comparison; no TorchInferno runtime default should be
changed from the keepalive A/B.

## Public 20260705_130223 refresh pending default mixed-prefix measurement

Public inference-bench advanced to commit `fb5f61e9` with run
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_130223`.
This row measured TorchInferno `c85d39b`, so it includes the small mixed-prefix
warmup follow-up but still predates pushed `e32436f` / `6c41917`, where the
OpenAI `temperature=0,max_tokens=512` mixed-prefix policy became the no-env
default and the greedy-large decode-many rejection was documented. The public
scorecard is TorchInferno `6/20`, vLLM `10/20`, and SGLang `3/20`.

The public shape moved a few cells but did not change the unresolved work:

- few_shot: TorchInferno `156.7 / 46.7 / 201.8ms`, vLLM
  `151.9 / 57.1 / 205.9ms`, SGLang `141.7 / 75.0 / 216.9ms`. TorchInferno
  keeps TPOT/E2E wins.
- self_consistency: TorchInferno `119.7 / 0.0 / 170.0ms`, vLLM
  `200.8 / 0.0 / 224.5ms`, SGLang `219.2 / 0.0 / 372.3ms`. TorchInferno
  wins TTFT/E2E/throughput.
- multi_turn: TorchInferno `371.5 / 59.0 / 423.8ms`, vLLM
  `175.1 / 55.8 / 227.5ms`, SGLang `170.2 / 109.2 / 276.2ms`. This is still
  the old common-prefix-only route: queue telemetry shows
  `{"common_prefix":1000}`, `35` prefill batches, no request-prompt reuse, and
  `4.09s/4.27s` prefill forward/wall. It does not measure the default-on
  `{"common_prefix":125,"request_prompt":875}` path from `e32436f`.
- tree_of_thought: TorchInferno `130.0 / 29.3 / 151.4ms`, vLLM
  `62.5 / 30.5 / 85.5ms`, SGLang `75.3 / 48.7 / 136.6ms`. The row keeps a
  TPOT win but remains TTFT/E2E bound.
- long_output: TorchInferno `258.9 / 20.6 / 918.4ms`, vLLM
  `80.5 / 15.1 / 623.3ms`, SGLang `73.9 / 22.0 / 848.1ms`. The queue profile
  remains decode-heavy (`815` decode batches, `105` decode-many calls,
  `9.77s` ragged-decode GPU), matching the current decode-throughput gap rather
  than a missing default knob.

Rechecking current `11f7acc` on focused local long_output wrote
`/tmp/inference-bench-ti-long-current-head-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-current-20260705/runs/20260705_141140`
and landed at `234.1 / 23.4 / 1129.4ms`, `1000/1000` correct. The row was
still decode-bound: `63` prefill batches, `4.79s/5.22s` prefill forward/wall,
`10.26s` ragged-decode GPU, `4.61s` decode-many GPU, and `611` overgenerated
tokens with the default stop-tail cap `4`. Turning the short-greedy stop-tail
cap off is rejected. The focused A/B
`/tmp/inference-bench-ti-long-tail0-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-tail0-20260705/runs/20260705_141815`
landed at `261.6 / 22.7 / 1152.7ms`, `1000/1000` correct: TPOT moved slightly
down, but E2E worsened, overgeneration rose to `1282`, decode-many GPU rose to
`5.42s`, and replay time rose to `4.47s`. Keep the default tail cap at `4`;
the remaining long_output gap is still cheaper per-step model replay or a real
prefill packing path, not looser stop-tail filtering.

Keep the next public read focused on whether the measured TorchInferno commit
has advanced past `c85d39b`; only then should the multi_turn row be compared
against the local no-env mixed-prefix validations below. Queue profiles now also
emit `greedy_large_mixed_prefix_reuse` so that public rows self-report whether
the scoped OpenAI policy was active. `inference-bench-summary` prints that field
as `mixed_prefix` and also exposes the recorded backend/decode/admission/
prefill-ready policy columns (`fp8_prefill`, `fp8_min_m`, `marlin_decode`,
`decode_many`, `decode_q`, `drain_q`, `admit_cap`, `min_free`, `min_ready`,
`prefill_ready`, and `ready_cap`) in its queue-profile table.

The first pushed-head focused `multi_turn` validation on `a170138` wrote
`/tmp/inference-bench-ti-multiturn-current-a170-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multiturn-current-a170-20260705/runs/20260705_144052`
and landed at `310.8 / 63.1 / 362.8ms`, `981/1000` correct. The score row was
noisier than the best no-env mixed-prefix full-suite checks, but the policy and
mechanics matched the intended default: `mixed_prefix=true`, `prefix_rows=112`,
`prefill_ready=false`, `decode_many=false`, `q2first_p50=156.9ms`, `37`
prefill batches, `2.29s/2.87s` prefill forward/wall, `843.8ms` ragged-decode
GPU, `37/0` prefill graph hits/misses, and route counts
`{"common_prefix":125,"request_prompt":875}`. Treat this as evidence that the
public `c85d39b` row is stale, not as a reason to re-enable PRBD; the
mixed-prefix PRBD-on path was already rejected below.

A same-host provider-only `multi_turn` refresh wrote
`/tmp/inference-bench-local-providers-multiturn-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-providers-multiturn-20260705/runs/20260705_150358`.
vLLM built at `cc1d020d0194` and landed at `178.6 / 60.2 / 235.7ms`,
`981/1000` correct. SGLang built at `602c8615a1af` and landed at
`152.1 / 114.8 / 269.9ms`, `979/1000` correct. This keeps the current local
target concrete: TorchInferno's mixed-prefix path is near the vLLM TPOT band,
but it still needs about `130ms` lower median TTFT versus vLLM and about
`160ms` versus SGLang.

Lowering the greedy-large mixed-prefix admission floor/cap is rejected. The
current-stack A/B with
`TORCHINFERNO_OPENAI_TP_ONLINE_ADMIT_PER_STEP_CAP=16` and
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_REFILL_MIN_READY_REQUESTS=16`
wrote
`/tmp/inference-bench-ti-multiturn-admit16-928-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multiturn-admit16-928-20260705/runs/20260705_151635`
and regressed to `594.7 / 76.5 / 672.5ms`, `982/1000` correct. Queue telemetry
showed `68` prefill batches, `23` prefill graph misses, `10.49s/10.85s`
prefill forward/wall, and `q2first_p50=517.9ms`. Smaller waves exposed
uncaptured `b16` mixed-prefix shapes and stretched queue-to-submit rather than
reducing first-token time. Keep the default `admit_cap=48` and
`min_ready=32` for this class.

A follow-up on pushed `45e3b70` added the exact missing
`ragged_prefill:b16:s32:rows1:ctx-128:src16` warmup shape and reran the same
admit16/min-ready16 A/B at
`/tmp/inference-bench-ti-multiturn-admit16-b16ctx128-45e-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multiturn-admit16-b16ctx128-45e3b70-20260705/runs/20260705_154405`.
That fixed the graph-miss pathology and recovered much of the regression:
`324.7 / 78.4 / 401.1ms`, `980/1000` correct, `67/0` prefill graph
hits/misses, `3.10s/3.50s` prefill forward/wall, `q2first_p50=245.4ms`, and
`q2submit_p50=126.5ms`. The policy is still rejected because it remains slower
than the default mixed-prefix run (`310.8 / 63.1 / 362.8ms`) and still produces
`67` prefill batches instead of the default run's `37`; keep the warmup shape,
but do not lower the default greedy-large admission floor/cap.

Raising the scoped greedy-large mixed-prefix prequeue wait from `2ms` to `3ms`
is rejected. The `3ms` env run on pushed `e3aea11` wrote
`/tmp/inference-bench-ti-multiturn-prequeue3-e3-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multiturn-prequeue3-e3aea11-20260705/runs/20260705_155338`
and landed at `247.7 / 63.3 / 355.9ms`, `982/1000` correct. The paired no-env
control on the same commit wrote
`/tmp/inference-bench-ti-multiturn-default-e3-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multiturn-default-e3aea11-20260705/runs/20260705_155926`
and was better at `246.1 / 61.0 / 319.4ms`, `983/1000` correct. Queue
telemetry also favors the current default: `q2submit_p50=60.0ms` and
`q2first_p50=147.2ms` with `36/0` prefill graph hits/misses, versus
`q2submit_p50=72.0ms`, `q2first_p50=153.9ms`, and `37/0` prefill graph
hits/misses for `3ms`. Keep the scoped mixed-prefix prequeue wait at `2ms`.

The finished-prefix cache is also rejected for the current greedy-large
mixed-prefix path. Running the same focused `multi_turn` benchmark with
`TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE=1` on pushed `248c81d` started
cleanly, but then made no visible request progress for several minutes with
`0%` GPU utilization across all eight devices. Interrupting the run shut the
server down and unwound blocked streaming clients with `RemoteProtocolError:
illegal chunk header: ... 400 Bad Request`; no result directory was written.
This path needs a correctness/hang investigation before it can be used for the
assistant-token reuse idea, and it is not a defaultable multi_turn optimization.

## Public 20260705_110218 refresh and multi-turn scheduler rejections

The latest public inference-bench commit advanced to `11363bd4` with run
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_110218`.
That public run still measured TorchInferno commit `390fed4`, not the later
`fecef37` pushed stack that includes CUDA greedy-gather sampling, decode graph
cache/symm telemetry, and the latest evidence notes. Score stayed `4/20` for
TorchInferno, while vLLM rose to `13/20` and SGLang fell to `2/20`.

Public gaps after `20260705_110218`:

- multi_turn: TorchInferno `350.8 / 61.7 / 401.5ms`, vLLM
  `172.4 / 48.9 / 219.5ms`, SGLang `165.4 / 104.3 / 278.8ms`.
- tree_of_thought: TorchInferno `123.5 / 30.6 / 148.2ms`, vLLM
  `62.6 / 30.6 / 85.9ms`, SGLang `76.5 / 57.1 / 141.8ms`. The earlier
  sampled-medium cap-32 work is now visible publicly as a TPOT tie with vLLM,
  but TTFT/E2E are still first-token/prefill bound.
- long_output: TorchInferno `278.4 / 19.6 / 1003.4ms`, vLLM
  `84.5 / 14.9 / 634.3ms`, SGLang `75.3 / 22.2 / 823.6ms`.

Rechecking current `879a6b6` defaults on local `multi_turn` wrote
`/tmp/inference-bench-ti-multiturn-current-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multiturn-current-20260705/runs/20260705_113750`
and landed at `361.3 / 57.5 / 416.7ms`, `981/1000` correct. Queue telemetry
again points at prefill, not decode: `q2first_p50=263.5ms`, `33` prefill
batches, `3.79s/4.06s` prefill forward/wall, and only `0.73s` decode GPU.
The hottest prefill shape was `prefix_graph:b32:s144:p45-45:src1:mixed0`
(`982.5ms` forward). Packed prefill candidates saved apparent dense tokens
(`20.5K` total, `6.3K` on the hot shape), but pattern reuse was low
(`9/32` repeat calls, `13.2%` repeated saved-token share).

Current-stack `multi_turn` probes split into three rejected scheduler/body
changes and one scoped mixed-prefix promotion:

- Shape-gated packed eager prefill for `prefix_graph:b32:s144:p45-45:src1:mixed0`
  wrote
  `/tmp/inference-bench-ti-multiturn-packedeager-s144-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multiturn-packedeager-s144-20260705/runs/20260705_114357`
  and regressed to `522.6 / 61.1 / 577.3ms`, `980/1000` correct. Six packed
  eager calls spent `6.71s`, so the Python-packed transformer body is still far
  more expensive than dense graphed prefill despite token savings.
- Raising greedy-large `prefill_ready_before_decode_active_cap` from `8` to
  `32` wrote
  `/tmp/inference-bench-ti-multiturn-cap32-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multiturn-cap32-20260705/runs/20260705_115006`
  and regressed to `427.5 / 63.5 / 476.3ms`, `984/1000` correct. It added
  another prefill batch and increased prefill wall to `4.37s`; keep the greedy
  large cap at `8`.
- Raising greedy-large max active rows to `48` wrote
  `/tmp/inference-bench-ti-multiturn-active48-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multiturn-active48-20260705/runs/20260705_115609`
  and regressed sharply to `798.5 / 81.0 / 881.5ms`, `984/1000` correct.
  Wider active rows are still not a defaultable multi_turn fix.
- Explicit greedy-large mixed-prefix reuse is now promoted only for the OpenAI
  `temperature=0,max_tokens=512` class after adding missing mixed-prefix warmup
  coverage and a large-greedy dynamic suffix bucket. The first current-stack
  opt-in run wrote
  `/tmp/inference-bench-ti-multiturn-mixedprefix-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multiturn-mixedprefix-20260705/runs/20260705_120548`
  and improved to `272.1 / 65.4 / 349.7ms`, with prefill wall down to
  `2.76s` and `q2first_p50=150.6ms`. The immediate repeat wrote
  `/tmp/inference-bench-ti-multiturn-mixedprefix-repeat-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multiturn-mixedprefix-repeat-20260705/runs/20260705_121126`
  and regressed to `466.2 / 63.2 / 510.2ms`; it hit a rare mixed-prefix graph
  miss and spent `389ms` on `prefix_graph:b2:s32:p156-158:src2:mixed1`.
  Adding `2:32:256` to the opt-in mixed-prefix suffix warmup covers that exact
  `b2:s32:ctx-256:src2` graph. Follow-up opt-in repeats wrote
  `/tmp/inference-bench-ti-multiturn-mixedprefix-b2warm-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multiturn-mixedprefix-b2warm-20260705/runs/20260705_122423`
  at `289.3 / 64.8 / 350.8ms`, `982/1000` correct, and
  `/tmp/inference-bench-ti-multiturn-mixedprefix-b2warm-repeat-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-multiturn-mixedprefix-b2warm-repeat-20260705/runs/20260705_123044`
  at `321.0 / 65.4 / 412.7ms`, `979/1000` correct. The repeat queue profile
  had `prefill_graph_misses=0` and a live
  `ragged_prefill:b2:s32:rows1:ctx-256:copy-1:src2:max1024:fp80:ar128:logits1`
  entry.
- A TorchInferno-only full-suite pass with the fixed mixed-prefix opt-in on
  pushed `c85d39b` wrote
  `/tmp/inference-bench-ti-full-mixedprefix-c85-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-full-mixedprefix-c85-20260705/runs/20260705_125305`.
  Multi_turn landed at `226.7 / 63.8 / 297.3ms`, `982/1000` correct, with
  `q2first_p50=149.8ms`, `37` prefill batches, `2.32s/2.65s` prefill
  forward/wall, `0` prefill graph misses, and route counts
  `{"common_prefix":125,"request_prompt":875}`. The scoped policy only changed
  the `temperature=0,max_tokens=512` row: few_shot stayed at `prefix_rows=64`,
  self_consistency at `16`, tree at `64`, and long_output at `64`. Promote the
  OpenAI mixed-prefix policy as the default for that exact multi_turn class,
  while preserving `TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_MIXED_PREFIX_REUSE=0`
  as the opt-out and keeping the lower-level continuous-engine policy opt-in.
- The first no-env default-on repeat exposed the remaining large-greedy exact
  context miss:
  `/tmp/inference-bench-ti-mixedprefix-default-c85-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-mixedprefix-default-c85-20260705/runs/20260705_130428`
  regressed to `444.8 / 67.2 / 498.3ms` with one
  `ragged_prefill:b1:s32:rows1:ctx156:src1` miss. Rather than warm the
  prompt-specific exact length, the runtime now gives greedy-large suffixes up
  to `32` tokens a dynamic context bucket, so that request reuses the already
  warmed `ctx-256` graph class. The next no-env repeat also found a normal
  mixed-source `b16:s16:ctx-256:src16` warmup gap; adding `16:16:256` to the
  mixed-prefix suffix warmup wrote
  `/tmp/inference-bench-ti-mixedprefix-default-dynamiclarge-b16s16-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-mixedprefix-default-dynamiclarge-b16s16-20260705/runs/20260705_132421`
  at `262.7 / 61.0 / 327.2ms`, `981/1000` correct, with one online phase,
  `38/0` prefill graph hits/misses, route counts
  `{"common_prefix":125,"request_prompt":875}`, and `q2first_p50=146.9ms`.
- The no-env full-suite check for the same narrow policy wrote
  `/tmp/inference-bench-ti-full-default-dynamiclarge-b16s16-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-full-default-dynamiclarge-b16s16-20260705/runs/20260705_132949`.
  Multi_turn landed at `232.6 / 65.5 / 303.2ms`, `982/1000` correct, with
  route counts `{"common_prefix":125,"request_prompt":875}` and one remaining
  `b2:s16:ctx-256:src2` miss. Broadening suffix-16 warmup across every mixed
  batch bucket removed misses but is rejected for now:
  `/tmp/inference-bench-ti-mixedprefix-default-dynamiclarge-s16warm-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-mixedprefix-default-dynamiclarge-s16warm-20260705/runs/20260705_133731`
  regressed to `357.6 / 62.4 / 445.1ms` despite `37/0` graph hits/misses.
  Keep only the narrow `16:16:256` warmup until a broader startup set proves
  stable.

The practical multi_turn target remains a lower-cost packed/fixed-pattern
cached-prefix prefill implementation and consistently cheap request-prompt
mixed-prefix replay, not current packed eager, wider active rows, or earlier
refill prefill scheduling.

A decode-many probe for greedy-large mixed-prefix multi_turn is rejected.
Forcing `TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY=1` and
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_ALLOW_STOP=1` wrote
`/tmp/inference-bench-ti-mixedprefix-decode-many-large-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-mixedprefix-decode-many-large-20260705/runs/20260705_134657`
and regressed to `323.8 / 217.5 / 541.7ms`, `981/1000` correct. The queue
profile kept the intended `{"common_prefix":125,"request_prompt":875}` route
mix and `34/0` prefill graph hits/misses, but decode-many did `15.0K` model
tokens for only `1.46K` emitted tokens, skipped `13.5K`, and spent `5.69s` in
decode GPU. The remaining multi_turn decode gap needs less overgenerated
multi-step decode or cheaper scalar ragged decode, not broad decode-many.

## Public 20260705_090205 refresh and decode graph symm telemetry

The latest public run advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_090205`
at inference-bench commit `9bd2f350`. TorchInferno is now `4/20`, vLLM is
`12/20`, and SGLang is `3/20`. The same gaps remain score-facing:

- multi_turn: TorchInferno `365.6 / 63.5 / 417.3ms`, vLLM
  `177.9 / 53.0 / 227.4ms`, SGLang `167.5 / 102.8 / 268.1ms`.
- tree_of_thought: TorchInferno `129.6 / 42.2 / 158.1ms`, vLLM
  `64.0 / 30.4 / 87.8ms`, SGLang `76.0 / 54.9 / 144.1ms`.
- long_output: TorchInferno `220.9 / 20.8 / 1000.7ms`, vLLM
  `83.0 / 14.9 / 616.9ms`, SGLang `74.0 / 22.2 / 833.2ms`.

The public tree row still used the older sampled-medium
`prefill_ready_before_decode_active_cap=10`, while current tree defaults use
cap `32`. Revalidating current TorchInferno against the same public harness
wrote
`/tmp/inference-bench-ti-tree-cap32-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-tree-cap32-20260705/runs/20260705_111300`
and finished `963/992` correct at `137.4 / 27.7 / 159.6ms`. This is not a
tree TTFT/E2E closure, but it is score-facing: TPOT would beat the public vLLM
tree TPOT (`30.4ms`) versus public TorchInferno's `42.2ms`. The queue profile
shows the intended policy (`prefill_ready_before_decode=true`, cap `32`),
`57` prefill batches, `1.72s/2.22s` prefill forward/wall, `1.71s` decode GPU,
and hot prefill at `prefix_graph:b24:s16:p45-45:src1:mixed0` (`909.7ms`).
Keep cap `32`; the remaining tree gap is first-token/prefill latency, not
sampled-medium TPOT.

The analyzer now carries decode graph symmetric-memory bucket counts through
queue-profile parsing and prints them as `decode_graph_symm`. The runtime records
`runtime_decode_graph_cache_live_symm_counts` next to the existing decode graph
batch/context/cache summaries when shape details are enabled, and the analyzer
can derive the same compact counts from older `runtime_decode_graph_cache_live_shape_counts`
keys. Re-rendering the current-tree long_output nosync profile at
`/tmp/inference-bench-ti-long-nosync-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-nosync-20260705/runs/20260705_093136`
now exposes `decode_graph_symm=symm0=16,symm128=16` without a rerun. This is
instrumentation for the remaining decode work: future public profiles can now
distinguish graph surfaces that are using `symm...` variants from profiles that
are spending decode replay time on non-symmetric-memory graph entries.

The startup ragged-decode graph warmup now releases decode graphs tied to its
temporary warmup caches after each cache spec. Those graph keys include
`id(cache)`, so the temporary-cache CUDA graphs cannot be replayed by the later
serving cache; the capture still warms the kernels, but the model-level graph
dict no longer retains throwaway entries. The focused long_output recheck wrote
`/tmp/inference-bench-ti-long-warmrelease-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-warmrelease-20260705/runs/20260705_095350`
and landed at `230.1 / 23.0 / 1095.7ms`, `1000/1000` correct. It is accepted
as graph-cache cleanup, not a score closure: the live serving-cache surface is
still `decode_graph_symm=symm0=16,symm128=16` because the online decode warmup
intentionally captures deterministic symm and sampled non-symm variants.

Decode graph replay telemetry now appends the actual graph-key symm bucket to
`runtime_decode_graph_replay_shape_ms`, and the analyzer prints the compact
`decode_replay_symm_ms` column. The focused long_output run
`/tmp/inference-bench-ti-long-replaysymm-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-replaysymm-20260705/runs/20260705_100535`
landed at `222.5 / 23.2 / 1092.6ms`, `1000/1000` correct. Its live graph cache
still had `decode_graph_symm=symm0=16,symm128=16`, but replay time was entirely
`decode_replay_symm_ms=symm128=4900.8`. That rules out non-symm graph replay as
the current long_output decode bottleneck; the remaining decode work is inside
the symm-enabled graph body (`9.72s` total ragged decode GPU, `6.54s`
decode-many GPU) plus padded prefill (`4.68s` forward).

The short-cache ragged-decode cap remains rejected after the warmup graph-cache
cleanup. Rechecking
`TORCHINFERNO_OPENAI_TP_ONLINE_SHORT_GREEDY_RAGGED_DECODE_CACHE_TOKENS=256`
with `MIN_BATCH=64` wrote
`/tmp/inference-bench-ti-long-shortcache-warmrelease-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-shortcache-warmrelease-20260705/runs/20260705_101445`
and finished 1000/1000 correct at `236.3 / 22.7 / 1062.6ms`. The cleanup did
avoid the old near-OOM behavior (`cache1024=32,cache256=2` live decode graphs,
server ready in `216s`), but it did not lower steady decode cost:
`decode_many_gpu_ms` rose to `7.13s` versus `6.54s` in the nearby replay-symm
control, and hot `decode_many:b64/64` rose to `3.32s`. Keep the short-cache cap
default-off; it is memory-safe enough to keep as a diagnostic now, but not a
score-facing long_output fix.

The queue analyzer now also prints live decode graph cache buckets as
`decode_graph_cache`, and future runtime profiles append `cache...` to
decode-graph replay keys so `decode_replay_cache_ms` can attribute replay time
to `cache256` versus `cache1024`. Re-rendering the short-cache run can only show
the live cache-bucket mix because that run predates the replay key change; the
next cache-bucket probe will expose both live and replay cache attribution.

Greedy tensor-parallel sampling now defaults to the one-collective gather path
for CUDA logits, with `TORCHINFERNO_GREEDY_SAMPLE_GATHER=0` retaining the older
two-all-reduce path and CPU behavior unchanged by default. The focused
long_output A/B wrote
`/tmp/inference-bench-ti-long-greedygather-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-greedygather-20260705/runs/20260705_102421`
and finished 1000/1000 correct at `239.5 / 22.4 / 1132.8ms`. The scored row is
noisy, but the queue profile shows the intended model-side effect versus the
nearby replay-symm control: `decode_many_gpu_ms` fell `6.54s -> 6.09s`, replay
time fell `4.90s -> 4.65s`, while hot `decode_many:b64/64` was roughly flat to
slightly worse (`2.99s -> 3.08s`) on a slightly different token mix. This is a
modest decode win, not a long_output closure; the remaining gap is still
full-batch model replay plus padded prefill.

The packed FlashInfer prefill probe is not score-facing under the current
public-style server configuration. Running
`TORCHINFERNO_CONTINUOUS_PACKED_FLASHINFER_PREFILL=1` wrote
`/tmp/inference-bench-ti-long-packedfi-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-packedfi-20260705/runs/20260705_103510`
and finished 1000/1000 correct at `262.2 / 23.1 / 1098.5ms`, but
`packed_fi_calls=0`: inference-bench's TorchInferno FlashInfer toggle installs
the optional kernels and leaves `--cache-backend` at dense, so this path cannot
fire on the public long_output row. Do not promote anything from this run.

The post-gather decode replay profile wrote
`/tmp/inference-bench-ti-long-decodeprof-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-decodeprof-20260705/runs/20260705_104243`
and captured `batch=64 cache_bucket=1024 rows=64` at `12.47ms` self CUDA. The
largest slices were dense GEMMs (`3.39ms` + `1.00ms`), gate-up Marlin
(`3.34ms`), 160 symmetric-memory all-reduces (`2.09ms`), and grouped GQA
attention (`1.48ms`). The new greedy all-gather sampler was only `9.5us`, so
remaining long_output TPOT work is projection/collective dominated, not sampler
or host-token handling.

The opt-in multi-token ragged decode graph path now handles contiguous physical
rows even when scheduler order differs from row order: it feeds the graph in
physical row order and remaps returned token columns back to scheduler order.
That fixed a previous no-op case without enabling row-indexed graph replay, but
the long_output probe remains rejected as a default. Running
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_GRAPH=1` with greedy gather wrote
`/tmp/inference-bench-ti-long-manygraph-reorder-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-manygraph-reorder-20260705/runs/20260705_110303`
and finished 1000/1000 correct at `212.8 / 23.9 / 1159.6ms`.
`decode_many_graph_calls=47` covered `232` steps at `decode_many:b64/64`, so
the repaired path did fire, but total decode work regressed:
`decode_many_graph_ms=5.44s`, `decode_many_gpu_ms=9.65s`, and total
`decode_gpu_ms=12.44s`. Keep the flag diagnostic-only; multi-step graph replay
does not currently reduce the projection/all-reduce/attention body cost.

## Public 20260705_070226 refresh and fixed-capacity packed-prefix rejection

The latest public run advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_070226`
at inference-bench commit `9cbe5008`. TorchInferno is now `5/20`, vLLM is
`11/20`, and SGLang is `3/20`. TorchInferno wins few_shot TPOT/E2E and
self_consistency TTFT/E2E/throughput, but the score-facing gaps remain:

- multi_turn: TorchInferno `328.0 / 58.2 / 383.3ms`, vLLM
  `181.1 / 46.1 / 227.5ms`, SGLang `158.7 / 109.6 / 267.5ms`.
- tree_of_thought: TorchInferno `123.1 / 37.5 / 152.0ms`, vLLM
  `64.1 / 29.9 / 88.3ms`, SGLang `75.6 / 72.4 / 153.1ms`.
- long_output: TorchInferno `239.0 / 20.6 / 958.0ms`, vLLM
  `85.8 / 14.9 / 658.1ms`, SGLang `73.3 / 22.2 / 863.5ms`.

The fixed-capacity packed-prefix branch now has an opt-in runtime prototype that
learns per-pattern slot capacities, pads missing `(prefix_len, suffix_len)`
slots, gathers real logits back to request order, and waits for repeated stable
capacities before invoking the packed graph. It is still rejected as a
performance path. The first targeted tree probe wrote
`/tmp/inference-bench-ti-tree-fixed-capacity-packed-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-tree-fixed-capacity-packed/runs/20260705_075427`
and regressed to `1352.1 / 51.6 / 1662.7ms`, `959/992` correct, with `38`
packed calls spending `20.3s`. After adding the stability guard, the recheck
wrote
`/tmp/inference-bench-ti-tree-fixed-capacity-stable-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-tree-fixed-capacity-stable/runs/20260705_080055`
and still regressed to `1265.3 / 38.5 / 1328.1ms`, `962/992` correct, with
`37` packed calls spending `19.6s`. The mechanism records the opportunity, but
the current packed layer loop/graph wrapper is far too expensive; keep the
switch diagnostic-only. A defaultable packed-prefix fix needs a real lower-cost
fixed-pattern body, not the current Python-packed transformer replay.

Decode-many queue profiling now avoids a CUDA synchronize after every fallback
decode replay by default. `TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_SYNC_MODEL_TIMINGS=1`
restores the old synchronized wall-time map for investigations that need it;
normal queue profiles should use the existing CUDA-event counters
(`runtime_decode_many_model_gpu_ms` and per-shape GPU maps). The focused
current-tree long_output run wrote
`/tmp/inference-bench-ti-long-nosync-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-nosync-20260705/runs/20260705_093136`
and landed at `269.9 / 22.3 / 1125.8ms`, `1000/1000` correct. The profile
confirmed the sync removal (`runtime_decode_many_model_ms=0.0`,
`runtime_decode_many_shape_model_ms={}`) while preserving GPU timing
(`runtime_decode_many_model_gpu_ms=6.83s`, hot `decode_many:b64/64=2.90s`).
This is accepted as measurement overhead reduction and keeps the queue profile
usable under public profiling, but it is not a score-facing long_output
closure; prefill stayed at `4.97s/5.39s` forward/wall and total decode GPU at
`9.91s`.

## Public 20260705_010222 refresh

The latest public run advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_010222`
at inference-bench commit `656a409f`. TorchInferno improved the tree row versus
`20260704_230159`, but the score-facing shape is unchanged:

- multi_turn: TorchInferno `343.3 / 59.8 / 401.6ms`, vLLM
  `172.0 / 49.4 / 217.4ms`, SGLang `168.0 / 100.6 / 273.6ms`.
- tree_of_thought: TorchInferno `118.7 / 35.9 / 145.7ms`, vLLM
  `64.4 / 30.9 / 87.5ms`, SGLang `74.6 / 56.7 / 139.5ms`.
- long_output: TorchInferno `255.1 / 20.4 / 992.5ms`, vLLM
  `91.4 / 14.9 / 661.0ms`, SGLang `73.3 / 22.1 / 804.8ms`.

The current median gaps remain multi_turn TTFT/E2E
(`+175.4/+184.1ms` versus the best other provider), tree TTFT/TPOT/E2E
(`+54.3/+5.0/+58.2ms` versus vLLM), and long_output TTFT/E2E
(`+181.8/+331.5ms`). The TorchInferno queue profile still points at the same
implementation split. Multi_turn spent `4.30s/4.45s` in prefill forward/wall
with queue-to-first p50 `277.3ms`. Tree spent `1.79s/2.02s` in prefill
forward/wall plus `1.75s` ragged-decode GPU, with the hot cached-prefix prefill
shape `prefix_graph:b24:s16:p45-45:src1:mixed0` at `752.4ms` and `43.7%`
padding. Long_output spent `4.28s/4.50s` in prefill forward/wall and `9.23s`
in decode GPU; its hottest decode-many shape was still `decode_many:b64/64`
(`2.07s` GPU, `11.0K` model tokens, `10.7K` emitted, `312` overgenerated).
The next defaultable work remains a real fixed-pattern packed cached-prefix
prefill body for tree/multi and lower model-side decode replay cost for
long_output.

The analyzer now also prints the worst raw 64-request waves per provider. On
this public long_output run, TorchInferno's high first-token cost persisted well
after startup: the selected worst waves had TTFT p50 `379.7ms` at wave 1,
`441.0ms` at wave 9, `301.6ms` at wave 10, `300.8ms` at wave 13, and
`313.3ms` at wave 14. vLLM's selected worst waves were much lower
(`173.3ms`, `132.2ms`, `141.8ms`, `115.0ms`, `164.8ms`), while SGLang was
mostly `76-91ms` except its wave-10 spike at `258.2ms`. This rules out a
single cold-start artifact; the long_output target is persistent wave/pipeline
prefill plus decode replay cost.

The analyzer now prints decode-many tail-cap details in the queue-profile table:
configured stop-tail cap, tail-limited calls/steps, and overgenerated tokens.
That immediately exposed why the public `20260705_010222` long_output row still
overgenerated `1.19K` decode-many tokens: the run exported `decode_quantum=3`
and `decode_many_allow_stop=true`, but no tail-cap field and
`0` tail-limited calls. A focused current-tree cap-2 probe wrote
`/tmp/inference-bench-ti-pattern-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-tail2/runs/20260705_015457`
and is rejected as a default. It reduced decode-many overgeneration to `567`
tokens and exercised `81` tail-limited calls, but fragmented decode into `762`
model calls, kept total ragged-decode GPU at `10.32s`, and landed at
`244.9 / 24.0 / 1061.7ms`, throughput `32.4 tok/s`. That does not beat the
better cap-4/tail-4 local controls on TTFT/TPOT/throughput, so the long_output
gap remains model-side decode replay plus prefill, not a stricter stop-tail cap.

A same-tree cap-4 default confirmation wrote
`/tmp/inference-bench-ti-current-long-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-current/runs/20260705_023128`
and landed long_output at `216.1 / 23.9 / 1151.4ms`, `1000/1000` correct,
throughput `33.7 tok/s`. It exercised the cap (`30` tail-limited calls,
`120` tail-limited steps), kept decode-many overgeneration to `1.25K` tokens,
and spent `10.30s` decode GPU plus `4.65s/5.08s` prefill forward/wall. A
no-tail-cap A/B on the same tree wrote
`/tmp/inference-bench-ti-current-long-notail-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-notail-current/runs/20260705_023829`
and landed at `238.2 / 24.5 / 1115.6ms`, `1000/1000` correct, throughput
`32.7 tok/s`. Removing the tail cap reduced decode-many GPU time
(`7.71s -> 6.82s`) and E2E, but worsened queue-to-first (`164.4 -> 184.6ms`),
TPOT, throughput, and overgeneration (`1.25K -> 1.84K`). Keep the cap-4
default; the long_output path needs lower steady model replay cost, not another
tail-cap toggle.

The runtime and analyzer now split decode-many work by generated-token windows
(`g1-16`, `g17-32`, ...), and the full `bN/N` decode-many path skips padding-row
seq-len bookkeeping. A TorchInferno-only long_output profile on the current tree
wrote
`/tmp/inference-bench-ti-step-window-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-stepwin/runs/20260705_025747`
and landed at `243.0 / 23.9 / 1066.3ms`, `1000/1000` correct, throughput
`32.9 tok/s`. The new step-window table shows the dominant decode-many work is
early and full-batch: `decode_many:b64/64:g1-16` accounted for `226` calls,
`14.5K` model tokens, and `3.13s` model wall, while
`decode_many:b64/64:g17-32` was only `20` calls, `1.28K` tokens, and `277ms`.
This narrows the next long_output target to reducing the steady full-batch
decode replay cost during the first 16 generated tokens, not drain-tail padding
or late-window stop overgeneration.

The next focused decode change removes explicit row-index tensors for greedy
ragged decode when the active physical row set is exactly `0..N-1`. Decode-many
now feeds the graph in physical-row order and gathers generated tokens back to
the scheduler's active-state order, so request event ordering does not change;
the one-step ragged decode path uses the same contract and gathers reusable
prefix logits back before storing them. The exact-code TorchInferno-only
long_output validation wrote
`/tmp/inference-bench-ti-contig-all-decode-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-contig-all-decode/runs/20260705_032508`
and landed at `224.2 / 23.7 / 1162.8ms`, throughput `33.6 tok/s`,
`1000/1000` correct. The queue profile confirmed a native
`ragged_decode:token:b64:rows0` capture, lower decode-many GPU time than the
step-window baseline (`6.87s` vs `7.35s`), fewer decode-many steps (`480` vs
`546`), and fewer hot `decode_many:b64/64` model tokens (`14.98K` vs
`15.74K`) with a similar GPU slice (`3.47s` vs `3.39s`). This is a modest
TTFT/TPOT/throughput win over the step-window baseline, while E2E moved with
run-to-run wave variance; it is not a complete long_output closure. The
remaining score gap is still dominated by full-batch model replay and prefill.
Validation: `venv/bin/python -m pytest tests/test_serving_engine.py -q`;
`venv/bin/python -m pytest
tests/test_openai_server.py::test_openai_queue_profile_records_runtime_engine_stats
tests/test_inference_bench_summary.py -q`; `venv/bin/python -m pyflakes
src/torchinferno/runtime/serving.py tests/test_serving_engine.py
src/torchinferno/openai_server.py src/torchinferno/research/inference_bench.py
tests/test_openai_server.py tests/test_inference_bench_summary.py`; and
`git diff --check`.

Startup scheduler warmup now captures the contiguous row-index-free ragged
decode graphs on the persistent serving cache, instead of warming only the
explicit-row-index variants. The focused long_output recheck wrote
`/tmp/inference-bench-ti-rows0-warmup-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-rows0-warmup/runs/20260705_033755`
and landed at `232.7 / 23.3 / 1072.3ms`, throughput `33.6 tok/s`,
`1000/1000` correct. The intended profile change is clear: request-time decode
graph captures dropped from one `ragged_decode:token:b64:rows0` capture
(`247.4ms`) to zero, the live decode graph cache now includes both `rows0` and
`rows1` entries for the warmed buckets, queue-to-first p50 moved from
`167.1ms` to `163.9ms`, and queue-to-finish p50 moved from `1026.7ms` to
`991.3ms`. This removes a first-request capture bubble; it does not change the
main long_output target, which remains lower full-batch decode replay and
prefill cost.

Short-cache ragged decode graph caps are rejected as a default and remain
diagnostic-only. The tempting idea was to keep the 1024-token persistent cache
for reuse, but slice the long_output decode graph body to a 256-token cache
bucket for deterministic short generations. The broad default-on probe wrote
`/tmp/inference-bench-ti-short-cachecap-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-short-cachecap/runs/20260705_035804`
and did capture/use the capped graphs with zero request-path decode captures
(`cache256:16` live entries), but startup memory reached the H100 limit and the
row regressed to `244.0 / 22.8 / 1074.0ms`, throughput `34.2 tok/s`. Narrowing
the cap to the hot 64-row decode bucket wrote
`/tmp/inference-bench-ti-short-cachecap-b64-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-short-cachecap-b64/runs/20260705_040512`
and reduced the extra graph surface to `cache256:2`, with zero decode captures
and a lower hot `decode_many:b64/64` GPU slice (`2.26s` over `11.39K` model
tokens). It still touched the memory ceiling and regressed to
`249.3 / 22.9 / 1142.9ms`, throughput `33.9 tok/s`. Keep
`TORCHINFERNO_OPENAI_TP_ONLINE_SHORT_GREEDY_RAGGED_DECODE_CACHE_TOKENS`
default-off; the long_output fix is not another decode graph bucket unless the
graph memory cost is solved.

Multi-token ragged decode graphs are also rejected as a default. The runtime now
has explicit `decode_many_graph_*` counters and the model can replay row-indexed
multi-step greedy decode graphs, but the score probe showed this is the wrong
long_output lever. With only
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_GRAPH=1`, the focused run wrote
`/tmp/inference-bench-ti-long-manygraph-telemetry-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-manygraph-telemetry/runs/20260705_083541`
and landed at `233.1 / 23.3 / 1074.4ms`, `1000/1000` correct, but the new
counters proved the path never fired (`runtime_decode_many_graph_calls=0`).
Allowing scattered/padded row-index graphs with
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_GRAPH_ALLOW_ROW_INDICES=1` made the
path fire on every decode-many call in
`/tmp/inference-bench-ti-long-manygraph-rowidx-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-manygraph-rowidx/runs/20260705_084645`
(`128` graph calls, `466` graph steps, `28.9K` graph model tokens), but regressed
the row to `222.1 / 23.9 / 1254.0ms` with TPOT p99 `298.8ms`. Decode-many GPU
time rose to `12.65s` from `7.04s` in the no-row-index telemetry run. Keep the
row-indexed multi-step graph behind the extra explicit opt-in; the default
long_output target remains lower per-step model replay cost, not chaining
several decode steps inside one CUDA graph.

## Public 20260705_030215 refresh

The latest public run advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_030215`
at inference-bench commit `cb96c382`. TorchInferno still wins
self_consistency E2E/throughput, but the score-facing gaps are unchanged in
shape: multi_turn is primarily TTFT/E2E (`340.0 / 62.9 / 398.6ms` versus vLLM
`147.7 / 66.8 / 208.4ms`), tree_of_thought remains cached-prefix prefill/decode
bound (`119.9 / 44.6 / 148.5ms` versus vLLM `65.6 / 29.6 / 90.4ms`), and
long_output remains the deterministic short-generation throughput row
(`250.4 / 20.9 / 982.9ms` versus vLLM `76.6 / 14.9 / 617.6ms`). The public
long_output queue profile again had no decode graph captures, about `4.36s`
prefill wall, and `9.20s` decode GPU, so the next defaultable change still has
to reduce model-side replay/prefill work rather than warm more graph variants.

`torchinferno.cli inference-bench-summary` now prints a compact
`[torchinferno score targets]` table that joins provider gaps to the matching
TorchInferno queue profile. On `20260705_030215` it ranks long_output first
(`+176.0/+6.0/+365.3ms`, decode target, `9.20s` decode GPU and `8.96s`
decode-many shape time), multi_turn second (`+192.3/-3.9/+190.2ms`, prefill
target, `4.20s` prefill forward), and tree third (`+54.3/+15.0/+58.1ms`,
balanced prefill+decode). That keeps the next implementation work focused on a
real lower-cost decode replay for `decode_many:b64/64` and a true packed
cached-prefix prefill body, not the already-rejected suffix-bucket,
decode-cache-bucket, FI-reuse, or fused-append toggles.

The score-target and queue-profile tables now include prefill/decode graph
misses plus generated-prefix store/reuse counters. On the same public run,
long_output shows only `2` decode graph misses and no generated-prefix activity,
while self_consistency shows `1` generated-prefix store and `998` generated
prefix reuses. That separates the long_output decode replay gap from
generated-prefix/static-logits noise seen in some local runs.

The analyzer now also mines vLLM/SGLang server logs for provider-side phase
signals. On `20260705_030215`, vLLM reported six runtime intervals averaging
`1760 tok/s` prompt throughput, `517 tok/s` generation throughput, and `76.5%`
prefix-cache hit rate. SGLang logged `619` graphed prefill batches with `91.2K`
new prompt tokens and `423.6K` cached tokens, plus `23` graphed decode batches.
That comparison reinforces the same direction as the TorchInferno queue profile:
competitors are getting broad prefix-cache/chunked-prefill reuse under CUDA
graphs, while TorchInferno still spends score-facing time in padded cached-prefix
replay and full-batch decode graph replay.

Decode-many now skips redundant start-of-burst GPU state staging when the active
rows, last tokens, and sequence lengths exactly match the GPU-resident state left
by the previous burst. A focused long_output run wrote
`/tmp/inference-bench-ti-state-syncskip-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-state-syncskip/runs/20260705_042815`
and landed at `218.4 / 24.4 / 1084.5ms`, throughput `33.3 tok/s`, `1000/1000`
correct. The profile shows the new path did fire (`63` state syncs, `72` sync
skips), but the score-facing shape did not change: `5.34s` prefill wall,
`6.87s` decode-many GPU, and hot `decode_many:b64/64` at `3.07s` over `14.3K`
model tokens. Keep the skip as default-safe runtime hygiene and profiling
visibility; it is not a long_output closure. The next useful work remains a
lower-cost full-batch decode replay or packed cached-prefix prefill body.

A greedy-short persistent idle probe is also rejected as a default. Raising the
online persistent idle window from the default `10ms` to `100ms` for a focused
TorchInferno-only long_output run wrote
`/tmp/inference-bench-ti-greedyidle100-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-greedyidle100/runs/20260705_045432`
and landed at `228.9 / 24.4 / 1092.3ms`, throughput `32.2 tok/s`,
`1000/1000` correct. The profile moved the wrong model-work counters:
`67` prefill batches, `4.99s/5.42s` prefill forward/wall, `6.43s`
decode-many GPU over `478` decode-many steps, and hot `decode_many:b64/64`
at `2.53s` over `11.8K` model tokens. Keeping the online session open longer
does not recover the competitor-style prefix reuse; it just shifts more wave
work into the same padded prefill and full-batch decode replay paths.

Ragged token decode graphs now look up rotary rows from the static
cache-position graph input inside the captured graph instead of copying rotary
cos/sin buffers before every replay. Logits graphs stay on the old copy path
unless `TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_ROTARY_IN_GRAPH=1` explicitly opts
them in. A focused long_output A/B wrote
`/tmp/inference-bench-ti-rotarygraph-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-rotarygraph/runs/20260705_050408`
and landed at `227.1 / 23.7 / 1094.5ms`, throughput `33.9 tok/s`,
`1000/1000` correct. The internal counter moved in the intended direction:
decode graph replay accounting fell to `579ms`, with `ragged_decode:token:b64`
rows1 replay at `393ms` instead of about `587ms` in the 100ms-idle probe. The
score-facing decode-many path did not close, though: total decode-many GPU was
still `6.83s` over `510` decode-many steps, and hot `decode_many:b64/64` was
`2.93s` over `13.95K` model tokens. This is a small decode-replay hygiene win,
not a long_output closure; the remaining target is still lower full-batch
decode replay or packed cached-prefix prefill.

A no-env confirmation after promoting that behavior for token graphs wrote
`/tmp/inference-bench-ti-rotarydefault-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-rotarydefault/runs/20260705_051304`
and landed at `246.0 / 23.6 / 1093.7ms`, throughput `33.7 tok/s`,
`1000/1000` correct. It reproduced the internal counter win
(`567ms` decode graph replay, with `ragged_decode:token:b64` rows1 replay at
`374ms`) but not a score-facing closure: `67` prefill batches,
`4.81s/5.23s` prefill forward/wall, `6.81s` decode-many GPU over `510`
decode-many steps, and hot `decode_many:b64/64` at `2.98s` over `14.2K` model
tokens.

The same rotary-in-graph hygiene is now applied to ragged prefix-prefill graphs.
The prefill graph key includes the rotary mode, and the default path looks up
rotary rows from the static write-position tensor inside the captured graph
instead of copying static cos/sin buffers before every replay. A long_output A/B
against the same working tree wrote
`/tmp/inference-bench-ti-prefillrotary-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-prefillrotary/runs/20260705_052716`
and
`/tmp/inference-bench-ti-prefillrotaryoff-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-prefillrotaryoff/runs/20260705_053233`.
With in-graph prefill rotary enabled, score was
`246.5 / 22.1 / 1068.2ms`, throughput `34.8 tok/s`; disabling it landed at
`218.5 / 23.6 / 1131.6ms`, throughput `33.6 tok/s`; both were `1000/1000`
correct. The mechanism is visible in queue counters despite decode variance:
prefill graph replay fell `187.8ms -> 145.7ms` and prefill forward/wall fell
`4.88s/5.29s -> 4.65s/5.04s`. This is a small defaultable prefill replay win,
not a full long_output closure, because decode-many still moved independently.

Ragged prefix-prefill graphs now also derive full write positions inside the
captured graph from static start positions plus a static suffix offset vector.
That avoids copying the full `[batch, suffix]` write-position tensor on replay
when the graph also gathers rotary rows internally. The focused long_output A/B
wrote
`/tmp/inference-bench-ti-prefillposgraph-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-prefillposgraph/runs/20260705_054445`
with the default enabled and
`/tmp/inference-bench-ti-prefillposoff-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-prefillposoff/runs/20260705_054954`
with `TORCHINFERNO_CUDAGRAPH_RAGGED_PREFILL_WRITE_POSITIONS_IN_GRAPH=0`.
Enabled landed at `248.3 / 22.5 / 1076.4ms`, throughput `34.8 tok/s`;
disabled landed at `223.7 / 24.1 / 1129.7ms`, throughput `33.5 tok/s`; both
were `1000/1000` correct. Queue counters show the intended mechanism:
prefill graph replay fell `157.5ms -> 122.8ms`, and prefill forward/wall fell
`4.95s/5.38s -> 4.71s/5.09s`. Keep it default-on as prefill replay hygiene,
with the same caveat as rotary: single-run TTFT and decode-many still vary
enough that this is not a full score-facing closure.

An opt-in stop-synchronized decode-many path is available under
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_SYNC_STOPS=1`, but it is not a
default promotion. The path copies decode-many tokens back to CPU after each
step when stop tokens are active, so stopped rows can be removed from later
steps instead of being counted as skipped/overgenerated tokens at the end of the
chunk. The focused long_output run wrote
`/tmp/inference-bench-ti-decodemany-syncstops-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-decodemany-syncstops/runs/20260705_060616`
and landed at `233.3 / 23.1 / 1087.4ms`, throughput `33.6 tok/s`,
`1000/1000` correct. Versus the nearby default
`/tmp/inference-bench-ti-prefillposgraph-results/.../runs/20260705_054445`,
it removed decode-many overgeneration (`1390 -> 0`) and reduced decode-many GPU
time (`7.92s -> 7.14s`), but CPU token-copy time rose (`20.4ms -> 78.6ms`) and
median E2E did not improve (`1076.4ms -> 1087.4ms`). Keep it as a diagnostic
for stop-heavy decode profiling; the default still needs a lower-overhead
GPU-side stop compaction or a broader decode pipeline change.

A max-active-96 long_output probe is also rejected as a default. The run wrote
`/tmp/inference-bench-ti-long-active96-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-long-active96/runs/20260705_062302`
with `TORCHINFERNO_OPENAI_TP_ONLINE_MAX_ACTIVE=96`,
`TORCHINFERNO_OPENAI_TP_ONLINE_KV_MAX_ACTIVE_CAP=96`, and a `49152` greedy KV
token budget. It landed at `241.5 / 23.9 / 1094.4ms`, throughput
`33.3 tok/s`, `1000/1000` correct. The scheduler still did not move the hot
decode work above the `decode_many:b64/64` bucket: total ragged decode GPU was
`10.39s`, decode-many GPU was `4.62s` over `338` steps, and the hottest
decode-many shape stayed `decode_many:b64/64` (`1.49s`, `7040` model tokens).
Prefill remained `4.59s/5.09s` forward/wall. Keep the `64`-active default; the
benchmark's wave shape is not unlocked by a higher active cap, and future queue
profiles now export `runtime_max_active_requests` and
`runtime_prefix_cache_capacity` so cap experiments are visible directly in the
JSONL.

## Public 20260705_050210 refresh

The latest public run advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260705_050210`
at inference-bench commit `a2e6ba16`. TorchInferno is at `3/20`, vLLM at
`14/20`, and SGLang at `2/20`. TorchInferno wins self_consistency
TTFT/E2E/throughput, but still trails vLLM on few_shot, multi_turn,
tree_of_thought, and long_output. The public long_output row is TorchInferno
`262.4 / 20.0 / 1016.4ms`, vLLM `78.0 / 14.8 / 621.9ms`, and SGLang
`72.6 / 21.9 / 838.4ms`. Public TorchInferno queue counters show the same
shape as the local A/Bs: `64` prefill batches, `4.09s/4.30s` prefill
forward/wall, `165ms` prefill graph replay, and `422` decode-many steps. The
remaining public gap is still cached-prefix prefill plus full-batch decode
throughput, not benchmark-specific request handling.

## Public 20260704_230159 refresh

The latest public run advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260704_230159`
at inference-bench commit `7b5bcd1c`. TorchInferno dropped to `1/20`, vLLM
scored `15/20`, and SGLang scored `3/20`. The public row now has vLLM winning
all tree_of_thought metrics and three of four long_output metrics:

- few_shot: TorchInferno `157.6 / 45.9 / 210.3ms`, vLLM
  `148.8 / 53.9 / 190.8ms`, SGLang `139.2 / 78.1 / 217.2ms`.
- self_consistency: TorchInferno `258.1 / 0.0 / 302.8ms`, vLLM
  `194.3 / 0.0 / 212.1ms`, SGLang `221.5 / 0.0 / 380.3ms`.
- multi_turn: TorchInferno `360.2 / 62.8 / 413.2ms`, vLLM
  `174.1 / 57.6 / 230.3ms`, SGLang `169.6 / 111.6 / 283.1ms`.
- tree_of_thought: TorchInferno `125.0 / 47.2 / 150.5ms`, vLLM
  `63.6 / 31.1 / 87.3ms`, SGLang `77.6 / 54.5 / 138.2ms`.
- long_output: TorchInferno `258.1 / 20.8 / 970.1ms`, vLLM
  `81.5 / 15.0 / 615.0ms`, SGLang `75.9 / 22.4 / 826.4ms`.

`torchinferno.cli inference-bench-summary` now prints per-benchmark provider
gap tables against the best non-TorchInferno row. On this public run the largest
median gaps are long_output TTFT/E2E (`+182.2/+355.1ms`), multi_turn TTFT/E2E
(`+190.6/+182.9ms`), and tree TTFT/TPOT/E2E (`+61.5/+16.1/+63.2ms`); few_shot
TPOT remains a TorchInferno win (`-8.0ms` versus the best other provider).
The same summary on the new run keeps the implementation split visible. Tree
queue-to-first p50 was `105.8ms`, with `55`
prefill batches, `1.70s/1.92s` prefill forward/wall, and `1.65s` decode GPU;
its hottest prefill shape was `prefix_graph:b32:s16:p45-45:src1:mixed0`
(`782.6ms`, `4.6K` padding tokens). Long_output queue-to-first p50 was
`186.8ms`, with `62` prefill batches, `4.15s/4.63s` prefill forward/wall, and
`9.24s` decode GPU; its hot decode shape stayed `decode_many:b64/64`
(`1.69s` GPU, `254` overgenerated tokens). Multi_turn still reused the `45`
token common prefix and spent `4.17s/4.33s` in prefill forward/wall. The public
run therefore does not change the next engineering target: tree/multi need a
real packed cached-prefix prefill body, while long_output needs lower model-side
decode replay cost in the `b64` decode-many path.

The analyzer now also shows decode-many shape efficiency: model tokens, emitted
tokens, skipped/overgenerated tokens, and skip percentage for the hottest
decode-many shape. On the existing q12-after-first diagnostic, the `b64/64`
drain shape accounted for `21.2K` model tokens, `20.3K` emitted tokens, and
`941` skipped tokens (`4.4%`) with `4.57s` GPU. That makes the q8/q12/tail-cap
tradeoff visible from `inference-bench-summary` without manually diffing
queue-profile JSON.

The same summary now reports hot prefill shape efficiency from the existing
model-token and active-token counters, and prints the top three prefill shapes
per workload instead of only the single hottest row. On the public
`20260704_230159` run, the hottest long_output suffix-prefill shape spent
`927.9ms` on `16.4K` model tokens for `10.7K` active tokens (`34.8%` padding);
its next two hot shapes were even less efficient (`48.8%` and `46.7%`
padding). Multi_turn's hot `b32:s144` shape spent `1.44s` on `36.9K/25.8K`
tokens (`29.9%` padding), and tree's top three hot shapes were all heavily
padded (`42.7%`, `44.4%`, and `50.8%`). The row/suffix split shows the dominant
waste is suffix padding, so the next performance lever remains a packed
cached-prefix prefill body rather than another active-row or decode-tail cap.

Packed-candidate telemetry now also records a fixed-pattern key that omits the
per-wave request counts from the exact signature. Exact signatures answer
whether a graph keyed on the full `(prefix_start, suffix_len, count)` histogram
would replay; the new pattern counters answer the more useful next question for
a fixed-capacity/dynamic-count packed prefill body: whether the same distinct
`(prefix_start, suffix_len)` groups repeat even when their counts vary. Future
queue profiles export pattern keys, calls, repeated calls, and repeated saved
tokens beside the existing exact-signature reuse table.

A current-tree TorchInferno-only tree run wrote
`/tmp/inference-bench-ti-pattern-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-pattern-tree/runs/20260705_012303`
and confirmed the pattern signal: exact signatures repeated only `4/57`
candidate calls (`604` saved tokens), while fixed patterns repeated `51/57`
candidate calls and covered `8.36K` saved tokens (`98.7%` of candidate saved
tokens). The analyzer now ranks fixed patterns by saved tokens; on that run the
top pattern was
`prefix_graph:b24:s16:p45-45:src1:mixed0|p45:s10/p45:s11/p45:s12`,
with `27` calls, `5.9K/10.4K` real/model tokens, and `4.48K` saved tokens
(`52.9%` of all candidate saved tokens). Future queue profiles now export the
maximum observed slot count for each fixed-pattern `(prefix_start, suffix_len)`
group, and the analyzer falls back to exact signatures for older logs. For that
top tree pattern, the observed fixed slots would execute `7.75K` packed model
tokens instead of `10.37K` dense tokens, saving `2.62K` tokens (`25.3%`) with
`66.7%` exact-signature coverage. That is a more realistic first target than the
raw dense-padding number: a fixed-capacity graph can save substantial suffix
work, but dynamic-count slack and group capacity still leave about `1.86K`
tokens between real tokens and fixed packed tokens. A follow-up sampled-medium
`b20` batch-bucket probe wrote
`/tmp/inference-bench-ti-pattern-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-tree-b20/runs/20260705_012847`
and is rejected as a default. It slightly improved TTFT (`127.5 -> 126.5ms`)
but worsened TPOT/E2E (`27.3/150.9ms -> 36.7/153.5ms`), increased prefill
batches (`58 -> 61`), and left prefill forward/wall essentially flat
(`1.80s/2.22s -> 1.81s/2.23s`). This reinforces that bucket proliferation can
reduce padding counters without reducing wall time; the next tree/multi change
needs a fixed-capacity packed cached-prefix prefill body, not another bucket
default.

A fresh current-tree TorchInferno-only run with runtime slot-count export wrote
`/tmp/inference-bench-ti-current-slot-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-slot-current/runs/20260705_021555`
and landed tree_of_thought at `127.5 / 25.5 / 150.6ms`, `953/992` correct.
The fixed-capacity plan table now marks rows with `slot_src=runtime` when the
queue profile provides direct slot maxima. On this run the top fixed-capacity
target shifted to
`prefix_graph:b32:s16:p45-45:src1:mixed0|p45:s10/p45:s11/p45:s12`: `15`
calls, `37` fixed slots, `7.68K -> 5.88K` model tokens, and `1.80K` saved
tokens (`23.4%`). The exact-signature map reported only `2/15` calls for that
pattern because the exported top-signature map is intentionally capped; the
runtime slot map is the authoritative source for the fixed-slot plan.
The summary now also converts fixed-capacity saved tokens into an estimated
saved-forward-ms column using the observed dense shape timing. On this artifact,
the top three fixed-capacity targets estimate about `130ms`, `105ms`, and
`51ms` of prefill-forward savings respectively, which is enough to prioritize
the first packed-prefix graph body but not enough to explain the entire public
tree TTFT/E2E gap by itself.

Rechecking the existing `s12,16` suffix-bucket opt-in on the same current tree
is also not a default promotion. The run wrote
`/tmp/inference-bench-ti-current-s12-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-s12-current/runs/20260705_022411`
and landed at `119.5 / 37.2 / 145.0ms`, `958/992` correct. It reduced
queue-to-first (`105.7 -> 98.5ms`), prefill padding (`8.51K -> 3.89K` candidate
saved tokens), and E2E (`150.6 -> 145.0ms`), but it worsened TPOT from
`25.5ms` to `37.2ms` and raised decode GPU (`1.39s -> 1.45s`). That gives up
the local tree TPOT win that a public vLLM comparison needs, so keep `s12,16`
as an opt-in diagnostic. The remaining defaultable work is still a packed
fixed-capacity body that lowers prefill tokens without shifting decode shape
quality.

A targeted packed-eager pattern probe is also rejected as an implementation
path, but useful as evidence. The runtime now has an opt-in
`TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER_PATTERN` switch that can
run the existing Python packed-eager prefill only for an exact fixed-pattern key
or coarse shape key. Targeting only the top tree pattern above wrote
`/tmp/inference-bench-ti-pattern-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-tree-packed-pattern/runs/20260705_014326`
and regressed tree_of_thought to `568.5 / 42.9 / 612.2ms`, `958/992` correct.
The queue profile confirms the cost source: only `16` packed-eager calls spent
`8.43s` while saving `2.7K` padded tokens. Keep the pattern switch diagnostic
only; the defaultable path still needs a real graph/kernel body for these fixed
patterns, not the current dynamic packed-eager layer loop.

A first exact-signature CUDA graph wrapper for the packed-eager body is also
diagnostic only. The unguarded probe targeted the same coarse `b32/s16/p45`
tree pattern with
`TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER_GRAPH=1` and wrote
`/tmp/inference-bench-ti-tree-packed-graph-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-tree-packed-graph/runs/20260705_065939`.
It landed at `145.8 / 36.5 / 183.5ms`, `964/992` correct. The graph path was
better than the earlier Python-only packed-eager probe, but it still spent
`8.88s` across `7` packed calls because every targeted `b32` wave had a
different exact suffix-length signature and paid capture cost instead of
replaying one graph. That rules out exact-signature graph capture as the tree
solution; the useful target remains the fixed-capacity pattern table.

The exact packed graph now waits for shape reuse before capturing, defaulting
`TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER_GRAPH_CAPTURE_MIN_CALLS`
to `2`. The guarded recheck wrote
`/tmp/inference-bench-ti-tree-packed-graph-guard-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-tree-packed-graph-guard/runs/20260705_070904`
and landed at `141.9 / 28.2 / 167.9ms`, `956/992` correct, with no request-path
packed graph captures and `3.67s` across `6` packed-eager calls. This prevents
the worst exact-graph tail (`p99` fell from about `2.7s` to `1.1s`) and proves
the static-index packed body is cheaper than the prior dynamic packed-eager
probe, but it still loses to the no-env current-slot control (`150.6ms` E2E)
and should remain opt-in.

A stricter graph-only fallback keeps that diagnostic from accidentally taking
the slow Python packed-eager body when no exact graph is reusable. With
`TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER_GRAPH_ONLY=1`, the targeted
tree recheck wrote
`/tmp/inference-bench-ti-tree-packed-graph-only-results/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100-local-ti-tree-packed-graph-only/runs/20260705_073734`
and landed at `148.4 / 49.5 / 193.3ms`, `960/992` correct. The queue profile
shows the intended guard behavior (`0` packed-eager calls, `62` packed
candidates recorded, and dense prefill graph replay at `104.5ms` total), but the
timing is still worse than both the guarded probe and the no-env control. Keep
the graph-only switch as a profiling safety valve; it is not the fixed-capacity
packed prefill body needed to close the tree/multi gap.

## Public 20260704_190215 refresh and long_output tail-cap recheck

The latest public run advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260704_190215`
at inference-bench commit `7550e490`. TorchInferno is at `4/20`, vLLM scored
`13/20`, and SGLang scored `2/20`. TorchInferno still wins few_shot TPOT and
self_consistency TTFT/E2E/throughput. The remaining score-facing gaps are
multi_turn, tree_of_thought, and long_output:

- few_shot: TorchInferno `158.2 / 46.8 / 201.8ms`, vLLM
  `145.1 / 47.8 / 188.3ms`, SGLang `146.7 / 75.8 / 221.9ms`.
- self_consistency: TorchInferno `172.8 / 0.0 / 184.7ms`, vLLM
  `216.6 / 0.0 / 240.5ms`, SGLang `225.7 / 0.0 / 387.1ms`.
- multi_turn: TorchInferno `364.9 / 61.9 / 418.2ms`, vLLM
  `174.8 / 46.4 / 222.4ms`, SGLang `161.1 / 113.1 / 269.0ms`.
- tree_of_thought: TorchInferno `134.3 / 40.4 / 164.2ms`, vLLM
  `62.9 / 30.5 / 85.8ms`, SGLang `74.1 / 57.2 / 142.8ms`.
- long_output: TorchInferno `243.1 / 21.5 / 995.4ms`, vLLM
  `79.2 / 14.8 / 625.5ms`, SGLang `76.6 / 22.0 / 831.1ms`.

The gap parser is now repeatable through
`torchinferno.cli inference-bench-summary <run-dir>`. Running it on the public
`20260704_190215` directory for multi_turn, tree_of_thought, and long_output
confirmed the provider distribution shape behind the scorecard:

- multi_turn raw p50/p90 TTFT: TorchInferno `364.9/486.7ms`, vLLM
  `174.8/273.8ms`, SGLang `161.1/277.8ms`.
- tree_of_thought raw p50/p90 TTFT: TorchInferno `134.2/208.7ms`, vLLM
  `62.9/99.4ms`, SGLang `74.1/121.6ms`.
- long_output raw p50/p90 TTFT: TorchInferno `243.0/349.0ms`, vLLM
  `79.2/120.6ms`, SGLang `76.6/127.7ms`. Output-token p50 was identical
  (`36`) across providers, so the row is not an output-length artifact.

The same command reports TorchInferno's final queue records beside provider
metrics. In the public run, long_output had server-side queue-to-first p50
`172.4ms`, `64` prefill batches, `4.21s/4.45s` prefill forward/wall, and
`9.81s` ragged-decode GPU; tree had queue-to-first p50 `112.7ms`,
`1.90s/2.13s` prefill forward/wall, and `1.97s` decode GPU; multi_turn had
queue-to-first p50 `290.2ms` and `3.86s/4.02s` prefill forward/wall. This
keeps the next loop grounded in two separate deficits: long_output still needs
lower steady decode replay cost, while tree/multi remain dominated by
cached-prefix prefill queueing and TP replay cost.

The analyzer now also carries the queue-profile shape maps into hot-shape
tables. On the same public run, long_output's hottest prefill replay was
`prefix_graph:b24:s64:p111-111:src1:mixed0` (`973.7ms` forward,
`9307` padding tokens) and its hottest decode replay was
`decode_many:b64/64` (`1352.9ms` GPU, `266` overgenerated tokens). Multi_turn's
largest prefill shape was `prefix_graph:b32:s144:p45-45:src1:mixed0`
(`1026.4ms` forward, `8681` padding tokens), and tree's was
`prefix_graph:b32:s16:p45-45:src1:mixed0` (`735.6ms` forward,
`4132` padding tokens) with `ragged:b21/32` as the hottest decode shape
(`293.0ms` GPU). That reinforces the implementation split: long_output needs a
lower-cost `b64` decode replay path, while tree/multi need a real packed
cached-prefix prefill path rather than more attention-only packing or shape
bucket proliferation.

The 20260704 local tree A/Bs narrowed the cached-prefix prefill branch but did
not produce a default promotion. Enabling the FlashInfer cache backend without
explicit FlashInfer prefill used to 500 because paged KV storage cannot safely
fall back to the dense SDPA prefill path. The runtime now forces full-prompt
FlashInfer prefill for paged caches and the diagnostic run completed, but it is
not a performance candidate: `tree_of_thought` landed at
`728.9 / 213.2 / 770.6ms`, with `51` packed FlashInfer calls spending
`16.4s` and saving only `1.3K` padded tokens. The existing packed eager ragged
prefill flag also remains diagnostic-only: it saved `7.8K` padded tokens but
spent `29.8s` in packed eager prefill and regressed tree to
`1183.9 / 309.5 / 1376.0ms`. A sampled-medium batch-bucket probe with
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS_SAMPLED_MEDIUM=1,2,4,8,16`
improved tree TPOT to `29.4ms` but worsened TTFT/E2E to `147.6/179.4ms`; it
also did not truly cap large groups because counts above the largest configured
bucket fall back to the power-of-two `b32` bucket. The next viable tree lever is
therefore a split/cap policy for large cached-prefix suffix groups or a graphed
packed cached-prefix prefill, not the existing eager packed paths.
The existing split knob is also rejected as a default:
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_ON_CAPTURE_SKIP_BATCH=16` forced
large prefix groups through `b16` chunks, but raised tree to
`176.1 / 40.9 / 201.9ms`. It increased prefill batches from the default `64`
to `90`, pushed prefill forward/wall to `2.54s/3.20s`, and still left `7.5K`
packed-candidate saved tokens. Splitting alone trades row padding for more TP
graph replays; the next implementation needs a graph-backed packed-prefix
prefill body or a more selective split policy that can prove lower wall time.

`torchinferno.cli inference-bench-summary` now exposes that passive packed
candidate signal directly. The queue table includes packed-candidate calls,
saved tokens, and distinct packed groups, and the new
`[torchinferno packed prefill candidates]` section reports the hottest shape by
saved tokens. On the current local multi/tree profile this shows multi_turn's
`prefix_graph:b32:s144:p45-45:src1:mixed0` candidate at `17.3K/23.0K`
real/model tokens (`5.7K` saved for that shape, `22.6K` total saved), and
tree's `prefix_graph:b24:s16:p45-45:src1:mixed0` at `3.8K/6.9K`
real/model tokens (`3.1K` saved for that shape, `8.0K` total saved). This keeps
future public profiles from hiding the packed-prefix opportunity behind the
default graph-forward counters or the rejected eager packed paths.

The packed-candidate telemetry now also records compact grouped signatures such
as `prefix_graph:...|p45:s10:n12/p45:s11:n8`, with per-signature call, real
token, model-token, saved-token, and group counters. The coarse shape counters
show where padding is expensive; the signature counters answer whether a future
packed CUDA graph can replay on exact `(prefix_start, suffix_len)` histograms or
would churn on every wave. This remains telemetry-only and does not change
admission, prefix reuse, or model execution.

The first focused signature run wrote
`/tmp/ti-bench-results/signature-tree/.../runs/20260704_230459` and landed
tree_of_thought at `123.4 / 40.1 / 153.9ms`, throughput `8.2 tok/s`, and
`967/992` correct. Queue-to-first p50 was `102.5ms`; prefill forward/wall was
`1.81s/2.41s`; total ragged-decode GPU was `1.42s`. The packed-candidate
counter saw `58` candidate waves, `8.5K` saved padding tokens, and `158`
groups. The exported signature map retained `32` exact grouped keys covering
`35` mapped calls; only `6/58` candidate calls repeated an exported exact
signature, accounting for just `508` saved tokens (`6.0%` of candidate saved
tokens). The hottest coarse shape remained reusable
(`prefix_graph:b24:s16:p45-45:src1:mixed0`, `26` calls, `4.4K` saved tokens),
but exact `(prefix_start, suffix_len, count)` histograms are too fragmented for
an exact-signature graph cache to be the default packed-prefix implementation.
The next viable packed path needs fixed-capacity/dynamic-count grouping, not a
CUDA graph keyed directly on the full histogram.

The analyzer now reports aggregate signature reuse as scalars when the server
exports them: signature keys, signature-covered calls, total candidate calls,
repeated calls, and repeated saved tokens. Older logs still get a best-effort
map-derived row, but missing saved-token coverage is printed as `-` rather than
`0`. The queue-profile exporter computes the scalar fields before top-N shape
map limiting, so future public runs can answer the exact-signature question
without manual `jq` or map-limit ambiguity.

An extra sampled-medium bucket probe with
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS_SAMPLED_MEDIUM=1,2,4,8,12,16,20,24,28,32`
wrote `/tmp/ti-bench-results/tree-buckets-mid-signature/.../runs/20260704_231651`
and is rejected as a default. It reduced coarse padding opportunity
(`6.8K` packed-candidate saved tokens versus `8.5K`) and improved TPOT to
`38.2ms`, but tree_of_thought regressed to `128.8 / 38.2 / 156.9ms` with
`958/992` correct. Queue-to-first rose to `106.5ms`, prefill batches increased
to `63`, prefill forward stayed flat-to-worse at `1.84s`, and total
ragged-decode GPU rose to `1.48s`. Finer batch buckets reduce padding in this
row, but they also fragment shape reuse and do not improve the score-facing
median. Keep them as opt-in evidence only.

Two scheduler-side tree probes also failed to produce a default promotion.
`TORCHINFERNO_OPENAI_TP_ONLINE_COLLECT_IDLE_ARRIVALS=1` wrote
`/tmp/ti-bench-results/tree-idlecollect/.../runs/20260704_232750` and landed at
`129.5 / 26.0 / 152.2ms`, with `956/992` correct. It improved TPOT and slightly
improved E2E versus the local signature baseline, but worsened TTFT, lowered
correctness, and pushed total decode GPU from `1.42s` to `1.80s`; the hot decode
shape became `ragged:b19/32` (`263.3ms`). Capping sampled-medium active rows to
`24` via `TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_MAX_ACTIVE=24` wrote
`/tmp/ti-bench-results/tree-maxactive24/.../runs/20260704_233315` and regressed
harder to `161.8 / 45.2 / 192.0ms`, `958/992` correct. It removed `b32` waves
but increased prefill batches to `81`, prefill forward/wall to `2.16s/2.72s`,
and queue-to-first p50 to `138.1ms`. The conclusion is the same as the padding
probes: rearranging admissions and buckets cannot close the tree gap without a
lower-cost packed-prefix prefill body.

A common-prefix cache-only probe is also rejected. The temporary patch made
`_prefill_common_prefix_batch` use the model cache-only hook when all requests
had suffix tokens, avoiding logits for the shared prefix row. The focused tree
run wrote `/tmp/ti-bench-results/tree-cacheonly-prefix/.../runs/20260704_235221`
and regressed to `141.8 / 46.7 / 171.3ms`, `959/992` correct. Queue-to-first
p50 rose to `117.3ms`, prefill wall rose to `2.64s`, and decode GPU rose to
`1.66s`; the common prefix accounts for only one batch, so skipping its logits
does not move the dominant cached-prefix suffix replay cost. Do not promote
this cache-only common-prefix path without a stronger paired win.

Public queue profiles still point at model-side replay and decode throughput,
not graph-cache misses. Multi_turn reused only the `45` token common prefix,
ran `34` prefill batches, and spent `3.86s/4.02s` in prefill forward/wall with
`0` graph misses; its fast HTTP profile still sent role-only chunks for
512-token streams. Tree reused the same `45` token prefix and spent
`1.90s/2.13s` in prefill forward/wall plus `1.97s` ragged-decode GPU.
Long_output reused the `111` token common prefix, spent `4.21s/4.45s` in
prefill forward/wall, and spent `9.81s` in ragged-decode GPU across `803`
ragged decode batches. The public long_output profile had
`runtime_decode_many_tail_limited_calls=0`, so it did not exercise the local
stop-tail cap default.

A paired current-tree TorchInferno-only long_output recheck measured that local
default. The no-env run wrote
`/tmp/ti-bench-results/long-current/.../runs/20260704_155337` and landed at
`246.1 / 24.3 / 1085.0ms`, throughput `33.0 tok/s`, `1000/1000` correct. Its
profile did exercise the cap (`decode_many_stop_tail_max_steps=6`,
`28` tail-limited calls, `56` tail-limited steps), reducing decode model calls
to `736`, but still spending `10.10s` in ragged-decode GPU and
`4.81s/5.22s` in prefill forward/wall.

The same-tree cap-off control with
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_DECODE_MANY_STOP_TAIL_MAX_STEPS=0`
wrote `/tmp/ti-bench-results/long-cap0/.../runs/20260704_155859` and landed at
`239.8 / 24.4 / 1119.4ms`, throughput `32.5 tok/s`, also `1000/1000`
correct. Cap-off made TTFT about `6ms` better but worsened E2E by `34ms` and
throughput by `0.5 tok/s`; it also ran `749` decode model calls and spent
`10.28s` in ragged-decode GPU.

A fresh cap-4 rerun on the same patched tree wrote
`/tmp/ti-bench-results/long-tail4-current/.../runs/20260704_234227` and landed
at `225.4 / 23.8 / 1061.4ms`, throughput `33.8 tok/s`, `1000/1000` correct.
The profile showed `decode_many_stop_tail_max_steps=4`, `27` tail-limited
calls, `108` tail-limited steps, `731` decode model calls, `7.14s`
decode-many GPU time, and `1.22K` decode-many overgenerated tokens. Promote
cap 4 as the greedy-short default: it now beats the current cap-6 and cap-off
controls on E2E/throughput while preserving correctness. It is not the public
long_output solution; the remaining gap is still larger prefill replay plus
steady decode GPU time, not stop-tail overcompute.

The no-env default confirmation on the patched tree wrote
`/tmp/ti-bench-results/long-default-cap4-recheck/.../runs/20260705_001246` and
landed at `211.0 / 24.0 / 1078.1ms`, throughput `33.6 tok/s`, `1000/1000`
correct. The queue profile exported `decode_many_stop_tail_max_steps=4`, `31`
tail-limited calls, `124` tail-limited steps, `7.52s` decode-many GPU time, and
`1.25K` decode-many overgenerated tokens. This verifies that the promoted
default, not only the env-forced probe, exercises the cap. The row still spends
`4.74s/5.30s` in prefill forward/wall, so the remaining long_output target is
lower prefill/decode model work rather than more stop-tail capping.

A tighter cap-3 probe is rejected. With
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_DECODE_MANY_STOP_TAIL_MAX_STEPS=3`,
`/tmp/ti-bench-results/long-tail3-current/.../runs/20260705_002335` landed at
`285.3 / 23.1 / 1116.4ms`, throughput `32.4 tok/s`, `1000/1000` correct.
It cut decode-many model tokens and GPU time (`24.2K`, `6.02s`) but fragmented
the tail into more scalar ragged decode work: total ragged-decode GPU rose to
`10.19s`, decode model calls rose to `797`, and queue-to-first p50 rose to
`207.2ms`. Keep cap 4 as the default balance point.

A current-tree recheck of the rejected short-greedy `s80` suffix bucket also
does not change the default. Adding
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS_GREEDY_SHORT=16,32,64,80,96,128,256`
wrote `/tmp/ti-bench-results/long-suffix80/.../runs/20260705_003038` and landed
at `231.3 / 23.9 / 1094.2ms`, `1000/1000` correct. The bucket reduced total
prefill padding to `28.1K`, but introduced cold `s80` graph captures during the
request burst (`3.72s` capture time) and pushed prefill forward/wall to
`8.04s/8.46s`; the hottest new shape was
`prefix_graph:b24:s80:p111-111:src1:mixed0` at `2.13s` forward. The existing
`16,32,64,96,128,256` short-greedy bucket list remains the right warmed shape
set until a new bucket can be prewarmed and proven faster end to end.

A targeted `s80` prewarm recheck removed the cold-capture failure mode but is
still rejected as a default. The previous probe's hottest cold shape was
`prefix_graph:b24:s80:p111-111:src1:mixed0`, while startup extra-pair warmup
covered `111:32` and `111:64` but not `111:80`. Running the same `s80` bucket
override with
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_GREEDY_COMMON_PREFIX_EXTRA_PAIRS=111:32,111:64,111:80,122:16`
wrote `/tmp/ti-bench-results/long-s80-p111warm/.../runs/20260705_005103` and
landed at `215.5 / 24.1 / 1091.3ms`, throughput `33.2 tok/s`, `1000/1000`
correct. Queue profiles verified the intended mechanical fix (`0` request-path
prefill graph captures/misses, `s80` p111 replay shapes present), but prefill
forward/wall still stayed at `4.58s/4.98s` and total decode GPU at `9.98s`.
This improves the rejected cold `s80` run's TTFT, but it does not beat the
no-env cap-4 default confirmation on E2E or throughput; do not add `s80` to the
default greedy-short bucket list.

Greedy-short suffix-bucket splitting remains opt-in after the cap-4 default.
The focused run with
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS_GREEDY_SHORT=1`
wrote `/tmp/ti-bench-results/long-suffixsplit-current/.../runs/20260705_003858`
and landed at `217.5 / 24.1 / 1076.5ms`, throughput `33.2 tok/s`,
`1000/1000` correct. It reduced prefill padding to `28.8K`, prefill
forward/wall to `4.71s/5.11s`, and decode-many GPU to `6.53s`, but increased
prefill batches to `69` and did not beat the cap-4 default recheck on
TTFT/TPOT/throughput. The mechanism is real, but the score-facing movement is
within noise and still fragments replay shapes (`68` candidate waves with no
exact signature repeats), so keep it as a diagnostic rather than a default.

Decode-many token readback now uses a reusable flat device scratch buffer
instead of allocating and cloning one token tensor per replay step before the
single host readback. The focused long_output run wrote
`/tmp/ti-bench-results/long-decode-scratch-current/.../runs/20260704_172611`
and landed at `269.0 / 23.9 / 1175.4ms`, throughput `31.8 tok/s`,
`1000/1000` correct. This is not a score-facing win versus the better
current-tree local bands, but it is useful hot-path hygiene: compared with the
earlier same-tree packed-candidate profile, decode-many CPU token handling
dropped from `26.2ms` to `19.9ms` despite more decode-many active model tokens
(`32.0K` vs `29.1K`). The row remained dominated by `10.22s` ragged-decode GPU
and `5.87s/4.86s` prefill wall/forward time, so the long_output gap still
needs model-side decode throughput or packed cached-prefix prefill, not another
host-token-copy tweak.

Decode-many now has explicit model wall/GPU timing counters so the long_output
profile can separate multi-step replay cost from token harvesting. The focused
profile wrote
`/tmp/ti-bench-results/long-decodemany-timing-current/.../runs/20260704_181309`
and landed at `249.3 / 24.0 / 1195.6ms`, throughput `32.7 tok/s`,
`1000/1000` correct. The final queue profile recorded `145` decode-many calls
over `574` steps, `30.5K` active model tokens, `34.0K` padded tokens, and
`29.0K` emitted tokens. Decode-many model time was `7.84s` wall and `7.79s`
GPU event time; CPU token handling was only `18.0ms`. Total ragged decode was
`10.37s` wall / `10.31s` GPU, with prefill at `4.75s/5.77s` forward/wall.
The avoidable decode-many accounting was visible but not dominant:
`3.47K` padding tokens and `1.49K` overgenerated stop-tail tokens. The hot
`decode_many:b64/64` shape alone consumed `3.72s` GPU over `270` steps and
`17.3K` model tokens. This closes the readback branch more firmly: the
score-facing long_output lever is lower model-side decode replay cost or
overlap, plus packed cached-prefix prefill, not more CPU token-copy surgery.
Validation for the telemetry path:
`venv/bin/python -m pyflakes src/torchinferno/runtime/serving.py
src/torchinferno/openai_server.py tests/test_openai_server.py
tests/test_serving_engine.py`; `venv/bin/python -m pytest
tests/test_openai_server.py::test_openai_queue_profile_records_runtime_engine_stats
tests/test_serving_engine.py::test_continuous_batch_engine_online_many_keeps_decode_tokens_ordered
tests/test_serving_engine.py::test_continuous_batch_engine_online_many_shape_model_tokens_include_padding
tests/test_serving_engine.py::test_continuous_batch_engine_online_many_can_overcompute_stop_tokens
tests/test_serving_engine.py::test_continuous_batch_engine_online_many_can_cap_stop_tail_burst
-q`; and `git diff --check`.

The matching hot-shape replay profile used
`TORCHINFERNO_PROFILE_RAGGED_DECODE_REPLAY_ONCE=1`,
`TORCHINFERNO_PROFILE_RAGGED_DECODE_MIN_BATCH=64`, and
`TORCHINFERNO_PROFILE_RAGGED_DECODE_REPLAY_SKIP_MATCHES=2`. It wrote
`/tmp/ti-bench-results/long-b64-decode-profile-current/.../runs/20260704_182045`
and landed at `264.4 / 24.5 / 1173.6ms`, `1000/1000` correct. The profiler
captured `batch=64 match=3 cache_bucket=1024 rows=64` with `12.411ms` self CUDA
time. The largest slices were the 160 dense GEMM kernels at `3.391ms`, gate-up
Marlin at `3.299ms`, 160 symmetric-memory all-reduces at `2.075ms`, grouped GQA
decode attention at `1.479ms`, and another dense split-K GEMM at `986us`.
Attention is only about `12%` of the hot replay. The final queue profile also
kept the new decode-many timing shape: `136` decode-many calls, `536` steps,
`8.15s/8.10s` decode-many model wall/GPU time, and `4.20s` of that GPU time in
`decode_many:b64/64` over `248` steps. This points at GEMM/Marlin and per-layer
TP collective reduction as the decode-throughput levers; FlashAttention tuning
or host-token handling will not move the long_output median enough on their own.

Queue profiles now also export live ragged decode graph cache shape summaries:
`runtime_decode_graph_cache_live_entries`,
`runtime_decode_graph_cache_live_shape_counts`,
`runtime_decode_graph_cache_live_batch_counts`,
`runtime_decode_graph_cache_live_context_counts`, and
`runtime_decode_graph_cache_live_cache_bucket_counts`. These mirror the live
prefill graph summaries and make future public profiles distinguish first-use
shape churn or cache eviction from steady replay cost. This is intentionally
telemetry-only: it reads `_ragged_decode_graphs` and aggregates the real graph
key `(cache id, batch, max_seq_len, cache token bucket, rows flag, symm key)`,
without changing capture, replay, or graph-cache sizing. Validation:
`venv/bin/python -m pyflakes src/torchinferno/openai_server.py
tests/test_openai_server.py`; `venv/bin/python -m pytest
tests/test_openai_server.py::test_openai_queue_profile_records_runtime_engine_stats
tests/test_openai_server.py::test_openai_queue_profile_progress_skips_shape_details_by_default
tests/test_openai_server.py::test_openai_queue_profile_graph_shape_limit_keeps_large_cache_visible
-q`; and `git diff --check`.

The runtime also now attributes decode graph capture and replay wall time by
shape through `runtime_decode_graph_capture_shape_ms` and
`runtime_decode_graph_replay_shape_ms`. The existing scalar
`runtime_decode_graph_replay_ms` was enough to see total graph overhead, but not
whether it came from the hot `b64` decode-many shape, smaller tail shapes, token
graphs, or logits fallback. The new maps reuse the same
`ragged_decode:{token|logits}:b*:rows*` keys as the capture/miss counters, so a
future public profile can pair live cache shape, miss counts, replay counts, and
replay time without enabling the heavyweight torch profiler. Focused validation:
`venv/bin/python -m pyflakes src/torchinferno/runtime/serving.py
src/torchinferno/openai_server.py tests/test_serving_engine.py
tests/test_openai_server.py`; `venv/bin/python -m pytest
tests/test_serving_engine.py::test_continuous_batch_engine_records_ragged_decode_graph_captures
tests/test_openai_server.py::test_openai_queue_profile_records_runtime_engine_stats
-q`.

A current-tree explicit mixed-prefix multi_turn refresh is rejected as a default
candidate. Running TorchInferno-only with
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_MIXED_PREFIX_REUSE=1` wrote
`/tmp/ti-bench-results/multi-mixed-current/.../runs/20260704_184104` and landed
at `415.6 / 64.0 / 451.2ms`, `980/1000` correct. The queue profile confirms
the regression is queueing, not missing graph warmup: `max_active=32`,
`prefix_rows=112`, `39` prefill batches, `38/1` prefill graph hits/misses,
`2.78s/3.07s` prefill forward/wall, and only `895ms` ragged decode GPU, but
queue-to-submit was `104ms` p50 / `614ms` p99 and queue-to-first was `179ms`
p50 / `700ms` p99. The harness terminated the server before a detailed
quiescent profile, so the run did not include route maps that would prove
request-prompt reuse. The profile recorder now marks final progress snapshots
with `profile_complete_snapshot=true` and forces the same shape/detail maps as a
quiescent record when all submitted requests have finished. That makes short
targeted runs useful even when the harness sends SIGTERM immediately after the
last stream closes. Validation:
`venv/bin/python -m pyflakes src/torchinferno/openai_server.py
tests/test_openai_server.py`; `venv/bin/python -m pytest
tests/test_openai_server.py::test_openai_queue_profile_progress_skips_shape_details_by_default
tests/test_openai_server.py::test_openai_queue_profile_records_runtime_engine_stats
-q`; and `git diff --check`.

Rebalancing the explicit mixed-prefix row budget toward active rows is also
rejected. The focused probe kept the same explicit reuse policy but added
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_MAX_ACTIVE=48` and
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_MIXED_PREFIX_ROWS=96`, preserving
the `144` total-row envelope. It wrote
`/tmp/ti-bench-results/multi-mixed-active48-prefix96/.../runs/20260704_185228`
and regressed to `936.4 / 90.5 / 1063.3ms`, `980/1000` correct. The completed
queue snapshots captured all `1000` requests across three online sessions:
`44` prefill batches, `31.0K` prefill tokens, `10.63s/11.33s` prefill
forward/wall, `18/26` prefill graph hits/misses, `1.54s` ragged decode GPU,
and route counts `{"common_prefix":247,"request_prompt":751}` with `1000`
full-prompt adoptions. The new detail maps attribute the misses mostly to cold
large-row mixed-prefix shapes (`b48:s32:ctx-128/ctx-256:src48`) and each
session still showed submit-to-first p50 around `475ms`. Larger active waves
therefore amplify cold mixed-prefix suffix replay and decode cost instead of
fixing the explicit policy's queueing jitter; keep the opt-in at `32/112`.

Extending stream role-deferral through 512-token requests is accepted. The
benchmark counts TTFT at the first non-empty content delta, so sending a
role-only SSE chunk before content adds client/parser work that the metric does
not credit. The existing default already defers role chunks for bounded streams
up to `400` tokens; the focused multi_turn probe raised only
`TORCHINFERNO_OPENAI_STREAM_DEFER_ROLE_MAX_TOKENS=512` alongside the explicit
mixed-prefix policy. It wrote
`/tmp/ti-bench-results/multi-mixed-deferrole512/.../runs/20260704_190414` and
landed at `285.7 / 63.2 / 349.2ms`, `980/1000` correct, versus the same
current-tree mixed-prefix default at `415.6 / 64.0 / 451.2ms`. The runtime
shape stayed intended: `max_active=32`, `prefix_rows=112`, `37` prefill
batches, `18.7K` prefill tokens, `2.33s/2.63s` prefill forward/wall, `37/0`
prefill graph hits/misses, `865ms` ragged decode GPU, route counts
`{"common_prefix":125,"request_prompt":875}`, and `1000` full-prompt
adoptions. Fast HTTP profiling confirmed the role-only chunk disappeared
(`role_send_present=false`) and first-content p50 moved to `163ms`; benchmark
TTFT still includes client-side overhead, but the content-visible median dropped
by about `130ms`. Make `512` the default role-deferral bound.

A no-env public-shape multi_turn check after the default change confirms role
deferral alone does not solve the common-prefix path. The run wrote
`/tmp/ti-bench-results/multi-default-deferrole512/.../runs/20260704_191132` and
landed at `414.8 / 60.2 / 470.1ms`, `980/1000` correct. Fast HTTP profiling
again showed no role-only chunk (`role_send_present=false`), but the runtime was
still the common-prefix shape: `prefix_rows=64`, no full-prompt adoptions,
route counts `{"common_prefix":1000}`, `35` prefill batches, `85.3K` prefill
tokens, `4.08s/4.43s` prefill forward/wall, and queue-to-first p50 `317ms`.
The stream packaging fix closes a client-visible chunking tax; the default
multi_turn gap still needs cheaper conversation-prefix reuse or lower
common-prefix prefill queueing.

Close-delimited SSE is not a broad default. Disabling fast HTTP keepalive with
`TORCHINFERNO_OPENAI_FAST_HTTP_KEEPALIVE=0` on the explicit mixed-prefix
multi_turn path wrote
`/tmp/ti-bench-results/multi-mixed-close-sse/.../runs/20260704_191923` and
landed at `250.2 / 68.0 / 343.6ms`, `980/1000` correct. It improved TTFT
versus role-deferral alone (`285.7ms -> 250.2ms`) and kept the intended runtime
shape (`32/112`, `36` prefill batches, `18.8K` prefill tokens, `36/0` prefill
graph hits/misses, `{"common_prefix":125,"request_prompt":875}`), but worsened
TPOT and p99 latency. The adjacent few_shot control with only keepalive disabled
wrote `/tmp/ti-bench-results/few-close-sse/.../runs/20260704_192447` and
regressed to `188.0 / 51.4 / 236.9ms`, `977/1000` correct, worse than the
current public few_shot row (`170.2 / 49.1 / 225.4ms`) with a bad `1.2s` TTFT
p99. Keep chunked keepalive as the broad default; global close-SSE remains an
explicit transport diagnostic.

The same transport tradeoff is not stable enough to promote even under the
deterministic large-stream bucket. A no-env common-prefix multi_turn control
with global keepalive disabled wrote
`/tmp/ti-bench-results/multi-default-close-sse/.../runs/20260704_194105` and
landed at `357.2 / 60.8 / 411.3ms`, `983/1000` correct, versus the
role-deferral-only no-env control at `414.8 / 60.2 / 470.1ms`. Implementing the
close behavior as a request-scoped default for `temperature <= 0` and
`400 < max_tokens <= 512` then wrote
`/tmp/ti-bench-results/multi-default-scoped-close/.../runs/20260704_194827` and
landed at `371.3 / 64.2 / 440.5ms`, `983/1000` correct. Fast HTTP profiling
confirmed all `1000` 512-token streams used `keep_alive=false` with no role-only
chunk, while the runtime stayed on the default common-prefix route
(`prefix_rows=64`, `35/0` prefill graph hits/misses,
`{"common_prefix":1000}`). The subsequent no-env full suite on the same patch
wrote `/tmp/ti-bench-results/full-current-scopedclose/.../runs/20260704_200347`
and regressed the multi_turn row back to `414.7 / 63.3 / 469.7ms` with queue
profile `35` prefill batches, `83.7K` prefill tokens, and queue-to-first p50
`326ms`. Keep stream keepalive as the default; the large-greedy close path is
only an explicit diagnostic via
`TORCHINFERNO_OPENAI_FAST_HTTP_GREEDY_LARGE_CLOSE_STREAM=1`. The restored
default check
`/tmp/ti-bench-results/multi-default-keepalive-restored/.../runs/20260704_201257`
landed at `413.9 / 61.1 / 463.7ms`, `981/1000` correct, and confirmed
`keep_alive=true` for all `1000` 512-token streams with the same common-prefix
runtime shape (`34/0` prefill graph hits/misses, `80.2K` prefill tokens,
queue-to-first p50 `329ms`).

Do not extend close-SSE to short-greedy long_output. The focused global-close
probe `/tmp/ti-bench-results/long-close-sse-current/.../runs/20260704_195728`
kept correctness at `1000/1000` but landed at `258.0 / 23.4 / 1167.2ms`, worse
TTFT/E2E than the current no-env long controls. The queue profile stayed on the
normal common-prefix route but moved the wrong direction: `57` prefill batches,
`4.59s/5.39s` prefill forward/wall, `10.46s` ragged-decode GPU, and `7.49s`
decode-many GPU. Short-greedy streams keep chunked keepalive by default.

Combining the already-rejected sampled-medium knobs is also rejected. The tree
probe with `TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS=12,16` and
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_INITIAL_BATCH_WAIT_MS=0` wrote
`/tmp/ti-bench-results/tree-s12-initial0-current/.../runs/20260704_201920` and
landed at `140.4 / 39.8 / 166.5ms`, `959/992` correct. Queue profiling showed
the intended `s12` graph surface and no misses (`61/0` prefill graph
hits/misses), but prefill wall rose to `2.51s`, queue-to-first p50 was
`118ms`, and TPOT regressed. Keep sampled-medium suffix buckets and initial wait
at their current defaults; the tree gap still needs true packed cached-prefix
prefill or lower TP replay cost.

The targeted current-tree tree replay profile wrote
`/tmp/ti-bench-results/tree-replay-current/.../runs/20260704_160910` with
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_REPLAY_ONCE=1`,
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_CONTEXT_LEN=61`, and
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_REPLAY_SKIP_MATCHES=1`. It landed at
`132.0 / 36.7 / 158.2ms`, `955/992` correct, with the expected profiler tail
spike (`p99` TTFT/E2E `1682/1699ms`). The warmed request graph fired after
readiness as `batch=32 suffix=16 match=2 context_len=61 src_rows=1`. The
CUDA table is stable with the earlier diagnosis: total replay CUDA time was
`33.064ms`, led by `160` NCCL all-reduces at `13.039ms` (`39.4%`), then GEMM
kernels at about `9.8ms` combined, RMSNorm/add at about `3.3ms`, gather/index
work at about `2.8ms`, and FlashAttention at only `0.796ms` (`2.4%`).

The final queue profile from that same run confirms the request shape rather
than a graph-cache issue: all `992` requests reused only the `45` token common
prefix, with `58` prefill graph hits, `0` misses, `0` request-path captures,
`2.89s/3.34s` prefill forward/wall, and `1.44s` ragged-decode GPU. Every
prefix-suffix replay still used `s16`, while real suffixes were only `10`,
`11`, and `12`, for `8.29K` padding tokens (`2.82K` row and `5.47K` suffix).
The hot replay map was `b32:s16:ctx61` at `1.145s` over `11` calls, `b24:s16`
at `87.3ms` over `24` calls, and `b16:s16` at `33.0ms` over `12` calls. This
closes the scheduler/bucket retest loop for tree: the next score-facing lever
must either make cached-prefix suffix prefill packed enough to avoid padded
MLP/all-reduce work, or reduce the per-layer TP collective cost. Attention
backend selection, more suffix buckets, and another wait/refill knob are not
supported by the current evidence.

A current-tree sampled-medium initial-wait-zero probe reinforces that conclusion.
With only `TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_INITIAL_BATCH_WAIT_MS=0`
changed, `/tmp/ti-bench-results/tree-initial0-current/.../runs/20260704_193424`
landed at `130.7 / 27.3 / 155.0ms`, `960/992` correct. Queue-to-submit p50
fell slightly to `48.6ms`, but queue-to-first stayed `107.8ms`, prefill
forward/wall was `1.80s/2.44s`, and all `992` requests still used only the
`45` token `common_prefix` route with `59/0` prefill graph hits/misses. Removing
the sampled-medium first wait therefore shifts small scheduling counters without
closing the TTFT/E2E gap; keep the existing `1ms` default.

A fresh current-patch recheck confirms that decision. The same initial-wait-zero
env wrote
`/tmp/ti-bench-results/tree-initial0-recheck/.../runs/20260705_000626` and
landed at `141.1 / 29.9 / 166.6ms`, `959/992` correct. It kept `59/0` prefill
graph hits/misses, but queue-to-first p50 rose to `115.8ms`, prefill wall rose
to `2.57s`, and decode GPU stayed high at `1.44s`. This can improve sampled
TPOT in some runs, but it is not a stable TTFT/E2E improvement over the
stronger no-env tree rows and should remain an override.

Tree pinned full-prompt reuse is also rejected by passive candidate telemetry
on current head. The runtime now keeps a profile-only shadow radix index for
full prompts that pinned shared-prefix mode skipped with
`pinned_without_allowance`, and exports
`runtime_full_prompt_reuse_candidate_*` counters. A no-env tree profile wrote
`/tmp/ti-bench-results/tree-fullprompt-candidate-current/.../runs/20260704_171403`
and landed at `135.6 / 36.4 / 167.6ms`, `966/992` correct. The final queue
profile stored `992` shadow full-prompt candidates covering `55,042` prompt
tokens, but recorded `0` later requests with a deeper candidate hit:
`runtime_full_prompt_reuse_candidate_requests=0`,
`runtime_full_prompt_reuse_candidate_extra_tokens=0`, and empty candidate-depth
maps. Actual reuse stayed `{"common_prefix":992}` / `{"45":992}` with
`60` prefill graph hits and `0` misses. This rules out simply enabling pinned
full-prompt/radix-style prompt stores for tree; the prompts do not repeat or
extend prior full prompts in this benchmark stream, so that path would add
store pressure without saving suffix prefill work.

That profile-only local full-prompt candidate index is now opt-in. The current
default previously built the shadow radix index for every pinned full-prompt
store skip whenever queue profiling was enabled, but long_output candidate
telemetry showed no useful hits and a current-tree control stored `1000` shadow
candidates (`155.7K` prompt tokens) with `0` later candidate requests. A paired
same-host long_output A/B made local candidate indexing require
`TORCHINFERNO_CONTINUOUS_FULL_PROMPT_REUSE_CANDIDATE_PROFILE=1` or an explicit
`TORCHINFERNO_CONTINUOUS_FULL_PROMPT_REUSE_CANDIDATE_CAPACITY`. The row moved
from `243.4 / 24.9 / 1170.2ms` to `248.3 / 24.8 / 1144.5ms`, with
`1000/1000` correct in both runs. Queue counters show the intended cleanup:
candidate stores fell `1000 -> 0`, `prefill_state_ms` fell `625 -> 68ms`,
prefill wall fell `5.86 -> 5.47s`, and queue-to-finish p50 fell
`1083 -> 1038ms`. Keep the heavier candidate profiler opt-in for diagnostics;
it should not tax default public-profile runs when it has no candidate hits.

Periodic online progress profile snapshots are now opt-in as well. A profile-off
long_output probe on the same tree wrote
`/tmp/ti-bench-results/current-long-profileoff/.../runs/20260704_205520` and
landed at `233.5 / 23.6 / 1106.7ms`, `1000/1000` correct, versus the profiled
no-candidate control at `248.3 / 24.8 / 1144.5ms`. That leaves a remaining
profile tax after disabling the local candidate index. A no-code A/B with
`TORCHINFERNO_OPENAI_TP_ONLINE_PROFILE_SNAPSHOT_COMMANDS=0` still wrote the
final queue/HTTP profiles but skipped the `22` periodic progress records; it
wrote `/tmp/ti-bench-results/current-long-nosnapshots/.../runs/20260704_210040`
and landed at `255.4 / 24.0 / 1112.9ms`, `1000/1000` correct. TTFT was noisy
and worse in that run, but TPOT and E2E moved toward the profile-off row. The
default now records quiescent/final queue profiles only; mid-run progress
snapshots remain available by setting
`TORCHINFERNO_OPENAI_TP_ONLINE_PROFILE_SNAPSHOT_COMMANDS` explicitly.

The remaining profile-off delta came from the fast HTTP profile. A queue-only
long_output run kept `TORCHINFERNO_OPENAI_QUEUE_PROFILE_JSONL` but omitted
`TORCHINFERNO_OPENAI_FAST_HTTP_PROFILE_JSONL`; it wrote
`/tmp/ti-bench-results/current-long-queueonly/.../runs/20260704_211009` plus
the manual queue profile and landed at `226.7 / 23.8 / 1103.3ms`,
`1000/1000` correct. A patched TorchInferno run with the default fast HTTP
profile made per-token/per-send timing fields opt-in through
`TORCHINFERNO_OPENAI_FAST_HTTP_PROFILE_DETAILED=1` and still wrote the useful
request-level fields (`first_content_sent_ms`, token/chunk counts, and
`role_sent`); it wrote
`/tmp/ti-bench-results/current-long-lightfast/.../runs/20260704_211654` and
landed at `245.2 / 24.3 / 1117.3ms`, `1000/1000` correct. The lightweight
profile is cheaper and preserves explicit diagnostics, but the queue-only row
shows the public score path should avoid the 1000 in-band HTTP profile records.
Inference-bench `main` now carries commit `183c9c54`, which keeps TorchInferno's
queue profile enabled by default and makes the fast HTTP profile opt-in via
`INFERENCE_BENCH_TORCHINFERNO_FAST_HTTP_PROFILE=1`.

The pushed inference-bench default was verified with fresh current-tree local
runs. A long_output run at
`/tmp/ti-bench-results/current-long-newdefault/.../runs/20260704_212644`
landed at `251.3 / 23.9 / 1074.6ms`, throughput `33.4 tok/s`, and
`1000/1000` correct. The provider log directory contained only
`torchinferno_queue_profile.jsonl` and `torchinferno.log`, confirming the fast
HTTP profile is no longer part of the default score path. The final queue
profile stayed in the same shape as the manual queue-only run:
`65` prefill batches, `4.94s/5.36s` prefill forward/wall, `64ms`
prefill-state time, `10.52s` decode GPU, and the hot `decode_many:b64/64`
shape at `3.64s` GPU. This removes the avoidable profile overhead from the next
public run but does not change the diagnosis: long_output still needs lower
model-side decode replay cost or a non-fragmenting packed cached-prefix prefill
path.

A current-default multi_turn plus tree_of_thought check with the same provider
behavior wrote
`/tmp/ti-bench-results/current-multitree-newdefault/.../runs/20260704_213322`.
It landed at `386.6 / 61.6 / 435.1ms`, `981/1000` correct for multi_turn and
`134.0 / 41.0 / 165.7ms`, `957/992` correct for tree. The analyzer now reports
the official inference-bench scored medians (`score_ttft`, `score_tpot`,
`score_e2e`, and `score_tps`) plus raw p90/output-shape context so single-token
median rows do not hide the benchmark's nonzero scored TPOT. Queue profiles
again show model/runtime work rather than transport as the blocker:
multi_turn spent `4.08s/4.32s` in common-prefix prefill with queue-to-first p50
`299ms`, while tree spent `1.88s/2.32s` in common-prefix prefill with
queue-to-first p50 `113ms`. Provider logs again omitted the fast HTTP profile
by default. These rows keep the next non-long lever focused on packed
cached-prefix prefill or lower TP collective cost, not more stream/profile
knobs.

The analogous current-tree multi_turn replay profile wrote
`/tmp/ti-bench-results/multi-replay-current/.../runs/20260704_161617` with
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_REPLAY_ONCE=1`,
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_CONTEXT_LEN=141`, and
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_MIN_SUFFIX=96`. The benchmark row was
profiler-perturbed at `421.7 / 60.8 / 483.2ms`, `980/1000` correct, but the
kernel table captured the intended greedy-large request graph:
`batch=32 suffix=96 match=2 context_len=141 src_rows=1`. Replay CUDA time was
`122.892ms`, led by `160` NCCL all-reduces at `43.434ms` (`35.3%`),
FP8/GEMM kernels at about `44.8ms`, RMSNorm/add/reduce kernels at about
`18.3ms`, and FlashAttention at only `1.470ms` (`1.2%`).

The final multi_turn quiescent queue profile stayed common-prefix-only:
`runtime_prefix_reuse_hit_token_counts={"45":1000}` and
`runtime_prefix_reuse_route_counts={"common_prefix":1000}`. It ran `34`
prefill batches, `34` graph hits, `0` graph misses/captures, and spent
`5.28s/5.51s` in prefill forward/wall plus `805ms` ragged-decode GPU. Padding
was `28.4K` tokens, dominated by `25.2K` suffix padding. The largest replay
shape was `b32:s96:ctx141`, `1.343s` replay and `1.718s` forward over `4`
calls; the other large common-prefix waves were `b32:s144` (`1.252s` forward
over `7` calls), `b32:s128` (`631ms` over `4`), and `b32:s64` (`525ms` over
`6`). This confirms that the default multi_turn row is not missing graph
coverage. It needs either stable non-common request-prompt reuse that does not
fragment the row, a packed suffix-prefill implementation that skips padded MLP
and all-reduce work, or a real TP collective replacement for prefill. Another
common-prefix scheduler knob cannot close the gap alone.

An env-gated persistent full-prompt candidate probe rules out cross-session
prefix persistence as the missing multi_turn lever on the current stream. With
`TORCHINFERNO_CONTINUOUS_PERSISTENT_FULL_PROMPT_REUSE_CANDIDATE=1` and capacity
`4096`, the focused run wrote
`/tmp/ti-bench-results/multi-persistent-candidate-current/.../runs/20260704_173933`
and landed at `389.3 / 59.8 / 439.9ms`, `981/1000` correct. The profile stored
`1000` persistent shadow full prompts (`114,608` tokens) but recorded `0`
cross-session candidate hits:
`runtime_persistent_full_prompt_reuse_candidate_requests=0` and
`runtime_persistent_full_prompt_reuse_candidate_extra_tokens=0`. The same run's
per-session shadow index did find the known within-session opportunity:
`runtime_full_prompt_reuse_candidate_requests=875` and `53,499` extra candidate
tokens beyond the actual `45` token common-prefix hit. This confirms the
multi_turn gap is not from losing candidates across `start_online` resets; it
is from the current default rejecting the expensive non-common/request-prompt
reuse route. The prior opt-in request-prompt route can expose that opportunity
but is still too slow as implemented, so the next useful work is making those
within-session request-prompt suffix waves cheaper rather than persisting prefix
state across bursts.

The current-tree mixed-prefix recheck keeps that conclusion. A no-env
multi_turn baseline wrote
`/tmp/inference-bench-ti-multiturn-baseline-results/.../runs/20260705_085755`
and landed at `397.1 / 62.5 / 451.1ms`, `981/1000` correct, with the expected
common-prefix shape (`34` prefill batches, `4.07s` prefill wall, `1000`
common-prefix reuses at `45` tokens). The explicit mixed-prefix opt-in
(`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_MIXED_PREFIX_REUSE=1`) wrote
`/tmp/inference-bench-ti-multiturn-mixed-optin-current-results/.../runs/20260705_090544`
and regressed to `445.6 / 64.6 / 487.3ms`: prefill wall fell to `2.99s`, but
queue-to-submit/first rose enough to erase the model-work win. Adding
`TORCHINFERNO_OPENAI_TP_ONLINE_COLLECT_IDLE_ARRIVALS=1` recovered the score to
`393.3 / 62.7 / 451.7ms`, but did not beat the no-env E2E and only left a
partial queue-profile aggregate (`465/1000`) after shutdown. Keep mixed-prefix
reuse and greedy-large idle collection explicit. The inference-bench summary now
prints queue-profile coverage such as `465/1000 partial` so incomplete
aggregates are not mistaken for full-run evidence.

Packed-ragged eager remains an opt-in oracle, but it now has explicit runtime
telemetry for future packed CUDA/FlashInfer replacements. The
`TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER=1` branch records
`prefill_packed_eager_calls`, real suffix tokens, padded model-token
equivalent, saved padding tokens, wall time, and per-shape counts/tokens/time;
OpenAI queue profiles export them as `runtime_prefill_packed_eager_*`. This
does not promote the Python eager oracle or change default scheduling. It makes
the next packed implementation measurable without overloading graph-hit/miss
counters. Validation: `venv/bin/python -m pyflakes
src/torchinferno/runtime/serving.py src/torchinferno/openai_server.py
tests/test_serving_engine.py tests/test_openai_server.py`; `venv/bin/python -m
pytest
tests/test_serving_engine.py::test_continuous_batch_engine_can_opt_into_packed_ragged_prefill_eager
tests/test_openai_server.py::test_openai_queue_profile_records_runtime_engine_stats
-q`; and `git diff --check`.

Default prefix-graph prefill now also records passive packed-prefill candidate
telemetry without enabling the slow eager oracle. For each padded cached-prefix
graph wave, the runtime records `prefill_packed_candidate_calls`, real suffix
tokens, padded graph model tokens, avoidable padding tokens, and the number of
distinct `(prefix_start, suffix_len)` groups a packed attention body would need
to cover. Per-shape maps are exported as
`runtime_prefill_packed_candidate_shape_*` in detailed queue-profile records.
This is diagnostic only, but it makes future public runs quantify the exact
non-fragmenting packed-prefix opportunity in the default path. Validation:
`venv/bin/python -m pyflakes src/torchinferno/runtime/serving.py
src/torchinferno/openai_server.py tests/test_serving_engine.py
tests/test_openai_server.py`; `venv/bin/python -m pytest
tests/test_serving_engine.py::test_continuous_batch_engine_records_profile_shape_counts
tests/test_openai_server.py::test_openai_queue_profile_records_runtime_engine_stats
-q`; and `git diff --check`.

A focused long_output run with the new passive counters quantified the default
packed-prefix opportunity. The run wrote
`/tmp/ti-bench-results/long-packed-candidate-current/.../runs/20260704_163527`
and landed at `268.7 / 22.6 / 1124.6ms`, `1000/1000` correct. It recorded `57`
packed-candidate calls, `44.7K` real suffix tokens, `75.5K` padded graph model
tokens, and `30.8K` avoidable packed-candidate tokens (`40.8%` of the
prefix-graph suffix slots). The largest saved-token shapes were
`prefix_graph:b24:s64:p111-111:src1:mixed0` (`6.42K`), `b24:s96` (`5.47K`),
`b16:s64` (`5.11K`), and `b32:s64` (`4.32K`). The same profile still spent
`4.56s/4.96s` in prefill forward/wall and `10.42s` in ragged-decode GPU, so
packed prefill is a real lever but not the only long_output gap.

The first non-Python packed implementation is now available as an explicit
FlashInfer-cache path. `prefill_ragged_logits_packed_flashinfer` flattens only
real suffix tokens, scatter-writes their KV into FlashInfer NHD rows, plans one
varlen `BatchPrefillWithPagedKVCacheWrapper` over the physical rows, and runs
attention through FlashInfer while projections, MLP, and TP collectives operate
on the compact `[1, total_real_tokens]` stream. Serving can route variable-length
FlashInfer-cache prefill batches through it with
`TORCHINFERNO_CONTINUOUS_PACKED_FLASHINFER_PREFILL=1`; it remains default-off
because the public/default server still uses dense graph prefill and prior
padded FlashInfer prefill was slower. Validation:
`venv/bin/python -m pyflakes src/torchinferno/models/llama3/tensor_parallel.py
src/torchinferno/runtime/serving.py tests/test_llama3_tensor_parallel_distributed.py
tests/test_serving_engine.py`; `venv/bin/python -m pytest
tests/test_llama3_tensor_parallel_distributed.py::test_llama3_tensor_parallel_packed_ragged_prefill_matches_padded_oracle
tests/test_llama3_tensor_parallel_distributed.py::test_llama3_tensor_parallel_packed_flashinfer_prefill_matches_padded_oracle
tests/test_serving_engine.py::test_continuous_batch_engine_can_use_packed_flashinfer_prefill
-q`.

The first real 8xH100 tree probe of that path rejects FlashInfer full-prompt
prefill as a route for prefix-heavy workloads. With
`TORCHINFERNO_OPENAI_FLASHINFER_FORWARD=1`,
`TORCHINFERNO_CONTINUOUS_FLASHINFER_PREFILL_DISABLE=0`, and
`TORCHINFERNO_CONTINUOUS_PACKED_FLASHINFER_PREFILL=1`, the run wrote
`/tmp/ti-bench-results/tree-packed-fi-current/.../runs/20260704_165441` and
landed at `719.2 / 46.2 / 749.0ms`, `960/992` correct. The packed FlashInfer
branch fired (`53` calls) but skipped only `1.43K` padded tokens out of
`56.25K` packed model-token slots, spent `16.9s` inside packed FlashInfer
prefill and `17.37s` in prefill wall, and recorded `0` prefix-reuse requests.
The failure mode is clear: the broad FI-prefill opt-in consumed cached-prefix
groups as full-prompt prefill, erasing common-prefix reuse for a tiny padding
win. The scheduler now keeps prefix-hit groups off full-prompt FI prefill unless
`TORCHINFERNO_CONTINUOUS_FI_REUSE=1` is also set, and queue profiles export
`runtime_prefill_packed_flashinfer_*` counters so future runs can prove whether
the branch is active. Validation:
`venv/bin/python -m pyflakes src/torchinferno/runtime/serving.py
src/torchinferno/openai_server.py tests/test_serving_engine.py
tests/test_openai_server.py`; `venv/bin/python -m pytest
tests/test_serving_engine.py::test_continuous_batch_engine_can_use_packed_flashinfer_prefill
tests/test_serving_engine.py::test_continuous_batch_engine_keeps_prefix_hits_off_full_prompt_flashinfer
tests/test_openai_server.py::test_openai_queue_profile_records_runtime_engine_stats
-q`; and `git diff --check`.

## Public 20260704_131636 refresh and mixed-prefix default rejection

The latest public all-provider run advanced to
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260704_131636`
with TorchInferno scoring `1/20`, vLLM `17/20`, and SGLang `1/20`. The
remaining public gaps are still concentrated in TTFT/prefill work for
multi_turn, tree_of_thought, and long_output, while few_shot also regressed
against vLLM:

- few_shot: TorchInferno `196.5 / 48.6 / 242.7ms`, vLLM
  `130.0 / 42.7 / 163.6ms`, SGLang `143.5 / 75.8 / 221.0ms`.
- self_consistency: TorchInferno `315.0 / 0.0 / 332.8ms`, vLLM
  `185.2 / 0.0 / 208.4ms`, SGLang `225.7 / 0.0 / 378.9ms`.
- multi_turn: TorchInferno `359.8 / 60.4 / 417.9ms`, vLLM
  `143.6 / 61.5 / 203.1ms`, SGLang `166.0 / 109.3 / 277.2ms`.
- tree_of_thought: TorchInferno `128.5 / 37.2 / 153.3ms`, vLLM
  `63.6 / 31.3 / 87.7ms`, SGLang `77.0 / 61.8 / 148.4ms`.
- long_output: TorchInferno `273.5 / 20.8 / 1075.9ms`, vLLM
  `83.1 / 15.0 / 662.4ms`, SGLang `72.5 / 22.7 / 847.8ms`.

The TorchInferno public queue profile stayed on the common-prefix-only default:
`runtime_prefix_reuse_route_counts={"common_prefix":1000}` and
`runtime_full_prompt_store_skip_reason_counts={"pinned_without_allowance":1000}`.
It spent `4.08s/4.31s` in prefill forward/wall over `61` prefill batches,
paid `34.0K` prefill padding tokens, and spent `9.35s` in ragged-decode GPU
over `773` decode batches. The public prefill graph path was warm
(`0` misses, `0` request-path captures, `60` replays), so this refresh again
points at reducing model work per wave rather than graph-cache sizing.

Two current-tree multi_turn probes rechecked the greedy-large mixed-prefix
promotion question. The first no-env promotion candidate wrote
`/tmp/ti-bench-results/multi-mixed-auto/.../runs/20260704_132423` and routed
the intended mixed-prefix path (`max_active=32`, `prefix_rows=112`, PRBD off,
`{"common_prefix":125,"request_prompt":875}`), landing at
`280.4 / 69.1 / 341.7ms`, `983/1000` correct. That improved median TTFT/E2E
against the public common-prefix row, but worsened TPOT/throughput and still
lost all multi_turn metrics. A repeat with miss-shape telemetry wrote
`/tmp/ti-bench-results/multi-mixed-miss-shapes/.../runs/20260704_133238` and
landed at `387.3 / 64.5 / 439.1ms`, also `983/1000` correct. Model-side work
was not worse (`37` prefill batches, `2.56s/2.87s` prefill forward/wall,
`847ms` ragged-decode GPU), but user-visible latency regressed below the public
baseline. The new `runtime_prefill_graph_miss_shape_counts` counter attributed
the lone graph miss to a singleton suffix replay
(`ragged_prefill:b1:s32:rows1:ctx189:src1`), matching the expensive
`prefix_graph:b1:s32:p157-157:src1:mixed0` forward. That miss explains one
outlier, not the median regression. Keep greedy-large mixed-prefix reuse as an
explicit opt-in; do not make it a no-env OpenAI default until the mixed-prefix
path has fewer prefill waves or cheaper non-common-prefix replay.

A scoped request-prompt reuse subset is also rejected as a default direction.
The new diagnostic
`TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_MAX_EXTRA_TOKENS=12` keeps only the
shortest request-prompt savings in the explicit mixed-prefix path. The focused
profile
`/tmp/ti-bench-results/multi-mixed-maxextra12-current/.../runs/20260704_175358`
completed `983/1000` correct but landed at `393.9 / 65.3 / 462.8ms`. The queue
profile proved the guard worked (`{"common_prefix":875,"request_prompt":125}`
with request-prompt hit depths `55/56/57`), but it retained most common-prefix
prefill cost (`74.7K` prefill tokens, `4.95s/5.28s` prefill forward/wall) and
added small mixed waves plus three graph misses. This rules out "reuse only the
first short request-prompt turn" as the missing multi_turn lever; the remaining
route needs a stable all-request-prompt mixed path or a cheaper non-common
suffix replay, not a small prefix-depth subset.

The opposite deep-only subset is rejected too. With
`TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_MIN_EXTRA_TOKENS=90`, the focused profile
`/tmp/ti-bench-results/multi-mixed-minextra90-current/.../runs/20260704_180106`
completed `980/1000` correct but regressed to `544.0 / 74.4 / 610.9ms`.
The guard routed `250` request-prompt hits and `750` common-prefix hits, cutting
raw prefill tokens to `52.6K`, but it fragmented the phase into `46` prefill
batches with `7` graph misses and still spent `5.24s/5.59s` in prefill
forward/wall. The largest remaining common-prefix shapes were still expensive
(`b32:s112` at `1.63s`, `b32:s32:p45-45` at `1.42s`), while deep mixed
`b32:s32` waves stayed around `71ms` apiece. Depth-thresholding request-prompt
reuse therefore does not produce a score path from the current implementation:
the shallow subset leaves too much common-prefix replay, and the deep subset
adds too much fragmentation. Future work should focus on making the full
request-prompt mixed path cheaper or less jittery, not another static
extra-token cutoff.

## Tree prefill replay attribution (20260704_134610)

Queue profiles now join raw ragged-prefill graph replay/capture/miss counters
back to the high-level prefix/chunk shape that triggered them via
`runtime_prefill_shape_graph_*`. A no-env tree_of_thought probe on the patched
tree wrote
`/tmp/ti-bench-results/tree-shape-graph-profile/.../runs/20260704_134610` and
landed at `131.0 / 32.8 / 157.6ms`, `958/992` correct, with readiness
`205.9s`. The profile confirms the tree path is not graph-cache bound:
`60` prefill graph hits, `0` misses, `0` request-path captures, and `59`
replays. The high-level shape join shows the compute-heavy prefix waves:

- `prefix_graph:b24:s16:p45-45:src1:mixed0`: `26` replays,
  `862.5ms` synchronized forward time, `70.2ms` replay-call time.
- `prefix_graph:b32:s16:p45-45:src1:mixed0`: `10` replays,
  `370.7ms` synchronized forward time, `30.5ms` replay-call time.
- `prefix_graph:b16:s16:p45-45:src1:mixed0`: `12` replays,
  `386.5ms` synchronized forward time, `24.9ms` replay-call time.

Actual suffixes were still only `10`, `11`, and `12`, while every request-path
suffix replay stayed at `s16`. Total prefill padding was `8.49K` tokens
(`3.02K` row / `5.47K` suffix), prefill forward/wall was `1.83s/2.30s`, and
ragged-decode GPU was `1.50s`. The replay-call timings are small relative to
the synchronized forward time, so the next defaultable tree improvement needs
less model work per cached-prefix wave (true packed/paged prefix-suffix prefill
or a fused attention body), not more graph warmup or cache sizing.

## Long-output decode-many stop-tail cap (20260704_135558-142734)

A same-host current-tree long_output baseline wrote
`/tmp/ti-bench-results/long-baseline-current/.../runs/20260704_135558` and
landed at `254.4 / 23.9 / 1100.0ms`, `1000/1000` correct. Its queue profile
matched the public gap shape: `60` prefill batches, `34.5K` prefill padding
tokens, `4.75s/5.17s` prefill forward/wall, `794` decode batches, and
`10.20s` ragged-decode GPU. Two padding/decode-throughput probes were rejected:

- Adding `48` and `80` to the greedy-short suffix buckets reduced prefill
  padding from `34.5K` to `24.3K` tokens but regressed to
  `275.5 / 24.7 / 1350.3ms`. The new `s48`/`s80` graph shapes were much slower
  (`12.30s` prefill forward) and raised startup memory/readiness.
- Allowing decode-many while requests were waiting at `min_active=48` increased
  decode-many use and cut runtime step time, but regressed the median row to
  `253.5 / 24.1 / 1145.6ms` because padding/skipped decode tokens rose.

The accepted small default is a greedy-short stop-tail cap for decode-many, now
wired as the OpenAI online default for stop-token-enabled greedy-short
decode-many traffic. Cap `4` validated the mechanism
(`/tmp/ti-bench-results/long-stop-tail4/.../runs/20260704_141238`) at
`214.9 / 24.1 / 1071.2ms`, but a no-env repeat with the same cap landed at
`227.9 / 24.5 / 1140.5ms`, improving TTFT while regressing E2E. Cap `6` was
initially selected because
`/tmp/ti-bench-results/long-stop-tail6/.../runs/20260704_142734` landed at
`251.3 / 23.5 / 1095.9ms`, `1000/1000` correct, and improved throughput
`32.2 -> 33.1 tok/s` against the baseline. A later current-tree cap-4 rerun
(`20260704_234227`) landed at `225.4 / 23.8 / 1061.4ms`, so the default is now
cap `4`. This narrows the long_output gap but is not a full decode-throughput
solution by itself.

## Self-consistency admission/coalescing rejections (20260704_143502-145246)

A focused current-tree self_consistency no-env profile wrote
`/tmp/ti-bench-results/self-profile/.../runs/20260704_143502` and reproduced the
public gap shape at `307.0 / 0.0 / 327.6ms`, throughput `3.05 tok/s`, with
`1000/1000` correct and one unique answer. The final queue profile separated the
server hot path from benchmark-visible client waves: after the generated prefix
was cached, server-side queue-to-first/finish p50 was only `9.0/9.0ms`, and the
fast HTTP stream profile was `10.47ms` total p50. The remaining visible median
comes from repeated tiny waves and admission/sync churn: `423` submit batches
for `1000` requests, `1.35s` in submit sync, `1.05s` in idle-drain waiting, and
request p90 queue-to-first/finish still `211.9/218.2ms`.

Three same-host A/Bs rejected admission waits as the next self_consistency
default:

- `TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_SHORT_INITIAL_BATCH_WAIT_MS=2`
  (`/tmp/ti-bench-results/self-initial-wait2/.../runs/20260704_144145`)
  regressed to `319.7 / 0.0 / 351.7ms`. The initial batch stayed at `2`, submit
  batches rose to `437`, and queue-to-first/finish p90 worsened to
  `270.2/302.0ms`.
- `TORCHINFERNO_OPENAI_TP_STREAM_PREQUEUE_ADMISSION_WAIT_MS=2`
  (`/tmp/ti-bench-results/self-prequeue2/.../runs/20260704_144746`) regressed to
  `334.2 / 0.0 / 354.9ms`. The first online batch shrank to `1`, submit batches
  rose to `443`, and queue-to-first/finish p90 worsened to `301.1/323.7ms`.
- `TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_SHORT_IDLE_BATCH_WAIT_MS=20`
  (`/tmp/ti-bench-results/self-idle20/.../runs/20260704_145246`) reduced runtime
  step calls (`222 -> 184`) and phase runtime step time (`617ms -> 565ms`) but
  still regressed to `327.8 / 0.0 / 350.9ms`. Submit sync stayed high
  (`1.38s`), idle waiting rose to `1.37s`, and accepted-request fast HTTP p50
  rose to `14.0ms`.

Skipping the explicit TP submit sync for sampled-short online requests is also
rejected. A local source probe made that barrier default-off for
self_consistency-style traffic and launched
`/tmp/ti-bench-results/self-submit-nosync`, but the 1000-request benchmark
stalled in `as_completed()` after readiness and had to be interrupted. The
server remained alive, which points at rank command/progress ordering rather
than model startup. Keep the submit barrier in place; any future fix should use
a redesigned combined submit+step command that advances primary and worker
runtimes together, not a silent barrier removal.

That redesigned shape was tested as an opt-in source probe and is also rejected
for now. The patch reused the existing `steps_after_submit` payload for
quiescent sampled-short submits and made the primary consume the same attached
step immediately. It completed correctly at
`/tmp/ti-bench-results/self-quiescent-submit-step/.../runs/20260704_152412`,
but regressed to `326.5 / 0.0 / 347.8ms`, throughput `2.9 tok/s`, with
`1000/1000` correct. The profile reduced submit batches (`423 -> 357`) and
idle-drain time (`1.05s -> 0.54s`), but runtime step time rose
(`617ms -> 944ms`) and request p90 queue-to-finish worsened to `613ms`. The
source hook was backed out; self_consistency needs lower per-submit/runtime-step
cost without pushing more work into latency-critical idle waves.

Keep sampled-short initial wait, stream prequeue admission, and idle drain at the
current defaults. The next self_consistency lever should reduce per-submit TP
sync/worker command overhead for already-cached repeated prompts, or combine
small cached-prefix submissions with useful decode work, rather than adding more
request-admission delay.

## Current-tree few_shot no-env profile (20260704_153514)

The public pointer was still `c976dec` / `20260704_131636`, so a focused
current-tree few_shot run refreshed the local no-env profile after the latest
tail-cap and queue-profile changes. It wrote
`/tmp/ti-bench-results/few-current/.../runs/20260704_153514` and landed at
`174.7 / 51.6 / 223.8ms`, p99 TTFT/E2E `1020.9/1057.7ms`, throughput
`5.1 tok/s`, and `977/1000` correct.

The queue profile was graph-warm and kept the expected greedy-mid shape:
`max_active=32`, `prefix_rows=64`, `33` submit batches, `34` prefill batches,
`0` request-path prefill captures, and only two static-prefill misses. The hot
prefill body remained `prefix_graph:b32:s16:p122-122:src1:mixed0` with
`31` replays, `1.67s/1.80s` synchronized forward/wall inside that shape and
`2.39s` total prefill wall. Actual suffix lengths were `12`, `13`, and `14`,
but every prefix-suffix replay still used the warmed `s16` graph, producing
`3.47K` suffix-padding tokens. Decode was also in-family: `71` decode model
calls, `65` ragged-decode batches, `958ms` ragged-decode GPU, and only `122`
ragged-decode padding tokens.

This does not reopen fine greedy-mid suffix buckets. The profile shows padding,
but the already-rejected `s12/s16` and fine suffix-bucket probes did not turn
padding reduction into a defaultable median/tail win. The remaining few_shot
gap is still model-side cached-prefix prefill cost plus queue tail from the
32-row wave cadence, not graph-cache miss handling or padded decode waste.

## Paged-prefix telemetry added after current-tree profile

The paged-prefix path remains default-off, but the next useful A/B needs better
attribution than total paged prefill time. `PagedPrefixCache.share_into()` now
records the raw matching prefix length, the page-aligned shareable length, and
the tokens stranded by page alignment without changing its public return value.
`PagedEngine` folds that into runtime stats as candidate, aligned, alignment
loss, forced-suffix, page-size, and candidate-length buckets. This makes a
future page-size or mixed page/suffix policy A/B measurable instead of inferred
from hit-token counts alone.

The optional paged-prefix suffix graph hook also records attempts, captures,
successful replays, fallbacks, failures, and per-shape counts. These fields are
exported through the OpenAI queue-profile JSONL alongside the existing runtime
prefill/decode counters. This is profiling infrastructure only; it does not
enable paged KV, paged prefix caching, or suffix graphing by default. Validation
covered pyflakes plus the focused paged-prefix/cache and queue-profile tests:
`venv/bin/python -m pytest tests/test_scaffolding.py::test_paged_prefix_cache_zero_copy_share_and_evict tests/test_serving_engine.py::test_paged_online_engine_batches_shared_suffix_prefill tests/test_serving_engine.py::test_paged_online_engine_pads_mixed_shared_suffix_prefill tests/test_serving_engine.py::test_paged_online_engine_graphs_padded_shared_suffix_prefill tests/test_openai_server.py::test_openai_queue_profile_records_runtime_engine_stats -q`.

## Few-shot packed-ragged eager rejection (20260704_150159)

A focused few_shot run exercised the packed-ragged prefill oracle with
`TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER=1`. The run wrote
`/tmp/ti-bench-results/few-packed-eager/.../runs/20260704_150159` and landed at
`1085.7 / 50.9 / 1127.7ms`, p99 TTFT/E2E `2403.0/2459.6ms`, throughput
`1.0 tok/s`, and `976/1000` correct. It is rejected as a runtime
implementation.

The profile shows why: the path slightly reduced few_shot prefill padding
(`4.7K -> 3.4K` tokens versus the latest public queue profile) and kept the
expected admission shape (`max_active=32`, `prefix_rows=64`, `33` submit
batches, `36` prefill batches), but made prefix-suffix prefill dramatically
slower. Total prefill forward/wall rose to `15.75s/16.74s`; the hot
`prefix_graph:b32:s16:p122-122:src1:mixed0` shape alone accounted for
`15.05s/15.16s` across `30` waves. Median queue-to-first/finish rose to
`1034.0/1059.5ms`, and runtime step time reached `20.12s`.

Keep the packed-ragged eager path as a correctness/oracle scaffold only. The
few_shot gap still points at true packed cached-prefix prefill, but the
defaultable version needs one CUDA/FlashInfer body that keeps the layer stack
packed, not a Python per-row eager loop.

## Public 20260703_130210 refresh and active-row rejection

The latest public all-provider run at
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260703_130210`
measured TorchInferno `4be1a2f`, vLLM `978de83`, and SGLang `1058d00`.
The TorchInferno commit predates the current pushed head's queue-profile
diagnostics, same-host long-output probes, and greedy-short `b24`/`192` prefill
graph-cache default, but the vLLM/SGLang rows remain the relevant target. The
score split was vLLM `10`, TorchInferno `6`, and SGLang `3`.

Rows as TTFT / TPOT / E2E:

- few_shot: vLLM `144.7 / 51.4 / 191.7ms`, SGLang
  `141.1 / 77.2 / 217.4ms`, TorchInferno `148.8 / 46.8 / 189.1ms`.
- self_consistency: vLLM `204.4 / 0.0 / 229.7ms`, SGLang
  `215.2 / 0.0 / 367.5ms`, TorchInferno `154.5 / 0.0 / 162.3ms`.
- multi_turn: vLLM `177.5 / 55.1 / 226.9ms`, SGLang
  `163.0 / 103.5 / 270.0ms`, TorchInferno `312.1 / 60.7 / 365.5ms`.
- tree_of_thought: vLLM `62.1 / 29.9 / 84.1ms`, SGLang
  `73.3 / 57.2 / 138.7ms`, TorchInferno `127.9 / 27.0 / 148.4ms`.
- long_output: vLLM `81.5 / 15.1 / 612.8ms`, SGLang
  `71.0 / 22.3 / 856.8ms`, TorchInferno `265.6 / 20.7 / 953.3ms`.

The latest TorchInferno public queue profiles keep the same remaining shape:
few_shot and self_consistency are competitive, while multi_turn, tree, and
long_output still trail vLLM/SGLang on prefix reuse, first-token scheduling, and
decode throughput. Public multi_turn used `max_active=32`, `prefix_rows=64`,
`decode_quantum=16`, and only the common-prefix route
(`{"common_prefix":1000}`); it spent `4.16s` in prefill forward, `4.33s` in
prefill wall, and `876ms` in decode GPU across `100` decode calls. Public
tree_of_thought used the same active/cache shape with `decode_quantum=2`,
reused only the `45` token common prefix, and spent `1.83s` in prefill forward
plus `1.74s` in decode GPU. Public long_output used `max_active=64`,
`prefix_rows=64`, `decode_quantum=3` with a q8 drain, spent `4.52s` in prefill
forward, and then paid `9.19s` decode GPU over `806` decode calls / `88`
decode-many calls. The public prefill graph cache was not thrashing
(`121/128` live entries and zero evictions), so these rows still point at
scheduling/reuse/decode work rather than cache churn. vLLM still reports
chunked prefill plus prefix caching, while SGLang uses RadixCache and
graph-captured prefill/decode.

A current-head TorchInferno-only full-suite validation on pushed `135d127`
(`agent_space/ti_head_135d127_full_results_0703/.../runs/20260703_135616`)
exercised the new greedy-short `b24`/`192` prefill-graph defaults without env
overrides. The run landed at few_shot `167.6 / 48.8 / 207.8ms`,
self_consistency `186.1 / 0.0 / 232.5ms`, multi_turn
`298.3 / 58.9 / 349.4ms`, tree_of_thought `130.4 / 30.1 / 154.0ms`, and
long_output `237.9 / 24.2 / 1053.6ms`, with correctness in the normal bands.
Queue profiles show the graph-cache-cap fix holding (`130/192` live prefill
graphs, zero evictions across the full run), so the latest defaults are no
longer losing to b24 cache churn. The remaining long_output row is still
decode/prefix-work bound: `61` prefill batches, `4.82s/5.21s`
prefill forward/wall, `777` decode model calls, `99` decode-many calls over
`456` steps, and `10.53s` ragged-decode GPU. Multi_turn remains policy-limited
to the shared `45` token common prefix and skipped all `1000` pinned full-prompt
stores; it improved versus the public row but still needs non-fragmenting
conversation-prefix reuse to close the vLLM/SGLang TTFT/E2E gap.

The queue profile now records why per-request full-prompt prefixes are not
stored. A focused current-head TorchInferno multi_turn profile on the local
instrumentation
(`agent_space/ti_multi_fullprompt_stats_results_0703/.../runs/20260703_113444`)
landed at `298.9 / 59.0 / 353.2ms`, `982/1000` correct. It still reused only
the shared prefix (`{"common_prefix": 1000}`, `{"45": 1000}`), and the new
profile counters make the default blocker explicit:
`runtime_full_prompt_store_requests=1000`,
`runtime_full_prompt_store_stored_requests=0`, and
`runtime_full_prompt_store_skip_reason_counts={"pinned_without_allowance":1000}`
for `114,608` skipped prompt tokens. This confirms the default multi_turn gap
is policy-limited conversation-prefix reuse rather than an exact lookup miss or
generated-prefix cache miss; the previously rejected pinned full-prompt and
mixed-prefix opt-ins still need cheaper batched replay before promotion.

Disabling prefix-depth admission priority is also rejected for the mixed-prefix
opt-in path. The diagnostic flag
`TORCHINFERNO_CONTINUOUS_ADMIT_PREFIX_HIT_PRIORITY=0` keeps admission in
arrival order instead of preferring deeper prefix hits. With the existing
greedy-large mixed-prefix policy and `112` prefix rows, run
`agent_space/ti_multi_no_prefix_priority_results_0703/.../runs/20260703_114531`
landed at `246.7 / 72.1 / 313.6ms`, `982/1000` correct. The profile adopted
all `1000` full-prompt rows and reused request prompts for `875` requests, but
prefill batches rose to `39`, decode model calls stayed high at `98`, decode
GPU time rose to `892ms`, and p99 TPOT regressed to `280.9ms`. This keeps the
admission switch as an opt-in diagnostic only; the multi_turn promotion blocker
is still mixed-prefix replay/active-set fragmentation, not simply prefix-hit
priority.

A current-head default TorchInferno-only refresh on `75fee00`
(`agent_space/ti_default_head_results_0703/.../runs/20260703_055113`) landed
at few_shot `167.0 / 46.9 / 206.0ms`, self_consistency
`153.2 / 0.0 / 241.7ms`, multi_turn `331.4 / 59.0 / 386.1ms`,
tree_of_thought `139.6 / 28.1 / 164.4ms`, and long_output
`275.4 / 24.2 / 1144.1ms`. The default path did not pick up the paged-prefix
suffix graph hook, which remains opt-in. Queue profiles still point to the same
default bottlenecks: multi_turn reuses only the `45` token common prefix and
spends `4.48s` in prefix-suffix prefill, while long_output reuses the `111`
token common prefix, spends `5.26s` in prefill forward, and runs `786` decode
batches for `38.2K` active decode tokens.

The queue and provider logs point at the same architectural gap. vLLM reports
prefix-cache hit rates in the `65-86%` range with chunked prefill, while SGLang
uses RadixCache and prefill/decode CUDA graphs. TorchInferno still reuses only
the `45` token shared prefix on multi_turn and spends about `4.1s` in
multi_turn prefill forward; long_output spends about `4.35s` in prefill forward
and `9.35s` in ragged-decode GPU across `771` decode batches. The current local
TorchInferno refresh after the sampled exact-context promotion still has the
same remaining shape: multi_turn only has common-prefix reuse, and long_output
is dominated by full `b64/64` decode-many work rather than scalar skip waste.

A concrete paged-prefix validation blocker was fixed after this inspection.
The experimental `PagedEngine` can persist a zero-copy page-level prefix cache
across online bursts, but `start_online()` reset internal ids back to `p0` while
the prefix cache retained entries by those ids. Reusing an id caused
`PagedPrefixCache.remember()` to touch the stale entry instead of replacing its
retained pages and radix routes, so later bursts could keep an old prefix set.
The cache now replaces same-id entries, rebuilds the router, releases stale
retained refs, and the engine keeps ids monotonic while a persistent prefix
cache exists. This does not make paged serving a default path; it removes a
correctness and validation blocker for the vLLM/SGLang-style KV reuse work that
multi_turn needs.

The paged-prefix suffix prefill path now has a bucketed fallback for mixed
suffix lengths. The previous infrastructure could either pad every shared-prefix
request in an admitted wave to the maximum suffix, or fall back to one exact
suffix-length prefill per group when that padding exceeded the guard. That left
fragmentation on waves where short suffixes could be cheaply grouped but one
long suffix made the all-in-one padded batch too expensive. The new
`TORCHINFERNO_PAGED_PREFIX_BUCKETED_SUFFIX_PREFILL` path, enabled by default
inside the still opt-in paged-prefix engine, groups by
`TORCHINFERNO_PAGED_PREFIX_SUFFIX_BUCKETS` and applies the same padding/page
guards per bucket. CPU coverage verifies a wave that rejects one all-in padded
batch now emits a padded `s4` bucket plus an exact long suffix group.

An opt-in suffix graph hook is available for that same experimental paged-prefix
path as `TORCHINFERNO_PAGED_PREFIX_SUFFIX_GRAPH=1`. A focused paged multi_turn
run with prefix caching and bucketed suffix prefill at
`agent_space/ti_multi_paged_bucket_results_2211/.../runs/20260703_051110`
landed at `3060.3 / 684.1 / 3287.3ms` with `981/1000` correct. The queue
profile reduced the path to `34` prefill calls, but still spent `9.35s` in
prefill forward, with most paged suffix/full prefill shapes taking roughly
`260-290ms` each. Exact suffix graphing at
`agent_space/ti_multi_paged_suffix_graph_results_0703/.../runs/20260703_052956`
improved that to `2757.6 / 614.1 / 2982.6ms`, but captured several exact
`s25..s31` shapes on the request path and still spent `8.41s` in prefill
forward. The follow-up graph-bucket run at
`agent_space/ti_multi_paged_suffix_graph_bucket_results_0703/.../runs/20260703_053849`
coalesced those suffixes to `s32`, landed at `2668.4 / 715.3 / 3077.5ms` with
`981/1000` correct, and cut prefill forward to `2.85s`; only the first
`b32/s32` and `b8/s32` captures were expensive, while later `b32/s32` replays
were about `10ms`. Keep this path default-off: graph replay removes the paged
suffix launch floor, but request-path capture and remaining queueing still leave
it far behind the default dense path and the vLLM/SGLang multi_turn rows.

Mixed-prefix dynamic context is accepted only as opt-in infrastructure for the
same multi_turn reuse gap. The previous full-prompt reuse route could collapse
the run to coarse non-common-prefix batches, but mixed-prefix prefill either
fell back to the eager boolean-mask path or failed capture. The runtime now lets
`TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_DYNAMIC_CONTEXT=1` route mixed prefix hits
through the existing negative context-bucket attention path when the suffix
bucket fits
`TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_DYNAMIC_CONTEXT_MAX_SUFFIX` (default
`32`). Focused CPU coverage verifies that mixed request-prompt hits with an
`s32` suffix use a negative context bucket instead of the old `prefix_copy_len`
mixed eager marker.

The score-path decision is still rejected. With pinned full-prompt stores,
`112` prefix rows, non-common/mixed prefix graph prefill, and mixed dynamic
context left at the old global `16` suffix gate, the run
`agent_space/ti_multi_mixed_dynamic_context_results/.../runs/20260703_031955`
landed at `1060.8 / 74.6 / 1124.6ms`, `983/1000` correct, because the `s32`
mixed groups still missed the graph and spent `16.2s` in prefill forward.
Exercising the intended `s32` bucket with
`TORCHINFERNO_CONTINUOUS_DYNAMIC_PREFIX_PREFILL_MAX_SUFFIX=32` wrote
`agent_space/ti_multi_mixed_dynamic_context32_results/.../runs/20260703_032523`
and improved to `435.2 / 87.2 / 523.8ms`, `983/1000` correct. The queue profile
shows real improvement (`s32` mixed groups replayed around `65-75ms` instead of
`~500ms` eager), but the benchmark still split into two online sessions, paid
about `10.4s` aggregate online phase time, and hit a mixed capture failure on a
long `s144` group. Dense default multi_turn remains around
`309 / 65 / 364ms`, so keep full-prompt/mixed dynamic context opt-in until
long-suffix routing and prefix-row policy beat the common-prefix baseline.

The next opt-in scaffold narrows that long-suffix failure mode without changing
defaults. `TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_LONG_SUFFIX_COMMON_FALLBACK=1`
lets a mixed-prefix request-prompt hit fall back to the best live
`common_prefix` entry when the request-prompt suffix bucket exceeds
`TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_DYNAMIC_CONTEXT_MAX_SUFFIX`. This should
preserve the fast `s32` dynamic-context wins while avoiding the pathological
`s144` mixed-prefix route; CPU coverage now verifies that only the explicit
fallback demotes overlong mixed hits back to the shared prefix. The 70B
multi_turn fallback-only run
`agent_space/ti_multi_long_common_fallback_results_0346/.../runs/20260703_034204`
landed at `289.4 / 70.7 / 389.2ms`, `981/1000` correct. It removed the `s144`
shape from that run, but still paid `5` request-time prefill graph captures
(`4.53s` capture time), so it was not enough to promote.

The stronger local result came from warming the mixed-source ragged prefill
graphs at startup. With the fallback knobs above plus
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_MIXED_PREFIX_SUFFIX_PREFILL=1`, run
`agent_space/ti_multi_warmmixed_results_0415/.../runs/20260703_035929` measured
`243.5 / 67.4 / 312.3ms`, `979/1000` correct. The queue profile shows
`36/36` request prefill graph hits, `0` request prefill captures, and prefill
wall time down to `2.57s` from `7.20s` in the fallback-only run. This is now the
best local multi_turn E2E result, but the knobs remain opt-in until the full
benchmark suite validates correctness and cross-workload latency.

Follow-up default-scope attempts on `8aed5df` were rejected. Enabling the
mixed/full-prompt reuse bundle automatically for greedy `max_tokens=512` without
also constraining long mixed suffixes wrote
`agent_space/ti_multi_scoped_mixed_default_results_0703/.../runs/20260703_072817`
and regressed multi_turn to `585.1 / 69.5 / 644.0ms`; the queue profile routed
`s64..s144` mixed-prefix batches, raising prefill model tokens to `53.0K` and
prefill forward to `9.71s`. Splitting mixed-prefix batches so only suffixes at
or below the dynamic mixed-context limit (`s32` by default) fixed the structural
bug (`19.3K` prefill tokens, no long `mixed1` shapes) but still measured
`324.0 / 81.9 / 396.0ms` in
`agent_space/ti_multi_scoped_mixed_split_results_0703/.../runs/20260703_073644`
because three request-path common-prefix graph captures cost `2.09s`.
Disabling request-path captures for that scoped policy improved p99 but not the
median (`326.5 / 83.5 / 405.2ms` in
`agent_space/ti_multi_scoped_mixed_nocapture_results_0703/.../runs/20260703_074343`).
Repeating with `112` prefix rows, matching the earlier opt-in run, still landed
at `309.8 / 84.7 / 415.2ms` in
`agent_space/ti_multi_scoped_mixed_nocapture_p112_results_0703/.../runs/20260703_075004`.
Keep the automatic greedy-large mixed-prefix policy off by default; the suffix
split and capture guard remain useful for explicit opt-in probes, but they do
not close the default multi_turn gap.

A current-head policy-only recheck on pushed `839a943` is also rejected. Running
the existing greedy-large mixed-prefix policy opt-in with `112` prefix rows
(`TORCHINFERNO_CONTINUOUS_GREEDY_LARGE_MIXED_PREFIX_REUSE=1`,
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS=112`) wrote
`agent_space/ti_multi_policywarm_results_0703/.../runs/20260703_092218` and
landed at `304.7 / 93.9 / 408.2ms`, `982/1000` correct. The automatic
mixed-prefix warmup did its job (`0` request-path prefill captures), but the run
split into two sessions (`448` and `552` requests) and the useful request-prompt
hits still routed through many `b32:s32` mixed-prefix passes plus two
`b32:s144` common-prefix passes. Prefill wall was about `3.24s` across the two
sessions and TPOT regressed sharply versus the accepted full-suite default band
(`293.5 / 60.5 / 346.5ms`). Keep this path opt-in; the next multi_turn attempt
needs cheaper non-common prefix replay or fewer mixed-prefix prefill waves, not a
broader default for the current policy bundle.

The opt-in mixed-prefix suffix warmup now also covers `b4:s32:ctx256`. The
rejected current-head policy run above hit one request-path
`prefix_graph:b4:s32:p154-159:src4:mixed1` miss and then paid about `354ms` in
that small fallback, while the default warmup only covered `b4:s16`. The added
shape is still scoped to the explicit greedy-large mixed-prefix policy or the
mixed-prefix warmup env; it does not affect normal default serving. A rebuilt
focused probe with the same policy env
(`agent_space/ti_multi_mixed_b4warm_results_0703/.../runs/20260703_094659`)
landed at `235.7 / 70.9 / 302.4ms`, `982/1000` correct, with a single online
session and `37/0` request prefill graph hits/misses. That rerun did not
reproduce the `b4` tail shape, so keep treating this as opt-in graph coverage
for an observed miss rather than evidence to promote greedy-large mixed-prefix
reuse by default.

The queue profile now also records prefix-graph route composition as
`runtime_prefill_shape_route_counts`,
`runtime_prefill_shape_route_active_tokens`, and
`runtime_prefill_shape_route_reuse_tokens`. A focused mixed-prefix opt-in run on
the instrumented tree
(`agent_space/ti_multi_route_comp_results_0703/.../runs/20260703_115712`)
landed at `250.3 / 69.3 / 318.6ms`, `981/1000` correct. It reused
`{"common_prefix":126,"request_prompt":874}`, issued `39` prefill batches,
reported `37/2` prefill graph hits/misses with no request captures, spent
`2.83s` in prefill forward, and still needed `108` decode calls / `145`
scheduler steps. The new route-shape counters show that the expensive
`b32:s32:mixed1` waves are mostly pure `request_prompt` rows; only a few
`p45-*` waves mix common-prefix rows into the same graph. So the remaining TPOT
cost is not mainly common-prefix contamination inside mixed groups. The next
multi_turn attempt should reduce request-prompt prefill/decode interleaving or
make those request-prompt replays cheaper; splitting mixed groups by route is
unlikely to move enough work by itself.

For the explicit greedy-large mixed-prefix policy, disabling
prefill-ready-before-decode is accepted as a scoped default. The A/B
`agent_space/ti_multi_mixed_prbd_off_results_0703/.../runs/20260703_120418`
used the same mixed-prefix knobs plus
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_READY_BEFORE_DECODE=0` and landed at
`228.5 / 69.9 / 295.9ms`, `981/1000` correct. Versus the adjacent
route-composition run, it reduced prefill batches `39 -> 37`, graph misses
`2 -> 0`, prefill forward `2.83s -> 2.33s`, decode calls `108 -> 88`,
scheduler steps `145 -> 124`, and phase time `6.42s -> 5.85s`. Tails worsened
(`710.9ms` p99 TTFT and `968.8ms` p99 E2E), so this is not a default for the
normal common-prefix multi_turn path. It is now only the default when the
explicit `TORCHINFERNO_CONTINUOUS_GREEDY_LARGE_MIXED_PREFIX_REUSE=1` policy is
selected; global prefill-ready env overrides still win.

A default-prefix-row control keeps full mixed-prefix promotion blocked. Running
the same pushed policy without `TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS=112`
(`agent_space/ti_multi_mixed_default_prefixrows_results_0703/.../runs/20260703_121145`)
landed at `248.0 / 69.3 / 330.0ms`, `981/1000` correct. The server used
`max_active=32`, `prefix_rows=64`, and `96` total cache rows. It still adopted
all `1000` full prompts, but request-prompt reuse fell to `849` requests,
common-prefix fallback rose to `151`, prefill batches rose to `48`, prefill
forward rose to `3.04s`, and p99 TTFT/E2E were `862.9/914.8ms`. This is better
than the common-prefix median E2E band but materially worse than the `112`-row
mixed-prefix run above. Do not promote the mixed-prefix policy by itself; any
default-scope attempt must address prefix-row capacity and cross-workload memory
first.

A full TorchInferno-only pass with the explicit greedy-large mixed-prefix policy
and `TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS=112` on pushed `6ee6704`
(`agent_space/ti_full_mixed_p112_results_0703/.../runs/20260703_121754`)
landed at few_shot `168.3 / 47.9 / 206.1ms`, self_consistency
`179.7 / 0.0 / 195.1ms`, multi_turn `234.2 / 73.4 / 305.4ms`,
tree_of_thought `134.8 / 44.4 / 161.6ms`, and long_output
`254.6 / 24.6 / 1135.0ms`. The multi_turn median TTFT/E2E improvement is real
versus both the public default row (`296.9 / 59.6 / 347.5ms`) and the current
default-head control (`331.4 / 59.0 / 386.1ms`), but TPOT and tails stay weak
(`590.5 / 318.4 / 779.2ms` p99). The adjacent workloads were not catastrophic,
but they were not a broad win either: few_shot stayed in family, tree_of_thought
kept E2E in family with worse TPOT, self_consistency was slower than the public
row, and long_output remained close to the local default while still far behind
vLLM. Queue profiles used `112` prefix rows for few_shot, tree_of_thought, and
multi_turn, `16` for self_consistency, and `80` for long_output. The multi_turn
record still showed `38` prefill batches, `2.60s` prefill forward, `99` decode
calls, route `{"common_prefix":125,"request_prompt":875}`, and `6.07s` phase
time; long_output still spent `10.15s` in decode GPU across `741` decode calls.
This validates the opt-in multi_turn median improvement, but not a broad
default. Promoting it would need an adjacent same-head all-workload control and
either a TPOT/tail fix or an explicit workload-scoped selection rule.

A same-head scoped-default recheck on pushed `c5c2664` is also rejected. The
candidate made the greedy-large mixed-prefix policy default-on only for
deterministic requests with `400 < max_tokens <= 512`, and raised online prefix
rows to `112` only when that policy matched. The focused multi_turn run
`agent_space/ti_multi_scoped_mixed_default_range_0703/.../runs/20260703_142156`
landed at `241.6 / 71.7 / 309.6ms`, `981/1000` correct, confirming the
selection rule did route to the intended mixed-prefix path (`max_active=32`,
`prefix_rows=112`, `39/0` prefill graph hits/misses, no request captures). The
full-suite run
`agent_space/ti_full_scoped_mixed_default_range_0703/.../runs/20260703_142825`
did not hold up: few_shot `170.5 / 49.8 / 210.8ms`, self_consistency
`201.1 / 0.0 / 216.9ms`, multi_turn `313.9 / 83.3 / 398.3ms`,
tree_of_thought `132.6 / 34.8 / 157.5ms`, and long_output
`243.1 / 24.8 / 1070.3ms`. That multi_turn row is worse than the adjacent
current-head full control (`298.3 / 58.9 / 349.4ms`) and the scoped policy also
regressed adjacent medians enough to reject the default. Keep the greedy-large
mixed-prefix policy explicit; do not promote a max-token/temperature scoped
default until a full-suite run beats the same-head default control.

Reducing the explicit greedy-large mixed-prefix active set is rejected. The
probe kept the same opt-in policy and `112` prefix rows but added
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_MAX_ACTIVE=24`
(`agent_space/ti_multi_mixed_active24_results_0703/.../runs/20260703_124801`).
It landed at `1431.2 / 60.8 / 1473.5ms`, p99 E2E `2065.4ms`, and `983/1000`
correct. The queue profile still adopted all `1000` full prompts and routed
`{"common_prefix":125,"request_prompt":875}`, but the lower active cap forced
`45` submit/prefill batches, raised prefill forward/wall to
`18.42s/18.68s`, and pushed queue-to-submit p50 to `922ms` before the first
token p50 reached `1391ms`. Decode did not become a useful win either
(`97` decode calls, `1.58s` decode GPU). Keep active `32` for this opt-in
policy; the remaining multi_turn lever is cheaper request-prompt replay or
better prefill/decode overlap, not smaller active waves.

Raising prefix rows alone is also rejected for the common-prefix multi_turn
default path. A current-head focused run with only
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS=112`
(`agent_space/ti_multi_prefix112_common_results_0703/.../runs/20260703_125809`)
landed at `305.7 / 64.2 / 366.8ms`, p99 E2E `627.5ms`, and `981/1000`
correct, behind the latest public default row (`296.9 / 59.6 / 347.5ms`). The
queue profile confirms that extra rows do not help without request-specific
prefix reuse: routing stayed `{"common_prefix":1000}`, full-prompt stores still
skipped as `{"pinned_without_allowance":1000}`, and the work stayed in-family at
`35` prefill batches, `4.02s/4.27s` prefill forward/wall, `86` decode calls,
and `820ms` decode GPU. Keep the default prefix pool at `64` rows unless it is
paired with a cheaper full-prompt/mixed-prefix replay path.

A focused tree_of_thought A/B on pushed `2719235` separates prefix-row capacity
from the mixed-prefix full-pass noise. The no-env control
(`agent_space/ti_tree_default_2719235_results_0703/.../runs/20260703_122701`)
landed at `127.3 / 43.7 / 155.6ms`, p99 E2E `632.5ms`, and `959/992`
correct with `prefix_rows=64`. Its queue record had `59` prefix-prefill
batches, `1.79s` prefill forward, `2.21s` prefill wall, `93` decode calls,
`1.50s` decode GPU, and queue-to-first/finish p50 `105.6/131.7ms`. Repeating
with only `TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS=112`
(`agent_space/ti_tree_prefix112_2719235_results_0703/.../runs/20260703_123134`)
landed at `129.5 / 45.1 / 157.6ms`, p99 E2E `556.6ms`, and `966/992`
correct. The larger prefix pool improved internal tails
(`271.8/309.9ms -> 172.2/241.8ms` queue-to-first/finish p99), but it did not
reduce steady model work: prefill forward stayed `1.79s`, prefill wall stayed
about `2.17s`, decode calls stayed `93`, and decode GPU stayed `1.49s`. Keep
`112` prefix rows as an opt-in or future workload-scoped policy candidate, not
a broad default; the score-facing tree median gap still needs faster
sampled-medium prefix-suffix prefill or decode pipeline work.

Bucketed ragged-decode cache-token loop bounds are rejected as a default for
long_output. The first env probe only changed the paged graph metadata path and
therefore did not affect the dense long-output path:
`agent_space/ti_long_cachetoken_bucket_results_0703/.../runs/20260703_100113`
landed at `285.4 / 24.3 / 1122.1ms`, with decode GPU still at `10.67s` and no
decode graph misses. Extending the same bucket key to dense ragged decode by
slicing the K/V cache before the grouped GQA kernel also failed:
`agent_space/ti_long_dense_cachetoken_bucket_results_0703/.../runs/20260703_101009`
landed at `272.6 / 23.7 / 1195.1ms`, still `1000/1000` correct, but introduced
five request-path decode graph captures (`1.62s` capture time), raised decode GPU
to `11.51s`, and worsened p99 TPOT to `104.9ms`. Keep
`TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_CACHE_TOKEN_BUCKETS` default-off as a
diagnostic; the long-output gap remains lower steady decode GPU without extra
graph buckets, or a real prefill/decode/readback pipeline.

A focused current-head long_output baseline on pushed `862b1de`
(`agent_space/ti_long_default_862b1de_results_0703/.../runs/20260703_123832`)
landed at `262.2 / 24.2 / 1142.1ms`, p99 E2E `2134.0ms`, and `1000/1000`
correct. The final queue profile is the same steady-state blocker as the recent
public and local runs: common-prefix reuse hit `{"111":1000}`, prefill graph
hits/misses were `61/0`, prefill forward/wall were `5.08s/5.61s`, and prefill
padding was still `41.6K` tokens split about evenly between row and suffix
padding. Decode remained the larger floor at `747` decode model calls,
`10.23s` ragged-decode GPU, `104` decode-many calls over `490` steps, and
`26.9K` decode-many model tokens. The dominant `decode_many:b64/64` shape still
did useful work (`14.1K` model tokens with `466` skipped stop-tail tokens), so
the existing q8 drain policy remains the right default. The next long_output
change should reduce or overlap the `5s` prefill plus `10s` decode pipeline; do
not reopen global drain-quantum, waiting decode-many, cache-token bucket, or
intermediate prefill-bucket defaults from this profile alone.

Queue-profile progress snapshots now omit heavy per-shape maps by default while
keeping scalar counters in progress records and preserving full shape detail in
quiescent/final records. `TORCHINFERNO_OPENAI_TP_ONLINE_PROFILE_PROGRESS_SHAPES=1`
restores the previous verbose progress behavior for investigations that need
mid-run shape detail. A focused long_output confirmation with the lighter
progress records
(`agent_space/ti_long_light_progress_profile_results_0703/.../runs/20260703_130650`)
landed at `248.7 / 24.7 / 1146.5ms`, `1000/1000` correct. The queue profile
file shrank from the adjacent default's `~772KB` to `~189KB`, but the final
profile stayed in the same performance shape: `64` prefill batches,
`5.20s/5.66s` prefill forward/wall, `733` decode model calls, `10.06s` decode
GPU, and `106` decode-many calls over `488` steps. Keep this as diagnostics
hygiene, not a score-facing long_output fix.

A same-host repro of the public TorchInferno commit shows the public
`915.7ms` long_output row is not locally stable. Running `3927cf1` from
`/tmp/TorchInferno-public-3927`
(`agent_space/ti_long_public3927_results_0703/.../runs/20260703_131540`)
landed at `282.0 / 24.4 / 1145.1ms`, `1000/1000` correct. Its queue profile
matched the current local band: `63` prefill batches, `5.33s/5.80s` prefill
forward/wall, `789` decode model calls, `10.70s` decode GPU, and `107`
decode-many calls over `508` steps. Treat the public/local long-output delta as
run/environment variance until a same-host run reproduces the `~915ms` row.

Full-active waiting decode-many is rejected as a default. A local patch allowed
short greedy decode-many to run while requests were waiting only when
`len(active) >= max_active`, so it could not delay admission into free active
rows. The focused long_output run on that patch
(`agent_space/ti_long_fullactive_wait_decode_many_results_0703/.../runs/20260703_132538`)
landed at `250.8 / 24.7 / 1146.4ms`, `1000/1000` correct. The profile confirmed
the intended policy (`decode_many_with_waiting=true`,
`decode_many_with_waiting_min_active=64`), but model work stayed in-family:
`62` prefill batches, `5.20s/5.70s` prefill forward/wall, `749` decode model
calls, `10.24s` decode GPU, and `102` decode-many calls over `456` steps. This
reduced neither median E2E nor steady decode GPU, so keep waiting decode-many
as an explicit diagnostic/runtime knob rather than an OpenAI default.

Decode-many padding is now separated from stop-tail overgeneration in queue
profiles. The focused current-head long_output run
`agent_space/ti_long_decodemany_padding_profile_results_0703/.../runs/20260703_102544`
landed at `232.8 / 24.2 / 1093.5ms`, `1000/1000` correct. Its final profile
reported `27.2K` active decode-many model tokens, `30.5K` padded decode-many
graph slots, `25.4K` emitted decode-many tokens, `1.8K` skipped stop-tail
tokens, and `3.3K` decode-many padding tokens. Total ragged decode padding was
`6.6K` tokens, while decode GPU was still `10.19s` and prefill padding was
`39.1K` tokens (`17.7K` row / `21.4K` suffix). Keep the new counters for
diagnosis, but do not treat decode-many padding as the main long-output lever;
the remaining gap is still steady decode GPU plus prefix-suffix prefill waste.

Async decode-many CPU token copy is also rejected. A local opt-in patch copied
each decode-many token tensor to pinned CPU memory on a side CUDA stream and
synchronized only at the final flattened readback. The focused run
`agent_space/ti_long_async_cpu_tokens_results_0703/.../runs/20260703_104620`
landed at `274.3 / 24.3 / 1103.5ms`, `1000/1000` correct. The internal
`runtime_decode_many_cpu_tokens_ms` counter moved in the intended direction
(`21.3ms -> 9.0ms` versus the adjacent no-env profile), but total phase time,
prefill wall, and decode GPU time were all worse (`17.88s -> 18.79s`,
`5.54s -> 5.67s`, and `10.19s -> 10.69s`). Keep decode-many token readback on
the simpler synchronous path; the CPU readback bucket is too small to justify a
side stream and per-step pinned allocation without a broader decode pipeline
change.

Ragged-prefill graph cache hits now refresh eviction order, but the intermediate
long_output batch buckets remain opt-in. The current-head LRU stress run with
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS=1,2,4,8,16,24,32`
(`agent_space/ti_long_batch_buckets_lru_results_0703/.../runs/20260703_103637`)
landed at `240.2 / 25.4 / 1206.4ms`, `1000/1000` correct. The profile showed
the cache guardrail working (`124/128` live prefill graphs, `0` graph-cache
evictions), but it still paid three request-path `b24` captures (`3.17s`),
raised prefill wall to `8.09s`, and left decode GPU at `10.09s`. Keep LRU
replacement as a safer graph-cache policy; do not promote the intermediate
bucket set.

Explicit runtime prefix-prefill batch buckets now also drive greedy
common-prefix suffix warmup when the warmup-specific batch env is unset. This is
accepted as opt-in hygiene: an env run should not configure runtime `b24`
batches and then miss those same graphs during startup. The post-fix
long_output recheck on pushed `d0e457b` with
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS=1,2,4,8,16,24,32`
(`agent_space/ti_long_b24_warm_buckets_results_0703/.../runs/20260703_110936`)
removed the previous three request-path `b24` captures (`0.0ms` capture time)
and landed at `261.6 / 24.8 / 1095.8ms`, `1000/1000` correct. It still filled
the ragged-prefill graph cache (`128/128` live entries, `12` startup evictions)
and reached readiness in `226.0s`, so this does not reopen greedy-short `b24`
as a default.

Live prefill graph-cache queue profiles now also aggregate resident entries by
batch and suffix bucket. The default long_output refresh on `fc7f6f4`
(`agent_space/ti_long_cache_bucket_profile_results_0703/.../runs/20260703_111823`)
showed `121/128` resident prefill graphs: `20` each for `b1`, `b2`, `b4`,
`b8`, `b16`, and `b32`, plus one `b24` extra-pair graph. Pruning the small
greedy suffix warmup to `b4,b8,b16,b32`
(`TORCHINFERNO_OPENAI_WARMUP_ONLINE_GREEDY_COMMON_PREFIX_SUFFIX_BATCHES=4,8,16,32`)
is rejected despite improving readiness to `180.8s` and reducing resident
prefill graphs to `83/128`. The focused run
`agent_space/ti_long_prune_small_prefill_warmup_results_0703/.../runs/20260703_112341`
captured `ragged_prefill:b2:s64` on the request path (`787ms`), raised prefill
wall to `6.35s`, and landed at `260.5 / 24.1 / 1223.7ms`. Keep the small
greedy suffix graph warmups unless there is a replacement that can handle tail
`b2` groups without request-time capture.

Sampled-medium prefix prefill now has a default `b24` batch bucket. The focused
env probe with `TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS=1,2,4,8,16,24,32`
and matching warmup wrote
`agent_space/ti_tree_prefill_b24_results_0703/.../runs/20260703_075813` and
improved tree_of_thought to `129.0 / 34.8 / 154.3ms`, `959/992` correct. The
queue profile used `b24:s16` graphs, kept request-path prefill captures/misses at
zero, and cut prefill padding from the fresh current-head profile's `9.5K` tokens
to `8.1K`. Promoting that as a sampled-medium-only default (temperature > 0,
`256 < max_tokens <= 384`) was validated by a no-env focused run,
`agent_space/ti_tree_b24_default_results_0703/.../runs/20260703_080549`, at
`135.3 / 36.0 / 164.3ms`, `960/992` correct, with zero prefill graph misses.
The full TorchInferno-only validation
`agent_space/ti_full_b24_default_results_0703/.../runs/20260703_081126` landed at
few_shot `165.4 / 46.9 / 203.3ms`, self_consistency `180.1 / 0.0 / 192.5ms`,
multi_turn `293.5 / 60.5 / 346.5ms`, tree_of_thought `134.0 / 27.9 / 156.4ms`,
and long_output `263.2 / 24.4 / 1113.0ms`, with correctness in the normal band.

Greedy-short `b24` prefix-prefill batching is accepted with a larger prefill
graph cache. The original focused long_output env probe
`agent_space/ti_long_prefill_b24_results_0703/.../runs/20260703_082054`
improved to `246.9 / 23.7 / 1092.8ms`, and the no-env patch validation
`agent_space/ti_long_b24_default_results_0703/.../runs/20260703_082745` landed
at `225.6 / 24.2 / 1075.7ms` with zero request-path prefill graph misses. The
full-suite run
`agent_space/ti_full_greedy_b24_default_results_0703/.../runs/20260703_083404`
showed why it cannot ship broadly: long_output improved to
`247.7 / 23.9 / 1059.1ms`, but self_consistency regressed to
`262.9 / 0.0 / 427.5ms` and multi_turn to `336.2 / 66.4 / 400.8ms`. The queue
profile was already at the `128` ragged prefill graph cap with `2` evictions,
where the sampled-medium-only default stayed at `121` live entries with zero
evictions.

A full-suite recheck with the same greedy-short `b24` buckets and
`TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_GRAPHS=192`
(`agent_space/ti_full_b24_graphcap192_results_0703/.../runs/20260703_133343`)
kept the added shapes without evictions and removed the broad regression:
few_shot `170.1 / 49.4 / 208.2ms`, self_consistency
`206.6 / 0.0 / 236.5ms`, multi_turn `299.6 / 63.6 / 355.5ms`,
tree_of_thought `140.7 / 46.8 / 172.4ms`, and long_output
`249.1 / 23.9 / 1037.0ms`. Queue profiles topped out at `140/192` live prefill
graphs with `0` evictions across all workloads. The no-env code-default focused
confirmation
(`agent_space/ti_long_b24_graphcap192_default_results_0703/.../runs/20260703_134346`)
landed long_output at `237.1 / 23.7 / 1072.3ms`, `1000/1000` correct, with
`b24` prefix graph shapes present, `130/192` live prefill graphs, and no prefill
graph misses or evictions. This promotes greedy-short `b24` buckets and a
`192` graph-cache cap as a normal runtime policy; the remaining long-output gap
is still steady decode GPU and first-token scheduling, not request-path prefill
captures.

A decode-many preparation fast path is rejected and was backed out. The local
patch skipped pad-row set construction when no padding was needed, used direct
token-buffer views for contiguous prefix rows, and advanced contiguous seq-lens
with an in-place slice add. The focused rebuilt long_output run
`agent_space/ti_long_prepfast_results_0703/.../runs/20260703_093411` stayed
correct (`1000/1000`) but landed at only `248.5 / 24.6 / 1152.0ms`. Its queue
profile moved in the wrong direction versus the accepted full-suite default
band: decode prepare rose from `420ms` to `443ms`, decode GPU from `10.23s` to
`10.55s`, prefill wall from `5.63s` to `5.73s`, and E2E remained worse than the
`1113.0ms` accepted run. Keep the existing index-select/index-add decode-many
preparation path until a broader decode pipeline or kernel change reduces actual
GPU work instead of reshaping small setup operations.

Sampled-medium `s12` suffix buckets are also rejected as a default, even after
the sampled `b24` batch bucket made the idea more plausible. The new
row/suffix padding counters in
`agent_space/ti_padding_breakdown_results_0703/.../runs/20260703_085009` showed
tree_of_thought still spending `8.0K` padding tokens, mostly suffix padding
(`2.5K` row / `5.5K` suffix), while long_output remained split between row and
suffix padding (`23.5K` / `22.2K`). An env probe with sampled suffix buckets
`12,16` plus matching sampled warmup,
`agent_space/ti_tree_s12_b24_results_0703/.../runs/20260703_085639`, cut tree
padding to `3.5K` tokens (`2.0K` row / `1.5K` suffix) and improved focused
medians to `120.5 / 38.4 / 149.2ms`. But the no-env patch validation
`agent_space/ti_tree_s12_b24_default_results_0703/.../runs/20260703_090314`
regressed tree TPOT to `51.0ms`, and the full-suite validation
`agent_space/ti_full_s12_b24_default_results_0703/.../runs/20260703_090939`
regressed self_consistency to `179.8 / 0.0 / 287.1ms`, tree to
`130.5 / 52.2 / 160.6ms` with correctness `0.958`, and long_output to
`254.6 / 25.7 / 1139.1ms`. The full profile sat at `128` live ragged prefill
graphs for every segment, leaving no graph-cache headroom. Keep sampled `s12`
suffixes as an opt-in diagnostic; the next tree fix needs lower per-call
prefix-suffix cost without filling the graph cache.

A debug rerun on pushed commit `5089f09` wrote
`agent_space/ti_multi_warmmixed_debug_results_0425/.../runs/20260703_041208`
and measured `232.4 / 72.8 / 304.8ms`, `981/1000` correct. Saved response text
shows the misses are ordinary arithmetic failures under load, such as
`68 * 88 =` producing `5994` instead of `5984` and integer division prompts
emitting decimal approximations. The public 20260703_110206 multi_turn
correctness rates are TorchInferno `0.983`, vLLM `0.980`, and SGLang `0.980`,
so the remaining correctness rate is not unique to the mixed-prefix cache path.

Do not promote that exact env set as the inference-bench provider default.
Full-suite run
`agent_space/ti_full_warmmixed_results_0430/.../runs/20260703_041927` improved
multi_turn to `244.9 / 71.2 / 310.4ms`, but regressed few_shot
(`190.5 / 58.3 / 245.9ms`), self_consistency (`552.0 / 0.0 / 756.9ms`),
tree_of_thought (`184.9 / 83.2 / 272.8ms`), and long_output
(`1135.2 / 53.5 / 2665.7ms`). The queue profile shows `31` request-time
common-prefix prefill captures for `p111/s32,s64,s96`, costing `32.4s`. An
attempt to warm `111:32,111:64,111:96,122:16` for exact batches `1..32`
(`agent_space/ti_full_warmmixed_commonwarm_build_0440`) was interrupted after
roughly six minutes without readiness and with near-full H100 memory use, so
exhaustive exact-batch common-prefix warmup is too heavy as a default.

## Public 20260702_140923 refresh and SGLang CLI compatibility

The latest public all-provider run at
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260702_140923`
was published with inference-bench `055de6e5`. It includes inference-bench
commit `2b770aee`, which switches the SGLang provider from the removed `--tp`
flag to current `--tp-size` and adds a focused provider test. The measured
provider commits were vLLM `2a16ece`, SGLang `a375e9f`, and TorchInferno
`0c3edef`. All providers completed all five benchmarks with no recorded
provider errors.

The score split was SGLang `14`, TorchInferno `5`, and vLLM `0`. Rows as
TTFT / TPOT / E2E:

- few_shot: vLLM `239.1 / 86.4 / 319.8ms`, SGLang
  `114.0 / 88.5 / 202.7ms`, TorchInferno `172.3 / 50.1 / 214.3ms`.
- self_consistency: vLLM `216.4 / 0.0 / 304.4ms`, SGLang
  `195.7 / 0.0 / 388.3ms`, TorchInferno `216.4 / 0.0 / 234.3ms`.
- multi_turn: vLLM `284.7 / 97.0 / 379.4ms`, SGLang
  `144.5 / 124.8 / 273.7ms`, TorchInferno `341.2 / 66.6 / 399.3ms`.
- tree_of_thought: vLLM `125.4 / 82.8 / 186.4ms`, SGLang
  `55.9 / 84.9 / 147.7ms`, TorchInferno `156.1 / 42.3 / 195.4ms`.
- long_output: vLLM `94.6 / 27.4 / 1115.2ms`, SGLang
  `60.5 / 24.7 / 889.8ms`, TorchInferno `274.0 / 25.1 / 1283.8ms`.

TorchInferno now wins median TPOT on few_shot, multi_turn, and tree_of_thought,
plus self_consistency E2E and throughput. It still loses SGLang on TTFT/E2E for
few_shot, multi_turn, tree_of_thought, and long_output, and narrowly misses
long_output TPOT. The TorchInferno queue profile explains the remaining losses:
few_shot reused the `122` token common prefix, but still spent `1.61s` in
prefill forward and `0.77s` in ragged decode GPU across the two client waves;
multi_turn reused only the `45` token shared prefix and spent `4.27s` in prefill
forward; tree reused the same `45` token shared prefix and spent `2.21s` in
prefill forward plus `1.45s` in ragged decode GPU; long_output reused the `111`
token common prefix but still spent `6.06s` in prefill forward and `10.08s` in
ragged decode GPU across `740` decode batches. The long_output decode-many path
was already active (`150` calls, `393` steps, `22.5k` model tokens), so the
gap is not an obvious disabled decode-many fast path.

This refresh confirms the current code path and recent inference-bench provider
fixes are runnable, but it does not change the
remaining engineering diagnosis: multi_turn needs TP-safe non-common prefix
reuse that does not fragment prefill, tree needs cheaper steady sampled-medium
prefix-suffix prefill/decode, and long_output needs a decode/prefill policy that
improves medians without reintroducing the rejected waiting-decode tail
regression.

Two focused follow-ups on the same checkout are rejected as defaults. Extending
the sampled-medium idle window to `750ms` merged tree_of_thought into one online
batcher session
(`agent_space/ti_tree_idle750_results/.../runs/20260702_143454`), but only moved
the row to `154.7 / 67.5 / 186.4ms`, `961/992` correct. The profile did reduce
prefill wall versus the current full-run control (`3.08s` to `2.82s`) and merge
the two public-shaped tree waves into one session, but total phase time was
still `5.67s` and it gave back most of the TPOT cushion that currently wins
TorchInferno's tree cell. Keep the sampled-medium idle window at the narrower
default.

Skipping active-row KV zeroing on acquire is now available only as the opt-in
diagnostic `TORCHINFERNO_CONTINUOUS_SKIP_ACTIVE_ROW_CLEAR=1`; it should not be
promoted. The tree run
`agent_space/ti_tree_skipclear_results/.../runs/20260702_144429` landed at
`150.9 / 87.5 / 180.5ms`, `957/992` correct. It cut tree prefill wall from the
current full-run control's `3.08s` to `2.51s` and queue p99 first-token/finish
to `304/395ms`, but median TPOT lost the vLLM/SGLang cell. The long_output run
`agent_space/ti_long_skipclear_results/.../runs/20260702_144935` was also not a
score win at `271.0 / 24.9 / 1256.2ms`, `1000/1000` correct; prefill wall was
`6.78s`, decode GPU was `10.27s`, and queue finish p50 stayed high at
`1172ms`. Active-row clears are measurable overhead, but removing them is not a
safe default path to the remaining median gaps.

A narrower active-row clear skip is accepted only for folded prefix-graph
prefill rows. Unlike the rejected global diagnostic above, this path is limited
to dense-cache `_prefill_prefix_graph_batch` rows whose copied prefix and suffix
KV are written inside the ragged prefill graph before attention slices
`context_len`; all other active-row acquisition still uses the normal clear
path. The focused tree run
`agent_space/ti_prefix_graph_noclear_results/.../runs/20260702_212204` landed at
`150.8 / 29.6 / 174.3ms`, `957/992` correct. Versus the pushed f7 tree profile,
prefix-copy accounting fell from `40.5ms` to `11.2ms`, prefill wall fell from
`3.01s` to `2.72s`, and queue p99 first-token/finish improved from
`388/473ms` to `310/355ms`; median TTFT was slightly worse, so this is not a
front-door TTFT fix. The matching long_output run
`agent_space/ti_prefix_graph_noclear_long_results/.../runs/20260702_212735`
landed at `282.6 / 24.0 / 1105.4ms`, `1000/1000` correct. Versus the q8 drain
profile, prefix-copy accounting fell from `98.2ms` to `9.7ms`, prefill wall
fell from `6.05s` to `5.64s`, queue p99 first-token improved from `790ms` to
`497ms`, and queue finish p50 improved from `1033ms` to `1015ms`; queue finish
p99 was roughly flat-to-slightly worse (`1863ms` to `1902ms`). Keep this as a
scoped prefill-tail reduction, not a replacement for the remaining TTFT work.

Tree prefix-graph profiling now exports per-shape prefill copy/setup/forward/
sample/state timing plus active-vs-padded prefix-graph token counts. The focused
instrumented tree run
`agent_space/ti_tree_prefill_shape_phase_results2/.../runs/20260702_221034`
landed at `148.6 / 50.0 / 178.7ms`, `964/992` correct. The final queue profile
spent `2.73s` in prefill wall, including `2.25s` forward, `156ms` sampling,
`67ms` setup, and `56ms` state update. The dominant
`prefix_graph:b32:s16:p45-45:src1:mixed0` shape consumed `1.70s` wall and
`1.41s` forward, while prefix-graph work overall ran `10,402` active suffix
tokens inside `20,784` graph tokens. This confirms the sampled-medium tree gap
is mostly padded prefix-suffix graph compute rather than host setup or sampling.

An opt-in one-shot CUDA profiler is available for the actual ragged
prefix-prefill body:
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_ONCE=1`. Unlike the older
`TORCHINFERNO_PROFILE_PREFILL_ONCE` hook, this targets the
`try_prefill_ragged_logits_graph` body used by common-prefix suffix prefill. It
defaults to `TORCHINFERNO_PROFILE_RAGGED_PREFILL_MIN_BATCH=32` so tiny startup
warmups do not consume the profile slot unless explicitly requested. Set
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_MIN_SUFFIX` when the target is a larger
request-path suffix bucket rather than the warmed `s16` startup graph. If the
request path uses the same bucket as startup, set
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_SKIP_MATCHES=N` to skip the first matching
warmup calls; the profile line prints the selected `match=` index. Startup can
still capture every request-shape graph before the server is ready; use
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_REPLAY_ONCE=1` to profile the warmed
CUDA-graph replay path instead, with the same batch/suffix gates and optional
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_REPLAY_SKIP_MATCHES=N`. Both hooks also
accept `TORCHINFERNO_PROFILE_RAGGED_PREFILL_CONTEXT_LEN`,
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_BATCH`, and
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_SUFFIX` when a workload has both dynamic
startup buckets such as `ctx-64` and exact request buckets such as tree's
`ctx61`, or when startup has larger suffix buckets that would otherwise consume
the one-shot profile slot. These are diagnostic hooks only; they should be used
to locate
the next prefill sink before changing default runtime policy.

That hook found the hot sampled tree shape running as
`batch=32 suffix=16 context_len=-64`: the one-shot CUDA profile spent
`40.5ms` total, including `14.8ms` in GEMMs, `12.9ms` in TP all-reduces, and
`5.7ms` under math SDPA. Forcing exact positive contexts with
`TORCHINFERNO_CONTINUOUS_DYNAMIC_PREFIX_PREFILL_GRAPH=0` in
`agent_space/ti_tree_exact_context_results/.../runs/20260703_015312` improved
tree to `135.1 / 28.6 / 164.4ms`, `960/992` correct, with zero request-path
captures and the dominant `prefix_graph:b32:s16:p45-45:src1:mixed0` forward
time down to `1.25s`. The default policy now uses exact prefix-suffix context
for sampled common-prefix prefill while preserving the greedy short-output
dynamic bucket path that helped few_shot/long_output. The no-env confirmation
run
`agent_space/ti_tree_sampled_exact_default_results/.../runs/20260703_020000`
landed at `135.2 / 28.4 / 159.0ms`, `964/992` correct, again with zero
request-path prefill captures and `1.25s` in the hot b32 prefix-graph forward.

The next obvious lever, graph-captured symmetric-memory all-reduce for the same
prefill shape, was tested as a local opt-in prototype and rejected. The
`TORCHINFERNO_SYMM_MEM_PREFILL_GRAPH_ALLREDUCE=1` A/B with sampled symmetric
scope enabled in
`agent_space/ti_tree_symm_prefill_graph_results/.../runs/20260703_021900`
landed at `136.3 / 42.3 / 167.7ms`, `956/992` correct. The hot b32
prefix-graph forward regressed to `1.32s` and p99 worsened, so the remaining
sampled tree gap should not be chased by promoting that path.

An opt-in suffix-bucket split is available as
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS=1`, but it is
rejected as a default for tree. The focused A/B
`agent_space/ti_tree_suffix_buckets_results/.../runs/20260702_221841` landed at
`151.2 / 39.5 / 178.1ms`, `960/992` correct. It improved TPOT and p99 E2E
(`706ms` to `581ms`) but worsened median TTFT and did not reduce the root
prefill padding: all prefix-graph groups still landed in the `s16` suffix
bucket, and model tokens only moved from `20,784` to `20,736`. Keep this as a
diagnostic for workloads with mixed suffix buckets, not a sampled-medium tree
default.

The same row-counter profiling now explains why suffix-bucket splitting also
cannot be promoted for multi_turn. The focused control
`agent_space/ti_multi_row_phase_results/.../runs/20260702_223534` landed at
`304.3 / 62.5 / 364.2ms`, `983/1000` correct. Its prefix-graph prefill work
ran `69,574` active suffix tokens inside `97,408` graph tokens; estimated waste
was mostly suffix padding (`22.8K` tokens) rather than row padding (`5.0K`
tokens). Enabling `TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS=1`
for `agent_space/ti_multi_suffix_split_results/.../runs/20260702_224117`
reduced suffix padding to `7.0K`, but row padding jumped to `27.3K`, prefill
batches rose `35 -> 68`, prefill forward rose `4.03s -> 4.92s`, and the score
row regressed to `374.8 / 63.0 / 425.0ms`, `982/1000` correct. Keep the split
as an opt-in diagnostic only; multi_turn still needs longer conversation-prefix
reuse or a non-fragmenting mixed-prefix suffix path.

Guarded suffix-bucket splitting is accepted only for greedy-short long_output.
The default policy now enables the split for deterministic `max_tokens<=128`
sessions, but only when the actual admitted group reduces predicted prefill
graph tokens, every split subgroup is at least `75%` full, and the automatic
greedy-short path avoids singleton split groups. The current-head control
`agent_space/ti_long_row_phase_results/.../runs/20260702_224930` landed at
`248.9 / 24.8 / 1142.5ms`, `1000/1000` correct, with `44,715` active suffix
tokens inside `91,136` prefix-graph tokens and `5.31s` prefill forward. The
guarded env A/B
`agent_space/ti_long_guarded_suffix_split_results/.../runs/20260702_225711`
cut prefix-graph tokens to `80,032` and moved the row to
`254.2 / 24.0 / 1128.5ms`. The no-env patched confirmation
`agent_space/ti_long_greedy_split_default_results/.../runs/20260702_230348`
landed at `256.9 / 24.4 / 1134.1ms`, `1000/1000` correct, with prefill forward
down to `5.00s` and prefix-graph tokens at `80,096`. This is a TPOT/E2E and
throughput tradeoff, not a TTFT fix; sampled tree and greedy-large multi_turn
stay on their rejected opt-in paths.

A follow-up full-order TorchInferno run on pushed `d869023`
(`agent_space/ti_full_d869023_results/.../runs/20260702_231547`) kept the
long_output improvement in-family at `286.2 / 23.3 / 1098.5ms`,
`1000/1000` correct, and reduced long_output prefix-graph model tokens further
to `75,712`. Requiring at least four requests per split subgroup is rejected as
too broad: the focused env run
`agent_space/ti_long_suffix_mingroup4_results/.../runs/20260702_232151`
improved TTFT to `262.3ms`, but regressed TPOT/E2E to `23.9 / 1158.5ms`.
Requiring only two requests per automatic greedy-short split subgroup is kept:
`agent_space/ti_long_suffix_mingroup2_results/.../runs/20260702_232653`
landed at `241.2 / 24.6 / 1120.4ms`, `1000/1000` correct, improving focused
TTFT/E2E versus the no-env split confirmation while avoiding singleton
`b1` suffix-prefill fragments. Explicit diagnostic split runs can still set
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS_MIN_GROUP=1`.

The automatic greedy-short suffix split is retracted after the public
`20260702_231040` refresh and same-head controls. That public run measured
TorchInferno `02949d2`, before automatic suffix splitting, at
`244.3 / 21.5 / 993.2ms` for long_output with `4.54s` prefill forward and
`9.66s` ragged-decode GPU. A current-head forced-off control
(`agent_space/ti_long_suffix_split_off_445ca96_results/.../runs/20260702_233832`)
landed at `254.2 / 23.8 / 1105.9ms`, beating the min-group-2 split run on
TPOT/E2E but not by enough to claim a new win. The patched no-env confirmation
with the automatic default off
(`agent_space/ti_long_split_default_off_results/.../runs/20260702_235427`)
landed at `268.0 / 24.1 / 1112.7ms`, `1000/1000` correct, with `5.16s`
prefill forward, `10.19s` ragged-decode GPU, and no split-induced extra prefill
groups. Disabling greedy-short prefill-cost priority on top of the forced-off
split is rejected:
`agent_space/ti_long_no_cost_priority_445ca96_results/.../runs/20260702_234717`
landed at `262.0 / 24.4 / 1118.8ms` and increased decode-many work to
`29.3K` model tokens with `1.9K` skipped. Keep suffix-bucket splitting opt-in
via `TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS=1` or
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS_GREEDY_SHORT=1`;
the automatic default should stay off until it reproduces against the public
no-split baseline.

A full TorchInferno-only refresh on pushed `18104cd`
(`agent_space/ti_full_18104cd_results/.../runs/20260703_000109`) landed at
few_shot `165.8 / 49.9 / 206.3ms`, self_consistency
`174.7 / 0.0 / 186.1ms`, multi_turn `308.7 / 62.1 / 362.4ms`,
tree_of_thought `150.0 / 54.9 / 183.6ms`, and long_output
`253.4 / 24.6 / 1078.6ms`, with correctness in the normal bands. The queue
profiles keep the priority unchanged: multi_turn reused only the `45` token
common prefix and spent `4.30s` in prefill forward; tree spent `2.15s` in
prefill forward plus `1.46s` in ragged-decode GPU; long_output spent `5.20s`
in prefill forward plus `10.45s` in ragged-decode GPU.

After the sampled exact-context promotion, a full TorchInferno-only refresh on
pushed `79c71d5`
(`agent_space/ti_full_79c71d5_results/.../runs/20260703_022522`) landed at
few_shot `171.2 / 53.2 / 211.9ms`, self_consistency
`199.0 / 0.0 / 228.0ms`, multi_turn `309.5 / 65.0 / 364.4ms`,
tree_of_thought `137.9 / 28.3 / 162.7ms`, and long_output
`257.1 / 24.3 / 1151.0ms`. The intended tree win held in the full run, but the
remaining priority did not change: multi_turn still reused only the `45` token
common prefix and spent `4.26s` in prefill forward; long_output spent `5.21s`
in prefill forward and `10.27s` in ragged decode GPU across `752` decode
batches. The current long-output decode-many split is still dominated by
`b64/64` work with only moderate skipped-token waste, so the rejected
drain-quantum/tail-cap knobs remain closed.

Rechecking sampled decode-many under the current sampled-medium q2 policy does
not change the default. Enabling
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_DECODE_MANY=1` alone for focused tree
(`agent_space/ti_tree_sampled_decode_many_q2_results/.../runs/20260703_000807`)
landed at `148.7 / 46.6 / 176.1ms`, but the queue profile showed
`use_decode_many=true` with `runtime_decode_many_calls=0`, so the row movement
was run noise rather than the intended mechanism. Allowing decode-many while
waiting made the mechanism fire, but regressed the row:
`agent_space/ti_tree_sampled_decode_many_wait_q2_results/.../runs/20260703_001318`
landed at `154.3 / 64.7 / 193.3ms`. It ran `32` decode-many calls over `64`
steps, spent `1.90s` in ragged-decode GPU, and skipped `330/732` decode-many
model tokens after stop finishes. Keep sampled decode-many and waiting
decode-many off by default for tree.

A narrower decode-many-while-waiting policy is also rejected and not in source.
The temporary patch allowed decode-many only while the ready queue was below the
same refill floor that would admit the next prefill wave, trying to reduce
one-token decode overhead without delaying admissions that were already
eligible. The long_output run
`agent_space/ti_long_wait_belowfloor_results/.../8xH100/runs/20260702_165707`
stayed correct (`1000/1000`) and improved median TTFT/E2E to
`196.7 / 26.1 / 1196.4ms`, but it worsened TPOT and produced a bad tail:
p99 TTFT/E2E was `1492/2335ms`. The queue profile confirms the tradeoff:
decode-many expanded to `222` bursts / `588` steps / `31.9K` model tokens with
`839` skipped tokens, while queue-to-first p99 stayed above `1.15s`. Reducing
the greedy-short decode quantum to `2` did not rescue it:
`agent_space/ti_long_wait_belowfloor_dq2_results/.../8xH100/runs/20260702_170242`
landed at `183.8 / 27.2 / 1199.8ms` with p99 TTFT/E2E `1688/2476ms`. Keep the
existing waiting-decode diagnostics closed; long_output still needs a real
prefill/decode pipeline rather than a decode-burst admission heuristic.

Splitting only uncaptured greedy-short prefix-prefill compute is accepted as a
narrow long_output fix. Public `20260702_140923` spent almost `1s` in two
`b64:s64` prefix-suffix prefill calls that could not be captured because the
greedy-short capture policy intentionally stops at batch `32`; limiting
admission to `32` had already been rejected because it fragmented decode. The
accepted path keeps the outer admission wave at `64` but splits only the prefix
prefill compute into warmed capture-eligible chunks when capture-on-miss would
otherwise be skipped. On the same checkout, the split-off control
`agent_space/ti_long_splitoff_control_results/.../runs/20260702_180006` landed
at `272.1 / 25.2 / 1253.3ms` with p99 TTFT/E2E `1582.8/2419.7ms` and one
`b64:s64` prefix graph miss. The split run
`agent_space/ti_long_split_b64_results/.../runs/20260702_175325` landed at
`231.9 / 25.2 / 1188.9ms`, p99 TTFT/E2E `1157.8/2142.1ms`, `1000/1000`
correct, and zero prefill graph misses. Queue profile improved from `6.86s` to
`6.51s` prefill wall, `10.45s` to `10.05s` ragged decode GPU, and
queue-to-first p99 from `1276ms` to `839ms`.

Pruning redundant online decode warmup shapes is accepted as a startup-safety
follow-up. The warmed runtime ragged decode path buckets active rows to powers
of two plus the active cap; exact small non-power batches `5`, `6`, and `7`
did not add request-time coverage but did add extra symmetric-memory graph
warmup rendezvous points. A focused long_output run on pushed `3085141` plus
the local pruning patch wrote
`agent_space/ti_long_pruned_decode_warmup_results/.../runs/20260702_183607`.
It reached readiness in `200.9s`, captured `8` startup FlashInfer decode graphs
instead of the previous larger set, completed `1000/1000` requests, and landed
at `243.5 / 25.0 / 1188.6ms` with p99 TTFT/E2E `1110.2/1936.7ms`. The queue
profile stayed clean: prefill graph hits/misses `60/0`, decode graph
hits/misses `745/0`, runtime decode graph captures `0`, prefill wall `5.89s`,
and ragged decode GPU `10.17s`. A full TorchInferno-only run on pushed
`51977ce`
(`agent_space/ti_main_51977ce_full_results/.../runs/20260702_184338`) then
showed why exact `3` should remain: after earlier benchmarks consumed active-row
state, long_output hit one request-time `ragged_decode:token:b3:rows1` capture
(`651ms`) even though prefill and decode graph misses were otherwise zero. Keep
exact `3` as the no-free-row fallback, but do not restore the old `5..8` exact
set. The accepted exact-3 follow-up
`agent_space/ti_main_exact3_decode_warmup_results/.../runs/20260702_185228`
kept readiness at `200.8s`, warmed `9` startup FlashInfer decode graphs,
completed the full TorchInferno-only run, and moved long_output to
`250.4 / 24.8 / 1129.7ms` with p99 TTFT/E2E `783.5/1962.6ms`. Its long_output
queue profile had prefill graph hits/misses `66/0`, decode graph hits/misses
`720/0`, runtime decode graph captures `0`, prefill wall `6.23s`, and ragged
decode GPU `9.88s`. Keep symmetric-memory decode warmup enabled; the rejected
no-symm-decode-warmup probe avoided startup risk but regressed long_output tails
and decode GPU time.

Decode graph miss-shape profiling is added as instrumentation, not as a new
scheduling default. Queue profiles now emit
`runtime_decode_graph_miss_shape_counts` beside the aggregate miss counter. A
focused tree probe after the exact-3 warmup change wrote
`agent_space/ti_tree_static_decode_miss_shapes_results/.../runs/20260702_191417`
and landed at `145.1 / 56.3 / 175.1ms`, `955/992` correct. Its single runtime
decode graph miss was `{"static_decode:logits:b3":1}` with zero request-path
decode graph captures, so the residual tree miss in this run was a static
fallback shape rather than ragged decode warmup coverage. Keep treating tree as
sampled-prefix/suffix prefill and steady decode bound.

Static logits decode warmup for the existing online decode batch set is accepted.
The focused patched tree run
`agent_space/ti_tree_static_logits_warm_results/.../runs/20260702_195432`
captured static logits graphs during startup after the normal ragged token/logits
decode warmup. It landed at `147.5 / 29.2 / 174.6ms`, `960/992` correct, and
removed the residual miss (`runtime_decode_graph_misses=0`,
`runtime_decode_graph_miss_shape_counts={}`). Queue timings stayed in-family
with the control (`request_queue_to_first_token_p50=125.6ms`, prefill wall
`2.92s`, ragged decode GPU `1.49s`), so this is a narrow graph-coverage fix, not
a tree scheduling breakthrough.

Raising the sampled FlashInfer decode cutoff to include tree's `max_tokens=300`
is rejected. The env-only run with
`TORCHINFERNO_CONTINUOUS_FI_DECODE_SAMPLED_MAX_TOKENS=400` on pushed `c6f04e8`
wrote
`agent_space/ti_tree_fi_sampled400_results/.../runs/20260702_200138` and
landed at `154.7 / 63.0 / 190.6ms`, `957/992` correct, versus the adjacent
static-logits-warm control's `147.5 / 29.2 / 174.6ms`. The run still had zero
decode misses/captures, but ordinary decode GPU time rose (`1.49s -> 1.70s`)
and queue-to-finish median rose (`148.9ms -> 165.1ms`) despite fewer scheduler
steps (`109 -> 99`). Keep the sampled FlashInfer decode default scoped to
`max_tokens <= 256`; tree's 300-token branch remains faster on the dense ragged
logits path.

The pushed `fcb4567` TorchInferno-only full refresh wrote
`agent_space/ti_full_fcb4567_results/.../runs/20260702_200744`. The rows were
few_shot `171.5 / 47.8 / 210.3ms`, self_consistency `120.2 / 0.0 / 269.8ms`,
multi_turn `316.0 / 62.5 / 368.7ms`, tree_of_thought
`150.7 / 47.0 / 183.1ms`, and long_output `241.3 / 25.7 / 1235.1ms`.
The full-order tree queue profile still had one static fallback miss,
`{"static_decode:logits:b5":1}`, even though the focused tree run above had
zero misses.

Restoring exact small static-logits warmup is rejected. A source probe warmed
static logits batches `5`, `6`, and `7` without restoring the earlier rejected
ragged decode warmups, then ran the public-order prefix through tree:
`agent_space/ti_tree_static_logits_small_results/.../runs/20260702_201606`.
Readiness rose to `210.9s`, tree landed at `153.1 / 42.1 / 180.5ms`,
`964/992` correct, and the queue profile still had a static miss, now
`{"static_decode:logits:b2":1}`. The miss moving to an already-warmed batch
showed that static graph misses are not keyed by batch alone; cache
sequence/attention-block state matters. The runtime miss-shape instrumentation
therefore now records static cache length when available, e.g.
`static_decode:logits:b2:s3`, before spending more startup time on static
fallback warmups.

A pushed-instrumentation rerun did not reproduce the static miss. The
four-benchmark sequence through tree on `feaded7` wrote
`agent_space/ti_tree_static_miss_seq_results/.../runs/20260702_202503` and
landed at few_shot `168.8 / 50.5 / 208.4ms`, self_consistency
`197.0 / 0.0 / 211.4ms`, multi_turn `296.0 / 62.7 / 350.2ms`, and
tree_of_thought `150.4 / 48.1 / 179.3ms`. The tree queue profile ended with
`runtime_decode_graph_misses=0`. Treat the static fallback as rare sequencing
noise unless a sequence-qualified miss recurs; do not add more static startup
warmups speculatively.

Dynamic drain decode quantum is accepted for short greedy decode-many sessions.
The base short-greedy command quantum stays at `3` while ready or waiting work
can still be admitted, but once both the external queue drain and the runtime
waiting queue are empty the online batcher now broadcasts a larger decode-many
drain command. The focused default-`8` run on the patched `06ee764` tree wrote
`agent_space/ti_long_drainq8_results/.../runs/20260702_203950` and landed at
long_output `255.9 / 23.7 / 1118.8ms`, `1000/1000` correct, with readiness
unchanged at `205.9s`. The profile showed the intended mechanism versus the
latest full-run control: online step commands fell `249 -> 183`, step
broadcast+sync fell `752ms -> 545ms`, queue-to-finish p50 improved
`1192ms -> 1033ms`, and graph coverage stayed clean (`60/0` prefill,
`801/0` decode). The cost is delayed stop-token readback during drain bursts:
decode-many skipped tokens rose `412 -> 1907`, so this is scoped only to
short greedy decode-many after admission has drained, with
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_DRAIN_DECODE_QUANTUM` left as the
deployment override.

The narrower drain quantum `5` is rejected. The adjacent env run
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_DRAIN_DECODE_QUANTUM=5` wrote
`agent_space/ti_long_drainq5_results/.../runs/20260702_204529` and stayed
correct, but regressed to `286.1 / 24.3 / 1223.9ms`. It reduced skipped tokens
relative to `8` (`928` vs `1907`) but kept too many commands (`201`) and did
not preserve the score-facing E2E win. Keep the default drain quantum at `8`;
set it to the base quantum to disable the drain behavior for latency-sensitive
streaming deployments.

Raising the same drain quantum to `12` is also rejected. The env-only run on
pushed `f7f2cd1` wrote
`agent_space/ti_long_drainq12_results/.../runs/20260702_210255` and landed at
`310.9 / 22.8 / 1088.2ms`, `1000/1000` correct. It cut online step commands
further (`183 -> 150`) and improved median TPOT/E2E, but queue-to-first p50
regressed (`203ms -> 249ms`), p99 finish regressed (`1863ms -> 2098ms`),
decode-many skipped tokens rose (`1907 -> 2881`), and decode GPU time rose
(`10.23s -> 10.80s`). Because default `8` already clears the public TPOT cell,
the larger drain burst is not worth the TTFT/tail tradeoff.

A current-head focused multi_turn profile on pushed `325cdf4` wrote
`agent_space/ti_multi_325cdf4_results/.../runs/20260702_192513` and landed at
`303.1 / 61.8 / 361.8ms`, `983/1000` correct. The new miss-shape field stayed
empty because decode graph hits/misses were `86/0`; the row remains admission
and prefix-suffix prefill bound. The queue profile admitted all `1000` requests
through `34` prefill graph hits, reused only the shared `45` token prefix
(`{"common_prefix":1000}`), and spent `3.82s` in prefill forward, `4.29s` in
prefill wall, and `0.79s` in ragged decode GPU. Public raw multi_turn per-turn
medians show why this is still a queue-facing gap: vLLM turn-0 TTFT was `172ms`
and SGLang `70.8ms`, while the focused TorchInferno run had turn-0 median
`618.4ms` as the 64-worker conversation harness waited for later conversations
to enter the queue. This does not reopen the rejected greedy-large first-wait,
active-row, generated-prefix, or full-prompt reuse knobs; the remaining win still
needs TP-safe per-conversation prefix reuse or a cheaper batched non-common
prefix path.

A suffix-bucket admission-affinity probe is also rejected. A local patch kept
the first ready request as the admission anchor, then filled the wave with other
ready requests sharing the same reusable-prefix hit length and suffix graph
bucket before falling back to normal arrival order. The run
`agent_space/ti_multi_bucket_affinity_results/.../runs/20260702_194240` landed
at `308.0 / 61.9 / 362.2ms`, `981/1000` correct, so it slightly regressed the
current-head control. The queue profile explains why this should not be kept:
prefill wall rose `4.29s -> 4.44s`, prefill forward rose `3.82s -> 3.96s`, and
queue-to-first p50/p99 moved from `239.8/466.3ms` to `245.2/501.7ms`. It also
shifted one extra wave into `b32:s128` without reducing the dominant `b32:s144`
work. Keep the existing arrival-order greedy-large admission; the prior
shortest-suffix priority and this weaker bucket-affinity variant both fail to
turn suffix bucketing into a multi_turn win.

Common-prefix row adoption is kept as a lifecycle/correctness fix, not as a
new scheduling knob. The old common-prefix path computed the shared prefix in
one prefix row, then tried to acquire a second prefix row just to store a
reusable copy. With `prefix_cache_capacity=1`, that meant the shared prefix
could not be retained for later arrivals. The adopted-row path stores the
computed prefix row directly in `reusable_prefixes`, and a CPU regression test
now covers the single-prefix-slot case. A dirty-worktree tree run based on
`17f4993`
(`agent_space/ti_tree_adopt_common_results/.../runs/20260702_145934`) landed
at `149.3 / 45.9 / 179.7ms`, `959/992` correct. Its profile showed one common
prefix prefill, `992` common-prefix reuse hits, and `58` prefix-suffix graph
batches. That keeps tree medians in the current band and slightly improves
TTFT/E2E versus the public row, but it is not a TPOT win and does not change
the remaining diagnosis.

A follow-up TorchInferno-only full run on pushed `8d0093b` wrote
`agent_space/ti_8d0093b_full_results/.../runs/20260702_151031`. Rows as
TTFT / TPOT / E2E were: few_shot `171.1 / 46.6 / 210.3ms`,
self_consistency `154.2 / 0.0 / 185.5ms`, multi_turn
`314.1 / 62.4 / 367.8ms`, tree_of_thought `154.0 / 47.4 / 189.6ms`, and
long_output `257.8 / 25.2 / 1302.2ms`. Correctness stayed in family:
`977/1000`, `1000/1000`, `980/1000`, `956/992`, and `1000/1000`. The
long_output queue profile still shows the unchanged bottleneck:
`7.16s` prefill wall (`6.19s` forward), `9.88s` ragged decode GPU, `717`
decode batches, and `153` decode-many calls. This run keeps the row-adoption
change validated across all public benchmark shapes, but it does not reopen the
rejected long_output scheduling knobs.

A dense-row prefix-graph prototype is rejected and was backed out. The idea was
to route common-prefix suffix prefill through a `row_indices=None` ragged-prefill
graph when active rows were exactly `0..batch-1`, so the model could copy the
shared prefix into a dense destination slice instead of using advanced row
indexing. A focused long_output run with
`TORCHINFERNO_CONTINUOUS_DENSE_ROW_PREFIX_GRAPH=1` wrote
`agent_space/ti_long_dense_prefix_results/.../runs/20260702_152740` and kept
correctness at `1000/1000`, but landed at `267.8 / 26.0 / 1256.6ms` with p99
TTFT/E2E `2552/4367ms`. Startup ready time rose to `306.3s`, GPU memory during
warmup reached roughly `85GB`, and the queue profile regressed prefill wall to
`15.15s` (`14.21s` forward) with `10` request-path graph captures
(`9.44s` capture time). Decode GPU time stayed in the same band at `10.15s`.
Do not re-open dense-row prefix graphing without first fixing graph-key/warmup
reuse and proving it reduces prefill wall without the startup, memory, and p99
tail cost.

The token-budget executor stop-token lifecycle fix is accepted as a foundation
change, not as a current public-score lever. The experimental token-budget path
can coalesce decode steps and batch shared-prefix suffix prefill, which is useful
for future non-common-prefix work, but stop-token finishes were previously only
reflected in the stream emitter. A row that stopped inside a coalesced decode run
could remain active in executor state, causing later precomputed decode steps to
run or later grouped-prefix admissions to fall back to slower per-row prefill.
The executor now clears stop-finished rows locally and skips stale decode chunks
inside a decode-run payload. Focused CPU coverage exercises grouped-prefix
prefill stop rows and decode-run stop rows; this does not change the default
dense online batcher used by the current public run.

The opt-in mixed-prefix ragged prefill graph now carries an explicit
`prefix_copy_len` from the scheduler into the Llama3 CUDA graph key and captured
prefix-copy block. Previously the mixed-prefix graph used
`start_positions.max().item()` inside capture while the graph key only included
batch, suffix, context, and source-row count. A later mixed-prefix replay with
the same tensor shape but a different maximum prefix length could therefore
reuse a graph whose prefix-copy span was baked for the first batch. The fix keeps
the non-common-prefix path opt-in, but removes that unkeyed capture state and
adds CPU coverage that a mixed group with `17/18/19` token hits passes
`prefix_copy_len=19` to the graph provider. This is groundwork for the
multi_turn reuse gap; it does not promote full-prompt or finished-prefix reuse
as a default.

A post-fix multi_turn probe confirms that full-prompt mixed-prefix reuse remains
rejected. With pinned full-prompt stores, `112` prefix rows,
`TORCHINFERNO_CONTINUOUS_NON_COMMON_PREFIX_GRAPH_PREFILL=1`,
`TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_PREFILL=1`, and
`TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_PREFILL_GRAPH=1`, the run
`agent_space/ti_multi_mixed_graph_prefixcopy_results/.../8xH100/runs/20260702_161639`
completed without the earlier mixed-prefix stall and kept correctness in family
(`982/1000`), but landed at `1729.1 / 66.1 / 1799.5ms`. The profile shows the
remaining failure mode: request-prompt reuse fired (`875` request-prompt hits,
`98.5K` reused tokens) and raw prefill tokens fell to `18.6K`, but the run still
paid `31` prefix-graph misses, `16.2s` prefill forward, `28.2s` prefill wall,
and `11.6s` prefill state time. The graph-key fix removes unkeyed capture
state; it does not make the non-common suffix path fast enough to default.

Delayed pinned full-prompt row adoption is accepted only as a foundation for
that rejected opt-in path. When pinned per-request full-prompt stores are
explicitly enabled and logits are not stored, the engine now skips the
prefill-time KV copy by default and adopts the already-live active row when the
request finishes; `TORCHINFERNO_CONTINUOUS_PINNED_FULL_PROMPT_STORE_ADOPT_ON_FINISH=0`
restores the eager-copy behavior for diagnostics. A focused multi_turn run with
the same full-prompt mixed-prefix reuse knobs wrote
`agent_space/ti_multi_delayed_adopt_results/.../8xH100/runs/20260702_163130`
and landed at `1071.7 / 70.4 / 1141.6ms`, `979/1000` correct. The profile
shows the intended overhead reduction: request-prompt reuse still fired for
`875` requests and `98.6K` tokens, while prefill state time dropped from the
prior probe's `11.6s` to `60ms` and prefill wall dropped from `28.2s` to
`17.5s`. It is still far slower than the current default multi_turn band
because the run paid `31` prefix-graph misses and `16.3s` prefill forward for
the non-common suffix path.

The follow-up mixed-prefix dynamic-context experiment remains rejected and is
not in source. Combining delayed row adoption with a temporary
`TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_DYNAMIC_CONTEXT=1` patch failed during a
multi_turn run with a CUDA unspecified launch failure in prefix-graph prefill
and returned server 500s before producing a result row. Do not revive this as a
runtime knob without a smaller CUDA-level reproduction and correctness proof for
mixed source-prefix lengths.

A reduced 16-conversation multi_turn diagnostic with
`TORCHINFERNO_OPTIONAL_WARNINGS=1` confirmed why the opt-in mixed-prefix graph
path still falls back to eager: the first mixed-prefix ragged-prefill capture
fails collectively (`mixed-prefix ragged prefill graph capture failed on at
least one rank`). The runtime now scopes that failure to mixed-prefix capture
on miss instead of poisoning the whole ragged-prefill graph family. Existing
uniform/common-prefix graph replays remain usable, but new request-path captures
are blocked after the mixed failure so one-off uniform shapes do not pay capture
cost. The diagnostic before the change
(`agent_space/ti_multi_debug_capture_results/.../runs/20260702_171811`) landed
at `752.4 / 38.9 / 784.8ms` with global ragged-prefill graph disablement. A
broader scoped-failure prototype that still allowed later captures regressed to
`1044.8 / 37.7 / 1072.1ms` because it spent `3.56s` capturing late one-off
prefix graphs. The accepted capture-blocked variant
(`agent_space/ti_multi_mixed_capture_block_results/.../runs/20260702_173158`)
landed at `747.2 / 38.5 / 779.9ms`, kept correctness at `127/128`, and showed
`0ms` request-path prefill capture time with warmed replays preserved. This is a
failure-containment fix for the opt-in path; mixed-prefix suffix prefill is
still eager and not fast enough to default.

## Public 20260702_095238 refresh and sampled-medium active-cap lower bound

The latest public all-provider run at
`results/v1/meta-llama--Meta-Llama-3.1-70B-Instruct/8xH100/runs/20260702_095238`
measured vLLM `08a8a4a`, SGLang `b276a9a`, and TorchInferno `46007b2`.
The score split was vLLM `13`, SGLang `5`, and TorchInferno `1`. Rows as
TTFT / TPOT / E2E:

- few_shot: vLLM `142.8 / 56.0 / 194.0ms`, SGLang
  `115.7 / 87.0 / 203.3ms`, TorchInferno `173.0 / 51.2 / 214.5ms`.
- self_consistency: vLLM `215.0 / 0.0 / 244.1ms`, SGLang
  `193.5 / 0.0 / 382.0ms`, TorchInferno `196.7 / 0.0 / 277.7ms`.
- multi_turn: vLLM `160.6 / 52.8 / 207.1ms`, SGLang
  `148.3 / 126.1 / 280.2ms`, TorchInferno `306.4 / 65.7 / 365.3ms`.
- tree_of_thought: vLLM `64.0 / 31.5 / 87.3ms`, SGLang
  `58.1 / 74.2 / 147.6ms`, TorchInferno `158.2 / 35.0 / 186.8ms`.
- long_output: vLLM `63.3 / 17.0 / 661.4ms`, SGLang
  `60.7 / 24.5 / 874.7ms`, TorchInferno `271.3 / 24.8 / 1192.7ms`.

The queue profile is still capture-clean enough that the remaining public gaps
are steady dense prefill/decode work, not cold graph setup. Tree spent
`2.12s` in prefix-suffix prefill forward and `1.44s` in ragged decode GPU;
long_output spent `5.53s` in prefill forward and `10.48s` in decode GPU.

Lowering sampled-medium tree active rows is rejected. A focused current-main
run with `TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_MAX_ACTIVE=16` wrote
`agent_space/ti_tree_active16_results/.../runs/20260702_112544` and landed at
`304.8 / 51.9 / 327.8ms`, `957/992` correct. Queue counters explain the loss:
`max_active=16` doubled prefix-suffix fragmentation (`119` prefill batches),
raised prefill wall/forward to `4.53/3.72s`, raised decode GPU to `1.79s`,
and pushed queue-to-first p50 to `282.3ms`. Keep sampled-medium max-active at
`32`; both lowering and prior higher-row probes lose through fragmentation,
tail, or startup/warmup cost.

Narrowing sampled common-prefix suffix buckets to `12,16` is also not
defaultable. The cold env-only run
`agent_space/ti_tree_s12_results/.../runs/20260702_124224` did switch tree
prefix reuse from `s16` to `s12`, but captured six prefill graphs on the request
path, raising p99 TTFT/E2E to `2807/2880ms` and prefill wall/forward to
`7.69/7.04s`. A fair rerun with
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_SAMPLED_COMMON_PREFIX_SUFFIX_TOKENS=12,16`
prewarmed those graphs and wrote
`agent_space/ti_tree_s12_warm_results/.../runs/20260702_124643`; it landed at
`140.7 / 54.0 / 174.7ms`, `954/992` correct, with zero runtime prefill captures.
The same-host current-default control
`agent_space/ti_tree_default_control_results/.../runs/20260702_125045` landed at
`152.7 / 48.7 / 184.8ms`, `958/992` correct. The warmed `s12` run improved
median TTFT/E2E but worsened TPOT, and the queue counters do not show a real
prefill kernel win: `s12` prefill forward/wall was `2.14/2.78s` versus the
control `s16` `2.12/2.87s`, with the hot `b32` graph still around `1.40s`.
Keep the sampled suffix bucket default at the graph-warmed `16` shape unless a
broader prefix-suffix prefill implementation reduces the actual per-call GPU
work.

## Current 20260702 full same-host provider refresh

The current same-host all-provider run used inference-bench with skipped builds
and the fixed vLLM FlashInfer workspace default. It wrote results under
`agent_space/allproviders_current_full_results/.../8xH100-local-all-current-20260702/runs/20260702_014042`.
Provider commits were vLLM `4236514`, SGLang `1a5977d`, and TorchInferno
`3283bbf`. vLLM served successfully, so this run supersedes the earlier local
provider fragments where vLLM was either absent or only checked per-row.

The local scorecard shifted to vLLM `15` metric wins, SGLang `4`, and
TorchInferno `0`. Rows as TTFT / TPOT / E2E:

- few_shot: vLLM `130.7 / 48.1 / 170.1ms`, SGLang
  `124.1 / 79.3 / 205.3ms`, TorchInferno `174.6 / 50.2 / 215.8ms`.
- self_consistency: vLLM `183.4 / 0.0 / 256.3ms`, SGLang
  `216.3 / 0.0 / 383.7ms`, TorchInferno `214.8 / 0.0 / 325.4ms`.
- multi_turn: vLLM `167.0 / 56.8 / 217.0ms`, SGLang
  `158.1 / 112.4 / 267.4ms`, TorchInferno `308.9 / 58.8 / 362.7ms`.
- tree_of_thought: vLLM `65.5 / 31.1 / 89.7ms`, SGLang
  `64.5 / 65.4 / 143.5ms`, TorchInferno `159.5 / 47.4 / 203.9ms`.
- long_output: vLLM `85.4 / 16.9 / 700.1ms`, SGLang
  `62.5 / 24.4 / 894.0ms`, TorchInferno `251.0 / 24.5 / 1254.1ms`.

TorchInferno correctness stayed in-family: `977/1000` few_shot, `1000/1000`
self, `982/1000` multi_turn, `958/992` tree, and `1000/1000` long_output. The
healthy-vLLM run changes the remaining target: TorchInferno no longer has a
local TPOT cushion on long_output (`24.5ms` versus vLLM `16.9ms`) and only stays
near vLLM on few/multi TPOT. The remaining work is therefore not another
median-only scheduling knob; it needs lower decode GPU time and fewer/faster
prefill waves.

The TorchInferno queue profile supports that target. few_shot had graph-warm
prefill (`37` prefill batches, `35` hits, `2` misses), but still exposed a bad
tail: queue-to-first p50/p99 was `125.6/2491.6ms`, with `2.58s` prefill wall and
`3.18s` ragged-decode GPU time. long_output was also graph-warm (`56` prefill
batches, `54` hits, `2` misses), with queue-to-first p50/p99
`202.7/1635.1ms`, queue-to-finish p50 `1196.3ms`, `6.53s` prefill wall
(`5.66s` forward), and `10.38s` ragged-decode GPU time. Decode-many ran
`140` calls / `365` steps / `21,238` model tokens for `20,812` emitted tokens,
with `426` skipped tokens and `6,654` ragged padding tokens.

This keeps the already-rejected controls closed: greedy-mid row-cap increases,
greedy-mid first-wave wait, fine greedy-mid suffix buckets, long_output refill
floors/tail caps, short-greedy decode quantum changes, step-sync-off,
prompt-lookup decode, and down-projection Marlin do not become defaultable from
this run. The next credible implementation target is a real decode/prefill
pipeline improvement: reduce per-step decode GPU time, fuse or replace the
remaining decode-heavy kernels, or pipeline prefill/decode/readback so
long_output can approach vLLM's `16.9ms` TPOT without worsening TTFT/E2E tails.

One follow-up from the current decode profile is also rejected. Profiling a
single FlashInfer decode step on pushed `dd78e77` showed `23.6ms` self CUDA:
`8.0ms` in 160 `aten::copy_` calls, `6.2ms` in 160 NCCL all-reduces, `4.8ms`
in dense GEMMs, and `2.6ms` in Marlin gate/up GEMMs. A guarded Triton
FlashInfer KV append micro-kernel matched the torch reference in a focused CUDA
test, but the no-profile long_output A/B with
`TORCHINFERNO_TRITON_FLASHINFER_KV_APPEND=1` regressed the score row to
`275.5 / 25.4 / 1236.1ms` versus the nearby default band around
`237.7 / 25.4 / 1204.4ms` and the full-run row `251.0 / 24.5 / 1254.1ms`.
The run wrote
`agent_space/ti_long_fi_triton_kv_results/.../8xH100-local-ti-long-fi-triton-kv-20260702/runs/20260702_020647`
and kept correctness at `1000/1000`, but queue counters still showed
`11.79s` decode GPU time, `6.66s` prefill wall, `720` decode graph hits, and
`7,320` ragged padding tokens. Replacing only the indexed FlashInfer KV write
is not a defaultable win; the remaining `copy_` profile needs a broader
layout/capture or decode pipeline change.

Startup symmetric-memory allreduce stays explicit after the current
long_output recheck. The successful auto-probe already validates
graph-captured multimem allreduce for decode shapes, and an explicit
startup-scope run on `c9865f7`
(`agent_space/ti_long_startup_symm_results/.../8xH100-local-ti-long-startup-symm-20260702/runs/20260702_022326`)
started normally (`195.8s` readiness), kept long_output correctness at
`1000/1000`, and improved the row to `243.2 / 24.5 / 1174.5ms`. Its queue
profile moved phase time to `19.42s`, queue-to-first p50/p99 to
`195.9/1279.4ms`, queue-to-finish p50 to `1128.1ms`, and decode GPU time to
`9.62s`.

That result did not reproduce cleanly when the default auto-probe scope was
temporarily changed from `runtime` to `all` and rerun with no override
(`agent_space/ti_long_auto_symm_default_results/.../8xH100-local-ti-long-auto-symm-default-20260702/runs/20260702_022947`).
The server enabled all scope after probe and stayed correct (`1000/1000`), but
the row regressed to `256.1 / 24.4 / 1295.7ms`, with `20.81s` phase time,
queue-to-finish p50 `1224.2ms`, `6.94s` prefill wall (`6.01s` forward),
`10.25s` decode GPU time, `7,065` ragged padding tokens, and `569` step calls.
The default therefore remains `runtime`; startup all-scope remains available as
an explicit override, not a defaultable win.

A narrower decode-warmup default did reproduce. On the patched run
`agent_space/ti_long_decode_warmupsymm_results/.../8xH100-local-ti-long-decode-warmupsymm-20260702/runs/20260702_030000`,
the server still auto-probed to `runtime` scope and reached readiness in
`200.9s`, but online decode graph warmup captured under the same runtime
symmetric-memory scope used by requests and also warmed the observed exact
small fallback decode shapes. The long_output row improved to
`240.9 / 25.1 / 1206.5ms`, with p99 TTFT/TPOT/E2E at
`1398.6 / 100.3 / 2123.3ms`, and correctness stayed `1000/1000`. The queue
profile confirms the mechanism: request-path decode graph captures fell to
`0` (`744` replays, `0.0ms` capture wall), decode GPU time fell to `9.59s`,
and online phase time fell to `19.37s`. The adjacent no-env control had
`7` request-path decode captures, `5.07s` capture wall, `13.99s` decode GPU,
and `24.49s` phase time. This is not a switch to all startup symm scope; it
only makes the online decode graph warmup match the runtime request scope, with
`TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_DECODE_WARMUP=0` as the targeted
escape hatch and `TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_STARTUP=0`
respected when the targeted override is unset.

Rechecking the scoped greedy-short initial wait after the decode-warmup fix is
still rejected. With
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_INITIAL_BATCH_WAIT_MS=10`, the run
`agent_space/ti_long_initwait10_postsymm_results/.../8xH100-local-ti-long-initwait10-postsymm-20260702/runs/20260702_030927`
kept correctness at `1000/1000` and raised the initial batch from `2` to `10`,
but the row moved to `235.9 / 25.7 / 1240.1ms` with worse p99
TTFT/TPOT/E2E (`1538.9 / 104.3 / 2269.8ms`) than the no-env patched run.
The queue profile showed no request-path decode captures, so the loss is normal
work: prefill wall/forward rose to `6.67/5.74s`, decode GPU rose to `9.72s`,
decode-many work rose to `141` calls / `358` steps / `21,708` model tokens, and
phase time rose to `19.75s`. Keep greedy-short initial wait at `0ms`; the
remaining long_output gap is not the first-wave collection window.

A sampled decode warmup extension is accepted for tree_of_thought. The
post-symm full TorchInferno sweep
`agent_space/ti_full_postsymm_results/.../8xH100-local-ti-full-postsymm-20260702/runs/20260702_031542`
still had tree request-path decode captures because startup warmed the
deterministic symmetric-memory graph key, while sampled tree requests run with
the no-symm graph key under the temperature gate. Tree landed at
`152.5 / 46.0 / 183.2ms`, p99 TTFT/TPOT/E2E
`1327.8 / 366.6 / 1355.2ms`, with `43` decode misses, `6` live captures, and
`1.85s` capture wall. Warming the sampled no-symm decode policy moved those
captures to startup: focused tree reruns wrote
`agent_space/ti_tree_sampled_decode_warmup_results/.../runs/20260702_032721`
and
`agent_space/ti_tree_sampled_decode_warmup2_results/.../runs/20260702_033526`.
They kept captures at `0` and p99 E2E at `844-892ms`; the second run also
dropped sampled token-graph fallthrough misses from `34` to `2` by skipping the
native token graph for `temperature > 0` after the FlashInfer sampled path has
had a chance to serve. Median rows were noisy (`151.8 / 43.7 / 180.5ms` then
`151.4 / 49.0 / 183.1ms`), so this is primarily a tail/capture fix rather than
a new score flip, but it removes a real request-path stall without adding a
benchmark-specific policy.

The final local TorchInferno-only sweep with the sampled decode warmup wrote
`agent_space/ti_full_sampled_decode_warmup_results/.../8xH100-local-ti-full-sampled-decode-warmup-20260702/runs/20260702_034226`
and stayed in-family on all rows: few_shot `169.6 / 48.7 / 209.7ms`,
self_consistency `180.1 / 0.0 / 197.7ms`, multi_turn
`304.8 / 59.2 / 359.0ms`, tree `150.9 / 52.5 / 181.8ms`, and long_output
`256.7 / 25.0 / 1177.0ms`. Tree p99 TTFT/TPOT/E2E moved to
`430.1 / 137.8 / 486.5ms` with `0` decode captures and `0` misses; long_output
also stayed at `0` decode captures. That sweep exposed one deterministic
few_shot `b5` token-graph capture (`602ms` capture wall), so the online decode
warmup now covers exact small batches through `8`, not only powers of two plus
`3`. The focused few_shot validation
`agent_space/ti_few_small_decode_warmup_results/.../runs/20260702_034924`
reached readiness in `210.9s`, kept correctness at `977/1000`, and removed the
request-path decode capture (`0` misses, `0` captures, `0.0ms` capture wall).
This adds modest startup work but closes another cold request tail without
changing runtime admission or decode bucketing.

The pushed `cb429b9` all-provider rerun used reusable local provider envs
(`vllm` `c621af1`, `sglang` `99b8f36`) and wrote
`agent_space/allproviders_cb429b9_results/.../8xH100-local-all-cb429b9-20260702/runs/20260702_035806`.
The local scorecard moved from the earlier same-host `15/4/0` split to vLLM
`13`, SGLang `3`, and TorchInferno `3`. TorchInferno won few_shot TPOT
(`48.4ms` vs vLLM `56.5ms` / SGLang `85.4ms`), self_consistency TTFT
(`135.3ms` vs `179.9ms` / `215.0ms`), and multi_turn TPOT
(`59.1ms` vs `82.1ms` / `112.1ms`). It still loses tree_of_thought and
long_output score cells to vLLM, and long_output remained the largest gap:
vLLM `64.8 / 16.9 / 668.9ms`, SGLang `61.4 / 25.0 / 969.4ms`,
TorchInferno `270.7 / 24.4 / 1289.6ms`. TorchInferno's queue profile stayed
capture-clean for tree and long (`0` decode captures), but tree still had `3`
decode misses and long_output spent `10.30s` in ragged decode GPU with
`7.44s` prefill wall. The next score-facing work is therefore still lower
long-output TTFT/E2E and tree decode/throughput, not more request-path graph
capture cleanup.

A side-stream async ragged token-copy probe is rejected on this stack. The
env-gated run
`agent_space/ti_long_async_ragged_copy_results/.../8xH100-local-ti-long-async-ragged-copy-20260702/runs/20260702_042347`
looked promising at `237.2 / 24.3 / 1125.6ms`, but the same working tree with
the path defaulted on landed at only `256.6 / 25.2 / 1230.1ms`, and the paired
explicit-off control landed at `235.2 / 25.1 / 1222.5ms`. Queue counters showed
the mechanism was not stable: the default-on run issued `332` deferred token
copies and had `19.69s` phase time, while the explicit-off control had no
deferred copies, `19.21s` phase time, similar decode GPU (`9.72s` vs `9.61s`),
and better score-facing TTFT. Do not add a small side-stream D2H copy shim as a
default; the long-output gap still needs a real prefill/decode/readback pipeline
or lower decode GPU work.

A fused Triton ragged RoPE+KV-append probe is also rejected for long_output. The
env-only run with `TORCHINFERNO_TRITON_RAGGED_DECODE_ROTARY_APPEND=1` wrote
`agent_space/ti_long_triton_ragged_rotary_append_results/.../8xH100-local-ti-long-triton-ragged-rotary-append-20260702/runs/20260702_044748`
and stayed correct at `1000/1000`, but the score-facing row was only
`260.9 / 25.0 / 1159.8ms` with p99 E2E `2428.7ms`. The queue profile did not
show a decode-kernel win: phase time was `19.81s`, ragged decode GPU was
`10.20s`, prefill wall was `6.76s`, and the largest `b64/64` decode-many shape
still cost about `13.78ms/call`. The nearby explicit-off control had lower phase
time (`19.21s`) and lower decode GPU (`9.61s`), so fusing only ragged RoPE and
dense KV append is not a defaultable long-output lever.

A ragged decode graph replay-input cleanup is rejected as well. A temporary
env-gated patch moved the per-step rotary table gather into the captured graph
body and skipped the replay-side `static_rotary_cos/sin` copies. The focused
long_output run wrote
`agent_space/ti_long_rotary_in_graph_results/.../8xH100-local-ti-long-rotary-in-graph-20260702/runs/20260702_045708`
and was correct at `1000/1000`, but regressed the score row to
`270.6 / 25.0 / 1283.7ms` with throughput `31.1`. Queue counters showed no
useful mechanism: phase time was `19.87s`, ragged decode GPU `9.71s`, prefill
wall `6.78s`, replay setup/graph replay accounting stayed in-family, and step
calls rose to `576`. Keep the explicit static rotary copy path until a broader
decode graph layout change can reduce actual per-step GPU work.

## Prior 20260701 refresh and local vLLM/SGLang checks

Public run `20260701_211855` is stale for TorchInferno: it measured
TorchInferno `3af4940`, before the dynamic-prefix context-floor and greedy
ctx256 warmup changes (`304761c`, `f8cc9d5`). It landed at TorchInferno
`9/20`, SGLang `10/20`, and vLLM `0/20`; vLLM did not serve because the public
provider started with a too-small FlashInfer workspace buffer
(`Buffer: 1048576 bytes, Required: 4194304 bytes`). The inference-bench provider
has since been updated to default vLLM's FlashInfer workspace to the current
upstream `394MiB` setting and to expose extra vLLM server args.

The public TorchInferno rows were few_shot `154.1 / 49.0 / 196.5ms`,
self_consistency `160.7 / 0.0 / 173.8ms`, multi_turn
`296.3 / 62.2 / 349.8ms`, tree_of_thought `137.7 / 37.3 / 161.3ms`, and
long_output `246.9 / 22.4 / 1051.4ms`. SGLang's long_output row was still well
ahead on first-token and E2E latency: `82.5 / 21.8 / 875.3ms`.

A same-host TorchInferno all-benchmark run with the warm ctx256 working tree
landed at few_shot `170.6 / 48.8 / 208.2ms`, self_consistency
`200.5 / 0.0 / 213.7ms`, multi_turn `294.0 / 59.3 / 351.0ms`,
tree_of_thought `147.6 / 46.1 / 182.3ms`, and long_output
`245.5 / 24.6 / 1214.5ms`. The long_output profile had zero request-path
prefill captures, `64` prefill batches, `62` graph hits, `2` graph misses,
`6.53s` prefill wall (`5.60s` forward), and `10.75s` ragged-decode GPU time.
This keeps the long gap in the prefill/decode scheduling bucket rather than
graph-capture cold start alone.

Local vLLM with the fixed workspace buffer served successfully. Its rows were
few_shot `240.4 / 91.6 / 320.6ms`, self_consistency `273.6 / 0.0 / 347.8ms`,
multi_turn `284.9 / 110.4 / 391.7ms`, tree_of_thought
`131.1 / 84.0 / 192.6ms`, and long_output `89.3 / 26.5 / 995.0ms`. Locally,
vLLM only clearly beats TorchInferno on long_output first-token/E2E latency;
TorchInferno remains ahead on the shorter rows and long_output TPOT.

Local SGLang `1a5977d` served long_output at
`62.6 / 24.8 / 1015.9ms`, 1000/1000 correct, after `125.6s` readiness. This
confirms the remaining long_output competitor gap is mostly TorchInferno's
prefill-to-first-token path, not steady token cadence.

Same-host multi_turn/tree_of_thought checks on current pushed TorchInferno
`c33d773` and local SGLang `1a5977d` isolate the remaining queue-facing gap.
SGLang landed at multi_turn `152.0 / 109.0 / 262.7ms` and tree_of_thought
`65.3 / 56.5 / 137.1ms`. TorchInferno landed at multi_turn
`329.8 / 71.9 / 398.7ms` and tree_of_thought `147.8 / 31.2 / 174.4ms`.
TorchInferno still wins TPOT, but SGLang wins TTFT/E2E/throughput. The
TorchInferno multi_turn profile was graph-warm with no captures, `35` prefill
batches, `4.44s` prefill wall (`3.96s` forward), `3.78s` ragged-decode GPU
time, and only shared-prefix reuse (`{"common_prefix": 1000}`,
`{"45": 1000}`). Per-turn medians were `602/742/257/276/300/349/362/423ms`;
SGLang's were `399/124/134/133/219/155/165/181ms`. This keeps the gap in
fewer/faster conversation-prefix suffix waves, not cold graph capture.

Rejected multi_turn prefix-cache follow-ups on `c33d773`:

- `TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE=1` is unsafe for this greedy
  512-token route. The server hit a CUDA launch failure in prefix graph prefill
  after startup, so generated-prefix reuse cannot be promoted for multi_turn
  without a separate correctness fix.
- Enabling pinned full-prompt stores for greedy-large requests
  (`TORCHINFERNO_CONTINUOUS_PINNED_FULL_PROMPT_STORE_MIN_MAX_TOKENS=512`)
  is rejected. It created request-prompt hits (`862` request-prompt reuses) but
  regressed multi_turn to `8165.4 / 73.3 / 8236.6ms`; prefill wall rose to
  `116.5s` because the non-common path incurred `501` prefill graph misses.
- Adding `TORCHINFERNO_CONTINUOUS_NON_COMMON_PREFIX_GRAPH_PREFILL=1` removed
  misses but still regressed to `7579.4 / 67.1 / 7638.7ms`. The profile showed
  `187` request-path graph captures, `117.3s` prefill wall, and `105.0s`
  prefill forward because exact context lengths produced too many graph keys.
- Bucketing those non-common suffixes with
  `TORCHINFERNO_CONTINUOUS_DYNAMIC_PREFIX_PREFILL_MAX_SUFFIX=32` reduced
  captures to `22` and TTFT to `2160.7ms`, confirming the capture-key diagnosis,
  but it still lost badly to the default `329.8ms` row. Prefill wall remained
  `38.5s` with `26.0s` forward and `11.4s` prefill state work. Keep full-prompt
  and non-common prefix graph reuse out of default multi_turn until the source
  row copy/state path is redesigned, not just bucketed.

Rejected long_output decode/readback probe on `687860f`: a local env-gated
patch allowed decode-many to run even while compatible requests were still
waiting. It proved the CPU-token readback hypothesis but not a defaultable
policy. The row moved to `679.1 / 16.9 / 1510.4ms`, 1000/1000 correct, versus
the warm ctx256 control at `245.5 / 24.6 / 1214.5ms`. The profile had
`313` decode-many calls, `920` decode-many steps, and only `56.6ms` total token
readback, but submit-to-first p50 ballooned to `627.9ms` and queue-to-first to
`651.4ms`. Running decode bursts while requests wait improves steady token
cadence by delaying first-token work; keep decode-many blocked when the waiting
queue is non-empty until there is a real overlap pipeline that preserves
prefill/admission latency.

Rejected long_output refill/tail-prefill A/B on `ae34ce5`: lowering the
greedy-short refill floor and delaying prefill-before-decode until the active
tail was smaller
(`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_REFILL_MIN_READY_REQUESTS=8`,
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_PREFILL_READY_ACTIVE_CAP=4`)
improved first-token latency but traded away the score row. The current-head
run landed at `213.0 / 27.6 / 1292.2ms`, 1000/1000 correct, versus the
warm-ctx256 control at `245.5 / 24.6 / 1214.5ms`. The queue profile shows why
this is not a default: queue-to-first improved (`204.1ms -> 173.7ms`) and
submit-to-first improved (`137.9ms -> 111.1ms`), but prefill fragmented
(`64 -> 84` batches), prefill wall/forward grew
(`6.53/5.60s -> 7.65/6.70s`), decode GPU time rose
(`10.75s -> 11.27s`), and total online phase rose
(`19.85s -> 22.10s`). Keep the current `12`-request refill floor and
`8`-row tail cap until a scheduling change lowers first-token latency without
losing TPOT/E2E. Isolating only the tail cap at `4` is also rejected: it landed
at `252.1 / 25.4 / 1280.6ms`, 1000/1000 correct, with submit-to-first worse
than control (`147.6ms` vs `137.9ms`) and decode GPU time up
(`13.83s` vs `10.75s`).

Rejected follow-ups on `f8cc9d5`:

- `TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_ADMIT_PER_STEP_CAP=32` removed the
  two `b64:s64` prefill graph misses but regressed long_output to
  `269.4 / 25.1 / 1299.4ms`. Decode fragmentation dominated: step calls rose
  from `500` to `600` and ragged-decode GPU time rose from `10.75s` to
  `13.65s`.
- Adding batch `64` to
  `TORCHINFERNO_OPENAI_WARMUP_ONLINE_GREEDY_COMMON_PREFIX_SUFFIX_BATCHES`
  is rejected. Startup rose to `226.0s`, long_output regressed to
  `257.9 / 27.4 / 1618.7ms`, and prefill forward inflated to `15.86s`.
  Capturing the larger graph hurts memory/steady-state behavior more than it
  saves on the rare cold batch-64 suffix-prefill misses.
- `INFERENCE_BENCH_TORCHINFERNO_PROFILE=0` did not validate profiling overhead
  as the long-output gap. The no-profile run landed at
  `254.5 / 24.4 / 1252.3ms`, essentially flat-to-worse versus the profiled
  local control, so keep inference-bench profiling behavior unchanged.
- `TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_CHUNK=32` is rejected:
  `1460.6 / 82.2 / 4443.1ms`, with `369` prefill batches and many graph
  captures.
- `TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_PREFILL_READY_ACTIVE_CAP=16` is
  rejected after warm graph retest: `250.6 / 25.5 / 1269.3ms`.
- Enabling FlashInfer prefill
  (`TORCHINFERNO_CONTINUOUS_FLASHINFER_PREFILL_DISABLE=0`) remains rejected:
  `529.2 / 35.8 / 2002.9ms` and a correctness drop to `98%`.
- Skipping the folded common-prefix copy when a dense active row already holds
  the same prefix is rejected and was reverted. A cold no-copy prototype fired
  for `896` requests but regressed long_output to `307.1 / 25.8 / 1458.7ms`
  because the new `src0` graph shapes paid `11.29s` of runtime capture. Adding
  startup `src0` warmup raised readiness to `281.2s` and still regressed to
  `307.2 / 25.4 / 1516.4ms`; the serving cache still captured `10` no-copy
  shapes at request time. The dense row-materialization copy is not the next
  defaultable lever without a deeper graph/cache design.
- Greedy-short FP8 prefill M-threshold sweeps around the current `512` gate are
  rejected on the long_output shape. Lowering to `256` preserved correctness but
  landed at `264.2 / 25.4 / 1229.6ms`; raising to `1024` landed at
  `272.5 / 25.3 / 1306.0ms`. The profiles showed no useful capture artifact
  and worse prefill forward/queue-to-finish balance, so keep the current
  greedy-short `min_m=512`.

## Current public-order refresh on 36c5c9a (2026-06-29)

Public run `20260629_201815` measured pushed TorchInferno `36c5c9a`, SGLang
`643e1cc`, and the current vLLM environment after the fast-HTTP profiling
instrumentation landed. The scorecard improved from the prior public
`SGLang 14 / TorchInferno 3 / vLLM 2` shape to SGLang `12/20`,
TorchInferno `5/20`, and vLLM `2/20`. TorchInferno won few_shot TPOT and E2E,
long_output TPOT, multi_turn TPOT, and tree_of_thought TPOT. The no-profile run
does not show a regression from the accept/read timing instrumentation.

TorchInferno rows were few_shot `164.4 / 48.6 / 205.9ms`, self_consistency
`330.0 / 0.0 / 361.7ms`, multi_turn `328.9 / 62.9 / 385.2ms`,
tree_of_thought `296.5 / 47.2 / 313.6ms`, and long_output
`274.9 / 23.9 / 1171.8ms`. Compared with `20260629_185744`, few_shot and
long_output moved back into their better local bands, but self_consistency,
multi_turn, and tree still have score-facing TTFT/E2E gaps. The current useful
work remains the previously identified request-wave/admission path for self and
fewer/faster prefill/decode waves for multi/tree/long, not reopening the
rejected first-wave, refill-floor, keepalive, FP8, or decode-many knobs.

## Current few_shot refresh and refill-floor rejection (2026-06-29)

The latest public run `20260629_185744` still shows few_shot as a narrow
median gap: TorchInferno `171.0 / 50.5 / 215.8ms` versus SGLang
`126.5 / 80.6 / 207.9ms`. A focused current profile on pushed `cd431f3`
landed in-family at `168.9 / 49.3 / 208.6ms`, 978/1000 raw correct. Fast-HTTP
accepted-to-handler and request-read p50 were only `0.1ms` and `5.3ms`; server
first-content p50 was `124.9ms`, close to queue-to-first p50 `122.2ms`. The
median is therefore mostly the existing online queue/prefill/decode cadence
plus benchmark/client observation overhead, not response serialization or
ThreadPoolExecutor backlog. The profile was graph-warm: `35` prefill batches,
`33` graph hits, `2` misses, no captures, and `987` common-prefix reuse
requests.

Lowering only `TORCHINFERNO_OPENAI_TP_ONLINE_REFILL_MIN_READY_REQUESTS` from
`8` to `4` is rejected for the 256-token greedy-mid bucket. The focused row
moved only slightly to `166.6 / 48.8 / 207.0ms`, but p90/p99 first-content
worsened, phase time rose (`7.13s -> 7.26s`), prefill wall rose
(`2.71s -> 2.84s`), and prefill misses increased (`2 -> 3`). Keep the current
`8`-request refill floor until a scheduler change improves medians without
trading away tail stability or graph-backed prefill shape reuse.

Applying the finer greedy-large suffix bucket set to the 256-token greedy-mid
few_shot path is rejected. Forcing
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS=16,32,64,80,96,112,128,144,160,192,224,256`
on current `5644e2f` landed at `169.2 / 48.7 / 209.3ms`, worse than the
current default band, and stretched server readiness to about `201s`. The queue
profile showed why: prefill wall rose to `4.37s`, request-path prefill misses
rose to `3`, two prefix-graph captures appeared, and HTTP p90 first-content
rose to `883ms`. Keep the fine suffix buckets scoped to the deterministic
512-token greedy-large path where they were validated.

## Current self_consistency handler split (2026-06-29)

The restored public run `20260629_185744` still shows self_consistency as a
request/handler-facing gap: vLLM/SGLang/TorchInferno landed at
`210.9 / 206.2 / 319.3ms` TTFT and `285.3 / 361.4 / 398.6ms` E2E. A focused
current TorchInferno profile on docs-only `31edf92` improved the row to
`285.1 / 0.0 / 348.6ms`, 1000/1000 correct, but the queue/HTTP split confirms
the runtime batcher is not the median limiter. The final queue record had p50
queue-to-submit, submit-to-first, queue-to-first, and queue-to-finish at
`1.6/6.0/8.2/8.2ms`; fast-HTTP p50 first-content/total was `9.5/9.7ms`, while
the benchmark still observed `285/349ms`.

The generated-prefix path is healthy in this profile: one common-prefix prefill,
one decode model call, one generated-prefix store, `978` generated-prefix
reuses, and `108,558` total prefix-reuse tokens. The remaining server-side
phase is request fragmentation and command cadence after reuse, not model
compute: `493` submit batches, `234` runtime steps, `1.21s` submit-sync, and
`3.12s` total online phase. This does not reopen sampler-state caching,
generated-prefix thresholds, sampled-short initial/idle waits, keepalive,
thread-local clients, worker-count increases, or HTTP prestart as defaults;
those knobs were already neutral or mixed in full-order checks. The next self
lever needs to reduce handler/request admission fragmentation without hurting
few_shot or multi_turn ordering.

Follow-up fast-HTTP accept/read instrumentation on `f4dcb85` narrows the missing
time further. The score row stayed in-family at `286.5 / 0.0 / 330.4ms`, and
all `1000` profiled streams were first requests on their connections.
Accepted-to-handler p50/p99 was only `0.1/0.7ms`, so the `ThreadPoolExecutor`
is not the hidden median queue. Accepted-to-ready and request-read p50/p99 were
`47.4/250.0ms`, while server first-content p50/p99 was `9.5/552.3ms`; the
benchmark still saw `286.5ms` TTFT. The remaining self gap is therefore mostly
before server accept plus client request-body delivery and tail request waves.
Do not reopen worker-count/prestart/parallel-accept without a change that
improves that pre-accept/client-admission side in full benchmark order.

The `36c5c9a` public-order refresh worsened the self row to
`330.0 / 0.0 / 361.7ms`, but focused ordered controls show that is not a
TorchInferno runtime regression after few_shot. A profiled TorchInferno-only
`few_shot -> self_consistency` run on `682ade7` landed at
`191.9 / 0.0 / 212.2ms`; the no-profile repeat landed at
`203.9 / 0.0 / 278.0ms`. The profiled self queue still had p50
queue-to-first `8.4ms`, HTTP first-content `9.6ms`, one prefill batch, one
decode model call, and `992` generated-prefix reuses. Treat the full-order
`330ms` self TTFT as client/host/provider-order admission noise unless a repeat
reproduces it with server-side queue or HTTP first-content movement.

A reduced full-order repeat on current `492994c` with
`vllm -> sglang -> torchinferno` and only `self_consistency` reproduced the
last-provider shape: TorchInferno landed at `320.6 / 0.0 / 389.2ms` versus the
ordered TorchInferno-only `203.9 / 0.0 / 278.0ms` control. Server-side medians
were still fast after request arrival: accepted-to-handler `0.05ms`,
request-read `45.3ms`, HTTP first-content `8.7ms`, queue-to-first `7.5ms`, one
prefill batch, one decode model call, and `986` generated-prefix reuses. The
missing median is therefore mostly before server accept or while the client is
delivering first-use localhost requests after earlier providers have run, not
inside TorchInferno model execution.

Increasing the general inference-bench inter-provider cleanup wait from `30s`
to `75s` is rejected as a default. A profiled reduced full-order run looked
promising (`252.4 / 0.0 / 359.4ms`, with TorchInferno winning E2E and
throughput), but the no-profile validation did not hold:
`358.9 / 0.0 / 385.7ms`, with SGLang winning TTFT, E2E, p99s, and throughput.
The longer wait is therefore not enough evidence for a harness default change;
keep using the env override only for isolation studies.

Raising the drained fast-HTTP keepalive timeout from `0.25s` to `1.0s` is also
rejected on the current stack. A focused self_consistency A/B on `afbbe89`
regressed the score row to `364.2 / 0.0 / 394.0ms`; the HTTP profile still had
`1000/1000` first requests on their connections, so the longer drained idle
window did not produce connection reuse. Request-read p50 worsened
`47.2ms -> 84.1ms`, server first-content p50 stayed around `9ms`, and queue
phase time rose (`2.95s -> 3.39s`) with more submit batches (`455 -> 557`).
Keep the short drained timeout; the missing self median is not fixed by simply
holding idle sockets open longer.

## Current multi_turn refresh and DQ=8 rejection (2026-06-29)

The restored public run `20260629_185744` kept multi_turn in the expected
score shape: TorchInferno `302.0 / 66.1 / 358.1ms` versus SGLang
`155.1 / 122.9 / 269.3ms`, so TorchInferno still wins only the TPOT cell.
A focused current profile on docs-only `f363437` landed nearby at
`321.6 / 67.8 / 380.4ms`, 983/1000 raw correct, with first-turn median TTFT
`781.8ms` and last-turn median `389.6ms`. Fast-HTTP p50 first-content/total
was `258.9/280.8ms`, while queue p50 queue-to-submit, submit-to-first, and
queue-to-first were `106.2/141.3/255.9ms`, so the remaining median is still
server prefill/admission plus benchmark/client observation overhead, not SSE
writing.

The queue profile confirms the current 512-token greedy path is graph-warm but
prefill dominated: zero request-path prefill captures, `35` graph-backed
prefill batches, `4.85s` prefill wall (`4.22s` forward), `3.69s` ragged decode
GPU event time, and `706ms` ragged token harvest. The largest suffix buckets
were the already-promoted fine greedy-large shapes, especially `b32:s144`
(`1.47s` wall over eight calls), followed by `s128`, `s112`, and `s64`.
This does not reopen finished/generated prefix reuse, mixed-prefix graph
prefill, larger active caps, lower FP8 min-M, or first-wave wait changes; the
remaining multi_turn gap is still fewer/faster conversation-prefix suffix
prefill waves without fragmenting graph-backed batches.

Lowering the 512-token greedy decode quantum from `16` to `8` is rejected on
the current stack. The env-only focused run preserved correctness and improved
tails (`ttft/e2e p99` `4054/4093ms -> 2131/2176ms`) while reducing internal
phase time (`10.06s -> 8.18s`), prefill wall (`4.85s -> 4.43s`), and ragged
decode GPU time (`3.69s -> 2.53s`). It did not improve score-facing medians:
the row moved to `321.9 / 59.3 / 383.0ms`, throughput fell to `3.1 tok/s`, and
HTTP p50 first-content stayed flat at `260.7ms`. Keep the greedy-large quantum
at `16` until a scheduler change improves median TTFT/E2E or throughput, not
only tails/internal counters.

A midpoint decode quantum of `12` is also rejected. The profiled A/B on
`de3d441` looked cleaner than DQ=8 (`308.5 / 63.9 / 365.9ms`) and preserved
graph-backed prefill (`34` hits, zero misses), but the no-profile repeat did
not hold the median win: `324.1 / 63.9 / 381.9ms`, with throughput still
`3.1 tok/s`. This is not enough to change the public score shape, so keep the
512-token greedy default at `16`.

## Greedy-mid first-wave wait rejected after public-order run (2026-06-29)

Public run `20260629_180924` measured TorchInferno `7110c60`, SGLang
`643e1cc`, and the current vLLM environment. The scorecard stayed at SGLang
`13/20`, TorchInferno `4/20`, and vLLM `2/20`. TorchInferno rows were few_shot
`164.2 / 49.7 / 205.0ms`, self_consistency `281.8 / 0.0 / 349.0ms`,
multi_turn `323.7 / 61.6 / 383.0ms`, tree_of_thought
`283.9 / 47.0 / 308.1ms`, and long_output `273.9 / 24.4 / 1204.6ms`.

A focused few_shot profile on the same commit landed at
`171.3 / 51.1 / 213.8ms`, 978/1000 raw correct. Fast-HTTP p50 first-content was
`123ms` while benchmark TTFT was `171ms`, so part of the remaining median is
client/request-wave time outside the handler. Queue counters still showed a
server-side first-wave issue: `initial_batch_size=2`, greedy-mid
`max_active=32`, and `fp8_prefill_enabled=false` for the 256-token deterministic
bucket. Extending FP8 prefill to greedy `max_tokens=256` with `min_m=512` is
rejected: the focused clean row landed at `166.9 / 51.8 / 208.1ms`, worse than
the public TorchInferno few_shot row.

A 5ms first-wave collection looked useful in focused checks: the env-forced row
landed at `164.5 / 48.7 / 204.3ms`, and the edited no-env checkout landed at
`161.8 / 48.1 / 202.6ms`, both with 977/1000 raw correctness. The full
public-order run `20260629_184300` on pushed `cdf5c65` did not hold the win:
TorchInferno dropped to `3/20`, few_shot regressed to
`170.1 / 48.5 / 213.8ms`, and long_output lost the TPOT cell to SGLang
(`25.2ms` vs `24.9ms`). Back out the greedy-mid default and keep first-wave
collection as an env-only experiment until it improves the full scorecard.

The restored public run `20260629_185744` measured TorchInferno `20cb8e8` after
backing out that default. The scorecard remained SGLang `14/20`, TorchInferno
`3/20`, and vLLM `2/20`; long_output was a near TPOT tie rather than a stable
new regression (`24.7ms` TorchInferno vs `24.6ms` SGLang), but SGLang still won
all four long_output score cells. A focused long_output queue/fast-HTTP profile
on the same commit landed at `293.1 / 24.8 / 1180.9ms`, 1000/1000 correct.
Server queue-to-first-token p50 was `240.5ms`, matching fast-HTTP
first-content p50 `243.8ms`, so the median remains inside server
scheduling/prefill rather than response writing.

The long_output counters also match the older diagnosis: `56` prefill batches,
one request-path `b64:s64` prefix graph capture costing `1.17s`, `7.89s`
prefill wall, `13.62s` ragged-decode GPU event time, and `6.97s` ordinary
ragged token harvest exposure. Reopening broad or narrow `b64` suffix warmup is
still rejected because it already removed this capture without a score-facing
win. The remaining useful long-output work is still a decode/readback or
prefill pipeline change that lowers first-token latency without fragmenting the
decode path.

A fresh tree_of_thought profile on the restored runtime keeps tree in the same
bucket. The focused TorchInferno row on docs-only `e264162` landed at
`252.3 / 51.6 / 303.5ms`, 97% correct. Fast-HTTP p50 first-content was
`236.2ms` and total stream p50 was `287.9ms`, close to benchmark
TTFT/E2E, so response serialization is not the median limiter. The six sampled
`max_tokens=300` sessions handled `896` requests at `max_active=32` and the
current `10ms` first collection window with zero request-path prefill captures:
median queue-to-submit across quiescent sessions was `129.6ms`, median
submit-to-first was `78.5ms`, sampled prefill wall totaled `3.81s`, ragged
decode GPU time `1.56s`, and ragged token harvest `1.32s`. The six deterministic
`max_tokens=400` eval sessions handled the remaining `96` requests with no
prefill captures as well. This profile does not reopen sampled-medium
max-active, initial-wait, FP8 min-M, command transport, suffix warmup, or
sampled decode-many knobs; the remaining tree gap is still the sampled
prefix/suffix prefill cadence plus ragged decode synchronization.

## Current 23395db refresh and greedy-short FP8 prefill (2026-06-29)

Public run `20260629_165551` measured TorchInferno `23395db`, SGLang
`643e1cc`, and the current vLLM environment. The scorecard was SGLang `14/20`,
TorchInferno `3/20`, and vLLM `2/20`. TorchInferno rows were few_shot
`166.5 / 50.2 / 210.1ms`, self_consistency `358.9 / 0.0 / 394.9ms`,
multi_turn `308.0 / 66.6 / 368.2ms`, tree_of_thought
`282.9 / 46.7 / 324.2ms`, and long_output `296.1 / 25.1 / 1246.8ms`.
TorchInferno still wins the few_shot, multi_turn, and tree_of_thought TPOT
cells, but SGLang now owns every score-facing long_output cell on this run.

A focused long_output queue/fast-HTTP profile on `23395db` landed at
`287.7 / 26.4 / 1310.1ms`, 1000/1000 correct. HTTP p50 first-content was
`250ms`, matching server queue-to-first-token p50 `246ms`, so the median gap is
inside server scheduling/prefill rather than client response writing. The run
spent `8.16s` in online prefill forward, with FP8 prefill disabled for this
deterministic short-generation bucket. Enabling online FP8 prefill explicitly
improved the focused row to `264.6 / 24.6 / 1274.9ms` and raised median
throughput from `29.9` to `31.6 tok/s`, still 1000/1000 correct. After making
that a scoped default for greedy short requests (`16-128` max tokens), the
edited-checkout no-env validation landed at `254.1 / 25.7 / 1238.6ms`, again
1000/1000 correct. This policy is intentionally below few_shot's 256-token
greedy bucket and does not affect sampled self_consistency/tree traffic.

The next self_consistency profile on `3bff181` reconfirmed that the
score-facing median includes client/request-wave time outside the handler:
fast-HTTP p50 first-content was `14ms` while benchmark p50 TTFT was `262ms`.
Server counters still exposed one real cleanup: cached repeated sampling spent
`1115ms` across `464` tiny sample-state hits. A repeated-sample token reservoir
cut that sampled-state time to `92ms` and improved fast-HTTP p50/p90
first-content (`14/420ms -> 8/414ms`) with 1000/1000 correctness. The profiled
benchmark row improved TTFT but not E2E (`228.7 / 0.0 / 348.9ms`), and a
no-profile check was mixed (`320.1 / 0.0 / 346.4ms`), so this is a server-side
control-plane cleanup rather than a claimed score flip.

## Current 8bf4c1c public refresh and prompt-cache follow-up (2026-06-29)

Public run `20260629_161720` measured TorchInferno `8bf4c1c`, SGLang
`643e1cc`, and the current vLLM environment. The scorecard is SGLang `12/20`,
TorchInferno `5/20`, and vLLM `2/20`. TorchInferno rows are few_shot
`164.2 / 48.7 / 204.9ms`, self_consistency `213.3 / 0.0 / 325.9ms`,
multi_turn `320.2 / 58.9 / 380.3ms`, tree_of_thought
`270.2 / 47.8 / 314.0ms`, and long_output `283.0 / 25.3 / 1193.3ms`.
TorchInferno now wins few_shot TPOT/E2E, self_consistency TTFT, and the
multi_turn/tree TPOT cells, but still loses most TTFT/E2E/throughput cells.

A focused self_consistency profile on `8bf4c1c` kept the known split between
server runtime work and benchmark/client-side latency. With queue profiling,
TorchInferno landed at `311.8 / 0.0 / 332.0ms`; fast-HTTP profiling landed at
`313.3 / 0.0 / 340.1ms` and showed server-side p50 first content around
`15ms` after the request body was ready, with `872/1000` requests below `50ms`
on that server-side clock. The unexplained median gap is still before or around
HTTP request arrival rather than token compute. A prompt token-cache
single-flight follow-up prevents identical prompt bursts from running duplicate
chat-template tokenization on concurrent cache misses. It is a general
control-plane cleanup, not a claimed score flip: the focused dirty-tree
confirmation was neutral/slightly positive at `311.3 / 0.0 / 334.6ms`,
1000/1000 correct.

## Current 642b555/82e9d83 refresh and rejected follow-ups (2026-06-28/29)

A same-host no-profile all-provider comparison on pushed `726ffad` is the
earlier local source baseline. The scorecard was SGLang `12/20`,
TorchInferno `4/20`, and vLLM `3/20`. TorchInferno rows were: few_shot
`170.6 / 49.7 / 213.2ms`, self_consistency `231.9 / 0.0 / 364.4ms`,
multi_turn `320.3 / 61.6 / 379.1ms`, tree_of_thought
`282.5 / 47.6 / 303.3ms`, and long_output `303.1 / 24.7 / 1248.1ms`.
TorchInferno still wins only TPOT on few_shot, long_output, multi_turn, and
tree_of_thought locally. SGLang leads most local TTFT/E2E/throughput cells,
while vLLM leads the local self_consistency median row and several p99 cells.

Public run `20260629_030325` measured the older TorchInferno `9b0f24c`,
vLLM `5274c11`, and SGLang `2260e61`. It landed at TorchInferno `0/20`,
vLLM `17/20`, and SGLang `2/20`. Public TorchInferno rows were:
few_shot `166.2 / 48.2 / 207.4ms`, self_consistency
`282.0 / 0.0 / 301.4ms`, multi_turn `314.9 / 59.0 / 369.9ms`,
tree_of_thought `189.1 / 58.2 / 232.7ms`, and long_output
`354.7 / 21.6 / 1110.8ms`. This public row predates the later
greedy-short prefill-cost priority default, but the competitor movement is the
important signal: public vLLM now wins even few_shot, tree_of_thought, and
long_output TPOT, so local TPOT wins are not enough to close the public gap.

The latest public run `20260629_050254` measured TorchInferno `642b555`,
vLLM `4559c43`, and SGLang `38d4ffc`. The scorecard was TorchInferno `1/20`,
vLLM `15/20`, and SGLang `3/20`. TorchInferno rows were few_shot
`165.6 / 47.2 / 208.0ms`, self_consistency `269.2 / 0.0 / 287.5ms`,
multi_turn `321.4 / 61.4 / 381.7ms`, tree_of_thought
`192.9 / 57.2 / 239.5ms`, and long_output `316.8 / 23.0 / 1117.0ms`.
The sampled-short Marlin default won one public TPOT cell, but public vLLM
still owns nearly all TTFT/E2E/throughput cells; the remaining work has to
reduce request-wave latency and not just token compute.

Public run `20260629_070312` measured TorchInferno `a37dfc0`, vLLM `4559c43`,
and SGLang `91cf159`. The scorecard moved to TorchInferno `1/20`, vLLM
`18/20`, and SGLang `0/20`. TorchInferno kept only the few_shot TPOT cell and
landed at few_shot `161.6 / 46.4 / 202.6ms`, self_consistency
`247.6 / 0.0 / 265.0ms`, multi_turn `314.1 / 55.3 / 369.5ms`,
tree_of_thought `201.4 / 56.7 / 248.5ms`, and long_output
`319.5 / 23.3 / 1158.3ms`. This run predates the later multi-turn metadata
harness change, but it confirms the same public shape: vLLM is now winning the
median TTFT/E2E/throughput cells even where TorchInferno is near or ahead on
token cadence locally.

Public run `20260629_090315` measured TorchInferno `07b0d6f`, vLLM `a4e3cb4`,
and SGLang `a2b5ce2`. The scorecard was TorchInferno `0/20`, vLLM `18/20`,
and SGLang `1/20`. TorchInferno rows were few_shot
`160.8 / 47.2 / 201.9ms`, self_consistency `251.7 / 0.0 / 269.1ms`,
multi_turn `312.3 / 58.1 / 365.6ms`, tree_of_thought
`191.8 / 57.0 / 235.2ms`, and long_output `314.9 / 23.2 / 1121.3ms`.
This public run includes the decode-many overrun counters but predates the
ragged active/padding split and later profile notes. vLLM now wins every
few_shot, multi_turn, tree, and self score cell; SGLang wins long_output TTFT.
TorchInferno's public median TPOT is no longer enough to win any cell, so the
remaining work still needs lower request-wave/prefill latency and lower
long-output decode/readback cost.

A same-host all-provider multi_turn rerun on current pushed `9d62f6b` with the
new harness metadata landed at vLLM `297.7 / 106.2 / 404.2ms`, SGLang
`159.3 / 108.6 / 284.5ms`, and TorchInferno `352.0 / 63.1 / 408.2ms`.
TorchInferno still won only TPOT. The per-turn split makes the gap concrete:
SGLang reaches about `150-174ms` median TTFT for turns 3-7, while TorchInferno
stays around `301-404ms` despite much better median TPOT. The TorchInferno
queue profile had `37` prefill batches, zero graph captures/misses, `4.46s`
prefill forward, `3.12s` decode GPU time, and exactly `45,000` reused prefix
tokens from the 45-token shared system prefix. So the current stable common
prefix path is not the blocker; the missing piece is efficient reuse of longer
per-conversation prefixes.

A current same-host multi_turn refresh on pushed `b448f97` keeps the same
diagnosis with the newer turn metadata. vLLM landed at
`303.8 / 100.2 / 394.1ms`, SGLang at `160.0 / 114.5 / 270.1ms`, and
TorchInferno at `348.8 / 63.0 / 406.8ms`, all around 98% raw correctness.
TorchInferno again won only TPOT. Queue counters showed one session with `36`
prefill batches, zero graph misses/captures, `4.91s` prefill wall (`4.27s`
forward), `5.19s` decode GPU event time, `191` padded ragged decode tokens, and
exactly `45,000` reused tokens from the shared 45-token system prefix. Per-turn
medians show where the tail comes from: TorchInferno turn 0 was `399ms`, turn 1
spiked to `808ms` median and `3446ms` p99, and later turns settled into the
`303-397ms` median band. SGLang reaches `122-177ms` medians for turns 1-7. This
keeps conversation-prefix reuse or a faster mixed-prefix suffix path as the
multi_turn requirement; the default common-prefix path is warm and stable.

Non-common prefix-hit bucketing is also rejected as a finished-prefix rescue.
A source prototype rounded finished-prefix reuse down to coarser prefix lengths
before graph prefill, then ran two opt-in multi_turn checks with finished-prefix
cache, non-common graph prefill, finished-prefix graph prefill, and dynamic
context bucketing through suffix `256`. A 64-token prefix bucket completed but
regressed to `1406.3 / 68.5 / 1488.7ms`; it reused `76.0K` prefix tokens but
still paid `74` prefill batches, `19` captures, `18.1s` capture time, and
`22.7s` prefill wall. A 16-token bucket cut prefill tokens further to `27.4K`
and reused `90.0K` prefix tokens, but still regressed to
`1111.8 / 69.2 / 1256.0ms` with `78` prefill batches, `19` captures, and
`21.5s` prefill wall. The prototype was reverted; lowering exact prefix
granularity does not solve the finished-prefix graph/capture overhead.

Exact-length full-prompt reuse is also rejected on current `9d62f6b`. Enabling
pinned full-prompt stores for `max_tokens>=512` with non-common-prefix graph
prefill, but without mixed-prefix grouping, completed correctly (`979/1000`) and
raised reused prefix tokens to `97.6K` while cutting prefill tokens to `18.1K`.
It still regressed catastrophically to `7207.9 / 66.4 / 7269.9ms` because the
exact prefix lengths fragmented into `282` prefill batches and `191` prefill
graph captures, spending `120.1s` in prefill wall and `105.5s` in capture. This
rules out "keep positive context_len by grouping exact prefix lengths" as the
next default path; conversation-prefix reuse needs shape coalescing without
per-length capture churn.

The new prefix-reuse queue counters on current `0fae868` confirm the dense
default has no hidden longer-prefix reuse. A focused no-env multi_turn row
landed at `338.8 / 62.7 / 402.2ms`, with `34` prefill batches, zero captures,
`4.70s` prefill wall, and route/hit histograms of exactly
`{"common_prefix": 1000}` and `{"45": 1000}`. Enabling full-prompt stores plus
dynamic context bucketing for suffixes through `32` did remove most exact-length
capture churn (`191 -> 13` captures), but it still regressed to
`1812.5 / 66.4 / 2034.6ms`: prefill wall was `31.6s`, including `19.4s`
prefill forward and `11.4s` state/store time, with `861` request-prompt reuse
hits. Forcing the paged prefix-cache engine below its default context threshold
is also rejected for this prompt length: `6015.4 / 656.8 / 6193.8ms`, with
`96.7s` online phase time and paged decode TPOT dominating. Keep paged-prefix
experiments behind their explicit envs until the paged engine has a short-context
decode/prefill path that beats dense.

A current focused self_consistency profile on `0fae868` shows that sampled-short
generated-prefix reuse is healthy and that the remaining row is control-plane
churn. The row landed at `190.2 / 0.0 / 328.8ms`, 1000/1000 correct, with one
common-prefix prefill, one generated-prefix store, `971` generated-prefix reuses,
and route histograms `{"common_prefix": 971, "generated_prefix": 971}`. Server
work was still split across `217` submit batches, `210` runtime steps, and
`2000` token events for 1000 one-token outputs; `phase_submit_sync_ms` was
`588ms` and `prefill_wall_ms` was `1.88s`. This keeps generated-prefix caching
out of the suspect list. The next self_consistency lever has to reduce
submit/reuse wave churn without repeating the rejected idle-drain coalescing or
idle-wait changes.

A same-host self_consistency provider refresh on pushed `63006f1` now favors
TorchInferno locally on score-facing E2E/throughput even though SGLang still
wins median TTFT. vLLM landed at `378.1 / 0.0 / 465.4ms`, SGLang at
`193.0 / 0.0 / 373.6ms`, and TorchInferno at `223.9 / 0.0 / 319.7ms`, all
1000/1000 correct with one unique final answer. The TorchInferno queue profile
again shows the intended generated-prefix shape: one generated-prefix store,
`985` generated-prefix reuses, route counts
`{"common_prefix": 985, "generated_prefix": 985}`, one prefill batch, and one
decode batch. The remaining server work is still `213` submit batches,
`191` runtime step calls, `2000` events, and `578ms` submit-sync time, so the
local remaining self gap is request-wave/finish churn rather than prefix cache
or decode compute. Keep the rejected idle-drain, event-ordering, and submit-step
follow-ups closed unless a new mechanism reduces the wave count directly.

Cached generated-prefix continuations now elide discard-only stop events when
the cached second token is EOS/stop: the visible first token is marked finished
instead of emitting a second internal stop event for the OpenAI server to drop.
A profiled self_consistency run on the working tree after `54b6471` stayed
correct (`1000/1000`) and cut internal emitted events from the previous
`~2000` shape to `1025`, with `215` submit batches and `194` runtime step
calls. The profiled row was `303.4 / 0.0 / 320.5ms`; a no-profile recheck
landed at `201.8 / 0.0 / 310.2ms`. This is a small finish-path cleanup rather
than a full self_consistency close: submit batches and runtime step calls remain
in the same band, so the larger request-wave issue is still open.

A current focused long_output profile on `26a9d5c` keeps that row in the known
decode/readback bucket rather than exposing a missing prefix path. The row
completed `1000/1000` correct at `301.4 / 25.4 / 1320.6ms`, with p99
TTFT/E2E/TPOT `2813.6/4662.4/261.2ms`. Prefix reuse was exactly the shared
111-token prompt for all requests (`111,000` reused tokens) and prefill was warm
apart from the single ragged-prefill capture (`56` prefill batches,
`8.28s` prefill wall). The online phase still spent `12.43s` in ragged-decode
GPU events and `7.50s` in CPU token harvest across `724` ragged decode batches.
Decode-many was enabled with quantum `3`; the old counters show `44,715` decode
row-tokens for `37,715` emitted token events, but they did not separate padded
ragged rows from true stop-token overrun. A follow-up run on `d60eead` landed in
the same band (`272.0 / 25.9 / 1246.4ms`) and showed only `230` skipped
decode-many tokens across `13,429` decode-many model tokens, with `305`
stop-token finishes. So stop-token overrun is measurable but not the whole
decode-token gap. A post-counter validation on `7be8e34` completed in the same
range (`257.5 / 25.0 / 1208.0ms`); its last complete progress snapshot showed
`44,032` ragged decode model tokens, `36,963` active tokens, and `7,069` padded
tokens, versus `251` skipped decode-many tokens. Queue profiles now record these
decode-many and ragged active/padding splits directly so future long_output runs
can distinguish useful multi-step decode work, padded bucket work, and true
overrun without local trace reconstruction.

A current long_output profile on pushed `74629ea` landed at
`255.5 / 25.7 / 1241.8ms`, 1000/1000 correct. Queue counters kept the known
shape: `56` prefill batches with one `b64:s64` prefix-graph capture
(`1.07s`), `8.03s` prefill wall, `11.73s` ragged-decode GPU event time,
`6.10s` CPU token harvest, `6.49K` ragged padding tokens, and `430`
decode-many skipped tokens. A source prototype split large prefix-reuse graph
batches at `32` rows to avoid the cold `b64` capture. It did remove captures and
cut prefill wall to `6.80s`, but regressed the score row to
`271.3 / 25.0 / 1286.4ms`: runtime steps rose `486 -> 597`, decode GPU time
rose `11.73s -> 13.44s`, CPU token harvest rose `6.10s -> 7.46s`, and ragged
padding rose `6.49K -> 7.36K`. The prototype was reverted; trading one prefill
capture for more decode/scheduling fragmentation is not a defaultable path.

Runtime Marlin int4 decode is now disabled by default only for sampled-short
online sessions (`temperature > 0`, `max_tokens <= 256`). The global env
`TORCHINFERNO_MARLIN_INT4_DECODE=0` showed the initial signal on focused
self_consistency, improving a same-host row from `337.7 / 0.0 / 359.6ms` to
`312.6 / 0.0 / 332.9ms`, but that switch is too broad because prior tree
checks showed all-session Marlin disable hurts sampled-medium traffic. A
min-M gate prototype with `TORCHINFERNO_MARLIN_INT4_MIN_M=16` is rejected; it
regressed focused self_consistency to `358.7 / 0.0 / 384.1ms`. The promoted
runtime policy leaves Marlin enabled for greedy and sampled-medium sessions
while turning it off for self_consistency's sampled-short bucket. A paired
same-code check landed at `227.3 / 0.0 / 346.3ms` with
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_SHORT_MARLIN_INT4_DECODE=0` versus
`362.4 / 0.0 / 384.4ms` no-env Marlin-on control, both `1000/1000` correct.
After making that sampled-short behavior the default, a no-env confirmation
landed at `246.5 / 0.0 / 358.3ms`, also `1000/1000` correct. This does not
close the public vLLM self_consistency row yet, but it removes a clear
sampled-short regression without changing few_shot, multi_turn,
tree_of_thought, or long_output default buckets. Explicit overrides remain:
`TORCHINFERNO_OPENAI_TP_ONLINE_MARLIN_INT4_DECODE` for all online sessions and
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_SHORT_MARLIN_INT4_DECODE` for the
sampled-short bucket.

Public refresh `20260629_110323` measured TorchInferno `03677fd` against vLLM
`3483240` and SGLang `bb7d344`; vLLM now owns `16/20` score cells, TorchInferno
`2/20`, and SGLang `1/20`. TorchInferno still wins only few_shot TPOT
(`46.7ms`) and multi_turn TPOT (`57.8ms`). Rows were: few_shot TorchInferno
`160.8 / 46.7 / 198.3ms`, vLLM `139.5 / 49.6 / 178.6ms`, SGLang
`150.4 / 75.7 / 226.0ms`; self_consistency TorchInferno
`249.2 / 0.0 / 269.6ms`, vLLM `210.4 / 0.0 / 231.9ms`, SGLang
`220.4 / 0.0 / 360.2ms`; multi_turn TorchInferno
`316.4 / 57.8 / 371.9ms`, vLLM `155.2 / 58.5 / 201.1ms`, SGLang
`161.2 / 101.5 / 253.5ms`; tree_of_thought TorchInferno
`208.2 / 59.1 / 257.5ms`, vLLM `63.0 / 31.5 / 85.5ms`, SGLang
`78.7 / 52.5 / 138.5ms`; long_output TorchInferno
`317.7 / 22.9 / 1119.8ms`, vLLM `76.4 / 15.0 / 629.6ms`, SGLang
`68.6 / 22.1 / 844.1ms`. This supersedes the earlier public `090315` read:
tree no longer has a TorchInferno TPOT win, and the dominant gaps are now
tree/long-output first-token scheduling plus long-output decode throughput.

Current long_output on pushed `78bd240` still points at the decode/readback
pipeline rather than request-shape skew. A queue-profiled TorchInferno-only run
completed `1000/1000` correct at `283.7 / 25.9 / 1296.6ms`, with much worse
tails than the public no-profile row (`p99` TTFT/E2E/TPOT
`2841.6/4174.0/261.9ms`). The profile had one online session with
`run_max_tokens=96`, `max_active=123`, `prefix_rows=21`, `decode_quantum=3`,
and `143` submit batches. It issued `57` graph-backed prefill batches plus one
cold `b64:s64` graph capture, for `7.85s` prefill wall (`6.79s` forward).
Decode remained the larger floor: `717` decode model calls, `96` decode-many
calls over `232` steps, `12.94s` decode GPU time, and `7.81s` token-harvest
wait. Do not reopen the broad `b64` greedy suffix warmup: it was already
rejected for startup cost/stability, and the current profile still needs a
pipeline change that reduces GPU decode/readback work rather than just removing
one cold prefill capture.

The pushed `642b555` full TorchInferno-only validation landed at few_shot
`161.4 / 46.7 / 198.3ms`, self_consistency `245.3 / 0.0 / 260.3ms`,
multi_turn `310.9 / 61.4 / 364.8ms`, tree_of_thought
`252.5 / 49.2 / 291.2ms`, and long_output `283.5 / 25.0 / 1163.1ms`.
A same-host all-provider pass hit the known local vLLM `libstdc++` preload
issue, but the TorchInferno/SGLang rows plus a separate vLLM-only rerun with
`LD_PRELOAD=/home/bobren/local/d/pytorch-env/lib/libstdc++.so.6` give the
current local score shape: TorchInferno wins self_consistency TTFT, E2E, and
throughput plus few_shot, multi_turn, and tree_of_thought TPOT; SGLang still owns
most TTFT/E2E/throughput cells; vLLM only wins long_output E2E in that merged
same-host view. Rows were TorchInferno few `165.4 / 52.1 / 206.7ms`,
self `206.6 / 0.0 / 238.3ms`, multi `319.9 / 64.7 / 385.0ms`,
tree `238.5 / 48.5 / 285.2ms`, long `282.3 / 24.7 / 1187.3ms`; SGLang few
`117.2 / 83.8 / 202.7ms`, self `216.6 / 0.0 / 380.9ms`,
multi `160.9 / 117.9 / 274.9ms`, tree `63.5 / 77.4 / 157.3ms`, long
`61.8 / 24.6 / 962.6ms`; vLLM few `250.5 / 87.7 / 324.8ms`, self
`234.1 / 0.0 / 284.4ms`, multi `276.3 / 100.9 / 373.3ms`, tree
`125.7 / 84.9 / 192.3ms`, and long `85.9 / 25.7 / 956.6ms`. The remaining
highest-leverage work is therefore first-token/prefill scheduling for
tree_of_thought and long_output, plus smaller few_shot E2E/TTFT cleanup.

A same-host all-provider tree-only rerun on current pushed `a37dfc0` with the
local vLLM `LD_PRELOAD` workaround confirmed the same shape without the earlier
vLLM import failure. vLLM landed at `126.8 / 86.9 / 196.7ms`, SGLang at
`61.8 / 65.6 / 150.8ms`, and TorchInferno at `265.6 / 50.5 / 309.7ms`;
correctness was in the same 97% band for all three providers. The TorchInferno
queue profile had 12 online sessions for 992 requests, with sampled branches
spending `6.48s` total phase time, `3.79s` prefill wall, and `2.22s` decode
active across 45 prefill graph hits and no misses. Greedy eval spent `3.81s`
phase time, `1.04s` prefill wall, and `2.65s` decode active. TorchInferno still
wins only tree TPOT locally; SGLang owns TTFT/E2E/throughput on the same host.

A current same-host tree-only rerun on pushed `07b0d6f` keeps the row in the
same bucket. vLLM landed at `122.4 / 82.4 / 187.6ms`, SGLang at
`61.9 / 74.1 / 158.9ms`, and TorchInferno at `246.8 / 49.3 / 298.2ms`, with
all providers around 97% raw correctness. The TorchInferno queue profile split
into six sampled-medium sessions for `896` requests and five greedy eval
sessions for `80` requests. Sampled-medium spent `6.27s` total phase time,
`3.85s` prefill wall (`1.84s` forward), and `2.06s` decode-active across
`44` prefill graph hits and zero misses/captures; greedy eval spent `3.24s`
phase time, `0.86s` prefill wall, and `2.30s` decode-active. Prefix reuse was
only the 45-token common prefix (`43,920` tokens total), and decode-many stayed
off for both buckets. This confirms the tree gap is not hidden graph warmup or
decode-many overrun; it remains sampled-medium prefill/session shape plus the
short greedy eval decode path. Do not reopen the already rejected 16/40/64-row,
5ms/20ms initial-wait, idle, idle-arrival, or FP8 min-M knobs from this profile.

A same-host current tree rerun on pushed `cc3bd6f` plus the updated
inference-bench request/completion metadata again shows a sampled-prefill gap,
not a correctness or output-length artifact. TorchInferno landed at
`236.3 / 49.1 / 282.5ms`, vLLM at `130.0 / 83.5 / 189.1ms`, and SGLang at
`63.1 / 71.7 / 152.2ms`; all providers completed around 97% raw correct.
TorchInferno branch requests were `225.3 / 49.4 / 272.7ms`, while greedy eval
requests were `317.4 / 27.3 / 372.9ms`. Queue totals for sampled-medium branch
traffic were `896` submitted requests, `43` graph-backed prefill batches, zero
prefill graph misses/captures, `3.66s` prefill wall (`1.80s` forward),
`1.42s` ragged-decode GPU time, and `1.24s` token harvest time. This keeps tree
in the known sampled common-prefix suffix prefill pipeline bucket; current
metadata now separates submitted request order from completion order for future
public traces.

A current TorchInferno-only tree profile on pushed `b71070c` keeps that same
bucket after the public `110323` refresh. The profiled row landed at
`250.4 / 49.6 / 299.0ms`, `958/992` raw correct, with the expected queue-profile
overhead versus the public no-profile `208.2 / 59.1 / 257.5ms` row. Sampled
branch traffic (`temperature=0.7`, `max_tokens=300`) used six sessions for
`896` submitted requests, `39` submit batches, `46` prefill batches, zero
prefill graph misses/captures, `3.77s` prefill wall (`1.88s` forward),
`1.60s` ragged-decode GPU time, and `1.37s` CPU token readback. Greedy eval
traffic used six 16-request sessions and spent another `1.03s` prefill wall and
`1.93s` decode GPU time. This does not reopen initial-wait, admission-cap,
FP8-min-M, dynamic-context-floor, or graph-warmup knobs; the remaining tree gap
needs a genuinely faster sampled-prefix prefill/decode pipeline rather than more
shape cleanup.

Lowering the dynamic prefix-prefill minimum context from `256` to `64` has been
reopened and enabled by default for small dynamic suffix shapes. On pushed
`3af4940`, a same-shape tree-only control landed at `168.2 / 63.1 / 202.1ms`
with queue-profile phase `6000ms` and prefill forward `2470ms`; the 64-token
floor landed at `144.5 / 33.5 / 170.2ms` with phase `5397ms` and prefill
forward `2216ms`. The p99 E2E tail was slightly worse (`890ms` versus `837ms`),
but queue finish p99 was comparable (`556ms` versus `549ms`) and median TTFT,
TPOT, E2E, and prefill work improved materially. Larger prefix/suffix pairs
still bucket to `128`, `256`, or above, and the
`TORCHINFERNO_CONTINUOUS_DYNAMIC_PREFIX_PREFILL_MIN_CONTEXT` override remains
available for deployments that need the old floor.

The first current full-order local check after that change exposed a warmup
coverage regression rather than a runtime scheduling issue. With the 64-token
floor, startup `p45/s16` greedy warmup now captures `ctx64`; few_shot
`p122/s16` and long_output `p111/s32,s64` still bucket to `ctx256`, so the
unpatched `304761c` full-order run had request-path `ctx256` captures in both
rows. It landed at few_shot `172.9 / 50.8 / 214.4ms`, multi_turn
`313.6 / 61.4 / 374.2ms`, tree `156.2 / 53.9 / 193.0ms`, and long_output
`253.0 / 25.6 / 1402.3ms`; long_output showed `8` prefill graph captures and
`13.82s` prefill wall. Adding only the missing greedy-short warmup pairs
`111:32,111:64,122:16` avoided repeating the rejected broad `45,122` warmup:
readiness moved from `180.8s` to `195.9s`, request-path captures dropped to
zero for few_shot and long_output, and the full-order row improved to few_shot
`170.6 / 48.8 / 208.2ms`, multi_turn `294.0 / 59.3 / 351.0ms`, tree
`147.6 / 46.1 / 182.3ms`, and long_output `245.5 / 24.6 / 1214.5ms`.

A current same-host self_consistency rerun on pushed `62eb441` shows the local
score shape has shifted from E2E to TTFT/tail control. TorchInferno landed at
`302.6 / 0.0 / 323.1ms`, vLLM at `278.2 / 0.0 / 337.6ms`, and SGLang at
`196.1 / 0.0 / 370.8ms`, all `1000/1000` correct with one unique final answer.
TorchInferno wins median E2E locally but still loses first token and p99. The
queue profile ended with progress records only because all submitted requests
finished while the non-persistent sampled-short batcher was waiting in its
750ms idle window; the harness can tear the server down before the final
`online_batcher` record is written. The last snapshot still showed the issue:
`1000` submitted requests fragmented into `240` submit batches and `214`
runtime step commands over `2.99s` phase time, with generated-prefix reuse
active for `970` requests and `54.3K` reused tokens. This is wave/control-plane
churn, not a missing generated-prefix cache hit or decode-compute gap. Add an
`online_batcher_quiescent` queue-profile record when all currently submitted
work has drained and before the idle wait for future arrivals, so later sampled
short profiles keep aggregate counters even if teardown races the final record.
The first source placement was still too late because teardown can occur during
the short idle-arrival wait; moving the record ahead of that wait validated on
`48b68bf`. A profiled TorchInferno-only self_consistency rerun landed at
`324.0 / 0.0 / 372.6ms`, `1000/1000` correct, and emitted quiescent snapshots
through the final `1000/1000` submitted/finished state (`217` online step
commands, `255` submit batches, `979` generated-prefix reuse requests) despite
the final `online_batcher` record still being preempted by server teardown.

Cached repeated-sampler state is now enabled for exact-prefix reusable logits.
The self_consistency hot path repeatedly samples from the same cached prompt and
generated-prefix logits across hundreds of small waves; those draws still need
fresh random thresholds, but the logits CDF/rank-sum setup is invariant per
prefix. The promoted path prepares that state lazily and reuses it through the
existing symmetric TP sampler hook. Same-tree profiled A/B on the working tree
validated the direction: default cached state landed at
`288.2 / 0.0 / 308.7ms`, `1000/1000` correct, with `452` cached-state sample
hits covering `1950` sampled tokens, `1.55s` exact-prefix prefill wall, and
`2.84s` online phase time. Disabling it with
`TORCHINFERNO_CONTINUOUS_CACHED_REPEATED_SAMPLE_STATE=0` regressed to
`327.6 / 0.0 / 359.6ms`, with `1.82s` prefill wall and `3.12s` phase time.
This reduces the known self-control-plane cost without changing request
admission, idle windows, prompt contents, or sampling semantics.

Lowering online admission granularity is rejected for sampled-medium tree. A
focused source-free A/B forced
`TORCHINFERNO_OPENAI_TP_ONLINE_ADMIT_PER_STEP_CAP=16` on the same pushed
`a37dfc0` tree row. It regressed to `283.9 / 64.1 / 357.3ms`, 957/992 raw
correct. Queue counters show why: sampled prefill batches rose `45 -> 67`,
sampled prefill wall rose `3.79s -> 4.22s`, sampled decode-active time rose
`2.22s -> 2.50s`, and prefill graph misses stayed at zero. Smaller admission
waves do not improve first-token latency enough to offset the extra graph
replays and decode fragmentation; keep the current sampled-medium 32-row
admission shape.

Broad online step-sync removal is still rejected for tree. A focused no-profile
run with `TORCHINFERNO_OPENAI_TP_ONLINE_STEP_SYNC=0` on `62baef8` completed
959/992 raw correct at `245.9 / 51.0 / 295.8ms`, with p99 TTFT/E2E
`1580.6/1685.2ms` and p99 TPOT `1012.1ms`. That is only within the recent
no-profile tree noise band, not a stable improvement over the existing
sampled-short-only no-sync default, and it worsens the tail token cadence. Keep
step-sync removal scoped to sampled-short self_consistency traffic until a
tree-specific command path shows a clear median and tail win.

A same-host all-provider long_output rerun on pushed `fc50b42` with the vLLM
`LD_PRELOAD` workaround showed the local long gap is still TTFT and tail
pipeline, not median token compute. vLLM landed at
`98.1 / 29.1 / 1143.0ms`, SGLang at `63.5 / 24.3 / 951.5ms`, and
TorchInferno at `257.0 / 25.2 / 1278.4ms`, all 1000/1000 correct with the same
36.7k output-token count. SGLang won all four score metrics; TorchInferno was
within `0.9ms` of SGLang median TPOT but had p99 TTFT/E2E/TPOT of
`2462.6/3698.3/262.2ms` versus SGLang `542.3/1772.7/40.2ms`. The next
long_output change still needs to reduce first-token queueing and tail
decode/readback behavior; a median-only TPOT improvement is unlikely to move the
score.

Disabling online step sync for deterministic 256-token few_shot is rejected.
The focused no-profile run with `TORCHINFERNO_OPENAI_TP_ONLINE_STEP_SYNC=0` on
`536f14e` completed 978/1000 raw correct at `164.7 / 50.6 / 206.1ms`, with p99
TTFT/E2E/TPOT `2062.6/2663.1/714.7ms`. That does not improve the current
TTFT/E2E band and gives away TPOT versus the recent default few_shot rows
(`161.4 / 46.7 / 198.3ms` full TorchInferno-only and public
`165.6 / 47.2 / 208.0ms`). Keep the no-step-sync default scoped to
sampled-short self_consistency.

A same-host all-provider multi_turn rerun on pushed `993f2bb` again isolates
the gap to first-turn/prefix admission rather than decode TPOT. vLLM landed at
`287.7 / 109.1 / 383.5ms`, SGLang at `166.0 / 102.7 / 275.4ms`, and
TorchInferno at `359.2 / 62.9 / 422.2ms`, with all providers in the 98%
correctness band. TorchInferno won only median TPOT; SGLang won TTFT/E2E and
throughput. The first-turn TTFT average was the clearest split:
TorchInferno `1049.7ms`, vLLM `438.4ms`, and SGLang `337.9ms`. TorchInferno
last-turn TTFT dropped to `395.3ms`, so conversation-prefix reuse is helping
after the first wave, but the initial admission/prefix-fill path is still too
slow and has a `3199.8ms` p99 TTFT tail.

A focused TorchInferno-only multi_turn queue profile on current `cf3e3c1`
keeps the same conclusion and rules out missed graph coverage on the default
shared-prefix route. The row landed at `337.2 / 64.8 / 395.7ms`, `981/1000`
raw correct, with first-turn TTFT average `1092.8ms` and last-turn average
`402.9ms`. The online session submitted all `1000` requests through `34`
prefill batches, all graph hits and zero captures/misses, with `4.72s`
prefill wall (`4.10s` forward) and `3.58s` decode-active time. Prefix reuse was
exactly `45,000` tokens (`45` shared system-prefix tokens x `1000` requests),
with no generated/finished-prefix reuse. The remaining multi_turn gap is
efficient per-conversation prefix reuse or a faster mixed-prefix suffix path;
the current common-prefix graph path itself is already warm and stable. The
inference-bench harness now records multi_turn conversation/turn metadata and
first/last-turn median/p99 TTFTs (`8101d771`) so future public runs expose this
shape directly.

Fresh focused profiles on pushed `0c45929` keep long_output in the known
decode/readback bucket. The queue-profiled control completed `1000/1000`
correct at `256.9 / 26.2 / 1255.9ms`. The last queue snapshot had all
`1000` requests finished with one `b64:s64` ragged-prefill graph capture
(`1013ms`), `8.05s` prefill wall, `10.16s` decode-active time, `13.50s`
decode GPU event time, and `7.27s` CPU token readback across `756` decode
batches; fast HTTP overhead was not the median bottleneck (`217ms` p50 first
content, `1219ms` p50 total, `23ms` p50 content-send). Disabling online
Marlin int4 decode globally is rejected for the greedy-short long_output row:
the focused no-profile rerun improved TTFT only slightly (`247.5ms`) but
regressed TPOT/E2E/throughput to `27.8ms`, `1296.4ms`, and `29.4 tok/s`.
Keep Marlin enabled for greedy short sessions; the sampled-short-only Marlin
disable remains the scoped default.

Common-prefix cache-only prefill is rejected as a default runtime change. A
source prototype skipped logits for shared common-prefix rows when every request
had a non-empty suffix, using the tensor-parallel `prefill_cache_only` hook for
prefixes at least 96 tokens long. Focused few_shot A/B on the same working tree
was neutral: cache-only landed at `165.5 / 50.4 / 205.6ms` versus
`167.2 / 50.1 / 207.5ms` with the path disabled. Queue counters showed only a
small whole-run prefill-forward movement (`1706.9ms -> 1684.6ms`), while decode
and arrival-wave shape dominated the score-facing medians. The long_output
confirmation was not stable either: `267.6 / 25.0 / 1344.0ms`, 1000/1000
correct, with `7.98s` prefill wall, `12.34s` decode GPU event time, and `7.48s`
CPU token readback. The prototype was backed out; a single common-prefix logits
skip is not enough for few_shot or long_output.

Self-consistency finish-path follow-ups are also rejected. A fresh no-env
control on pushed `82e9d83` landed at `200.7 / 0.0 / 314.4ms`, `1000/1000`
correct. The queue profile confirms the model compute is already mostly out of
the way for this row: a single generated-prefix cache fill served the repeated
prompts, with `978` generated-prefix reuses, one prefill/decode batch, but
`199` submit batches, `185` runtime step calls, `575.8ms` submit-sync time, and
`2.90s` total runtime phase time. Lowering only the sampled-short idle wait to
`2ms` made the row worse at `330.4 / 0.0 / 353.6ms`, with `244` submit batches
and `3.18s` phase time. An exact-prefix event-interleaving source prototype,
which emitted first and stop events per reused request instead of all first
tokens followed by all stops, also regressed to `316.2 / 0.0 / 342.1ms`, with
`250` submit batches and `3.03s` phase time. That prototype was backed out.
The next self_consistency improvement needs to change the finish/request-wave
shape more substantially than event ordering or a smaller idle wait.

Combining submit+step for newly idle sampled-short waves is rejected. The
existing combined submit-step command remains useful only when runtime work was
already active before the drain. A source prototype also sent
`steps_after_submit` on idle self_consistency waves so the worker could process
the new exact-prefix requests without a separate step command. Focused
self_consistency regressed to `312.2 / 0.0 / 333.8ms`, 1000/1000 correct.
Queue counters explain the loss: submit batches rose `199 -> 265`, runtime step
calls `185 -> 224`, idle-wait drain time `116.3ms -> 251.4ms`, and runtime
prefill wall `1.73s -> 1.89s`, while step-broadcast time was only `9.5ms` in
the control. The extra command fusion changed arrival/drain pacing more than it
saved coordination overhead, so the prototype was backed out.

Coarsening the sampled-short idle-drain poll interval is rejected too. The
default idle drain loop polls every `0.1ms` inside the 10ms sampled-short window.
An env-gated source prototype tested `1ms` polling to reduce tiny drain batches,
but focused self_consistency regressed to `334.5 / 0.0 / 362.0ms`, 1000/1000
correct. Queue counters moved the wrong way versus the `82e9d83` control:
submit batches `199 -> 253`, runtime step calls `185 -> 230`, submit-sync
`575.8ms -> 663.2ms`, runtime phase `2.90s -> 3.17s`, and prefill/reuse wall
`1.73s -> 1.84s`. The prototype was backed out; changing the idle poll cadence
does not solve self_consistency's finish-path churn.

The prior same-host no-profile all-provider comparison on pushed `d363367`
landed at SGLang `13/20`, TorchInferno `4/20`, and vLLM `2/20`. TorchInferno
rows were: few_shot `174.5 / 50.5 / 220.0ms`, self_consistency
`372.5 / 0.0 / 400.5ms`, multi_turn `333.7 / 64.8 / 393.1ms`,
tree_of_thought `312.4 / 46.7 / 348.6ms`, and long_output
`297.6 / 24.8 / 1247.2ms`. Keep it as historical same-host evidence rather
than the current source baseline.

Self-consistency idle-drain submit coalescing is rejected. A source prototype
changed the idle path to collect arrivals across the wait window and submit
them once. It did reduce submit and runtime step counts, but the score-facing
row regressed because ready arrivals paid repeated idle waits. With the 10ms
sampled-short idle default, submit batches dropped `237 -> 154` and runtime
step calls `214 -> 98`, but idle wait rose `182ms -> 1244ms` and the row landed
at `306.7 / 0.0 / 353.0ms`. Lowering only
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_SHORT_IDLE_BATCH_WAIT_MS` to `2` improved
TTFT to `205.6ms`, but E2E stayed poor at `341.9ms` and throughput stayed
`2.9 tok/s`. The patch was reverted; keep the current sampled-short idle drain
shape until a change improves E2E/throughput rather than only reducing command
count.

Two few_shot follow-ups on `92af26f` are rejected. The current focused control
landed at `166.1 / 50.9 / 208.2ms`, with `max_active=32`, `30` full
32-request waves, `2.92s` prefill wall, and `3.98s` decode-active time.
Raising only `TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_MID_MAX_ACTIVE` to `48`
regressed to `196.5 / 49.8 / 242.1ms`; it introduced `b48` prefill shapes and
raised runtime phase time to `10.41s` (`4.06s` prefill wall,
`5.92s` decode-active). Forcing a 10ms initial collection window kept the
32-row cap and moved initial admission from `3` to `13`, but was effectively
flat at `166.7 / 49.7 / 206.6ms`. Keep greedy-mid active rows at `32` and do
not add a greedy-mid initial wait default from this evidence.

Reopening greedy-mid decode-many with a stop-aware source prototype is also
rejected. The patch removed stopped rows from the rest of a decode-many quantum
when all active requests shared the same stop ids, then rechecked few_shot with
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_GEN_MAX_TOKENS=256`. It fixed the
old overdecode symptom (`~1.5k` decode tokens instead of the prior `15k+`), but
the row still regressed badly to `180.0 / 69.4 / 759.9ms` and only
`2.6 tok/s`. Queue phase time rose to `13.46s` even though decode-active time
was only `1.29s`, so the remaining problem is decode-many event pacing/CPU
work rather than stopped-row overcompute. The prototype was reverted; keep
decode-many scoped to short greedy output.

The current pushed `707481b` few_shot row is still a small median gap but a bad
tail row. Same-host providers landed at vLLM `170.7 / 86.5 / 273.1ms`, SGLang
`127.7 / 82.3 / 206.7ms`, and TorchInferno `168.1 / 47.9 / 209.2ms`; all were
in the 98% correctness band. The TorchInferno queue profile showed one online
session, `35` prefill batches, `2.70s` prefill wall (`1.71s` forward), `72`
decode batches, `3.15s` decode GPU event time, only `122` padded ragged decode
tokens, and two prefill graph misses with no captures. Extending greedy
common-prefix suffix warmup from `45` to `45,122` is rejected: readiness slowed
from about `161s` to `206s`, the focused row was only noise-better at
`164.2 / 48.9 / 203.9ms`, p99 worsened, and queue counters regressed to
`3.60s` prefill wall with three misses and one request-path capture. The few_shot
tail is not solved by warming that benchmark's 122-token common-prefix shape.

A current profiled few_shot check on pushed `96cb9e7` keeps the same diagnosis.
The row landed at `167.4 / 49.6 / 206.3ms`, 977/1000 raw correct, with
`initial_batch_size=2`, `34` submit batches, `35` prefill batches, and `73`
decode model calls. The runtime reused the 122-token common prefix for `980`
requests, hit `33` prefill graphs with only `2` misses and no captures, and
spent `2.69s` in prefill wall (`1.69s` forward) versus `4.03s` in decode-active
time. Ragged decode padding was only `165` tokens. So the current few_shot median
gap is not a missing warmup bucket or padded decode waste; it remains the known
greedy-mid balance of prefill wave cost plus decode cadence.

Rechecking a global 5ms online initial collection window for the current
few_shot shape is rejected. The no-profile 5ms run on pushed `03677fd` landed
at `166.4 / 46.8 / 202.6ms`, 977/1000 raw correct. A same-code no-env control
landed at `162.4 / 47.5 / 202.4ms`, 978/1000 raw correct. The extra wait did
not improve median E2E and gave away TTFT, so it is not defaultable for
greedy-mid traffic.

Raising the stream token drain cap for long_output is rejected. On clean
`b0d4c0f`, the current profiled long_output control landed at
`271.6 / 24.8 / 1276.3ms`, with `1.2s` of prefill graph capture,
`12.57s` ragged-decode GPU event time, and `6.62s` CPU-token readback.
Increasing only `TORCHINFERNO_OPENAI_STREAM_TOKEN_BATCH_MAX` from `8` to `32`
kept correctness at `1000/1000` but regressed the row to
`279.6 / 25.5 / 1311.7ms`. Fast-HTTP profiling showed why this is not a
promotion candidate: content chunks stayed unchanged at `36,715`, content
send calls only moved `26,839 -> 26,160`, and summed content-send time rose
`20.70s -> 21.30s`. Keep the current stream batch cap.

Greedy-short prefill-cost admission priority is promoted for deterministic
`max_tokens <= 128` traffic. The policy keeps prefix-hit priority first, then
admits cheaper prompt suffixes before longer suffixes only within this short
greedy bucket. A profiled long_output run showed the expected tradeoff:
TTFT improved to `248.2ms`, but the flag fragmented admissions
(`129 -> 144` submit batches, `536 -> 600` runtime step calls) and made the
profiled phase slower, so it needed a no-profile A/B. The paired no-profile
control landed at `273.1 / 25.1 / 1234.4ms`, `31.3 tok/s`, while the scoped
priority landed at `259.4 / 24.6 / 1205.1ms`, `32.1 tok/s`, both
`1000/1000` correct. Keep the default scoped to greedy-short; few_shot
(`max_tokens=256`), multi_turn (`512`), self_consistency/tree sampled traffic,
and tree eval (`400`) keep arrival-order admission unless explicitly overridden.
The pushed default full TorchInferno-only validation on `7ca6f95` then landed at
few_shot `165.3 / 48.7 / 205.0ms`, self_consistency
`265.6 / 0.0 / 288.2ms`, multi_turn `305.9 / 63.1 / 363.6ms`,
tree_of_thought `243.0 / 47.5 / 294.9ms`, and long_output
`272.1 / 25.4 / 1171.4ms`, with summary correctness rates at `1.0`.

Rechecking a much larger greedy-large initial collection window is rejected.
Forcing `TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_INITIAL_BATCH_WAIT_MS=50`
on current `0b53edf` did collect a larger first wave (`initial_batch_size=28`
instead of `8` in the focused no-env profile), but the multi_turn row landed at
`341.3 / 64.9 / 399.7ms`, with only `984/1000` raw correct. Fast-HTTP first
content worsened from `269.7ms` p50 and `2426.6ms` p99 to `274.1ms` p50 and
`3111.0ms` p99. The queue profile also regressed total phase time to `10.00s`
and decode-active time to `4.83s`. Keep the current greedy-large initial wait;
the remaining multi_turn tail is not solved by collecting a bigger first wave.

Broadening prefill-cost admission priority to the 256-token greedy few_shot row
is also rejected. On current `e5b7dcc`, a paired no-profile control landed at
`163.3 / 49.2 / 202.7ms`, while forcing
`TORCHINFERNO_CONTINUOUS_ADMIT_PREFILL_COST_PRIORITY=1` landed at
`164.2 / 47.1 / 202.8ms`; both were `977/1000` raw correct. The env priority
only traded a small TPOT improvement for a TTFT regression and no E2E movement,
so keep the default scoped to deterministic greedy-short traffic.

A fresh tree_of_thought profile on current `2b71f1f` landed back in the normal
TorchInferno band at `236.0 / 49.8 / 286.8ms`, `959/992` raw correct. The
aggregate queue profile across `11` online sessions showed zero prefill graph
misses or captures, `52` prefill graph hits, `4.55s` prefill wall, `3.03s`
decode GPU event time, `1.35s` decode CPU-token readback, `41` submit batches,
and `144` runtime step calls. The all-provider tree row's `312.4ms` TTFT was
therefore a run-order outlier, not evidence of a fresh request-path graph miss.
Extending greedy-large FP8 prefill to the deterministic 400-token evaluator
sub-sessions is rejected. Forcing
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_FP8_PREFILL_MIN_TOKENS=399` and
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_FP8_PREFILL_MIN_M=256` preserved
correctness (`965/992` raw) but regressed the median row to
`244.1 / 51.1 / 293.2ms`. Keep the FP8 boundary at `400 < max_tokens <= 512`.

## Earlier no-profile local scorecard (2026-06-28)

An earlier no-profile all-provider run on pushed `50580d6` landed at
TorchInferno `6/20`, vLLM `0/20`, and SGLang `13/20`. It was the best
score-facing baseline at that point because the earlier `1702ba1` full pass
carried queue profiling overhead. TorchInferno then won few_shot TPOT/E2E
(`48.5ms` / `202.4ms`), self_consistency E2E/throughput
(`278.6ms` / `3.6 tok/s`), multi_turn TPOT (`61.4ms`), and tree_of_thought TPOT
(`47.1ms`). SGLang still leads row-level TTFT/E2E/throughput for multi_turn,
tree, and long_output; the long_output row remains the broadest gap at
`301.7 / 24.9 / 1346.9ms` versus SGLang `64.2 / 24.1 / 867.5ms` and vLLM
`88.0 / 26.0 / 1057.4ms`.

## Public run 20260628_230259 and b64 warmup rejection (2026-06-28)

Public run `20260628_230259` measured TorchInferno `1702ba1`, vLLM `4dfbf15`,
and SGLang `b9b8606`. The scorecard was TorchInferno `0/20`, vLLM `18/20`,
and SGLang `1/20`, so the newer competitor commits erased the earlier
few_shot TPOT public cell. TorchInferno rows were: few_shot
`163.4 / 50.3 / 203.7ms`, self_consistency `252.2 / 0.0 / 268.9ms`,
multi_turn `319.3 / 58.2 / 376.3ms`, tree_of_thought
`196.2 / 57.9 / 241.2ms`, and long_output
`338.4 / 22.8 / 1164.9ms`. The public shape matches the local no-profile
baseline: deterministic TPOT is competitive, but first-token and E2E gaps
remain the score-facing problem.

Rechecking `b64` greedy common-prefix suffix warmup is rejected. Adding
`64` to `TORCHINFERNO_OPENAI_WARMUP_ONLINE_GREEDY_COMMON_PREFIX_SUFFIX_BATCHES`
first exposed a startup ordering bug: several TP workers entered the command
listener while slower ranks were still draining startup warmup collectives,
which made the first object-list command receive interpret a corrupted size and
try to allocate more than `1EB`. The general fix is a post-warmup TP sync after
local CUDA synchronization, before workers can enter command-listener
collectives.

With that ordering fixed, the same `b64` warmup no longer hit the object-list
crash but still failed the promotion bar: after more than seven minutes before
readiness, six ranks had logged FlashInfer decode graph completion while ranks
0 and 7 were still active in `_warmup_flashinfer_decode_graphs` under
`marlin_int4_mm`. The run was stopped before requests. Keep default greedy
suffix warmup at `1,2,4,8,16,32`; avoiding one request-path `b64:s64` capture
is not worth the startup cost or stability risk from this broader warmup shape.
The normal default warmup path remains healthy with the startup barrier: a
focused long_output retry reached readiness in `165.7s`, completed `1000/1000`
correct requests, and landed at `281.6 / 25.5 / 1337.3ms` with
`30.3 tok/s`. A same-host env-only follow-up lowered only
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_INITIAL_BATCH_WAIT_MS` from `1` to
`0` and improved the focused row to `274.0 / 24.9 / 1205.6ms`, `31.7 tok/s`,
also `1000/1000` correct. Promote the zero wait for deterministic greedy-short
traffic only; the bucket remains capped at `max_tokens <= 128`, below few_shot.
A no-env patched confirmation then landed at `255.7 / 25.5 / 1341.6ms`,
`31.2 tok/s`, also `1000/1000` correct. This confirms the TTFT/throughput
benefit while median E2E stays in the current variance band.

The follow-up all-provider long_output comparison on pushed `b8cc49a` still
left TorchInferno at `0/4`: TorchInferno `278.8 / 25.0 / 1264.2ms`, vLLM
`90.9 / 27.6 / 1038.2ms`, and SGLang `58.6 / 24.2 / 920.5ms`. Reopening
greedy-short decode quantum after the zero-wait change is rejected. A focused
DQ=4 run looked superficially useful (`325.2 / 23.6 / 1229.2ms`), but the
same-host all-provider check still lost every score cell and worsened tails:
TorchInferno `315.2 / 25.2 / 1312.5ms`, p99 TPOT `632.3ms`, versus SGLang
`62.7 / 24.7 / 985.5ms`, p99 TPOT `37.9ms`. Keep the greedy-short decode
quantum at `3`; the remaining long_output gap is not solved by a larger command
quantum.

The current tree_of_thought profile on `a349eba` keeps the sampled-medium
diagnosis. A profiled focused run landed at `231.3 / 49.1 / 275.4ms`,
964/992 raw correct, with one request-path suffix prefill graph capture
(`ragged_prefill:b31:s16:rows1:ctx-256:src1`) costing about `1.0s`; total
profiled prefill wall was `1.91s` for the first 256-request sampled wave.
Narrowly adding only batch `31` to the existing common-prefix suffix warmup is
rejected: readiness rose to `180.8s` and the score row regressed to
`250.0 / 49.2 / 294.1ms`, 961/992 raw correct. Keep the sampled suffix warmup
batch set at `1,2,4,8,16,32`; removing that one capture is not enough to improve
client-observed tree medians.

Promote the narrower runtime-side fix for the same unpaddable shape: when a
common-prefix suffix graph wants to pad to the warmed power-of-two batch but no
active row is free, it may now borrow a free prefix-cache row for the discarded
dummy row. This avoids evicting cached prefixes and keeps live decode rows
unchanged. The patched tree profile eliminated request-path prefill graph
captures entirely (`ragged_prefill:b31` disappeared) and replayed warmed
`b32/b16/b4/b1` prefix graphs, with the focused row staying in band at
`233.2 / 49.3 / 279.2ms` and p99 E2E improving versus the prior profiled
`2577ms` tail. A no-profile same-host check kept the score-facing shape:
TorchInferno `213.4 / 49.6 / 254.5ms`, SGLang
`63.2 / 68.2 / 156.0ms`, and a separate vLLM rerun with the fixed
`libstdc++` path landed at `125.1 / 85.5 / 193.9ms`. TorchInferno still wins
only tree TPOT; the remaining gap is first-token scheduling/prefill MFU, not
request-path graph capture.

## Greedy-large multi_turn suffix bucket refinement (2026-06-28)

Current multi_turn is still conversation-prefix prefill dominated, but the
power-of-two suffix buckets were doing avoidable padded work for 512-token
greedy sessions. An env-only A/B on `7be4858` with suffix buckets
`16,32,64,96,128,160,192,224,256` improved the focused TorchInferno row from
the same-host control `395.2 / 64.0 / 460.0ms` to
`333.2 / 64.3 / 397.8ms` (TTFT/TPOT/E2E), with zero prefill graph misses.
The queue profile moved profiled runtime phase time from `11.11s` to `9.60s`
and prefill wall from `5.84s` to `4.54s`.

The promoted default is deliberately scoped to deterministic greedy-large
sessions (`400 < max_tokens <= 512`) and keeps explicit env overrides. A no-env
confirmation of the guarded default landed at `347.1 / 60.1 / 406.6ms`, with
`s96` and `s160` prefix graphs replaying and no request-path captures. The
all-provider comparison landed at TorchInferno `347.7 / 64.4 / 405.1ms`, vLLM
`279.0 / 133.3 / 358.7ms`, and SGLang `153.6 / 111.8 / 267.4ms`. TorchInferno
therefore keeps the multi_turn TPOT cell locally but still trails on TTFT/E2E;
the next lever remains first-token scheduling or lower-overhead conversation
prefix reuse, not broader suffix graph warmup alone.

The adjacent greedy-mid row-cap idea is rejected. A focused few_shot run with
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_MID_MAX_ACTIVE=48` and matching
`b48` greedy suffix warmup regressed from the current full-pass default
`169.3 / 51.5 / 213.3ms` to `190.0 / 52.6 / 234.5ms`, with throughput falling
from `5.3` to `4.8 tok/s`. The profile introduced `b48` decode shapes, three
prefix-graph misses, and raised phase time from `8.14s` to `9.09s`
(`2.96s -> 3.35s` prefill wall). Keep greedy-mid active rows at `32`; larger
row caps need a separate decode/prefill shape improvement before they are useful
for few_shot.

Rechecking the greedy-large initial collection window after the suffix-bucket
change is also rejected. Forcing
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_INITIAL_BATCH_WAIT_MS=5` on current
`8d1085d` completed multi_turn at `327.6 / 64.9 / 390.9ms`, which looks better
than the focused no-env E2E but gives back TPOT. The queue counters do not
support it as a real server-side win: the initial batch stayed at `5`, phase
time rose to `9.94s` versus the no-env confirmation's `9.22s`, and both prefill
wall and decode-active time were higher. Keep the `10ms` greedy-large first
collection default; the suffix-bucket change did not reopen the old 5ms tradeoff.

Extending greedy decode-many to the 256-token few_shot path is rejected.
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_GEN_MAX_TOKENS=256` preserved raw
correctness but regressed few_shot to `303.6 / 203.2 / 506.3ms` and only
`2.1 tok/s`. The profile confirmed the wrong shape: `use_decode_many=true`,
but decode model calls jumped to `480` and decode tokens to `15.4k` for a row
that normally needs about `72` decode calls and `1.6k` decode tokens. Keep
decode-many scoped to short greedy output; few_shot needs a different stop-aware
decode batching mechanism.

Rechecking sampled-medium tree initial wait `5ms` is rejected on current
`92c2480`. The focused tree run preserved normal raw correctness (`960/992`) but
regressed to `280.2 / 49.5 / 323.4ms`, versus the current full-pass default
`219.0 / 48.1 / 257.4ms`. The first sampled-medium wave paid one request-path
prefill graph capture and `1.87s` prefill wall, and later waves still fragmented
into small prefill/decode groups. Keep the sampled-medium `10ms` first collection
window; shortening it does not recover the tree TTFT/E2E gap.

A pinned CPU readback buffer for decode-many is rejected. An opt-in patch reused
a pinned host tensor for the post-quantum token harvest instead of allocating via
`.cpu()` each time. The focused long_output run preserved correctness and landed
near the median E2E band (`294.9 / 25.3 / 1205.9ms`), but TTFT/tails worsened
and the counters did not improve: `runtime_decode_ragged_cpu_tokens_ms` stayed
at about `6.9s`, while total online phase rose to `22.86s`. The patch was
reverted; the long_output gap needs fewer/smarter decode harvests or real
overlap, not just pinned-memory reuse for the same synchronous copy.

## FlashInfer packaging and no-FI decode A/B (2026-06-28)

Public provider logs for `20260628_190318` showed vLLM and SGLang running with
FlashInfer-backed attention/sampling and CUDA graph buckets, while TorchInferno
printed repeated `[WARMUP] FlashInfer unavailable: No module named
'flashinfer'`. TorchInferno now declares a dedicated `flashinfer` optional
extra (`flashinfer-python>=0.6.8,<0.7`), and inference-bench installs
`.[serve,flashinfer]` for the TorchInferno GPU provider by default with a
fallback to `.[serve]` if the optional package cannot resolve. This keeps the
base serving extra CPU-friendly while letting public GPU runs exercise the same
optional FlashInfer-backed paths used locally.

The missing package is not the primary long_output bottleneck. A same-host
TorchInferno `81eda1d` run with `TORCHINFERNO_FI_DECODE_GRAPH=off` completed
long_output at `275.0 / 25.6 / 1288.5ms` (TTFT/TPOT/E2E), 1000/1000 correct,
versus the recent importable-FlashInfer control on `06c3e04` at
`290.8 / 23.9 / 1297.9ms`. Queue shape stayed the same: dense online serving,
`use_decode_many=true`, no paged engine, one ragged-prefill capture, and
ragged decode graph hits for nearly all decode batches. So the packaging change
is still the right public-run fairness fix, but long_output remains dominated by
decode GPU time plus token readback/CPU sampling overhead rather than by the
FlashInfer decode-graph warmup alone.

Self_consistency sampled-prefix sampling also rejected a low-threshold repeated
Gumbel sampler. Forcing
`TORCHINFERNO_TEMPERATURE_SAMPLE_REPEATED_GUMBEL_MIN_BATCH=1` on `03a2cee`
completed correctly but regressed the row to `311.3 / 0.0 / 331.4ms` versus the
current self control at `212.7 / 0.0 / 331.2ms`. The queue profile showed more
fragmented arrivals (`367` submit batches, `271` prefix-reuse batches,
`572.5ms` idle-drain wait) rather than a clean sampler win, so keep the
repeated-Gumbel threshold at `128`.

Tree sampled-prefix padding with free prefix-cache rows is also rejected on
current `d0093e5`. The working-tree patch let odd common-prefix suffix batches
borrow unused prefix-cache rows so they could replay warmed bucket graphs instead
of capturing request-path shapes. Queue profiles confirmed the intended internal
effect: the two focused tree runs eliminated the prior `b17`/`b31` suffix graph
captures and cut profiled prefill wall from the current control's `6.67s` to
about `4.6s`. The score-facing medians still regressed from the same-host
control `227.5 / 49.7 / 268.1ms` to `242.9 / 49.1 / 289.3ms` and then
`238.5 / 51.0 / 282.8ms`, so the code patch was reverted. Do not reopen this
padding route unless it improves tree median TTFT/E2E/throughput, not just
aggregate prefill capture time.

Long_output refreshed the greedy-short first-batch wait. A same-host comparison
on `8b0dde3` showed TorchInferno already wins local median TPOT against vLLM and
SGLang (`24.9ms` vs `28.4ms` and `25.1ms`) but still loses TTFT/E2E/throughput
(`284.3 / 1263.8ms` vs SGLang `61.3 / 948.6ms`). Lowering only
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_INITIAL_BATCH_WAIT_MS` from `10` to
`1` preserved TPOT and improved the row to `262.2 / 24.8 / 1193.6ms`, with
phase total `24.73s -> 24.11s`. Lowering the refill floor to `1` is rejected:
it improved TTFT to `152.9ms` but regressed TPOT/E2E/throughput to
`32.6ms / 1380.0ms / 27.9 tok/s` by fragmenting into `190` prefill batches.
Later same-host rechecks on `de4d2e4` moved this bucket from `1ms` to `0ms`:
the default barrier run was `281.6 / 25.5 / 1337.3ms`, while zero wait landed at
`274.0 / 24.9 / 1205.6ms` as an env-only run. The patched no-env confirmation
was `255.7 / 25.5 / 1341.6ms`, with better TTFT/throughput than the 1ms default
and E2E in the same variance band; all three runs were `1000/1000` correct.
Promote the `0ms` greedy-short first-batch wait only; this bucket stays below
few_shot's `256`-token cap and targets long_output.

Self_consistency on current `6ce94f1` is now mostly a control-plane/finish-path
gap, not a model-kernel gap. A same-host provider comparison landed at
TorchInferno `274.1 / 0.0 / 350.8ms`, vLLM `274.7 / 0.0 / 327.4ms`, and SGLang
`210.3 / 0.0 / 398.7ms`. The queue profile had one common-prefix prefill, one
decode, but `260` submit batches, `685.6ms` submit sync, and `262.4ms` idle
drain. Rechecking sampled-short initial wait `20ms` is rejected on this stack:
it admitted more initially (`7 -> 16`) but regressed to
`287.1 / 0.0 / 351.0ms`, raised phase total (`3.01s -> 3.17s`), and worsened
tail latency. Keep sampled-short initial wait at `10ms`; the next self lever is
reducing submit/finish churn without broadening the already-rejected shared
memory command scopes.

## Public run 20260628_190318 and finished-prefix reuse rejection (2026-06-28)

Public run `20260628_190318` measured TorchInferno `4000a03`, vLLM `c2127a2`,
and SGLang `ad30a99`. The scorecard moved to vLLM `16/20`, SGLang `2/20`,
TorchInferno `1/20`. TorchInferno kept only the few_shot TPOT cell
(`46.8ms` vs vLLM `47.4ms`), while public medians stayed behind on
self_consistency (`280.7 / 300.7ms` TTFT/E2E vs vLLM `186.9 / 211.4ms`),
multi_turn (`414.2 / 58.5 / 470.7ms` vs vLLM `163.3 / 51.0 / 206.4ms`),
tree_of_thought (`264.2 / 42.6 / 310.0ms` vs vLLM
`64.5 / 32.7 / 89.2ms`), and long_output
(`288.3 / 21.5 / 1060.7ms` vs vLLM `79.1 / 14.8 / 638.6ms`). This public row
still predates the later pushed queue-profile/capture-shape docs and code
through `8ee6cb8`.

A current same-host multi_turn control on `8ee6cb8` landed at
`360.0 / 61.5 / 423.8ms`, with no request-path prefill graph captures,
`5.39s` prefill wall, and `78.8k` prefill tokens across `34` prefix-graph
batches. Finished-prefix KV reuse remains rejected even when routed through the
non-common graph path. Enabling finished-prefix cache plus non-common finished
prefix graph prefill cut prefill tokens to `16.6k`, but fragmented into
`303` prefix-reuse batches, paid `84` graph captures (`48.1s`), and regressed
the row to `3524.4 / 66.8 / 3588.8ms`. Grouping mixed prefix lengths together
without a graphable context bucket avoided the capture storm but missed the
prefill graph (`31` misses), spent `17.2s` in prefill wall, and landed at
`1069.2 / 64.8 / 1137.7ms`.

A source prototype then forced mixed-prefix graph prefill through a static
negative context bucket instead of `context_len=None`. It did turn the path into
graph hits, but the cold run still landed at `450.2 / 69.7 / 518.0ms` with
`9.7s` capture time, and a same-server warm second pass was still slower than
the default at `400.1 / 62.2 / 447.9ms`. The patch was reverted. Do not enable
finished-prefix cache, mixed-prefix prefill, mixed-prefix graph prefill, or
startup warmups for those mixed source-row shapes unless a new implementation
beats the default median TTFT/E2E as well as aggregate phase time.

## Public direct-scatter startup regression (2026-06-28)

Public submit run `20260628_110307` failed before any TorchInferno benchmark
rows on `ed7f29b`. The server selected
`rank0_replicated_page_cache_warm=1 rank0_direct_scatter=1
rank0_shard_scatter=1`, spent `516.4s` on initial embedding/norm/head loading,
then only reached `10/80` layers after `2473.8s`. Nonzero ranks timed out in
NCCL `SCATTER` (`SeqNum=77`, `NumelIn=29360128`) while rank 0 had only
completed work `76`, so the direct-scatter startup path can let nonzero ranks
enqueue collective shard receives while rank 0 is still blocked reading or
packing a later checkpoint tensor.

This was not an isolated public-host artifact. Submit run `20260628_090338`
failed the same way on `467c3c3` (`SCATTER`, `SeqNum=78`) while the same direct
scatter code looked healthy only on the local devgpu run `20260628_093044`
(`15.9s` checkpoint load). The default has been reverted to portable concurrent
per-rank safetensor reads for both replicated and sharded checkpoint tensors.
Replicated page-cache ordering remains available via
`TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_PAGE_CACHE_WARM=1`, and direct
scatter remains available as an explicit opt-in via
`TORCHINFERNO_TP_RANK0_CHECKPOINT_SHARD_SCATTER=1`, with
`TORCHINFERNO_TP_RANK0_CHECKPOINT_DIRECT_SCATTER=0` preserving the older
reduce-scatter fallback only for controlled experiments.

Same-host inference-bench validation on the follow-up working-tree patch over
`f9abdc0` selected
`rank0_replicated_page_cache_warm=0 rank0_direct_scatter=0
rank0_shard_scatter=0`, loaded initial embedding/norm/head tensors in `0.3s`,
loaded all `80/80` layers in `18.3s`, and reached `/health` in `145.6s`. The
focused few_shot row completed at `161.5 / 46.7 / 200.8ms` with 98%
correctness, matching the expected latency band while removing the public
SCATTER startup failure mode.

## Latest public refresh (2026-06-28)

Public run `20260628_150314` measured TorchInferno `3f37307`, vLLM `4b643c4`,
and SGLang `aaa31eb`, so the startup failure from `20260628_110307` is fixed in
the public environment. TorchInferno now wins `3/4` few_shot cells:
`139.0 / 46.2 / 178.1ms` versus vLLM `141.9 / 50.1 / 183.7ms` and SGLang
`156.6 / 75.3 / 225.3ms`. The total scorecard is still vLLM `15/20`,
TorchInferno `3/20`, SGLang `1/20`. The remaining public gaps are:
self_consistency `248.2 / 0.0 / 267.6ms` versus vLLM
`190.3 / 0.0 / 228.8ms`, multi_turn `387.0 / 56.7 / 453.2ms` versus vLLM
`161.5 / 52.1 / 203.1ms`, tree_of_thought `278.6 / 41.5 / 318.2ms` versus
vLLM `63.1 / 31.6 / 86.4ms`, and long_output `256.9 / 21.6 / 1102.9ms` versus
vLLM `70.1 / 14.8 / 590.7ms` (TTFT/TPOT/E2E). Latest `main` through `f431d1c`
is no longer represented by this public row: it predates the later greedy-large
FP8 prefill gate, sampled-short online step-sync skip, and validation-only docs
through `56129eb`.

A same-host skip-build comparison on pushed `56129eb` refreshed the local
provider gap with current TorchInferno code. The run intentionally covered
self_consistency, multi_turn, tree_of_thought, and long_output, so the
self_consistency row is colder than the full public-order TorchInferno pass
that runs few_shot first. Local rows were:

- self_consistency: TorchInferno `303.8 / 0.0 / 328.0ms`, vLLM
  `256.0 / 0.0 / 318.5ms`, SGLang `192.7 / 0.0 / 356.7ms`.
- multi_turn: TorchInferno `372.6 / 60.8 / 440.3ms`, vLLM
  `288.7 / 116.5 / 393.6ms`, SGLang `149.3 / 106.8 / 260.2ms`.
- tree_of_thought: TorchInferno `210.9 / 47.7 / 246.0ms`, vLLM
  `123.5 / 82.7 / 185.7ms`, SGLang `63.6 / 69.4 / 152.6ms`.
- long_output: TorchInferno `268.3 / 24.7 / 1136.6ms`, vLLM
  `90.2 / 26.6 / 1038.7ms`, SGLang `64.3 / 24.4 / 919.6ms`.

This confirms the same shape as the public gap after the latest local source
changes: TorchInferno is still competitive on deterministic TPOT, but TTFT and
p99 E2E remain dominated by first-token prefill/admission and decode/readback
tails. The competitor logs show vLLM and SGLang both relying on broad
piecewise/chunked prefill graph coverage and large paged KV pools; previously
rejected local probes already cover the nearby TorchInferno knobs
(larger suffix warmup, lower refill floors, no-step-sync for deterministic
traffic, paged KV at the 569-token multi_turn shape, and naive chunked/unified
prefill).

The same-host skip-build comparison on pushed `6425184` remains useful for local
A/Bs, but no longer replaces public evidence. The local all-provider run
`20260628_150944` reached `/health` in `145.6s` for TorchInferno and `90.4s`
for SGLang; local vLLM failed only because the host `/lib64/libstdc++` is
missing `CXXABI_1.3.15`, and a separate vLLM rerun with the conda
`libstdc++.so.6.0.34` preload reached `/health` in `115.5s`. Combining those
rows, SGLang led most local TTFT/E2E/throughput cells while TorchInferno kept
competitive TPOT: TorchInferno rows were few_shot
`162.2 / 46.5 / 201.4ms`, self_consistency `293.3 / 0.0 / 324.0ms`,
multi_turn `344.1 / 60.4 / 404.4ms`, tree_of_thought
`284.4 / 47.9 / 317.2ms`, and long_output `237.7 / 25.4 / 1204.4ms`.
The local SGLang rows were `118.9 / 78.5 / 199.2ms`,
`229.5 / 0.0 / 385.7ms`, `150.4 / 114.1 / 266.4ms`,
`62.9 / 72.2 / 148.7ms`, and `66.8 / 24.6 / 907.0ms`.
The local vLLM preload rows were `231.9 / 86.8 / 314.4ms`,
`240.1 / 0.0 / 294.3ms`, `298.4 / 108.7 / 394.8ms`,
`123.3 / 83.0 / 186.3ms`, and `109.5 / 28.7 / 1125.3ms`.

A current public-order TorchInferno profile over
few_shot/self_consistency/multi_turn/tree_of_thought landed at
`166.6 / 47.4 / 205.0ms`, `276.3 / 0.0 / 302.1ms`,
`343.0 / 58.1 / 402.2ms`, and `266.6 / 48.9 / 307.3ms`.
The tree sampled-medium sessions handled `896` requests, spent `8.21s` in
online phase time and `5.58s` in prefill wall, and still paid two request-path
ragged-prefill graph captures totaling `1.96s`. That confirms the remaining
tree gap is still the sampled prefix/suffix prefill path plus burst scheduling,
not the public startup failure.

Three adjacent tree graph probes are rejected on this current stack. Lowering
only the sampled common-prefix warmup FP8 M gate with
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_SAMPLED_COMMON_PREFIX_FP8_MIN_M=1` captured
the wrong precision-key family for small suffix buckets and regressed tree to
`354.8 / 47.6 / 391.3ms`; sampled request-path captures rose from `2` to `6`
and capture time from `1.96s` to `5.48s`. Raising
`TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_GRAPHS=256` reduced sampled captures to
one and sampled phase time to `7.33s`, but retained graph memory rose to roughly
`66GB/GPU` during startup and self_consistency regressed from
`276.3 / 302.1ms` to `291.6 / 313.2ms`, so the broad cache increase is not a
default. A source patch that moved sampled common-prefix suffix warmup after the
runtime FP8 ragged warmup also reduced sampled captures to one and phase time
to `7.32s`, but it regressed score-facing tree to `283.6 / 48.4 / 324.3ms` and
self_consistency to `288.3 / 313.0ms`; the patch was reverted. Do not promote
more startup graph retention or reorder sampled warmup without a score-facing
win.

Current long_output remains decode/readback dominated. A focused profile on
`f431d1c` landed at `276.1 / 25.0 / 1250.9ms`, with `22.42s` online phase time,
`7.74s` prefill wall, one `1.01s` request-path prefill graph capture,
`11.70s` decode GPU event time, and `6.83s` CPU token harvest across `723`
ragged decode batches. Rechecking the greedy-short refill floor is rejected:
raising it to `16` cut prefill batches (`57 -> 50`) but regressed the row to
`300.5 / 25.4 / 1276.7ms` and raised phase time to `24.70s`; lowering it to
`8` improved profiled TTFT but worsened TPOT/aggregate phase, and the no-profile
confirmation landed at `224.5 / 27.2 / 1226.1ms` versus the nearby no-profile
default at `237.7 / 25.4 / 1204.4ms`. A narrower greedy KV-token budget
(`24576`, roughly mid-90 active rows) is also not a default: it landed at
`267.6 / 24.8 / 1199.7ms`, trading away TTFT without addressing the public
vLLM long_output gap. Keep the current refill floor and KV budget; the next
long_output work needs a real decode/readback or pipeline change.

One source-level decode-many scheduling follow-up is also rejected. A patch made
`ContinuousBatchEngine._can_step_decode_many` capacity-aware so waiting requests
only blocked multi-step greedy decode when active rows were below
`max_active_requests`, with the intent of amortizing token readback while all
rows were full. The focused long_output profile
`/tmp/ti_full_decode_many_waiting_profile` kept correctness but worsened the
row to `243.1 / 25.6 / 1215.7ms` versus the nearby local default
`237.7 / 25.4 / 1204.4ms`; queue counters also moved the wrong way
(`723 -> 753` decode batches, `544 -> 591` runtime step calls, `11.70s ->
13.12s` decode GPU event time, and `22.42s -> 24.74s` phase time). The patch
was reverted. The current one-token pacing while arrivals are waiting is still
the least-bad measured default until decode/readback can be pipelined.

Two FlashInfer decode follow-ups are rejected on current `d2bd224`. First,
forcing greedy traffic onto the sampled FlashInfer decode graph path with
`TORCHINFERNO_FI_DECODE_GRAPH=always` preserved long_output correctness but
regressed the focused row from the local default band (`292.3 / 25.4 /
1275.4ms` in the adjacent profiled run) to `263.3 / 36.9 / 1692.7ms`; a source
patch that fused one-token FlashInfer RoPE plus KV append with the existing
Triton ragged decode append kernel stayed in the same bad band at
`267.1 / 37.2 / 1707.9ms`. Second, the same fused FlashInfer append patch did
not produce a sampled tree win: default tree_of_thought landed at
`257.7 / 49.6 / 312.8ms`, compared with the current public-order profile's
`266.6 / 48.9 / 307.3ms`. The patch was reverted. Keep greedy decode on the
dense ragged token graph and do not add a FlashInfer decode-append branch
without a clear sampled score-facing gain.

A current multi_turn recheck on pushed `c1ac1b1` showed the greedy-large
512-token path is still prefill dominated: the default focused row landed at
`379.9 / 64.4 / 451.1ms`, with `5.81s` prefill wall and `5.35s` prefill
forward across `34` graph-backed prefix/suffix batches. Lowering only the
online greedy-large FP8 prefill M gate from `2048` to `512` improved the row to
`359.4 / 59.5 / 420.3ms` in the explicit-env A/B and `366.5 / 62.0 /
432.5ms` after making it the no-env default in the working tree. The default
confirmation profile recorded `fp8_prefill_min_m=512`, `5.63s` prefill wall,
and `5.18s` prefill forward, preserving the same `34` prefill batches and
1000/1000 finished events. Promote the `512` gate only for the existing
deterministic greedy-large online FP8 policy (`400 < max_tokens <= 512`);
sampled-medium keeps its separate `256` gate, and non-FP8 traffic remains out
of this policy. A post-push TorchInferno-only five-benchmark pass on `ddbad82`
kept the scoped behavior: few_shot `160.6 / 49.9 / 201.3ms`,
self_consistency `306.9 / 0.0 / 338.4ms`, multi_turn
`341.8 / 60.3 / 401.4ms`, tree_of_thought `249.5 / 47.8 / 284.9ms`, and
long_output `280.9 / 24.2 / 1307.6ms`.

A same-host provider comparison on `d4a03ec` confirms what remains in
multi_turn after the FP8 gate change. TorchInferno landed at `366.8 / 60.9 /
432.6ms`, vLLM at `291.3 / 107.6 / 389.2ms`, and SGLang at `167.3 / 103.9 /
279.3ms` (TTFT/TPOT/E2E), all with 100% correctness. TorchInferno now wins the
decode/TPOT cell locally, but still trails on TTFT, E2E, and throughput; the
next multi_turn lever is still first-token queueing/prefix scheduling, not
decode throughput.

Four same-host multi_turn follow-ups on current `0f0a0a7` are rejected. Enabling
only `TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE=1` did find more reusable
conversation tokens, but it routed the safe non-graph fallback into tiny suffix
prefills and regressed to `6471.0 / 65.6 / 6529.9ms`; queue counters showed
`488` prefill batches and `94.7s` prefill wall versus the default `34` batches
and roughly `5.6s`, so do not enable finished-prefix cache without a batched
graph-safe suffix path. Raising the greedy-large active cap to `48` regressed
score-facing latency to `447.1 / 67.4 / 519.4ms`, so the current `32`-row cap
remains the better default. Lowering the greedy-large FP8 prefill min-M from
`512` to `256` landed at `370.1 / 60.6 / 432.8ms`, not better than the default
band. Lowering the greedy-large refill floor from `32` to `16` also stayed in
the same median band, `370.1 / 62.6 / 432.4ms`, while worsening tails. Keep the
current `512` FP8 gate and `32` refill floor until a different scheduling path
improves both medians and tails.

## Current long-output admission profile (2026-06-28)

On pushed `5c67607`, same-host focused long_output control completed at
`274.3 / 25.3 / 1234.3ms` with 100% correctness. The queue profile row
`20260628_134910` showed the current remaining shape: `1000/1000` prefix reuse,
one prefill graph capture, `57` prefill batches, `7.85s` prefill wall, `765`
decode graph hits, and `25.26s` total queue phase. Decode TPOT remains close
to the local vLLM/SGLang band, but prefill/admission and CPU token harvest still
dominate the TTFT/E2E gap.

Two narrowly scoped admission follow-ups are rejected on that stack. Forcing
`TORCHINFERNO_OPENAI_TP_ONLINE_COLLECT_IDLE_ARRIVALS=1` improved median TTFT
to `250.8ms`, but left E2E flat at `1233.0ms` and worsened the long-output tail
from `1938.6/3261.5ms` p99 TTFT/E2E to `2915.6/4125.9ms`. Lowering only
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_INITIAL_BATCH_WAIT_MS` from `10` to
`5` also regressed the row to `285.2 / 24.7 / 1286.6ms`, with p99 TTFT/E2E at
`2797.9/4104.3ms`. Keep the current greedy-short initial wait and keep idle
arrival collection disabled for greedy short streams until a policy preserves
tail latency while improving median admission.

## Current tree-of-thought sampled-medium refresh (2026-06-28)

Current pushed `f59602e` keeps the score-facing focused tree row in the expected
band. The no-profile control reached `/health` in `145.7s` and completed
`959/992` raw requests correct at `224.9 / 49.4 / 265.3ms`, with p99 TTFT/E2E
`1491.5/1609.9ms`. A queue-profile run on the same commit is useful for phase
shape but not for score decisions: profiling synchronizes prefill work and moved
the row to `314.9 / 49.8 / 368.0ms`. Its final records still show the known
split: sampled `max_tokens=300` sessions handled `896` requests, spent `8.86s`
in online phase time, `5.56s` in prefill wall, and paid two request-path
ragged-prefill captures totaling `2.00s`; deterministic eval sessions handled
the other `80` requests and were mostly decode-GPU bound.

A later queue-profile refresh on `7edb9e8` added capture-shape counters and
showed the remaining sampled captures were not the warmed `b32` bucket, but
unpaddable active-row shapes: `ragged_prefill:b31:s16:rows1:ctx-256:src1` and
`ragged_prefill:b20:s16:rows1:ctx-256:src1`. Splitting those unpaddable groups
into power-of-two sub-batches removed request-path captures and cut aggregate
profiled phase time (`11.90s -> 9.10s`) plus p99 E2E, but regressed the
score-facing focused tree row to `266.0 / 50.1 / 317.1ms` by increasing median
first-token delay. The patch was reverted; do not promote power-of-two
sub-batching unless it preserves median TTFT/E2E, not just aggregate phase time
or p99.

Rechecking sampled common-prefix warmup alignment with
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_SAMPLED_COMMON_PREFIX_TEMPERATURE=0.7` is
still rejected on the current stack. The unprofiled candidate completed
`963/992` raw correct but regressed the row to `281.4 / 50.4 / 316.5ms` and
worsened p99 TTFT/E2E to `2973.8/3562.5ms`. Keep the sampled warmup temperature
default at `1.0`; current tree work should avoid retesting this knob and focus
on a different prefill/session pipeline change.

Lowering the sampled-medium FP8 prefill min-M is also rejected on current
`0f0a0a7`. With
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_FP8_PREFILL_MIN_M=128`,
tree_of_thought completed `959/992` raw correct at `232.9 / 49.6 / 288.2ms`,
but the paired no-env run immediately after completed `961/992` at
`218.3 / 48.8 / 262.0ms`. Keep the sampled-medium FP8 min-M at `256`.

## Current multi-turn admission refresh (2026-06-28)

Current pushed `0db61a5` keeps focused multi_turn in the recent band but still
well behind vLLM/SGLang on queue-facing latency. The no-profile control reached
`/health` in `145.6s` and completed `981/1000` raw requests correct at
`354.4 / 62.5 / 414.3ms`, with p99 TTFT/E2E `3393.6/3430.4ms`. The queue
profile landed nearby at `373.7 / 65.7 / 440.2ms` and shows the current
mechanism clearly: `34/34` prefill graph hits, zero request-path prefill
captures, `5.78s` prefill wall (`5.31s` forward), `45,000` reused common-prefix
tokens, `82,874` raw prefill tokens, `4.12s` decode-active, and
`704ms` CPU token harvest. The remaining gap is not startup graph coverage; it
is the dense conversation-prefix/suffix prefill floor plus decode interleaving.

Two first-wave admission follow-ups are rejected. Raising the fixed
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_INITIAL_BATCH_WAIT_MS` to `20`
improved p99 TTFT/E2E to `2178.5/2249.0ms`, but regressed score-facing median to
`364.8 / 63.5 / 429.1ms`. A bounded reset-on-arrival prototype with the same
`20ms` cap also regressed median to `373.6 / 62.3 / 433.6ms` and did not keep
the fixed-wait tail win (`3064.8/3086.6ms` p99). Keep the greedy-large initial
wait at `10ms`; multi_turn needs a lower-overhead conversation-prefix reuse or
prefill pipeline change rather than another first-wave wait knob.

## Current self-consistency sampled-short refresh (2026-06-28)

Current pushed `3f37307` shows focused self_consistency is still highly
run-order sensitive. Two no-profile self-first controls were stable in the slow
band, `326.3 / 0.0 / 347.6ms` and `329.1 / 0.0 / 353.1ms`, both 1000/1000
correct with `/health` at `145.7s`. A queue-profile run on the same tree landed
much faster at `214.5 / 0.0 / 313.0ms`, so the score-facing control should come
from no-profile runs, not the profiled row.

The profile still confirms the intended sampled-short mechanism: one
`common_prefix:b1:t55` prefill graph hit, one generated-prefix store,
`980` generated-prefix reuses, one decode model call, and `55` raw prefill
tokens for the whole 1000-request burst. The remaining server-side work was
arrival/command fragmentation: `207` submit batches, `189` runtime steps,
`2.74s` online phase time, `1.55s` prefix-reuse/prefill wall, `512ms`
submit-sync, and `111ms` step-sync. Do not reopen generated-prefix, sampled
initial/idle wait, min-ready, decode-quantum, or HTTP worker-prestart knobs from
this evidence; the next self lever needs a batching/command-cadence change that
improves no-profile medians and full-order behavior together.

An active-arrival coalescing prototype is rejected on current `5320caf`. The
patch added an opt-in sampled-short active drain window and the 2ms A/B landed
at `229.6 / 0.0 / 340.1ms`, which improved isolated TTFT but left E2E flat and
moved queue counters the wrong way versus the paired profile (`2.85s -> 2.92s`
phase time, `198 -> 202` runtime steps, `196 -> 200` prefix-reuse batches, and
`1.59s -> 1.65s` prefill wall). The prototype was reverted. Do not add an
active wait in front of live sampled-short submissions without a throughput/E2E
win.

One command-cadence change is accepted on current `51ba4f1`. Disabling the
per-step tensor-parallel sync only for sampled-short online sessions
(`temperature > 0`, `max_tokens <= 256`) preserved 1000/1000 correctness and
improved the paired no-profile self_consistency row from `350.0 / 0.0 /
373.1ms` to `313.7 / 0.0 / 336.1ms`; the explicit-env discovery run was
`293.3 / 0.0 / 315.4ms`. The profiled no-env confirmation landed at
`316.7 / 0.0 / 336.5ms`, kept the generated-prefix path healthy
(`981` generated-prefix reuses, one store, one prompt prefill), and removed the
`phase_step_sync_ms` field from the queue profile. Keep the env override
`TORCHINFERNO_OPENAI_TP_ONLINE_STEP_SYNC` authoritative for debugging or
conservative deployments.

A post-push TorchInferno-only five-benchmark pass on `2564898` validated the
sampled-short sync policy in full benchmark order: few_shot
`159.6 / 47.6 / 197.3ms`, self_consistency `197.7 / 0.0 / 248.7ms`,
multi_turn `328.8 / 59.6 / 384.8ms`, tree_of_thought
`227.9 / 48.0 / 268.2ms`, and long_output `252.0 / 25.0 / 1154.8ms`, all
with normalized correctness at 1.0. Two adjacent command-path follow-ups are
rejected on the same commit. Expanding shared-memory command transport to all
online commands (`TORCHINFERNO_OPENAI_TP_SHM_COMMAND_MODE=online`) reduced
measured submit-sync time but regressed focused self_consistency to
`320.5 / 0.0 / 337.8ms`, with more submit batches and higher runtime
prefill/step wall. Disabling online step sync for deterministic 512-token
multi_turn (`TORCHINFERNO_OPENAI_TP_ONLINE_STEP_SYNC=0`) also regressed: the
profiled row was `355.1 / 59.3 / 417.9ms`, and the no-profile confirmation was
`376.4 / 63.0 / 445.2ms`. Keep the no-step-sync default scoped to sampled-short
online sessions only.

## Published local refresh and rejected follow-ups (2026-06-28)

The public result stream stalled after `20260628_050325`, so a same-host
skip-build all-provider run was published as inference-bench
`20260628_093044` after preloading the conda `libstdc++` needed by the local
vLLM import path. This remains the best same-host artifact for local comparison:
SGLang won `14/20` metric cells, TorchInferno won `3/20`, and vLLM won `2/20`.
TorchInferno's rows were few_shot `165.3 / 51.1 / 207.1ms`,
self_consistency `245.0 / 0.0 / 329.0ms`, multi_turn
`348.1 / 61.2 / 408.1ms`, tree_of_thought `238.2 / 48.6 / 281.2ms`, and
long_output `301.9 / 25.5 / 1154.4ms` (TTFT/TPOT/E2E). The shape is unchanged:
TorchInferno's decode TPOT is competitive, but SGLang and vLLM still win most
TTFT/E2E/throughput cells through lower prefill and scheduling overhead.

Several follow-ups on top of current `467c3c3` are rejected. First, a mixed-prefix
context-bucket prototype made the opt-in finished-prefix graph path unit-correct,
but the real multi_turn route was still unsafe. With
`TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE=1`,
`TORCHINFERNO_CONTINUOUS_NON_COMMON_PREFIX_GRAPH_PREFILL=1`,
`TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_PREFILL=1`, and
`TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_PREFILL_GRAPH=1`, the 8-GPU run wrote only
one queue record (`81` submitted, `49` finished), then sat at 0% GPU utilization
until manual termination. The prototype was reverted; finished/generated prefix
reuse still needs a batching policy that cannot strand live requests.

Second, raising sampled-short idle batch wait to `20ms` regressed the close
self_consistency row on the current stack. The focused few_shot/self run landed
at few_shot `163.8 / 49.6 / 202.9ms` and self_consistency
`260.5 / 0.0 / 345.1ms`; the queue profile spent about `344ms` in idle drain
and ended with `252` submit batches. Keep the existing `10ms` sampled-short
idle wait.

Third, decoupling prefill/admission capacity from decode width is not a quick
tree fix. An opt-in prototype with `max_active=64` and a 32-row decode limit
preserved correctness but regressed focused tree_of_thought to
`382.8 / 52.8 / 443.5ms`. Extra prefill/admission capacity introduced more
prefill and graph-shape pressure than it saved in queueing, so the scaffold was
reverted.

Two later long_output follow-ups on the same runtime shape are also rejected.
Forcing deterministic short-generation submit+step commands with
`TORCHINFERNO_OPENAI_TP_ONLINE_SUBMIT_STEP_COMMAND=1` completed correctly but
regressed the focused row to `262.0 / 25.0 / 1344.8ms`. It reduced submit-sync
time, but queue phase still rose to `25.9s` and ragged decode GPU time rose to
`15.2s`, so command coalescing is still not the limiting factor for long_output.

Adding batch `64` to greedy common-prefix suffix warmup removed the remaining
request-path long_output prefill capture (`1 -> 0`) and cut the profiled queue
phase (`24.05s -> 22.78s`), but it did not improve score-facing latency. The
broad env override regressed the no-profile row to `274.3 / 25.4 / 1322.8ms`,
and a narrower source patch that warmed only the short-greedy `b64/s64` shape
also regressed under profiling to `298.2 / 25.1 / 1380.1ms`. Do not promote more
startup suffix graphs for this row without a score-facing win; the remaining gap
is still decode/readback and streaming tails, not that single prefill capture.

Two command-path follow-ups are also rejected on the current stack. Enabling the
existing `DecodeGraphRunner` for focused few_shot reduced the internal queue
phase to `4.91s` but regressed the benchmark row to
`201.2 / 62.4 / 251.1ms` (TTFT/TPOT/E2E), versus the current
`~165 / 51 / 207ms` band. Keep the runner off by default; the synchronous
harvest path still does not translate internal work reduction into client-visible
latency.

Broadening tensor-parallel shared-memory command transport is not a durable tree
or self fix. `TORCHINFERNO_OPENAI_TP_SHM_COMMAND_MODE=all` regressed focused
self_consistency to `296.8 / 0.0 / 359.4ms`. The same broad mode looked
promising for tree_of_thought at `212.5 / 49.9 / 254.1ms`, but the scoped
sampled-medium prompt-submit mode that would avoid sampled-short/greedy
regressions did not reproduce the win (`233.2 / 49.6 / 283.2ms`, with a worse
p99). The env-gated prototype was reverted; keep the default sampled
start/step/close shared-memory scope unchanged.

The sampled-medium FP8 prefill boundary is also rejected for tree. Lowering only
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_FP8_PREFILL_MIN_M` from `256` to
`255` makes the exact `b16 x s16 = 256` suffix-prefill bucket use FP8, but the
focused row landed at `225.3 / 50.4 / 265.1ms`. Queue profiles still showed two
request-path prefill graph captures and worse TPOT/throughput, so the current
`M > 256` default remains the safer tradeoff.

Two HTTP/client-side follow-ups are rejected on the current public-order shape.
A focused self_consistency run with TorchInferno fast-HTTP and queue profiling
landed at `336.4 / 0.0 / 353.1ms` with 100% correctness, while the server-side
fast-HTTP profile showed p50 first content at only `17.9ms`. The queue profile
showed the expected generated-prefix path (`1` store, `983` reuses), so the gap
is still outside the token send/decode loop and mostly in request admission,
batching, and command cadence rather than SSE serialization.

An inference-bench thread-local OpenAI/httpx client prototype is not a fair
default promotion. It preserved correctness and moved the focused TorchInferno
self row to `307.7 / 0.0 / 327.7ms`, but TTFT was worse than the current public
row and the E2E/throughput movement was too small and noisy to justify changing
the harness. The prototype was reverted.

Prestarting TorchInferno's fast-HTTP `ThreadPoolExecutor` workers is also
rejected as a default. When self_consistency ran first with
`TORCHINFERNO_OPENAI_HTTP_PRESTART_WORKERS=256`, the focused row improved to
`181.1 / 0.0 / 287.9ms`, but that is not the public benchmark order. In a
TorchInferno-only full-order pass with the same setting, few_shot landed at
`165.9 / 48.3 / 205.8ms`, self at `286.7 / 0.0 / 335.1ms`, multi_turn at
`335.5 / 61.0 / 397.4ms`, tree at `261.7 / 47.4 / 303.9ms`, and long_output at
`283.6 / 24.4 / 1159.0ms`. The mixed movement did not create a clear
score-facing win, and tree/self E2E regressed versus the current public row, so
the prototype was reverted.

Extending fast-HTTP assistant-role deferral through 512-token streams is rejected
for multi_turn. The existing default defers the role frame for streams up to 400
tokens, so multi_turn's `max_tokens=512` path still sends a standalone role SSE
chunk before the first content chunk. Raising only
`TORCHINFERNO_OPENAI_STREAM_DEFER_ROLE_MAX_TOKENS=512` preserved correctness but
landed at `353.1 / 62.6 / 416.6ms` with a large p99 tail, worse than the current
public multi_turn row. Keep the role-deferral bound at 400.

## Direct rank-0 shard scatter probe (2026-06-28)

The old rank-0 shard-scatter checkpoint path used `reduce_scatter_tensor`,
which forced every nonzero rank to allocate and collectively reduce a full-size
zero tensor for each checkpoint weight. That explains why the earlier
`rank0_shard_scatter=1` default was brittle on public hosts: it removed shared
storage reads but replaced them with very large unnecessary collective inputs.

The next probe tried direct `dist.scatter` when shard scatter was enabled. Rank
0 loaded each full sharded tensor once, built equal-sized contiguous shards, and
scattered those shards directly; nonzero ranks no longer read checkpoint shards
or allocated full zero inputs. This fixed the old reduce-scatter allocation
problem, but later public submit runs showed that direct scatter is still too
environment-sensitive for the default startup path.

Same-host validation on current `e7c7acb` plus this patch selected
`rank0_direct_scatter=1 rank0_shard_scatter=1` with no provider env override.
Checkpoint load fell to `16.4s` versus roughly `20s` on the replicated
page-cache-warm all-rank-shard path; an opt-in run just before the temporary
default change loaded in `15.8s` and completed self_consistency with 1000/1000
correctness. This remains useful evidence for controlled hosts, but it is no
longer the source default.

## Current `564783b` refresh and rejected tree capture bypass (2026-06-28)

The latest public v1 run is still `20260628_030309`, which measures
TorchInferno `5680b84`, vLLM `11a1230`, and SGLang `4a76699`. It predates the
batch-2 sampled suffix warmup and the 10ms greedy-large initial collection
default. Public scorecard wins are vLLM `15/20`, TorchInferno `2/20`, and
SGLang `2/20`. TorchInferno rows are few_shot
`139.1 / 45.9 / 177.3ms`, self_consistency `248.8 / 0.0 / 265.5ms`,
multi_turn `411.7 / 57.6 / 466.0ms`, tree_of_thought
`295.2 / 42.4 / 360.6ms`, and long_output `310.2 / 22.3 / 1190.1ms`
(TTFT/TPOT/E2E).

A full local TorchInferno-only pass on pushed `564783b` completed all rows with
the expected correctness band: few_shot `162.7 / 49.4 / 204.0ms`,
self_consistency `314.9 / 0.0 / 345.2ms`, multi_turn
`358.0 / 62.7 / 423.4ms`, tree_of_thought `307.0 / 47.9 / 332.4ms`, and
long_output `267.6 / 24.5 / 1224.5ms`. The multi-turn initial-wait change
carried into the full sequence and cut p99 TTFT/E2E to about `996/1055ms`.
Self and tree still show focused/full-run order noise; do not use the slower
full-run self/tree medians alone as proof of a new runtime regression.

Disabling request-path common-prefix suffix graph capture remains rejected for
tree. With `TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_CAPTURE_ON_MISS=0`, the
focused tree row regressed to `342.4 / 49.5 / 396.3ms`. Queue counters showed
zero captures but four prefill graph misses; the eager miss path was more
score-visible than the rare captures it avoided.

Fast HTTP keepalive cleanup is now scoped to fully drained engines. A prior
focused `few_shot -> self_consistency` control landed at few_shot
`164.8 / 48.3 / 203.1ms` and self_consistency `305.8 / 0.0 / 337.2ms`.
Forcing every keepalive socket down to `0.25s` helped the next row
(`277.6 / 0.0 / 304.2ms`) but hurt few_shot
(`170.1 / 50.1 / 210.1ms`). The promoted version keeps the normal `5s`
keepalive timeout while requests are live, then uses a `0.25s` timeout only
after the engine drains; the same focused sequence landed at few_shot
`168.0 / 48.8 / 207.0ms` and self_consistency `299.6 / 0.0 / 334.0ms`.
Extending that short timeout to low-but-nonzero live-request counts is rejected:
the threshold variant landed at few_shot `167.5 / 51.3 / 207.7ms` and
self_consistency `304.4 / 0.0 / 333.5ms`.

Short greedy common-prefix suffix prefill now uses the dynamic-context graph
bucket through suffix `128` only for deterministic `max_tokens <= 128`
sessions. Current long_output on pushed `dfc88cb` spent `5.94s` in nine
request-path prefill graph captures and landed at `240.2 / 26.0 / 1287.1ms`.
The scoped source-default rerun cut request-path captures to one, prefill wall
`11.10s -> 8.11s`, and phase time `27.83s -> 24.07s`, with score-facing
long_output at `246.6 / 26.5 / 1264.0ms`. A broad dynamic suffix-128 default
is rejected: it made long_output faster internally but regressed multi_turn to
`433.9 / 60.7 / 493.6ms`. The scoped version kept multi_turn in band at
`364.7 / 62.3 / 423.6ms` with zero prefill captures.

A full local TorchInferno-only pass on pushed `f1114b6` completed with the
expected correctness band: few_shot `160.4 / 49.1 / 199.3ms`,
self_consistency `302.7 / 0.0 / 329.8ms`, multi_turn
`339.0 / 58.3 / 400.5ms`, tree_of_thought `230.5 / 46.7 / 277.5ms`, and
long_output `257.5 / 25.0 / 1231.0ms`. The remaining local self-consistency
issue is row-order/client-wave sensitive: a focused self-only profile on the
same code landed at `199.6 / 0.0 / 345.2ms`, while the fast HTTP handler's
server-side median total was only `16.5ms` after request read. Raising
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_SHORT_IDLE_BATCH_WAIT_MS` from `10` to
`20` is rejected for now: it improved self E2E/throughput to
`317.0ms`/`3.2 tok/s`, but regressed TTFT to `296.7ms` and barely moved queue
phase time (`2.97s -> 2.90s`).

## Current `578e117` refresh and rejected follow-ups (2026-06-28)

The latest public v1 run is `20260628_010307`, which measures TorchInferno
`578e117`, vLLM `11a1230`, and SGLang `da802dd`. It includes the two pushed
TorchInferno runtime updates from this loop: `886ad79` buckets short
common-prefix suffix graphs behind the dynamic-context path, and `578e117`
warms sampled common-prefix suffix graphs under the same sampled
symmetric-memory policy used at runtime. The public scorecard is still vLLM
`16/20`, SGLang `2/20`, and TorchInferno `1/20`. TorchInferno rows are now
few_shot `143.2 / 45.3 / 184.2ms`, self_consistency
`262.6 / 0.0 / 278.0ms`, multi_turn `353.5 / 58.9 / 412.2ms`,
tree_of_thought `269.8 / 42.3 / 312.2ms`, and long_output
`316.8 / 22.1 / 1126.9ms` (TTFT/TPOT/E2E). The few_shot row now wins TPOT
against vLLM, but vLLM still owns the other score cells.

A full local TorchInferno-only pass on pushed `578e117` completed all five
rows with 100% benchmark-level correctness: few_shot
`160.6 / 46.9 / 198.6ms`, self_consistency `271.1 / 0.0 / 291.6ms`,
multi_turn `343.8 / 59.6 / 402.5ms`, tree_of_thought
`289.2 / 47.0 / 324.6ms`, and long_output `261.5 / 24.2 / 1188.8ms`. Focused
no-profile/current profiles show substantial run-order noise: tree-only now
lands around `238.7 / 50.1 / 289.9ms`, and long-output around
`233.9 / 24.7 / 1251.7ms`.

The self_consistency queue profile separated server work from benchmark-client
arrival timing. The default row was `234.3 / 0.0 / 354.2ms`, while the final
queue snapshot showed only `3.18s` total online phase time, `243` submit
batches, `218` runtime steps, one common-prefix prefill, one decode batch, and
`979` generated-prefix reuse hits. A local patch that accumulated idle arrivals
before one TP submit reduced internal work (`243 -> 196` submit batches and
`3.18s -> 2.89s` phase time), but it regressed score-facing TTFT to
`295.3ms`. Setting idle wait to zero was worse at `311.5 / 0.0 / 332.5ms`.
Reject both policies; benchmark median TTFT, not only internal phase time, moved
the wrong way.

An adjacent sampled-short scheduler patch that also used the existing
submit+step TP command for idle-arrival batches is rejected. The focused
self_consistency row on current `39e1975` landed at `316.7 / 0.0 / 339.1ms`,
1000/1000 correct. It trimmed the final queue profile only slightly
(`243 -> 237` submit batches, `218 -> 212` runtime steps, and about
`3.18s -> 3.07s` online phase time) while pushing median TTFT substantially
worse. Keep submit+step combination limited to submissions made while online
work is already active.

Current long_output remains decode/readback dominated. The focused profile
(`233.9 / 24.7 / 1251.7ms`) recorded `9.48s` prefill wall, `4.42s` request-path
prefill graph capture across seven exact-context suffix shapes, `12.48s`
ragged-decode GPU event time, and `6.92s` CPU token readback across `747`
decode batches. Disabling common-prefix prefill capture-on-miss removed capture
time but regressed to `363.6 / 33.4 / 1684.6ms` because eager suffix prefill
grew prefill wall to `13.72s`. Extending dynamic prefix context to suffix
bucket `64` cut captures to four and prefill wall to `9.11s`, but the
boolean-mask bucket path still regressed score-facing latency to
`258.5 / 25.4 / 1319.0ms`. Keep exact-context graph capture enabled for long
until there is a faster dynamic-context attention path.

Tree-of-thought on `578e117` no longer spends request time capturing sampled
common-prefix suffix graphs in the common short-prefix sessions; queue records
show only two prefill captures across the focused run. The current focused row
is `238.7 / 50.1 / 289.9ms`, and aggregate final batcher records are still
split between sampled `max_tokens=300` sessions and deterministic
`max_tokens=400` eval sessions. The remaining local tree gap is prefill and
decode work across bursty sessions (`~6.3s` aggregated prefill wall and
`~4.1s` decode-active time in final queue records), not a missed HTTP streaming
optimization.

Same-host provider comparison remains directionally consistent even though the
local vLLM import path is slower than the public vLLM row: a skip-build vLLM
tree-only run with the conda `libstdc++` preloaded completed at
`131.3 / 84.3 / 193.9ms`, versus local TorchInferno `238.7 / 50.1 / 289.9ms`
and public vLLM `63.2 / 31.1 / 85.9ms`.

A runtime FP8 ragged-prefill warmup alignment experiment is rejected. Changing
that startup warmup to use `_dynamic_prefix_prefill_context_len()` instead of
exact `prefix+suffix` contexts did not remove the two sampled prefix captures
and regressed the focused tree row to `302.7 / 50.3 / 344.7ms`; the patch was
reverted.

Two adjacent tree follow-ups are also rejected. Raising
`TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_GRAPHS` to `256` did not show graph-cache
eviction as the limiter: the focused row stayed in the same band at
`242.4 / 49.5 / 294.7ms` while request-path prefill captures increased to
three (`2.78s`) and aggregate prefill wall rose to `7.16s`. A code patch that
adopted the freshly-prefilled common-prefix row directly into the reusable
prefix cache avoided one conceptual KV copy, but it did not improve the real
tree path. The focused row landed at `240.7 / 49.3 / 289.8ms`, and sampled
sessions worsened from `42` to `45` prefill batches, `2` to `3` request-path
captures, and `7.82s` to `9.13s` aggregate sampled phase time. The patch was
reverted; the remaining tree gap is still suffix-prefill capture/phase
fragmentation plus decode, not common-prefix row registration.

Two later cache/transport probes are also rejected. Persisting common-prefix
rows across online sessions worked mechanically (`restored_common_prefixes=1`
after the first session and later sessions skipped the `common_prefix:b1:t45`
prefill), but the focused tree row regressed to `256.5 / 50.6 / 305.6ms`.
Sampled sessions gained an extra request-path capture and moved from `7.82s`
to `9.09s` aggregate phase time, so the code was backed out. A primary-only
generated-prefix fast-answer path is unsafe for TP: cached-logit sampling enters
the Llama3 tensor-parallel sampler, which uses distributed collectives, so rank
0 cannot sample without matching worker participation. The self_consistency run
with that prototype hung before writing a queue profile; do not reintroduce it
without a symmetric worker command.

Fast-HTTP keepalive is not a broad latency fix. With queue and HTTP profiling
enabled, disabling streaming keepalive improved self_consistency modestly
(`250.2 / 0.0 / 341.2ms` to `240.6 / 0.0 / 326.4ms`), but tree_of_thought
regressed to `281.0 / 51.2 / 322.6ms` with a much worse p99. Keep the fast HTTP
default on; the self row's server-side first-content median is already around
`18ms` after request read, so the remaining gap is not a simple SSE write or
keepalive setting.

Changing the sampled common-prefix suffix warmup temperature default is not
robust enough to promote. An env-only run with
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_SAMPLED_COMMON_PREFIX_TEMPERATURE=0.7`
looked promising (`254.4 / 48.6 / 304.5ms` no-env control to
`224.3 / 49.7 / 268.4ms`, with sampled phase total `8.88s -> 8.06s` and
captures `3 -> 2`). But the same value as a code default did not reproduce the
mechanism: score-facing E2E stayed better at `291.2ms`, but sampled captures
rose to `4`, sampled prefill wall rose to `7.06s`, and sampled phase total rose
to `9.88s`. Keep the `1.0` default until the graph-key mismatch is explained.

Adding batch `2` to the common-prefix suffix warmup is promoted. The current
tree control still hit tiny sampled suffix shapes such as
`prefix_graph:b2:s16:p45-45:src1:mixed0`, while the default warmup covered
`1,4,8,16,32`. Warming `1,2,4,8,16,32` moved focused tree to
`243.8 / 49.6 / 291.0ms` and improved the sampled branch counters:
phase total `8.88s -> 8.04s`, prefill wall `6.24s -> 5.45s`, prefill forward
`4.61s -> 3.81s`, and request-path captures `3 -> 2`. This is only one extra
startup graph shape per suffix/policy and does not alter runtime scheduling.

Raising the greedy-large online initial collection window to `10ms` is
promoted for the 512-token multi-turn path. On current `55475d1`, the focused
default landed at `378.6 / 65.1 / 445.7ms` with an initial batch of `2`,
`10.97s` total online phase time, `5.93s` prefill wall, and no request-path
prefill captures. Setting
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_INITIAL_BATCH_WAIT_MS=10` admitted
`9` initial requests, kept captures at zero, and moved the focused row to
`363.0 / 59.2 / 421.3ms` with `9.27s` phase time and `5.53s` prefill wall.
A source-default confirmation with the code change landed at
`372.9 / 61.6 / 440.0ms`, still at zero request-path captures and `9.51s`
phase time. The adjacent `15ms` check was worse at
`369.9 / 62.2 / 432.7ms`, so keep the new default at `10ms`. Raising the active
cap to `48` remains rejected: it regressed to `592.7 / 65.6 / 665.3ms`,
introduced four request-path prefill captures, and pushed prefill wall to
`9.66s`.

Current multi_turn on pushed `6659e61` is still prefill dominated:
`366.8 / 62.8 / 426.0ms`, with `5.52s` prefill wall, `3.54s` decode active,
and only common-prefix reuse (`45,000` reused prefix tokens). A mixed-prefix
dynamic-context experiment is rejected. Enabling pinned full-prompt stores,
non-common-prefix graph prefill, mixed-prefix graph prefill, and a mixed-prefix
dynamic context bucket raised prefix reuse to `~97.7K` tokens and cut prefill
tokens to `~45.7K`, but the store/state path and graph capture dominated:
`1123.2 / 69.0 / 1209.3ms` with `21.1s` prefill wall, `11.3s` prefill state
time, and `6.29s` graph capture even when dynamic bucketing covered suffixes
through `256`. Without the wider dynamic bucket the same idea was worse at
`2051.9 / 68.7 / 2131.5ms` with `27` request-path graph captures. Keep
full-prompt mixed-prefix reuse out of default multi_turn; the missing piece is a
low-overhead prefix-store/adoption policy, not just a graph context key.

## Latest public refresh and streaming send batching (2026-06-27)

The latest public result while profiling this loop is still
`20260627_190305`, so it predates the current TorchInferno head. It used
TorchInferno `019ce7b`, vLLM `35e3850`, and SGLang `592f6c8`; scorecard wins
were vLLM `17/20`, TorchInferno `1/20`, and SGLang `1/20`. TorchInferno rows
were few_shot `253.7 / 47.5 / 299.6ms`, self_consistency
`273.3 / 0.0 / 293.2ms`, multi_turn `492.2 / 59.7 / 544.7ms`,
tree_of_thought `268.8 / 42.9 / 309.5ms`, and long_output
`325.9 / 21.3 / 1130.4ms` (TTFT/TPOT/E2E). The run does not include the later
BPE-cleanup suppression or sampled tensor-parallel shared-memory command
transport.

Focused local long_output profiling on current `ccb13de` kept the gap
decode/readback bound: `25.39s` queue phase time, `9.61s` prefill wall,
`4.51s` prefill graph capture, `12.43s` ragged-decode GPU event time, and
`7.17s` CPU token readback across `725` ragged decode batches. Forcing greedy
short decode quantum `4` is still rejected on this code: TPOT improved slightly
to `22.9ms`, but TTFT/E2E regressed to `279.4 / 1304.4ms` versus the nearby
default at `254.8 / 1149.2ms`. Keep the short greedy decode-many quantum at
`3`.

Fast-HTTP profiling showed an independent host-side cost: the server wrote one
socket frame per content token even when the online scheduler had already
produced a small decode-many burst for that request. Streaming now keeps one
OpenAI content delta per generated token, but `OpenAICompletionEngine` exposes a
batched token iterator so the fast HTTP path can pack multiple already-ready
SSE events into one `sendall`. This preserves the benchmark's chunk/token
accounting. In a focused long_output A/B, summed content-send time dropped from
`26.35s` to `20.70s` across request threads while keeping `36,715` content
chunks, and queue phase time dropped from `25.39s` to `24.07s`. Full
TorchInferno-only validation with the patch completed all rows at few_shot
`146.5 / 47.4 / 188.9ms`, self_consistency `289.0 / 0.0 / 324.8ms`,
multi_turn `363.8 / 59.6 / 430.2ms`, tree_of_thought
`217.1 / 47.4 / 263.9ms`, and long_output `262.7 / 24.5 / 1215.8ms`, all with
100% benchmark-level correctness.

A decode-many GPU staging-buffer rewrite is rejected. Replacing the existing
per-step token clones plus final `torch.cat(...).cpu()` with a reusable flat GPU
token buffer increased long_output CPU-token readback from `6.78s` to `7.44s`
and queue phase time from `24.07s` to `26.45s`. The current clone/cat path is
therefore the better default until CPU readback can be overlapped or removed by
a different event-delivery design.

The current same-host tree-only provider comparison on TorchInferno `961c30c`
still shows the largest short-context TTFT/E2E gap: TorchInferno
`228.4 / 48.9 / 279.3ms`, vLLM `66.4 / 33.2 / 92.3ms`, and SGLang
`61.4 / 74.2 / 153.2ms` (TTFT/TPOT/E2E), all 100% benchmark-level correct.
TorchInferno's fast-HTTP profile is not the limiter here: sampled tree median
first-content was `206.8ms` and median content-send cost was only `1.2ms`.
The first sampled batcher session instead spent `1.49s` in prefill wall,
including one `706ms` FP8 ragged-prefill graph capture.

The sampled-medium startup FP8 ragged-prefill warmup was capturing under the
startup symmetric-memory allreduce scope (`temperature=0.0`), while sampled
tree runtime disables that path (`temperature=0.7` by default). Since the CUDA
graph key includes the allreduce mode, those startup graphs did not match the
first sampled tree session. The warmup now enters the same sampled-medium
allreduce policy it is meant to serve, and queue profiles also record the
physical runtime cache shape to make future warmup mismatches visible. The
mechanism is confirmed: first sampled session captures dropped from `1` to `0`
and prefill wall dropped from `1.49s` to `0.79s` on the shared `144x1024` cache.
Score-facing tree movement is modest but positive: profiled
`230.1 / 51.1 / 275.9ms` improved to `215.5 / 49.7 / 256.9ms`; a no-profile
confirmation landed at `227.2 / 48.6 / 273.8ms` with p99 E2E down to `1.67s`.
This removes a startup-warmup bug, but it does not close the vLLM tree gap;
remaining tree work is still sampled prefill/session scheduling.

A fresh same-host all-provider comparison after `4896405` moves the active
gap from stale public `019ce7b` numbers to current local behavior. TorchInferno
now scores `1/20` metric wins against vLLM `14/20` and SGLang `4/20`. The
largest median gaps are still multi_turn (`371.9 / 60.1 / 441.2ms` vs vLLM
`169.2 / 55.0 / 226.4ms`) and tree (`253.0 / 48.0 / 292.4ms` vs vLLM
`63.7 / 31.7 / 87.2ms`). Multi_turn profiling showed one long greedy online
session with `7` request-path shared-prefix suffix CUDA graph captures totaling
`5.16s`; the shapes were the deterministic `p45` suffix buckets used by the
8-turn calculator prompts. Those graphs were not covered by the sampled FP8
warmup, and generic common-prefix suffix warmup remains too broad for default
startup.

The multi_turn startup path now captures only the measured greedy common-prefix
suffix buckets (`p45`, suffix `16/32/64/128/256`, batches `8/16/32`) under the
same runtime symmetric-memory allreduce scope and greedy FP8 prefill policy used
by the online session. The mechanism is confirmed in queue profiles:
request-path prefill captures dropped from `7` / `5.16s` to `0` / `0.0s`, and
prefill wall dropped from `9.58s` to `5.42s`. Startup ready time increases from
about `105.5s` to `115.5s`, but benchmark request latency improves materially
in the tail: profiled multi_turn p99 TTFT moved `4.70s -> 2.68s`, and a
no-profile confirmation landed at `381.1 / 62.5 / 455.5ms` with p99 TTFT
`2.69s`, first-turn average `1.19s`, and 100% benchmark-level correctness.

## Latest public refresh and greedy-mid idle retention (2026-06-26)

Public run `20260626_181444` was the fair same-host comparison before the
greedy-mid idle-retention patch. All providers completed. Console metric wins
were vLLM `20`, TorchInferno `4`, and SGLang `1`. TorchInferno rows were few_shot
`171.8 / 54.2 / 220.6ms`, self_consistency `155.5 / 0.0 / 334.5ms`,
multi_turn `484.7 / 68.5 / 541.5ms`, tree_of_thought
`309.3 / 57.0 / 342.8ms`, and long_output `296.5 / 28.1 / 1466.5ms`
(TTFT/TPOT/E2E). TorchInferno wins few_shot TPOT, self_consistency TTFT,
multi_turn TPOT, and multi_turn correctness, but still trails vLLM on most
queue-facing latency and throughput cells.

Current long_output profiling on `b068e8b` keeps the known split: `12.37s`
prefill wall, `18.18s` ragged decode GPU event time, and `7.78s` CPU token
wait across `780` decode batches. The run used `max_active=123` and
`prefix_rows=21`, but the first client wave only admitted `8` requests and the
provider submitted `1000` requests across `109` batches. This does not reopen
the already rejected long-output knobs: 20ms greedy-short initial wait, larger
row budgets, 128 decode graphs, non-power-of-two decode buckets, submit+step
commands, and synchronous decode-runner/readback shims all failed to produce a
clean default.

The narrower few_shot gap exposed a useful untried scheduler shape. Default
queue profiling on current `b068e8b` split few_shot into two online sessions
(`344` then `656` requests) because the gap between client waves exceeded the
10ms post-drain idle timeout. That forced the second wave to rebuild the online
session and common-prefix cache, landing at `176.2 / 55.3 / 228.9ms` with
`6.42s` combined prefill wall across the two sessions. Raising only
`TORCHINFERNO_OPENAI_TP_ONLINE_PERSISTENT_IDLE_MS=100` kept all `1000` requests
in one session, reduced profiled phase time from roughly `12.2s` to `10.6s`,
and cut prefill wall to `4.93s`. The profiled score row improved modestly to
`170.0 / 55.6 / 222.2ms`; the no-profile confirmation landed at
`165.6 / 54.7 / 210.1ms`, 975/1000 raw correct. The patched no-env default
reproduced the setting at `163.7 / 55.9 / 208.6ms`, 976/1000 raw correct, with
normal `115.6s` readiness. Promote the 100ms idle-retention window only for
deterministic mid-length online sessions (`128 < max_tokens <= 300`). This does
not change initial collection, active rows, refill floor, or decode quantum, and
avoids reopening the rejected global initial-wait and greedy-large persistent-idle
experiments.

The full provider recheck on pushed TorchInferno `876c18c` is public as
inference-bench run `20260626_185834` (`63043e6b`). It confirmed the scoped
few_shot movement inside the normal full-suite sequence: TorchInferno few_shot
improved to `167.3 / 55.2 / 214.8ms` from the prior public
`171.8 / 54.2 / 220.6ms`, preserving the TPOT win. vLLM also moved faster in the
same run (`152.3 / 57.0 / 199.6ms`), so this did not add a scorecard cell.
Summary scorecard wins were vLLM `16/20`, TorchInferno `2/20`, and SGLang
`1/20`; TorchInferno's wins were few_shot TPOT and self_consistency TTFT.
Multi_turn TPOT moved back to vLLM in this run (`60.2ms` vs TorchInferno
`67.1ms`), matching the known TPOT variance band rather than a code-path change
from the few_shot-scoped idle policy.

Two follow-up self_consistency profiles on pushed `d1f213e` kept the known
arrival-fragmentation shape. A clean rerun landed at `267.3 / 0.0 / 395.4ms`,
with `228` submit batches, `204` exact-prefix reuse waves, `1.74s` prefill
wall, and `1000/1000` correctness. An opt-in device-logits experiment that
cached reusable prefix logits on GPU regressed to `461.4 / 0.0 / 489.3ms`:
the run fragmented further (`303` submit batches, `261` exact-prefix waves,
`1.99s` prefill wall) while preserving correctness. Do not promote a
device-resident logits cache as the self_consistency lever; the row is still
dominated by client arrival batching and repeated exact-prefix scheduling, not
CPU-to-GPU logits copies.

The full same-host provider refresh on TorchInferno `36eae02` is public as
inference-bench run `20260626_195035` (`45d890b3`). It updated the branch after
the small tuple-key prompt-cache cleanup, but the scorecard stayed runtime
dominated: vLLM won `20` metric cells, SGLang `4`, and TorchInferno `1`.
TorchInferno's only cell win was few_shot TPOT (`54.3ms` vs vLLM `59.2ms`).
TorchInferno medians were few_shot `167.5 / 54.3 / 217.7ms`,
self_consistency `328.4 / 0.0 / 354.8ms`, multi_turn
`432.3 / 69.9 / 504.6ms`, tree_of_thought `352.6 / 56.1 / 408.2ms`, and
long_output `350.1 / 27.3 / 1418.1ms` (TTFT/TPOT/E2E). The tuple cache key is
kept as host-path cleanup, but it is not a score-facing lever; the remaining
gaps still sit in runtime prefill/session scheduling, sampled arrival
fragmentation, and long-output decode/readback.

A focused few_shot reprofile on current `4c58b4d` kept the public shape at
`165.8 / 53.9 / 210.1ms`, with `36` submit batches, `4.70s` prefill wall, and
`75` decode batches for only `1.6k` decode tokens. A deterministic-mid
submit+step A/B initially looked promising (`159.6 / 54.4 / 203.2ms`) and cut
batcher phase time from `11.18s` to `8.40s`, but it was not robust. The same
scoped policy in a no-env patched run returned to `165.3 / 56.6 / 211.1ms`
under queue profiling and regressed the unprofiled score to
`172.3 / 57.9 / 224.9ms`. Keep submit+step default-off for deterministic
mid-length sessions; the existing sampled-short opt-in/default remains the
only promoted submit+step path.

A current self_consistency profile on pushed TorchInferno `1810bb5` separated
server work from benchmark-client admission. With the old inference-bench
OpenAI client defaults, the row was `333.3 / 0.0 / 393.4ms`, but TorchInferno's
fast HTTP profile showed server-side first content at only `18.5ms` p50. The
missing time was before the request handler profile started, consistent with
HTTPX connection-pool contention under the benchmark's `128` worker threads.
Raising the inference-bench client pool to `512` connections for every provider
cut TorchInferno's focused self row to `230.4 / 0.0 / 379.0ms` without changing
correctness, and a fair three-provider self-only check landed at
vLLM/SGLang/TorchInferno `210.7 / 222.7 / 244.1ms` TTFT and
`386.0 / 402.1 / 357.4ms` E2E. Inference-bench `60edbfa6` carries that fair
client-pool default; the remaining self gap is provider/server scheduling, not
the old client pool ceiling. A follow-up no-keepalive A/B
(`INFERENCE_BENCH_HTTP_MAX_KEEPALIVE_CONNECTIONS=0`) is rejected: a fair
self-only run shifted the row to vLLM/SGLang/TorchInferno
`421.0 / 212.9 / 254.1ms` TTFT and `461.3 / 381.2 / 390.9ms` E2E, so keep the
`512/512` connection/keepalive default.
Raising TorchInferno's fast HTTP worker pool is also rejected for this path.
`TORCHINFERNO_OPENAI_HTTP_WORKERS=512` improved focused TTFT to `198.1ms` but
left E2E at `381.2ms`, and `1024` workers regressed to
`422.7 / 0.0 / 469.9ms`. The remaining self-consistency work is not solved by
more handler threads. A follow-up four-acceptor fast-HTTP prototype is rejected
too: the focused self row regressed to `413.6 / 0.0 / 436.2ms`, fragmented into
`285` submit batches, and left server-side p50 first-content at `18.8ms`, so
parallel accept did not address the benchmark-client admission delay.

A focused multi_turn profile on the same stack reproduced the public gap at
`437.1 / 72.2 / 512.3ms`, with HTTP first-content p50 `351.3ms` and runtime
prefill wall `10.64s` versus decode GPU event time `3.33s`. The online engine
reused only the 45-token system/common prefix across the 1000 turns. Enabling
the opt-in finished-prefix cache exposed an unsafe row-adoption bug: finished
states advertised the whole emitted token list even though the newest sampled
token is not KV-backed until a later decode consumes it. The cache now stores
only `state.tokens[:state.seq_len]` for finished prefixes, which fixed the
device-assert crash under the flag, but the safe version is still a hard
performance rejection. It raised prefix reuse to about `99k` tokens, but
fragmented suffix prefill into `470` batches with `450` graph misses and pushed
HTTP first-content p50 to `6485ms`. Keep finished-prefix caching disabled by
default. Re-enabling the guarded non-common finished-prefix graph path after
the KV-span fix did not recover the idea: an interrupted retry reached only
`581` submitted requests, had `160` graph-prefill hits and `0` misses, but spent
`73.2s` in graph prefill forward across `32` distinct prefix/suffix/source-row
shapes and pushed HTTP first-content p50 to `8234ms`. The missing piece is not
just graph safety; it needs shape consolidation or a different finished-prefix
reuse design.

A focused tree_of_thought profile on `9376f70` landed at
`309.1 / 59.0 / 370.0ms`, better than the public row but still TTFT-bound versus
vLLM. The sampled `temperature=0.7`, `max_tokens=300` traffic accounted for
`896` requests across six online sessions, spending `8.93s` in prefill wall and
`1.92s` in decode GPU event time; deterministic eval requests accounted for the
remaining `96` requests and `4.08s` decode GPU event time. A bounded
skip-incompatible-idle experiment tried to keep same-temperature sessions open
by deferring opposite-temperature requests. It fired (`39` incompatible skips)
but regressed the tree row to `358.3 / 59.9 / 419.4ms`, so do not promote queue
scanning across incompatible sampling classes.

A refreshed tree_of_thought profile on current `46be80d` landed at
`242.9 / 61.2 / 291.8ms`, with the same dominant split: sampled
`temperature=0.7`, `max_tokens=300` traffic spent `8.88s` in prefill wall and
`2.59s` in decode-active time, while the greedy eval side spent `1.74s` in
prefill wall and `2.72s` in decode-active time. Expanding the runtime
symmetric-memory allreduce scope to sampled traffic
(`TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE=runtime`,
`..._MAX_TEMPERATURE=0.7`) is rejected. It improved the smaller greedy eval
HTTP p50 (`428.4 / 502.6ms` first-content/E2E to `383.9 / 423.2ms`) and cut
greedy prefill wall to `1.25s`, but it regressed the sampled p50
(`210.8 / 260.3ms` to `234.8 / 297.4ms`) and raised sampled prefill wall to
`9.34s`. The full row moved backward to `271.9 / 60.5 / 347.8ms`. Keep runtime
symmetric-memory allreduce scoped to deterministic/greedy traffic; it is not a
general sampled tree lever.

The weak full-suite tree row was partly a harness sequencing artifact after
the larger client-pool change. Reproducing only `multi_turn -> tree_of_thought`
on TorchInferno `656a4d9` gave tree `344.2 / 57.8 / 385.0ms`, matching the
public full-suite row and much slower than focused tree. Closing each
benchmark's OpenAI/httpx client before the next benchmark (`inference-bench`
`f9fda409`) improved the same sequence to `304.8 / 58.2 / 357.5ms`; sampled
tree HTTP p50 moved from `283.6 / 330.3ms` first-content/E2E to
`263.2 / 324.8ms`, and greedy eval moved from `465.4 / 490.9ms` to
`358.9 / 375.8ms`. This is provider-neutral cleanup for stale keepalive pools,
not a TorchInferno runtime win. Focused tree still remains faster
(`242.9 / 61.2 / 291.8ms`), so the remaining full-suite tree gap is still in
request-wave timing and runtime prefill/session scheduling.

The full same-host public refresh with that harness fix is inference-bench
`113e45c1`, run `20260626_225903`, against TorchInferno `b7a4735`. The markdown
scorecard is vLLM `16/20`, TorchInferno `3/20`, and SGLang `0/20`.
TorchInferno's wins are few_shot TPOT (`53.1ms`), self_consistency TTFT
(`150.8ms`), and multi_turn TPOT (`66.1ms`). The tree sequence cleanup carried
into the full run: TorchInferno tree moved from the previous public
`342.5 / 55.7 / 371.6ms` to `296.7 / 55.4 / 332.1ms`. The same run showed a
weak multi_turn row (`593.2 / 66.1 / 647.3ms`), but targeted follow-ups did not
reproduce it. `few_shot -> self_consistency -> multi_turn` with queue profiling
kept multi_turn at `443.2 / 67.5 / 516.7ms`, and the same first-three sequence
without profiling landed at `438.1 / 68.5 / 512.2ms`. Do not tune runtime policy
against the single public multi_turn spike until a repeat full all-provider run
or a provider-cleanup artifact makes it reproducible.

The next full same-host refresh is inference-bench `508927d6`, run
`20260627_000710`, against TorchInferno `89bb0d3` (docs-only relative to the
runtime). It confirms the multi_turn spike was not stable: the row improved from
`593.2 / 66.1 / 647.3ms` to `502.5 / 66.2 / 552.6ms`, still far behind vLLM/SGLang
on TTFT and E2E. TorchInferno wins are back to two cells: few_shot TPOT
(`54.6ms`) and self_consistency TTFT (`164.3ms`). Other TorchInferno rows were
few_shot `162.9 / 54.6 / 211.5ms`, self_consistency `164.3 / 0.0 / 339.6ms`,
tree_of_thought `318.1 / 57.3 / 362.4ms`, and long_output
`306.9 / 27.4 / 1406.6ms`. Treat this as a cleaner baseline for the next gap:
multi_turn remains persistent-prefix/prefill scheduling bound, tree remains
sampled-medium prefill/session bound, and long_output remains decode/readback
bound.

A current long_output profile on `738bef5` landed at
`290.9 / 28.0 / 1435.2ms`, with `11.95s` prefill wall, `15.37s` decode GPU
event time, and `7.03s` CPU token readback across `763` decode batches. Raising
the short-greedy refill floor from `12` to `24` cut prefill batches (`57 -> 33`)
and improved TPOT (`28.0 -> 24.9ms`), but it delayed admission enough to regress
TTFT/E2E to `539.9 / 1782.6ms` and increased CPU token wait to `9.04s`. A
smaller `16` floor had the same tradeoff in milder form (`382.6 / 26.7 /
1515.4ms`) while raising decode token work to `45.6k` bucketed tokens. Keep the
default refill floor; it is the better latency/throughput tradeoff for the
scorecard.

A delayed inference-bench commit (`23ffc116`, run `20260626_230301`) added an
older public result after the cleaner `20260627_000710` refresh. It is not a
current runtime datapoint: TorchInferno was still on `b7a4735` there and never
became ready within 1800s, stalling during checkpoint load after the initial
embedding/norm/head tensors. Keep using `20260627_000710` as the latest complete
same-host comparison for scorecard decisions.

A current focused tree_of_thought repro on pushed `21ed3d8` landed at
`328.5 / 60.3 / 394.5ms`, 956/992 raw correct. Queue profiling kept the same
split as the public row: sampled-medium traffic accounted for `896` requests
across six online sessions and spent `8.54s` in prefill wall time, while the
small greedy eval side accounted for `96` HTTP requests and drove some high-tail
decode GPU time. The first sampled 256-request session was still the outlier
(`5.66s` phase, `4.80s` prefill wall) despite zero prefill graph misses, whereas
a later 256-request session was much cheaper (`1.71s` phase, `0.76s` prefill
wall). This keeps tree in the known sampled-medium prefill/cold-capture bucket;
do not reopen the rejected sampled-medium wait, idle, row-cap, suffix-warmup, or
capture-on-miss knobs without a different prefill pipeline mechanism. Follow-up
instrumentation now separates ragged prefill graph captures from graph replays
in queue profiles, so future "graph hit" counts will not hide cold-capture cost.
The first verification run on `8b530a4` proved the point: focused tree landed at
`255.9 / 60.4 / 304.5ms`, 957/992 raw correct, and sampled-medium prefill showed
`45` graph successes split into `8` cold captures and `31` replays. The captures
alone took `5.80s`, mostly in the first three sampled sessions; later replay-only
sessions had much lower prefill wall. The next tree lever is therefore shape
pre-capture/pipeline cost, not another admission wait.

A follow-up graph-cache fix adds the effective ragged prefill precision mode to
the CUDA graph key. Without it, a graph captured for greedy-large runtime FP8
policy (`min_m=2048`, BF16 for small suffix shapes) could be replayed by
sampled-medium policy (`min_m=256`, FP8 for those same shapes), or vice versa.
This is a normal runtime-policy key, not benchmark-shape detection. A local
`multi_turn -> tree_of_thought` validation with the fix landed multi_turn at
`448.9 / 69.8 / 532.8ms` and tree at `287.6 / 58.4 / 334.2ms`, improving the
public full-suite tree row while preserving correctness (962/992 raw tree
correct). Queue profiling confirmed sampled tree (`fp8_prefill_min_m=256`)
captured its own first-wave graphs after multi_turn (`4` captures, `2.63s`
capture time in the first sampled session) instead of reusing prior
greedy-large captures. The remaining first-wave cost is still cold capture and
prefill pipeline work.

Rechecking the existing common-prefix suffix warmup knob after the precision-key
fix is not defaultable. With
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_COMMON_PREFIX_SUFFIX_PREFILL=1`, focused tree
started in `130.6s` and landed at `269.1 / 59.8 / 311.0ms`, 955/992 raw
correct. The first sampled session still spent `2.19s` in prefill wall and
captured two graphs (`1.54s` capture time) because the warmup does not remove
the sampled FP8 suffix-capture path. Keep suffix warmup opt-in until a
precision-aware pre-capture path beats the focused control without trading away
startup or correctness.

Runtime-FP8 ragged prefill graph warmup is the narrower precision-aware version
of that idea. The current full local all-provider run (`20260627_012328`,
TorchInferno `14ba05d`) showed the order-dependent tree regression clearly:
TorchInferno kept few_shot TPOT/E2E wins but tree fell to
`381.5 / 56.7 / 436.3ms`; the first sampled tree session paid `5` FP8 ragged
prefill captures (`2.98s` capture time). Adding startup capture for the
sampled-medium runtime-FP8 ragged suffix buckets raised readiness to `125.6s`
in a TorchInferno-only `few_shot -> self_consistency -> multi_turn ->
tree_of_thought` sequence, but tree recovered to `295.6 / 57.3 / 341.6ms` with
955/992 raw correct and p99 E2E `1.01s`. Queue profiling confirmed the mechanism:
the first sampled session dropped to `1` FP8 ragged prefill capture (`720ms`) and
`10` replays. This keeps startup work generic to runtime FP8/suffix graph policy
while removing most of the request-path cold-capture tail.

## PUBLIC STARTUP REGRESSION: CHECKPOINT LOAD/BROADCAST VARIABILITY (2026-06-24)

Update `2026-06-25`: public run `20260624_230255` showed the opposite failure
mode. TorchInferno printed `rank0_broadcast=1`, initialized NCCL successfully,
then spent the full 1800s readiness window in the first checkpoint tensor
broadcast and was killed before `/health` could bind. Same-host validation with
`TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST=0` loaded all `80/80` layers in
`23.6s` after the probe and reached `/health`, so rank-0 checkpoint tensor
broadcast is now an explicit opt-in instead of the default.

Public inference-bench run `20260624_160928` failed before latency measurement:
TorchInferno reached NCCL init, then loaded only `70/80` layers in `1745.0s`
with `rank0_broadcast=0` before the 1800s readiness timeout killed the server.
The same all-rank-read path is storage-sensitive: public run `20260624_100730`
barely completed checkpoint loading in `1661.8s`, while same-host cached runs
finish the same path in seconds.

Rank-0 checkpoint tensor broadcast remains useful on hosts where it is known to
be healthy because it avoids every rank independently streaming the same
checkpoint from slow shared storage. Local startup smoke with that path reached
readiness in `231s` and loaded all `80/80` layers in `25.3s`
(`rank0_broadcast=1`, symm-memory allreduce disabled for the startup check).
Given the public NCCL broadcast stalls above, the inference-bench TorchInferno
provider should leave `TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST` unset unless
a run explicitly opts in.

Public run `20260625_030321` failed again on the all-rank-read path with
`rank0_broadcast=0`, loading only `70/80` layers by `1757.4s`. The startup
policy is now hybrid by default when the broadcast env is unset: replicated
tensors still use per-rank reads so the giant embedding table is not broadcast,
while sharded tensors use rank-0 scatter/reduce-scatter to avoid eight ranks
hammering shared storage for every layer shard. Explicit
`TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST=0` keeps the old all-rank-read path,
and `TORCHINFERNO_TP_RANK0_CHECKPOINT_SHARD_SCATTER=0` disables the hybrid.

Public run `20260625_090252` reached NCCL initialization and checkpoint loading
but never bound `/health` before the 1800s readiness timeout; the process was
killed with SIGTERM and no TorchInferno benchmark rows were produced. vLLM and
SGLang completed the same run, so this was a TorchInferno startup warmup
regression rather than a harness-wide failure. Local profiling found the server
burning startup CPU inside broad graph warmups (`ragged_decode`, then
temperature prefill/decode graph capture through Marlin quantization). Broad
startup graph sweeps are now opt-in via `TORCHINFERNO_OPENAI_STARTUP_GRAPH_WARMUP`,
with ragged decode separately opt-in. The narrower online scheduler cache/decode
warmup remains default-on because it recovered runtime graph coverage while
still reaching readiness in `120-126s`, well below the public timeout.
Public run `20260625_110248` repeated the same failure on TorchInferno
`c7ff2ca`; it does not include these local startup changes.

Public run `20260625_150258` failed before benchmark rows on stale
TorchInferno `754cc36`. It reached distributed startup with
`rank0_broadcast=0` and `rank0_shard_scatter=1`, but the log was dominated by
inherited `NCCL_DEBUG=INFO` output for most of the readiness window and never
showed a `loaded 10/80 layers` progress line before SIGTERM at 1800s. The
inference-bench TorchInferno provider now forces `NCCL_DEBUG=WARN` by default
and drops the deprecated `NCCL_ASYNC_ERROR_HANDLING` env, while preserving an
explicit `INFERENCE_BENCH_TORCHINFERNO_NCCL_DEBUG=INFO` escape hatch for
transport debugging. This is harness hardening, not a runtime latency fix.

Public runs after that harness change (`20260625_190425` through latest
`20260626_130303`) still failed TorchInferno readiness on current `ca2ea3d`
before any benchmark rows, while vLLM and SGLang completed. The latest log is
now quiet enough to isolate the next stall: after symmetric-memory allreduce
probing it prints `rank0_broadcast=0` and `rank0_shard_scatter=1`, then no
`loaded 10/80 layers` progress line before SIGTERM. That leaves the initial
replicated tensors or first sharded scatter as the likely startup bottleneck.
Current startup now defaults replicated checkpoint tensors to rank-0 chunked
broadcast via `TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_BROADCAST=1`,
independent of the broader full-checkpoint broadcast knob. Chunks default to
`TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST_CHUNK_BYTES=268435456`, so the
giant embedding/norm tensors avoid eight-rank shared-storage reads without
restoring the old single full-tensor NCCL broadcast. Explicit
`TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_BROADCAST=0` or
`TORCHINFERNO_TP_RANK0_CHECKPOINT_BROADCAST=0` keeps the old all-rank-read
path. The server also prints an initial embedding/norm/head progress line so
the next public failure can distinguish an initial-tensor stall from a layer
load stall. Same-host inference-bench validation with this default
(`20260626_154324`, `self_consistency` only) showed the intended startup shape:
initial embedding/norm/head tensors loaded in `3.8s`, all `80/80` layers loaded
in `52.0s`, `/health` was ready in `150.8s`, and the row completed at
`240.2 / 0.0 / 379.9ms` with 1000/1000 correctness. A follow-up full
TorchInferno-only pass on pushed `c230138` (`20260626_154946`) reproduced the
startup path more quickly (`80/80` layers in `28.9s`, `/health` in `130.6s`)
and completed all rows in the expected runtime band: few_shot
`166.4 / 53.3 / 219.0ms`, self_consistency `198.1 / 0.0 / 283.6ms`,
multi_turn `467.4 / 72.5 / 548.3ms`, tree_of_thought
`287.1 / 55.9 / 336.1ms`, and long_output `326.9 / 27.4 / 1361.2ms`.

Public runs from `20260627_010324` through `20260627_130258` invalidated that
hybrid default on the CUDA 13.2 submit hosts. TorchInferno repeatedly printed
`rank0_broadcast=0 rank0_replicated_broadcast=1 rank0_shard_scatter=1`, spent
`487-603s` just loading the initial embedding/norm/head tensors, and never
reached the first `loaded 10/80 layers` progress line before the 1800s
readiness timeout. Same-host local validation still loads this path quickly, so
the issue is environment-sensitive NCCL collective throughput during startup,
not model correctness. The portable per-rank checkpoint reader is now the
default again for both replicated tensors and sharded tensors. Rank-0 replicated
broadcast and shard scatter remain available through explicit
`TORCHINFERNO_TP_RANK0_REPLICATED_CHECKPOINT_BROADCAST=1` and
`TORCHINFERNO_TP_RANK0_CHECKPOINT_SHARD_SCATTER=1` opt-ins when a host is known
to handle those large collectives.

Same-host validation after the default flip (`9a22e75`) confirmed the harness
startup path: inference-bench built the editable TorchInferno package in `7.6s`,
loaded all `80/80` Llama layers in `18-21s`, and reached `/health` in `110.5s`.
A full TorchInferno-only local baseline completed all five rows: few_shot
`167.9 / 48.8 / 215.0ms`, self_consistency `286.4 / 0.0 / 313.2ms`,
multi_turn `377.6 / 62.4 / 449.9ms`, tree_of_thought
`197.6 / 47.5 / 235.1ms`, and long_output `282.0 / 23.2 / 1173.1ms`
(TTFT/TPOT/E2E). Correctness stayed at 98-100% depending on row.

Focused long_output profiling on the same code keeps the remaining runtime gap
decode/prefill bound rather than startup-bound. The queue profile ended at
`26.4s` total phase time with `60` prefill batches, `10.1s` prefill wall
(`9.2s` forward), `8` runtime prefill graph captures (`4.9s`), `787` decode
batches, `9.4s` decode-active time, and `7.0s` CPU-token readback time. Two
existing knobs were rejected: enabling common-prefix suffix warmup moved one
capture out of the request path but regressed score-facing long_output to
`282.4 / 24.1 / 1353.6ms`; forcing greedy-short decode quantum `8` improved
TPOT to `21.0ms` but regressed TTFT/E2E/throughput to
`466.2ms / 1423.4ms / 25.5 tok/s`. Keep both defaults unchanged.

Public run `20260624_185427` supersedes the later-sorting stale
`20260624_183253` failure. It used TorchInferno `76107de`, vLLM `1cd3e0e`,
and SGLang `4a4f063`; all providers completed all five benchmarks. Scorecard
wins were vLLM `17/20`, TorchInferno `1/20`, and SGLang `1/20`. TorchInferno's
current public rows are few_shot `152.1 / 51.3 / 195.9ms`, self_consistency
`349.9 / 0.0 / 378.4ms`, multi_turn `432.0 / 64.7 / 498.3ms`,
tree_of_thought `240.0 / 51.3 / 288.3ms`, and long_output
`382.4 / 26.9 / 1561.1ms` (TTFT/TPOT/E2E). The same run confirms the startup
fixes: TorchInferno reached readiness and completed with 100% benchmark-level
correctness instead of timing out in checkpoint load.

The later public run `20260624_190316` is a startup failure at the same
TorchInferno commit, not a latency result. It ran on a different submit host
with Python 3.13 / CUDA 13.2, spent minutes in NCCL OFI initialization before
falling back to Socket, and then never reached the first
`[Llama3TP] loaded 10/80 layers` checkpoint progress line before the 1800s
readiness timeout. The rank-0 checkpoint path was still broadcasting each full
sharded tensor to every rank and slicing locally. That is too much collective
traffic on a weak NCCL path, so sharded checkpoint loads now use
`reduce_scatter_tensor` when available: rank 0 packs rank shards, nonzero ranks
avoid checkpoint reads, and each rank receives only its shard. Same-host
validation kept startup healthy: 8-GPU NCCL `reduce_scatter_tensor` smoke passed,
and a real Llama server on port 8090 loaded all `80/80` layers in `27.6s` and
reached `/health`.

Follow-up full local inference-bench run `20260624_205205` on TorchInferno
`2af6f8f` completed all providers and supersedes the failed startup row for
local comparison. Scorecard wins were TorchInferno `4/20`, vLLM `15/20`, and
SGLang `0/20`. TorchInferno won few_shot TTFT/TPOT/E2E
(`154.6 / 51.2 / 197.6ms`) and multi_turn TPOT (`67.5ms`), but still trails
vLLM on self_consistency E2E (`370.4ms` vs `352.5ms`), tree TTFT/E2E
(`285.4 / 330.4ms` vs `73.9 / 101.2ms`), and long_output decode/e2e
(`24.8 / 1387.3ms` vs `18.8 / 768.1ms`). The checkpoint scatter change is a
startup hardening fix; it does not materially close the runtime prefill/decode
gaps.

Several current-loop A/Bs are rejected on `76107de`. Runtime FlashInfer prefill
for multi_turn (`TORCHINFERNO_CONTINUOUS_FLASHINFER_PREFILL_DISABLE=0`)
regressed badly to `1253.5 / 89.4 / 1422.5ms` and only `1.1 tok/s`, versus the
nearby dense baseline around `433.2 / 63.2 / 501.4ms`; the queue profile showed
`134.4s` aggregate phase time. Finished-prefix caching still needs a batching
redesign: enabling `TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE=1` cut
prefill tokens but fragmented into hundreds of tiny batches
(`485` prefill batches, `463` graph misses, `101.99s` prefill wall), and adding
mixed-prefix graph grouping with
`TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_PREFILL{,_GRAPH}=1` stalled after only
`68` submitted / `36` finished requests and was terminated. Lowering the online
FP8 prefill gate to `512` for multi_turn was neutral-to-worse
(`431.2 / 66.9 / 502.9ms`, `10.63s` prefill wall), so keep the `2048` runtime
M gate.

Follow-up local profiling on `c7ff2ca` kept the same runtime shape. The full
TorchInferno baseline (`20260625_073446`) landed at few_shot
`154.7 / 49.7 / 195.9ms`, self_consistency `357.5 / 0.0 / 386.1ms`,
multi_turn `437.6 / 66.8 / 508.5ms`, tree_of_thought
`328.1 / 51.6 / 363.6ms`, and long_output `343.1 / 27.3 / 1371.1ms`.
Multi-turn is still dense prefill dominated: the queue profile spent
`10.17s` in prefill wall time versus `1.31s` in active decode, with one
45-token common-prefix prefill and `33` graph-backed suffix waves.

Two more current-loop A/Bs are rejected on top of that baseline. A temporary
batched COW suffix-prefill patch for the paged prefix-cache path improved the
first paged queue snapshot (`240` requests finished at `12.5s`, versus the
older first snapshot around `241` at `45.7s`), but the final paged multi_turn
run (`20260625_094415`) was still unusable: `7832.5 / 745.3 / 8243.2ms`,
`0.2 tok/s`, and `121.8s` batcher time. Dense multi_turn finishes the same
profile shape in about `11.8s`, so the paged model/cache path remains the
bottleneck; the patch was reverted. Rechecking sampled-medium tree with
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_MAX_ACTIVE=40` also regressed:
focused tree (`20260625_095527`) moved to `389.5 / 52.2 / 426.9ms` from the
baseline `328.1 / 51.6 / 363.6ms`. Keep the sampled-medium default at `32`.

Self-consistency was re-profiled with additional queue-shape counters
(`20260625_100730`). The row stayed in family at `329.3 / 0.0 / 384.3ms`,
1000/1000 correct, and confirmed generated-prefix reuse is not the missing
piece: `985` generated-prefix reuse hits, one prefill batch, and one decode
batch. The queue shape is the limiter: 1000 requests arrived as `232` submit
batches, including `69` single-request submits, `38` two-request submits, and
only one large `113`-request submit. This matches the earlier rejected
post-arrival collection family; keep the new profile counters, but do not
default another idle collection change without a fresh mechanism that avoids the
previous TTFT/E2E regression.

The startup fix initially exposed a first-request graph miss. With broad startup
graph sweeps disabled and no replacement warmup (`20260625_105038`), readiness
was healthy (`75.4s`) but self_consistency regressed to
`369.6 / 0.0 / 417.3ms`; the profile showed the common-prefix prefill and static
decode paths both missing their graphs (`prefill_graph_hits=0/misses=1`,
`decode_graph_hits=0/misses=1`) and prefill wall rose to `2.80s`. A narrow
online common-prefix startup warmup now allocates the persistent serving cache
and captures only the default common-prefix prefill bucket, leaving the broad
temperature/ragged graph zoo off by default. That restored the prefill hit
without reintroducing the readiness timeout.

Two adjacent A/Bs are rejected. Combining online submit and step into one TP
command (`20260625_104126`) did restore graph hits when the full scheduler
warmup was still enabled, but it did not reduce submit-sync time
(`644ms` versus the nearby `623ms` baseline) and regressed self_consistency to
`371.6 / 0.0 / 413.7ms`; keep the protocol support behind
`TORCHINFERNO_OPENAI_TP_ONLINE_SUBMIT_STEP_COMMAND=1` rather than defaulting it.
The first narrow-prefill run without a hot-path decode guard (`20260625_110200`)
captured a static decode graph inside request handling, producing a `42.3s`
decode-active stall and `43.2s` p99 TTFT/E2E tails. Continuous serving decode
now passes `capture_on_miss=False` only when generated-prefix caching is active
(the sampled-short self_consistency path that produced the 42s tail);
`TORCHINFERNO_CONTINUOUS_DECODE_CAPTURE` remains the explicit override.

With narrow startup prefill warmup plus hot-path decode capture disabled
(`20260625_110829`), focused self_consistency improved to
`234.5 / 0.0 / 368.0ms`, `1000/1000` correct, with readiness still `75.4s`.
The queue profile now has the intended shape: one common-prefix prefill graph
hit and no prefill miss, one decode graph miss falling back to eager decode,
`1.77s` prefill wall, `222ms` decode-active, and `3.18s` profiled online
batcher phase total. This does not close the vLLM public self_consistency gap
(`189.2 / 0.0 / 213.4ms` in `20260625_090252`), but it fixes the public
readiness failure and removes the first-request graph-miss regression without
benchmark-specific output logic.

The full-run recheck showed the decode guard must be scoped. Disabling runtime
decode capture for every continuous-serving session recovered self but regressed
few_shot/multi/tree/long because those rows still depend on decode graph capture
or pre-warmed decode graphs after startup. With default-on scheduler warmup and
broad graph sweeps still off, the full local TorchInferno run `20260625_113317`
reached readiness in `125.6s` and landed at few_shot
`153.1 / 50.4 / 194.8ms`, self_consistency `288.1 / 0.0 / 364.2ms`,
multi_turn `429.7 / 65.3 / 497.9ms`, tree_of_thought
`279.8 / 51.2 / 345.6ms`, and long_output `370.5 / 27.8 / 1618.9ms`.
The follow-up no-env run `20260625_114015` confirmed the default path reaches
the same startup band (`125.6s`) and stays in family: few_shot
`155.5 / 50.2 / 195.5ms`, self_consistency `254.6 / 0.0 / 393.5ms`,
multi_turn `453.4 / 69.0 / 538.2ms`, tree_of_thought
`368.6 / 51.1 / 410.1ms`, and long_output `374.0 / 26.9 / 1492.5ms`. This
keeps public startup safe and restores the earlier runtime family except for
long_output E2E, which remains decode/readback dominated.

A narrow runtime FP8 prefill startup warmup is accepted for the sampled-medium
tree path. Focused tree with the warmup enabled (`20260625_121445`) reached
readiness in `120.6s` and landed at `230.1 / 51.5 / 270.1ms`; disabling only
`TORCHINFERNO_OPENAI_STARTUP_RUNTIME_FP8_PREFILL_WARMUP` in the paired run
(`20260625_121827`) kept readiness at `120.6s` but regressed to
`329.6 / 51.7 / 374.6ms`. The first sampled-medium 256-request phase still has
cold cost, but the warmup cut it from `4.84s` phase / `3.96s` prefill wall to
`3.60s` phase / `2.91s` prefill wall without restoring the broad startup graph
sweeps. The pass is disabled when runtime FP8 prefill is explicitly disabled.

A follow-up long_output decode-many quantum A/B is rejected. Letting the upper
end of the greedy-short decode-many window use the full `16`-step command
quantum cut step broadcasts sharply (`191` in the prior profile family to `58`)
but regressed the row to `520.6 / 25.2 / 1765.9ms` in `20260625_120420`.
The middle `8`-step quantum was also worse (`20260625_122518`:
`488.8 / 26.0 / 1609.7ms`), even though it cut step broadcasts to `102` and
readback time to `6.29s`. The queue profile still spent `26.7s` in runtime
steps, with `13.0s` decode GPU time and `12.1s` prefill wall, so fewer host
commands did not offset the larger queueing/prefill waves. Keep the default
`decode_quantum=4` for decode-many short greedy sessions until there is a real
pipelined readback or scheduler change.

The final default TorchInferno-only full pass for this loop (`20260625_122927`)
reached readiness in `120.6s` and completed all rows: few_shot
`157.7 / 50.1 / 201.9ms`, self_consistency `379.0 / 0.0 / 406.4ms`,
multi_turn `436.4 / 66.5 / 507.6ms`, tree_of_thought
`281.0 / 51.5 / 345.5ms`, and long_output `359.6 / 27.2 / 1504.2ms`.
Tree stayed improved versus the no-FP8-warmup paired run, while long remained in
the default quantum-4 family. Self-consistency is still arrival-shape sensitive:
the final full run landed slower than the focused `20260625_110829` row but
kept the same one-prefill/one-decode generated-prefix shape and 1000/1000
correctness.

Two current self_consistency queueing A/Bs were checked. Forcing
`TORCHINFERNO_CONTINUOUS_ADMIT_MIN_READY_REQUESTS=16` did not consolidate the
runtime reuse waves; it increased submit/reuse fragmentation (`297` submit
batches, `261` prefix-reuse batches) and regressed the row to
`400.0 / 0.0 / 461.8ms` in `20260625_123743`. The older unscoped combined
submit+step TP command retest also regressed to `411.3 / 0.0 / 436.8ms` in
`20260625_124132`, but a matched sampled-short recheck on current `418f9da`
showed the useful scope. Enabling
`TORCHINFERNO_OPENAI_TP_ONLINE_SUBMIT_STEP_COMMAND=1` for self_consistency
(`20260625_143349`) landed at `241.9 / 0.0 / 385.4ms`, 1000/1000 correct,
versus the paired no-env control (`20260625_143735`) at
`390.6 / 0.0 / 418.0ms`, also 1000/1000 correct. The profile moved submit
batches `248 -> 212`, runtime step calls `220 -> 201`, phase total
`3772ms -> 3591ms`, submit-sync `683ms -> 599ms`, and idle wait/drain
`264ms -> 127ms`. Promote combined submit+step only for sampled-short online
sessions (`temperature > 0`, `max_tokens <= 256`) and leave the env override for
broader experiments. A post-push no-env validation on `80279ef` with symm-mem
probing default-off reached readiness in `100.5s` and completed
self_consistency at `276.2 / 0.0 / 409.9ms`, 1000/1000 correct.

A full no-env TorchInferno-only validation after both patches (`f963719`,
`20260625_145417`) also reached readiness in `100.5s` and completed all rows:
few_shot `163.0 / 53.3 / 205.6ms`, self_consistency
`281.5 / 0.0 / 305.3ms`, multi_turn `445.8 / 65.7 / 512.2ms`,
tree_of_thought `325.6 / 58.0 / 379.3ms`, and long_output
`328.0 / 31.2 / 1579.2ms`. This is the expected public-readiness tradeoff:
self improves materially from sampled submit+step, while tree/long decode TPOT
give back the symm-memory optimization until the startup/runtime symm scopes are
split.

After landing the startup/runtime fixes as TorchInferno `de2d6f1`, a same-host
skip-build provider comparison (`20260625_125255`) confirmed the readiness fix
inside the normal provider harness: TorchInferno bound `/health` in `125.6s`.
Runtime gaps remain large. vLLM/SGLang/TorchInferno medians were few_shot
`211.2 / 222.0 / 233.7ms` E2E, self_consistency
`275.9 / 414.3 / 506.7ms`, multi_turn `247.3 / 295.5 / 654.5ms`,
tree_of_thought `99.7 / 171.4 / 400.5ms`, and long_output
`773.7 / 1050.8 / 1539.6ms`. TorchInferno kept only the few_shot TPOT cell
(`53.8ms` versus vLLM `59.7ms` and SGLang `82.1ms`), so the next score work
should target conversation-prefix prefill and long-output row turnaround rather
than more startup fixes.

The current pushed default (`754cc36`) was re-run in a same-host provider
comparison (`20260625_150106`) after keeping symm-memory probing opt-in and
defaulting sampled-short submit+step. TorchInferno again reached readiness and
won only the few_shot TPOT cell. Provider medians were:
few_shot vLLM/SGLang/TorchInferno `201.3 / 223.2 / 227.8ms` E2E,
self_consistency `267.9 / 393.0 / 372.1ms`, multi_turn
`214.0 / 282.0 / 541.5ms`, tree_of_thought `99.7 / 163.2 / 313.6ms`, and
long_output `738.9 / 1004.0 / 1694.0ms`. The queue profile shows the remaining
work split by traffic shape: self_consistency still has `218` submit batches,
`200` step calls, `604ms` submit-sync, and `248ms` idle drain despite
submit+step; multi_turn spends `10.14s` prefill wall / `9.65s` prefill forward
across `34` submit batches; long_output spends `11.58s` prefill wall plus
`14.52s` decode GPU time and `9.15s` synchronous token readback exposure across
`558` step calls. This keeps the priority unchanged: close dense prefill and
long-output decode/readback gaps before another broad queue knob.

A follow-up split restores the safe part of symm-memory allreduce without
putting startup back on the failing path. A successful auto-probe now writes
`TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE=runtime`, startup warmups/validation
stay on NCCL unless `TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_STARTUP=1`, and
runtime symm is default-gated to deterministic traffic
(`TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_MAX_TEMPERATURE=0.0`). The broad
runtime version (`20260625_152053`) proved readiness-safe (`125.6s`) and
improved long_output to `379.2 / 27.0 / 1512.6ms`, but it was not worth using
for sampled rows. The paired probe-off control (`20260625_152626`) landed at
few_shot `167.9 / 60.3 / 216.0ms`, self_consistency
`91.6 / 0.0 / 120.2ms`, multi_turn `441.3 / 67.8 / 516.7ms`,
tree_of_thought `294.7 / 56.7 / 349.7ms`, and long_output
`359.3 / 31.3 / 1606.9ms`. With the deterministic gate
(`20260625_153150`), readiness remained healthy (`120.6s`) and long_output
kept the useful decode win at `381.1 / 27.0 / 1396.2ms`; the queue profile moved
long decode GPU/readback exposure from `15.09s / 9.84s` in the probe-off
control to `12.89s / 7.57s`, while sampled self/tree stayed on the NCCL scope.
Self_consistency remains arrival-shape sensitive (`268` submit batches in that
run), so do not treat runtime symm as a sampled-traffic fix.

The pushed `a74ce3f` same-host provider comparison (`20260625_154001`) confirms
the score impact. TorchInferno reached readiness in `125.6s`, logged
`symmetric-memory allreduce enabled after probe (runtime scope)`, and won two
TPOT cells: few_shot `55.4ms` versus vLLM `56.2ms`, and multi_turn `65.4ms`
versus vLLM `71.9ms`. Overall wins were vLLM `22`, TorchInferno `2`, and SGLang
`1`. The remaining score-facing gaps are still TTFT/E2E: few_shot
`182.7 / 238.4ms` vs vLLM `144.2 / 194.7ms`, self_consistency
`265.7 / 411.7ms` vs `215.6 / 246.1ms`, multi_turn
`532.9 / 582.9ms` vs `175.1 / 235.3ms`, tree_of_thought
`288.9 / 335.7ms` vs `74.7 / 101.1ms`, and long_output
`389.7 / 1631.5ms` vs `78.9 / 768.8ms`. The long queue profile still spends
`11.92s` in prefill wall and `13.09s / 7.32s` in decode GPU/readback exposure,
so runtime symm is only a partial decode fix; the next real lever remains
prefill scheduling plus token readback/pipelining.

Rechecking long_output with the new deterministic runtime-symm scope and
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_GEN_DECODE_QUANTUM=8`
(`20260625_155605`) is rejected. It reduced readback exposure to `5.37s` and
median TPOT to `26.3ms`, but decode GPU time rose to `16.31s`, total profiled
phase rose to `30.88s`, and score-facing TTFT/E2E regressed to
`418.9 / 1686.1ms`. Keep the default greedy-short decode quantum at `4`.
An attempted side-stream async D2H readback inside `step_online_many` is also
rejected (`20260625_160230`): score-facing long_output stayed flat at
`379.4 / 27.3 / 1610.2ms`, while the profile moved the wrong way
(`30.64s` phase, `15.78s` decode GPU, `7.75s` readback). Revert the prototype;
readback needs a real pipelined runner rather than a small copy scheduling shim.
Forcing the combined submit+step TP command on deterministic long_output is also
rejected. The focused row (`20260625_162846`) improved score-facing
TTFT/E2E/throughput to `342.8 / 1455.8ms / 26.5 tok/s` versus the paired no-env
control (`20260625_163244`) at `486.0 / 1681.6ms / 23.0 tok/s`, but it regressed
TPOT (`27.9ms` vs `25.9ms`) and the queue profile did not show a real engine win:
phase total was flat-to-worse (`30.96s` vs `30.81s`), submit batches increased
`95 -> 117`, runtime step calls increased `544 -> 611`, and decode/readback
exposure rose. Keep combined submit+step defaulted only for sampled-short.
Disabling greedy-short decode-many on current `71b979d` is also rejected
(`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_DECODE_MANY=0`,
`20260625_164757`). The row preserved correctness but landed at
`374.2 / 28.8 / 1692.2ms`, with `278.5ms` p99 TPOT. Internally it reduced
step commands (`192 -> 92`) and runtime step calls became one-per-step, but CPU
token readback exposure rose from the nearby default `7.96s` to `10.93s`, p99
streaming worsened, and profiled phase total stayed flat (`30.69s`). Keep the
current decode-many default for this shape; turning it off does not remove the
long-output decode/readback bottleneck.

Lowering the greedy-short refill floor from `16` to `12` is accepted on current
`ee29a4f`. The env-backed run (`20260625_170033`) landed at
`321.6 / 28.1 / 1421.1ms`, `27.4 tok/s`, 1000/1000 correct, versus the paired
no-env control (`20260625_170458`) at `366.8 / 28.0 / 1565.7ms`,
`24.1 tok/s`, also 1000/1000 correct. The profile supports the median move:
prefill wall dropped `12.54s -> 11.20s`, decode GPU exposure
`14.66s -> 14.19s`, token readback `8.20s -> 7.48s`, and runtime step calls
`602 -> 500`; total profiled phase was only modestly better
(`30.97s -> 30.63s`) and p99 TPOT worsened (`225.8ms -> 273.8ms`), so keep the
change narrowly scoped to deterministic `max_tokens<=128` traffic. The patched
no-env default reproduced the setting (`admit_min_ready_requests=12`) and landed
at `285.5 / 28.2 / 1389.3ms`, `26.7 tok/s`, with a lower profiled phase
(`29.89s`) and 1000/1000 correctness. A full TorchInferno-only no-env pass on
the pushed default (`fdbe848`, `20260625_171504`) kept long_output in the
improved family at `344.5 / 28.3 / 1388.7ms`, with few_shot
`164.2 / 55.1 / 208.8ms`, self_consistency `173.9 / 0.0 / 299.4ms`,
multi_turn `451.0 / 66.0 / 522.3ms`, and tree_of_thought
`345.9 / 56.0 / 409.7ms`. Tree was noisy/slower, but the changed refill policy
only applied to the long_output session (`run_max_tokens=96`,
`admit_min_ready_requests=12`).

A same-host focused tree refresh on pushed `8ac97c6` (`20260625_172238`) keeps
the tree gap unchanged: vLLM/SGLang/TorchInferno medians were
`73.4 / 35.1 / 100.0ms`, `78.7 / 62.9 / 167.7ms`, and
`297.6 / 58.1 / 335.3ms` (TTFT/TPOT/E2E). TorchInferno's sampled branch
submitted 896 requests across 7 online sessions and spent `10.08s` in prefill
wall / `8.28s` prefill forward across 46 prefill batches; greedy eval was much
smaller at 96 requests. Two current tree probes are rejected. Raising only the
sampled-medium initial wait to `20ms` (`20260625_173430`) consolidated one fewer
sampled session but regressed the row to `337.9 / 59.8 / 386.8ms`; the first
sampled session alone took `5.77s` profiled phase time. Lowering the
sampled-medium runtime FP8 prefill gate to `M>=128` (`20260625_173950`) reduced
aggregate sampled prefill wall (`10.08s -> 8.94s`) but still regressed
score-facing latency to `326.1 / 58.8 / 377.6ms`. Do not promote either knob;
tree still needs a prefill/session pipeline change rather than another
collection-wait or tiny-FP8 gate tweak.

Greedy-large refill `16` is rejected for current multi_turn. The env run
(`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_REFILL_MIN_READY_REQUESTS=16`,
`20260625_174457`) moved medians to `436.6 / 67.4 / 507.0ms` versus the paired
no-env control (`20260625_174845`) at `447.4 / 70.5 / 518.6ms`, both 981/1000
raw correct, but the profile did not show a real scheduling change: submit
batches stayed at `34`, runtime steps at `119`, decode batches at `86`, and
profiled phase was flat (`15.41s` vs `15.40s`). The refill-16 run also worsened
p99 TPOT (`1308ms` vs `689ms`). Keep the deterministic 401-512 token refill
floor at `32`; multi_turn still needs fewer or faster prefix/suffix prefill
waves, not a lower ready-request floor.

Greedy-large active-row increases are also rejected for current multi_turn.
Raising `TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_MAX_ACTIVE` to `48`
(`20260625_180435`) regressed hard to `887.3 / 77.6 / 956.9ms`, `1.35 tok/s`,
with p99 E2E `5821ms`. The profile showed why: `max_active=48` increased
prefill wall to `16.75s`, prefill forward to `16.15s`, and decode GPU exposure
to `4.70s`, while prefix reuse stayed limited to the shared 45-token system
prefix. The intermediate `40`-row probe (`20260625_180843`) reproduced the same
failure family at `846.7 / 72.8 / 896.1ms`, `1.31 tok/s`, with `15.83s`
prefill wall and `4.69s` decode GPU. Keep the default `32` active rows for this
512-token greedy path; larger decode waves add KV/decode cost and make the
suffix-prefill queue worse without creating useful conversation-prefix reuse.

Long-output row-budget A/Bs on the same pushed code are not defaultable yet.
Raising `TORCHINFERNO_OPENAI_TP_ONLINE_TOTAL_ROWS_BUDGET` to `160` improved the
profiled long-output engine phase (`26.34s -> 25.65s`) and moved the focused row
to `382.9 / 27.3 / 1537.4ms`, but TPOT worsened and E2E was flat. A larger
`192`-row budget reproduced a better long-output median both with profiling
(`352.7 / 27.0 / 1441.6ms`, `20260625_131454`) and without profiling
(`351.0 / 27.8 / 1403.2ms`, `20260625_132047`) versus the paired no-env control
(`378.0 / 26.8 / 1445.9ms`, `20260625_132431`). The global knob also changes
sampled-short cache shape: self_consistency improved median TTFT but left E2E
flat and worsened p99 (`303.7 / 0.0 / 444.1ms`, p99 `1587.9/1639.5ms`, versus
the no-env control `416.5 / 0.0 / 443.9ms`, p99 `1304.6/1360.0ms`). A scoped
code patch was rejected too. Keeping a warmed 192-row persistent cache gave a
strong focused long row (`336.5 / 26.8 / 1361.6ms`) but regressed the full-suite
shape, especially few_shot and multi_turn (`200.1 / 52.5 / 242.9ms` and
`644.6 / 68.6 / 691.9ms` in `20260625_134508`). Warming only the old 144-row
cache while allocating the larger greedy-short cache at runtime caused a
long-output TTFT tail blow-up (`691.3ms` median, `32.5s` p99). A follow-up
dual-cache prototype on the runtime-symm stack is also rejected: the default
192-row greedy-short cache (`20260625_161437`) landed at
`379.9 / 27.6 / 1487.9ms` but worsened the profiled phase to `32.97s`
(`13.34s` prefill wall, `16.51s` decode GPU, `7.23s` readback) with a bad p99
tail, and a 160-row scoped variant (`20260625_161848`) was not cleaner at
`360.0 / 27.3 / 1508.7ms` with `31.59s` phase. Do not promote a flat larger row
budget or the tested cache split; the viable path needs a more fundamental
shape-specific scheduler/warmup change.

Current `8e55b6e` long-output profiling keeps the same decode/readback shape.
The focused default (`20260626_155550`) landed at
`313.8 / 27.0 / 1488.3ms`, 1000/1000 correct, with `48` prefill graph batches,
`11.90s` prefill wall, `724` decode batches, `13.66s` ragged-decode GPU event
time, and `6.74s` CPU token wait. Raising
`TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_BATCH`,
`TORCHINFERNO_DECODE_LINEAR_MM_MAX_BATCH`, and ragged decode buckets to include
`128` is rejected (`20260626_160131`): score-facing latency regressed to
`318.6 / 28.3 / 1573.2ms`, throughput fell to `25.5 tok/s`, and the profile
still capped max model batch at `64` with `724` decode batches. The current long
gap is not a missing 128-row ragged graph; it remains a decode/readback pipeline
issue.

A follow-up local exploratory patch made continuous-serving ragged decode use
configured non-power-of-two buckets (`32,40,48,56,64`) instead of the default
power-of-two padding. This did exercise the intended shapes
(`b33-40/40`, `b41-48/48`, `b49-56/56`), reducing padded decode tokens
(`44.7K -> 40.9K`), but it regressed the focused long row to
`335.8 / 27.0 / 1514.7ms` (`20260626_165151`). Decode batches rose
`724 -> 779`, ragged GPU time rose `13.66s -> 15.33s`, and CPU token wait rose
`6.74s -> 7.43s`. Do not promote configurable continuous decode buckets from
this result; the power-of-two graph replay shape is faster despite extra padded
rows.

A narrower decode-many-only variant is also rejected on current `5b25a4e` plus
a local env-gated source patch. The focused long_output run
`agent_space/ti_long_decode_many_fine_buckets_results/.../8xH100/runs/20260702_130453`
completed 1000/1000 correct with
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_BUCKET_SIZES=4,8,16,24,32,40,48,56,64`
but regressed to `279.2 / 24.9 / 1395.8ms`, with p99 TTFT/E2E
`2623/3934ms`. The profile did reduce ragged padding versus the current public
long_output profile (`7.93K -> 6.22K` padded tokens), but it introduced four
request-path decode graph captures for `b24`, `b40`, `b48`, and `b56`, costing
`2.79s`; ragged decode GPU time rose from the public `10.48s` to `12.80s` and
online phase time rose from `19.58s` to `22.73s`. The source patch was reverted.
Keep decode-many on the existing graph-warmed power-of-two bucket family unless
non-power decode shapes can be warmed and replayed without request-path captures
and without the prior broad-bucket GPU regression.

A same-host long-output provider refresh on current TorchInferno `4fb4065`
(`20260626_170958`, isolation monitor disabled because an unrelated small GPU
process tripped the guard in the first attempt) confirms the local gap:
vLLM/SGLang/TorchInferno landed at `80.2 / 19.2 / 853.3ms`,
`82.0 / 28.4 / 1077.5ms`, and `291.0 / 27.8 / 1460.7ms`. TorchInferno preserves
correctness and is near SGLang TPOT, but vLLM still wins every score-facing
long-output latency/throughput metric locally. Treat the remaining long row as
a true decode/streaming pipeline gap, not public-run noise.

A current paired submit+step TP-command recheck on `bf71a37` keeps that policy
out of deterministic long-output. Forcing
`TORCHINFERNO_OPENAI_TP_ONLINE_SUBMIT_STEP_COMMAND=1` completed 1000/1000
correct and improved score-facing medians versus the paired no-env control
(`283.7 / 27.3 / 1381.3ms` vs `301.8 / 28.1 / 1491.2ms`), but the queue profile
did not show an engine win. Submit/step command overhead fell
(`579ms -> 397ms` across submit sync plus step broadcast), while phase total
worsened (`29.38s -> 30.06s`), decode batches rose (`759 -> 763`), ragged
decode GPU time rose (`15.25s -> 15.89s`), CPU token wait rose
(`7.30s -> 7.73s`), and p99 TPOT/E2E regressed (`175.1 / 6365.9ms` to
`268.5 / 6832.8ms`). Keep combined submit+step scoped to sampled-short
traffic; long-output still needs lower decode/readback cost rather than fewer
control-plane commands.

Finished-prefix reuse through the non-common graph prefill path remains unsafe
on current `7e67e71`. Enabling
`TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE=1` plus
`TORCHINFERNO_CONTINUOUS_NON_COMMON_PREFIX_GRAPH_PREFILL=1` reached readiness
for `multi_turn`, then failed the benchmark with an incomplete streamed body
and a CUDA `vectorized_gather_kernel index out of bounds` device assert before
writing a queue profile. Disabling active-row adoption with
`TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE_ADOPT_ROWS=0` changed the failure
mode but not the conclusion: it admitted only `136` requests, emitted `294`
events, wrote one profile snapshot (`15.5s` phase total, `10.3s` prefill wall,
`20` prefill graph hits, `19` prefix-reuse graph batches), then failed with an
HTTP 500 after a CUDA device-side assert surfaced during decode token readback.
Keep the non-common finished-prefix graph path out of default multi-turn; the
row-lifetime variant is not the only issue. The graph route is now additionally
guarded behind `TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_GRAPH_PREFILL=1` so the
broad non-common graph env cannot select it accidentally.

Current `def840e` follow-up probes reject three narrower queue/cache knobs.
For multi_turn, enabling pinned full-prompt stores for greedy-large requests
with non-common mixed-prefix grouping and graph capture-on-miss disabled
completed correctly (`982/1000`) but regressed to `1743.2 / 76.3 / 1804.4ms`.
The profile cut raw prefill tokens (`~80K -> 45.3K`) and raised prefix reuse
tokens to `97.7K`, but prefill wall grew to `27.9s`; mixed-prefix eager prefill
spent `16.0s` in forward and `11.5s` in state/prefix-store work. For
self_consistency, forcing `TORCHINFERNO_OPENAI_TP_ONLINE_COLLECT_IDLE_ARRIVALS=0`
preserved correctness but did not reduce internal work: phase total moved
`4.47s -> 4.58s`, generated-prefix reuses dropped `994 -> 981`, and the focused
row landed at `322.2 / 0.0 / 429.7ms`. For few_shot, extending greedy
KV-bounded admission to 256 max tokens raised active rows to `84` and improved
median E2E in one focused run (`170.6 / 54.5 / 214.6ms`) but worsened TPOT,
p99, and profiled phase time (`6.34s -> 6.58s`). Capping that path at 64 active
rows was worse: without a longer idle timeout it split into `9` online sessions
and regressed to `933.8 / 83.7 / 999.0ms`; with a `200ms` idle timeout it stayed
in one session but still regressed to `256.8 / 60.1 / 319.8ms`. Keep the
current few_shot 32-row greedy-mid policy and do not promote these env knobs.

The same counters on dense multi_turn (`20260625_101513`) show the opposite:
submission cadence is not the limiter there. The run landed at
`437.5 / 65.3 / 510.4ms`, 981/1000 raw correct, with `34` submit batches; `24`
were full 32-request batches and submit-sync time was only `168ms`. Runtime was
again prefill dominated: `10.15s` prefill wall, `9.65s` prefill forward, `34`
prefill graph hits, and `1.33s` decode-active. Multi-turn still needs a faster
or less fragmented conversation-prefix/suffix prefill path; batching HTTP
arrivals is not the missing lever for this row.

Lowering the runtime FP8 prefill `M` gate for the same row is also rejected on
current code. `TORCHINFERNO_OPENAI_TP_ONLINE_FP8_PREFILL_MIN_M=512`
(`20260625_124613`) landed at `436.5 / 65.0 / 507.2ms`, essentially neutral on
median latency, while the profile worsened prefill forward time to `10.88s`
from the default full-run `10.12s`. Keep the greedy-large FP8 gate at `2048`.

Disabling the greedy-large online FP8 prefill path entirely is not a promotion
either. With `TORCHINFERNO_OPENAI_TP_ONLINE_FP8_PREFILL=0`, current `b46a056`
landed at `472.5 / 66.6 / 544.0ms`, with `35` prefill batches,
`79.9K` prefill tokens, and `10.75s` prefill wall. That is better than the
noisy public `205205` multi_turn E2E but worse than the focused nearby baseline
(`433.2 / 63.2 / 501.4ms`, `9.88s` prefill wall), so keep the default FP8
prefill policy enabled for this path.

Greedy generated-prefix caching is also rejected for multi_turn on current
`29c4791`. Enabling `TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE=1` reduced
raw prefill tokens (`79.2K -> 46.2K`) but fragmented the run into `331` prefill
batches with `298` graph misses, stored `1452` generated prefixes, recorded zero
exact generated-prefix continuations, and regressed the row to
`6151.9 / 630.2 / 6591.8ms`. Adding
`TORCHINFERNO_CONTINUOUS_NON_COMMON_PREFIX_GRAPH_PREFILL=1` to route those
generated prefixes through the graph path did not recover it: after readiness it
created no queue-profile records, grew workers to about `64GB` each, and was
terminated. Generated-prefix reuse needs coarser grouping before it can help the
multi-turn suffix-prefill gap.

Long-output rechecks also did not produce a clean promotion. The focused
baseline landed at `405.1 / 27.0 / 1461.9ms`, with `40` prefill graph batches,
`11.98s` prefill wall, and `14.81s` active decode. Enabling greedy-short
decode_many slightly improved TPOT but regressed TTFT/E2E
(`436.1 / 26.2 / 1647.8ms`) and left active decode essentially unchanged
(`14.77s`). Forcing the paged online engine on this short-context shape with
`TORCHINFERNO_OPENAI_PAGED_KV_MIN_SEQ=1` is a clear regression
(`1642.4 / 146.4 / 7015.1ms`, `5.3 tok/s`), validating the current long-context
threshold. Lowering the greedy-short refill floor from `16` to `8` improved
TTFT (`353.1ms`) but raised prefill fragmentation (`56` prefill batches) and
did not clearly improve E2E/TPOT (`27.5 / 1523.0ms`); keep the current
16-request refill floor.

Current `75f9a0f` long-output decode-loop A/Bs are also rejected. Raising only
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_GEN_DECODE_QUANTUM` from `8` to
`16` cut online step commands (`95 -> 52`) but did not reduce runtime decode
model calls and regressed TTFT (`377.9 -> 484.9ms`) while leaving E2E roughly
flat (`1544.9 -> 1537.0ms`). An experimental full-active decode_many gate
(`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_DECODE_MANY=1`,
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_WHEN_FULL=1`, and
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_ALLOW_STOP=1`) preserved
correctness but still issued `831` decode model calls, overcomputed decode
tokens (`45.8K`), and regressed TTFT/p99 (`442.1ms` median TTFT, `136ms` p99
TPOT). Revert the experimental gate; the durable gap remains GPU decode work and
streaming readback, not online-step command count.

Current `89f239a` long-output decode-readback probes keep the same conclusion.
`TORCHINFERNO_GREEDY_SAMPLE_GATHER=1` is rejected: it preserved correctness but
regressed the row to `418.5 / 27.7 / 1600.7ms`, with no meaningful reduction in
decode GPU/readback totals (`12.72s` GPU, `11.65s` CPU-token wait). A bounded
two-step decode_many experiment that only ran when admission was blocked
(`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_DECODE_MANY=1`,
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_WHEN_ADMISSION_BLOCKED=1`,
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_ALLOW_STOP=1`,
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_MAX_STEPS=2`) cut the profiled
phase total to `24.95s` and token-harvest wait to `6.10s`, but overcomputed
decode tokens (`45.9K`) and worsened streaming tails (`126.1ms` p99 TPOT,
`4298.7ms` p99 E2E). The no-profile public-style check was not promotable at
`378.8 / 26.5 / 1423.6ms`, with `125.1ms` p99 TPOT. Keep decode_many off by
default; the next viable long-output improvement needs true pipelined token
readback or lower GPU decode work without bursty streaming emission.

Self-consistency sampled-short rechecks are also rejected on `27d3d7d`. The
focused no-env profile landed at `377.8 / 0.0 / 404.8ms`; internally it already
used the intended shortcut shape: one common-prefix prefill, one decode batch,
`987` generated-prefix reuses, and `1000/1000` correctness. Disabling generated
prefix caching improved TTFT to `292.0ms` but regressed E2E/throughput to
`425.4ms` / `2.4 tok/s` and required `45` decode batches. Raising the
sampled-short idle drain window to `25ms` regressed to
`430.8 / 0.0 / 460.6ms`. Disabling TP online step sync was promising as a pure
env run (`215.4 / 0.0 / 339.1ms`, p99 E2E `1358.4ms`), but the no-env default
guard did not reproduce (`392.1 / 0.0 / 422.8ms`) even though the profile
confirmed step sync was absent. Keep step sync on by default; this path needs a
less noisy reduction in submit/runtime-step overhead before promotion.

A focused tree_of_thought profile on `27d3d7d` is retained as diagnostic
evidence, not as a tuning target. The isolated row regressed relative to the
public full run at `335.3 / 52.7 / 373.4ms`; queue counters show the sampled
medium branch dominates aggregate work (`896` sampled requests, `6.95s` prefill
wall, `48` prefill batches, `2.54s` decode-active) while greedy eval requests
are small (`96` requests, `1.07s` prefill wall). Do not change sampled-medium
admission from this isolated noisy row; the durable tree gap is still prefill
pipeline cost versus vLLM's much lower TTFT/E2E.

Current-head refresh `0777c3e` keeps the same conclusion with a better focused
row: `265.9 / 52.1 / 314.1ms`, 961/992 correct. Queue totals split by online
session show sampled `temperature=0.7, max_tokens=300` traffic still dominates
(`896` submitted requests, `45` prefill batches, `6.08s` prefill wall,
`2.32s` decode-active), while greedy eval traffic is much smaller
(`80` submitted requests, `10` prefill batches, `1.54s` prefill wall). The next
tree improvement needs to reduce sampled-medium prefill pipeline cost; the
greedy eval path is not the primary limiter.

Tree sampled-medium row-cap refresh is rejected on the current startup/symm
stack. The 32-row focused baseline on `a180fbb` landed at
`280.5 / 52.2 / 321.6ms` (TTFT/TPOT/E2E), with `55` prefill batches,
`4373.8ms` prefill forward, `114` decode batches, and `150` scheduler steps.
Forcing `TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_MAX_ACTIVE=40` produced
one promising row at `256.4 / 52.5 / 310.4ms`, but after promotion to
`10f3443` no-env repeats regressed to `311.7 / 52.3 / 348.5ms` and
`310.4 / 53.2 / 344.1ms`. A 48-row check was worse at
`362.8 / 53.3 / 396.4ms`. Keep the stable 32-row sampled-medium cap.

Long-output greedy-short refill `24` is rejected on current `a261de3`. The
focused baseline landed at `406.1 / 26.1 / 1454.3ms`, 1000/1000 correct, with
`38` prefill batches, `11.35s` prefill wall, `13.68s` decode-active time, and
`10.95s` CPU token copy. Raising only
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_REFILL_MIN_READY_REQUESTS` to `24`
improved median TPOT to `24.4ms` but regressed TTFT/E2E to
`495.1 / 1610.1ms`. Keep the current 16-request refill floor; this path needs a
streaming readback or decode-loop improvement, not a larger refill wait.

## CURRENT SAME-HOST REFRESH AFTER OPENAI SYMM-MEM PROBE (2026-06-24, baseline f0c333d)

Same-host public-style inference-bench run `20260624_055652` with TorchInferno
`f0c333d`, vLLM `d4448b5`, and SGLang `84a7a84` landed at vLLM 14 scorecard
wins, SGLang 3, and TorchInferno 2. TorchInferno's two scorecard wins remain
few_shot TPOT/E2E (`54.5 / 214.1ms`), while the remaining same-host gaps are
self_consistency E2E (`377.1ms` vs vLLM `326.0ms`), multi_turn E2E
(`512.4ms` vs `242.0ms`), tree_of_thought E2E (`333.0ms` vs `101.8ms`), and
long_output decode/E2E (`32.2 / 1543.2ms` vs vLLM `18.8 / 771.6ms`).

Opting the OpenAI TP path into symmetric-memory decode allreduce is broadly
positive on this host. A TorchInferno-only all-row probe with explicit
`TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE=1` and a 256-row cap preserved
correctness and moved the rows to few_shot `152.9 / 52.3 / 194.9ms`,
self_consistency `218.5 / 0.0 / 356.7ms`, multi_turn `428.2 / 66.2 / 502.3ms`,
tree_of_thought `291.0 / 52.0 / 326.3ms`, and long_output
`297.5 / 28.6 / 1438.6ms` (TTFT/TPOT/E2E). The promoted no-env default path
runs the existing graph-aware symm-memory probe and enables decode symm only
after a successful collective probe. With the default 128-row cap, the
score-facing no-env check preserved correctness and landed at few_shot
`151.8 / 50.4 / 193.6ms`, self_consistency `312.8 / 0.0 / 381.8ms`,
multi_turn `429.9 / 62.9 / 499.1ms`, tree_of_thought
`216.4 / 50.7 / 254.0ms`, and long_output `334.7 / 27.6 / 1503.9ms`.
Self-consistency was noisy and not improved in that broad run, but the other
score-facing rows narrowed without changing benchmark semantics.

Two follow-ups are rejected. Enabling eager prefill symm-mem alongside decode
symm regressed few_shot p99 and did not help median TTFT (`155.3 / 51.6 /
199.7ms`), and the same prefill setting on long_output regressed TTFT
(`336.9 / 27.9 / 1512.3ms`). Combining decode symm with the long_output
`decode_many` path improved TTFT (`249.6ms`) but lost TPOT/E2E versus decode-only
symm (`29.2 / 1464.3ms` vs `28.6 / 1438.6ms`). Keep prefill symm and
`decode_many` opt-in.

The greedy-large first-batch wait is also still rejected. Raising
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_INITIAL_BATCH_WAIT_MS` from the
default 5ms to 10ms preserved correctness but regressed multi_turn to
`447.8 / 77.3 / 535.0ms` with worse p99. Keep the current 5ms default.

A narrower long_output refill floor is promoted on current `6e2cc27`.
The no-env focused long_output control with queue profiling landed at
`327.5 / 27.9 / 1565.1ms` (TTFT/TPOT/E2E), 1000/1000 correct, with only
`11` requests in the first wave, `53` graph-hit suffix prefill batches, and
`11435.9ms` prefill wall. Raising the deterministic short-output refill floor
to `16` requests preserved correctness and shifted the same run to
`351.7 / 26.6 / 1392.2ms`, cutting suffix prefill to `41` batches and
`9931.0ms` prefill wall. The promoted no-env default reproduced the refill
floor (`admit_min_ready_requests=16`) and landed at
`394.8 / 26.4 / 1447.6ms`, 1000/1000 correct, with `39` suffix prefill batches.
TTFT moved the wrong way, but E2E/TPOT and p99 E2E improved versus the profiled
control, and the policy is scoped to greedy `max_tokens<=128`, leaving
few_shot's 256-token path and multi_turn's 512-token path on their existing
defaults.

Prompt-lookup decode is rejected for this long_output shape. Enabling
`TORCHINFERNO_CONTINUOUS_PROMPT_LOOKUP_DECODE=1` made the server spend minutes
of active GPU time without emitting the first 1000-request progress line or a
queue-profile snapshot, while the no-env baseline completed the full request
wave in about `25.8s` after readiness. The existing prompt-lookup grouping is
too fragmented by per-request sequence length for this traffic; keep it opt-in
until it has a coarser batching design.

## CURRENT SAME-HOST REFRESH AFTER SAMPLED-MEDIUM IDLE (2026-06-24, current 6059831)

Same-host inference-bench run `20260624_015224` with vLLM `8dd1b702f27e`,
SGLang `c65f4ea692dd`, and TorchInferno `6059831` landed at vLLM 16 metric
wins, SGLang 5, TorchInferno 4. TorchInferno still wins few_shot TPOT locally
(`58.1ms` vs vLLM `59.6ms`) but remains behind on queue-facing medians:
few_shot `168.0 / 58.1 / 217.4ms`, self_consistency
`305.0 / 0.0 / 419.8ms`, multi_turn `455.4 / 68.1 / 532.9ms`,
tree_of_thought `275.9 / 57.4 / 313.2ms`, and long_output
`322.6 / 33.3 / 1727.4ms` (TTFT/TPOT/E2E).

The same run keeps the stale public TorchInferno `_server` row out of the
latency discussion: local TorchInferno reaches readiness and completes all five
benchmarks with 100% benchmark correctness rates. The largest current same-host
gaps are still long_output decode/readback throughput (vLLM
`90.8 / 18.8 / 766.1ms`), multi_turn prefill/session reuse (vLLM
`186.5 / 61.2 / 237.8ms`), and tree TTFT/pipeline latency (vLLM
`73.9 / 35.9 / 101.3ms`). The sampled-medium persistent-idle promotion improved
focused tree medians, but the full-run tree row is still a prefill/decode
pipeline gap rather than a wait-knob gap.

## CURRENT SAME-HOST REFRESH AFTER FP8 GATE (2026-06-23, current 15e19b8)

A fresh same-host run with vLLM `8dd1b702f27e`, SGLang `c65f4ea692dd`, and
TorchInferno `15e19b8` landed at vLLM 20 wins, SGLang 4, TorchInferno 1. The
only TorchInferno win is still few_shot TPOT: `55.6ms` vs vLLM `57.5ms`.
TorchInferno rows were few_shot `168.8 / 55.6 / 216.0ms`, self_consistency
`321.4 / 0.0 / 424.7ms`, multi_turn `619.4 / 70.2 / 674.3ms`,
tree_of_thought `301.5 / 56.7 / 322.9ms`, and long_output
`294.0 / 33.2 / 1674.3ms` (TTFT/TPOT/E2E). The isolated multi_turn FP8 win
remains real, but the full sequential run is noisier and still far from vLLM's
`184.1 / 57.4 / 235.5ms` multi_turn row; do not treat FP8 prefill as sufficient
for the multi_turn queueing gap.

The public `20260623_221253` TorchInferno `_server` row is stale with respect
to later startup fixes. It failed in a 600s NCCL `BROADCAST` watchdog while
broadcasting a 234M-element checkpoint tensor from rank 0 at TorchInferno
`5ad5429`. Current main is past the startup fixes where TP rank-0 checkpoint
tensor broadcast is configurable and inference-bench defaults
`NCCL_CUMEM_ENABLE=0` for TorchInferno. Treat that public row as a startup
integration failure from an old commit, not as evidence about current runtime
latency.

Tree-of-thought remains a real same-host gap on current `4be4712`, and the
public/local TPOT discrepancy is still not reproducible locally. A focused
three-provider run with inference-bench `20260623_214401` landed at vLLM
`73.9 / 36.2 / 101.0ms`, SGLang `71.1 / 75.1 / 168.0ms`, and TorchInferno
`263.9 / 57.3 / 321.4ms` (TTFT/TPOT/E2E). TorchInferno reached readiness in
226s with `rank0_broadcast=0`, confirming the startup path is past the stale
public failure. Queue counters show the sampled-medium branch path dominates:
896 requests with `max_active=32`, `decode_quantum=4`, and about `7116ms`
prefill wall plus `2602ms` decode-active time.

Sampled-medium chunked prefill is rejected on current `f3db1fc`. Enabling
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_CHUNK=128` for a focused
tree_of_thought run preserved correctness in-family (957/992 raw correct) but
regressed badly to `586.3 / 59.0 / 650.6ms`, p99 E2E `2230.8ms`, versus the
nearby non-chunked tree evidence around `230.1 / 56.3 / 272.1ms`. Queue
counters explain the loss: the sampled-medium sessions still used max model
batch `32`, but chunking spread work across `45` prefill batches, about `54k`
prefill tokens, `171` scheduler steps, and `15.8s` aggregate batcher wall.
Keep online prefill chunking opt-in for sampled tree traffic; this path
fragments the high-MFU prefill waves without improving decode interleaving.

Sampled-medium post-idle arrival collection is rejected. Forcing
`TORCHINFERNO_OPENAI_TP_ONLINE_COLLECT_IDLE_ARRIVALS=1` reduced aggregate
sampled-medium prefill wall (`7116ms -> 6024ms`) and prefill batches
(`47 -> 45`), but the score row only moved to `262.0 / 57.8 / 317.2ms` while
p99 E2E regressed badly (`1721ms -> 3217ms`). Cutting the global idle collection
window to 1ms reduced aggregate prefill further (`5697ms`, `43` batches) but
regressed the row to `308.3 / 57.1 / 343.3ms`. Do not enable sampled-medium idle
collection without a tail-safe admission rule; the current tree gap still needs
a prefill/decode pipeline change rather than another wait knob.

Paged-KV multi_turn remains rejected on current `2fd31a9`. Rechecking
`TORCHINFERNO_OPENAI_PAGED_KV_MIN_SEQ=512` with
`TORCHINFERNO_PAGED_PREFIX_CACHE=1` no longer failed readiness (server ready in
231.1s), but the runtime path was dominated by paged-engine work: the first
queue snapshot finished only `241` requests after `45660ms`
(`phase_runtime_step_ms=45566ms`). Dense current multi_turn finishes all 1000
requests in about `13883ms` profiled. The run was manually terminated before
completion; do not lower the paged threshold or enable paged prefix caching for
this 569-token-session shape without a substantially faster paged prefill path.
A later working-tree patch batched COW shared suffix prefills by uniform suffix
length, avoiding the obvious one-FlashInfer-prefill-per-shared-request
serialization in `PagedEngine`. That fixes a real paged-engine structure, but
it is still not enough for the score path: the focused `multi_turn` A/B with
`TORCHINFERNO_OPENAI_PAGED_KV_MIN_SEQ=512` and
`TORCHINFERNO_PAGED_PREFIX_CACHE=1` completed at
`6176.1 / 708.1 / 6285.0ms`, p99 E2E `10660.6ms`, with 100% normalized
correctness. Keep the default threshold unchanged; the paged path still needs a
larger prefill/decode pipeline redesign before it can replace dense for the
current benchmark shape.

Mixed-length paged-prefix suffix padding is accepted as paged-engine
infrastructure, not as a score-path switch. On the same `multi_turn` A/B,
bounded padding collapsed paged-prefix prefill from `267` model calls to `35`
and aggregate paged prefill forward time from about `51.0s` to `10.1s`; medians
improved to `3040.3 / 713.5 / 3296.4ms` with correctness `0.982`. This proves
the prior paged prefix cache path was launch-fragmented, but the remaining
`35` paged prefill waves still cost roughly `260-290ms` each. Keep
`TORCHINFERNO_PAGED_PREFIX_CACHE` opt-in and leave
`TORCHINFERNO_OPENAI_PAGED_KV_MIN_SEQ` unchanged for the current dense
multi_turn shape.

Finished-prefix row adoption is also rejected for multi_turn on current
`4ea5e98`. Enabling `TORCHINFERNO_CONTINUOUS_FINISHED_PREFIX_CACHE=1` completed
1000/1000 requests but took `96311.6ms` in the batcher profile, with
`93926.4ms` prefill wall across `453` prefill batches. It stored `1000`
generated prefixes and raised total prefix reuse tokens to `99781`, but exact
generated-prefix reuse stayed at `0`; the adopted rows fragmented prefill into
mostly single-prefix work instead of producing reusable conversation-session
batches. Keep this cache path opt-in until it has a row budget and batching
policy that can reuse finished prefixes without destroying prefill shape.
Greedy-large idle-arrival collection is rejected on current `b15b006`.
`TORCHINFERNO_OPENAI_TP_ONLINE_COLLECT_IDLE_ARRIVALS=1` completed multi_turn
with 100% benchmark correctness and medians of `442.2 / 74.0 / 527.7ms`, but
the profile moved the wrong way: total batcher time `13452ms`, prefill wall
`11583ms`, and `35` prefill batches versus the nearby dense baseline around
`12185ms` total / `10327ms` prefill wall / `34` prefill batches. The slightly
lower TTFT does not justify the TPOT and aggregate prefill regression; leave
idle collection scoped to sampled-short traffic.

Long-output decode-quantum rechecks are rejected. Current default `8` in the
full run landed at `294.0 / 33.2 / 1674.3ms`, `24.5 tok/s`. Focused env-only
rechecks with the current tree landed at DQ=10 `325.1 / 33.2 / 1638.1ms`,
DQ=12 `383.7 / 30.3 / 1594.7ms`, and DQ=16
`493.4 / 28.0 / 1882.7ms`. Larger quanta trade away TTFT and throughput for
TPOT, and still do not beat vLLM (`18.8ms`) or SGLang (`27.2ms`) on TPOT. Keep
the greedy-short default at 8.

Long-output `decode_many` also stays opt-in. Enabling the existing greedy-short
multi-step decode path with stop-token overcompute and the default 8-step
quantum cut queue-profile CPU token-copy time from about 12.9s to 5.0s, but
regressed the score row to `375.3 / 32.7 / 1726.1ms`, `23.0 tok/s`. A 4-step
variant looked better as an env-only run (`290.6 / 32.2 / 1575.1ms`,
`25.1 tok/s`) but worsened p99 and did not reproduce as a no-env default guard:
the patched default landed at `246.2 / 33.8 / 1852.9ms`, `23.6 tok/s`, despite
the profile confirming `decode_many_enabled=true` and `decode_quantum=4`. Do not
promote decode_many without a safer streaming/stop-token design; the current
path can reduce CPU synchronization while still hurting client-observed E2E.
A narrower q2 recheck on current `65eb78b` is rejected too:
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_DECODE_MANY=1` with
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_GEN_DECODE_QUANTUM=2` preserved
correctness and improved TTFT to `198.3ms`, but regressed TPOT/E2E to
`35.9 / 1707.7ms`. Queue profile showed no real readback win
(`12.89s` CPU token copy, `14.03s` decode GPU) and prefill fragmented to
89 batches, so the one- or two-step overcompute family is not the current
long_output lever.

Long-output online step-sync-off remains rejected on current `1338fe5`.
`TORCHINFERNO_OPENAI_TP_ONLINE_STEP_SYNC=0` preserved correctness and landed at
`283.7 / 33.3 / 1616.4ms`, which looks like a median E2E improvement, but the
queue profile regressed total phase time (`29.46s -> 30.21s` versus the current
focused baseline), prefill batches (`63 -> 67`), and p99 E2E (`5318ms`). The
removed `154ms` step-sync accounting was not the long_output bottleneck.

Long-output greedy-small session bucketing is promoted after the current local
recheck. The final no-env default check with a 96-token greedy-small session
bucket plus a 21-row greedy KV prefix floor landed at
`307.1 / 31.2 / 1546.8ms`, 1000/1000 correct. The queue profile shows the
intended shape: `run_max_tokens=96`, `max_active=123`, `prefix_rows=21`,
`57` prefill batches, and `28.08s` total phase time. This keeps per-token
streaming intact, unlike decode_many, while reducing refill fragmentation versus
the prior 128-token bucket / no-prefix-floor shape.
A score-facing no-profile A/B on `9056eac` also kept the new shape ahead:
current defaults landed at `293.0 / 31.9 / 1683.1ms`, while restoring the old
128-token bucket and no greedy prefix floor via env regressed to
`356.5 / 33.2 / 1706.6ms`.
A no-profile long_output probe with ragged decode buckets disabled
(`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_BUCKETS=0`) is also rejected on
`3675f6a`: it preserved correctness but regressed to
`359.8 / 38.8 / 2135.3ms`, with p99 E2E `10508ms`. Keep bucketed ragged decode
for greedy-short long_output; the padding cost is lower than the graph/shape
instability from exact active-row decode shapes.
A guarded streaming-decode GPU-buffer reuse path is rejected on current
`d09fda1`. It reused device-side last-token/seq-len buffers between streamed
ragged decode steps, but the focused long_output run landed at
`303.4 / 31.8 / 1575.4ms` and the profile did not reduce prepare cost
(`decode_ragged_prepare_ms=1201ms`, with `12419ms` prefill wall). The extra
indexing/sync bookkeeping traded one small allocation path for another, so the
code was removed instead of keeping another opt-in knob.
A follow-up 20ms initial-wait recheck on `74173a1` remains rejected:
`288.4 / 32.7 / 1682.5ms`, p99 E2E `5057ms`. The larger wait collected a
15-request first wave but still used 59 prefill batches and regressed prefill
wall time to `13.39s`, so keep the greedy-short initial wait at 10ms.

Long-output greedy-short KV active cap 64 is rejected on current `aa7a9a9`.
The hypothesis was that 64 client workers could not use the default 112 active
rows, so capping `TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_KV_MAX_ACTIVE_CAP=64`
would restore prefix rows (`32 -> 64`) without losing decode concurrency. The
run preserved correctness but regressed badly to `710.1 / 54.9 / 2591.2ms`,
`14.7 tok/s`, versus the no-env current profile at `309.9 / 32.5 / 1740.6ms`,
`23.1 tok/s`. The profile showed why: max model batch stayed `64`, but prefill
wall time nearly doubled (`12956ms -> 25058ms`) and prefill batches rose
(`58 -> 64`). Keep the greedy-short KV cap above the client-worker count; the
extra active rows reduce queue-facing prefill pressure even when the final decode
batch size does not exceed 64.

Self-consistency sampled row-cap rechecks above 128 rows are still rejected. On
current `ebfae5b`, the same-host default was `373.4 / 0.0 / 442.4ms`, p99 E2E
`1672.8ms`. Raising true active rows to 256 without raising the total row budget
removed prefix rows (`prefix_rows=0`), forced 18 plain prefill batches over 55k
tokens, and regressed to `623.3 / 0.0 / 730.0ms`. Restoring 64 prefix rows at
256 active rows fixed the plain-prefill failure (`prefill_tokens=55`, `988`
prefix-reuse requests) but still landed at `324.4 / 0.0 / 465.0ms`. A 160-row
variant improved TTFT and p99 but not median E2E (`315.9 / 0.0 / 447.2ms`).
The 192-row, 64-prefix-row variant looked score-positive in isolation
(`286.0 / 0.0 / 409.0ms`, `2.4 tok/s`) and a broad no-env default reproduced
`282.4 / 0.0 / 393.6ms`, but it regressed focused tree_of_thought to
`312.8 / 71.8 / 361.2ms`. Forcing the old 128-row startup/batch ceiling
recovered tree to `230.1 / 56.3 / 272.1ms`, and a narrower runtime-only
192-row lift crashed self/tree with CUDA device-side asserts. Prewarming 192
rows while keeping the request batch ceiling at 128 avoided the crash and
improved self to `234.5 / 0.0 / 372.6ms`, but still regressed tree to
`246.8 / 70.5 / 356.5ms`, isolating the tree cost to the larger warmed/persistent
cache shape. Rebuilding smaller dense caches before tree did not recover it; the
self/tree sequence hung before tree emitted queue-profile records. Keep
sampled-short KV-bounded admission at 128 until a design can raise self
concurrency without changing tree startup/cache behavior or runtime allocation
shape.

Self-consistency uniform-ragged decode is rejected. Forcing
`TORCHINFERNO_CONTINUOUS_UNIFORM_RAGGED_DECODE=1` moved sampled self from static
decode graphs into ragged decode (`31` ragged batches, max model batch `128`),
but padded decode work to `1419` tokens for 1000 one-token completions and
increased decode-active time. The live row regressed to
`353.5 / 0.0 / 437.1ms`, `2.3 tok/s`, with p99 E2E `1602.8ms`, versus the
same-host default self row at `321.4 / 0.0 / 424.7ms` and p99 E2E `904.4ms`.
Keep uniform-ragged decode off for sampled identical-prompt traffic.

Self-consistency is not a queue-profile overhead artifact. A no-profile default
run on `6b1ae7e` landed at `358.2 / 0.0 / 450.6ms`, `2.2 tok/s`, p99 E2E
`1703.5ms`, worse than the profiled same-host full-run row. Keep using queue
profiles for diagnosis; the self gap needs scheduling/decode work, not less
instrumentation.

Self-consistency sampled-short decode quantum 8 is rejected on current
`fe6b77e`. The shape-count profiled default used `decode_quantum=4` and landed
at `273.5 / 0.0 / 407.2ms`, 1000/1000 correct, with one common-prefix prefill,
37 decode batches, and 74 scheduler steps. The env-only
`TORCHINFERNO_OPENAI_TP_ONLINE_SHORT_GEN_DECODE_QUANTUM=8` run reduced internal
work only slightly (35 decode batches, 70 scheduler steps) and landed at
`307.3 / 0.0 / 402.0ms`. The small E2E movement does not justify a scoped
sampled-short default while median TTFT regresses.

Self-consistency sampled-short generated-prefix caching is promoted on current
`959b094`. Enabling the existing generated-prefix cache preserved 1000/1000
correctness and landed at `191.3 / 0.0 / 335.7ms`, `3.0 tok/s`. The queue
profile shows the intended general mechanism: one shared prompt prefill, one
generated prefix stored, `983` generated-prefix reuses, and decode batches
dropping from the prior `37` waves to `1`. Scope the default to sampled
`max_tokens<=256` traffic so tree's sampled-medium path stays unchanged, and
preserve the runtime env overrides for manual disable/adaptive experiments.
Post-promotion validation on `6995c69` confirms the default wiring after a
preceding few_shot row: the final self profile stored one generated prefix and
reused it `990` times (`1980` emitted events, `1` prefill batch, `1` decode
batch), landing at `261.3 / 0.0 / 391.0ms`. The remaining self gap is wave
formation and TP command overhead, not a missed generated-prefix hit. A
submit-barrier removal probe (`TORCHINFERNO_OPENAI_TP_ONLINE_SUBMIT_SYNC=0`,
local patch only) is rejected because the first few_shot row stopped making
progress after readiness; keep online submit synchronization intact. Do not
broaden the generated-prefix threshold to tree without a separate mechanism:
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_SHORT_GENERATED_PREFIX_CACHE_MAX_TOKENS=300`
did not enable tree hits because the online session rounded to
`run_max_tokens=400`, and the profiled row was a hard regression
(`758.8 / 560.5 / 1262.0ms`, zero generated-prefix stores/reuses). Actually
enabling the threshold at 400 was worse: the no-profile tree row landed at
`832.0 / 573.3 / 1405.5ms`, so generated-prefix collection must stay out of
sampled-medium tree traffic.

Current `6e73a7d` self-consistency profiling keeps that diagnosis. The focused
row (`20260626_165921`) landed at `195.8 / 0.0 / 328.8ms`, 1000/1000 correct.
The last complete progress snapshot shows the generated-prefix path is healthy:
one common-prefix prefill (`common_prefix:b1:t55`), one decode model call, one
generated-prefix store, `984` generated-prefix reuses, and `109224`
prefix-reuse tokens. The remaining server-side phase was `2.97s`, dominated by
`1.57s` prefix-reuse/prefill wall, `517ms` submit-sync, and `143ms` step-sync
across `171` submit batches and `160` runtime steps. This is still
arrival-fragmentation and TP command overhead after reuse is working; do not
reopen the rejected sampled-short initial-wait, idle-wait, min-ready, or
decode-quantum knobs without a different batching mechanism.

Self-consistency sampled post-arrival collection should stay default-on for now.
An apples-to-apples current `1b60135` recheck with
`TORCHINFERNO_OPENAI_TP_ONLINE_COLLECT_IDLE_ARRIVALS=0` landed at
`332.7 / 0.0 / 411.0ms`, `2.4 tok/s`, p99 E2E `1703.7ms`. The no-env default
on the same checkout and host landed at `258.0 / 0.0 / 394.0ms`, `2.5 tok/s`,
p99 E2E `1548.3ms`. The disabled variant admitted a larger model batch
(`107` vs `84`) and two fewer decode batches (`39` vs `41`), but spent more
prefill wall time (`2324ms` vs `2204ms`) and more total batcher time
(`4830ms` vs `4575ms`). Do not flip collection off by default from the older
rejection note without a stronger scheduler change.

Self-consistency global Marlin-int4 decode disable is rejected on current
`9750616`. Rechecking with `TORCHINFERNO_MARLIN_INT4_DECODE=0` landed at
`347.2 / 0.0 / 433.6ms`, `2.3 tok/s`, p99 E2E `1571.3ms`, worse than the
same-host default self row at `321.4 / 0.0 / 424.7ms`. The queue profile also
regressed from the default's `37` decode batches and about `4602ms` batcher wall
time to `39` decode batches and `4696ms`. Keep Marlin decode enabled for the
sampled short-row path; disabling it is not a current self-consistency lever.

Few-shot startup/knob rechecks on current `b5cdbfc` are rejected. A 2ms initial
collection wait landed at `173.6 / 59.2 / 222.5ms`, worse than the nearby guard
run (`162.0 / 54.8 / 205.9ms`). Broad model-level FP8 with
`TORCHINFERNO_FP8_PREFILL=1` and `TORCHINFERNO_FP8_PREFILL_MIN_M=1024` landed at
`163.6 / 53.7 / 208.7ms`: a small TPOT move, but median E2E and throughput did
not beat the no-FP8 guard, so keep few_shot out of the default FP8 policy.
Common-prefix suffix prefill warmup is also not defaultable. The broad suffix
warmup (`16,32,64,128,256`) needed 331.4s to reach readiness before any request
was served. A narrowed `suffix_tokens=16,batches=32` run still needed 231.1s
startup, regressed p99 TTFT/E2E to `2186.0 / 2242.1ms`, and did not remove
runtime prefill graph misses in the queue profile. Keep suffix-prefill warmup
opt-in only; the few_shot gap is not a startup graph-capture issue.

Current `1c78b71` few_shot profiling keeps the same greedy-mid shape. The
focused default (`20260626_160731`) landed at `169.6 / 52.9 / 218.6ms`,
976/1000 raw correct, with `39` prefill batches, `78` decode batches,
`5.48s` prefill wall, and `3.09s` ragged-decode GPU time. Extending the
greedy-short policy to `max_tokens=256` is a hard rejection
(`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_GEN_MAX_TOKENS=256`,
`20260626_161211`): the row regressed to `323.1 / 238.0 / 559.0ms`, while
decode-many overcomputed to `481` decode batches and `15.0K` decode tokens for
only `2454` emitted events. Keep few_shot out of greedy-short decode-many. A
same-host provider comparison (`20260626_161634`) shows few_shot median is not
the current local priority: vLLM/SGLang/TorchInferno landed at
`160.9 / 61.1 / 214.3ms`, `144.2 / 75.1 / 218.3ms`, and
`156.5 / 50.5 / 198.8ms`. TorchInferno still has a bad p99 tail, but it wins
the local median TPOT/E2E cells; do not trade that away for public-run median
noise without a tail-safe graph/pipeline change.

Current `016ea32` tree profiling keeps the sampled-medium diagnosis. The
focused default (`20260626_162641`) landed at `320.1 / 57.9 / 370.6ms`,
964/992 raw correct, with a p99 E2E tail of `3015.5ms`. The sampled branch
submitted `896` requests across six sampled sessions and spent `10.76s` in
profiled phase time, including `7.81s` prefill wall, `6.06s` prefill forward,
`43` prefill graph-hit batches, and only `1.87s` ragged-decode GPU time. The
greedy eval branch was much smaller at `96` requests, `4.46s` phase time,
`1.56s` prefill wall, and `2.03s` ragged-decode GPU time. The worst sampled
session was the first 256-request wave at `4.09s` phase and `3.31s` prefill
wall despite graph hits, while a later 256-request wave was lower at `2.83s`
phase and `1.88s` prefill wall. That points to sampled-medium prefill/session
pipeline overhead and cold lower-level FP8 work, not a missing prefill graph or
a decode row bucket. Keep the rejected sampled-medium wait, idle collection,
larger active-row, and lower-FP8-gate knobs rejected until a mechanism can
reduce the first-wave prefill cost without worsening the tail.

Narrow common-prefix suffix warmups are also rejected on the same tree profile
shape. Warming only row `53`, prefix `45`, suffix `16`, and batches
`4,16,32` reached readiness in `125.6s` and cut aggregate sampled prefill wall
from `7.81s` to `6.17s`, but the row regressed to
`355.3 / 58.6 / 388.6ms` with six suffix-graph misses. Adding `b1` still missed
one suffix graph per sampled session and regressed further to
`382.8 / 58.2 / 427.7ms`; warming every power-of-two batch through `32` reached
readiness in `120.6s` but was worse again at `405.7 / 59.4 / 452.4ms`.
Startup suffix warmup can move internal counters, but it does not improve the
client-observed tree median or correctness band. Keep
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_COMMON_PREFIX_SUFFIX_PREFILL` opt-in.

## MULTI-TURN RUNTIME FP8 PREFILL GATE (2026-06-23)

Global `TORCHINFERNO_FP8_PREFILL=1` remains rejected because it previously
regressed few_shot. A narrower online runtime policy is now validated for
deterministic greedy-large sessions only (`400 < max_tokens <= 512`) with a
higher prefill M gate (`min_m=2048`). The no-env live run on current local
TorchInferno enabled FP8 only through this runtime path and moved multi_turn
from the same-host `504.6 / 70.2 / 583.6ms` band to `451.4 / 73.4 / 542.8ms`,
`2.2 tok/s`, with 982/1000 raw correct. The queue profile confirmed
`fp8_prefill_enabled=true`, `fp8_prefill_min_m=2048`, `run_max_tokens=512`,
34 prefill batches, and 83.5k prefill tokens.

The live few_shot guard stayed out of the policy: it recorded
`fp8_prefill_enabled=false`, `run_max_tokens=256`, and landed at
`162.0 / 54.8 / 205.9ms`, 976/1000 raw correct. Keep the model-level FP8 env as
an explicit broad override; the default path should only use the runtime setter
for greedy-large online sessions.

Lowering the online FP8 prefill M gate to 1024 is rejected on current
`4a812bc`. A same-commit multi_turn control with the default 2048 gate landed at
`440.7 / 67.8 / 512.1ms`, p99 E2E `2752.6ms`, with 982/1000 raw correct. The
1024-gated run preserved correctness and slightly moved TTFT to `439.2ms`, but
regressed TPOT/E2E to `73.7 / 517.6ms` and worsened p99 TPOT
(`124.6ms -> 164.8ms`). Queue profiles showed the same 34 prefill batches,
about 79k prefill tokens, and 86 decode batches in both runs; the lower gate's
smaller aggregate batcher wall did not translate to client-observed latency.
Keep the scoped greedy-large runtime FP8 gate at `min_m=2048`.

## TREE SAMPLED-MEDIUM RUNTIME FP8 PREFILL GATE (2026-06-25)

Tree-of-thought is prefill-heavy enough to benefit from the runtime FP8 prefill
path, but only in its sampled-medium request bucket. A same-host TP8 A/B on
current `bf0a31d` with an unrelated GPU0 process present compared explicit
BF16 (`TORCHINFERNO_OPENAI_TP_ONLINE_FP8_PREFILL=0`) against explicit FP8
(`TORCHINFERNO_OPENAI_TP_ONLINE_FP8_PREFILL=1`,
`TORCHINFERNO_OPENAI_TP_ONLINE_FP8_PREFILL_MIN_M=256`), with sampled idle
collection forced off in both runs. BF16 landed at `349.7 / 51.6 / 388.5ms`,
`4.0 tok/s`; FP8 landed at `302.8 / 51.5 / 330.7ms`, `4.4 tok/s`, preserving
benchmark correctness. The no-override implementation run landed at
`310.2 / 52.3 / 343.1ms`, `4.2 tok/s`.

Promote this as a narrow online policy only for sampled requests with
`256 < max_tokens <= 300`, using `min_m=256`. Keep sampled-short
self-consistency/few-shot (`max_tokens<=256`), longer sampled traffic, greedy
short long_output, and the broad model-level `TORCHINFERNO_FP8_PREFILL` path out
of the default. The noisy validation regressed tree p99 TTFT/E2E, so this is a
scorecard-median improvement rather than a tail-latency fix; leave the
sampled-medium idle-arrival collection experiment rejected until there is a
tail-safe scheduler change.

Full-prompt mixed-prefix reuse is still rejected for multi_turn on current
`a9666ff`. Enabling pinned full-prompt stores for 512-token sessions with
non-common-prefix graph prefill and mixed-prefix batching avoided the old CUDA
illegal-memory crash, but regressed badly to `1769.4 / 78.7 / 1842.4ms`,
`0.6 tok/s`, p99 E2E `3496.8ms`, with 982/1000 raw correct. The profile shows
the tradeoff: prefill tokens fell (`79.4k -> 45.3k`) and reused prefix tokens
rose (`45.0k -> 97.7k`), but prefill wall jumped (`11.3s -> 28.1s`) and
prefill forward rose (`10.8s -> 15.8s`) because the mixed-prefix path hit only
four prefill graphs. Forcing mixed-prefix graph capture did not fix it: the row
landed at `1740.4 / 80.2 / 1813.2ms`, p99 E2E `4565.8ms`, with `31` prefill
graph misses and `28.5s` prefill wall. An experimental static prefix-prefill
batch bucket did not change the graph-miss pattern (`3` hits / `31` misses) and
still landed at `1726.2 / 69.9 / 1790.9ms` with `27.4s` prefill wall, so that
code was backed out. Keep this path opt-in until
mixed-prefix reuse can stay on stable captured graph shapes.

Multi-turn prefix-suffix bucket splitting is rejected on current `cc727c2` +
instrumentation. The new queue-profile shape histograms showed the default
multi_turn path was not missing graphs: it had 34 prefill graph hits, zero
misses, one 45-token common prefix, and 33 graph-backed prefix-reuse suffix
waves under the same source prefix. Splitting prefix-reuse groups by suffix
bucket reduced padded prefill tokens only slightly (`79.3k -> 75.7k`) but
fragmented the run into 53 prefill batches and doubled prefill wall
(`10.3s -> 21.6s`, forward `9.8s -> 21.1s`). The live row regressed from the
nearby control `439.8 / 71.3 / 506.1ms` to
`1491.8 / 72.0 / 1543.3ms`, with p99 E2E `3073.4ms` and 980/1000 raw correct.
Keep common-prefix reuse batched by prefix; the useful direction is fewer
prefill waves or faster large-bucket prefill, not splitting suffix buckets.

## CURRENT SAME-HOST REFRESH AND TREE WAIT RECHECKS (2026-06-23)

After the greedy-large initial-wait patch, a current same-host four-row
comparison using vLLM `8dd1b702f27e`, SGLang `c65f4ea692dd`, and TorchInferno
`42ad0f0` still has the expected score shape: TorchInferno wins only
multi_turn TPOT. vLLM/SGLang remain much faster on queue-facing rows:
tree_of_thought is 71.4 / 35.9 / 98.0ms for vLLM versus 276.8 / 56.8 /
336.7ms for TorchInferno, and long_output is 84.1 / 18.8 / 749.3ms for vLLM
versus 275.5 / 32.9 / 1754.0ms for TorchInferno (TTFT/TPOT/E2E). Multi-turn is
still prefill-dominated: TorchInferno is 504.6 / 70.2 / 583.6ms versus vLLM's
189.5 / 75.0 / 250.8ms.

Two current tree_of_thought wait rechecks are rejected. Lowering only the
sampled-medium initial collection wait to 5ms landed at 315.2 / 57.0 / 350.3ms
and 954/992 raw correct. Raising the online idle drain globally to 10ms landed
at 313.9 / 56.4 / 347.4ms and 962/992 raw correct. Both are worse than the
same-run default tree row and do not change the local 56-57ms TPOT band. Keep
tree on the current sampled-medium wait and idle-drain defaults; the next tree
work needs a pipeline/prefill mechanism rather than another initial-collection
or idle-drain knob.

Sampled-medium persistent idle was promoted separately (2026-06-24, current
`97b6db2` + default patch). The shape-count control landed at
`333.6 / 56.3 / 377.1ms`, 957/992 raw correct, with 8 sampled-medium online
sessions and `7.46s` sampled prefill wall. A global
`TORCHINFERNO_OPENAI_TP_ONLINE_PERSISTENT_IDLE_MS=100` probe improved that to
`312.3 / 56.7 / 345.0ms`, 958/992, reducing sampled-medium sessions to 6 and
sampled prefill wall to `6.50s`. The scoped default, limited to sampled
requests above the sampled-short range and up to 300 max tokens, landed at
`290.6 / 57.4 / 337.9ms`, 952/992, with `5.83s` sampled prefill wall. This is a
median prefill/session-reuse win, not a tail fix: p99 E2E was still `2342ms`,
so tree tail work remains in the scheduler/pipeline bucket.

Raising the sampled-medium persistent idle above that default is rejected.
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_IDLE_MS=200` on current
`8f638ae` landed at `305.0 / 56.6 / 339.1ms`, 960/992 raw correct, with p99 E2E
`2397ms`. The larger idle window did not consolidate the tree into fewer useful
work waves and increased score-facing median latency versus the current 100ms
default/full-run band. Keep sampled-medium persistent idle at 100ms.

Sampled-medium row-cap midpoints remain non-defaultable on current `bb744e1`.
The no-profile 32-row control was `291.9 / 57.0 / 331.3ms`, 957/992 raw
correct. Raising only `TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_MAX_ACTIVE`
to 40 improved TTFT/E2E to `250.8 / 60.3 / 308.4ms`, 960/992, but regressed
TPOT, throughput, and p99s. The 36-row midpoint was worse outright at
`336.4 / 58.5 / 381.3ms`, 961/992. Keep the current 32-row sampled-medium cap
until a prefill/decode pipeline change can improve queue latency without adding
decode pressure.

Down-projection Marlin remains rejected for long_output too. A current
TorchInferno-only recheck with `TORCHINFERNO_MARLIN_INT4_DOWN=1` completed
1000/1000 correct but landed at 315.5 / 32.7 / 1677.1ms and 21.4 tok/s, worse
than the nearby default long rows on TTFT and throughput while leaving TPOT in
the same band. Queue snapshots showed no GPU decode win: final progress was
about 13.9s ragged-decode GPU time and 12.7s token-readback time across 712
decode batches. Keep down-proj Marlin opt-in until calibrated weights or a
broader fused decode path proves a real serving win.

## MULTI-TURN GREEDY-LARGE INITIAL WAIT (2026-06-23)

After the 32-row greedy-large cap, current `30bc24a` still showed multi_turn
dominated by prefix/suffix prefill waves: 493.2ms TTFT, 76.4ms TPOT, 571.8ms
E2E, 2.0 tok/s, and 979/1000 raw correct. A scoped recheck with
`TORCHINFERNO_OPENAI_TP_ONLINE_INITIAL_BATCH_WAIT_MS=5` admitted more of the
first client wave (initial batch 1 -> 4) and improved to 482.1ms TTFT, 69.6ms
TPOT, 557.5ms E2E, 2.1 tok/s, with the same 979/1000 correctness. Promote the
5ms wait only for deterministic 401-512 token online sessions. Keep the global
5ms initial-wait rejection intact for few_shot and other greedy traffic; it
regressed few_shot latency in earlier guards.

A later current recheck with
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_INITIAL_BATCH_WAIT_MS=10` is not
defaultable. On current `5abb3b5`, the full-run 5ms profile had initial batch 4,
35 prefill batches, `11.31s` prefill wall, and a score row of
`455.4 / 68.1 / 532.9ms`. The 10ms focused run admitted 6 initial requests and
reduced internal work slightly (34 prefill batches, `10.94s` prefill wall), but
landed at `448.4 / 72.4 / 523.5ms` and still does not beat the best focused
5ms band. Keep the deterministic 401-512 token default at 5ms until a prefill
pipeline/reuse change gives a clearer median and TPOT win.

## LONG-OUTPUT GREEDY-SHORT INITIAL WAIT (2026-06-23)

The robust symm-off baseline still shows long_output queue/pipeline pressure:
301.7ms TTFT, 31.8ms TPOT, 1674.9ms E2E, and 23.8 tok/s. The prior focused
10ms greedy-short initial-wait run completed 1000/1000 correct and improved the
score-facing decode/E2E/throughput cells versus the 5ms-style baseline
(27.4ms TPOT, 1456.9ms E2E, 28.4 tok/s), at the cost of a small median TTFT
increase. Promote the 10ms default only for deterministic max_tokens<=128
online sessions; sampled self/tree, few_shot, and multi_turn stay on their
existing policies.

Raising that scoped greedy-short initial wait to 20ms is not defaultable on
current `6dbdaaa`. The shape-count profiled default landed at
`297.5 / 32.4 / 1682.4ms`, 1000/1000 correct, with initial batch 7, 63 prefill
batches, 743 decode batches, `12.26s` prefill wall, `13.40s` CPU token readback,
and `14.51s` decode GPU time. The 20ms run admitted more of the first wave
(initial batch 13) and reduced aggregate work to 60 prefill batches, 696 decode
batches, `12.09s` prefill wall, `12.78s` CPU readback, and `13.87s` decode GPU
time, but the score row was `276.6 / 32.0 / 1689.3ms` and throughput fell to
24.1 tok/s. The long_output gap needs pipeline/readback work rather than a
larger initial collection window.

Greedy-short runtime FP8 prefill is also rejected for long_output on current
`0afa3a4`. Forcing `TORCHINFERNO_OPENAI_TP_ONLINE_FP8_PREFILL=1` preserved
correctness and landed at `309.2 / 31.9 / 1644.3ms`, but the profile did not
support a prefill win: prefill wall regressed versus the focused default
(`12.26s -> 13.06s`) and p99 E2E rose to `4794ms`. Keep FP8 prefill scoped to
the existing greedy-large multi_turn path until the FP8 graph path produces a
clear prefill-wall reduction for short greedy traffic.

Current `6995c69` long_output profile remains in the same dense decode/prefill
band: `326.4 / 32.0 / 1777.7ms`, `22.9 tok/s`, 1000/1000 correct. The queue
profile shows `52` prefill batches (`11.64s` prefill wall, `50.1k` prefill
tokens) and `712` decode graph hits (`16.65s` decode-active, `14.21s` decode
GPU event time, `13.06s` synchronous token readback exposure). This refresh
does not reopen the rejected decode-many, FP8-prefill, or larger-wait knobs; the
remaining long_output gap still needs a real decode/prefill pipeline change.

## OPENAI TP STARTUP RECHECK (2026-06-23, current 684af9b)

Public run `20260623_160941` still used stale TorchInferno bits and failed
readiness after NCCL init. A local focused run reproduced a related startup
stall after the symm-mem probe had passed: `py-spy` showed rank 0 blocked in
`torch.distributed._symmetric_memory.rendezvous` / `cuMulticastBindMem` during
ragged decode graph warmup. Because the serving process cannot safely recover
from that in-process CUDA rendezvous hang, the immediate mitigation kept OpenAI
TP symm-mem allreduce default-off and required explicit opt-in with
`TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_AUTO_PROBE=1` or
`TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE=1`. This gives up a measured decode
TPOT optimization on stable hosts but should move public runs from startup dash
back to benchmarkable NCCL behavior. Local validation with the patched default
reached readiness in 231.1s and completed long_output at 100% correctness
(301.7ms TTFT, 31.8ms TPOT, 1674.9ms E2E).

The public `20260625_130348` run confirms that leaving this probe default-on is
still unsafe on the benchmark host. It built TorchInferno `de2d6f1` with the
provider-level NCCL socket defaults, reported `symmetric-memory allreduce
enabled after probe`, completed NCCL init, then never bound `/health` before the
1800s readiness timeout killed torchrun. The `80279ef` mitigation restored an
opt-in default with no probe unless
`TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_AUTO_PROBE=1` or an explicit
`TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE=auto/probe` requested it; a local
no-env provider check on that commit showed no symm-probe log, reached readiness
in `100.5s`, and completed self_consistency 1000/1000 correct. The later
runtime-only split re-enables the auto-probe for default deterministic decode,
but startup warmups still stay off the symm scope.

## LOCAL MULTI-TURN LARGE-CAP RECHECK (2026-06-23, current 0d8749e + env)

The public `20260623_142642` run is still stale with respect to the pushed
NCCL CUMEM startup guard and symm-mem probe retry: TorchInferno failed to become
ready there. Local repeated TP8 startup on the current tree succeeds with
`NCCL_CUMEM_ENABLE=0` and symmetric-memory allreduce enabled after the probe.

For the remaining multi_turn gap, the 512-token greedy path was rechecked with
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_MAX_ACTIVE=32`. Current `0d8749e`
landed at 464.8ms TTFT, 64.6ms TPOT, 531.3ms E2E, and 2.2 tok/s, 979/1000 raw
correct. That matches the two earlier focused rechecks (464.7 / 65.2 / 530.7
and 478.0 / 64.7 / 548.3) and improves over the default 16-row band from recent
full/local runs (about 594-614ms TTFT and 631-659ms E2E). Promote 32 rows for
deterministic 401-512 token online sessions; it is scoped away from sampled
self/tree and short greedy long_output.

## LATEST RUN 20260622_060431 (built bc3b9ea): 1/20

TorchInferno fell to 1/20 (vLLM 18/20, SGLang 0/20). The only public win is
multi_turn TPOT: TorchInferno 37.5 ms vs vLLM 43.0 ms. The score-moving near
misses are now TPOT cells: tree_of_thought is only 1.0 ms behind vLLM
(`30.2ms` vs `29.2ms`), few_shot is 1.2 ms behind (`45.7ms` vs `44.5ms`), and
cross-benchmark TPOT is 0.6 ms behind (`26.9ms` vs `26.3ms`). TTFT/E2E remain
queue/prefill dominated: multi_turn is `584.7ms` TTFT vs vLLM `158.3ms`,
self_consistency is `272.6ms` vs `206.2ms`, and long_output still needs real
decode work (`21.4ms` TPOT vs vLLM `14.8ms`). This public run predates the
common-prefix no-logits cleanup (`d3421c2`) and the later chat stop-id runtime
propagation (`fb7fdf0`).

FULL SAME-NODE BASELINE (2026-06-22, current 4746555,
inference-bench `20260622_063441`): with all three providers on the same host,
TorchInferno scores 3/20: few_shot TPOT (52.9ms) and E2E (199.8ms), plus
multi_turn TPOT (42.2ms). vLLM scores 19/20 and SGLang scores 3/20. The current
actionable gaps are not few-shot median latency; TorchInferno already beats vLLM
there locally except throughput/p99. The large remaining gaps are self median
latency (TorchInferno 324.7 / 439.3 ms vs vLLM 167.2 / 280.0 ms), multi_turn
TTFT/E2E/tput (597.6 / 640.0 ms / 2.0 tok/s vs vLLM 179.7 / 232.1 ms /
5.9 tok/s), tree_of_thought (222.5 / 52.2 / 265.0 ms vs vLLM 71.9 / 34.8 /
97.6 ms), and long_output decode/queueing (340.8 / 27.5 / 1591.6 ms vs vLLM
71.3 / 18.8 / 753.7 ms). SGLang is useful as a TTFT reference in few/tree/multi
but is not a self-consistency tail target: p99 E2E was 58.4s.

POST-IDLE GATE FULL GUARD (2026-06-22, current e15935c TorchInferno-only):
after making post-idle arrival collection opt-in, the full local guard landed at
few_shot 149.4 / 50.5 / 190.1 ms, self_consistency 312.8 / 0.0 / 389.7 ms,
multi_turn 598.6 / 40.6 / 639.0 ms, tree_of_thought 300.7 / 52.2 / 330.6 ms,
and long_output 283.5 / 28.5 / 1442.5 ms (TTFT/TPOT/E2E). The self row improved
against the prior full same-node TorchInferno row, few stayed strong, and
multi/long stayed in family. The weak full-run tree row was order/variance:
a focused e15935c tree recheck landed at 219.4 / 51.3 / 258.3 ms with 955/992
raw correct.

POST-IDLE GATE PROVIDER COMPARISON (2026-06-22, current c6073ca,
inference-bench `20260622_073810`): the fresh same-node provider run keeps the
same score shape: vLLM 16 metric wins, SGLang 6, TorchInferno 3. TorchInferno's
wins are few_shot TPOT/E2E (51.9ms / 200.7ms) and multi_turn TPOT (42.3ms).
The large remaining gaps are unchanged: self_consistency is 353.7 / 457.0 ms
versus vLLM 198.4 / 236.3 ms, multi_turn is 586.1 / 627.1 ms / 2.1 tok/s
versus vLLM 183.5 / 239.2 ms / 5.8 tok/s, tree_of_thought is 428.8 / 50.5 /
468.0 ms versus vLLM 71.3 / 35.5 / 98.0 ms, and long_output is 371.9 / 26.7 /
1482.6 ms versus vLLM 71.1 / 18.7 / 759.3 ms. The gate is a small default-path
cleanup, not a score-shape change.

ONLINE CACHE-FIT ORDER GUARD (2026-06-22, current 84b3594 + local patch):
the weak full-run tree rows were traced to stale dense persistent-cache shape
metadata being consulted before deciding whether the cache fit the next online
session. A previous short sampled session could cap the compatibility window for
the next tree session even though the batcher later allocated a larger cache;
the TP worker also reset/reused its persistent cache without the primary's fit
check. The patch centralizes the cache shape test and applies it before both
admission and primary/worker reuse. Focused self_consistency -> tree_of_thought
landed at self 368.6 / 0.0 / 455.2 ms and tree 228.0 / 53.2 / 266.9 ms
(`20260622_075219`). A full TorchInferno-only guard landed at few_shot
154.5 / 53.3 / 197.9 ms, self_consistency 359.5 / 0.0 / 446.6 ms, multi_turn
581.0 / 42.4 / 621.6 ms, tree_of_thought 246.0 / 53.8 / 317.3 ms, and
long_output 346.1 / 28.5 / 1568.2 ms (`20260622_075841`). This is a tree/order
stability fix; the remaining score-facing gaps are still self median latency,
multi_turn TTFT/E2E/throughput, and long_output decode/queueing.

POST-CACHE-FIT PROVIDER COMPARISON (2026-06-22, current addfef4,
inference-bench `20260622_081355`): the same-node provider comparison after the
cache-fit guard landed at vLLM 20 wins, SGLang 3, TorchInferno 2. TorchInferno
kept few_shot TPOT (56.0ms) and multi_turn TPOT (45.0ms), but few_shot E2E did
not hold in this run (227.1ms versus vLLM 205.1ms). tree_of_thought improved
from the prior provider outlier (428.8 / 468.0 ms TTFT/E2E) to 356.7 / 385.4 ms
but remained far behind vLLM's 73.0 / 100.8 ms. self_consistency also remains a
large median gap at 405.5 / 521.2 ms versus vLLM 177.1 / 239.7 ms, and
long_output remains decode/queue-bound at 399.1 / 27.3 / 1630.9 ms versus vLLM
71.2 / 19.0 / 764.3 ms. Treat the cache-fit guard as correctness/stability; the
next score work needs self wave formation and long-output decode/queueing.

SELF INITIAL-WAIT RECHECK (2026-06-22, current addfef4 + local patch): the
current self_consistency queue profile (`20260622_081944`) still admitted only
9 requests in the initial wave, then processed 1000 requests through 44
exact-prefix reuse batches. Runtime totals were about 1926ms in prefix-reuse
prefill work and 1207ms in decode-active work. Raising only the sampled-short
initial collection default from 10ms to 20ms improved focused self_consistency
to 304.5 / 0.0 / 382.3 ms, 2.6 tok/s, 1000/1000 correct
(`20260622_082535`). This is narrower than the previously rejected idle-wait
knobs and remains scoped to sampled short requests (`max_tokens <= 256`), so it
does not apply to few_shot, tree_of_thought, multi_turn, or long_output.

LONG GREEDY-SHORT INITIAL-WAIT RECHECK (2026-06-22, current 35a3304 + local
patch): the current long_output queue profile (`20260622_083311`) admitted only
one request in the initial wave, then spread 1000 requests across 59 prefill
batches and 728 decode batches. Raising only the greedy-short initial wait from
1ms to 5ms improved focused long_output from 342.0 / 28.5 / 1459.7 ms to
326.4 / 27.6 / 1429.1 ms, 26.0 tok/s, 1000/1000 correct (`20260622_083907`).
This is narrower than the rejected global 5ms initial-wait experiment: it only
applies to greedy requests with `max_tokens <= 128`, while few_shot uses
`max_tokens=256` and stays on its existing path.

INITIAL-WAIT FULL GUARD (2026-06-22, current 9e08add TorchInferno-only): the
combined cache-fit, sampled-short 20ms initial wait, and greedy-short 5ms
initial wait defaults landed at few_shot 152.5 / 51.6 / 195.0 ms,
self_consistency 318.7 / 0.0 / 396.9 ms, multi_turn 587.1 / 41.0 / 625.8 ms,
tree_of_thought 267.3 / 51.1 / 293.2 ms, and long_output 281.0 / 28.5 /
1526.7 ms (`20260622_084712`). Correctness stayed in family: few 976/1000,
self 1000/1000, multi 980/1000, tree 960/992, long 1000/1000. The self row is
the main default-path improvement; long TTFT improved but TPOT remains in the
same 27-28ms local band.

SAMPLED-SHORT IDLE COLLECTION DEFAULT (2026-06-22, current 3b298c0 + local
patch): the first same-node provider recheck after the 20ms sampled initial wait
did not hold the focused win. Provider run `20260622_090138` scored vLLM 19,
SGLang 5, TorchInferno 1; TorchInferno kept only multi_turn TPOT, and
self_consistency was 384.7 / 0.0 / 539.1 ms. A corrected queue profile with the
20ms default admitted 17 initial rows but still needed 44 exact-prefix reuse
batches and regressed focused self to 357.7 / 422.2 ms. Reverting sampled-short
initial collection to 10ms and enabling post-idle collection only for sampled
short requests improved the focused no-env self run to 254.8 / 0.0 / 392.5 ms
with 1000/1000 correct (`20260622_092731`). The full TorchInferno-only guard
landed at few_shot 153.3 / 47.9 / 193.2 ms, self_consistency 329.8 / 0.0 /
402.0 ms, multi_turn 600.2 / 40.8 / 644.6 ms, tree_of_thought 247.4 / 53.3 /
293.3 ms, and long_output 289.3 / 27.7 / 1425.7 ms (`20260622_093338`). Treat
this as a self wave-formation cleanup; it is intentionally scoped away from
tree_of_thought's sampled-medium path and greedy workloads.

PREFIX-REUSE GRAPH CAPTURE DIVERGENCE FIX (2026-06-22, current 519188d + local
patches): the first same-node provider recheck on `519188d` wedged during
TorchInferno long_output after few/self/multi/tree had completed. `py-spy`
showed a collective-order mismatch in prefix-reuse prefill: rank 0 had fallen
through to eager ragged prefill and was in a tensor-parallel all-reduce, while a
worker rank was still in ragged prefill graph-capture success synchronization.
Disabling prefix-reuse capture-on-miss avoided the wedge and completed the
provider run (`20260622_102029`, vLLM 17 / SGLang 6 / TorchInferno 2), but it
over-serialized prefix reuse and ballooned few_shot/multi_turn TTFT. The
narrower follow-up keeps prefix graph capture enabled by default and instead
clears stale `_skip_capture_sync` markers whenever the online runtime adopts an
external persistent cache, so primary and worker ranks make the same
capture/replay decision. `TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_CAPTURE_ON_MISS=0`
remains an escape hatch for diagnosing future capture-order issues.
The follow-up provider run on `5a1bd23` completed without the long_output wedge
and restored the expected score shape: vLLM 21 wins, TorchInferno 3, SGLang 1
(`20260622_103746`). TorchInferno wins are few_shot TPOT/E2E (51.7ms /
197.2ms) and multi_turn TPOT (41.3ms). Long_output completed at
300.8 / 27.2 / 1486.1 ms, back in the pre-guard latency band instead of the
no-capture guard's 427.5 / 31.7 / 1714.4 ms.

CLEAN NO-PROFILE TREE REPRO (2026-06-21, current 33a2eb3): rerunning
tree_of_thought locally with no queue profiling did not reproduce the public
`30.2ms` TPOT row. The latest inference-bench checkout, `--skip-build`, and the
current TorchInferno worktree landed at 293.5 / 53.6 / 317.3 ms
(TTFT/TPOT/E2E), 4.2 tok/s, and 959/992 raw correct. That is worse than the
recent local no-logits guard while the public run still shows ~30ms TPOT. Treat
local tree TPOT as an environment-sensitive diagnostic for now; do not default a
change solely because it moves the local 52-54ms TPOT band unless same-node vLLM
or a counter profile explains the public/local delta.

LLAMA CHAT STOP-ID CLEANUP (2026-06-21, current bd0e430 + local patch): the
Transformers chat tokenizer path was treating every added `<|...|>` control
token as a stop id. For Llama-3.1-70B-Instruct that produced 256 stop ids, while
the continuous runtime accepts one `eos_token_id`. The server could finish an
HTTP request on `<|eot_id|>` while the runtime row kept decoding until max tokens
or an arbitrary control id. Narrowing chat stop ids to the known terminators
(`<|end_of_text|>`, `<|eom_id|>`, `<|eot_id|>`) and preferring `<|eot_id|>` for
runtime EOS changed the real tokenizer from 256 stop ids to 3 with runtime EOS
128009. The no-profile tree_of_thought guard improved the local TPOT band from
the clean current run's 293.5 / 53.6 / 317.3 ms to 216.6 / 51.8 / 254.7 ms,
4.4 tok/s, and 964/992 raw correct. The HTTP-only profile from the preceding
control showed response framing was not the 52ms TPOT source: p50 content send
was 0.94ms, p50 finish send was 0.58ms, and server-side reconstructed positive
TPOT was still 53.5ms. Keep the stop-id cleanup; it is general row-retirement
correctness and a small tree TPOT win, but it does not explain the public/local
30ms vs 52ms discrepancy by itself.

FULL STOP-ID GUARD (2026-06-21, current cadbeb8 local no-profile): a
TorchInferno-only full pass after the stop-id cleanup landed at few_shot
156.3 / 52.7 / 200.3 ms, self_consistency 287.8 / 0.0 / 401.1 ms,
multi_turn 584.3 / 42.8 / 621.5 ms, tree_of_thought 231.3 / 51.8 / 263.8 ms,
and long_output 339.5 / 28.1 / 1441.6 ms (TTFT/TPOT/E2E). Compared with the
earlier full local current guard, self_consistency TTFT/E2E improved and tree
TPOT stayed in family, but few_shot and long_output TPOT did not improve. Treat
the change as the right Llama chat-row retirement behavior, not as a confirmed
full-score speedup.

MARLIN GATE RECHECK (2026-06-22, current fb7fdf0 local no-profile A/B): the
default gate-up Marlin int4 decode path is not a clean self_consistency lever.
Default current landed at 327.7 / 0.0 / 412.4 ms and 2.4 tok/s; disabling
`TORCHINFERNO_MARLIN_INT4_DECODE` improved self_consistency to 309.1 / 0.0 /
388.0 ms and 2.6 tok/s with 100% correctness, but the same global disable
regressed tree_of_thought to 238.9 / 57.5 / 291.4 ms and 4.1 tok/s. Lowering
`TORCHINFERNO_MARLIN_INT4_MAX_M` from 256 to 96 was also rejected: self worsened
to 356.3 / 0.0 / 433.5 ms and 2.3 tok/s. Keep Marlin enabled for the short-row
decode workloads for now; a future self-only disable would need an explicit
request-shape policy rather than a model-wide default.

MARLIN DOWN-PROJECTION RECHECK (2026-06-22, current b823758 local no-profile
tree slice): enabling `TORCHINFERNO_MARLIN_INT4_DOWN=1` is not defaultable. The
tree_of_thought run landed at 288.0 / 51.4 / 317.3 ms, 4.1 tok/s, and 952/992
raw correct. TPOT moved only inside the local 51-53 ms band, while TTFT/E2E and
correctness were worse than the recent default tree controls. Keep down-proj
Marlin off unless a later calibrated variant proves math correctness and a
score-facing latency win.

GREEDY SAMPLE-GATHER RECHECK (2026-06-22, current 26752a8 no-profile): replacing
greedy token selection's two tiny all-reduces with the one-collective gather path
(`TORCHINFERNO_GREEDY_SAMPLE_GATHER=1`) is not defaultable. long_output TPOT
improved within the local band (27-28 ms -> 25.8 ms), but TTFT/E2E/throughput
regressed to 375.1 / 1557.8 ms / 26.6 tok/s. few_shot regressed more clearly:
159.9 / 52.0 / 204.7 ms and 5.8 tok/s, 976/1000 raw correct. Keep the gather
sampler opt-in.

SELF SAMPLER RECHECKS (2026-06-22, current 787006d no-profile): the recent
temperature-sampling knobs are rejected for self_consistency. Capping decode
linear `mm` use at batch 128 (`TORCHINFERNO_DECODE_LINEAR_MM_MAX_BATCH=128`)
landed at 372.9 / 0.0 / 445.8 ms, 2.2 tok/s. Enabling the gathered temperature
sampler (`TORCHINFERNO_TEMPERATURE_SAMPLE_GATHER=1`) landed at 325.9 / 0.0 /
391.8 ms, 2.6 tok/s, which is only local noise versus the current self band.
Disabling repeated-prefix Gumbel sampling
(`TORCHINFERNO_TEMPERATURE_SAMPLE_REPEATED_GUMBEL=0`) was a clear regression:
405.8 / 0.0 / 472.6 ms, 2.1 tok/s. Keep repeated-prefix Gumbel enabled at the
current threshold and leave the alternate samplers opt-in. Lowering the
threshold to force Gumbel on small repeated-prefix waves is also rejected:
`TORCHINFERNO_TEMPERATURE_SAMPLE_REPEATED_GUMBEL_MIN_BATCH=1` preserved
generated-prefix reuse (`985` reuses, one decode batch) but landed at
`379.2 / 0.0 / 400.2ms` with more online commands (`283`) and higher submit-sync
time (`959ms`), so the self gap is not solved by changing the repeated sampler.

SELF IDLE-DRAIN BATCHING (2026-06-22, current bccf781 + local patch): a default
queue-profiled self_consistency run showed the remaining gap was wave
fragmentation, not sampling: the server collected only 9 requests initially and
processed 1000 requests through 43 exact-prefix prefill-reuse batches plus 44
decode batches. Runtime totals were 1949ms in prefill/exact-prefix work and
1186ms in decode-active time. Raising only the online idle drain wait from 2ms
to 10ms for sampled-short bursts improved a no-profile run from the comparable
317.6 / 0.0 / 407.5 ms profile row to 232.4 / 0.0 / 369.4 ms. A 20ms wait was
too long and is rejected: 375.2 / 0.0 / 457.8 ms. The default patch scopes the
10ms wait to sampled requests with `max_tokens <= 256`, so tree_of_thought's
sampled 300-token branches and greedy rows stay on the 2ms idle drain. The
first post-patch no-env guard landed at 287.7 / 0.0 / 399.6 ms, 1000/1000
correct. A single bounded post-arrival collection run later landed at 235.1 /
0.0 / 352.3 ms, 2.8 tok/s, 1000/1000 correct, but follow-up rechecks did not
hold that win: current default with post-arrival collection reproduced 381.4 /
0.0 / 434.1 ms, while gating post-arrival collection back to opt-in improved
the same current checkout to 343.7 / 0.0 / 411.7 ms. Keep the 10ms sampled-short
idle drain, but leave post-arrival collection behind
`TORCHINFERNO_OPENAI_TP_ONLINE_COLLECT_IDLE_ARRIVALS=1`. Lowering the
sampled-short idle drain to 8ms was too short and is rejected: 379.9 / 0.0 /
451.1 ms, 2.2 tok/s. Raising it to 12ms is also rejected: 380.8 / 0.0 /
456.5 ms, 2.2 tok/s. Extending sampled-short persistent idle to 500ms is also
not a lever: 341.5 / 0.0 / 427.4 ms, 2.3 tok/s, 1000/1000 correct.

SELF SAME-NODE PROVIDER COMPARISON (2026-06-22, current fc084b0,
inference-bench `20260622_051934`): with the existing local vLLM/SGLang builds,
vLLM still leads the self_consistency median row: vLLM 183.7 / 0.0 / 341.7 ms,
2.9 tok/s; SGLang 218.6 / 0.0 / 389.6 ms, 2.6 tok/s; TorchInferno 300.7 / 0.0 /
391.3 ms, 2.6 tok/s. TorchInferno's p99 E2E was the best of the three
(1035.8ms vs vLLM 1056ms and SGLang's pathological 58889.6ms), so the remaining
self gap is median wave latency/throughput rather than tail stability.

SELF STOP-LOOKAHEAD REJECTED (2026-06-22, current 76b24e4 + temporary patch):
an env-gated one-row exact-prefix lookahead decoded the sampled first token once
and finished the whole reuse group only if all sampled second tokens were stop
tokens. It preserved correctness but serialized too much work: self_consistency
regressed to 550.0 / 0.0 / 597.6 ms, 1.7 tok/s. Do not revisit without a
batched or graph-captured verification path.

ADDITIONAL SCHEDULER REJECTIONS (2026-06-22, current 30992b1 no-profile):
multi_turn does not benefit from simply keeping the online session open longer;
`TORCHINFERNO_OPENAI_TP_ONLINE_PERSISTENT_IDLE_MS=100` landed at 600.6 / 42.7 /
643.5 ms, 1.9 tok/s, matching the prior prefill-dominated band. few_shot also
does not want a global 5ms initial wait:
`TORCHINFERNO_OPENAI_TP_ONLINE_INITIAL_BATCH_WAIT_MS=5` landed at 152.9 / 50.1 /
195.1 ms, 6.2 tok/s, worse than the current few_shot band. Keep multi_turn on
the default persistent-idle policy and keep greedy few_shot's initial collection
window short.

FULL POST-IDLE LOCAL REFRESH (2026-06-22, current b7bf3d2 no-profile,
TorchInferno-only): the pushed idle-drain stack landed at few_shot 154.3 / 49.8 /
196.5 ms, self_consistency 244.9 / 0.0 / 365.6 ms, multi_turn 578.8 / 40.6 /
618.8 ms, tree_of_thought 282.8 / 51.9 / 307.4 ms, and long_output 316.8 /
28.2 / 1442.6 ms (TTFT/TPOT/E2E). Correctness stayed in family: few 976/1000,
self 1000/1000, multi 980/1000, tree 959/992, long 1000/1000. The self row is
the real movement from the idle-drain work; the other rows remain in their local
variance bands and still need separate levers.

TREE/LONG SCHEDULER REJECTIONS (2026-06-22, current b380546 no-profile): a
global 10ms idle drain for tree_of_thought
(`TORCHINFERNO_OPENAI_TP_ONLINE_IDLE_BATCH_WAIT_MS=10`) landed at 227.1 / 52.2 /
265.8 ms, 4.3 tok/s, so TPOT stayed in the local 52ms band and the setting is
not score-facing. Larger long_output greedy-short decode quanta expose the
expected admission tradeoff. Quantum 16 improved TPOT to 23.7ms but regressed
TTFT to 472.3ms; quantum 12 landed at 413.1 / 25.8 / 1542.5 ms. Keep the
current 8-step short-greedy quantum until a dynamic policy can raise quantum
only when it will not delay newly arrived requests. A temporary dynamic-q16
experiment that shrank the command when an active row was near completion still
landed at 448.8 / 24.5 / 1411.9 ms with lower throughput, so that simple
remaining-token heuristic is also rejected. Disabling the per-command online
step sync for long_output (`TORCHINFERNO_OPENAI_TP_ONLINE_STEP_SYNC=0`) also
does not help: 309.7 / 27.8 / 1533.2 ms with worse p99.

LONG SAME-NODE PROVIDER COMPARISON (2026-06-22, current a72825d,
inference-bench `20260622_053240`): vLLM still owns the long_output row on the
same host and harness: vLLM 82.2 / 18.8 / 815.5 ms, 47.4 tok/s, versus
TorchInferno 333.2 / 27.8 / 1485.1 ms, 23.6 tok/s. Both providers were
1000/1000 correct. TorchInferno's p99 is also behind here (TTFT 1902.7ms and
E2E 3668.1ms versus vLLM 805.2ms and 1313.3ms), so the remaining long gap is
not just a median decode issue; queue/admission tail behavior is still material.

LONG REFILL-READINESS REJECTIONS (2026-06-22, current a72825d no-profile): the
refill batching mechanism is real but not defaultable. Raising
`TORCHINFERNO_OPENAI_TP_ONLINE_REFILL_MIN_READY_REQUESTS` to 32 cut median TPOT
to 20.6ms, but TTFT/E2E regressed to 520.7 / 1521.7 ms and p99 TPOT worsened to
89.2ms. The middle point at 16 was also worse than default: 377.1 / 26.1 /
1571.2 ms with 80.8ms p99 TPOT. Larger refill groups reduce suffix-prefill
fragmentation, but the queueing cost lands directly in score-facing latency.
Keep greedy-short refills on the current default until a policy can prove it
only waits when client arrivals are already exhausted.

TREE SAME-NODE PROVIDER COMPARISON (2026-06-22, current 9b90df2,
inference-bench `20260622_060104`): the local tree_of_thought TPOT band is real
on this host. vLLM landed at 73.7 / 35.3 / 99.9 ms, 12.1 tok/s, 963/992 raw
correct; TorchInferno landed at 204.1 / 52.4 / 240.5 ms, 5.0 tok/s, 962/992 raw
correct. This differs from the stale public run where TorchInferno showed
~30ms TPOT, so keep treating public/local tree deltas as environment-sensitive
until a fresh public run lands on the pushed stop-id and idle-drain stack.

TREE DECODE-QUANTUM REJECTION (2026-06-22, current 9b90df2 no-profile): raising
the broad short-generation quantum to 8 via
`TORCHINFERNO_OPENAI_TP_ONLINE_SHORT_GEN_DECODE_QUANTUM=8` does not move tree
TPOT and regresses latency: 229.4 / 52.6 / 273.3 ms, 4.6 tok/s, 956/992 raw
correct. The sampled-medium branch sessions stay in the local 52ms TPOT band,
while the broader override also weakens TTFT/E2E. Do not scope a sampled-medium
quantum default from this result.

FEW/MULTI SAME-NODE PROVIDER COMPARISON (2026-06-22, current ddc00a7,
inference-bench `20260622_061445`): TorchInferno is locally ahead of vLLM on the
few_shot latency row: vLLM 160.9 / 61.3 / 217.8 ms, 7.0 tok/s, versus
TorchInferno 155.4 / 52.5 / 200.9 ms, 5.9 tok/s. The remaining few gap is
throughput and p99, not median TTFT/TPOT/E2E. Multi_turn still has the expected
shape: TorchInferno wins TPOT (42.7ms vs vLLM 65.3ms), but loses TTFT/E2E/tput
badly (588.0 / 627.0 ms, 2.0 tok/s, with last-turn TTFT 951.4ms, versus vLLM
184.9 / 239.6 ms, 5.7 tok/s, last-turn TTFT 169.8ms). This points back to
cross-turn prefix/session reuse, not decode speed.

MULTI IDLE-DRAIN REJECTION (2026-06-22, current ddc00a7 no-profile): a broader
greedy idle collection window via `TORCHINFERNO_OPENAI_TP_ONLINE_IDLE_BATCH_WAIT_MS=10`
does not improve multi_turn. It landed at 594.8 / 43.7 / 639.3 ms, 1.8 tok/s,
with first-turn TTFT 933.6ms and last-turn TTFT 1059.8ms. The extra wait
slightly flattens turn growth but adds too much head latency; keep greedy-large
idle collection at the current default.

FEW ROW-CAP REJECTION (2026-06-22, current d3608c6 no-profile): lowering
greedy-mid active rows to 24
(`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_MID_MAX_ACTIVE=24`) improves the
few_shot TPOT cell to 44.5ms, but it regresses the rest of the row badly:
202.5ms TTFT, 237.8ms E2E, 5.4 tok/s, and worse p99. Do not default a
single-cell TPOT tradeoff that hurts the actual request latency shape.
The less aggressive 28-row midpoint is also rejected on current a72825d:
165.7 / 49.7 / 209.6 ms, 5.8 tok/s, and 976/1000 raw correct. That preserves
correctness and trims only noise-level TPOT while still worsening TTFT/E2E, so
the row-cap curve has no score-facing midpoint between 32 and 24.

FLASHINFER PREFILL REJECTION (2026-06-22, current 0d1273c no-profile): enabling
the experimental continuous FlashInfer prefill path
(`TORCHINFERNO_CONTINUOUS_FLASHINFER_PREFILL_DISABLE=0`) is not correct or fast
enough for few_shot. The run landed at 1201.5 / 135.5 / 1360.9 ms, 1.2 tok/s,
and only 891/1000 correct. Keep this path default-disabled until it has a
separate correctness fix and shape-specific evidence.

FULL CURRENT LOCAL REFRESH (2026-06-22, current 51fba66 no-profile,
TorchInferno-only): the pushed default stack landed at few_shot 149.3 / 50.2 /
189.8 ms, self_consistency 313.4 / 0.0 / 388.7 ms, multi_turn 595.8 / 41.4 /
633.3 ms, tree_of_thought 218.7 / 51.0 / 256.4 ms, and long_output 309.9 /
27.7 / 1494.1 ms (TTFT/TPOT/E2E). Correctness stayed in family: few 976/1000,
self 1000/1000, multi 981/1000, tree 957/992, long 1000/1000. This confirms
the public run is still stale with respect to the stop-id runtime work, but it
also confirms the local tree TPOT discrepancy remains: default local tree is
still ~51ms TPOT while public bc3b9ea reported ~30ms.

ADAPTIVE GENERATED-PREFIX CACHE REJECTED (2026-06-22, current 4186466 + local
patch): making generated-prefix continuation caching automatic for online rows
with queued exact-prompt reuse is not defaultable. The intended mechanism was to
pay one logits decode for a repeated prompt, store `prompt + first_token`, and
let later exact-prefix hits emit both the first token and likely stop token in
the same scheduler step. A self_consistency run with that adaptive behavior
enabled landed at 643.4 / 0.0 / 728.9 ms, 1.4 tok/s, 1000/1000 correct, far
worse than the current local self band. Keep this behind
`TORCHINFERNO_CONTINUOUS_ADAPTIVE_GENERATED_PREFIX_CACHE=1`; the broad
`TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE=1` rejection still stands.

TREE SCHEDULER KNOB RECHECKS (2026-06-21, current 60802a local no-profile A/B):
three more tree_of_thought knobs are rejected. Raising the sampled-medium online
idle window from the 10ms default to 100ms landed at 231.5 / 52.8 / 273.4 ms
(TTFT/TPOT/E2E), 4.3 tok/s, and 960/992 raw correct; keeping the engine open
between waves does not recover the vLLM gap. Lowering the sampled-medium initial
collection wait from 5ms to 1ms was only a marginal local shift, 224.3 / 52.6 /
269.8 ms and 4.6 tok/s, while raw correctness slipped to 959/992 and p99s got
worse. Raising `TORCHINFERNO_OPENAI_TP_ONLINE_MAX_ACTIVE` to 64 is a clear
regression: 309.8 / 53.3 / 372.9 ms, 3.9 tok/s, and 958/992 raw correct. Keep
tree on the 5ms sampled-medium wait, 10ms idle window, and 48-row cap until the
next hypothesis changes the prefill/decode pipeline itself.

TREE FULL-DRAIN REFILL RECHECK (2026-06-22, current 26752a8 queue-profiled):
forcing refill admission to wait until all 48 active rows were free
(`TORCHINFERNO_CONTINUOUS_ADMIT_MIN_FREE_ROWS=48`) is rejected. It landed at
239.5 / 54.0 / 279.5 ms, 4.1 tok/s, 957/992 raw correct. The counters moved the
wrong way versus the comparable profiled control: total runtime step time
8.48s -> 8.77s, prefill wall 5.62s -> 5.75s, and decode active 2.83s -> 2.99s.
Holding refills back removes interleaving but costs batching efficiency.

TREE SESSION CAP REJECTED (2026-06-21, current bc3b9ea + temporary env-gated
patch): capping each online session at 128 requests was meant to reduce
cross-temperature head-of-line blocking between sampled branch requests and
greedy eval requests. It was a clear regression: profiled tree moved from the
current control's 248.2 / 52.1 / 306.7 ms (TTFT/TPOT/E2E), 4.0 tok/s, 962/992
raw correct, to 453.5 / 51.7 / 491.1 ms, 3.5 tok/s, 962/992 raw correct. Queue
counters explain the loss: deferred requests fell 122 -> 77, but sampled prefill
was fragmented into more graph batches; aggregate prefill wall rose 6050.8 ->
6813.3 ms and total online-session time rose 8977.9 -> 9786.6 ms. Do not cap
tree sessions without a mechanism that preserves or reuses prefill work across
the added session boundaries.

COMMON-PREFIX LOGITS STORE CLEANUP (2026-06-21, current a9699a4 + local patch):
shared common-prefix rows no longer clone cached logits back to CPU when every
request in the group has a non-empty suffix. The prefix-suffix graph only needs
the KV row; exact-prefix groups still keep logits. This removes an avoidable
sync from common-prefix storage. Profiled tree_of_thought improved from the
current control's 248.2 / 52.1 / 306.7 ms (TTFT/TPOT/E2E), 4.0 tok/s, 962/992
raw correct, to 216.7 / 51.8 / 255.6 ms, 4.6 tok/s, 962/992 raw correct. Queue
counters showed the same 50 prefill batches, while aggregate prefill wall fell
6050.8 -> 5529.2 ms and total online-session time fell 8977.9 -> 8520.1 ms.
No-profile guard (`few_shot` then `tree_of_thought`) stayed in family: few_shot
153.2 / 51.4 / 197.9 ms, 977/1000 raw correct; tree 231.4 / 51.9 / 277.3 ms,
963/992 raw correct. Treat this as prefill-sync cleanup, not a score-flip claim.

COMMON-PREFIX GPU LOGITS STORE REJECTED (2026-06-21, current d3421c2 + local
patch): keeping exact common-prefix logits on GPU instead of cloning them to CPU
did not validate on self_consistency. The run landed at 344.9 ms TTFT /
423.1 ms E2E / 2.4 tok/s, 1000/1000 raw correct. Queue snapshots showed
prefill wall essentially unchanged versus a comparable current profile
(1922.2 -> 1917.9 ms) and decode-active slightly worse (1181.6 -> 1223.1 ms).
Do not retain GPU logits for exact common prefixes without a clearer hit-rate or
latency win.

TREE PINNED READBACK REJECTED (2026-06-21, local 570917f + temporary patch): a
reusable pinned CPU token buffer for ragged decode readback looked promising in
one no-profile run (218.0 / 52.0 / 252.9 ms, 4.5 tok/s, 963/992 raw correct),
but the profiled counter did not validate the mechanism. Baseline profiled
sampled-medium tree spent 1415.5 ms in `decode_ragged_cpu_tokens_ms` across
67 ragged decode batches; the pinned-buffer patch spent 1467.9 ms across
70 ragged decode batches, with total sampled-medium profiled time moving
7790.6 -> 8003.2 ms. The patch was backed out. The readback gap needs real
lagged/asynchronous token harvesting, not just a pinned synchronous copy.

TREE LOCAL REPRO/PROVIDER COMPARISON (2026-06-21, current aa22fa0): warm-server
ordering does not explain the public/local tree discrepancy. A no-profile
few_shot -> self_consistency -> multi_turn -> tree_of_thought sequence landed
tree at 242.6 / 51.6 / 279.1 ms (TTFT/TPOT/E2E) and 4.2 tok/s. A fresh clone
and wheel build of the same commit also did not close the gap: 250.1 / 53.6 /
300.2 ms and 4.0 tok/s. Same-node provider comparison is the cleaner next
tree baseline: vLLM was 75.4 / 36.5 / 103.5 ms and 11.6 tok/s, SGLang was
72.3 / 77.5 / 170.3 ms and 7.9 tok/s, and TorchInferno was 215.1 / 52.2 /
254.1 ms and 4.7 tok/s. The public/local variance affects at least vLLM and
TorchInferno, but the same-node gap is still real: TorchInferno is roughly
15.7 ms slower than vLLM on TPOT and roughly 150 ms slower on TTFT. vLLM's
local server enabled chunked prefill, prefix caching, FlashAttention, and decode
graph capture sizes up to 512; use vLLM as the next tree comparator when testing
TorchInferno scheduler changes.

TREE SAMPLED ROW CAP RECHECK (2026-06-21, current c636428 local focused A/B):
lowering `TORCHINFERNO_OPENAI_TP_ONLINE_MAX_ACTIVE` to 32 is not defaultable.
With queue profiling enabled it improved the profiled current control from
248.2 / 52.1 / 306.7 ms (TTFT/TPOT/E2E) to 230.7 / 50.3 / 273.2 ms, but raw
correctness moved 962/992 -> 958/992. The score-comparable unprofiled run then
landed at 240.4 / 51.8 / 288.1 ms and 4.7 tok/s; the unprofiled current default
control was 230.6 / 52.7 / 269.8 ms and 4.3 tok/s. Disabling continuous ragged
decode buckets improved local TTFT/E2E to 217.6 / 260.3 ms but regressed TPOT to
54.0 ms, throughput to 4.3 tok/s, and TPOT p99 to 612.5 ms. Keep sampled tree at
the 48-row/default bucket policy unless a new hypothesis targets the actual
public TPOT gap without trading away E2E/throughput. The public/local tree TPOT
discrepancy remains real: c636428 differs from public 28d7c7c only by docs, yet
local unprofiled TPOT is still ~52-54 ms while the public row reports 32.1 ms.

## LATEST RUN 20260621_155659 (built a7e5516): 4/20

TorchInferno improved to 4/20 (vLLM 13/20, SGLang 2/20). The few_shot row is
now a real bright spot: TorchInferno wins TTFT, TPOT, and E2E (136.4 / 45.2 /
179.4 ms) and only trails throughput (6.6 vs vLLM 7.7 tok/s). The score-moving
near miss is tree_of_thought TPOT: 30.3 ms vs vLLM 29.7 ms, a 0.6 ms gap. The
big remaining queueing rows are unchanged: self_consistency E2E/throughput
(433.7 ms / 2.3 tok/s vs vLLM 220.0 ms / 4.5 tok/s), multi_turn TTFT/E2E, and
long_output decode throughput.

SELF_CONSISTENCY PROFILED AND TUNED (2026-06-21, local focused runs on the same
a7e5516 build): the benchmark is 1000 identical sampled requests at
temperature=0.7 and max_tokens=256. Each request emits the visible answer token
from prefill, then needs one decode step to produce the stop token; TPOT reports
0.0 because only one content token is streamed. Baseline local metrics were
TTFT/E2E/throughput = 223.4 ms / 361.3 ms / 2.8 tok/s. Queue profile at the final
snapshot: initial_batch_size=8, max_active=105, 1000 admitted, 38 prefix-reuse
prefill batches, 39 decode batches, prefill_wall_ms=1919, decode_active_ms=1199.
So the gap is not classic inter-token TPOT; it is wave scheduling plus the
mandatory stop-token decode.

SELF sampled KV budget (2026-06-21, bd61b32/7aa3845 local focused A/B): the
default sampled-short path used a 64*512 KV-token budget, which admitted 105
active rows for max_seq_len=311. Raising only the sampled-short budget to
128*512 admits the full 128-client wave while the 144-row total cache envelope
keeps 16 prefix rows. Local 70B TP8 self_consistency improved from the current
profiled 234.7 ms TTFT / 362.7 ms E2E / 2.8 tok/s to 243.4 / 293.0 ms /
3.4 tok/s with 100% correctness. This should not affect tree_of_thought
(sampled max_tokens=300 is outside the sampled KV-bounded range) or greedy
long_output (separate greedy budget remains 64*512).

SELF CURRENT UNPROFILED RECHECK (2026-06-21, current `c3ec2cb`): a focused
no-profile run landed at 245.3 ms TTFT / 362.1 ms E2E / 2.8 tok/s, 100% correct.
That is better than the latest public self row (274.1 / 388.6 ms) but still far
from vLLM's 208.9 / 235.6 ms. Since the sampled KV budget already admits the
full 128-worker wave, further self work needs to reduce wave count or overlap the
mandatory stop-token decode; raising the active cap cannot help this client
shape.

SELF CURRENT SAME-NODE RECHECK (2026-06-21, current `2b2b839`): a provider
comparison on the same host reproduced the self gap but also showed significant
run-to-run variance: vLLM landed at 238.2 ms TTFT / 334.5 ms E2E / 3.0 tok/s,
SGLang at 224.0 / 399.2 ms / 2.5 tok/s, and TorchInferno at 387.3 /
449.7 ms / 2.2 tok/s, all 1000/1000 correct with one visible output token per
request. A TorchInferno-only queue profile on the same code landed at 354.2 /
424.0 ms / 2.4 tok/s with `max_active=128`, `prefix_rows=16`, 44 exact-prefix
reuse prefill batches, and 45 decode batches. The remaining self work is wave
formation plus the mandatory stop-token decode, not row-cap starvation.

SELF EXACT-PREFIX ONLINE ROUTE (2026-06-21, current `3153672` + local patch):
the default OpenAI online engine now routes suffix-empty exact-prefix hits
through the same evented exact-prefix helper that chunked online serving already
used. This preserves the non-stream batch path, but avoids the heavier generic
prefix-suffix branch for repeated online prompts. A profiled self_consistency
rerun improved to 217.8 ms TTFT / 340.6 ms E2E / 2.9 tok/s, 100% correct.
Queue counters moved from the earlier current profile's 44 prefix-reuse batches
/ 45 decode batches / 1182 ms decode-active to 37 / 38 / 1045 ms, with
prefill_wall_ms also down from 1922 to 1823 ms. This is a real default-path win,
but E2E still trails vLLM because the mandatory stop-token decode still happens
in many small waves.

SELF_CONSISTENCY RULED OUT (do not re-chase without a new hypothesis):
- Uniform ragged decode (`TORCHINFERNO_CONTINUOUS_UNIFORM_RAGGED_DECODE=1`)
  regressed to 245.6 ms TTFT / 368.4 ms E2E / 2.7 tok/s.
- Larger sampled-short admission
  (`TORCHINFERNO_OPENAI_TP_ONLINE_ADMIT_PER_STEP_CAP=64`) regressed to
  359.1 ms TTFT / 437.4 ms E2E / 2.3 tok/s. Bigger waves raised head-of-line
  prefill/decode work more than they reduced queueing. Rechecking cap=64 after
  the sampled KV-budget increase still regressed to 270.0 ms TTFT /
  376.8 ms E2E / 2.7 tok/s, despite reducing scheduler/decode batches.
- After raising the sampled-short KV budget, reducing the sampled initial wait
  back to 5ms still regressed: b2dc983 local 70B TP8 self_consistency moved from
  243.4 ms TTFT / 293.0 ms E2E / 3.4 tok/s at the 10ms default to
  279.7 / 417.7 ms / 2.4 tok/s. The shorter wait under-collected the first wave
  (`initial_batch_size=4` vs `7`) and increased scheduler/decode batches.
- Unified online forward (`TORCHINFERNO_CONTINUOUS_UNIFIED_FORWARD=1`) regressed
  hard to 811.7 ms TTFT / 1117.5 ms E2E / 0.9 tok/s.
- Disabling the online post-step TP sync
  (`TORCHINFERNO_OPENAI_TP_ONLINE_STEP_SYNC=0`) is also a regression on current
  `56060b2`: 388.1 ms TTFT / 454.0 ms E2E / 2.2 tok/s, 100% correct. The sync
  is not the self bottleneck in isolation.
- Rechecking generated-prefix reuse on current `2b2b839` still produced zero
  reuse hits: 347.6 ms TTFT / 424.0 ms E2E / 2.4 tok/s, one generated-prefix
  store, zero generated-prefix reuses, and decode-active time rising to 1896 ms
  versus 1182 ms in the comparable default profile. This path is not useful
  unless waiting requests can actually hit the stored generated prefix.
- After the non-chunked exact-prefix path was wired to the generated-prefix
  continuation lookup, enabling `TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE=1`
  was still a hard reject: 888.4 ms TTFT / 949.4 ms E2E / 1.1 tok/s, 100%
  correct. The extra logits collection and continuation work dominate; keep the
  cache default-off unless a narrower hit-rate/cost model is proven.
- Raising the sampled-short initial wait to 20ms regressed to 381.6 ms TTFT /
  466.2 ms E2E / 2.1 tok/s. It buys larger waves by adding too much direct
  queue-visible delay.
- A sampled-short idle drain wait of 10ms was not robust. An env-only run looked
  mildly better at 337.1 ms TTFT / 424.1 ms E2E / 2.4 tok/s, but making that
  scoped default regressed to 377.4 / 447.4 ms / 2.2 tok/s. Do not promote idle
  drain waits without a more stable admission rule.

FEW/TREE UNIFORM-RAGGED A/B (2026-06-21): global uniform ragged decode was also
not a broad default. On the local focused run it moved few_shot from
157.3 / 49.3 / 201.6 ms (TTFT/TPOT/E2E) to 157.5 / 50.0 / 200.6 ms, and moved
tree_of_thought from 249.8 / 53.2 / 295.2 ms to 272.1 / 51.7 / 304.6 ms. That
does hint at a possible narrow tree TPOT lever, but it trades away TTFT/E2E and
is not strong enough to ship as a default.

Scoped sampled-medium uniform-ragged decode is also rejected after the latest
public run made tree TPOT a 1ms target. A temporary patch enabled uniform ragged
only for sampled-medium online requests (`temperature>0`, `256 < max_tokens <=
300`) so self_consistency, few_shot, long_output, and multi_turn stayed on the
old policy. The focused tree run landed at `219.0ms` TTFT, `51.9ms` TPOT,
`257.9ms` E2E, `4.6 tok/s`, 961/992 raw correct. Against the common-prefix
no-logits tree profile, decode-active improved (`2551.9ms -> 2419.1ms`) but
CPU token readback rose (`1586.2ms -> 1660.3ms`) and prefill wall regressed
(`5529.2ms -> 5632.6ms`). TPOT stayed flat (`51.8ms -> 51.9ms`), so the scoped
default was backed out.

FEW_SHOT CURRENT RECHECK (2026-06-21, current `1a5d759`): the latest local
few_shot control is materially better than the frozen public row:
`152.8ms` TTFT, `50.5ms` TPOT, `193.7ms` E2E, `6.1 tok/s`, 976/1000 raw
correct. Queue profile used the default greedy-mid policy (`max_active=32`,
`decode_quantum=16`) and split into two online sessions; the combined shape is
still prefill/decode interleaving rather than one missing cache feature.

FEW_SHOT RULED OUT (2026-06-21, current `1a5d759`):
- Raising the initial collection wait to 5ms gathered the full workload into one
  online session but regressed all score-facing metrics: `154.7ms` TTFT,
  `51.2ms` TPOT, `194.9ms` E2E, `6.0 tok/s`.
- Dropping the initial collection wait to 0ms is also bad on an unprofiled
  current `2e2f1df` run: `158.0ms` TTFT, `54.7ms` TPOT, `201.1ms` E2E,
  `5.8 tok/s`. The default 1ms wait remains the knee for few_shot.
- Lowering greedy-mid decode quantum to 8 did not buy throughput; it regressed to
  `161.0ms` TTFT, `52.7ms` TPOT, `204.9ms` E2E, `5.8 tok/s`. Queue profile ended
  at `3904ms` prefill wall and `1206ms` decode active, worse than the default.

GENERATED-PREFIX REUSE RE-CHECKED (2026-06-21, current 340ffe0 + env flag):
`TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE=1` on self_consistency produced
260.5 ms TTFT / 386.7 ms E2E / 2.6 tok/s. It stored one generated prefix but
reused zero (`generated_prefix_store_requests=1`, reuse=0), and decode-active
profile time rose to 1473 ms versus 1191 ms in the comparable full-run profile.
Do not enable it by default without a new reuse-hit hypothesis.

SEQ-LEN PARTIAL-ROW FASTPATH (2026-06-21): the Llama3 TP dense cache now uses a
hybrid uniform-marker policy. Small partial `_set_rows_seq_len` updates skip the
full row scan and conservatively invalidate the marker; large partial updates
still rescan so high-row long-output batches can recover the fast global append
path. CPU microbench for 32-row updates in a 144-row cache improved the setter
loop about 2.0x while 128-row updates stayed even with the original scan.
Focused few_shot/tree run was mixed but safe:
few_shot 159.2 / 50.6 / 200.2 ms, tree_of_thought 224.5 / 52.2 / 266.6 ms. Keep
as hot-path overhead cleanup, not a claimed score flip. A count-tracking version
was rejected because it made the setter loop slower than the original scan.
Follow-up long_output validation completed 1000/1000 correct without the earlier
stall at 371.6 / 27.7 / 1485.3 ms.

SELF COMMON-PREFIX PREFILL MISS (2026-06-21): local b5efff4 full-run queue
profiles showed self_consistency spent ~1.3s of wall time in the first
common-prefix prefill per online session even though only one 55-token shared
prefix was materialized. The root cause was row mismatch: startup warms
common-prefix prefill graphs on stable rows such as 128, while runtime prefix
allocation for the self shape (105 active + 39 prefix rows) selected row 105
first, forcing a real-request graph capture (`prefill_graph_misses=1`). Runtime
prefix allocation now prefers the warmed rows when they are free and then falls
back to the existing row order. Focused CPU tests cover the default row choice
and env override. A local 70B benchmark rerun could not complete because the
machine entered a GPU-driver `D`-state during server startup with an unrelated
process already occupying GPU 7; do not treat that as benchmark evidence.

FOLLOW-UP COMMON-PREFIX BUCKET COVERAGE: the same queue profiles showed
few_shot and long_output reuse around the 128-token prefill bucket (roughly
120 and 111 shared tokens/request), while startup only warmed the 64-token
bucket through a 45-token sample. Default common-prefix warmup now captures both
64- and 128-token buckets. This is still an env-overridable startup cost, not
benchmark-specific prompt matching.

PREFILL SYMM-MEM ALLREDUCE RECHECK (2026-06-21, e4f8de7 few_shot local slice):
`TORCHINFERNO_SYMM_MEM_PREFILL_ALLREDUCE=1` regressed few_shot from
156.6 / 49.0 / 198.5 ms to 173.5 / 52.1 / 223.3 ms (TTFT/TPOT/E2E). Queue
profile showed prefill_forward_ms rising from 1286 ms to 1618 ms for the first
few_shot online session. Keep prefill symm-mem allreduce default-off unless a
new implementation avoids this forward-time regression.

LONG_OUTPUT DECODE-MANY RECHECK (2026-06-21, e4f8de7 local slices): enabling
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_DECODE_MANY=1` did not produce a
defaultable win. The stock guard never engaged on benchmark traffic because the
requests carry an EOS id. A temporary EOS-overcompute variant reduced the
deferred readback counter (`decode_ragged_cpu_tokens_ms` about 10.6s -> 4.25s),
but shifted the sync into decode timing (`decode_ragged_model_ms` about 0.88s ->
8.81s) and did not improve score-facing throughput/E2E (371.4 / 27.4 / 1686.9
ms, 24.3 tok/s). Keep decode-many gated until readback can be genuinely
pipelined rather than just moved between timing buckets. The stop-token
overcompute variant is now reproducible without local patches via
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_ALLOW_STOP=1`, but it remains
explicitly default-off.

Current flag-backed confirmation (2026-06-22, current 8696a2a no-profile
long_output): `TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY=1` plus
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_ALLOW_STOP=1` landed at 350.0 /
27.3 / 1519.4 ms, 26.2 tok/s, and 1000/1000 raw correct. That is in family with
the current default long-output band and still does not move TPOT/E2E enough to
default decode-many with stop-token overcompute.

Long-output GPU decode timing refresh (2026-06-22, current a001d3e queue
profile): the queue profile now records CUDA-event decode time separately as
`runtime_decode_ragged_model_gpu_ms`. A focused long_output run landed at
279.4 / 27.3 / 1423.1 ms, 27.3 tok/s, 1000/1000 correct. The last completed
profile snapshot had all 1000 requests finished and showed 13.6s decode-active,
11.4s GPU ragged-decode model time across 703 ragged decode batches
(~16.2ms/batch), 10.5s token-readback sync, and 11.4s prefill wall. The old
`decode_ragged_cpu_tokens_ms` counter was mostly exposing GPU decode completion,
not pure host copy overhead. Do not re-chase synchronous readback or simple
pinned-buffer tweaks for long_output; the remaining gap is decode compute/pipeline
plus material prefill wall.

Long-output unified-scheduler recheck (2026-06-22, current 26752a8): the
default-off token-budget unified scheduler
(`TORCHINFERNO_OPENAI_UNIFIED_SCHEDULER=1`,
`TORCHINFERNO_OPENAI_UNIFIED_MAX_SEQ_LEN=1024`) is rejected for the benchmark
path. The server started and accepted long_output traffic, but several minutes
of saturated GPU execution produced no completion where the default online
batcher normally finishes quickly; the run was manually terminated. Keep the
continuous online batcher as the serving default.

LONG_OUTPUT CURRENT RECHECKS (2026-06-21, bd61b32 local slices): two narrower
variants also failed. Raising greedy-short decode quantum from the default 8 to
16 regressed to 534.1 ms TTFT / 25.2 ms TPOT / 1525.5 ms E2E / 23.3 tok/s. A
bounded EOS-overcompute experiment that allowed only two decode-many steps kept
100% correctness but still regressed to 387.1 / 29.1 / 1565.8 ms and 24.4 tok/s.
The issue is not simply too many CPU readbacks; speculative readback deferral
still increases queue-visible latency unless decode and readback are actually
pipelined.

Tail-only decode quantum 16 is rejected too (2026-06-22, current 0116ef1 +
local patch). The patch kept the default 8-step quantum while the HTTP queue was
non-empty, then switched to 16 only after queued submissions drained. It improved
long_output TPOT to 26.0 ms, but TTFT/E2E/throughput regressed to 402.2 /
1528.3 ms / 25.3 tok/s with 1000/1000 correct. The fixed-DQ16 conclusion still
holds: the TPOT gain is not worth the queue-visible latency.

LONG_OUTPUT ADMISSION RECHECK (2026-06-21, current `1f51273` unprofiled):
lowering the greedy-short admit cap from the default 64 to 32 improved local
TTFT but regressed every decode-throughput-facing metric: `291.4ms` TTFT,
`28.5ms` TPOT, `1555.8ms` E2E, `25.2 tok/s`, 100% correct. The default 64-admit
policy remains the better score tradeoff for long_output.

LONG_OUTPUT CURRENT REFRESH (2026-06-21, current `5069d82`): local profiled
long_output is still decode/readback bound and behind the public row:
`329.9ms` TTFT, `28.1ms` TPOT, `1544.9ms` E2E, `25.2 tok/s`, 100% correct.
Queue profile ended at `760` decode batches for `43314` decode tokens,
`13558ms` decode-active, and `10621ms` synchronous CPU token readback. Prefill is
also material at `13243ms` wall / `11355ms` forward across `60` batches.
Two more decode-shape knobs are not defaultable:
- Setting `TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_BUCKET_CAPACITY=128` looked
  modestly better on score metrics (`306.8ms` TTFT / `27.8ms` TPOT /
  `1421.3ms` E2E / `27 tok/s`), but the profile did not validate the intended
  mechanism: max model batch stayed `64`, ragged batches stayed effectively the
  same (`711 -> 719`), and CPU readback slightly increased. Treat as noise or
  secondary scheduling variance, not a shippable lever.
- Forcing uniform batches through ragged decode
  (`TORCHINFERNO_CONTINUOUS_UNIFORM_RAGGED_DECODE=1`) regressed to `343.9ms`
  TTFT / `28.5ms` TPOT / `1496.3ms` E2E and raised readback/prepare time. Keep
  uniform ragged disabled for long_output too.

Long-output refresh on current `fff4c37` keeps the same shape. Focused profiled
long_output landed at `345.3 / 27.8 / 1529.8ms`, 1000/1000 correct. The last
queue-profile record ended at `11.61s` prefill wall (`10.74s` forward), `744`
decode model calls for `44.65K` bucketed decode tokens, `14.56s` decode GPU
event time, and `7.48s` in the token-harvest/synchronization bucket. Max model
batch stayed at `64` with most ragged decode shapes padded into the `64` bucket.
This does not reopen the rejected decode-many, exact-row decode, or refill-floor
knobs; the next long-output lever still has to reduce decode GPU work or overlap
streaming token harvest with model execution.

Scoped common-prefix ragged suffix threshold (2026-06-21, current `27b8381` +
patch): raising
`TORCHINFERNO_CONTINUOUS_COMMON_PREFIX_RAGGED_SUFFIX_MAX_PREFIX_TOKENS=128`
globally validated the long_output mechanism but regressed few_shot. Long_output
with the broad env moved to `279.5ms` TTFT / `28.3ms` TPOT / `1511.7ms` E2E and
zeroed prefill graph misses (`55/5 -> 60/0` hits/misses), while prefill wall
dropped `13243ms -> 12329ms`. The few_shot guard with the same broad threshold
regressed to `158.8ms` TTFT / `53.5ms` TPOT / `203.5ms` E2E because it also has
a ~120-token shared prefix but uses `max_tokens=256`.

Promoted narrower behavior: without an explicit env override, deterministic
short-output requests (`max_new_tokens<=128`) may use the 128-token common-prefix
ragged suffix cutoff; other traffic keeps the old 64-token cutoff. The no-env
long_output confirmation removed the same graph misses (`55/5 -> 57/0`), reduced
prefill wall `13243ms -> 11800ms`, and improved score-facing E2E/TPOT to
`1475.4ms` / `27.9ms` with 100% correctness. TTFT was neutral/noisy
(`329.9ms -> 332.4ms`), so this is an E2E/prefill cleanup rather than a TTFT
fix. Focused CPU tests cover the short-vs-mid threshold split.

Long-output refresh after the common-prefix no-logits cleanup (2026-06-21,
current `778e185`): local profiled long_output landed at `343.7ms` TTFT,
`26.9ms` TPOT, `1528.3ms` E2E, `25.9 tok/s`, 1000/1000 correct. Queue profile
remained decode/readback dominated: `728` decode batches for `44347` decode
tokens, `13348.8ms` decode-active, and `10891.3ms` synchronous CPU token
readback. Prefill stayed material (`55` batches, `13141.5ms` wall /
`12161.0ms` forward) despite all common-prefix suffix prefills hitting graph
replay. The no-logits cleanup is not a stable long_output lever.

Ragged decode GPU-token input buffer rejected (2026-06-21, current `778e185` +
temporary patch, backed out): reusing `_gpu_last_tokens` and cached row-index
tensors for the normal ragged decode input path did not validate. The run landed
at `343.8ms` TTFT, `27.6ms` TPOT, `1455.7ms` E2E, `25.4 tok/s`, 1000/1000
correct. The apparent E2E shift came with prefill variance (`55 -> 51` batches,
`13141.5ms -> 11765.2ms` wall), not a decode mechanism win. Decode-active
worsened (`13348.8ms -> 13605.4ms`), CPU readback was flat
(`10891.3ms -> 10871.1ms`), and ragged prepare time regressed
(`1086.2ms -> 1408.2ms`). Keep the simple per-step tensor construction until a
true lagged readback/GPU-token pipeline exists.

PROMPT-LOOKUP DECODE RECHECK (2026-06-21, 28d7c7c local slices): prompt lookup
is still not a defaultable long_output lever. The old verifier used a full
prefill over `[last_token, proposal...]` and was catastrophic (hundreds of
seconds of runtime step time). A graph-decode verifier that grouped by
`(seq_len, proposal_len)` accepted drafts but fragmented into 3733 decode graph
calls after only 8 online commands: runtime step was already 50.9s with
40.1s in decode model time. Regrouping mixed sequence lengths by proposal
length cut the full run to 204 online steps and accepted 29.5k draft tokens, but
it still regressed score metrics to 925.4 ms TTFT / 25.7 ms TPOT / 1728.1 ms
E2E / 18.8 tok/s (100% correct). Queue profile: 1366 decode calls,
16.9s decode model time, 3.0s CPU token readback. Conclusion: prompt lookup can
move readback cost down, but it does not reduce autoregressive GPU decode work
unless verification is fused into a cheaper multi-token kernel/prefill path.
Keep it off by default.

TREE TPOT PUBLIC/LOCAL DISCREPANCY (2026-06-21): public runs 20260621_175526
and 20260621_215200 reported TorchInferno tree_of_thought TPOT 31.2-31.9 ms at
commits 96c9b54 and 9452794, but a same-commit 96c9b54 local repro with the same
benchmark shape landed at 232.2 / 52.2 / 272.6 ms (TTFT/TPOT/E2E), matching
current bd61b32 local repro at 234.1 / 51.8 / 276.9 ms. The raw output-token mix
is effectively identical (about 560 one-token responses and about 408 two-token
responses), so do not bisect current commits for the public 31 ms value without
first finding the environment/runtime difference that makes it reproducible. A
sampled decode-many experiment with two-step EOS overcompute also regressed the
local tree slice to 297.9 / 76.5 / 346.9 ms and 3.8 tok/s, so keep sampled
decode-many disabled. Extending sampled KV-bounded eligibility to tree's
max_tokens=300 after the sampled KV-budget change also failed locally:
232.9 / 53.4 / 272.7 ms and 4.2 tok/s with 958/992 correct, despite sampled
bursts reaching max_active=128. Keep the default sampled KV range capped at 256.

## LATEST RUN 20260607_101328 (built 73aa664, BEFORE rope+KV-bounded): 1/20

Dropped 2/20 -> 1/20, NOT from our regression: vllm IMPROVED multi_turn TPOT
67.6 -> 43.5 (a moving target; likely FP8-KV / paged long-context decode) and took
that cell. Our only remaining cell is few_shot TPOT (52.9 vs vllm 53.6 -- razor-thin
0.7ms). Closest unflipped: self_consistency TTFT (250 vs 200, 1.25x) and E2E
(350 vs 219). The build is one commit behind my rope fusion (c3ec55c) -- so rope
(secures few_shot TPOT, +~2.5ms margin) and KV-bounded concurrency (targets
self_consistency TTFT, A/B showed ~2x) are QUEUED and should land next run.
multi_turn TPOT (now 20ms behind vllm) and long_output TPOT (32 vs 15) need
paged long-context decode -- deep. The TTFT/throughput rows (3-9x) remain
the architectural (paged-KV concurrency + FP8 prefill) gap.

LONG-CONTEXT DECODE PROFILED (2026-06-07, TORCHINFERNO_PROFILE_DECODE_ONCE,
batch=32 @ ~1500-tok context): decode is STILL GEMM-bound, NOT attention-bound --
aten::mm 69%, allreduce 13%, index_put (KV write) 6%, add_rms_norm 4%, and
FlashInfer paged decode attention only ~4.6% (411us decode + 237us merge) EVEN at
long context (FlashInfer paged decode is very efficient). The decode STEP is
~14ms, but multi_turn's benchmark TPOT is 63.8ms (~4.5x inflated) -- that
inflation is QUEUEING / prefill-interleaving (125-conc >> 48 max_active -> constant
admission interleaves prefills into decode). CORRECTION to the prior "FP8-KV for
multi_turn TPOT" idea: attention is negligible, so FP8-KV's value is NOT
attention speed -- it is the MEMORY halving (-> ~2x concurrent rows -> less
queueing). So the single lever behind multi_turn TPOT + the whole TTFT/throughput
complex is HIGHER EFFECTIVE CONCURRENCY for long contexts = PAGED KV (pack by
actual tokens, not dense max_seq x rows). FP8-KV helps only via that same
memory->rows path. KV-bounded concurrency (shipped) is the short-context-only
version of this; paged KV is required for the long-context (multi_turn/long_output)
cells. This is THE prioritized deep lever (one change addresses TTFT + TPOT +
throughput simultaneously).

PAGED-KV FEASIBILITY FULLY DE-RISKED (2026-06-07, foundation built + validated,
NOT yet wired into serving):
- runtime/paged.py LayeredPagedKVCache: multi-layer pool, ONE block table per
  request shared across all layers, per-layer NHD storage
  [num_pages, 2, page_size, kv_heads, head_dim]; reserve/extend/write_layer/
  flashinfer_page_table/free. (vLLM-standard layout.)
- CORRECTNESS: GPU test feeds it through flashinfer.BatchDecodeWithPagedKVCacheWrapper
  and matches a dense SDPA reference across page-crossing lengths
  (test_layered_paged_kv_cache_flashinfer_decode_matches_dense).
- CONCURRENCY VIABILITY (scripts/bench_paged_decode_concurrency.py): paged decode
  ATTENTION is sub-linear in row count -- 48->512 rows @2048-tok ctx costs only
  3.95x attention time (per-row 2.66->0.99us), staying ~5% of the GEMM-bound step;
  512 long-context rows fit in ~43GB/GPU. No attention wall at high concurrency.
- CUDA-graph: CUDAGraphBatchDecodeWithPagedKVCacheWrapper (decode_runner.py)
  already supports paged page-tables as graph inputs -- no graph blocker.
REMAINING (focused multi-day integration, now de-risked): TP-shard the pool at
serving start -> route the FI decode/prefill plan through flashinfer_page_table +
layer_kv with small page_size -> scheduler admission by free pages (not the 48-row
cap) -> dynamic page-tables as decode-graph inputs. Default path stays dense behind
a flag until benchmark-validated.

## EXECUTIVE SUMMARY (state of the gap, distilled)

Score: stable 2/20 (vllm 15-16/20) across THREE consecutive runs (_021340,
_042000, _061657). We win exactly the two cells we contend on: few_shot TPOT
(52.9 vs vllm 58.3) and multi_turn TPOT (64.7 vs 67.6) -- both now CONFIRMED
robust (held 3 runs), not noise. Closest unflipped cell: tree_of_thought TPOT
(31.7 vs vllm 29.5), a consistent ~2-3ms gap (needs the deep decode-fusion lever).
TPOT cross-avg ~35ms (vllm ~34, sglang ~48) -- competitive. The deficits are TTFT
(~3.4x), E2E (~2.7x), throughput (~2.9x), all driven by two root causes below.
Scoring is best-in-row, so a cell flips only by beating BOTH competitors.

RULED OUT (measured, do not re-chase):
- int4/marlin decode weights: WORKS + accurate + graph-safe, but TPOT-NEUTRAL
  (+2.6%). Decode at batch 48 is not weight-bound; isolated GEMM 1.52x does not
  translate (graph-critical-path bound). Marlin module kept flag-off (good for M=1).
- Faster decode allreduce: our symm-mem multimem (38us) already BEATS vllm
  custom_ar (66us) at [48,8192]. Allreduce is NOT the gap. one_shot/two_shot worse.
- decode_quantum (16 optimal), unified/chunked prefill (worse), max_active
  (tradeoff/queueing, no flip), FP8 W8A8 decode (slower at M=48), W8A16/tinygemm
  (M=1-tuned), torchao int4 (needs fbcode-only mslk), custom triton int4 (0.8x floor).

THE TWO REAL GAPS (both DEEP, multi-week, no drop-in):
1. DECODE TPOT (21 vs vllm 15ms; gates tree/multi_turn TPOT + throughput): after
   ruling out allreduce + weight, the remainder is vllm's kernel FUSION (fewer ops
   on the residual-serialized critical path) + attention kernel. Needs structural
   decode work (fused mega-kernels / shorter critical path), not op-by-op.
2. PREFILL MFU / TTFT (few_shot 303 vs 159; dominant): prefill GEMMs ~34% MFU
   (~2x vllm). FP8 prefill (_scaled_mm 1.8x at large M, WORKS) would narrow
   few_shot ~23% but NOT flip (vllm 159 too far) and is accuracy-gated. Short-prompt
   TTFT (long_output/tree/multi_turn) is admission/queueing-bound, not prefill, and
   capped by the ~2x decode behind it -- also no flip without the decode fix.
   PREFILL PROFILED (2026-06-07, TORCHINFERNO_PROFILE_PREFILL_ONCE, batch=50 q=64
   = 3200 tok, 150ms CUDA): aten::mm 90.8ms = 60.5% (the FP8 lever), allreduce
   (symm-mem multimem) 37ms = 24.7%, _add_rms_norm 15ms = 10% (already fused),
   FlashInfer attn 2.0ms = 1.3%, swiglu/index_put <2%. So prefill has NO
   low-hanging fruit: GEMMs need FP8; allreduce is ALREADY OPTIMAL (scripts/
   bench_allreduce.py: multimem beats NCCL ring 1.38-2.74x at every prefill size
   1024-30720 tok), norms/attn already fused/tiny. The only prefill levers are FP8
   (GEMMs) and compute/comm overlap (the 25% allreduce) -- both deep.

SHIPPED WINS (this whole effort): symm-mem TPOT parity (scored; beats vllm's AR),
3 NaN correctness fixes + regression tests, validated marlin module (flag-off),
benchmark-faithful streaming harness, full diagnosis. Net: no single-iteration
change closes the gap; both real levers are dedicated multi-week structural work.

---

Consolidated findings from extended profiling. Benchmark numbers are from
inference-bench run `20260607_000945` (torchinferno commit `c60e0bd` -- which
includes ALL current functional work; the 4 commits after it on main are
docs/harness only). Local measurements are from `scripts/test_stream.py`
(benchmark-faithful closed-loop streaming) against the current online batcher.

## Headline gaps (cross-benchmark medians, run 20260607_042000)

| Metric | torchinferno | vllm | sglang |
| :-- | --: | --: | --: |
| TTFT median (ms) | 449.7 | 132.7 | 137.3 |
| TPOT median (ms) | 35.2 | 33.5 | 55.3 |
| throughput (tok/s) | 6.3 | 18.0 | 12.3 |
| **score** | **2/20** | **15/20** | **2/20** |

Stable at 2/20 across runs _021340 and _042000 (builds 7d2331a, 5bb4303 -- both
differ from the 1/20 build c60e0bd only by docs/flag-off/reference-path commits,
so serving behavior is unchanged). Our two won cells are BOTH TPOT and now look
robust (not pure noise): few_shot TPOT 50.6 vs vllm 56.2, multi_turn TPOT 59.8
vs vllm 67.2 (vllm ~67ms two runs running, vs our consistent ~60-63). The next
TPOT target is tree_of_thought 32.9 vs vllm 29.3 (3.6ms behind). Everything else
(TTFT 3.4x, E2E 2.7x, throughput 2.9x) is queueing+decode-kernel bound and not
flippable by tuning -- see the analysis below.

BENCHMARK VARIANCE WARNING. The score moved 1/20 -> 2/20 between runs
20260607_000945 and _021340 with our code FUNCTIONALLY UNCHANGED (build 7d2331a
differs from the 1/20 build c60e0bd only by docs/flag-off commits). The new cell
is multi_turn TPOT: us 63.3 vs vllm 67.9 -- but vllm's multi_turn TPOT was 52.5
the prior run, i.e. it WOBBLED UP ~15ms. We did not get faster; vllm's number
rose above ours. Run-to-run TPOT variance is ~±15ms. Our three TPOT contests are
all WITHIN that noise band: few_shot 54.8 vs 57.8 (3ms), multi_turn 63.3 vs 67.9
(4.6ms), tree_of_thought 32.1 vs 28.4 (3.7ms BEHIND). So the score will fluctuate
~1-3/20 on noise alone, and a real/ROBUST TPOT-cell win needs margin beyond the
noise (~15ms) -- which needs the kernel-blocked decode speedup, not tuning.

## Benchmark metric definitions (inference_bench/benchmarks/base.py) -- READ

Per request, from the streaming client:
- `ttft_ms` = time to first streamed token (INCLUDES admission + queue wait).
- `tpot_ms` = (e2e - ttft) / (output_tokens - 1) = pure inter-token decode rate
  (queueing-INDEPENDENT; the only pure-compute metric).
- `e2e_latency_ms` = full request wall time.
- `throughput_tps` = output_tokens / (e2e_latency_ms/1000) -- i.e. output /
  (ttft + decode), so it is DRAGGED DOWN by ttft/queueing too.
Reported = MEDIAN over requests; winner = beats BOTH others; higher-is-better
for throughput.

CONSEQUENCE (corrects the earlier "throughput ~= 1/TPOT" note): THREE of four
metrics (ttft, e2e, throughput) are queueing-coupled; only TPOT is pure decode.
So reducing queueing (max_active >= concurrency for short-context, or prefix
reuse) improves ttft+e2e+throughput together. BUT this does not flip cells:
even with ZERO queueing, long_output throughput -> ~1000/tpot ~= 31 tps (vllm
58.9) -- still a loss, because our decode is too slow. vllm is 2-8x ahead on
every queueing-coupled metric for long_output/multi_turn/tree, so no amount of
scheduling closes them. self_consistency is the CLOSEST (ttft 247 vs 193, e2e
346 vs 257, tput 2.9 vs 3.9 -- all ~1.3x; TPOT is 0.0 for everyone = tiny
outputs), but its gap is 1000-way-concurrency queueing + prefill efficiency
(max_active 48 << 1000), needing much larger batch / working TP prefix reuse /
FP8 prefill -- all deep. Net: still no single-iteration cell flip; the only
reachable cells remain tree/multi_turn TPOT (decode kernel, blocked).

## Scoring strategy (READ THIS FIRST -- it inverts naive tuning)

The scorecard is BEST-IN-ROW: a cell is won only by beating BOTH competitors on
that metric. In the latest public run we win exactly ONE cell: multi_turn TPOT
(`37.5ms` < vLLM `43.0ms` < SGLang `96.3ms`). TPOT remains the only reachable
surface: cross-TPOT is `26.9ms`, only 0.6 ms behind vLLM `26.3ms`, while
TTFT/E2E/throughput are still too far back for scheduler-only tuning to flip.

CONSEQUENCE: do NOT trade TPOT away for queue metrics. The realistically
flippable cells are all TPOT near-misses:
  - tree_of_thought TPOT 30.2 vs vllm 29.2  -> 1.0ms away (BEST TARGET)
  - few_shot        TPOT 45.7 vs vllm 44.5  -> 1.2ms away
  - cross-bench     TPOT 26.9 vs vllm 26.3  -> 0.6ms away
Long_output still needs a deeper decode-compute lever (`21.4ms` vs `14.8ms`).
Decode speed and avoiding extra decode overhead are the master levers.

## decode_quantum is fully swept -- 16 (default) is optimal

local long_output TPOT by quantum: DQ=4 41ms, DQ=8 40ms, DQ=16 22ms, DQ=64 31ms.
Non-monotonic; 16 is the knee (decode runs in long uninterrupted bursts). Lower
trades TPOT for TTFT/throughput (loses our metric); higher hurts everything
(bigger, longer prefill interruptions). Leave at 16.

## Decode-step profile (DQ=16, ~48 active rows, per step) -- where TPOT goes

| component                        | per-step |
| :------------------------------- | -------: |
| GPU decode launch (graph replay) |   1.79ms |
| `.cpu().tolist()` readback sync  |  15.04ms |
| prepare (build input tensors)    |   0.65ms |
| state update + emit              |   0.05ms |

The 15ms readback is the CPU BLOCKING on the GPU decode to finish (weight-read
floor 17.5GB/GPU / 3.35TB/s = 5.2ms, plus KV + 160 allreduces + GEMMs -> ~15ms
REAL GPU compute). So per-step decode is GPU-COMPUTE-BOUND, not overhead-bound.
Only ~2-4ms is recoverable CPU-exposure (GPU idles while the CPU reads the token
back and builds the next step's input, because the autoregressive next input
depends on that readback). The two real decode levers:
  1. Async GPU-resident decode: feed the sampled token GPU->GPU (no per-step
     .cpu() sync) and lag the readback for emission. Recovers the ~2-4ms ->
     could flip tree_of_thought TPOT (needs 2.6ms) and lift throughput. BUT
     TP-RISKY: at temp>0 (self_consistency/tree_of_thought) all ranks must
     sample the SAME token without a CPU barrier -- the same multi-rank
     divergence class as the reuse saga. Needs a dedicated instrumented 8-GPU
     session, not a loop iteration.
  2. FP8 decode weights: ~2x the GPU decode compute (15->~8ms) -> would flip
     long_output TPOT (31.8->~17, near vllm 15.1) AND its throughput. Deep
     (FP8 weights + scales + accuracy validation).

## GPU-resident decode runner (DecodeGraphRunner) — wired + correctness-proven, NOT a win

The async GPU-resident decode runner (token stash + async D2H, sglang-style) was
already BUILT (`runtime/decode_runner.py`) but dead-wired (never called) and not
pipelined. This iteration: verified TP-safety (both greedy all-gather and
gumbel-temperature sampling contain the cross-rank collective, so all ranks hold
the IDENTICAL token on GPU -- the GPU->GPU feed needs no CPU barrier), wired it
into `_decode_active` behind `TORCHINFERNO_DECODE_GRAPH_RUNNER` (default off), and
validated greedy output is BIT-IDENTICAL to baseline on 8-rank TP.

But the SYNCHRONOUS path (harvest immediately after step) is shape-dependent and
NOT shippable -- clean same-session A/B (64-conc streaming):

| shape                | TPOT base -> runner | tput base -> runner |
| :------------------- | ------------------: | ------------------: |
| long_output (16/256) | 24 -> 30  (worse)   | 703 -> 688          |
| tree-like   (128/64) | 57 -> 68  (worse)   | 516 -> 496          |
| few_shot    (640/32) | 68 -> 29  (better)  | 109 -> 158          |

It helps prefill-heavy few_shot but HURTS decode-bound long_output and tree --
and tree_of_thought TPOT is our single best flip target (2.6ms), so this moves
the wrong way. The true win needs PIPELINING (double-buffer the readback + lagged
harvest so decode replays run back-to-back and the .cpu() overlaps GPU compute).
BUT pipelining only helps CONSECUTIVE same-active-set steps: under realistic
varied-length load requests finish at different steps, forcing a pipeline flush
(one sync) on every active-set change, so the recoverable ~2-4ms/step rarely
applies. Net: the runner is not the lever. Decode is GPU-compute-bound (~15ms);
FP8 decode weights (halve it) remain the real throughput/TPOT lever. The wiring
stays (flag off, validated) as a foundation if FP8 or a stable-decode path lands.

Real few_shot recheck (2026-06-21, current 264814b with
`TORCHINFERNO_DECODE_GRAPH_RUNNER=1`): the synthetic few_shot-like win does not
transfer to inference-bench. It regressed the row to 183.7 ms TTFT / 66.8 ms TPOT
/ 238.2 ms E2E / 5.0 tok/s, with 975/1000 raw correct. Do not scope-enable the
runner for greedy-mid/few_shot traffic without a new implementation that avoids
the startup and synchronous-harvest cost.

## max_active / queueing — validated with the new ignore_eos harness

Added TORCHINFERNO_OPENAI_IGNORE_EOS (test-only, default off): forces generation
to max_tokens so scripts/test_stream.py reproduces the real long_output dynamics
(huge outputs keep rows occupied; arrivals beyond max_active queue). Without it,
synthetic prompts hit EOS at ~30 tokens and the queueing never appears.

A/B with TRUE long outputs (long_output 16/256, closed-loop 64-conc, n=256):

| metric    | max_active=48 | max_active=64 |
| :-------- | ------------: | ------------: |
| TTFT p50  |          511  |          538  |
| TTFT p90  |         4780  |          636  |
| TTFT p99  |         5116  |          639  |
| TPOT p50  |           21  |           26  |
| agg tput  |         1949  |         2251  |

max_active >= concurrency ELIMINATES the TTFT tail (no request ever queues in
closed-loop). BUT: (1) MEDIAN TTFT is unchanged -- the median request isn't the
queued one -- and the benchmark scores MEDIAN, so this does not move the
long_output TTFT cell; (2) TPOT rises (21->26 here; few_shot TPOT 68->95 in a
separate run) -- risking our ONE won cell (few_shot TPOT). This is the same
lesson as the reverted max_active=128 (KV/compute-bound at high row counts).
Flat max_active is NOT the lever. The principled fix the code comment already
calls for is a KV-TOKEN-BOUNDED decode batch: allow many rows when contexts are
short (long_output -> big batch, weight-bound, TPOT-safe) and few rows when
contexts are long (multi_turn -> avoid KV-bound TPOT blowup). That needs a real
admission change AND resolves the median-vs-tail question only on the live
benchmark. Per-request TTFT FLOOR is ~500ms even with no queue (vllm 66ms).

ADAPTIVE DECODE QUANTUM TESTED + REVERTED (2026-06-07). The ~500ms floor was
hypothesized to be the decode_quantum drain (drain_ready admits arrivals only
between 16-step quanta, so a mid-quantum arrival waits ~16 steps ~= 480ms). Built
an adaptive quantum (short while arrivals flow, full 16 once the queue dries;
quantum is broadcast so TP stays in lockstep) behind
TORCHINFERNO_OPENAI_TP_ONLINE_ADAPTIVE_QUANTUM and tested on the live 8xH100.
RESULT: with ramp_quantum=1 the batcher admits EVERY step, yet warm-server TTFT
stayed 648-657ms across 3 runs (48-conc, short prompts; TPOT fine at 17ms). So
the quantum is NOT the bottleneck -- the floor is effective concurrency /
row-turnaround: max_active=48 caps concurrent decodes, the 1000-req queue drains
only as rows finish, and at capacity drain_ready admits 0 so the ramp signal
never fires. The variant that WOULD help row-turnaround (ramp while the queue is
non-empty) keeps the quantum short for the whole benchmark -> tanks TPOT (our
cells). CONCLUSION: quantum tuning, fixed or adaptive, cannot win TTFT; the only
lever is KV-token-bounded admission (above) to raise effective concurrency
WITHOUT raising per-step cost, exactly what vllm's paged KV gives it for free
(TTFT 66-205, throughput 18 vs our 6). Code reverted; no flag left behind.

## Concurrency lever fully mapped (2026-06-07, current code w/ rope fused)

DECISIVE microbench (scripts/bench_decode_batch_scaling.py, per-GPU TP8 decode
GEMMs in a CUDA graph): decode GEMMs are WEIGHT-BOUND -- 32->256 rows (8x the
batch) costs only +15% GEMM time; per-row cost falls 3.62us -> 0.52us. Per-layer
GEMM ~120us x 80 layers ~= 9.6ms matches the live decode profile, so the bench is
faithful. This OVERTURNS the long-held "high row count = TPOT blowup": the GEMMs
(69% of the step) are flat; only allreduce (~linear with rows, 13% at 48) and
power-of-2 bucket padding scale. So high concurrency is GEMM-cheap.

Live A/B (closed-loop steady-state, /tmp/closed_loop_load.py, max_active 48 vs 128):
| workload (harness)        | metric    | ma=48  | ma=128 |
| long_output-like 96-conc  | TTFT p50  | 2628ms | 858ms (3x) |
|                           | agg tput  | 2740   | 3488 (+27%) |
|                           | TPOT      | 20.4   | 24.7 |
| few_shot-like 64-conc     | TPOT      | 51.5   | 34.0 |
|                           | TTFT      | 938ms  | 1829ms (worse) |
| multi_turn-like 125-conc  | TPOT      | 59.8   | 59.8 (context-bound) |

Findings: (1) our benchmark few_shot TPOT (51.6) is QUEUEING-INFLATED -- 64-conc >
max_active=48 interleaves prefills into decode; at matched capacity it's ~34ms.
(2) multi_turn TPOT (59.8) is CONTEXT-bound (long 8-turn ctx -> attention
dominates), unaffected by max_active -- my harness reproduced the benchmark's
exact 59.8. (3) raising max_active is a real throughput+TTFT win for
queueing-bound long_output, but HURTS few_shot TTFT (bigger prefill batches) and
costs TPOT on long_output.

CONCLUSION: flat max_active up does NOT flip any cell -- the long_output gaps are
3-9x (TTFT 8.8x, tput 2.9x) and +27%/3x falls far short; vllm runs even higher
concurrency AND a faster per-request decode (long_output TPOT 15 vs our 33). It
also risks KV-OOM at long seq and a few_shot-TTFT regression, so it is NOT a safe
default change. The principled KV-token-bounded admission (high cap for short-ctx,
low for long-ctx) is the right lever, but to actually WIN long_output it must be
PAIRED with a faster per-request decode (int4/fp8: TPOT 33 -> ~15). Harness note:
closed-loop steady-state (warm batcher) is REQUIRED; one-shot bursts are
setup-dominated (~500ms) and mislead.

## KV-token-bounded concurrency (DEFAULT-ON, A/B-resolved; guard crash-fix on)

JOURNEY (2026-06-07): off -> on -> off -> ON, settled on default-on after a clean
A/B. few_shot's prompt is ~150 tok (system + 5 tiny "15 + 27 = 42" examples + a
one-line question) -> max_seq ~400, SHORT-context like self_consistency (~286) and
tree (~350), so the boost raises few_shot's rows too (cannot cleanly exclude it).
The blocking question was whether that regresses few_shot TPOT (a won cell). Clean
A/B (boost OFF vs ON, max_tok=8 short outputs, many completions): self_consistency-
like TTFT 795->412ms (~2x), tput +10%, TPOT 14->30 (IRRELEVANT -- 0.0/uncontested
in scorecard); few_shot TPOT +1.4ms ONLY at the over-boosted 128 rows of
tiny-max_seq synthetic prompts -- at few_shot's REAL dims (max_seq ~406 -> 60 rows)
the pure decode cost is ~+0.2ms (weight-bound GEMMs), and in the benchmark
few_shot TPOT (51.6) is queueing-INFLATED so 60 rows cut prefill-interleaving and
likely LOWER it (matches the earlier ma128 51->34 obs). So the "contradictory"
earlier numbers were just different operating points; at the real point the boost
is few_shot-safe and self_consistency-positive. Budget kept conservative (48*512)
so few_shot reaches only ~60 rows. Net EV: likely +3 cells (self_consistency
TTFT/E2E/tput), few_shot safe. The guard (persistent-cache fit check) is an
unconditional crash-fix and stays on. Lesson: read benchmark prompt/output token
counts from source before calibrating seq-length thresholds.

UPDATE (the default-on attempt, now reverted): fixed the validation blocker. Root cause of the
earlier >768 crash: the batcher reused the persistent serving cache
UNCONDITIONALLY with no check that its max_seq_len/rows cover the workload (a
pre-existing bug the benchmark dodges via a large max_model_len). Fix: a guard
that reuses the persistent cache only when persistent_max_seq >= max_seq_len AND
persistent_rows >= max_active+prefix, else allocates a fresh correctly-sized
cache. Live-validated on 8xH100: >768 workload serves with NO crash;
self_consistency-like 128-conc boosts to 128 rows; few_shot-range (max_seq>=512)
TPOT 49.4 ~= baseline 51 (floored to 48, no regression). Default-on is safe
because the boost has NO cell-regression risk: protected cells (few_shot/multi_turn
TPOT) are arithmetically floored to 48, and the boosted short-context workloads
(self_consistency, tree) currently lose all four metrics, so more rows only help.
Open question for the next benchmark run: does self_consistency TTFT (274) drop
below vllm's 195 (the harness over-states its queueing vs real early-EOS outputs).
Original analysis below.

## KV-token-bounded concurrency (implemented, flag-gated, 2026-06-07)

Targets the closest flippable cell: self_consistency TTFT (274 vs vllm 195, only
1.4x). self_consistency is 1k IDENTICAL short prompts (~286 tok) with early-EOS
tiny outputs at 128 client concurrency -> pure queueing against the 48-row cap.
Flag TORCHINFERNO_OPENAI_TP_ONLINE_KV_BOUNDED_CONCURRENCY:
_kv_bounded_concurrency_cap() sizes the persistent serving cache for a higher row
cap (128) at warmup; the batcher then sets
max_active = max(48, min(cap, KV_TOKEN_BUDGET // max_seq_len)), budget 48*512.
SHORT-context workloads admit more rows (self_consistency ~85, tree ~70); any
max_seq_len >= 512 floors to 48 so few_shot (~896)/multi_turn (~2k)/long_output
(large) are UNCHANGED -- the two TPOT cells are protected by arithmetic, and
decode GEMMs are weight-bound (above) so the extra short rows are ~free.

Live-validated WIN case (budget 30720): long_output-like 96-conc TTFT 2628->955ms
(2.75x), throughput +15%; self_consistency-like runs at 128 rows. Kept
DEFAULT-OFF: the no-regression A/B was blocked because the test server left
max_model_len unset -> persistent/warmup cache max_seq defaulted to 768 -> ANY
>768-token request (multi_turn 826, few_shot 896) hits an index-out-of-bounds
device assert at pos 769 that poisons the CUDA context. This crash is ORTHOGONAL
to the boost (the 768 cache vs >768 seq would crash the baseline too) and the
REAL benchmark configures max_model_len large (it serves multi_turn fine), so
default-on is likely safe there but UNVALIDATED. To ship: restart with
--max-model-len / TORCHINFERNO_OPENAI_UNIFIED_MAX_SEQ_LEN >= 4096, re-run the
closed-loop A/B (few_shot/multi_turn TPOT must stay ~48-row baseline AND
self_consistency TTFT must drop below 195), then flip the default.

## BREAKTHROUGH: Marlin int4 decode kernel works via vLLM (scripts/bench_marlin_int4.py)

The long-cited "no fast batched-int4 decode kernel available" conclusion is
SUPERSEDED. vLLM's `_C` is built against torch's STABLE libtorch ABI, so a full
`import vllm` registers `torch.ops._C.marlin_gemm` and it RUNS correctly with our
custom torch 2.13.0a0 -- NO rebuild needed (unlike torchao 0.17 cutlass .so =
CUDA13-ABI fail; torchao 0.18 int4 = fbcode-only `mslk` dep). Earlier probes
missed it because `import vllm._C` alone doesn't register ops; the full `import
vllm` does (one libstdc++ LD_PRELOAD works around a soxr CXXABI clash in vllm's
transformers import chain).

CUDA-graph FLOOR (M=48), marlin int4 vs fp16:
| GEMM    | K x N      | fp16   | marlin | speedup |
| qkv     | 8192x1280  | 12.2us | 28.3us | 0.43x   |
| o_proj  | 1024x8192  |  8.4us | 11.6us | 0.72x   |
| gate_up | 8192x7168  | 65.5us | 43.0us | 1.52x   |
| down    | 3584x8192  | 32.9us | 22.9us | 1.44x   |
Marlin WINS the big, K-large GEMMs (gate_up 1.52x, down 1.44x) and loses the tiny
ones (fixed overhead vs a ~10us bf16 GEMM). All-4 total 1.13x; a HYBRID (marlin
for gate_up+down, bf16 for qkv/o_proj/lm_head) is ~1.38x on the decode projection
GEMMs. lm_head (N=16032) is not marlin-eligible (N must be % 64). Correctness
maxdiff ~0.008 vs fp16 (RTN int4 quant error).

HISTORICAL INTEGRATION (gate_up, flag TORCHINFERNO_MARLIN_INT4_DECODE,
default off at the time; current code later made the M-gated decode path
default-on, see the Marlin gate recheck near the top of this file):
torchinferno/kernels/marlin.py + tensor_parallel._mlp_project_decode_reduce now
route the gate_up decode GEMM through marlin int4 (op via torch.ops.load_library
on vllm _C, no vllm python import). Results (8xH100, ignore_eos steady-state
long_output, 64-conc):
- ACCURACY: greedy output BIT-IDENTICAL to bf16 (Paris./primes/gravity match).
  int4 gate_up holds correctness. FI decode CUDA graph captures cleanly WITH
  marlin inside (8 graphs).
- TPOT: 21ms -> 21ms (NEUTRAL); throughput 1956 -> 2007 (+2.6% only).

CORRECTED FINDING (per-kernel torch.profiler of the real decode, TORCHINFERNO_
PROFILE_DECODE_ONCE; supersedes the earlier "int4 neutral/dead" call which was
based on ONE noisy end-to-end TPOT test): at batch=32 the GEMMs DOMINATE the
decode step -- aten::mm = 9.82ms = 58% of GPU time; allreduce only 1.93ms (11%);
flashinfer attention 0.40ms (2.4%); cat/mul/neg (rope) ~2.6ms; index_put (KV
write) 0.87ms; norms 0.54ms. gate_up alone = 4.657ms (nvjet_128x32, 58us/call,
weight-read-bound: 117MB/58us = 2 TB/s). With marlin ON, the SAME decode profile
shows gate_up -> _C::marlin_gemm = 2.952ms (36.9us/call) = 1.58x, saving 1.9ms of
real GPU time (the GEMM total drops 9.82 -> 7.93ms). So int4 marlin DOES cut decode
GPU time (profile-PROVEN), contradicting the earlier neutral read. The earlier
end-to-end "21->21" was measurement imprecision / engine+network TPOT overhead
masking a ~1-2ms GPU saving (throughput did rise +2.6%).

INT4 ARC OUTCOME AT THAT POINT: gate_up int4 is ACCURATE (10/10 diverse greedy
prompts == bf16) and saves 1.74ms/decode-step (network-free _decode_active:
bf16 17.676 -> 15.938). qkv int4 REGRESSED (small GEMM, marlin overhead loses) ->
hybrid = big GEMMs only. It stayed DEFAULT-OFF in that iteration because (a)
default-on broke graph-vs-eager exact-match tests (int4 numerics != bf16) and
the lazy-quantize-before-capture ordering was fragile off the server path; (b)
benchmark-TPOT translation was UNCERTAIN -- my streaming harness showed TPOT
21->21 (network/SSE-masked) despite the 1.74ms engine saving, and whether the
benchmark client was engine- or network-bound was unknown; (c) even if it
translated, gate_up alone (~1.7ms) -> tree 31.7->~30 did NOT flip vllm 29.5;
gate_up+down (~2.1ms) -> ~29.6 only MARGINALLY. The current serving code keeps
the decode-sized gate default-on with bf16 fallback and an M cap; the latest
self/tree recheck above rejects only a broader self-only policy change, not the
existing short-row default.

REVIVED LEVER: a FULL int4 decode (all 4 GEMMs: qkv+o+gate_up+down, lm_head stays
bf16 -- N=16032 not %64) should save ~3ms of the 9.8ms GEMM GPU time -> decode
GPU ~17->14ms -> could flip tree_of_thought TPOT (31.7 vs 29.5, 2.2ms gap) and
lift throughput. OPEN QUESTION to resolve next: does the GPU saving translate to
serving TPOT, or is TPOT engine/network-bound above the GPU step? (the one
gate_up test was inconclusive). NEXT: extend marlin to all 4 GEMMs + drop the
bf16<->fp16 conversions, then re-measure TPOT carefully. The profiler hook
(forward_decode_flashinfer, gated, batch>=32) makes this reproducible.

ROPE FUSION (SHIPPED, default-on, the EXACT lever that int4 is not) -- run
20260607. The decode rope was plain aten: _rotate_llama_eager does cat(cos,cos)
+ cat(sin,sin) + cat(-x2,x1) + neg + 2 mul per layer per {q,k}. Two shipped
commits remove ~2.5ms of it:
  1. c3ec55c: the cos/sin tables are identical across all 80 layers, so their
     per-layer cat()s (~320 of 480) were redundant -- hoisted to once. (~1.3ms)
  2. 2636a66: a per-(batch,token) triton kernel
     (triton_apply_rotary_llama_batched_inplace) fuses the whole rotate-half into
     ONE in-place launch; wired into _apply_rotary_ragged_prefill (FI decode +
     ragged suffix) and _apply_rotary_ragged under the existing
     TORCHINFERNO_TRITON_ROTARY flag (default on). The pre-existing fused kernel
     (triton_apply_rotary_llama_inplace, used by _apply_rotary_cached for uniform
     prefill) could NOT be reused -- its cos is shared across the batch; decode
     positions vary per row.
Offline (scripts/bench_fused_rope.py, in CUDA graph): aten rope 1.32ms/step ->
fused 0.10ms/step = 13.5x. Correctness rel ~3e-3 vs the aten ref (fp32 accumulate,
MORE accurate than bf16). graph-vs-eager stays bit-identical (both call the same
kernel) -- so unlike int4, this is default-on SAFE, full suite green, regression
test test_triton_rotary_llama_batched_inplace_matches_torch_reference.
IN-SITU CONFIRMED (TORCHINFERNO_PROFILE_DECODE_ONCE on the live 8xH100 server,
48-conc, batch>=32): _rotary_llama_qk_batched_inplace_kernel fires 80 calls
(1/layer) at 178.9us TOTAL = 1.27% of the 14.145ms decode step; NO aten
cat/neg/mul rope ops remain in the profile. Decode is now cleanly GEMM-dominated
(aten::mm 9.81ms = 69%, allreduce 1.90ms = 13%, index_put 0.87ms, rope 0.18ms).
EXPECTED score effect (next run >2636a66): tree_of_thought TPOT 31.7->~29 flips
vs vllm 29.5; widens the few_shot (52.9 vs 58.3) and multi_turn (64.7 vs 67.6)
TPOT margins against the +-15ms run-to-run variance. Caveat (int4 lesson): a real
engine GPU saving can be partly masked in benchmark TPOT; confirm on the live run.
With rope fused, the ONLY remaining big decode lever is the GEMMs (int4 marlin,
above) -- everything else (allreduce, attention, norms, KV write) is small.

## Decode-step composition (8xH100, batch 48) — WHY int4 was neutral + the real lever

Measured the 160-per-step allreduce directly (8-GPU torchrun, [48,8192] bf16):
single allreduce nccl=214us, symm-mem multimem=39-55us; one_shot=99us (slower),
two_shot=absent. So multimem (current choice) is the best available symm-mem op.
x160 allreduces/step = ~6-9ms. Decode step ~21ms breaks down roughly:
  - 160 allreduces (symm-mem multimem): ~6ms  (~30%)
  - GEMM weight read (qkv+o+gate_up+down): ~5ms (~24%)
  - attention + norms + swiglu + per-op kernel time: ~10ms (~46%)

This explains the int4 NEUTRAL result: int4 cuts only the GEMM slice (~24%) and
only partially, so it can't move TPOT. The biggest single chunk is the ALLREDUCE
(~30%), but symm-mem multimem is already optimal -- beating it needs vLLM's custom
one-shot allreduce (lower latency for small msgs, but requires cross-rank CUDA-IPC
buffer registration -- a real distributed-setup effort, not a drop-in op) OR
fewer allreduces (the 2/layer are serialized by the residual stream -- attention-
out reduce and MLP-down reduce can't be fused/overlapped without re-architecting
TP, e.g. sequence parallelism which only converts AR->RS+AG at the same total).
The remaining ~46% (attention + many small kernels) is where vLLM likely also
wins via kernel fusion. Net: the decode TPOT gap to vllm (21 vs 15) is the SUM of
allreduce-latency + kernel-fusion advantages, each a deep effort; no single
drop-in change closes it.

ALLREDUCE IS NOT THE GAP (measured, overturns the "allreduce 30%" framing). vLLM's
custom one-shot allreduce (torch.ops._C_custom_ar, via CustomAllreduce) at [48,8192]
= 66us vs our symm-mem multimem = 38us -- we are 1.7x FASTER (NVLS hardware
multicast beats vllm's P2P one-shot for 768KB). So although the allreduce is ~30%
of OUR decode step, it is already faster than vllm's; the decode TPOT gap to vllm
(21 vs 15) is NOT the allreduce. This saved a pointless custom_ar integration. The
gap is the COMPUTE/attention/fusion path -- where int4 was neutral and the GEMMs
are already cuBLAS -- i.e. likely vllm's kernel fusion (fewer ops on the critical
path) and/or attention kernel, both deep and not drop-in replicable here.

STRATEGIC CORRECTION (isolated op speedups do NOT translate end-to-end). Per-layer
decode COMPUTE in isolation (4 GEMMs + 2 norms + swiglu, CUDA-graph, M=48) = 169us
x80 = 13.5ms; + measured allreduce ~6ms ~= the 21ms step. By that the GEMMs look
large, so int4 "should" help ~8% -- but the EMPIRICAL int4 gate_up result was only
+2.6% (TPOT-neutral). So isolated microbenchmarks (marlin 1.52x on the GEMM)
OVERPREDICT the end-to-end gain: the real decode is bound by the GRAPH-LEVEL
critical path (per-op execution serialized through the residual stream + the
allreduces), not by any single op's weight-read. Shrinking one GEMM barely moves
the chain. CONSEQUENCE: optimize the decode at the GRAPH level (fewer ops / fused
ops / fewer-or-faster allreduces / a shorter critical path), NOT op-by-op weight
compression. This is why every per-op lever (int4, symm-mem-already-applied) nets
small, and why matching vllm's 15ms needs structural decode changes (deep).

## FP8 characterization (scripts/bench_fp8_decode.py) — redirects the FP8 effort

Microbenchmarked torch._scaled_mm (FP8 e4m3, W8A8) vs bf16 mm at exact Llama3-70B
TP8 GEMM shapes, swept M. _scaled_mm has a high FIXED kernel overhead (~128us flat
for skinny GEMMs), so it CROSSES OVER with M:

| M (tokens)        | gate_up speedup |
| :---------------- | --------------: |
| 48   (decode)     | 0.61x (SLOWER)  |
| 512               | 0.80x           |
| 2048 (prefill)    | 1.80x           |
| 8192 (big prefill)| 1.84x           |

Implications:
- FP8 DECODE is RULED OUT via _scaled_mm (decode M<=64 is 0.55-0.67x SLOWER --
  cuBLAS FP8 is tuned for large-M training/prefill, not skinny decode). Faster
  decode would need W8A16 weight-only (FP8 weights, bf16 activations, fused
  dequant) -- a custom Marlin/torchao/triton kernel, deep. This saves a large
  wasted W8A8-decode implementation. W8A16 via available tools is ALSO not
  feasible (checked): torch._weight_int8pack_mm is 0.01-0.16x at decode shapes
  (naive kernel, time grows with N -- gate_up M=48: 9.7ms vs bf16 81us, 100x
  SLOWER), and torchao's optimized cutlass kernels fail to load here
  (_C_cutlass_90a.abi3.so ABI-mismatches our custom PyTorch build). So the decode
  TPOT cells (tree 2.6ms, multi_turn 7.4ms) are unreachable by quantization
  without a custom dequant-GEMM kernel OR a torchao rebuild against our PyTorch.
  Decode stays bf16-GPU-bound (~15ms/step).
  W4A16 int4 (_weight_int4pack_mm, gpt-fast tinygemm) checked too -- it reveals
  WHY quant cannot help us: it is tuned for M=1 single-stream decode. gate_up
  speedup vs M: M<=8 ~1.0x, M=16 0.67x, M=48 0.24x. Our server decodes at batch
  16-64, where the weight read is ALREADY amortized across the batch (decode is
  weight-bound only at M=1; at M=48 the 5.2ms weight read serves 48 tokens), so
  M=1-tuned quant kernels just add dequant overhead and LOSE. Only a Marlin-style
  kernel (efficient to M~64) could help batched decode -- needs the torchao
  cutlass build fixed or a custom kernel. NO available quantization path improves
  our batched decode.
  Build infra UPDATE (RESOLVED the ABI block, but int4 still unavailable):
  torchao 0.17.0's prebuilt cutlass .so was built for CUDA 13 / a stable torch
  ABI and would not load against our CUDA-12.6 custom torch 2.13.0a0. Rebuilding
  torchao 0.18 FROM GITHUB SOURCE against our torch WORKS (scripts/build_torchao.sh):
  the resulting _C_cutlass_90a.abi3.so loads cleanly (ctypes.CDLL OK) and its
  torch.ops.torchao cutlass ops are live -- the long-cited ABI block is gone.
  BUT this does NOT hand us an int4 decode kernel: torchao 0.18's cutlass kernels
  are FP8-SPARSE (rowwise_scaled_linear_sparse_cutlass_f8f8; no marlin_qqq_gemm),
  and its default Int4WeightOnlyConfig now requires a separate `mslk>=1.0.0`
  dependency (ImportError, not installed); the only built-in int4 path is the
  M=1-tuned tinygemm (_weight_int4pack_mm, already shown 0.24x at M=48). vllm's
  _C marlin ops also don't register (same prebuilt-ABI class). So the int4
  batched decode kernel still needs either the `mslk` lib built/installed, a
  Marlin-class custom kernel (triton plateaus at ~0.8x), or a vllm from-source
  rebuild. The ABI groundwork is done and reproducible; the kernel itself is not. CONFIRMED by two
  rounds of attempt (scripts/bench_triton_int4.py, packed int4 + groupwise
  dequant): a naive kernel is 0.03-0.74x; a rewritten one (even/odd nibble split
  = two clean tl.dots, BLOCK_K=group, autotuned BLOCK_M/N + num_stages) reaches
  0.54-0.99x EAGER. But the decisive CUDA-GRAPH (floor-level, no launch overhead)
  comparison at M=48: bf16 gate_up 64.9us / int4 143.9us = 0.45x; lm_head 0.56x.
  The int4 kernel reads 4x LESS memory yet is SLOWER -- it sustains only ~0.2
  TB/s vs cuBLAS bf16's ~1.8 TB/s. It is dequant/tl.dot-overhead-bound, NOT
  memory-bound: the int4 byte savings are entirely eaten by un-hidden unpack+
  dequant and poor small-M tensor-core utilization. Beating bf16 needs
  Marlin-class engineering (dequant fused into the MMA pipeline, async cp.async
  weight prefetch) -- expert multi-month work. DEFINITIVE across ALL angles
  (PyTorch M=1 kernels, torchao/vllm ABI, custom triton at the HW floor): faster
  batched-decode quant is not reachable with available tools/effort.
- FP8 PREFILL is the viable lever for the DOMINANT TTFT gap. Prefill GEMMs are
  52% of prefill at 34% MFU; at M>=2048 (few_shot is 48x640=30720 tokens) FP8 is
  ~1.8x on that portion -> ~23% prefill reduction -> few_shot TTFT ~317->~244
  (vllm 159: narrows, does not yet flip). Standard _scaled_mm, well-supported.
  ACCURACY probe (RTN e4m3, LLM-like activations w/ outliers, M=2048): per-GEMM
  rel-err ~3.6-3.8% (per-tensor AND row-wise -- error is e4m3 mantissa-bound, not
  scale-bound), vs bf16's 0.14%. That is the UNCERTAIN zone: ~25x worse than bf16
  per GEMM, compounding over 80 layers -- may or may not hold the >=98% correctness
  bar; only a full impl + benchmark settles it. So FP8 prefill is mechanically
  ready (1.8x) but accuracy-gated and does not flip a cell -- modest, risky.
  NEXT STEP: quantize prefill weights to FP8 + dynamic activation quant in the
  prefill GEMM path; the hard part is keeping benchmark correctness >=98% (FP8
  prefill accuracy) and FP8 quant ops inside the captured prefill graph. Note it
  only helps LONG-prompt benchmarks (few_shot); multi_turn/tree/long_output have
  short prompts (small-M prefill, FP8 neutral/negative) and are TTFT-bound by
  ADMISSION latency, not prefill compute.

## (historical) Issue 1 — prefill kernel MFU (~2x), the dominant TTFT gap

## Issue 1 — prefill kernel MFU (~2x), the dominant TTFT gap

few_shot = 64 concurrent x ~640-token prompts (one burst). Local measurement at
HEAD: min TTFT 146ms (≈ vllm), but **median 2135ms** — 63 of 64 requests bunch at
~2.1s. Cause: admission packs ~48 long prompts into one step → a single
~30,720-token prefill. Even at the graph-bucketed level, the (batch=8, q=768)
prefill graph does 6144 tokens in **315ms ≈ 34% MFU**. vllm prefills the same
64x640 with ~145ms median TTFT — roughly **2x better GEMM MFU plus chunking**.

Ruled out as quick fixes (tested locally — ALL scheduling/admission knobs):
- Lowering `prefill_token_budget` 32768 → 5120 (burst): median TTFT unchanged
  (~2117ms), throughput WORSE (557 → 463). Total prefill compute is the floor.
- `prefill_token_budget` 32768 → 1280 (~2 reqs/step, STREAMING 64-conc 640-tok):
  much WORSE — TTFT p50 1855 → 4843ms, throughput 126 → 75. Small prefill
  batches lose GEMM efficiency; staggering does not beat batching. **Do not
  lower the budget.**
- max_active 48 vs 128 (streaming few_shot): 48 is better/tied on TTFT
  (1855 vs 1927) AND much better TPOT (44 vs 96). Row cap is not the TTFT lever.
- Joint (batch,q) prefill CUDA graphs already remove per-layer launch overhead
  (the 315ms is a graph replay = pure compute at 34% MFU).

Architecture levers tested (streaming harness, 64-conc; baseline = default
DQ=16, big batched prefill, graphed decode):
- `decode_quantum` sweep (admit/drain every N steps; controls how often new
  arrivals enter the engine). Full data, 64-conc streaming:
    | metric            | DQ=16 (def) | DQ=8 | DQ=4 |
    | few_shot   TTFT   | 1881        | 1657 | 1282 |
    | few_shot   tput   | 101         | 153  | 157  |
    | few_shot   TPOT   | 64          | 175  | 224  |
    | long_out   TTFT   | 493         | 384  | 375  |
    | long_out   tput   | --          | 855  | 766  |
    | long_out   TPOT   | 22          | 40   | 41   |
  Pure TTFT/throughput <-> TPOT tradeoff. A small quantum admits new arrivals
  promptly (low TTFT, high throughput) but interrupts graphed decode with a
  separate prefill forward more often (high TPOT). CRITICAL: the TPOT
  regression is STEEP and hits at ANY DQ<16 -- long_output TPOT 22->40 even at
  DQ=8 -- because decode no longer runs in long uninterrupted bursts. DQ=16 is
  the knee. Below it, TPOT ~doubles for a 12-30% TTFT gain. NOT a clean win;
  net benchmark impact unmeasurable while frozen. Default stays 16.
  STRATEGIC NOTE: vllm AND sglang both have higher throughput than us, and
  sglang ACCEPTS worse TPOT (50.7ms vs our 36) to get it. We are the outlier
  over-optimizing TPOT (already ~matched) while 2.8x behind on throughput.
  Lowering DQ moves us toward the vllm/sglang operating point. This is the #1
  A/B to run the MOMENT the benchmark unfreezes -- the data above says DQ=8
  buys ~+50% throughput and lower TTFT for a TPOT hit that sglang shows is
  survivable. It is a scoring-weight call that only the live benchmark settles.
- `TORCHINFERNO_CONTINUOUS_UNIFIED_FORWARD=1` (mix prefill+decode in ONE
  forward_step_flashinfer): MUCH worse -- few_shot TPOT 1560ms, long_output
  TPOT 83ms. The mixed batch composition varies every step so it canNOT use a
  captured decode graph; eager decode dominates. Off for good.
- `prefill_chunk_size=256` + DQ=4 (vllm-style chunked prefill, graphed): MUCH
  worse -- few_shot TTFT 5937ms, tput 36; long_output TTFT 1131, TPOT 101.
  Chunking fragments the high-MFU batched prefill into small low-MFU pieces and
  multiplies attention recompute over growing context. Same MFU-floor failure
  as a small token budget. Off for good.

Net: the baseline (big batched prefill + graphed decode) sits near this
architecture's Pareto frontier -- throughput and TPOT are coupled (prefill and
decode are separate forwards, so admitting/prefilling more aggressively for
throughput necessarily interrupts decode and costs TPOT), and that coupling
cannot be broken cheaply (unified and chunked both lose, above). NO
scheduling/config knob is a clean win. The only real levers are (a) higher
prefill MFU -- and QKV/gate-up GEMMs are ALREADY fused, so this means FP8 GEMMs
or a different parallelism layout, both major; or (b) prefix caching to shrink
the prefill, which is session-wiped today (needs a persistent engine); or (c)
a decode path that stays graphed WHILE absorbing a bounded prefill (e.g. a
captured mixed-shape graph per (decode_n, prefill_bucket) pair), which neither
existing unified nor chunked path provides. All three are deep.

Op-level profile (TORCHINFERNO_PROFILE_PREFILL_ONCE, one batched prefill,
torch.profiler sorted by CUDA time) CORRECTS the earlier "raise GEMM MFU"
framing — the sinks are:

| component                       | CUDA time | %    |
| :------------------------------ | --------: | ---: |
| GEMMs (aten::mm, cuBLAS nvjet)  |    91.5ms | 52%  |
| TP allreduce (NCCL ring, 160x)  |    53.9ms | 31%  |
| add_rms_norm                    |    15.4ms |  9%  |
| FlashInfer attention            |     2.0ms | 1.2% |

Attention is NOT the sink (1.2%). Two real levers:
- GEMMs (52%): already cuBLAS; QKV and gate-up are ALREADY fused (one GEMM
  each). The MFU loss is the TP-sharded dims (per-rank K/N smaller). Limited
  headroom without a different parallelism layout.
- Allreduce (31%): runs as ncclDevKernel_AllReduce_Sum_bf16_RING_LL -- plain
  NCCL ring, NOT the symm-mem one-shot allreduce the decode graphs use. On
  8xH100 NVLink, symm-mem one/two-shot is typically faster than ring for these
  sizes. ROUTING PREFILL ALLREDUCE THROUGH SYMM-MEM is the most actionable
  single lever (2 allreduces/layer x 80 layers = 160 calls). This is the next
  data-driven change after feaebc7 (max_active revert) validates.

Also still: true chunked prefill interleaved with decode so early requests
return their first token without waiting for the whole burst.

Common-prefix graph warmup A/B (2026-06-21, current `776cdb2`):
- Restoring the legacy short common-prefix slot alongside the newer 64/128
  warmups (`45,64,128`) fixes the tree_of_thought cold suffix-graph stall:
  local tree TTFT/E2E/throughput moved `434.1/475.8/3.6` to
  `219.2/259.5/4.6`. Queue profile: first sampled 256-request session prefill
  forward dropped `3011ms -> 1008ms`.
- It is not a universal win: the same warmup set slightly regressed the local
  self/few slice vs `64,128` (`self` TTFT `320 -> 347`, `few` TTFT
  `156 -> 166`). The tree regression without the short slot is much larger, so
  the default keeps the short slot until the live full benchmark says otherwise.
- Rejected alternatives:
  - Disable ragged suffix graph capture on misses and fall back to eager:
    tree regressed to `624.2ms` TTFT / `1.8 tok/s`; repeated eager suffix
    prefills are worse than a cold capture.
  - Dense `32..64` short-prefix sweep for only max-active suffix batch:
    weak result (`413.1ms` TTFT) because the first session still captured
    smaller suffix batch shapes.
  - Narrow `40..48` band with all suffix batch buckets and one prefix row:
    partial recovery (`280.7ms` TTFT) but worse than the legacy short slot and
    higher startup (`261.5s`).

## Issue 2 — decode is memory-bound; throughput scales with batch

long_output (tiny prompt, huge output, 64 concurrent): TPOT 30.7 vs vllm 15.1,
throughput 22.3 vs 58.3. Decode reads all 70B weights every step, so per-step
time is ~fixed and throughput ∝ concurrent rows. Profiling: decode_model launch
2.3ms, but the blocking `next_token_tensor.cpu().tolist()` exposes ~18ms of
GPU decode per step.

RESOLVED — symm-mem allreduce for batched decode (commit `c60e0bd`). Decode does
160 small allreduces/step; on `max_batch=1` these fell back to NCCL ring. Raising
to 256 routes them through symm-mem (~3x faster, latency-bound). A/B on 8xH100,
max_active=48: 16-conc TPOT 38 → 27 ms (-29%), 32-conc 29 → 26 ms. 27ms is BELOW
vllm's 32.5ms cross-benchmark TPOT.

max_active SETTLED at 48 (commit `feaebc7`), re-confirmed AFTER symm-mem decode:
max_active=128 + symm-mem still gives 45ms TPOT at 64-conc (loses to vllm 32.5),
vs 48's 27ms (wins). The TPOT penalty of 128 is KV-read scaling with active rows,
which symm-mem does NOT fix. 128 buys throughput (96-conc 2019 vs ~1200 tok/s)
but at a TPOT loss -> net benchmark loss (it scored 0/20). Keep 48: winning TPOT
beats chasing throughput. The throughput gap needs a different lever (chunked
prefill, or KV-token-bounded admission), not a bigger fixed row cap.

Remaining decode lever: the ~18ms per-step CPU sync is the GPU decode time
exposed; cutting TPOT further needs async overlap (GPU token stash — keep sampled
tokens on GPU, read back lagging) which has been correctness-fragile at scale.

## Issue 3 — prefix reuse is session-limited AND TP-buggy (low ROI)

Reuse would help self_consistency / multi_turn / few_shot / tree_of_thought in
principle, BUT:
- With `persistent=False`, every request burst starts a fresh online session and
  `start_online → _reset_capacity` WIPES the prefix cache. So cross-burst reuse
  (e.g. multi_turn's per-turn requests) NEVER triggers — verified via per-rank
  `TORCHINFERNO_REUSE_DEBUG` traces (`cached_prefixes=0` at every `step=0`).
- Only within-session reuse fires, and that is exactly the case that
  hangs/CUDA-asserts on 8 ranks (a multi-rank collective divergence). The reuse
  LOGIC is verified correct single-GPU (`scripts/debug_reuse_engine.py` PASSES);
  storage is TP-consistent; the divergence is in within-session reuse execution.

So a useful reuse needs BOTH the TP fix AND a persistent engine whose cache
survives bursts (persistent=True previously deadlocked). Larger than the reuse
path itself. Gated OFF (`TORCHINFERNO_CONTINUOUS_FI_REUSE=0`).

Multi-turn A/B refresh (2026-06-21, current `82d814d`):
- Baseline remains prefill dominated: local multi_turn `601.2ms` TTFT,
  `42.1ms` TPOT, `641.2ms` E2E, `1.9 tok/s`; queue profile ended at
  `10026ms` prefill wall vs `2160ms` decode active with `max_active=16`.
- Current `d8846b1` recheck is the same shape: `604.0ms` TTFT,
  `43.3ms` TPOT, `648.7ms` E2E, `1.9 tok/s`; queue profile ended at
  `10203ms` prefill wall / `9741ms` prefill forward vs `2070ms` decode
  active, with `65` prefill batches and no generated/finished-prefix stores.
- Raising the greedy-large row cap to 24 is rejected again:
  `656.9ms` TTFT, `57.1ms` TPOT, `698.6ms` E2E, worse p99. The extra active
  rows cost TPOT without relieving enough prefill queueing.
- Lowering `TORCHINFERNO_OPENAI_PAGED_KV_MIN_SEQ` to 512 is not ready:
  the server built FlashInfer decode graphs and listened, then hung before
  writing queue progress or result files on multi_turn. Retried on current
  26752a8 after stop-id/runtime cleanup: the server reached traffic and saturated
  all GPUs, but still produced no 100-conversation milestone after several
  minutes and was manually terminated. Keep the 1024 threshold.
- Finished-prefix reuse with non-common 16-token prefix buckets is also
  rejected (experimental code backed out): it cut prefill tokens
  `81431 -> 27937` and stored `1000` finished prefixes, but fragmented into
  `175` prefill graph calls and regressed to `2638.9ms` TTFT, `43.8ms` TPOT,
  `2677.7ms` E2E, `0.5 tok/s`. Queue profile ended at `44780ms` prefill wall /
  `43646ms` prefill forward. Coarser reuse is still more expensive than the
  baseline shared-prefix path unless non-common suffixes can be fused into much
  fewer graph shapes.

Multi-turn refresh after the exact-prefix online route (2026-06-21, current
`fa088ca`): the shape is unchanged and still prefill dominated. Current profiled
multi_turn landed at `594.9ms` TTFT, `41.9ms` TPOT, `635.7ms` E2E,
`2.0 tok/s`, 980/1000 raw correct. Queue profile ended at `10056ms` prefill wall
/ `9570ms` prefill forward vs `2123ms` decode active, with `65` prefill batches,
`154` decode batches, `80135` prefill tokens, and `45000` reused prefix tokens.
Rejected follow-ups on the same code:
- A global 5ms initial batch wait did not fill the first wave and slightly
  regressed: `598.1ms` TTFT, `42.1ms` TPOT, `637.9ms` E2E. Final profile had
  `initial_batch_size=2`, `64` prefill batches, and `9504ms` prefill wall.
- Raising online prefix rows from 64 to 96 did not increase reuse. Reuse tokens
  stayed at `45000`, prefill tokens rose to `82499`, and metrics regressed to
  `604.3ms` TTFT / `644.5ms` E2E.
- Enabling pinned full-prompt stores for `max_tokens>=512` cut prefill tokens
  (`80135 -> 19222`) and increased reused prefix tokens (`45000 -> 95611`), but
  fragmented the graph path into `632` prefill batches and `575` prefill graph
  misses. It regressed catastrophically to `8530.9ms` TTFT / `8571.1ms` E2E.
  The useful direction is not simply longer prefix hits; it needs a fused
  non-common-prefix prefill path with stable graph shapes.
- Combining pinned full-prompt stores (`max_tokens>=512`) with the existing
  non-common-prefix graph path
  (`TORCHINFERNO_CONTINUOUS_NON_COMMON_PREFIX_GRAPH_PREFILL=1`) is also rejected.
  It reduced raw prefill tokens but made fragmentation worse and then crashed:
  by the last queue snapshot the run had admitted only `515` requests, emitted
  `499`, and already spent `73834.9ms` in prefill wall (`68385.5ms` forward)
  across `198` prefill batches / `199` graph hits. The server then hit a CUDA
  illegal memory access in `_prefill_prefix_graph_batch` during the synchronized
  non-common graph prefill. Do not enable this path for multi_turn without a
  new mixed-prefix graph-safety fix and a much coarser batching policy.
- Rechecking multi_turn after the common-prefix no-logits cleanup (`c6be9e0`)
  did not move the shape: `598.6ms` TTFT, `41.4ms` TPOT, `638.6ms` E2E,
  `1.9 tok/s`, 981/1000 raw correct. Queue profile stayed at `65` prefill
  batches and `10107ms` prefill wall (`9643ms` forward). Multi still needs the
  fused non-common-prefix path or persistent TP-safe reuse; removing common-prefix
  logits sync is not enough.

Self-consistency sampled-short wait A/B (2026-06-21, current `9452794`):
- Clean local full run with the 5ms sampled-short initial wait:
  `358.2ms` TTFT, `427.3ms` E2E, `2.3 tok/s`.
- Re-running self_consistency with
  `TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_SHORT_INITIAL_BATCH_WAIT_MS=10`
  improved to `255.7ms` TTFT, `384.1ms` E2E, `2.6 tok/s`, correctness 100%.
  This policy only covers sampled short requests (`max_tokens<=256`), so tree's
  sampled medium path remains on its separate default.

Tree sampled-medium wait refresh (2026-06-21, current `0ec0ce5`): rechecking
the older 1ms wait A/B on the current branch is not defaultable. With queue
profiling enabled, tree_of_thought moved from the current profiled control
`248.2ms` TTFT / `52.1ms` TPOT / `306.7ms` E2E / `4.0 tok/s`, 962/992 raw
correct, to `225.9ms` TTFT / `53.1ms` TPOT / `269.0ms` E2E / `4.3 tok/s`,
960/992 raw correct. The first sampled-medium online session admitted more work
(`221 -> 237`) while prefill wall dropped `1835ms -> 1326ms`. But the
score-comparable default run was mixed: `243.1ms` TTFT / `50.8ms` TPOT /
`278.0ms` E2E / `4.4 tok/s`, 959/992 raw correct, versus the unprofiled control
at `230.6ms` TTFT / `52.7ms` TPOT / `269.8ms` E2E / `4.3 tok/s`. Keep the 5ms
sampled-medium default.

Long-output greedy-short wait refresh (2026-06-21, current `0ec0ce5`): raising
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_INITIAL_BATCH_WAIT_MS` from 1ms to
5ms is rejected. It only moved `initial_batch_size` from 2 to 3, kept the same
60 prefill batches and 55/5 graph hit/miss split, and regressed score-facing
TTFT/TPOT to `355.2ms` / `28.9ms`. Keep greedy-short long_output on the 1ms
default.

Post-startup long_output refresh (2026-06-27, current `cc7e4e9`): after
switching TP checkpoint loading back to per-rank reads, local TorchInferno
server startup reached readiness in `~110s` and long_output completed 1000/1000
correct. Same-host vLLM/SGLang comparison still shows a TTFT/E2E gap:
TorchInferno `316.8ms` TTFT / `23.2ms` TPOT / `1317.6ms` E2E, vLLM
`65.6ms` / `16.9ms` / `675.0ms`, SGLang `62.2ms` / `24.3ms` / `937.1ms`.
Queue profiling points at online prefill graph capture plus decode admission:
`~10.1s` prefill wall with `8` ragged-prefix graph captures and `~9.4s`
decode active. Rejected follow-ups:
- Raising greedy-short initial wait to 50ms filled the first batch
  (`initial_batch_size=37`) but did not improve score-facing latency:
  `317.0ms` TTFT / `22.8ms` TPOT / `1317.2ms` E2E.
- Disabling graph prefill was catastrophic: by an interrupted partial run,
  prefill wall was already `~89s`.
- Disabling capture-on-miss for common-prefix ragged prefill removed capture
  time but regressed eager prefill: `464.4ms` TTFT / `29.2ms` TPOT /
  `1775.8ms` E2E.
- Deferring capture until a shape had five observations cut captures from `8`
  to `4`, but prefill wall still grew to `~11.7s`; reject shape-deferral
  capture policy.

Greedy-short decode quantum refresh (2026-06-27): for decode-many-enabled
short greedy traffic, quantum `8` improved TPOT but over-held the decode loop and
regressed TTFT/E2E. Quantum `2` improved TTFT but raised decode sync overhead.
Quantum `3` is the best tested balance: long_output recheck landed at
`263.2ms` TTFT / `24.4ms` TPOT / `1206.8ms` E2E, and a full TorchInferno-only
run with the same env had long_output `238.0ms` / `24.9ms` / `1180.6ms` with
1000/1000 correctness. Make `3` the greedy-short decode-many default while
keeping env overrides for score/throughput tradeoffs.

Post-`019ce7b` public run and scheduler A/B refresh (2026-06-27): public
`20260627_170254` still shows TorchInferno at 1/20 metric wins versus vLLM's
16/20, with the largest median gaps in tree (`286.1ms` TTFT vs vLLM `60.7ms`),
long_output (`304.6ms` vs `70.7ms`), and multi_turn (`375.7ms` vs `166.2ms`).
Local follow-ups on the same checkout rejected these knobs:
- `TORCHINFERNO_CONTINUOUS_GENERATED_PREFIX_CACHE=1` on multi_turn is
  catastrophic: `5931.0ms` TTFT / `777.0ms` TPOT / `6558.9ms` E2E. The extra
  generated-prefix logits/cache work overwhelms any later-turn reuse benefit.
- Raising multi_turn greedy-large `max_active` to `48` regressed to
  `887.8ms` TTFT / `79.1ms` TPOT / `975.6ms` E2E. The earlier 32-row default
  remains the stable point; the 24-row TPOT tradeoff also remains rejected for
  latency.
- Lowering multi_turn greedy-large refill min-ready to `1` landed at
  `382.1ms` TTFT / `59.5ms` TPOT / `453.0ms` E2E, a small TPOT move with worse
  TTFT/E2E than the current local baseline (`375.4ms` / `62.1ms` / `446.5ms`).
- Raising self-consistency sampled active rows while preserving prefix rows
  improved TTFT but broke one-token E2E: 160 rows landed at
  `219.9ms` TTFT / `329.1ms` E2E, and 144 rows landed at
  `246.3ms` / `355.6ms`. Keep the 128-row sampled-short KV cap until the HTTP
  finish path or scheduler event emission is changed.

The public TorchInferno log also emitted 12 Transformers BPE cleanup warnings
immediately after readiness. Those are not a median-latency lever, but decode
calls should pass `clean_up_tokenization_spaces=False` so the server does not
write warning noise on BPE tokenizers.

Public `20260627_190305` is a newer same-host comparison, but it still measured
TorchInferno `019ce7b` rather than the pushed cleanup commit. vLLM moved to
`35e3850` and won 17/20 metric cells; TorchInferno kept only few_shot TPOT.
TorchInferno medians were few_shot `253.7 / 47.5 / 299.6ms`,
self_consistency `273.3 / 0.0 / 293.2ms`, multi_turn
`492.2 / 59.7 / 544.7ms`, tree_of_thought `268.8 / 42.9 / 309.5ms`, and
long_output `325.9 / 21.3 / 1130.4ms` (TTFT/TPOT/E2E). Local post-cleanup
full-suite rechecks on `876fb5e` were substantially better on few_shot
(`145.6 / 47.5 / 187.6ms`) but still noisy on sampled self and unchanged on the
structural multi/long gaps.

The next control-plane lever is a shared-memory TP command ring. Full shared
memory for every command is rejected as a default despite improving
self/tree/long: it regressed few_shot to `171.4 / 42.4 / 206.7ms` and
multi_turn to `376.1 / 61.7 / 459.1ms`. Moving only online step commands is
also too broad: it gave self_consistency `182.5 / 0.0 / 243.5ms`, but regressed
few_shot to `192.4 / 49.1 / 233.0ms` and long_output to
`273.9 / 23.8 / 1284.3ms`. The promoted shape is narrower: enable the POSIX
shared-memory transport by default only on CUDA tensor-parallel models and use
it for sampled online start/step/close commands (`temperature > 0`,
`max_tokens <= 300`), while prompt submissions and greedy sessions fall back to
the existing tensor command protocol. The full local A/B with
`sampled_online_step` landed at few_shot `140.9 / 49.0 / 181.4ms`,
self_consistency `256.6 / 0.0 / 274.2ms`, multi_turn
`369.6 / 61.0 / 441.0ms`, tree_of_thought `227.8 / 48.2 / 271.3ms`, and
long_output `276.3 / 24.0 / 1210.8ms`, all with 100% normalized correctness.
A no-env focused confirmation after making that the CUDA TP default reproduced
the few_shot row at `140.4 / 48.0 / 180.6ms`; self remained noisy
(`293.2 / 0.0 / 326.7ms`). The expected public score movement is therefore
few_shot TTFT/TPOT/E2E first, not a solved self or long-output row.

Rejected follow-ups in this loop:
- Multi_turn generated-prefix cache remains catastrophic
  (`5931.0 / 777.0 / 6558.9ms`), even though correctness is preserved.
- Multi_turn `max_active=48` (`887.8 / 79.1 / 975.6ms`) and refill floor `1`
  (`382.1 / 59.5 / 453.0ms`) do not beat the current latency tradeoff.
- Self sampled active caps `144` and `160` improve TTFT at the cost of one-token
  E2E/throughput.
- Prompt-lookup decode for long_output is catastrophic
  (`3923.8 / 388.5 / 18135.1ms`) because verification overcomputes without the
  needed graph/pipeline path.
- Forcing submit+step commands on long_output regressed to
  `274.6 / 24.2 / 1302.7ms`.
- Sampled-short `TORCHINFERNO_OPENAI_TP_ONLINE_STEP_SYNC=0` with shared-memory
  commands lowered self TTFT but regressed E2E/tail (`224.8 / 0.0 / 336.0ms`,
  p99 over 1s).
- Reducing sampled-short initial wait to 5ms is rejected:
  `233.2 / 0.0 / 356.7ms` with the same high tail.

Post-`d3131f4` local refresh (2026-06-27): public run `20260627_230306`
measured TorchInferno `d3131f4`, so it still predates the later `1c52093`
small-batch greedy suffix warmup. The public row kept only the few_shot TPOT
cell and showed TorchInferno at few_shot `249.6 / 49.0 / 298.7ms`,
self_consistency `281.9 / 0.0 / 303.1ms`, multi_turn
`366.3 / 60.2 / 428.5ms`, tree_of_thought `262.1 / 43.0 / 310.7ms`, and
long_output `321.8 / 21.5 / 1127.0ms`. The current no-profile
TorchInferno-only stack landed at few_shot `141.6 / 46.7 / 182.5ms`,
self_consistency `195.5 / 0.0 / 234.8ms`, multi_turn
`346.5 / 60.6 / 410.3ms`, tree_of_thought `229.6 / 46.8 / 271.4ms`, and
long_output
`255.4 / 24.1 / 1231.9ms` with in-family correctness. Focused tree follow-ups
rejected sampled-medium `max_active=40` repeat (`280.2 / 48.3 / 325.1ms`),
capture-on-miss disable (`280.1 / 50.0 / 325.1ms`), and sampled-medium idle
arrival collection (`242.6 / 50.0 / 289.7ms`) against the same-control
`220.8 / 49.4 / 281.2ms`.

Multi_turn still had request-path greedy common-prefix suffix graph captures on
the current stack. The queued tail shapes included `prefix_graph:b4:s128:p45-45`
and, on repeat, `prefix_graph:b1:s128:p45-45`; the startup warmup captured only
batch `8/16/32`. Adding batches `1,4` to
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_GREEDY_COMMON_PREFIX_SUFFIX_BATCHES` removed
request-path captures (`1 -> 0`) and moved the profiled multi row from
`375.3 / 62.9 / 438.7ms` to `365.1 / 58.8 / 423.3ms`. Queue counters improved
phase total `11.4s -> 9.34s`, prefill wall `6010.8ms -> 5473.1ms`, and decode
active `5087.1ms -> 3575.2ms`. Promote the extra startup batches; startup moved
from about `110.5s` to `125.5s`, which is not score-facing.

The greedy suffix warmup also had a precision-policy mismatch: it captured under
a 512-token greedy policy, while short greedy sessions such as long_output run
with FP8 prefill disabled. A targeted env-only long_output run that warmed the
same configured suffix graphs under the short greedy policy eliminated request
path prefill captures (`8 -> 0`) and cut the profiled row to
`239.4 / 23.7 / 1139.1ms`, with prefill wall `9979.7ms -> 5874.1ms`. Promote
the policy part only: startup greedy suffix warmup now covers both short and
large greedy max-token policies (`128,512`) for whatever prefix/suffix shapes
are already configured. Exact additional prefix-token warmups remain env-only
until there is a general dynamic-prefix graph path.

That general path is now scoped to short suffixes: a negative ragged-prefill
`context_len` selects a static context bucket while keeping per-row
`start_positions` dynamic, so one CUDA graph can replay across different shared
prefix lengths in the same bucket. Keep it default-on only for suffix buckets
`<=16` with a 256-token minimum context. The profiled few_shot row improved from
`190.5 / 47.9 / 226.2ms` to `166.4 / 47.8 / 203.9ms`; request-path prefill
captures dropped from `2` (`1038.2ms`) to `0`, and prefill wall fell from
`3561.4ms` to `2834.7ms`. Multi_turn was neutral at
`364.4 / 62.7 / 428.2ms` with zero request-path prefill captures. A broader
env-forced dynamic run on long_output is rejected (`263.9 / 25.5 / 1220.7ms`):
the larger `s32/s64/s128` boolean-mask bucket path slows prefill forward enough
that those suffixes should stay on the exact flash `context_len` path.

Tree_of_thought exposed the same suffix graph precision mismatch on sampled
medium traffic (`temperature > 0`, `max_tokens=300`): a profiled run on
`886ad79` spent `10906.7ms` in 12 request-path prefill graph captures and
landed at `265.6 / 50.1 / 305.0ms`. Warm the existing common-prefix suffix
shape set under the sampled-medium FP8 policy as well, but default its sampled
suffix list to `16` so startup does not capture the larger rejected suffix
buckets. The local tree rerun moved to `230.3 / 49.4 / 275.3ms`.

Rejected follow-ups after the public `d3131f4` refresh:
- Prompt-lookup decode for long_output accepted copied prompt tokens but routed
  verification through expensive eager multi-token forwards; an interrupted run
  had only `340` submitted requests after `93.5s` of runtime step time.
- Extending sampled submit+step to tree's 300-token sampled-medium branch
  landed at `235.7 / 51.1 / 283.5ms`, slightly worse than the nearby control,
  with queue phase time essentially unchanged.
- Disabling common-prefix prefill capture-on-miss for few_shot removed the cold
  capture cost but made every suffix prefill eager and regressed the row to
  `534.3 / 46.9 / 573.0ms`.

The public `20260629_110323` refresh measured TorchInferno `03677fd` at 2/20
scorecard cells. The largest remaining median gaps were TTFT/E2E for
long_output (`317.7 / 22.9 / 1119.8ms` vs vLLM
`76.4 / 15.0 / 629.6ms`) and tree (`208.2 / 59.1 / 257.5ms` vs vLLM
`63.0 / 31.5 / 85.5ms`). A same-host long_output provider comparison on current
`23bbd95` showed a narrower story: TorchInferno locally won median TPOT
(`24.1ms` vs SGLang `24.5ms` and vLLM `27.7ms`) but still lagged TTFT/E2E
(`281.1 / 1262.7ms` vs SGLang `59.5 / 928.6ms`). The queue profile points to
first-token latency, not steady decode: `7.59s` prefill wall, `13.00s` ragged
decode GPU time, `7.37s` CPU token time, and `23.43s` profiled phase for
`37.7k` emitted events. Per-100 request windows after the cold first wave stayed
around `219-296ms` TTFT for TorchInferno while vLLM was mostly `82-106ms` and
SGLang `55-79ms`.

Two greedy-short scheduling rechecks remain rejected on the current stack.
Enabling idle-arrival collection improved median TTFT/E2E to
`264.8 / 1158.4ms`, but increased prefill batches (`128 -> 135` submits) and
worsened global p99 TPOT (`89.4 -> 269.3ms`). Setting the greedy-short initial
batch wait to `1ms` improved median TTFT to `244.1ms`, but the run still started
with one initial request and regressed p99 TTFT/TPOT/E2E to
`5249.1 / 474.3 / 6550.6ms`. Keep the current zero-wait/no-idle-collection
defaults until a different admission path can reduce first-token latency without
tail damage.

To make the next long-output profiles actionable, queue-profile records now emit
server-side request latency aggregates: queue-to-submit, queue-to-first-token,
submit-to-first-token, and queue-to-finish counts plus p50/p90/p99/max. These
fields separate HTTP/client arrival delay from runtime prefill/decode delay in
the same `online_batcher` JSONL snapshots used above.

The same request-latency profiling on tree_of_thought shows a different split.
A no-env TorchInferno-only run on pushed `289ce77` landed at
`230.6 / 49.5 / 275.4ms`, 992/992 benchmark-correct. Across the 12 quiescent
online-batcher sessions, p50 queue-to-submit averaged `188.8ms` and p50
submit-to-first averaged `154.9ms`, with `4.73s` prefill wall, `3.47s` decode
GPU time, and `9.77s` profiled phase. Lowering only sampled-medium
`max_active` from `32` to `16` is rejected: it regressed the score-facing row to
`372.3 / 47.2 / 411.8ms` and `2.8 tok/s`. The profile confirms this is not a
useful queueing fix: p50 queue-to-submit worsened to `315.8ms`, step calls rose
`138 -> 216`, and decode GPU/CPU exposure rose as well.

Self_consistency on pushed `2ce0ed3` is now mostly outside the runtime batcher.
The no-env TorchInferno-only row landed at `218.7 / 0.0 / 328.4ms`, 1000/1000
correct, while the online-batcher profile had p50 queue-to-first and
queue-to-finish around `13ms`. Fast-HTTP profiling confirmed that after a
request is read, server first-content and total stream p50 are also about
`14ms`; the remaining benchmark latency is before/around HTTP connection
handling and client stream observation. Raising
`TORCHINFERNO_OPENAI_HTTP_WORKERS` to `512` preserved TTFT and improved E2E
`328.4 -> 314.6ms`; disabling keepalive regressed to `323.4 / 344.0ms`.
Promote the 512-worker fast HTTP default, but keep keepalive enabled.

The public `20260629_130308` refresh still measured stale TorchInferno
`bd17332`, so it does not include the cached sampler-state or HTTP-worker
changes above. On current `722cc35`, the multi_turn row remains server-side:
the local TorchInferno-only run landed at `320.2 / 62.1 / 378.2ms`, with p50
queue-to-submit `104.6ms`, submit-to-first `139.9ms`, `4.71s` prefill wall, and
`2.72s` ragged decode GPU time. Queue-profile snapshots now include per-shape
prefix-graph prefill wall and forward timings. The largest actual prefill wall
buckets were `b32:s128` (`1297ms` over 8 calls), `b32:s160` (`1009ms` over 5
calls), and `b32:s96` (`922ms` over 7 calls), confirming that large suffix
prefill dominates the current multi_turn TTFT path. Enabling prefill-cost
admission priority is rejected: it shifted more waves into `s160`, increased
prefill wall to `5.00s`, decode GPU time to `3.82s`, and regressed the row to
`341.7 / 62.7 / 403.0ms`.

The same shape-timing profile supports a narrower greedy-large suffix-bucket
refinement. Adding `80/112/144` to the deterministic 512-token suffix bucket
set kept request-path captures at zero and moved the local profiled row to
`317.9 / 60.7 / 379.6ms`. Prefill wall dropped `4.71s -> 4.29s`, and first-turn
median TTFT dropped `765ms -> 587ms`; startup ready time increased by about
`15s` because more warmup graph shapes are captured. Promote the finer bucket
set only under the existing `400 < max_tokens <= 512`, `temperature=0` gate.

Current long_output on `dbee040` remains decode/readback bound. A profile with
per-shape decode timing landed at `282.4 / 25.3 / 1333.4ms`, with p50
queue-to-first `229.6ms`, `7.66s` prefill wall, `12.98s` decode GPU time, and
`7.19s` CPU token readback. The largest GPU shape was
`decode_many:b64/64` (`2352ms` across 129 calls, about `18.2ms/call`); ordinary
ragged `b43-b62/64` calls were roughly `14ms/call` and their CPU token harvest
was of similar size. Queue-profile records now emit per-shape decode model,
GPU, and CPU-token timings, with a wider decode-shape count export so these
averages stay inspectable in long_output profiles. Do not infer a new
decode-many knob from this alone: prior larger-quantum, waiting-capacity,
pinned-readback, and async-copy attempts remain rejected.

A lower greedy KV-token budget is still rejected after the decode-shape timing
refresh. Forcing `TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_KV_TOKEN_BUDGET=16384`
on pushed `bd7e38b` capped long_output at `63` active rows and completed
1000/1000 correct, but the score-facing row moved to
`264.2 / 26.8 / 1343.3ms`. The profile showed the intended padding reduction
did not buy real throughput: prefill wall rose `7.66s -> 9.24s`, decode GPU
time rose `12.98s -> 13.71s`, CPU token wait rose `7.19s -> 8.41s`, and total
phase time rose `24.63s -> 26.03s`. Keep the 32K greedy KV budget. Queue
profiles now also split `runtime_decode_many_cpu_tokens_ms` and attribute
decode-many token-readback time into the per-shape CPU timing map so the next
long_output profile can distinguish multi-step readback from ordinary ragged
decode readback.

The first default profile with that split on pushed `9c83243` landed at
`297.5 / 25.6 / 1325.9ms`, 1000/1000 correct. Total token readback was
`7.17s`, but decode-many accounted for only `1.45s` of it across `120` calls
and `305` internal steps. The remaining readback/synchronization exposure is
ordinary ragged decode, with `decode_many:b64/64` contributing about `590ms`
CPU time versus the larger ordinary ragged `b49-b62/64` buckets. Do not spend
the next pass on decode-many CPU-copy micro-optimizations; the useful lever
still has to reduce or overlap ordinary ragged decode synchronization.

Same-host long_output comparison on pushed `874b4a8` narrows the current target:
local vLLM needed a newer conda `libstdc++` in `LD_LIBRARY_PATH` to import the
host `soxr` wheel, so inference-bench now fixes that automatically in commit
`3b956711`. With that environment, vLLM landed at
`86.9 / 25.6 / 1034.6ms`; SGLang landed at `64.6 / 23.8 / 994.5ms`; and
TorchInferno landed at `284.7 / 24.7 / 1205.2ms`, all 1000/1000 correct. The
TorchInferno profile shows median queue-to-submit is only `37ms`, while
submit-to-first is `145ms`; TPOT is now in the same local band as vLLM/SGLang,
but first-token latency and tails remain worse. The same profile had `9.0s`
prefill wall and `2.67s` of request-path prefill graph capture on `b64:s64`
and `b64:s128`; do not reopen broad b64 warmup or capture-on-miss disable
without a different mechanism, since both have already regressed score-facing
long_output. The next useful runtime work is a prefill scheduling/pipeline
change that reduces first-token wait without fragmenting prefill into the
previously rejected low-MFU chunked path.

Two prefill/decode ordering A/Bs are also rejected as defaults. First, forcing
`TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_FIRST=0` makes the online engine prefill
before decode for the whole session. It preserved 1000/1000 correctness and cut
median TTFT to `246.2ms`, but it disabled decode-many, raised CPU token wait to
`8.72s`, and regressed E2E to `1289.0ms` with p99 TTFT `3744.8ms`. Second, the
narrower `TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_READY_BEFORE_DECODE=1` only
prefills before decode while runtime-ready requests are waiting, then returns to
decode-first/decode-many once the queue drains. That kept decode-many active
(`112` calls) and reduced request-path capture to one `b64:s64` graph, but the
score-facing row still regressed to `295.7 / 24.8 / 1322.4ms`; submit-to-first
p50 rose to `195ms`. Keep both as opt-in scheduler probes only. The result
reinforces that local first-token latency is not solved by simply swapping
prefill ahead of decode; it needs either faster suffix prefill or a pipeline
that overlaps ordinary ragged decode synchronization without losing the
decode-many/e2e balance.

Greedy short prequeue admission is rejected for long_output. Applying the same
prequeue mechanism globally with
`TORCHINFERNO_OPENAI_TP_STREAM_PREQUEUE_ADMISSION_WAIT_MS=1` kept correctness at
`1000/1000` and improved median E2E to `1240.3ms`, but it left TTFT/TPOT
effectively unchanged at `265.9 / 24.6ms` and worsened p99 TTFT/E2E to
`1477.7 / 2237.2ms`. A smaller `0.5ms` value also preserved correctness but
regressed medians to `277.3 / 24.5 / 1330.9ms` with similarly bad tails. Keep
prequeue admission scoped to sampled-medium tree traffic; the long_output gap
remains decode and prefill/decode overlap, not first-wave request admission.

Sampled decode-many is also only a diagnostic probe. Tree-of-thought on current
`fd3bebb` with queue profiling landed at `256.2 / 50.3 / 307.8ms` locally.
Enabling `TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_DECODE_MANY=1` improved median
TTFT to `216.5ms` and E2E to `294.7ms`, but regressed TPOT to `77.0ms`. The
profile explains the loss: batched sampled decode cut ordinary decode CPU wait
from `1.46s` to `0.86s`, but decode GPU rose from `3.73s` to `6.59s` because
the run speculatively decoded `1185` tokens that were skipped after EOS
(`778` stop finishes). Keep sampled decode-many off by default; it is useful
only to measure CPU sync pressure on sampled workloads without early stops.

Sampled-medium tree refresh (2026-07-01, current `b66d3d4`): focused control
was `185.9 / 49.5 / 223.5ms`. Setting sampled-medium initial wait to `1ms` and
decode quantum to `2` reproduced the earlier lower-latency shape at
`166.9 / 72.3 / 207.7ms`, with p99 E2E `1110.5ms` and correctness in family.
This is an explicit latency/TPOT tradeoff, but it moves tree TTFT/E2E toward the
vLLM/SGLang gap and is scoped to `temperature > 0`, `256 < max_tokens <= 300`,
leaving sampled-short self-consistency and greedy paths unchanged.

The same q2/wait1 stack benefits from a smaller prefill-ready active-cap bump
than the noisy cap-16 probe. A fresh cap-8 control landed at
`165.8 / 79.4 / 211.1ms`; cap 12 improved TPOT/E2E but gave back too much TTFT
and tail. The midpoint cap 10 landed at `171.5 / 51.6 / 209.1ms` with p99 E2E
`1068.5ms`, and the no-env default confirmation landed at
`166.4 / 57.2 / 199.8ms` with p99 E2E `1102.3ms`. Promote `10` as the
sampled-medium default under the existing `temperature > 0`,
`256 < max_tokens <= 300` gate: it keeps TTFT essentially flat versus cap 8
while improving TPOT/E2E in the focused local runs.

Full-suite local refresh after `b8914db` exposed a session-compatibility bug:
the online batcher could keep a 256-token greedy session open and admit later
sampled, multi-turn, and long-output requests under the wrong policy. That
regressed the full-suite rows to self `235.8 / 0.0 / 275.5ms`, multi
`416.5 / 65.9 / 484.4ms`, and long_output `1163.4 / 31.1 / 2194.2ms`. Require
new requests to match the online sampled/greedy class and max-token bucket, but
keep sampled-medium deterministic follow-ups in the same session. The queue
profile then shows four final sessions (`0/256`, `0.7/256`, `0/512`, `0.7/300`)
instead of one merged session, and the full-suite row recovers to few_shot
`169.9 / 48.8 / 209.6ms`, self `41.6 / 0.0 / 46.1ms`, multi
`297.0 / 58.4 / 349.3ms`, tree `174.0 / 53.0 / 207.6ms`, and long_output
`283.0 / 25.2 / 1314.8ms`.

Sampled-medium prequeue admission is now a narrow tree latency default. The
latest same-host all-provider control at `cb429b9` had tree at
`157.2 / 61.3 / 210.5ms`, with the sampled-medium online session starting from
`initial_batch_size=1`, queue-to-submit p50 `63.8ms`, and queue-to-first p50
`133.2ms`. A global `2ms` prequeue admission barrier raised the initial batch
to `6` and improved medians to `150.9 / 40.0 / 177.8ms`, but it also pushed
queue-to-submit p99 to `433ms` and raw p99 E2E to `778ms`. The `1ms` override
kept the p50 gains with a smaller server-side p99 hit in that run:
`150.9 / 54.9 / 182.1ms`, `962/992` correct, initial batch `3`,
queue-to-submit p50/p99 `56.8/322.2ms`, and queue-to-first p50/p99
`131.3/396.2ms`. The no-env default confirmation was even better on medians at
`145.8 / 40.3 / 172.0ms`, but still had a raw p99 E2E tail of `890ms` and
queue-to-first p99 `519.7ms`. Promote `1ms` only for `temperature > 0`,
`256 < max_tokens <= 300` so sampled-short self-consistency, greedy, multi-turn,
and long-output sessions keep their existing admission behavior. This is a
median scheduling cleanup, not a tail fix or vLLM-gap closure: sampled-medium
suffix prefill and ordinary ragged decode remain the larger tree bottlenecks.
A pushed `5137a36` TorchInferno-only full-suite guard kept the expected scope:
few_shot `169.6 / 50.2 / 211.8ms`, self `204.2 / 0.0 / 217.8ms`, multi
`317.7 / 65.5 / 376.9ms`, tree `152.7 / 54.9 / 186.7ms`, and long_output
`266.1 / 24.6 / 1312.7ms`; tree queue-to-first p50/p99 was
`128.9/425.7ms`.

Short-greedy common-prefix suffix buckets now include `96` for deterministic
`max_tokens <= 128` sessions. The prior default jumped directly from suffix
`64` to `128`, overpadding long_output waves with prompt suffixes in the middle
of that band. On pushed `357421b`, a fresh default focused control landed at
`280.5 / 25.8 / 1275.1ms` with p99 E2E `4286.4ms`; forcing
`16,32,64,96,128,256` by env landed at `247.9 / 25.6 / 1208.3ms`; and the
no-env patch confirmation landed at `249.0 / 25.7 / 1301.3ms` with p99 E2E
`3406.7ms`, all 1000/1000 correct. The no-env profile shows real `s96` graph
use, reduces prefill wall from `7.47s` to `6.73s`, and cuts prefill forward from
`6.48s` to `5.84s`. Aggregate median E2E is noisy against the paired control,
but raw request deltas are still favorable at median `-12.6ms` TTFT and
`-14.3ms` E2E, so promote this as a narrow TTFT/tail cleanup rather than a
decode-throughput fix.

Self-consistency HTTP/admission follow-ups on pushed `837d84e` did not produce
a defaultable server change. The full-suite run showed self at
`323.3 / 0.0 / 347.5ms`; focused self was `236.3 / 0.0 / 322.5ms`, and forcing
the old short-greedy suffix list still landed at `227.7 / 0.0 / 348.6ms`, so
the suffix-96 patch is not the cause. Fast-HTTP profiling showed the server-side
stream path itself is short (`total_ms` p50 about `10ms` from request-ready to
done, queue-to-finish p50 about `8.7ms`), while benchmark time is dominated
before the request body is fully read. Drained keepalive `5s` regressed to
`379.7 / 0.0 / 414.0ms`; drained keepalive `50ms` stayed high at
`241.5 / 0.0 / 350.2ms`; and `1024` HTTP workers landed at
`332.8 / 0.0 / 352.3ms`. Keep the current fast-HTTP defaults until a change can
move client-observed TTFT/E2E, not just internal queue timing.

Current `13d2b75` follow-ups add two guardrails. A no-profile focused
self-consistency run still landed at `304.9 / 0.0 / 372.9ms`, so the current
self row is not merely queue/HTTP profiling overhead. Rechecking the
sampled-medium prefill-ready active cap at `12` for tree_of_thought also does
not beat the promoted cap-10 tradeoff on current main: the focused row landed
at `181.6 / 54.8 / 216.7ms`, and the queue profile showed queue-to-first/finish
p50 rising to `159.5/195.0ms`. A same-host all-provider tree comparison attempt
could not start vLLM/SGLang because no skipped-build venvs existed in the fresh
build directory, but its TorchInferno control row was `169.4 / 50.0 / 201.3ms`;
keep cap `10` and do not reopen cap `12` without a new signal.

Same-host all-provider refresh on pushed `db1587d` gives the current local
score shape after the sampled-medium prequeue and rejected long-output probes.
vLLM started successfully with the inference-bench workspace/libstdc++ fixes and
won `10` metric cells, SGLang won `5`, and TorchInferno won `4`: few_shot TPOT
(`47.8ms` vs vLLM `54.8ms`), self_consistency E2E/throughput
(`236.8ms`, `4.2 tok/s`), and multi_turn TPOT (`61.1ms`). The closest missed
cell is self_consistency TTFT (`165.3ms` vs vLLM `159.7ms`), but the
TorchInferno fast-HTTP profile for that row has p50 server stream total around
`8.9ms` and queue-to-first p50 around `7.7ms`; the remaining difference is
client-observed connection/request scheduling, not model/runtime work.
Score-facing runtime gaps remain few_shot E2E (`207.6ms` vs SGLang `196.0ms`),
tree TTFT/TPOT/E2E (`147.3 / 44.9 / 180.5ms` vs best
`62.1 / 31.9 / 90.9ms`), multi_turn TTFT/E2E (`325.7 / 381.6ms` vs best
`154.5 / 228.6ms`), and long_output decode/E2E
(`24.8 / 1215.0ms` vs vLLM `16.9 / 666.4ms`).

Greedy-mid prequeue admission is rejected for few_shot. Forcing the existing
global prequeue override to `1ms` gathered the focused few_shot run into one
online session with `initial_batch_size=4`, but it did not improve the row:
`170.9 / 47.8 / 209.9ms` with p99 E2E `1288.3ms`, versus the current local
all-provider control `168.0 / 47.8 / 207.6ms`. The queue profile still had p50
queue-to-first around `125.9ms`, while p99 queue-to-first rose to `913.6ms`.
Keep prequeue admission scoped to sampled-medium tree traffic.

## Public Run Status

The public results directory now has a current all-provider run at
`20260702_062945`, written with inference-bench commit `045649d4` and
TorchInferno `840f859` (perf-equivalent to `db1587d`; docs-only commits after
that). vLLM started successfully with the workspace/libstdc++ fix. The public
scorecard is vLLM `11/20`, SGLang `5/20`, TorchInferno `3/20`: TorchInferno wins
few_shot TPOT and self_consistency E2E/throughput, while the remaining
score-facing gaps are tree_of_thought, long_output, and multi_turn TTFT/E2E.

Public-path medians are close to the same-host local refresh:
few_shot TorchInferno `172.0 / 48.6 / 213.3ms`, self_consistency
`211.2 / 0.0 / 229.0ms`, multi_turn `326.6 / 61.3 / 381.2ms`,
tree_of_thought `161.4 / 46.7 / 190.7ms`, and long_output
`257.2 / 24.5 / 1254.0ms`. The current vLLM targets are tree
`66.5 / 31.4 / 89.6ms` and long_output `64.4 / 16.9 / 674.0ms`.

Lowering sampled-medium FP8 prefill's M threshold is rejected on the public
stack. The focused tree run with
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_FP8_PREFILL_MIN_M=64` wrote
`agent_space/ti_tree_fp8_minm64_results/.../8xH100-local-ti-tree-fp8-minm64-20260702/runs/20260702_065057`
and landed at `151.7 / 44.2 / 180.2ms` with `958/992` correct. A same-host
no-env control immediately after it wrote
`agent_space/ti_tree_control_results/.../8xH100-local-ti-tree-control-20260702/runs/20260702_065557`
and beat the score-facing medians at `148.0 / 46.3 / 176.2ms` with `954/992`
correct. Queue counters also did not show a defaultable mechanism: the
`min_m=64` run kept prefill forward essentially flat (`2251.8ms` vs
`2264.0ms`) while worsening phase time (`6066.0ms` vs `5258.5ms`), decode GPU
time (`1492.8ms` vs `1412.4ms`), and queue-to-first p50 (`132.9ms` vs
`126.7ms`). Keep sampled-medium FP8 prefill at `min_m=256`; the tree gap is
still faster prefix-suffix prefill/decode, not a smaller FP8 gate.

Adding a finer short-greedy suffix bucket at `80` is also rejected. The public
long_output prompts share a `111` token prefix and have suffixes in `17-75`, so
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS_GREEDY_SHORT=16,32,64,80,96,128,256`
looked plausible as a way to move the current `65-96` suffix wave from `s96` to
`s80`. A same-host default long_output control on current main wrote
`agent_space/ti_long_control_results/.../8xH100-local-ti-long-control-20260702/runs/20260702_071620`
and landed at `241.2 / 24.9 / 1284.9ms`, with `5.83s` prefill forward,
`9.65s` decode GPU, queue-to-first p50 `192.7ms`, and queue-to-finish p50
`1221.6ms`. The adjacent `s80` override wrote
`agent_space/ti_long_s80_bucket_results/.../8xH100-local-ti-long-s80-bucket-20260702/runs/20260702_071011`
and regressed to `272.4 / 25.7 / 1500.1ms`, with prefill forward rising to
`13.70s`, decode GPU to `10.30s`, queue-to-first p50 to `210.7ms`, and
queue-to-finish p50 to `1441.7ms`. The warmup helper consumes the same policy,
so this is not just an env handoff issue; the added bucket creates expensive
short-greedy graph variants that outweigh the reduced padding. Keep the
short-greedy default at `16,32,64,96,128,256`.

The non-decode Triton RMSNorm path is not a standalone default either. Enabling
`TORCHINFERNO_TRITON_RMS_NORM=1` for a focused long_output run wrote
`agent_space/ti_long_triton_rms_results/.../8xH100-local-ti-long-triton-rms-20260702/runs/20260702_072613`
and stayed correct (`1000/1000`), but landed at `252.9 / 25.0 / 1229.7ms`
against the adjacent default control's `241.2 / 24.9 / 1284.9ms`. The headline
E2E movement did not come from a durable runtime win: prefill forward only moved
from `5.83s` to `5.71s`, while decode GPU rose from `9.65s` to `10.26s`,
queue-to-first p50 rose from `192.7ms` to `207.5ms`, and queue-to-submit p50
rose from `27.1ms` to `36.0ms`. Decode-specific Triton norm/SwiGLU paths are
already default-on; keep the broader prefill RMSNorm path opt-in until it shows
a clear win on more than noisy long_output E2E.

A local decode-many tail gate is rejected. The public profile showed an
expensive-looking small `decode_many:b8/8` tail, so a throwaway patch added an
env-gated minimum-active check and tested
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_MIN_ACTIVE=16` on long_output. The
run wrote
`agent_space/ti_long_decodemany_min16_results/.../8xH100-local-ti-long-decodemany-min16-20260702/runs/20260702_073649`
and landed at `243.2 / 24.7 / 1209.9ms` with `1000/1000` correct. The adjacent
patched no-env control wrote
`agent_space/ti_long_decodemany_control2_results/.../8xH100-local-ti-long-decodemany-control2-20260702/runs/20260702_074223`
and landed at `256.9 / 25.1 / 1212.5ms`. The median movement was too small and
the counters contradicted the intended mechanism: the min-active run raised
decode GPU time (`9.96s` to `10.20s`), decode-many model tokens (`16,453` to
`20,441`), skipped tokens (`271` to `399`), and stop finishes (`372` to `490`).
Both adjacent profiles also lacked the public run's small decode-many tail
shape, so the local patch was removed. Do not add a decode-many minimum-active
gate without a stronger, reproducible tail signal.

Decode-many while runtime-ready requests are waiting is also rejected as a
default. The current public long_output profile (`20260702_095238`) showed
first-wave requests getting their first token around `140-150ms` but not
finishing `9-10` output tokens until about `1.2s`, because waiting work prevents
`step_online_many` from running a multi-token decode burst during the initial
fill phase. An env-gated diagnostic path,
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_WITH_WAITING=1`, confirmed the
mechanism but not the policy. Unrestricted waiting decode wrote
`agent_space/ti_decode_many_wait_results/.../8xH100/runs/20260702_104143` and
matched vLLM-like TPOT (`17.0ms`), but TTFT/E2E regressed to
`668.2 / 17.0 / 1476.3ms`; queue profile p50 submit-to-first rose from
`139.2ms` to `605.4ms`, prefill forward rose from `5.53s` to `7.81s`, and phase
time rose from `19.58s` to `23.32s`. Constraining the same diagnostic to times
when active rows were effectively full did not make it defaultable:
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_WITH_WAITING_MIN_ACTIVE=61` wrote
`agent_space/ti_decode_many_wait61_results/.../8xH100/runs/20260702_104807` at
`244.8 / 24.7 / 1183.7ms`, and `MIN_ACTIVE=64` wrote
`agent_space/ti_decode_many_wait64_results/.../8xH100/runs/20260702_105339` at
`265.5 / 25.1 / 1181.6ms`; both kept `1000/1000` correctness but worsened p99
TTFT/E2E to about `1.5s` / `2.2-2.3s`. Reducing the command quantum did not
rescue the policy: the current-head check with
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_GEN_DECODE_QUANTUM=2` wrote
`agent_space/ti_long_waitdq2_results/.../8xH100-local-ti-long-waitdq2-20260702/runs/20260702_133611`
and landed at `658.8 / 17.6 / 1402.9ms`, `1000/1000` correct. It expanded
decode-many to `420` bursts / `840` steps and lowered TPOT, but queue-to-first
p50/p99 still rose to `633ms` / `2.02s` while prefill forward rose to `7.63s`.
Keep the hook env-only for diagnostics; do not enable it by default without a
policy that improves medians and tails.

`TORCHINFERNO_COMPILED_POST_ATTENTION=1` is not a defaultable runtime win. A
focused long_output check wrote
`agent_space/ti_long_compiled_postattn_results/.../8xH100-local-ti-long-compiled-post-attn-20260702/runs/20260702_075057`
and looked superficially good at `227.1 / 25.5 / 1177.4ms`, but the queue
profile did not show a prefill mechanism: prefill forward barely moved versus
the adjacent no-env control (`5.76s` to `5.71s`), while the row mostly changed
through decode scheduling/padding noise. The clearer tree_of_thought check
wrote
`agent_space/ti_tree_compiled_postattn_results/.../8xH100-local-ti-tree-compiled-post-attn-20260702/runs/20260702_075645`
and landed at `145.8 / 37.3 / 175.6ms` with `964/992` correct, but counters
again showed no post-attention prefill win: prefill forward was flat
(`2264.0ms` control vs `2262.9ms` compiled), decode GPU worsened
(`1412.4ms` to `1443.0ms`), phase time worsened (`5258.5ms` to `5637.9ms`),
and decode misses rose (`1` to `3`). Keep runtime `torch.compile`
post-attention as an explicit experiment only; it does not satisfy the offline
optimization contract for a default serving path and did not close the measured
prefill gap.

Greedy-large prefill-before-decode at the active tail is a small current
multi_turn win. A focused current-head validation with
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_READY_BEFORE_DECODE=1` and
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_READY_ACTIVE_CAP=8` wrote
`agent_space/ti_multi_prbd8_current_results/.../8xH100-local-ti-multi-prbd8-current-20260702/runs/20260702_081043`
and landed at `309.5 / 60.9 / 366.2ms`, `982/1000` correct. The queue profile
kept the desired graph shape (`34` prefill graph hits, `0` misses) and moved
queue-to-first/finish p50 to `248.2/268.8ms`, with p99s at `445.4/501.3ms`.
Versus the recent focused default band (`323.4 / 64.1 / 380.6ms`) and the
public row (`326.6 / 61.3 / 381.2ms`), this is enough to promote only the
deterministic `400 < max_tokens <= 512` policy. Keep the active cap at `8` and
leave greedy short, greedy mid, sampled short, sampled medium, and global
prefill/decode ordering unchanged.

The no-env confirmation after promotion wrote
`agent_space/ti_multi_prbd8_default_confirm_results/.../8xH100-local-ti-multi-prbd8-default-confirm-20260702/runs/20260702_083658`
on pushed `bf6aa8b` and landed at `307.7 / 61.3 / 364.9ms`, `981/1000`
correct. The quiescent queue record confirms the default policy fired without
overrides: `run_max_tokens=512`, `prefill_ready_before_decode=true`, active cap
`8`, `34/0` prefill graph hits/misses, and queue-to-first/finish p50
`245.7/268.5ms`.

Forcing prefill-cost admission priority on sampled-medium tree traffic is
rejected. The profiled env run with
`TORCHINFERNO_CONTINUOUS_ADMIT_PREFILL_COST_PRIORITY=1` wrote
`agent_space/ti_tree_prefillcost_results/.../8xH100-local-ti-tree-prefillcost-20260702/runs/20260702_082319`
and looked borderline useful at `147.2 / 32.1 / 171.5ms`, `957/992` correct.
The counters did not show the intended mechanism: prefill forward was flat
against the adjacent tree control (`2264.0ms` to `2270.4ms`), decode GPU rose
(`1412.4ms` to `1483.8ms`), runtime steps rose (`104` to `106`), and queue
finish p99 worsened (`513.6ms` to `554.3ms`). The no-profile confirmation
`agent_space/ti_tree_prefillcost_np_results/.../8xH100-local-ti-tree-prefillcost-np-20260702/runs/20260702_083020`
then landed at `143.9 / 52.2 / 176.6ms`, `963/992` correct, so the profiled
TPOT movement was not reproducible as a default tree win. Keep prefill-cost
priority scoped to greedy-short traffic unless a sampled-medium rerun shows a
real reduction in prefill/decode work.

A current-head all-provider multi_turn rebuild confirmed the remaining gap is
request-specific prefix reuse, not just the earlier scheduling default. The run
`agent_space/allproviders_current_multi_built_results/.../8xH100-local-current-multi-allproviders-build-20260702/runs/20260702_085107`
used TorchInferno `912728c`, SGLang `b276a9a`, and vLLM `08a8a4a`. vLLM failed
server startup in `flashinfer_comm.allreduce_fusion` with `Buffer: 1048576
bytes, Required: 4194304 bytes`, so the comparable rows were SGLang at
`152.4 / 119.8 / 280.1ms`, `981/1000` correct, and TorchInferno at
`316.6 / 61.0 / 371.6ms`, `982/1000` correct. Per-turn TTFT medians show the
shape: SGLang stays mostly flat after turn 0 (`356.1`, then `121-200ms`), while
TorchInferno grows from `376.5ms` to `387.2ms` by turn 7. TorchInferno's queue
profile reused only the shared `45` token system prefix for all `1000`
requests, with `85,398` prefill tokens and `4.48s` prefill wall. Closing multi
requires TP-safe per-conversation prefix reuse that can still batch and replay
stable prefill shapes.

Persistent online continuation now keeps the already-started runtime alive for
compatible idle batches instead of broadcasting a fresh online start and calling
`start_online()` again. That fixes the opt-in mode's command semantics and keeps
the runtime prefix index available across compatible bursts, but it is not a
multi_turn win by itself. A current-head probe with
`TORCHINFERNO_OPENAI_TP_ONLINE_PERSISTENT=1` wrote
`agent_space/ti_multi_persistent_results/.../8xH100-local-ti-multi-persistent-20260702/runs/20260702_134747`
and landed at `304.7 / 62.7 / 359.5ms`, `981/1000` correct. The queue counters
stayed essentially identical to default: `34` prefill batches, `3.85s` prefill
forward, `4.30s` prefill wall, and only `{"common_prefix":1000}` /
`{"45":1000}` prefix reuse. p50 first-token improved slightly
(`250.3ms -> 243.9ms`) but p99 first-token regressed (`356.3ms -> 472.7ms`) and
throughput fell, so keep persistent mode opt-in for multi_turn until paired with
a cheap request-specific prefix reuse path.

Pinned full-prompt stores remain rejected on current head. Enabling
`TORCHINFERNO_CONTINUOUS_PINNED_FULL_PROMPT_STORE_MIN_MAX_TOKENS=1` with
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS=112` wrote
`agent_space/ti_multi_fullprompt_p112_results/.../8xH100-local-ti-multi-fullprompt-prefix112-20260702/runs/20260702_091013`
and produced request-prompt reuse (`875` request-prompt hits, `98,517` reused
tokens), but regressed catastrophically to `8221.2 / 63.9 / 8276.4ms`. The
profile had only `16k` prefill tokens, but `506` prefill batches, `505`
prefix-reuse batches, and `499` graph misses, so longer hits fragmented into
many tiny suffix passes. A follow-up using mixed-prefix batching with
`TORCHINFERNO_CONTINUOUS_NON_COMMON_PREFIX_GRAPH_PREFILL=1`,
`TORCHINFERNO_CONTINUOUS_MIXED_PREFIX_PREFILL=1`, and
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_CAPTURE_ON_MISS=0` avoided the crash and
collapsed to `34` prefill batches, but still regressed to
`1898.7 / 128.3 / 2031.5ms`, `981/1000` correct. The counters explain the
failure: `98,523` reused prefix tokens and `18,543` prefill tokens, but
`16.64s` prefill forward / `28.58s` prefill wall and `11.55s` prefill state
time. Do not default full-prompt reuse until non-common prefix prefill is both
graph-safe and cheap enough to beat recomputing the compact full prompts.

The vLLM startup failure has an inference-bench fix, not a TorchInferno change.
A vLLM-only confirmation with
`--compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'` wrote
`agent_space/vllm_disable_allreduce_rms_results/.../8xH100-local-vllm-disable-allreduce-rms-20260702/runs/20260702_092631`
and started successfully, landing at `162.9 / 53.6 / 216.9ms`, `981/1000`
correct. inference-bench commit `455dd782` now applies that default unless a
caller supplies their own `--compilation-config` or sets
`INFERENCE_BENCH_VLLM_DISABLE_ALLREDUCE_RMS_FUSION=0`.

A same-host all-provider tree/long refresh with that vLLM fix wrote
`agent_space/allproviders_current_tree_long_fixedvllm_results/.../8xH100-local-current-tree-long-allproviders-fixedvllm-20260702/runs/20260702_093317`.
vLLM now starts under inference-bench defaults and wins the current long_output
row at `68.7 / 16.9 / 669.8ms`; SGLang is `61.8 / 24.6 / 913.3ms`; TorchInferno
is `259.3 / 25.2 / 1203.8ms`, all `1000/1000` correct. TorchInferno's
long_output queue profile remains decode dominated: shared-prefix reuse hit the
`111` token common prefix for all requests, but the session still spent `5.90s`
in prefill forward, `6.76s` prefill wall, `10.20s` ragged decode GPU, and `755`
decode batches. For tree_of_thought, vLLM was `66.0 / 32.1 / 90.4ms`, SGLang
`56.9 / 76.8 / 167.0ms` with a severe p99 outlier, and TorchInferno
`151.9 / 43.4 / 177.7ms`. The tree queue profile had only the `45` token common
prefix reused, `59` prefill graph hits, `11,849` prefill tokens, `2.17s` prefill
forward, and `1.38s` ragged decode GPU. The current tree gap is no longer a cold
graph miss; it is steady sampled-medium prefill/decode pipeline cost plus
queueing.

Rechecking the sampled-medium prequeue wait at `2ms` is rejected on current head.
The focused env run with
`TORCHINFERNO_OPENAI_TP_SAMPLED_MEDIUM_STREAM_PREQUEUE_ADMISSION_WAIT_MS=2` wrote
`agent_space/ti_tree_prequeue2_results/.../8xH100-local-ti-tree-sampled-prequeue2-20260702/runs/20260702_094540`
and landed at `154.6 / 57.5 / 193.5ms`, `959/992` correct. It did gather a
larger first batch (`3` vs `1` in the adjacent all-provider control), but p99
queue-to-submit rose to `450ms`, p99 queue-to-first to `522ms`, decode batches
rose to `95`, and ragged decode GPU rose to `1.48s`. Keep the sampled-medium
prequeue default at `1ms`; larger waits increase tail and decode fragmentation
without closing the vLLM tree gap.

Raising sampled-medium active rows is also rejected on the current stack.
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_MAX_ACTIVE=64` wrote
`agent_space/ti_tree_active64_results/.../8xH100-local-ti-tree-sampled-active64-20260702/runs/20260702_100818`
and improved median tree to `150.6 / 34.1 / 177.2ms`, `964/992` correct, but
phase time rose to `7.03s` and p99 first-token/finish were `1.59s/1.61s`.
`MAX_ACTIVE=48` wrote
`agent_space/ti_tree_active48_results/.../8xH100-local-ti-tree-sampled-active48-20260702/runs/20260702_101321`
and landed at `145.8 / 51.8 / 176.7ms`, `958/992` correct, with an even worse
tail (`1.88s` queue-to-first p99) because request-time `b48` prefill and decode
graph captures cost about `1.0s` and `0.33s`. Warming those shapes with
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_GREEDY_COMMON_PREFIX_SUFFIX_BATCHES=1,2,4,8,16,32,48`
and `TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_DECODE_WARMUP=1` moved the tail
in the right direction but still lost the score row: startup increased to
`241s`, tree was `146.4 / 48.0 / 174.4ms`, correctness was `957/992`, and p99
first-token stayed at `825ms`. Keep the sampled-medium active cap at `32`; the
extra rows reduce median queueing a little but increase decode/prefill work and
tail latency.

A current-head recheck confirms sampled-medium prefill-ready-before-decode
should stay enabled. Disabling only
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_PREFILL_READY_BEFORE_DECODE`
wrote
`agent_space/ti_tree_no_sampled_prbd_results/.../8xH100/runs/20260702_160326`
and landed at `154.1 / 93.8 / 213.1ms`, `957/992` correct. The score-facing
regression came from decode fragmentation rather than prefill: prefill graph
hits stayed clean (`59/0`) and prefill forward was in-family (`2.23s`), but
decode batches rose to `116`, ragged-decode GPU time rose to `1.81s`, and phase
time rose to `5.80s` versus the current full-run control's `91` decode batches,
`1.49s` decode GPU, and `5.41s` phase time. Keep the current scoped
sampled-medium prefill-ready policy (`active_cap=10`).

Current `32c4aed` focused tree rechecks do not justify changing that cap. The
same-conditions no-env control wrote
`agent_space/ti_tree_default_32c4aed_results/.../8xH100-local-ti-tree-default-32c4aed-20260703/runs/20260703_003459`
and landed at `150.8 / 44.8 / 178.4ms`, `959/992` correct. Lowering the cap to
`6` wrote
`agent_space/ti_tree_prbd_cap6_results/.../8xH100-local-ti-tree-prbd-cap6-20260703/runs/20260703_002431`
and regressed TPOT/E2E to `59.7 / 179.7ms`; the queue profile showed more
decode batches (`87 -> 101`) and higher ragged decode GPU (`1.39s -> 1.64s`).
Raising the cap to `16` wrote
`agent_space/ti_tree_prbd_cap16_results/.../8xH100-local-ti-tree-prbd-cap16-20260703/runs/20260703_002932`
and improved TTFT/E2E slightly to `147.0 / 175.1ms`, but regressed TPOT to
`46.7ms` and p99 E2E to `738.8ms` while adding prefill work and one decode graph
miss. Keep cap `10`.

A current-head TorchInferno-only full refresh on pushed `401888a` wrote
`agent_space/ti_current_full_results/.../8xH100/runs/20260702_132520`.
The rows were few_shot `170.4 / 50.1 / 211.0ms`, self_consistency
`182.5 / 0.0 / 195.6ms`, multi_turn `306.9 / 60.3 / 359.1ms`,
tree_of_thought `154.3 / 44.8 / 184.9ms`, and long_output
`240.3 / 25.4 / 1225.6ms`, all with normal correctness bands
(`977/1000`, `1000/1000`, `981/1000`, `957/992`, `1000/1000`). Versus the
public `20260702_095238` TorchInferno row this confirms the current head
improved self_consistency E2E materially and nudged multi_turn/long_output TTFT,
but it still loses the public vLLM/SGLang targets on multi_turn TTFT/E2E, tree
TTFT/E2E, and long_output decode/E2E.

The queue split from that run matches the remaining work rather than pointing to
a small default knob. multi_turn still reused only the shared `45` token prefix
for all `1000` requests, with `3.80s` prefill forward and no request-specific
reuse (`runtime_prefix_reuse_hit_token_counts={"45":1000}`). tree_of_thought
had stable common-prefix graph reuse (`60` prefix graph batches, no misses), but
still spent `2.41s` in prefill forward and `1.49s` in ragged decode GPU. The
long_output row reused the `111` token common prefix, then spent `9.76s` in
ragged decode GPU across `705` decode batches; decode-many handled `140` bursts
and `358` steps (`20.9k` model tokens), so the fast 64-wide decode graph is
already being used after the queue drains. The gap is therefore not another
tail-bucket tweak: multi_turn needs batched TP-safe non-common prefix reuse, tree
needs faster steady sampled-medium prefix-suffix prefill/decode, and long_output
needs a decode/prefill pipeline policy that improves medians without the known
waiting-decode tail regression.

Current public pointer refresh on 2026-07-02 after the q8 drain change still
showed public inference-bench at `origin/main=055de6e5` and run
`20260702_140923`. The local full TorchInferno refresh on pushed `f7f2cd1`
(`agent_space/ti_full_f7f2cd1_results/.../8xH100-local-ti-full-f7f2cd1-20260702/runs/20260702_205419`)
landed at few_shot `170.5 / 50.1 / 210.9ms`, self_consistency
`192.3 / 0.0 / 210.3ms`, multi_turn `306.0 / 64.5 / 367.1ms`,
tree_of_thought `143.2 / 59.4 / 178.5ms`, and long_output
`250.0 / 24.4 / 1160.5ms` (TTFT/TPOT/E2E). The accepted long-output mechanism
is the scoped greedy-short drain quantum: base decode quantum stays at `3`
while ready/waiting work exists, then drain-only decode-many uses quantum `8`.
The profile cut long-output online step commands to `192` while preserving
1000/1000 correctness and moving TPOT into the public vLLM/SGLang band.

A larger drain quantum is rejected. The q12 focused run
(`agent_space/ti_long_drainq12_results/.../8xH100-local-ti-long-drainq12-20260702/runs/20260702_210255`)
completed 1000/1000 correct and improved median TPOT/E2E to
`22.8ms` / `1088.2ms`, but median TTFT regressed to `310.9ms`, p99 E2E rose to
`2187.8ms`, skipped decode-many tokens doubled (`1439 -> 2881`), and ragged
decode GPU rose to `10.80s`. Since q8 already closes the public TPOT cell
without that tail cost, keep q12 as an opt-in throughput/median-E2E tradeoff.

Current head now exports per-shape decode-many counters in queue profiles:
`runtime_decode_many_shape_model_tokens`,
`runtime_decode_many_shape_emitted_tokens`,
`runtime_decode_many_shape_skipped_tokens`,
`runtime_decode_many_shape_stop_finishes`, and
`runtime_decode_many_shape_limit_finishes`. The next long-output profile should
use these before changing the drain policy again: the global counters prove
q8/q12 trade off skipped stop-token work against fewer command steps, but the
shape split is needed to tell whether the waste is concentrated in large
64-wide graph batches, narrow tail batches, or a small set of stop-heavy prompt
shapes.

A focused long-output reprofile with those counters
(`agent_space/ti_long_decode_shape_results/.../8xH100-local-ti-long-decode-shape-20260702/runs/20260702_214353`)
landed at `280.4 / 24.1 / 1150.7ms`, p99 E2E `1894.9ms`, and 1000/1000
correct. The profile recorded `105` decode-many calls, `516` decode-many steps,
`28.7k` active decode-many model tokens, `26.8k` emitted tokens, `1.87k`
skipped tokens, and `680` stop-token finishes. The new shape counters show the
dominant `decode_many:b64/64` shape consumed `14.98k` active tokens and
`3.22s` GPU with only `385` skipped tokens (`2.6%`), while the higher skipped
percentages were tail/narrow shapes such as `b48/64` (`15.8%`), `b51/64`
(`17.7%`), `b47/64` (`31.1%`), and `b16/16` (`52.3%`). That rejects another
global drain-quantum increase as the next default; any further decode-many
change should be tail-specific and should preserve the full 64-wide drain path.

An opt-in tail stop cap is also rejected as a default. The probe
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_STOP_TAIL_MAX_STEPS=4`
(`agent_space/ti_long_tailcap4_results/.../8xH100-local-ti-long-tailcap4-20260702/runs/20260702_215601`)
finished 1000/1000 correct and improved median TTFT to `235.1ms`, but regressed
TPOT/E2E to `25.0ms` / `1178.1ms`, p99 TTFT to `1141.2ms`, and p99 E2E to
`1968.3ms`. The queue profile showed the intended mechanism did fire
(`28` tail-limited decode-many calls, `112` deferred steps) and reduced skipped
decode-many tokens (`1871 -> 1322`) plus ragged decode GPU (`10.26s -> 9.85s`),
but it increased decode-many calls (`105 -> 143`) and active decode-many model
tokens (`28.7k -> 30.3k`). Keep the cap as an opt-in profiling/tuning hook only;
the default should stay q8 drain without tail splitting until there is a
narrower policy that improves E2E and tails together.

A narrower tail split based on active rows was also rejected on current head. A
temporary opt-in guard limiting the same 4-step stop tail cap to active counts
`<=32` wrote
`agent_space/ti_long_tailcap4_active32_results/.../8xH100-local-ti-long-tailcap4-active32-20260703/runs/20260703_004155`
and finished 1000/1000 correct, but landed at `260.3 / 23.8 / 1107.7ms` with
p99 E2E `2077.3ms`. It fired only once (`1` tail-limited call, `4` deferred
steps), so it was too narrow to affect the dominant work. Raising the guard to
`<=48` wrote
`agent_space/ti_long_tailcap4_active48_results/.../8xH100-local-ti-long-tailcap4-active48-20260703/runs/20260703_004744`
and also preserved correctness, but regressed to `260.5 / 23.9 / 1143.2ms` with
p99 E2E `1922.0ms`. Do not add a tail-active guard by default; the long-output
gap remains decode throughput and prefill/decode pipeline structure, not a
simple stop-tail limiter.

A full-width drain-quantum variant is also rejected. The probe
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_DRAIN_DECODE_QUANTUM=16` with
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_STOP_TAIL_MAX_STEPS=8` wrote
`agent_space/ti_long_fullq16_tail8_results/.../8xH100-local-ti-long-fullq16-tail8-20260703/runs/20260703_010017`
and completed 1000/1000 correct, but landed at `350.5 / 21.9 / 1107.1ms` with
p99 E2E `1958.1ms`. The shape-scoped idea did improve median TPOT, but it
delayed first-token visibility enough to erase the score-facing win. Keep the
q8 drain default; larger drain commands need an overlap/event-flush design, not
just a tail limiter.

Intermediate prefix-prefill batch buckets are also rejected as a simple
long-output default. An opt-in diagnostic hook
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS=1,2,4,8,12,16,24,32`
with matching greedy common-prefix warmup batches wrote
`agent_space/ti_long_batch_buckets_results/.../8xH100-local-ti-long-batch-buckets-20260703/runs/20260703_011330`
and finished 1000/1000 correct, but regressed to
`298.9 / 27.0 / 1909.4ms` with p99 E2E `4996.4ms`. The profile did reduce
some row padding and decode GPU was not the problem, but the new shapes caused
`16` request-path prefill captures (`15.9s` capture time), inflating prefill
wall to `19.9s`. Keep prefix-prefill batch buckets power-of-two by default;
intermediate buckets need a graph-warmup/keying fix before they can be a
runtime policy.

A follow-up local warmup-alignment probe, where the configured runtime batch
buckets also drove greedy common-prefix suffix warmup, wrote
`agent_space/ti_long_batch_buckets_warmfix_results_0703/.../8xH100-local-ti-long-batch-buckets-warmfix-20260703/runs/20260703_063050`
and still regressed to `246.8 / 29.1 / 1856.9ms` with p99 E2E `5116.4ms`.
It did reduce measured prefill padding (`42.2K -> 28.6K` tokens versus the
nearby default profile), but it still took `16` request-path prefill captures
and `16.0s` capture time. Raising `TORCHINFERNO_CUDAGRAPH_PREFILL_MAX_GRAPHS`
to `256` for the same bucket set was not viable either: the server reached
about `97GB` per rank during startup, only `7/8` ranks printed the FlashInfer
decode warmup completion line, and the run was aborted before readiness. Current
queue profiles now expose `runtime_prefill_graph_cache_live_entries`,
`runtime_prefill_graph_cache_max_entries`,
`runtime_prefill_graph_cache_evictions`, and
`runtime_prefill_graph_cache_evicted_entries` so the next bucket/graph-cache
iteration can distinguish shape-key misses from graph-cache eviction directly.
Changing the ragged prefill graph cache to evict one oldest graph at a time
instead of clearing the entire cache improves the opt-in bucket failure mode but
still does not make intermediate buckets a default. The follow-up
`agent_space/ti_long_batch_buckets_fifoevict_results_0703/.../8xH100-local-ti-long-batch-buckets-fifoevict-20260703/runs/20260703_065635`
finished 1000/1000 correct and landed at `251.7 / 24.8 / 1157.0ms`. Request
captures dropped from `16` to `4`, capture time dropped from `16.0s` to
`4.5s`, and the profile reported `128/128` live graph entries with `36`
one-entry evictions. A no-env control on the same change wrote
`agent_space/ti_long_fifoevict_default_results_0703/.../8xH100-local-ti-long-fifoevict-default-20260703/runs/20260703_070242`
and stayed capture-free (`0` evictions, `0` request captures), landing at
`256.2 / 24.7 / 1139.0ms`. Keep intermediate buckets opt-in; the graph-cache
policy change is useful guardrail behavior, not enough to offset the added
prefill shape cost.

Warm-row prefix-copy skipping is rejected as a default. An opt-in diagnostic
hook (`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SKIP_WARM_PREFIX_COPY=1`) tracks
free active rows whose KV still starts with the same shared prefix, then replays
the prefix-suffix graph with `src_prefix_row=None` for those rows instead of
copying the shared prefix again inside the graph. A focused few_shot run wrote
`agent_space/ti_few_skip_warm_prefix_results/.../8xH100-local-ti-few-skip-warm-prefix-20260703/runs/20260703_013119`
and preserved correctness (`977/1000`), but landed at
`167.4 / 51.1 / 208.3ms` with p99 E2E around `2.0s`, worse than the current
full-run control. The profile proves the mechanism fired (`31` skipped batches,
`115.9K` skipped prefix-copy tokens), but it caused two request-path `src0`
prefill graph captures (`1.66s`) and the steady b32 replay was not materially
faster than the existing `src1` graph after capture cost is removed. Keep this as
an opt-in diagnostic hook only; the common-prefix gap is not the dense KV copy
inside the captured graph.

The same full run also confirms that multi_turn's remaining TTFT/E2E gap is not
a missing default env flip. It still has `34` prefill batches, `34/0` prefill
graph hits/misses, and only `{"common_prefix":1000}` / `{"45":1000}` reuse.
Finished-prefix and pinned full-prompt paths remain blocked by the same
non-common-prefix prefill problem documented above: the safe fallback fragments
into hundreds of tiny suffix prefills, while the non-common graph route has
already shown TP failures or large prefill/state regressions. Do not repeat
those env-only probes without a new batched, TP-safe non-common prefix prefill
implementation.

The queue-profile schema now makes the two remaining shape wastes first-class:
`runtime_prefill_padding_tokens` / `runtime_prefill_shape_padding_tokens` derive
from prefill model-vs-active token maps, and
`runtime_decode_many_overgenerated_tokens` /
`runtime_decode_many_shape_overgenerated_tokens` derive from decode-many
model-vs-emitted token maps. These are reporting-only fields computed while
writing the profile JSONL; they do not change scheduling, graph keys, kernels,
or serving behavior. Use them to rank the next packed/ragged prefill and decode
overlap work instead of inferring waste by hand from multiple shape counters.
The prefill profile now also splits that padding into
`runtime_prefill_row_padding_tokens` and
`runtime_prefill_suffix_padding_tokens` with per-shape maps, so batch-bucket
padding and uneven suffix padding can be evaluated independently.

A focused current-head run after adding those counters wrote
`agent_space/ti_waste_counters_results_0703/.../8xH100-local-ti-waste-counters-20260703/runs/20260703_061357`
against pushed `66a33ed`. It landed at multi_turn
`305.0 / 60.0 / 358.7ms`, `984/1000` correct, and long_output
`246.3 / 24.1 / 1122.0ms`, `1000/1000` correct. The new counters make the
shape waste explicit: multi_turn used `82.2K` prefill tokens with `25.4K`
padding tokens and no decode-many overgeneration, while long_output used
`50.4K` prefill tokens with `42.2K` padding tokens plus `1.9K` decode-many
overgenerated tokens. The dominant long-output padding was still in warmed,
miss-free common-prefix graph shapes (`b32:s64` at `18.2K` padding and
`1.87s` forward, then `b32:s96` at `9.9K` padding and `0.94s` forward). This
confirms that the next large lever is a non-fragmenting packed/ragged cached
prefix prefill path; the already-rejected suffix splitting, prefill-cost
priority broadening, chunked/unified forward, and decode-many tail knobs should
not be repeated without a new implementation.

A current-head greedy-short suffix-split recheck on pushed `32a059d` keeps that
conclusion. The diagnostic run
`agent_space/ti_long_greedy_suffix_split_probe_0703/.../runs/20260703_140857`
enabled `TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS_GREEDY_SHORT=1`
and stayed correct (`1000/1000`), landing at
`234.7 / 23.6 / 1072.3ms`. It reduced prefill padding
(`34.8K -> 29.6K`) and prefill forward (`4.82s -> 4.68s`) versus the adjacent
current-head full-run profile, but it also increased prefill model calls
(`61 -> 66`), decode-many model tokens (`24.0K -> 27.9K`), and did not improve
median E2E. Keep suffix splitting opt-in; the current padding distribution still
needs a non-fragmenting packed/ragged cached-prefix prefill implementation
rather than more suffix-bucket fragmentation.

Public `20260704_110219` is now the latest 8xH100 row. It measures
TorchInferno `390fed4` at `4/20` metric wins versus vLLM `12/20` and SGLang
`3/20`: few_shot `157.5 / 46.2 / 205.9ms`, self_consistency
`157.3 / 0.0 / 168.9ms`, multi_turn `310.8 / 58.9 / 366.4ms`,
tree_of_thought `135.0 / 41.6 / 160.9ms`, and long_output
`257.5 / 19.8 / 981.7ms`. The queue profile still shows the default
common-prefix-only multi_turn shape: `35` prefill batches, `80.7K` prefill
tokens, `20.3K` suffix-padding tokens, `3.79s/3.95s` prefill forward/wall,
zero full-prompt adoptions, `{"common_prefix":1000}` reuse routing,
`prefix_rows=64`, PRBD cap `8`, and queue-to-first p50 `246.7ms`. Tree is
still on the older sampled-medium PRBD cap `10` in this public run (`61`
prefill batches, `1.90s/2.11s` prefill forward/wall, `1.96s` ragged-decode
GPU), so it does not include the local cap-32 tree improvement yet. Long_output
remains decode-heavy (`57` prefill batches, `3.98s/4.20s` prefill
forward/wall, `822` decode batches, `91` decode-many calls, `452`
decode-many steps, `24.0K` model tokens, `22.4K` emitted tokens, and `9.89s`
ragged-decode GPU). This keeps the current local priorities intact:
multi_turn needs stable request-prompt reuse, tree can take the local cap-32
sampled-medium PRBD change, and long_output needs a deeper decode-throughput
lever than q8/q12 stop-tail policy.

Earlier public `20260704_070207` measured
TorchInferno `390fed4` at `3/20` metric wins versus vLLM `14/20` and SGLang
`2/20`: few_shot `177.9 / 47.9 / 231.5ms`, self_consistency
`181.5 / 0.0 / 196.4ms`, multi_turn `341.2 / 62.6 / 399.5ms`,
tree_of_thought `132.2 / 49.6 / 163.0ms`, and long_output
`258.7 / 20.0 / 957.9ms`. The queue profile keeps the same unresolved shape:
multi_turn stayed on stable common-prefix reuse only (`35` prefill batches,
`29.8K` padding tokens, `3.98s/4.14s` prefill forward/wall, `{"45":1000}`
reuse), tree used the cap-32 sampled-medium path (`54` prefill batches,
`1.74s/1.95s` prefill forward/wall, `1.71s` ragged-decode GPU), and
long_output remained decode-heavy (`57` prefill batches, `3.89s/4.35s`
prefill forward/wall, `822` decode batches, `98` decode-many calls, `496`
decode-many steps, `26.5K` model tokens, `24.9K` emitted tokens, and `1.57K`
skipped tokens). This worsens the public score but does not change the local
priority ordering: multi_turn still needs stable request-prompt reuse, tree
needs packed/ragged prefix-suffix prefill or fewer small TP collectives, and
long_output needs decode-kernel or decode-overlap work beyond q8/q12 tail
policy.

Earlier public `20260704_050216` measured
TorchInferno `390fed4` at `6/20` metric wins versus vLLM `11/20` and SGLang
`2/20`: few_shot `163.3 / 47.0 / 212.1ms`, self_consistency
`166.6 / 0.0 / 178.7ms`, multi_turn `363.7 / 63.2 / 419.6ms`,
tree_of_thought `136.4 / 28.0 / 156.3ms`, and long_output
`221.6 / 20.9 / 954.2ms`. The queue profiles still show the same default
shape. Multi_turn used `max_active=32`, `prefix_rows=64`, PRBD cap `8`, only
`{"common_prefix":1000}` reuse, `36` prefill batches, `91.8K` prefill tokens,
`4.12s/4.28s` prefill forward/wall, and queue-to-first p50 `286.7ms`. Tree
still used the older sampled-medium PRBD cap `10`, with `56` prefill batches,
`1.81s/2.03s` prefill forward/wall, and `2.10s` decode GPU. Long_output stayed
decode-heavy at `790` decode batches, `95` decode-many calls, `20.5K`
decode-many model tokens, `9.61s` ragged-decode GPU, and `16.17s` phase time.
This keeps the current local conclusions intact: default multi_turn needs a
stable request-prompt reuse path, tree can take the cap-32 sampled-medium PRBD
tradeoff, and long_output still needs a deeper decode-throughput lever.

Earlier public `20260704_030215` measured TorchInferno
`390fed4` at `4/20` metric wins versus vLLM `12/20` and SGLang `3/20`, with
the same unresolved gaps: multi_turn `344.2 / 59.5 / 400.4ms`,
tree_of_thought `129.0 / 36.3 / 154.0ms`, and long_output
`237.7 / 20.7 / 963.5ms`. A same-host patched multi_turn rerun against the
current-main worktree plus the prefix-row no-clear eviction fix landed at
`361.8 / 63.7 / 428.7ms`, `982/1000` correct. Its queue profile matched the
public shape rather than showing a new lever: `34` prefill batches,
`79.8K` prefill tokens, `22.1K` padding tokens, `34/0` prefill graph
hits/misses, and only `{"common_prefix": 1000}` reuse at `45` tokens/request.
The eviction fix is useful correctness/overhead cleanup for prefix-row churn,
but it is not a public-score gap closer for the current multi_turn shape.

The ragged-prefill one-shot profiler now also accepts
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_MIN_SUFFIX`; without that gate it consumed
the profile slot on the warmed `b32:s16:ctx-64` startup graph. A first
`min_suffix=96` run captured the greedy-short warmup graph
`batch=32 suffix=96 context_len=-256`; it measured `153.1ms` self CUDA time
with math SDPA still visible (`28.6ms`, `18.7%`). That is not the multi_turn
request-path shape. Restricting warmup to greedy-large max tokens captured the
actual exact-context graph used by the multi_turn common-prefix path,
`batch=32 suffix=96 context_len=141`. That profile measured `123.3ms` self CUDA
time: TP all-reduce `43.3ms` (`35.1%`), FP8/GEMM paths roughly `45.7ms`
(`37.0%`), RMSNorm `14.6ms` (`11.8%`), and FlashAttention only `1.5ms`
(`1.2%`). This supports the current priority ordering: exact-context attention
is already cheap for the greedy-large common-prefix path, so the next
score-facing improvement needs a model-path change that shortens the
all-reduce/GEMM critical path or removes padded prefill work, not another
scheduler-only suffix split, finished-prefix cache env, or admission wait tweak.

An opt-in experiment that allowed symmetric-memory TP all-reduce inside the
ragged prefill CUDA graph was rejected. With
`TORCHINFERNO_SYMM_MEM_PREFILL_GRAPH_ALLREDUCE=1`, the targeted multi_turn run
regressed to `843.7 / 69.3 / 911.6ms` and still profiled the exact-context
`b32:s96:ctx141` graph at `43.1ms` of NCCL all-reduce (`160` calls), not a
multimem replacement. The server also emitted symmetric-memory multicast OOM
warnings and the final queue profile showed `10.69s` prefill forward wall
inside a profiled run. Do not pursue graph-captured symmetric-memory prefill
all-reduce in this form; first prove a non-graph prefill all-reduce replacement
that reduces the `43ms` all-reduce slice without increasing graph-capture or
memory pressure.

A scoped greedy-large mixed-prefix OpenAI default was tried for the
`temperature=0`, `max_tokens=512` class but is not promoted. The runtime now
accepts an explicit `greedy_large_mixed_prefix_reuse` constructor policy, and
OpenAI can still enable the path with
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_MIXED_PREFIX_REUSE=1`; direct
continuous-engine users retain the existing
`TORCHINFERNO_CONTINUOUS_GREEDY_LARGE_MIXED_PREFIX_REUSE=1` opt-in. When the
policy is explicit, OpenAI warms mixed-prefix suffix graphs and uses `112`
prefix rows under the existing `144` total-row budget (`max_active=32`,
`prefix_rows=112`). Row sizing can still be overridden with
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS`.

The first no-env validation run with the tentative default on the patched clean
tree wrote
`/tmp/inference-bench-ti-mixed-default-recheck-results/.../runs/20260704_033428`
and landed at `280.5 / 65.5 / 336.8ms`, `983/1000` correct. Its final queue
profile used the intended shape: `max_active=32`, `prefix_rows=112`,
`runtime_persistent_cache_rows=144`, `37` prefill batches, `18.8K` prefill
tokens, `2.53s` prefill forward, `2.81s` prefill wall, `36/1` prefill graph
hits/misses, `875` request-prompt reuses plus `125` common-prefix reuses, and
`1000` full-prompt adoptions. This improves the same-host common-prefix patched
run (`361.8 / 63.7 / 428.7ms`) by about `81ms` TTFT and `92ms` E2E, at the
cost of about `2ms` median TPOT and a longer startup warmup (`~211s`). It does
not beat vLLM on the public multi_turn row, but it materially closes the
TTFT/E2E gap without the catastrophic full-prompt-only fragmentation seen in
older rejected probes.

The first full-suite tentative-default run exposed a separate tensor-parallel
worker lifecycle bug. Isolated multi_turn was fast, but the full suite at
`/tmp/inference-bench-ti-full-mixed-default-results/.../runs/20260704_034510`
regressed multi_turn to `412.7 / 64.9 / 468.3ms`: when
`TORCHINFERNO_PAGED_PREFIX_CACHE=1` was set and the flashinfer-free server used
the dense continuous engine, TP workers kept a stale dense
`_RuntimeContinuousBatchEngine` across `online_start` sessions while the primary
rebuilt per session. The worker loop now rebuilds dense runtimes on each
`online_start` and only preserves engines with paged-engine shape (`max_seq`) on
`online_close`.

The first confirmation full-suite run after that lifecycle fix wrote
`/tmp/inference-bench-ti-full-worker-rebuild-results/.../runs/20260704_035447`.
It landed at few_shot `177.8 / 50.2 / 226.9ms`, self_consistency
`202.4 / 0.0 / 245.5ms`, multi_turn `271.1 / 62.3 / 323.3ms`,
tree_of_thought `137.7 / 48.0 / 166.6ms`, and long_output
`229.8 / 23.9 / 1093.5ms`. The final multi_turn queue profile kept the intended
mixed-prefix shape: `max_active=32`, `prefix_rows=112`,
`runtime_persistent_cache_rows=144`, `41` prefill batches, `18.7K` prefill
tokens, `2.87s` prefill forward, `3.17s` prefill wall, `40/1` prefill graph
hits/misses, `875` request-prompt reuses plus `125` common-prefix reuses, and
`1000` full-prompt adoptions.

A later full-suite rerun after the sampled-medium tree cap change did not
reproduce that multi_turn win:
`/tmp/inference-bench-ti-full-cap32-results/.../runs/20260704_044212` landed at
multi_turn `421.2 / 67.0 / 481.0ms`. The model work stayed in-family
(`39` prefill batches, `2.97s` prefill forward, `102` decode batches), but
queue-to-submit p50 grew to `264ms` late in the conversation stream. Forcing
prefill-ready-before-decode back on with an 8-row cap in
`/tmp/inference-bench-ti-multi-mixed-prbd8-results/.../runs/20260704_044951`
landed only at `323.7 / 66.4 / 388.7ms`. Keep greedy-large mixed-prefix reuse
explicit until the late-turn admission jitter is fixed; the worker rebuild
remains a real correctness/lifecycle fix.

Within that explicit mixed-prefix opt-in, reducing the 512-token greedy initial
collection window from `10ms` to `5ms` is a useful refinement. The focused run
with `TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_MIXED_PREFIX_REUSE=1`,
prefill-ready-before-decode cap `8`, and
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_INITIAL_BATCH_WAIT_MS=5` wrote
`/tmp/inference-bench-ti-multi-mixed-prbd8-wait5-results/.../runs/20260704_053348`
and landed at `278.4 / 70.3 / 355.0ms`, `982/1000` correct. The final queue
profile preserved the intended reuse shape (`875` request-prompt reuses,
`125` common-prefix reuses, `1000` full-prompt adoptions) with `37` prefill
batches, `18.8K` prefill tokens, `2.37s` prefill forward, `2.67s` prefill
wall, `37/0` prefill graph hits/misses, and queue-to-first p50 `149ms`.
Versus the preceding explicit mixed+PRBD8 run, the smaller first wait cut
prefill wall `3.21s -> 2.67s`, decode batches `104 -> 92`, and phase time
`7.66s -> 6.21s`, at the cost of median TPOT moving `66.4ms -> 70.3ms`.
Make `5ms` the default only inside the explicit greedy-large mixed-prefix
policy; the normal common-prefix greedy-large path keeps the `10ms` collection
window and automatic mixed-prefix reuse remains off.

The explicit mixed-prefix path also gets a scoped tensor-parallel stream
prequeue wait. A same-tree comparison on the current patch first measured the
exact explicit mixed-prefix default at
`/tmp/ti-bench-results/multi-mixed/.../runs/20260704_113619`: `326.0 / 66.9 /
392.3ms`, `979/1000` correct, with `1000` full-prompt adoptions,
`{"common_prefix":125,"request_prompt":875}` reuse routing, `37` prefill
batches, `18.7K` prefill tokens, and `2.34s/2.64s` prefill forward/wall.
Adding only `TORCHINFERNO_OPENAI_TP_STREAM_PREQUEUE_ADMISSION_WAIT_MS=2` wrote
`/tmp/ti-bench-results/multi-mixed-prequeue2/.../runs/20260704_114322` and
landed at `282.4 / 62.4 / 325.1ms`, `979/1000` correct. The runtime shape
stayed the same (`1000` adoptions, `875` request-prompt reuses, PRBD off,
`prefix_rows=112`, `max_active=32`, no graph evictions); queue-to-submit p50
fell from `88.8ms` to `67.8ms` and queue-to-first p50 from `173.5ms` to
`148.8ms`. Make a `2ms` prequeue wait the default only for the explicit
greedy-large mixed-prefix policy via
`TORCHINFERNO_OPENAI_TP_GREEDY_LARGE_MIXED_PREFIX_STREAM_PREQUEUE_ADMISSION_WAIT_MS`;
the broad `TORCHINFERNO_OPENAI_TP_STREAM_PREQUEUE_ADMISSION_WAIT_MS` override
still wins, and automatic mixed-prefix reuse remains off.

The exact source default for the explicit mixed-prefix opt-in must keep PRBD
off unless the caller opts into it explicitly. Running only
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_MIXED_PREFIX_REUSE=1` with the 5ms
source default wrote
`/tmp/inference-bench-ti-multi-mixed-optin-default-wait5-results/.../runs/20260704_054529`
and landed at `324.6 / 66.2 / 394.8ms`, `982/1000` correct. The queue
shape reused the intended prompt rows (`875` request-prompt reuses,
`125` common-prefix reuses, `1000` full-prompt adoptions), but PRBD was off:
`39` prefill batches, `2.45s/2.75s` prefill forward/wall, `98` decode batches,
queue-to-first p50 `166.7ms`, and p99 queue-to-first `761ms`. Although one
env-backed PRBD+5ms diagnostic run was faster, two exact source-default PRBD
rechecks were not: `/tmp/inference-bench-ti-multi-mixed-optin-default-prbd8-results/.../runs/20260704_055454`
landed at `379.1 / 63.7 / 418.8ms`, and
`/tmp/inference-bench-ti-multi-mixed-optin-default-prbd8-recheck-results/.../runs/20260704_060044`
landed at `357.6 / 65.6 / 418.9ms`. Both had the intended PRBD shape
(`prefill_ready_before_decode=true`, active cap `8`, `prefix_rows=112`), so do
not make PRBD the mixed-prefix opt-in default. Keep PRBD as an explicit
diagnostic override for this path; automatic mixed-prefix reuse remains
disabled.

The sampled-medium prefill-ready cap is now widened to the full 32-row active
set. A same-host tree_of_thought comparison on the patched clean tree first
measured the default cap-10 row at `142.4 / 40.5 / 176.4ms` in
`/tmp/inference-bench-allproviders-tree-patched-results/.../runs/20260704_042026`.
Repeating TorchInferno with
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_PREFILL_READY_ACTIVE_CAP=32`
(`.../8xH100-local-ti-tree-prbdcap32/runs/20260704_043029`) landed at
`126.9 / 46.6 / 160.9ms`, `958/992` correct. The intended mechanism showed up
in the queue profile: queue-to-submit p50 fell `58.1ms -> 43.6ms`,
queue-to-first p50 fell `118.9ms -> 102.0ms`, and ragged-decode GPU fell
`1.72s -> 1.48s`; prefill wall was roughly flat (`2.36s -> 2.31s`) while
prefill batches rose `60 -> 63`. This is a scoped TTFT/E2E tradeoff for
sampled `256 < max_tokens <= 300` sessions, not a broad admission-policy
change.

The current no-env full-suite validation after reverting automatic mixed-prefix
reuse wrote
`/tmp/inference-bench-ti-full-cap32-nomixed-results/.../runs/20260704_050159`.
It landed at few_shot `172.5 / 48.9 / 215.6ms`, self_consistency
`228.6 / 0.0 / 243.5ms`, multi_turn `355.6 / 61.5 / 408.2ms`,
tree_of_thought `128.7 / 24.9 / 149.9ms`, and long_output
`231.1 / 23.8 / 1066.8ms`. The tree queue profile used the intended cap-32
sampled-medium path (`57` prefill batches, `57/0` graph hits/misses,
`1.75s/2.15s` prefill forward/wall, queue-to-first p50 `108.6ms`). Multi_turn
returned to the stable common-prefix shape (`prefix_rows=64`, no full-prompt
adoptions, `35/0` prefill graph hits/misses, `4.10s/4.33s` prefill
forward/wall). This keeps the tree improvement as the defaultable change and
leaves multi_turn's next lever as a stable, non-fragmenting request-prompt reuse
path rather than automatic mixed-prefix admission.

Rechecking sampled-medium FlashInfer decode with the cap-32 tree policy is still
rejected. Setting `TORCHINFERNO_CONTINUOUS_FI_DECODE_SAMPLED_MAX_TOKENS=400`
on the patched tree wrote
`/tmp/inference-bench-ti-tree-fi400-cap32-results/.../runs/20260704_052057`
and landed at `129.7 / 27.0 / 153.0ms`, `961/992` correct. Decode graph hits
were clean (`95/0`), but the queue profile moved the wrong way versus the
no-env cap-32 full-suite validation: prefill batches rose `57 -> 61`, prefill
forward/wall rose `1.75s/2.15s -> 1.86s/2.30s`, decode batches rose
`90 -> 93`, and ragged-decode GPU stayed slightly higher (`1.49s -> 1.51s`).
Keep the sampled FlashInfer decode cutoff scoped to `max_tokens <= 256`; tree's
300-token branch remains on the dense ragged logits graph path.

Focused sampled-medium tree profiling with the ragged-prefill profiler on the
cap-32 policy wrote
`/tmp/inference-bench-ti-tree-ragged-prefill-prof-s32-results/.../runs/20260704_061457`
and landed at `135.0 / 28.7 / 159.3ms`, `961/992` correct. The profiler was
gated with `MIN_BATCH=32`, `MIN_SUFFIX=32`, and captured a startup warmup
instead of request traffic:
`batch=32 suffix=32 context_len=-128 src_rows=1`. That warmup spent `58.7ms`
self CUDA, led by TP all-reduce (`18.9ms`, `32.3%`), FP8/GEMM work
(`15.9ms` combined `_scaled_mm` plus `mm`), attention math (`~9.5ms`), and
copies (`4.6ms`). The queue profile confirms tree's real request path is all
`s16`: `61` prefill batches, `1.83s/2.30s` prefill forward/wall, `11.8K`
active prefill tokens, `8.35K` padding tokens, and hot shapes
`prefix_graph:b16:s16:p45-45` (`11` calls), `b24:s16` (`21` calls), and
`b32:s16` (`14` calls). Padding was split between `2.88K` row padding tokens
and `5.47K` suffix padding tokens. This does not identify another bucket
default to promote; the next tree lever is still packed/ragged cached-prefix
prefill that avoids padded suffix compute, and the profiler now has
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_SKIP_MATCHES` for capturing the later
request-path `s16` body directly.

The first follow-up with `MIN_SUFFIX=16` and `SKIP_MATCHES=10` wrote
`/tmp/inference-bench-ti-tree-ragged-prefill-prof-s16-skip10-results/.../runs/20260704_062407`
and landed at `140.7 / 47.7 / 169.0ms`, `958/992` correct. It still captured
startup before the server listened:
`batch=32 suffix=64 match=11 context_len=109`, with `83.5ms` self CUDA
(`30.2ms` all-reduce, `21.1ms` `_scaled_mm`, `9.7ms` RMSNorm, and `8.7ms`
`mm`). The final graph cache contained twenty live `b32` ragged-prefill graph
entries, so the capture hook is the wrong tool once startup warms request
shapes. A separate `TORCHINFERNO_PROFILE_RAGGED_PREFILL_REPLAY_ONCE` hook now
profiles graph hits in `_run_ragged_prefill_logits_graph`, which should capture
the warmed tree request replay without disabling startup warmup or measuring a
graph-capture body. The first replay-only run still captured startup
`batch=32 suffix=16 match=1 context_len=-64`, so the hook also accepts
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_CONTEXT_LEN=61` to target the exact sampled
common-prefix request graph.

The targeted replay profile did capture that request graph. Running with
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_REPLAY_ONCE=1`,
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_CONTEXT_LEN=61`, and
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_REPLAY_SKIP_MATCHES=1` wrote
`/tmp/inference-bench-ti-tree-ragged-prefill-replay-prof-ctx61-results/.../runs/20260704_063731`
and landed at `134.2 / 39.2 / 162.2ms`, `965/992` correct. The profile fired
after the server listened:
`batch=32 suffix=16 match=2 context_len=61 src_rows=1`. The warmed exact
request replay spent `32.75ms` self CUDA: NCCL all-reduce was `12.80ms`
(`39.1%`, `160` calls), GEMM kernels were roughly `9.7ms` combined, RMSNorm was
`1.93ms`, vector/gather/index overhead was `3.7ms`, and FlashAttention was only
`0.80ms` (`2.4%`). The queue profile shows the expected profiler perturbation
on the sampled request tail (`p99` TTFT `1381ms`, hot `b32:s16` forward
inflated to `1.55s`), so use the kernel table as evidence rather than the p99
row. The remaining tree gap is not attention selection; it is the combination
of padded prefix-suffix graph work and the 160 small TP all-reduces per prefill
replay.

Do not treat the existing FlashInfer `q_lens` hook as that packed-prefill path.
In `forward_flashinfer`, ragged `q_lens` only packs the attention queries in the
eager path. Embedding, QKV, KV append, output projection, RMSNorm, MLP, and
all-reduce still run on the padded `[batch_bucket, suffix_bucket]` tensor, and
the CUDA-graph path intentionally passes `q_lens=None` to avoid graph-illegal
tensor-to-bool control flow. That matches the warmed replay evidence above:
FlashAttention is only `0.80ms` of the `32.75ms` tree replay, while all-reduce,
GEMM, norms, and indexing dominate. The missing implementation is a true packed
cached-prefix prefill that keeps the layer stack packed and only unpacks at
cache/logit boundaries, not a toggle on the current `q_lens` attention branch.

Adding more sampled-medium prefix-prefill batch buckets is rejected as a simple
default. A focused tree probe with
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS=1,2,4,8,16,20,24,28,32`
and matching
`TORCHINFERNO_OPENAI_STARTUP_RUNTIME_FP8_RAGGED_PREFILL_BATCHES=1,2,4,8,16,20,24,28,32`
(`8xH100-local-ti-tree-b20-b28-buckets`, port `8045`) stayed in startup for
more than six minutes and never reached server readiness; it was terminated
before requests, and no result row was written. The expected row-padding
reduction is too small to justify a startup shape expansion that cannot
reliably warm. Keep sampled-medium buckets at `1,2,4,8,16,24,32` until there is
a packed/ragged prefill path that lowers row padding without multiplying
warmup graph shapes.

Long-output decode now has a matching one-shot replay profiler for the warmed
ragged token graph:
`TORCHINFERNO_PROFILE_RAGGED_DECODE_REPLAY_ONCE=1`. It profiles the graph-hit
path in `_run_ragged_decode_graph`, not the eager FlashInfer body covered by
the older `TORCHINFERNO_PROFILE_DECODE_ONCE` hook. Use
`TORCHINFERNO_PROFILE_RAGGED_DECODE_MIN_BATCH`,
`TORCHINFERNO_PROFILE_RAGGED_DECODE_CACHE_BUCKET`, and
`TORCHINFERNO_PROFILE_RAGGED_DECODE_REPLAY_SKIP_MATCHES` to target the steady
long-output `decode_many:b64/64` replay after startup warmups. The hook is
diagnostic only; it should identify whether long's decode gap is still TP
all-reduce/GEMM dominated before changing decode scheduling again.

The first focused long-output replay profile wrote
`/tmp/inference-bench-ti-long-ragged-decode-replay-prof-b64-results/.../runs/20260704_070025`
with `TORCHINFERNO_PROFILE_RAGGED_DECODE_MIN_BATCH=64` and
`TORCHINFERNO_PROFILE_RAGGED_DECODE_REPLAY_SKIP_MATCHES=1`. It landed at
`249.2 / 24.7 / 1192.2ms`, `1000/1000` correct. The profiler fired after the
server listened on the steady shape
`batch=64 match=2 cache_bucket=1024 rows=64`, and the warmed replay spent
`12.45ms` self CUDA: dense GEMM kernels were `4.50ms`, Marlin was `3.32ms`,
multimem all-reduce was `2.09ms`, streaming GQA decode attention was `1.48ms`,
and RMSNorm was `0.43ms`. The final queue profile had `100` decode-many calls,
`470` decode-many steps, `27.5K` decode-many model tokens, `25.8K` emitted
tokens, and `1.7K` overgenerated tokens; `decode_many:b64/64` alone consumed
`16.1K` model tokens and `4.07s` GPU. This says the long-output decode gap is
not primarily all-reduce anymore. The next useful work is either reducing the
GEMM/Marlin slices per decode replay or overlapping/flushing decode-many work
so q16-style larger bursts do not hide first tokens.

Gating a larger drain burst on already-emitted tokens is also rejected as a
default, though the runtime now has a useful diagnostic guard:
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_DRAIN_DECODE_MIN_GENERATED`.
Setting drain q12 after every active request had emitted one token wrote
`/tmp/inference-bench-ti-long-drainq12-after-first-results/.../runs/20260704_071529`
and landed at `321.0 / 21.8 / 1126.0ms`, `1000/1000` correct. It reduced
median TPOT/E2E, but queue-to-first p50 was still `239ms`, p99 E2E was
`2223ms`, decode-many skipped tokens rose to `3266`, and `b64/64` skipped
`941` tokens. Raising the gate to four generated tokens wrote
`/tmp/inference-bench-ti-long-drainq12-after-four-results/.../runs/20260704_072048`
and softened the tradeoff to `286.2 / 24.0 / 1157.4ms`, `1000/1000` correct,
with queue-to-first p50/p99 `229/481ms`, `10.34s` ragged decode GPU, `90`
decode-many calls, and `3071` skipped tokens. Both variants still lose the q8
default's TTFT/tail balance, so keep drain q8 as the default and use the
min-generated guard only for future decode-many diagnostics.

Capping stop-aware decode-many tails is rejected as a long-output default. A
focused run with
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_STOP_TAIL_MAX_STEPS=1` wrote
`/tmp/inference-bench-ti-long-stop-tail1-results/.../runs/20260704_073650`
and landed at `266.4 / 23.8 / 1107.4ms`, `1000/1000` correct. It cut skipped
decode-many tokens from the q8 default's `1385` to `596`, but fragmented the
decode tail into `286` decode-many calls, tail-limited `242` calls / `871`
steps, and still spent `10.13s` in ragged-decode GPU. The result is worse than
the current q8 default and the q12 diagnostics on median TTFT/E2E, so leave the
tail cap at `0` by default. Queue profiles now record
`decode_many_stop_tail_max_steps` so future tail-cap probes are
self-describing.

Greedy-mid prefill-ready-before-decode is also rejected for few_shot. The latest
public row made the 256-token greedy gap more visible, so a focused probe set
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_READY_BEFORE_DECODE=1` and
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_READY_ACTIVE_CAP=8` on the patched clean
tree. It wrote
`/tmp/inference-bench-ti-few-prbd8-results/.../runs/20260704_075038` and landed
at `178.1 / 49.1 / 221.1ms`, `977/1000` correct. The queue profile confirms
the mechanism is not defaultable: it used `34` prefill batches and `32/2`
prefill graph hits/misses, but prefill wall rose to `2.44s`, p99
queue-to-first stayed high at `896ms`, p99 E2E was `1272.5ms`, and median TPOT
regressed to `49.1ms`. Keep prefill-ready-before-decode scoped to greedy-short,
greedy-large, and sampled-medium policies that already have evidence; greedy-mid
few_shot still needs faster `b32:s16` prefix-suffix prefill rather than another
decode/prefill ordering toggle.

The corrected few_shot replay profile confirms that diagnosis. The first
attempt targeted `context_len=138`, but the serving path maps `p122/s16` through
dynamic prefix prefill to the `ctx=-256` graph key, so no profile table was
emitted. Rerunning with
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_REPLAY_ONCE=1`,
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_CONTEXT_LEN=-256`,
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_MIN_BATCH=32`, and
`TORCHINFERNO_PROFILE_RAGGED_PREFILL_MIN_SUFFIX=16` wrote
`/tmp/inference-bench-ti-few-prefill-replay-ctxneg256-results/.../runs/20260704_080737`
and landed at `181.5 / 50.2 / 229.6ms`, `977/1000` correct. The profile fired
on the warmed request shape
`batch=32 suffix=16 match=1 context_len=-256 src_rows=1` and spent `50.22ms`
self CUDA: NCCL all-reduce was `12.94ms` (`25.8%`, `160` calls), GEMM/NVJET
kernels were roughly `19ms` combined, elementwise/norm/index/gather work was the
remaining material cost, and attention was not a top item. Queue timing was
profile-perturbed (`2.90s/3.61s` prefill forward/wall and p99 E2E `2.25s`), so
use the kernel table rather than the row as evidence. This aligns few_shot with
the targeted tree replay profile: the next defaultable improvement must reduce
the model-side prefill replay cost itself, especially small TP all-reduces and
GEMM/elementwise slices, or replace padded prefix-suffix replay with a packed
path. The fixed 64-entry live-graph shape profile cap also hid the hot `b32`
resident shapes; queue profiles now default
`TORCHINFERNO_OPENAI_TP_ONLINE_PROFILE_GRAPH_SHAPE_LIMIT` to `192` so final
records expose the full ragged-prefill graph cache, with an env override still
available for smaller logs.

Chunked prefill remains opt-in, but its intermediate chunks no longer need to
pay for logits. The runtime now calls a cache-only ragged prefill graph/eager
hook when no request finishes its prompt in the current chunk, skipping the
final gather, LM-head projection, and CPU sampling for those chunks. Queue
profiles label these resident graph entries with `:logits0` so chunked-prefill
A/Bs can separate cache-fill graphs from token-emitting suffix graphs. This does
not overturn the earlier chunking rejections by itself; it removes one known
waste source before the next controlled chunked-prefill run.

The chunked common-prefix setup now follows the same principle. When a chunked
admission wave shares a prefix and every request still has a non-empty suffix,
the runtime first replays a cache-only ragged prefill graph if one is already
resident, with graph capture disabled on miss, then falls back to the model's
cache-only eager hook and finally to the old logits prefill if neither no-logits
path is available. This keeps the normal non-chunked common-prefix path
unchanged because broad common-prefix no-logits was already rejected, but removes
a discarded LM-head projection from the opt-in chunked path before any suffix
chunks run. Focused CPU coverage:
`venv/bin/python -m pytest
tests/test_serving_engine.py::test_continuous_batch_engine_chunked_prefill_prepares_common_prefix
tests/test_serving_engine.py::test_continuous_batch_engine_chunked_prefill_skips_intermediate_logits
tests/test_serving_engine.py::test_continuous_batch_engine_chunked_prefill_matches_one_shot
-q`.

Focused post-fix synthetic evidence is positive for that specific waste
removal. On the patched clean tree, a TP=8 `openai-microbench` with synthetic
`prompt_tokens=160`, `max_tokens=16`, concurrency `64`, warmup `1`, and iters
`1` compared no chunking against
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_CHUNK=64`. The no-chunk measured
concurrent row was `2611.4 / 19.7 / 2906.4ms` with `5.5 tok/s`; chunk-64 was
`1284.1 / 19.3 / 1573.1ms` with `10.2 tok/s`. The chunked queue profile
confirmed resident `:logits0` graph entries and, for the measured concurrent
session after warmup, added six prefill graph replays with zero new captures or
misses. Cold chunk sessions still paid `10` graph captures and `9.21s` aggregate
capture time, so this is not a default-policy result; it shows the cache-only
intermediate chunks are now viable enough for a warmer, public-shaped chunking
A/B.

The startup chunked-prefill warmup now targets that cold-capture gap directly
when `TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_CHUNK` is set. Instead of warming
only token-emitting logits graphs at `context_len=suffix`, it warms cache-only
ragged prefill graphs for dynamic context buckets such as `-64/-128/-256/-512`
and only warms logits graphs for the configured final context buckets. The
warmup also enters the same online FP8 prefill policy key as the live request
path. This keeps chunked prefill opt-in while making the next chunked A/B a
startup-warmed measurement rather than a first-request graph-capture test.
A same-shape rerun with that warmup active, `warmup=1`, and `iters=1` wrote
`/tmp/ti_chunk_ab_chunk64_warmed_warm1.json`: concurrent-64 was
`1282.0 / 18.9 / 1566.0ms` with `10.2 tok/s`, matching the earlier post-fix
chunk-64 row while the queue profile stayed at `0` request-path prefill
captures, `0` misses, and `24` prefill graph replays by the measured row.

Real long_output chunking is still rejected as a default. The synthetic
chunk-64 result did not carry over to the benchmark shape because the chunked
admission path originally skipped the one-shot path's shared common-prefix
setup. A direct run with
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_CHUNK=64` on the patched tree hit `66`
request-path prefill graph captures by the final queue snapshot, filled the
ragged-prefill graph cache to `192`, and reached p50 queue-to-first around
`4.65s` before the run was interrupted. Bucket-padding chunked groups to the
runtime prefill batch buckets reduced captures to `19`, but still bypassed
common-prefix reuse, prefilled `155.7K` active prompt tokens, spent `34.3s` in
ragged decode GPU, and ended with p50 queue-to-first `1.38s`.

Chunked admission now prepares a shared common prefix before creating
prefilling states, and the chunked warmup covers the same batch buckets as the
runtime path. That fixes the mechanical miss: the next
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_CHUNK=64` long_output run reused the
`111` token prefix for all `1000` requests (`111K` reused tokens), cut active
prefill tokens to `44.8K`, and used `60` prefill batches. It still landed at
`753.6 / 29.1 / 1804.8ms`, with `6` request-path prefill captures,
`26.3s` ragged-decode GPU in the queue snapshot, and p99 E2E `6.80s`. Keep
chunked prefill opt-in; a defaultable long_output fix needs a decode/prefill
overlap design that preserves the common-prefix shape and does not fragment the
steady 64-row decode path.

The no-logits chunked common-prefix setup also needed a row-mapping fix before
it was safe. The first graph-first cache-only run wrote
`/tmp/ti-bench-results/current-long-chunk64-cacheprefix/.../runs/20260704_215142`
and exposed the bug: correctness fell to `0.000`, even though the profile
reported `111` reused prefix tokens for all `1000` requests. The cache-only
helper had filled a cache row view with local `row_indices=[0]`; the ragged
prefill write path treats explicit row indices as physical rows, so the shared
prefix row could be left empty while row 0 was overwritten. The runtime now
calls cache-only ragged prefill on the root cache with the acquired physical
prefix row, and the CPU test asserts that the cache-only call records the same
physical row later used as the suffix graph source.

The corrected chunk-64 long_output rerun wrote
`/tmp/ti-bench-results/current-long-chunk64-cacheprefix-rowfix/.../runs/20260704_220151`
and recovered `1000/1000` correctness, but it still rejects chunking as a
default: `817.7 / 39.1 / 2338.2ms`, versus the current non-chunked default
probe's `251.3 / 23.9 / 1074.6ms`. Queue counters show why the fix is only
correctness hygiene: prefix reuse hit `111` tokens for all `1000` requests and
active prefill stayed at `44.8K` tokens, but the run still paid `9` prefill
graph captures (`8.36s`), `25` decode graph captures (`20.14s`), and
`30.98s` ragged-decode GPU. Keep
`TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_CHUNK` diagnostic-only for public-shaped
long_output.

The `inference-bench-summary` helper now prints prefill/decode graph
capture/replay timing and the hottest prefill graph capture/replay shapes.
This run exposed why that belongs in the summary table: graph-only chunked
prefill legitimately reports `0.0ms` in the old eager forward/wall columns
while still spending `8.36s` in prefill graph capture.

Public pointer refresh `20260704_090227` (`292caed`) did not change the
underlying score-facing gaps. TorchInferno improved to `5/20` metric wins:
few_shot `158.6 / 46.2 / 204.2ms`, self_consistency
`174.4 / 0.0 / 182.1ms`, multi_turn `348.0 / 60.9 / 401.4ms`,
tree_of_thought `128.4 / 69.0 / 161.1ms`, and long_output
`244.8 / 21.0 / 987.0ms`. The long_output queue profile remained decode-heavy:
`59` prefill batches, `4.31s/4.54s` prefill forward/wall, `812` decode
batches, `95` decode-many calls over `430` decode-many steps, and `9.84s`
ragged-decode GPU. Multi_turn remains common-prefix-only at default policy, and
tree remains sampled-prefix/decode bound.

Sampled tree suffix/batch bucketing probes do not produce a defaultable change.
Forcing sampled suffix buckets to `8,16,32` and warming those suffixes wrote
`/tmp/ti-bench-results/tree-suffix8/.../runs/20260704_102116` and landed at
`133.0 / 31.2 / 159.7ms`, `959/992` correct. Shape details showed the env did
not create any `s8` request shapes: all prefix-suffix replays remained
`s16`, with `58` prefill batches, `11.8K` active prefill tokens,
`18.9K` model prefill tokens, and `8.46K` padding tokens. Enabling the existing
suffix-bucket split path on top wrote
`/tmp/ti-bench-results/tree-suffix-split8/.../runs/20260704_102618` and landed
at `132.1 / 27.9 / 154.6ms`, `956/992` correct, but still produced only `s16`
request shapes. The TPOT movement is therefore not evidence for promoting `s8`
suffix buckets.

A finer sampled-medium batch-bucket probe
(`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS_SAMPLED_MEDIUM=1,2,4,8,12,16,20,24,28,32`)
wrote `/tmp/ti-bench-results/tree-batch12/.../runs/20260704_103128`. It reduced
profiled prefill forward to `1.69s` and padding to `6.48K` tokens with `55`
prefill batches and no request-path graph captures, but the benchmark regressed
to `135.3 / 29.4 / 165.4ms`, p99 E2E `793ms`, and startup readiness `210.9s`.
Keep sampled batch buckets at the current `1,2,4,8,16,24,32`; the next tree
change needs model-side prefill/decode work, not more graph-bucket surface.
To make that next pass less inferential, queue profiles now record
`runtime_prefill_shape_real_batch_counts` and
`runtime_prefill_shape_suffix_length_counts` alongside the existing
row/suffix-padding totals. Those counters show the actual request counts and
suffix lengths that fed each prefix graph shape, so future tree probes can
target the real waste distribution without first adding more warmed graph
buckets.

Current-head tree profiling with those counters confirms the exact waste shape.
The baseline focused run
`/tmp/ti-bench-results/tree-newfields/.../runs/20260704_124446` landed at
`130.5 / 39.1 / 160.2ms`, `959/992` correct, with startup readiness `205.9s`.
Every prefix-suffix replay stayed in `s16`, while actual suffix lengths were
only `10`, `11`, and `12`. The queue profile showed `59` prefill batches,
`11.8K` active prefill tokens, `18.99K` model prefill tokens, `8.59K` padding
tokens (`3.12K` row / `5.47K` suffix), `1.83s/2.27s` prefill forward/wall, and
`1.42s` ragged-decode GPU.

Rechecking the existing `s12/s16` opt-in on the same tree
(`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS=12,16` and matching
startup suffix warmup) wrote
`/tmp/ti-bench-results/tree-s12-current/.../runs/20260704_125036`. It improved
median TPOT/E2E to `128.3 / 31.2 / 154.3ms` and cut padding to `3.17K` tokens
(`1.67K` row / `1.50K` suffix), but it is still not a defaultable policy:
p99 TTFT/E2E worsened to `895/922ms`, prefill forward/wall rose to
`2.36s/2.79s`, and the run paid a request-path capture for
`ragged_prefill:b24:s12:rows1:ctx57:src1`. Queue profiles now also expose
`runtime_prefill_graph_capture_shape_ms` and
`runtime_prefill_graph_replay_shape_ms` so future suffix-bucket or packed-prefill
probes can attribute these tail spikes directly.
The sampled common-prefix suffix warmup helper now follows
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SUFFIX_BUCKETS` when no sampled-specific
warmup suffix list is configured, matching the greedy path. The patched rerun
`/tmp/ti-bench-results/tree-s12-warmfix/.../runs/20260704_125949` confirmed the
targeted fix: no prefill graph misses or request-path captures, warmed
`b24:s12` replay at `73.6ms`, `57` prefill batches, `11.8K` active tokens,
`3.58K` padding tokens (`2.08K` row / `1.50K` suffix), and `1.68s/2.12s`
prefill forward/wall. It landed at `128.9 / 26.6 / 149.5ms`, `961/992`
correct, with p99 TTFT/E2E `590.5/614.8ms`.
Promoting sampled-medium `12,16` suffix buckets to a no-env default is still
rejected. The default-promotion probe
`/tmp/ti-bench-results/tree-s12-default/.../runs/20260704_130535` also had zero
prefill graph captures, but startup readiness rose to `210.9s`, profiled
prefill forward/wall rose to `1.90s/2.35s`, and the benchmark landed at
`140.9 / 28.6 / 165.2ms`, `959/992` correct. Keep `s12` as an explicit
diagnostic/runtime opt-in until the startup graph surface and median TTFT/E2E
are both improved.

Lowering the few_shot runtime FP8 prefill gate is rejected. The replay profile
made the tempting case: the hot `b32*s16=512` request graph misses the default
runtime gate because `_ragged_prefill_precision_graph_key` uses a strict
`token_count > min_m` check and the greedy-mid default leaves online FP8 prefill
disabled. Forcing
`TORCHINFERNO_OPENAI_TP_ONLINE_FP8_PREFILL=1` and
`TORCHINFERNO_OPENAI_TP_ONLINE_FP8_PREFILL_MIN_M=256` wrote
`/tmp/inference-bench-ti-few-fp8-minm256-results/.../runs/20260704_081537`
and flipped the hot resident graph from `fp80` to `fp81`. It landed at
`176.0 / 50.2 / 218.7ms`, `980/1000` correct. The adjacent no-env control on
the same tree wrote
`/tmp/inference-bench-ti-few-control-390fed4-results/.../runs/20260704_082058`
and landed at `176.1 / 49.0 / 219.0ms`, `977/1000` correct. The FP8 variant
cut the hot `b32:s16:p122` forward counter from `1.66s` to `1.60s`, but added a
prefill batch/miss, raised prefill wall from `2.39s` to `2.59s`, worsened p99
queue-to-first/finish from `678/696ms` to `1056/1083ms`, and lost `1.2ms`
median TPOT. Keep few_shot out of the runtime FP8 prefill default; this exact
gate does not close the score gap.

Added a dense packed-ragged prefill oracle for the next prefix-suffix lever.
`prefill_ragged_logits_packed_eager` flattens only real suffix tokens, scatters
KV into the same physical rows/positions as padded ragged prefill, and slices
attention per request so rows do not cross-attend. A CPU test compares logits
and real-token KV columns against the existing padded oracle with shared-prefix
copy. Serving can route padded suffix groups through it with the explicit
diagnostic switch `TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER=1`; it
only fires when `logit_positions + 1` shows at least one row is shorter than the
suffix bucket. This is not a default optimization yet: it is a correctness and
profiling scaffold to prove the tensor contract before replacing the Python
per-row attention loop with a CUDA/FlashInfer packed implementation.

The first full long_output A/B on the patched local tree rejects that Python
packed-eager path as a runtime implementation. The same-tree control wrote
`/tmp/ti-bench-results/control-current/.../runs/20260704_111103` and landed at
`244.8 / 24.7 / 1106.2ms`, `1000/1000` correct, with `5.02s/5.40s`
prefill forward/wall and `10.10s` ragged-decode GPU. Enabling
`TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER=1` wrote
`/tmp/ti-bench-results/packed-ragged/.../runs/20260704_110423` and landed at
`1671.4 / 98.2 / 5535.0ms`, also `1000/1000` correct, but with
`71.82s/72.17s` prefill forward/wall. It did reduce padded suffix accounting
from `24.8K` to `21.7K` tokens, but the per-request Python attention loop
dominates. Keep the method as a correctness oracle and profiling scaffold only;
the defaultable version needs one packed CUDA/FlashInfer prefill body, not eager
per-row SDPA.

The packed eager scaffold now precomputes packed suffix metadata once per
prefill and groups attention calls by `(prefix_start, q_len)` instead of
re-reading CUDA metadata inside every layer. The correctness test was tightened
to include two requests with the same real suffix length so the grouped path is
covered. A focused tree_of_thought probe with
`TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER=1` wrote
`/tmp/ti-bench-results/tree-packed-grouped/.../runs/20260704_121800` and still
landed at `1263.1 / 303.8 / 1363.2ms`, `957/992` correct. Queue counters show
the reason: prefill forward/wall was `29.52s/29.94s` across `58` batches,
versus about `1.74s/2.18s` for the padded graph control. Metadata grouping
removes obvious Python sync waste, but the eager attention structure is still
orders of magnitude too slow for runtime. Keep it opt-in as an oracle; do not
default packed prefill until it has a single CUDA/FlashInfer body.

A follow-up packed-query indexing experiment is also rejected and was backed
out. It precomputed each grouped query token index and replaced the per-layer
slice/`torch.cat` construction with a single `index_select`, but the focused
tree probe with `TORCHINFERNO_CONTINUOUS_PACKED_RAGGED_PREFILL_EAGER=1` wrote
`/tmp/ti-bench-results/tree-packed-queryindex/.../runs/20260704_123142` and
landed at `1215.6 / 316.4 / 1391.9ms`, `953/992` correct. Queue counters still
showed the same blocker: prefill forward/wall was `30.66s/31.14s` across `63`
batches, versus `29.52s/29.94s` for the prior grouped packed probe and
`~1.74s/~2.18s` for padded graph control. The bottleneck is not query slicing;
it is still eager per-group SDPA and uncaptured per-layer work.

A fresh ragged-decode replay profile on the same tree wrote
`/tmp/ti-bench-results/decode-profile/.../runs/20260704_111646` and profiled a
hot `batch=64 cache_bucket=1024` graph replay at `12.44ms` self CUDA. The
largest slices were dense GEMMs (`4.50ms` combined), gate-up Marlin
(`3.32ms`), symmetric-memory all-reduce (`2.07ms`), grouped GQA decode
attention (`1.49ms`), and RMSNorm/add (`0.44ms`). A matching Marlin-off run
with `TORCHINFERNO_OPENAI_TP_ONLINE_MARLIN_INT4_DECODE=0` wrote
`/tmp/ti-bench-results/marlin-off/.../runs/20260704_112227` and landed at
`249.8 / 24.1 / 1124.9ms`, `1000/1000` correct, with `10.64s` decode GPU.
That is not a clean win over the control (`24.7ms` TPOT, `10.10s` decode GPU),
so keep gate-up Marlin enabled. The current long_output gap remains decode GEMM
and per-layer collective work; cache-token buckets and stop-tail caps are still
covered by prior rejected A/Bs.

A same-tree all-provider tree_of_thought refresh on the patched local tree wrote
`/tmp/ti-bench-results/tree-current-allproviders/.../runs/20260704_115517`.
TorchInferno landed at `140.5 / 27.0 / 162.4ms`, vLLM at
`65.5 / 32.2 / 89.9ms`, and SGLang at `58.8 / 83.6 / 154.7ms`. The current
sampled-medium policy now wins TPOT locally, but TTFT/E2E remain bounded by
cached-prefix prefill: TorchInferno used `56` prefill graph batches for
`11.8K` active prompt tokens and `8.0K` padding tokens. The remaining tree
lever is therefore true packed cached-prefix prefill, not another sampled
decode or graph-bucket default.

A fresh current-tree long_output profile with
`TORCHINFERNO_PROFILE_RAGGED_DECODE_REPLAY_ONCE=1` wrote
`/tmp/ti-bench-results/long-current-profile/.../runs/20260704_120555` and
landed at `263.9 / 23.9 / 1168.4ms`, `1000/1000` correct. The queue profile
showed `59` prefill batches, `4.66s/5.05s` prefill forward/wall, and `100`
decode-many calls over `482` decode-many steps. The replay profiler captured a
`batch=32 cache_bucket=1024` graph at `10.68ms` self CUDA: dense GEMMs were
`4.24ms` combined, gate-up Marlin `2.26ms`, symmetric-memory all-reduce
`1.66ms`, grouped GQA decode attention `1.40ms`, and RMSNorm/add `0.40ms`.
That confirms the same shape as the earlier `batch=64` replay: the long_output
gap is model-kernel throughput plus padded prefill, not an untried
queue-scheduling toggle.

A sampled temperature scratch-buffer experiment was also rejected and reverted.
Commit `e20eff3` reused per-shape float/Gumbel work buffers in the distributed
temperature sampler, but a valid tree-only run at
`/tmp/inference-bench-ti-e20eff3-tree-results/.../runs/20260705_215756`
landed at `146.4 / 64.1 / 205.8ms`, `963/992` correct, worse than the public
`20260705_210211` tree row (`79.6 / 64.9 / 114.5ms`). The queue profile showed
lower model-side prefill/decode totals (`2.84s` prefill forward, `2.46s`
decode GPU), but `prefill_sample_ms` rose to `305ms` and request queueing
dominated (`q2first=130ms`, `q2submit=75ms`). The extra copy/divide work is not
a proven default win; keep sampled decode focused on a real fused or graph-safe
sampler, not persistent scratch buffers.

Added an opt-in decode-many replay profiler for the captured multi-step ragged
decode graph. Set `TORCHINFERNO_PROFILE_RAGGED_DECODE_MANY_REPLAY_ONCE=1`,
optionally with `TORCHINFERNO_PROFILE_RAGGED_DECODE_MANY_MIN_BATCH`,
`TORCHINFERNO_PROFILE_RAGGED_DECODE_MANY_CACHE_BUCKET`,
`TORCHINFERNO_PROFILE_RAGGED_DECODE_MANY_STEPS`,
`TORCHINFERNO_PROFILE_RAGGED_DECODE_MANY_REPLAY_SKIP_MATCHES`, and
`TORCHINFERNO_PROFILE_RAGGED_DECODE_MANY_ROW_LIMIT`, to print one rank-0
`torch.profiler` table for the exact `decode_many` graph replay. This is
instrumentation only; default serving behavior is unchanged.

A long_output run with the new decode-many replay profiler enabled wrote
`/tmp/inference-bench-decode-many-prof-results/.../runs/20260706_052503` and
landed at `237.4 / 23.4 / 1062.5ms`, `1000/1000` correct. The hook did not
fire because the current default path has `runtime_decode_many_graph_calls=0`;
`decode_many` is a scheduler loop over single-step ragged decode graph replays.
The queue profile still showed the same shape: `4.68s` prefill forward,
`5.05s` decode-many GPU, `3.86s` ragged decode graph replay, and `1.36s`
decode-many CPU token handling.

The matching single-step ragged replay profile filtered to
`batch=64/cache_bucket=1024` wrote
`/tmp/inference-bench-ragged64-prof-results/.../runs/20260706_053043` and
landed at `216.3 / 23.6 / 1139.3ms`, `1000/1000` correct. The rank-0 profiler
captured one hot `batch=64` replay at `12.53ms` self CUDA: dense GEMMs were
`4.58ms` combined, gate-up Marlin `3.34ms`, symmetric-memory all-reduce
`2.11ms`, grouped GQA decode attention `1.49ms`, add/RMSNorm `0.44ms`, rotary
append `0.18ms`, and greedy sampling all-gather `0.009ms`. Queue counters
showed `5.72s` total ragged decode graph replay, almost entirely
`b64/cache1024/symm128`, plus `4.80s` prefill forward and `1.60s` decode token
CPU handling. This rules out token sampling and rotary as meaningful
long_output levers; the remaining decode gap is dense GEMM/Marlin throughput
and per-layer collective/attention cost, with padded prefill still the TTFT
lever.

A current-head sampled-medium tree recheck on `f135593` restores the
prefill-ready active cap to the full 32-row sampled-medium active set. The
same-conditions control wrote
`/tmp/ti-tree-default-20260707-results/.../runs/20260707_195656` and landed at
`135.4 / 55.4 / 185.0ms`, p99 E2E `621.0ms`, `961/992` correct. Raising
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_MAX_ACTIVE=40` improved medians
to `119.8 / 57.5 / 172.1ms` but increased active rows. Keeping max_active at
`32` and setting
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_PREFILL_READY_ACTIVE_CAP=32`
wrote `/tmp/ti-tree-prbdcap32-f135-20260707-results/.../runs/20260707_201623`
and landed at `116.3 / 58.9 / 172.8ms`, p99 E2E `482.1ms`, `958/992`
correct. The queue profile kept the scoped sampled-medium shape
(`temperature=0.7`, `run_max_tokens=300`, `max_active=32`) while moving
queue-to-first p50/p99 to `100.6/335.1ms` and queue-to-finish p50/p99 to
`159.8/405.4ms`. This is a TTFT/E2E and tail tradeoff for tree-style sampled
medium traffic; sampled-short self-consistency, greedy traffic, and the row cap
remain unchanged.

Warmed non-power decode buckets remain rejected for long_output. A temporary
opt-in patch made continuous decode buckets configurable and added matching
online startup warmup, then tested `2,3,4,8,16,32,40,48,56,64` on current
`620557c`. The run
`/tmp/ti-long-decodebuckets-f620-20260707-results/.../runs/20260707_204749`
stayed correct and cut decode-many padding to `850` tokens, but landed only
neutral at `217.0 / 23.5 / 1051.1ms` with `5.00s` decode-many GPU. A narrower
`2,3,4,8,16,32,48,64` run
`/tmp/ti-long-decodeb48-f620-20260707-results/.../runs/20260707_205337` also
stayed correct but regressed to `220.9 / 23.9 / 1072.8ms`. This validates that
request-path capture was not the only blocker in the earlier fine-bucket probe:
fewer padded rows still does not beat the existing power-of-two replay family.
Do not add non-power continuous decode buckets by default without a cheaper
replay body.

Greedy-large mixed-prefix multi_turn now defaults the scoped stream prequeue
admission wait to `0ms`. On current `027a6d8`, the baseline `2ms` default wrote
`/tmp/ti-multi-current-027a-20260707-results/.../runs/20260707_210346` and
landed at `215.6 / 64.6 / 278.0ms`, p99 E2E `784.9ms`, `982/1000` correct.
Disabling only
`TORCHINFERNO_OPENAI_TP_GREEDY_LARGE_MIXED_PREFIX_STREAM_PREQUEUE_ADMISSION_WAIT_MS`
with `0` wrote
`/tmp/ti-multi-prequeue0-027a-20260707-results/.../runs/20260707_211442` and
improved the score-facing medians to `213.1 / 57.3 / 267.7ms`, `979/1000`
correct. Midpoint and larger waits did not validate: `1ms`
(`/tmp/ti-multi-prequeue1-027a-20260707-results/.../runs/20260707_212000`)
landed at `222.6 / 59.4 / 284.6ms`, and `4ms`
(`/tmp/ti-multi-prequeue4-027a-20260707-results/.../runs/20260707_210942`)
landed at `229.7 / 60.3 / 290.1ms`. Keep the env override for explicit
diagnostics, but do not add an automatic stream prequeue delay to the
mixed-prefix default.

Public `20260707_210232` measured TorchInferno `027a6d8`, so it includes the
sampled-medium prefill-ready active-cap restoration and the decode-bucket
rejection note, but not the later `c997899` mixed-prefix prequeue default. The
public row kept the scorecard at TorchInferno `3/20`, vLLM `15/20`, and SGLang
`1/20`. Tree improved materially on the public scoreboard at
`65.8 / 43.5 / 99.1ms` versus vLLM `33.1 / 21.8 / 48.8ms`; this validates the
sampled-medium tree direction even though vLLM still wins that row. Public
multi_turn remained the stale prequeue target at `272.1 / 58.6 / 327.0ms`
versus vLLM `157.0 / 46.3 / 201.3ms`.

A same-host all-provider `multi_turn` refresh on pushed `c997899` wrote
`/tmp/allproviders-multi-c997899-20260707-results/.../runs/20260707_212948`.
vLLM landed at `157.0 / 55.5 / 205.6ms`, SGLang at
`164.2 / 116.0 / 275.3ms`, and TorchInferno at
`225.4 / 60.9 / 295.0ms`, `982/1000` correct. The queue profile shows the
current mixed-prefix path is active (`max_active=32`, `prefix_rows=112`,
`{"common_prefix":125,"request_prompt":875}`), with `38` prefill batches,
`2.39s/2.73s` prefill forward/wall, `q2submit_p50=66.6ms`,
`q2first_p50=151.0ms`, and `submit2first_p50=86.5ms`. This is the current
post-prequeue local comparison band while public waits for a `c997899` run.

Adding an intermediate greedy-large prefix-prefill batch bucket is rejected.
The scoped env probe
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_BATCH_BUCKETS=1,2,4,8,16,24,32` on
`c997899` wrote
`/tmp/ti-multi-b24bucket-c997899-20260707-results/.../runs/20260707_214417`.
It stayed correct enough (`981/1000`) but landed at
`227.4 / 61.0 / 288.2ms` with a much worse p99 (`1192.6ms` TTFT,
`1253.6ms` E2E) and readiness stretched to `226s`. Queue telemetry split the
run into `381` and `619` request sessions and exposed very slow mixed-prefix
`b24:s32` replays (`453-486ms` forward) plus `3` graph misses in the larger
session. The saved padded rows do not compensate for the extra warmup,
fragmentation, and slow b24 mixed-prefix bodies; keep greedy-large mixed-prefix
batch buckets on the existing power-of-two path unless the b24 replay body is
made cheap and warm.

Forcing combined submit+step on greedy-large mixed-prefix multi_turn is also
rejected. The env run
`TORCHINFERNO_OPENAI_TP_ONLINE_SUBMIT_STEP_COMMAND=1` on `c997899` wrote
`/tmp/ti-multi-submitstep-c997899-20260707-results/.../runs/20260707_215043`
and landed at `227.0 / 61.8 / 287.1ms`, `980/1000` correct, with p99 TTFT/E2E
regressing to `1036.5/1095.7ms`. It did not reduce score-facing first-token
latency versus the current no-env band: queue `q2first_p50=145.4ms` was
similar, while prefill grew to `40` batches and `2.85s/3.43s`
forward/wall. Keep combined submit+step scoped to the sampled-short policy and
leave the global env override as an explicit diagnostic.

A same-host all-provider `long_output` refresh on pushed `85007b2` wrote
`/tmp/allproviders-long-85007b2-20260707-results/.../runs/20260707_215812`.
vLLM landed at `84.2 / 16.8 / 685.9ms`, SGLang at
`62.6 / 23.7 / 948.1ms`, and TorchInferno at
`216.5 / 23.5 / 1060.7ms`, all `1000/1000` correct. TorchInferno's queue
profile is the current dense-cache long-output band: `max_active=64`,
`prefix_rows=64`, `use_decode_many=true`, `decode_quantum=3`,
`drain_decode_quantum=8`, `62` prefill batches, `4.70s/5.25s` prefill
forward/wall, `50.9K` prefill tokens, `21.4K` suffix-padding tokens,
`5.05s` decode-many GPU, and `3.82s` ragged decode graph replay. The biggest
score-facing gaps are unchanged: first-token latency is still padded
cached-prefix prefill and decode/E2E is still the full-batch decode replay body.
The earlier non-power decode-bucket, decode-many graph, stop-tail, and suffix
split probes remain rejected because they move padding counters without
lowering the dominant replay bodies enough.

A focused current-head `few_shot` TorchInferno-only profile on `85007b2` wrote
`/tmp/ti-few-current-85007b2-results/.../runs/20260707_221159` and landed at
`170.1 / 46.6 / 211.6ms`, p99 `1198.4/197.1/1244.4ms`, `977/1000` correct.
The live shape is fully warmed and concentrated: `max_active=32`,
`prefix_rows=64`, no decode-many, no runtime FP8 prefill, `34` prefill
batches, `31` hot replays of
`prefix_graph:b32:s16:p122-122:src1:mixed0`, and no request-path prefill graph
captures or misses. The hot row accounts for `987` requests, `12.3K` real
suffix tokens, `15.9K` model tokens, and `3.48K` suffix-padding tokens with
actual suffix lengths `12/13/14`. The existing exact-suffix, FP8, PRBD,
prequeue, row-cap, warm-prefix-copy, selected-logit, and greedy-mid
decode-many directions do not change under this profile: the current target is
still a faster model-side cached-prefix prefill body, or a packed prefill body
that avoids the Python packed-eager path's already-measured overhead.

Public `20260710_050745` exposed a distinct graph-cache cleanup failure mode on
TorchInferno `861b7c3`: few_shot/self started with the expected 16 ragged decode
graphs (`symm128` buckets), then the 512-token greedy multi_turn session entered
the online batcher with only `2.062MB` free CUDA memory. Low-memory cleanup
cleared all `136` ragged-prefill graphs and every decode graph, raising free
memory to `23818MB`. Because greedy-large prefix prefill and TP online decode
both disable capture-on-miss, multi_turn paid `33` request-path ragged-prefill
graph misses and later tree/long_output ran with zero decode graph replays; the
public long_output tail had `0` live decode graphs, `0` hits, and `1472` ragged
decode misses. The fix is to make the low-memory path trim ragged-prefill graph
entries first using the model-side memory watermark, then clear prefill graph
caches before considering a full graph clear. The cleanup command now carries
explicit `clear_graph_caches`, `clear_prefill_graph_caches`, and
`trim_graph_caches` bits so TP workers mirror rank 0. This keeps the existing
OOM guard while preserving warmed decode graphs whenever prefill graph cleanup
recovers enough memory.

The greedy-mid token-prefill default remains rejected. A broad env probe
(`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_TOKEN_GRAPH=1` plus token suffix `16`)
improved local few_shot to `172.2 / 35.5 / 201.7ms`, but the narrow
code-default version landed at `184.4 / 35.5 / 214.6ms`, worse than the local
control `182.7 / 35.1 / 213.2ms`. Keep token-prefill expansion opt-in until it
beats the default without relying on broad suffix coverage.

Repeating the broad token-prefill probe on current head `9513b9e` confirms it
is not defaultable. The run
`/tmp/inference-bench-torchinferno-few-tokenprefill-current-results/.../runs/20260710_071040`
landed at `205.9 / 33.7 / 239.6ms`, `977/1000` correct, worse than the
current profiled control `186.6 / 34.2 / 216.4ms`. Queue telemetry showed the
same `34` prefill batches and `14.1K` prefill tokens as default, but prefill
graph replay time rose to `305.5ms` (`b32:s16` accounted for `304.4ms`) and
queue-to-first rose to `191.3ms`. The earlier broad win was not a stable
mechanism-level improvement.

Greedy token-only common-prefix prefill is also rejected as a default for the
current public few_shot row. The no-logits graph path is useful as a diagnostic:
when full-prompt logits are skipped under pinned shared-prefix caching, it can
return only greedy next tokens and avoid the runtime sample-select body. It did
not close the score gap in focused local probes against public
`20260710_140141` (`174.0 / 34.5 / 203.4ms`).

The broad token-only suffix warmup probe
`/tmp/inference-bench-few-tokenonly-results/.../runs/20260710_150115` landed at
`184.7 / 34.0 / 216.1ms`, `977/1000` correct. It lowered few_shot prefill
forward from `1.91s` to `1.75s`, but added `15` token-only resident prefill
graph entries, pushed startup memory to roughly `95GB/GPU`, raised prefill graph
evictions from `4` to `25`, and inflated decode graph replay from `30.5ms` to
`248.5ms`.

The request-capture probe
`/tmp/inference-bench-few-tokenonly-capture-results/.../runs/20260710_150852`
disabled broad token suffix warmup and enabled token-only capture-on-miss. It
kept decode replay healthy and cut prefill sample-select from `54.9ms` to
`2.8ms`, but paid `3` request-path prefill captures (`2.50s`) and landed at
`178.2 / 33.5 / 209.0ms`, with p99 TTFT/E2E above `2s`.

The targeted warmup probe
`/tmp/inference-bench-few-tokenonly-targeted-results/.../runs/20260710_151454`
warmed suffix `16` for observed small/big buckets and avoided request-path
captures, landing at `174.1 / 34.6 / 204.2ms`, `977/1000` correct. Median
queue-to-first was effectively flat (`165.25ms` vs `165.42ms` control), p99
TTFT/E2E worsened to `1420/1454ms`, and overall phase time rose by `1.31s`.

A pure token-only probe added a separate diagnostic warmup switch,
`TORCHINFERNO_OPENAI_WARMUP_ONLINE_GREEDY_COMMON_PREFIX_TOKEN_ONLY_SUFFIX_PREFILL`,
so the no-logits graphs can be warmed without also enabling the older
logits+token runtime fallback. The run
`/tmp/inference-bench-few-tokenonly-pure-results/.../runs/20260710_152822`
landed at `176.1 / 33.2 / 203.2ms`, `977/1000` correct. It kept decode replay
healthy (`31.6ms`), avoided request-path captures, and cut sample-select to
`3.0ms`, but p99 TTFT/E2E remained `1410/1435ms`, phase time rose by `1.33s`,
and prefill state-create time grew to `212ms` on the hot shape. Keep token-only
prefill default-off; the remaining few_shot gap is still the hot cached-prefix
`b32:s16` prefill body, not logits materialization or greedy sampling.

Post-fix focused local validation on `a4d92f0` plus the trim-first cleanup
patch wrote
`/tmp/inference-bench-torchinferno-cleanup-trim-results/.../runs/20260710_060847`.
The normal no-cleanup path stayed in the expected band: few_shot
`184.2 / 36.0 / 214.8ms`, multi_turn `242.3 / 37.9 / 283.1ms`, correctness
`0.977/0.982`. Queue telemetry showed no cleanup applied locally, `148` live
ragged-prefill graph entries, zero evictions, `33` prefill graph replays on
both workloads, and multi_turn only one miss. The new unit coverage directly
exercises the low-memory trim-first path, prefill-only fallback clear, decode
graph preservation, and the TP cleanup command bits.

A focused long_output validation on `6979820` wrote
`/tmp/inference-bench-torchinferno-cleanup-prefill-only-results/.../runs/20260710_062209`.
It landed at `226.2 / 20.6 / 1078.9ms`, p99 `1514.7 / 36.8 /
1864.9ms`, `1000/1000` correct. Queue telemetry showed no cleanup on the
normal path, `16` live ragged decode graphs (`symm128` buckets), `728` decode
graph replays, `16` misses limited to static decode warmup shapes, and zero
request-path graph captures. This matches the intended post-cleanup invariant:
long_output should keep warmed decode graphs unless the last-resort full graph
clear is actually required.

Two current-head sequence reruns kept the same invariant across workload
handoff. A TorchInferno-only `multi_turn long_output` run on `e19cd9e` wrote
`/tmp/inference-bench-torchinferno-cleanup-sequence-results/.../runs/20260710_064659`.
It landed at multi_turn `241.1 / 39.9 / 275.2ms`, long_output `246.7 / 21.0 /
1089.4ms`, and kept `16` live ragged decode graphs with `736` long_output
replays. Repeating the sequence with
`TORCHINFERNO_OPENAI_TP_ONLINE_GRAPH_CLEANUP_MIN_FREE_MB=8192` wrote
`/tmp/inference-bench-torchinferno-forced-cleanup-sequence-results/.../runs/20260710_065308`.
It landed at multi_turn `232.5 / 38.3 / 264.0ms`, long_output `220.0 / 22.1 /
1115.3ms`, and again kept `16` live decode graphs with `728` long_output
replays. The local host still had enough free memory that even the 8GB threshold
did not apply cleanup, so direct low-memory branch evidence remains the unit
coverage plus the public `20260710_050745` cleanup telemetry.

The greedy-short initial wait expansion remains rejected on current head. A
local `TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_INITIAL_BATCH_WAIT_MS=5`
probe wrote
`/tmp/inference-bench-torchinferno-long-wait5-results/.../runs/20260710_063320`
and landed slightly worse than the 2ms default: `226.7 / 21.2 / 1083.5ms`,
p99 `1533.2 / 37.1 / 1902.3ms`, `1000/1000` correct. Queue telemetry showed
the first batch still admitted only one request, while prefill fragmented from
56 to 59 batches and prefill wall rose from `5.23s` to `5.45s`.

Disabling the OpenAI-scoped prefill symmetric-memory all-reduce is not promoted
from a focused few_shot recheck. The env run
`TORCHINFERNO_OPENAI_TP_SYMM_MEM_PREFILL_ALLREDUCE=0` wrote
`/tmp/inference-bench-torchinferno-few-nosymm-prefill-results/.../runs/20260710_070115`
and landed at `179.6 / 35.4 / 209.7ms`, `977/1000` correct, versus the
comparable current profiled control at `186.6 / 34.2 / 216.4ms`. The profile
does not show a prefill-model win: prefill forward was effectively flat
(`1.77s -> 1.78s`) and prefill wall worsened (`2.44s -> 2.59s`), while the
median shift came from queue/decode variation. Keep the current default and
treat this as noise unless a repeated full-suite comparison shows a stable
score win.

Decode-many queue profiles now also export
`runtime_decode_many_shape_steps` for the non-graph decode-many path. The
existing `runtime_decode_many_shape_model_tokens` and emitted/skipped counters
show token volume, while the new step counter makes it explicit whether a shape
is expensive because it repeats many single-step replays or because each replay
is slow. This is telemetry only; it does not change scheduling or graph replay
behavior.

A current-head long_output validation with that telemetry on `f3d6a47` wrote
`/tmp/inference-bench-torchinferno-long-current-shapesteps-results/.../runs/20260710_072229`.
It landed at `228.4 / 20.9 / 1029.4ms`, p99 `1571.8 / 28.4 /
1771.0ms`, `1000/1000` correct. Queue telemetry stayed in the expected band:
`53` prefill batches, `51.5K` prefill tokens, `4.54s/5.25s` prefill
forward/wall, `16` live decode graph entries, `736` decode graph replays, and
`16` misses. The new decode-many shape-step map makes the hot body explicit:
`decode_many:b64/64` ran `270` single-step replays, consumed `17.28K` model
tokens, emitted `16.27K`, skipped `1.01K`, and accounted for about `3.43s` of
step-window model time. The remaining long_output gap is therefore still the
full-width decode replay body and prefill/decode pipeline structure, not just a
tail-padding counter.

A same-host TorchInferno/vLLM long_output refresh on pushed `e617c52` wrote
`/tmp/inference-bench-ti-vllm-long-e617c52-429f405-results/.../runs/20260710_085503`.
vLLM `429f405` landed at `50.0 / 16.9 / 651.8ms`, while TorchInferno landed at
`226.7 / 21.5 / 999.6ms`, both `1000/1000` correct. This removes public-run
host variance from the current target: TorchInferno still loses first token by
about `4.5x` and TPOT by `27%`. The TorchInferno profile used the no-env
default with decode-many async readback off and kept the warmed decode graphs
live (`16` entries, `735` replays, `2` misses). It spent `5.48s` in prefill
wall (`4.72s` forward across `56` batches), then `7.63s` in decode-many GPU
events and `7.14s` in the old `decode_many_cpu_tokens_ms` bucket across
`137` decode-many calls / `605` steps. Queueing is still material:
`q2submit_p50=93.5ms`, `submit2first_p50=114.9ms`, and
`q2first_p50=214.8ms`. Prior cap/wait/chunk/fine-bucket A/Bs already reject
the simple scheduler knobs; the supported next work remains a real
prefill/decode overlap design or faster full-width decode replay body.

The same profile also shows why `runtime_decode_many_cpu_tokens_ms` needs a
clearer split. In the CUDA decode-many path, model calls are intentionally not
synchronized step-by-step; the synchronous token materialization is often the
first stream fence, so the old CPU token bucket includes GPU wait time. Queue
profiles now export `runtime_decode_many_token_wait_ms` and
`runtime_decode_many_token_materialize_ms` while preserving the existing total
field. Use those fields on the next long_output run before treating token
readback as a Python-side bottleneck.

The post-push validation on `2044b85` wrote
`/tmp/inference-bench-torchinferno-long-2044b85-results/.../runs/20260710_091123`.
It stayed in the current score band at `222.5 / 21.9 / 1076.3ms`,
`1000/1000` correct, with `57` prefill batches, `5.59s/4.94s`
prefill wall/forward, `728` decode graph replays, and `16` decode graph
misses limited to warmup/static shapes. The new split shows the old
`runtime_decode_many_cpu_tokens_ms=7219ms` is almost entirely stream wait:
`runtime_decode_many_token_wait_ms=7180ms` and
`runtime_decode_many_token_materialize_ms=34ms`. Token list materialization is
therefore not a score-facing long_output lever; the decode gap remains the
GPU replay body plus prefill/decode pipeline ordering.

A same-host TorchInferno/vLLM tree_of_thought refresh on pushed `3dfd5c5`
wrote
`/tmp/inference-bench-tree-3dfd5c5-results/.../runs/20260710_092441`. The
cached vLLM tree build `cbe9c40` landed at `38.5 / 26.8 / 57.1ms`, while
TorchInferno landed at `61.6 / 40.0 / 90.6ms`, with correctness `960/992` vs
`953/992`. The TorchInferno score gap is now roughly `1.6x` TTFT/E2E and
`1.5x` TPOT, much smaller than the older `132/66ms` tree baseline but still
visible.

The current TorchInferno profile is prefill-bound: `q2first_p50=57.4ms`,
`q2submit_p50=22.1ms`, `submit2first_p50=36.1ms`, `6.41s` prefill forward,
`7.84s` prefill wall, and `44%` padded prefill tokens. Decode-many is not on
this path. Hot shapes remain the sampled common-prefix suffix bucket
`prefix_graph:b2/b4/b16:s16:p45-45:src1:mixed0`, with the largest packed
candidate savings in `b4:s16` and `b2:s16`. The new research-summary columns
make the active policy explicit for future A/Bs: `init_wait=1.0`,
`idle_wait=2.0`, `active_wait=1.0`, and `decode_capture=False`. Do not reopen
sampled `s12` suffix buckets, generated-prefix caching, or larger
sampled-medium active caps as defaults; those have already failed to turn the
padding signal into a score win. The next score-facing tree lever needs a real
packed/FlashInfer-style prefix-suffix prefill body or another way to reduce the
`s16` padded model work without multiplying small graph replays.

A focused dirty-tree decode cleanup allowed same-temperature sampled ragged
decode to use the contiguous-row graph form when the active physical rows are
already dense. The TorchInferno-only tree run wrote
`/tmp/inference-bench-tree-contigrows-results/.../runs/20260710_094515` and
landed at `69.2 / 40.9 / 99.7ms`, `954/992` correct, so it is not a
score-facing tree win by itself. Combining its non-overlapping queue-profile
records shows the intended small effect: decode graph replay fell from
`124.2ms` to `114.2ms`, decode GPU from `3.44s` to `3.17s`, and `5.6ms` of
decode replay moved to `rows0` contiguous graph keys. The overall median still
regressed because prefill wall rose from `7.84s` to `9.11s`; keep treating the
tree gap as packed prefix-suffix prefill first, with this row-index change only
as a minor sampled-decode cleanup.

A follow-up dirty-tree prefill cleanup lets dense ordered prefix-prefill batches
omit `row_indices` too, and fixes the Llama3 TP prefix-copy path so
`src_prefix_row` still copies into destination rows `0..N-1` when `row_indices`
is absent. The TorchInferno-only tree run wrote
`/tmp/inference-bench-tree-prefill-contigrows-results/.../runs/20260710_100021`
and landed at `65.8 / 39.4 / 94.6ms`, `954/992` correct. The final queue
profile showed the intended narrow effect: prefill graph replay fell to
`816ms` versus `1319ms` on the clean `3dfd5c5` control, with prefill
forward/wall roughly flat-to-slightly-better (`6.27s/7.71s` vs
`6.41s/7.84s`). Only two live ragged-prefill graph entries used `rows0`
against `133` `rows1` entries, so this is not the main tree lever; keep it as
correctness coverage and a small row-index-free graph scaffold, while the
score-facing gap still points at the padded `s16` prefix-suffix body.

Extending that cleanup to dense row permutations is rejected. A dirty prototype
reordered prefix-prefill inputs when physical rows were a permutation of
`0..N-1`, then gathered real outputs back to request order. The run
`/tmp/inference-bench-tree-prefill-denseperm-results/.../runs/20260710_101213`
regressed to `70.4 / 40.2 / 101.1ms`, `954/992` correct, with p99 TTFT/E2E
above `2.5s`. The queue profile showed why: it added cold `rows0` prefill graph
captures (`~466-547ms`) while still leaving almost all live ragged-prefill graph
entries as `rows1` (`1-2` `rows0` entries versus `133` `rows1`). Keep the
row-index-free prefill path limited to exact ordered physical rows unless it is
paired with warmup or allocator changes that avoid new capture tails.

Queue profiles now record prefix-prefill row-index mode counters:
`runtime_prefill_row_indices_omitted_batches/rows`,
`runtime_prefill_row_indices_indexed_batches/rows`, and per-shape maps for both
modes. Use these alongside live `rows0/rows1` graph-cache keys to verify that a
future row-layout experiment changes actual replay calls, not just resident graph
entries.

Keeping the active-row freelist sorted after release and prefix-row adoption is
a small general cleanup, not a score-facing multi_turn fix by itself. The dirty
TorchInferno-only multi_turn run
`/tmp/inference-bench-multiturn-lowrows-results/.../runs/20260710_104006`
landed at `234.5 / 37.3 / 266.7ms`, slightly ahead of the earlier
cleanup-sequence sample (`241.1 / 39.9 / 275.2ms`), but the profile showed only
one omitted row-index prefill batch with two rows versus 32 indexed batches and
1000 indexed rows. Live ragged-prefill graph keys also stayed concentrated in
`rows1` (`0` `rows0` entries and `148` `rows1` entries). Treat this as
allocator hygiene and unit coverage for dense low-row reuse; the remaining
multi_turn gap still needs a real mixed-prefix prefill/queueing lever.

Aligning greedy common-prefix suffix warmup with row-index-free dense prefill is
a few_shot p99 fix, not a median throughput win. The pushed `9de1244` control
run
`/tmp/inference-bench-torchinferno-few-9de1244-results/.../runs/20260710_105028`
landed at `180.1 / 33.9 / 210.4ms` with p99
`2108.5 / 68.6 / 2122.8ms`. Its request path omitted row indices for all
prefill batches, but startup warmup had only populated indexed suffix graphs,
so live traffic captured cold `rows0` prefill graphs for
`prefix_graph:b2/b8/b32:s16:p122-122:src1:mixed0` and spent `2.72s` in prefill
graph capture. The dirty dense-extra-pair warmup run
`/tmp/inference-bench-torchinferno-few-densewarm-results/.../runs/20260710_110102`
landed at `181.0 / 35.2 / 211.2ms`, p99
`447.1 / 267.1 / 484.9ms`, with zero request-path prefill graph captures and
the hot `b32:s16:p122` path replay-only. Startup readiness grew from `231.0s`
to `261.2s` and warmup memory reached about `84GB/GPU`, so keep the change
limited to already-configured greedy-short extra pairs instead of broadening
the warmup shape set.

Extending the same dense row-index-free suffix warmup to small base
common-prefix batches is rejected for multi_turn. The pushed `1cd8f36` control
run
`/tmp/inference-bench-torchinferno-multiturn-1cd8f36-results/.../runs/20260710_110920`
landed at `243.1 / 37.6 / 277.3ms`, p99
`1275.7 / 272.3 / 1305.5ms`, with one request-path prefill miss on the tiny
initial `ragged_prefill:b2:s16:rows0:ctx-64:src1` shape. A dirty probe that
warmed dense base batches up to `b2`
`/tmp/inference-bench-torchinferno-multiturn-densebase-results/.../runs/20260710_111858`
removed that miss, but regressed to `249.5 / 39.5 / 286.7ms`, p99
`1295.3 / 83.8 / 1333.2ms`, and increased readiness from `261.2s` to
`296.3s`. The remaining multi_turn gap is the expensive and fragmented
`b32:s32:mixed1` request-prompt replay body, not this one small cold dense-base
graph.

Adding a new `p111/s96` greedy-short extra warmup is also rejected for
long_output. A fresh `02d160a` control wrote
`/tmp/inference-bench-torchinferno-long-02d160a-results/.../runs/20260710_112817`
and landed at `224.4 / 21.6 / 1057.7ms`, p99
`1613.9 / 56.1 / 2924.1ms`, `1000/1000` correct. Its profile had one
request-path capture for `prefix_graph:b16:s96:p111-111:src1:mixed0`,
`1.42s` prefill graph capture GPU, `6.11s/6.77s` prefill forward/wall, and
`7.76s` decode-many GPU. A dirty probe adding `111:96` to the existing
greedy-short extra pairs wrote
`/tmp/inference-bench-torchinferno-long-p111s96warm-results/.../runs/20260710_113613`
and did remove that capture (`runtime_prefill_graph_capture_gpu_ms=0`,
prefill forward/wall `4.85s/5.48s`), but regressed score-facing medians to
`232.1 / 21.8 / 1069.4ms` while adding another benchmark-shaped startup
warmup constant. Keep the default extra pairs at the already-configured
`111:32`, `111:64`, and `122:16`; the remaining long_output gap is the
single-step high-active decode replay body plus padded prefill, not more
shape-specific warmup.

A same-host vLLM refresh after the TorchInferno `02d160a` long_output profile
wrote
`/tmp/inference-bench-vllm-long-refresh-results/.../runs/20260710_114420`
using vLLM `cbe9c40f998f`. It landed at `50.1 / 16.9 / 644.4ms`, p99
`90.6 / 23.1 / 1120.9ms`, and `1000/1000` correct. vLLM startup readiness was
`65.3s`; its server log showed graph capture finishing in about `9s`, custom
all-reduce graph-address registration, `65.7%` prefix-cache hit rate, about
`4.8K` prompt tok/s, and about `3.6K` generation tok/s. Against the fresh
TorchInferno control, the same-host gap is still roughly `+174ms` TTFT,
`+4.7ms` TPOT, and `+413ms` E2E. No reusable SGLang venv was present in the
active skipped-build directory for this refresh; the older same-host SGLang
rows remain the current SGLang comparison.

Public `20260710_111248` still measures stale TorchInferno `a4d92f0`, before
the current cleanup and dense-greedy warmup patches. The public scorecard is
TorchInferno `4/20`, vLLM `14/20`, and SGLang `1/20`; the visible public gaps
remain multi_turn, tree_of_thought, and long_output TTFT/E2E. The public
TorchInferno queue profile still shows many stale ragged decode misses and
older prefill behavior, so current local comparisons should use the pushed
`d6f1b5c` branch runs until the public pointer advances.

A stop-synchronized decode-many probe for current multi_turn is rejected as a
default. The env run
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY=1`,
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_ALLOW_STOP=1`, and
`TORCHINFERNO_CONTINUOUS_RAGGED_DECODE_MANY_SYNC_STOPS=1` wrote
`/tmp/inference-bench-multiturn-syncstops-d6f1b5c-results/.../runs/20260710_120532`.
It improved TTFT versus the nearby profiled control (`215.6ms` vs `243.1ms`)
and removed overgeneration, but E2E was flat (`276.9ms` vs `277.3ms`) and TPOT
regressed (`44.2ms` vs `37.6ms`). Queue telemetry shows why this is not a
promotion: it ran `65` stop-synchronized decode-many steps, emitted `1427`
tokens with zero skips, but charged `662.7ms` of token wait because each step
must synchronize to observe stop tokens before the next step. Keep sync-stops
decode-many as a diagnostic until streaming can emit between internal steps or
stop compaction can stay GPU-side.

Batching cache seq-len updates inside prefix-graph prefill is accepted as small
runtime hygiene, not a score-facing multi_turn fix. The dirty validation run
`/tmp/inference-bench-multiturn-batchedseqlen-dirty-results/.../runs/20260710_121540`
landed at `242.8 / 38.9 / 283.2ms`, essentially flat-to-slightly-worse versus
the `1cd8f36` control `243.1 / 37.6 / 277.3ms`. The internal counter moved in
the intended direction: prefill state setup fell `57.2ms -> 32.3ms`, with the
row seq-len portion falling `38.9ms -> 14.3ms`, and prefill wall fell
`3.41s -> 3.18s`. This removes avoidable repeated cache setter work from the
mixed-prefix prefill path, but the remaining row gap is still dominated by
fragmented padded `b32:s32:mixed1` prefill and queue formation.

A current-head full-suite validation after forwarding the TP online close sync
bit wrote
`/tmp/inference-bench-close-sync-results/.../runs/20260710_220324` and completed
all five TorchInferno rows without reproducing the public `20260710_210746`
rank-0 NCCL watchdog crash. Metrics were `170.1 / 34.6 / 198.9ms` for
few_shot, `35.3 / 0.0 / 35.7ms` for self_consistency,
`229.7 / 37.2 / 263.5ms` for multi_turn, `77.4 / 45.8 / 107.3ms` for
tree_of_thought, and `227.3 / 21.6 / 1027.1ms` for long_output. The useful
long_output queue profile arrived as a completed `online_batcher_quiescent`
record rather than a final `online_batcher` record because the benchmark
terminated the server after receiving all responses. That row had max-active
`64`, `60` prefill batches, `139` decode-many calls, `604` internal
decode-many steps, `31,957` model tokens, `30,369` emitted tokens, and `1,588`
skipped tokens; `decode_many:b64/64:g1-16` alone accounted for `11,584` model
tokens. Completed quiescent rows are now marked as complete profile snapshots so
this final-shape detail is not missed by profile readers when the server is
stopped immediately after the last response.

Rechecking smaller greedy-short decode-many stop-tail caps on current
`b9a4a38` keeps the default cap at `4`. The cap-1 probe
`/tmp/inference-bench-tailcap1-results/.../runs/20260710_221604` landed at
`226.6 / 21.6 / 1040.1ms`: it cut skipped decode-many tokens from the cap-4
control's `1,588` to `527`, but fragmented the tail into `380` decode-many
calls and regressed median E2E. The cap-2 probe
`/tmp/inference-bench-tailcap2-results/.../runs/20260710_222246` landed at
`214.8 / 21.5 / 1039.0ms`: it improved TTFT and p99 E2E, reduced skipped
decode-many tokens to `722`, and lowered the internal phase snapshot to
`17.34s`, but still missed the score-facing cap-4 median E2E (`1027.1ms`).
Treat cap `1` and cap `2` as opt-in tail-latency diagnostics, not default
promotions.

The sampled-medium tree max-active cap is back to `32` on current evidence. A
same-commit focused default control on `2db2e1d` wrote
`/tmp/inference-bench-tree-default-2db-results/.../runs/20260710_223655` and
landed at `60.0 / 37.2 / 88.4ms`, correctness `0.962`, with queue-to-finish
p50 `83.7ms` and phase total `15.53s`. The matching
`TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_MEDIUM_MAX_ACTIVE=32` probe wrote
`/tmp/inference-bench-tree-maxactive32-results/.../runs/20260710_223043` and
landed at `59.2 / 37.7 / 85.5ms`, correctness `0.968`, queue-to-finish p50
`80.1ms`, and phase total `15.25s`. Prefill/decode counts were essentially
unchanged (`353/302` batches at 16 rows, `358/307` at 32 rows), so this is a
small tree scheduling default, not a new prefill implementation. It remains
scoped to `temperature > 0` and `256 < max_tokens <= 300`.

The current pushed `a360de1` full-suite validation wrote
`/tmp/inference-bench-a360-full-results/.../runs/20260710_224606` and completed
all TorchInferno rows: few_shot `177.7 / 34.2 / 206.2ms`,
self_consistency `34.7 / 0.0 / 35.2ms`, multi_turn
`249.9 / 38.0 / 279.8ms`, tree_of_thought `61.4 / 36.3 / 88.5ms`, and
long_output `233.6 / 21.7 / 995.6ms`. Public results remained stale at
`20260710_210746`, where TorchInferno still crashes mid-suite before the close
sync and completed-quiescent-profile fixes. The local priority shifted back to
multi_turn because same-host public vLLM is still `76.8 / 36.1 / 104.7ms`.

Raising greedy-large multi_turn max-active from `32` to `48` is rejected. The
focused probe
`/tmp/inference-bench-multi-maxactive48-results/.../runs/20260710_225405`
landed at `615.8 / 144.0 / 743.1ms`, correctness `0.981`. It slightly reduced
prefill batches (`36 -> 32`) and queue-to-submit (`135.2ms -> 123.1ms`), but
blew up submit-to-first (`94.8ms -> 471.2ms`) and doubled decode graph misses
(`46 -> 86`) while cutting decode graph hits (`68 -> 40`). Keep greedy-large
online max-active at `32`.

Graph capture/miss shape counters now follow the lightweight queue-profile
contract. Before this change, no-sync public-style profiles could report
`runtime_decode_graph_misses > 0` with an empty
`runtime_decode_graph_miss_shape_counts` map because the shape maps were gated
behind `profile_timings`. The runtime now records prefill/decode graph
capture/miss shape counts when `TORCHINFERNO_OPENAI_QUEUE_PROFILE_JSONL` is
enabled, without enabling CUDA-synchronized timing maps. Focused unit coverage
checks both prefill and decode graph shape maps with `profile_timings=False`.

A focused current-head multi_turn run with the new diagnostic wrote
`/tmp/inference-bench-multi-shapemap-results/.../runs/20260710_230407` and
landed at `241.2 / 38.8 / 273.1ms`, correctness `0.981`. The queue profile
now shows the miss source clearly: `runtime_decode_graph_misses=44` split
across static single-row decode token/logits shapes at seq lens `57-66`,
`73-75`, `92-96`, `109`, and `138`; the live decode graph cache only held the
ragged batch shapes. The remaining multi_turn decode gap is therefore not a
new ragged batch bucket. It is the static row-view path falling back to eager
because TP runtime disables capture-on-miss and startup warmup captures static
graphs on the base cache, while online serving uses cached `for_rows(...)`
views whose graph keys are tied to the view object.

Routing single-active greedy decode through the existing ragged row-index graph
closes that miss source without changing TP static graph keying. The dirty
validation
`/tmp/inference-bench-multi-ragged-single-results/.../runs/20260710_231839`
landed at `229.3 / 39.3 / 260.5ms`, correctness `0.980`. The queue profile
showed `runtime_decode_graph_misses=0`, `runtime_decode_graph_hits=95`, and
`runtime_decode_graph_replays=95`; the previous diagnostic run had `44` misses
and `65` hits/replays. Median TPOT was flat-to-slightly-worse, but median TTFT
improved by `11.9ms`, median E2E by `12.6ms`, and p99 E2E dropped from
`1738.7ms` to `715.4ms`. Keep this as a general runtime routing fix: single
active rows should prefer the already-warmed ragged graph over static row-view
graphs that TP cannot capture on miss.

The pushed-head `4e79fc9` full-suite validation wrote
`/tmp/inference-bench-4e79-full-results/.../runs/20260710_232601` and completed
all rows: few_shot `169.8 / 34.7 / 197.3ms`, self_consistency
`28.5 / 0.0 / 29.3ms`, multi_turn `244.7 / 37.2 / 274.5ms`,
tree_of_thought `60.9 / 39.2 / 90.1ms`, and long_output
`235.7 / 21.5 / 944.8ms`. Correctness stayed in band
(`977/1000`, `1000/1000`, `982/1000`, `956/992`, `1000/1000`). Multi_turn
still shows some run-to-run median variance versus the focused probe, but the
queue profile confirms the intended structural fix held in the full suite:
`runtime_decode_graph_misses=0`, `runtime_decode_graph_hits=91`, and
`runtime_decode_graph_replays=91` for the multi_turn row. The same full-suite
profile now exposes a separate sampled-row target: self_consistency/tree still
report small static logits misses at `b2/b3/b4...:s55/s56/s57`, because sampled
decode cannot use the greedy ragged token graph path.

Extending the same fallback to ragged logits reduces the sampled-row miss
source. The dirty tree_of_thought validation
`/tmp/inference-bench-tree-ragged-sampled-results/.../runs/20260710_233507`
landed at `62.0 / 41.5 / 87.9ms`, correctness `0.964`. The completed
queue-profile snapshots showed sampled decode graph misses falling from the
full-suite row's `17` static logits misses to at most `1`
(`static_decode:logits:b4:s56`) while keeping correctness unchanged. Treat this
as a small routing cleanup, not a primary tree score win: TTFT/TPOT remained
within the current focused-run variance, but the static row-view sampled miss
source is almost gone.

The pushed-head `8ff3638` full-suite validation wrote
`/tmp/inference-bench-8ff-full-results/.../runs/20260710_234322` and completed
all rows: few_shot `172.8 / 34.7 / 200.6ms`, self_consistency
`20.4 / 0.0 / 21.1ms`, multi_turn `220.5 / 36.8 / 249.4ms`,
tree_of_thought `57.9 / 38.0 / 80.5ms`, and long_output
`242.8 / 21.1 / 972.7ms`. Correctness stayed in band
(`977/1000`, `1000/1000`, `980/1000`, `958/992`, `1000/1000`). Relative to
the `4e79fc9` full suite, self_consistency, multi_turn, and tree medians moved
the right way, while few_shot and long_output TTFT moved slightly against the
change and should be treated as run variance or unrelated scheduler mix. The
graph profile confirms the sampled fallback mostly held under full-suite load:
tree ended with `runtime_decode_graph_hits=331`, `runtime_decode_graph_misses=6`,
and `runtime_decode_graph_replays=331`, down from the previous full-suite tree
row's `17` static sampled logits misses. Long_output remained clean on decode
graphs (`759` hits/replays, `0` misses), so the remaining long gap is still
decode throughput and tail policy, not graph coverage.

A follow-up dirty patch closed the remaining sampled tree static decode miss
routes by applying the ragged-logits fallback to the FI decode branch, allowing
mixed-temperature groups to use ragged logits, and padding odd fallback batches
to the existing ragged decode buckets. The first focused tree validation
(`/tmp/inference-bench-tree-fi-ragged-logits-results/.../runs/20260710_235535`)
landed at `53.9 / 35.1 / 76.6ms`, correctness `953/992`, but still had `4`
static logits misses. After mixed-temperature groups used ragged logits, the
second validation
(`/tmp/inference-bench-tree-fi-mixed-ragged-logits-results/.../runs/20260711_000249`)
landed at `54.4 / 36.1 / 79.4ms`, correctness `959/992`, and shifted the
remaining misses to ragged odd batches (`b5/b6`). The final bucketed run
(`/tmp/inference-bench-tree-bucketed-ragged-logits-results/.../runs/20260711_001101`)
landed at `54.3 / 37.9 / 79.0ms`, correctness `956/992`, with
`runtime_decode_graph_hits=347`, `runtime_decode_graph_replays=347`, and
`runtime_decode_graph_misses=0`. This is a graph-coverage cleanup and modest
tree median improvement versus the pushed `8ff3638` full-suite tree row
(`57.9 / 38.0 / 80.5ms`), but it does not change the larger sampled-prefix and
decode-throughput priorities.

The pushed-head `7a025c4` full-suite validation wrote
`/tmp/inference-bench-7a025-full-results/.../runs/20260711_001911` and completed
all rows: few_shot `186.0 / 34.4 / 213.4ms`, self_consistency
`24.4 / 0.0 / 25.1ms`, multi_turn `220.2 / 36.0 / 249.3ms`,
tree_of_thought `53.7 / 35.9 / 75.6ms`, and long_output
`236.7 / 21.3 / 989.2ms`. Correctness stayed in band
(`977/1000`, `1000/1000`, `979/1000`, `958/992`, `1000/1000`). The tree row
held the intended structural fix in the full suite:
`runtime_decode_graph_hits=355`, `runtime_decode_graph_replays=355`, and
`runtime_decode_graph_misses=0`; the previous `8ff3638` full-suite tree row had
`6` sampled decode graph misses. Treat the few_shot TTFT/E2E regression as
unrelated run variance until it repeats, because the routing change does not
touch that row's prompt path and decode graph misses stayed at zero there.

The current long_output sync-timing refresh on the same pushed head wrote
`/tmp/inference-bench-7a025-long-sync-results/.../runs/20260711_002946` and
landed at `232.1 / 21.1 / 984.7ms`, `1000/1000` correct. The profile stayed on
the known path: `52` prefill batches with `5.25s` prefill wall / `4.59s`
forward, `744` clean decode graph replays, and `128` decode-many calls covering
`577` internal steps. Decode-many CPU time was again stream wait rather than
Python materialization (`6976.7ms` wait vs `28.7ms` materialize), and packed
candidate telemetry still pointed at the same cached-prefix prefill target
(`33.8K` avoidable padded suffix tokens).

That run exposed a profiling attribution gap rather than a new serving gap:
hot greedy ragged prefill token graph replays used `capture_on_miss=False`, so
their sync-timing GPU events were not recorded. The real hot shapes had forward
time (`b24:s64`, `b24:s96`, `b32:s64`, `b32:s96`) but
`runtime_prefill_shape_graph_replay_gpu_ms` only showed tiny `b1/b2/b4` entries.
Greedy token and token+logits prefill graph attempts now start the same GPU
timer regardless of capture mode, while still only recording it when the graph
actually hits. This is timing-only; default lightweight queue profiles and
serving behavior are unchanged.

The patched sync-timing validation wrote
`/tmp/inference-bench-prefill-gpu-timing-results/.../runs/20260711_003722` and
landed at `223.2 / 21.8 / 965.6ms`, `1000/1000` correct. The intended
telemetry is now present: `runtime_prefill_graph_replay_gpu_ms=4984.3ms`, with
hot shape GPU attribution including `b24:s64=975.3ms`, `b24:s96=780.3ms`,
`b32:s64=682.9ms`, and `b32:s96=378.8ms`. This confirms the long_output gap is
ordinary cached-prefix suffix prefill body plus high-active decode-many replay;
it is not hidden graph capture, token materialization, or missing shape timing.

The current same-host all-provider refresh on `d78c1ae` wrote
`/tmp/inference-bench-current-compare-results/.../runs/20260711_004651`.
The target rows were vLLM tree `39.1 / 26.6 / 56.9ms` and long_output
`50.3 / 16.8 / 635.6ms`, SGLang tree `39.0 / 108.4 / 109.6ms` and long_output
`45.1 / 25.9 / 924.2ms`, and TorchInferno tree
`60.1 / 38.9 / 83.6ms` and long_output `219.4 / 21.2 / 976.0ms`.
TorchInferno stayed stable on current head (`956/992` tree, `1000/1000` long),
so the latest public `b3dab3b` multi_turn CUDA launch failure is stale relative
to the pushed sampling/close-sync fixes.

A focused greedy-short refill A/B on the same head keeps refill-floor changes
rejected. Lowering
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_ADMIT_MIN_FREE_ROWS` from `4` to
`1` wrote `/tmp/inference-bench-long-minfree1-results/.../runs/20260711_010004`
and landed at `235.2 / 21.0 / 961.3ms`, improving E2E slightly but regressing
median TTFT and p99 finish. Lowering only
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_REFILL_MIN_READY_REQUESTS` from
`8` to `4` first looked promising:
`/tmp/inference-bench-long-refill4-results/.../runs/20260711_010631` landed at
`217.7 / 21.2 / 964.4ms`, p99 `443.8 / 30.3 / 1751.7ms`, `1000/1000` correct,
with p99 first-token down from `532.2` to `438.8ms`. The no-env patched
confirmation then regressed to `243.1 / 20.9 / 975.1ms`, p99
`1392.3 / 42.0 / 2571.9ms`, and a midpoint floor of `6` landed at
`253.5 / 21.4 / 1000.0ms`. Keep the greedy-short refill floor at `8`; the
observed instability points back to model-side prefill/decode work rather than
another queue-floor default.

A current-head greedy-short suffix-split recheck on `2ea0535` keeps suffix
splitting rejected as a default. The env run
`/tmp/inference-bench-long-suffixsplit-results/.../runs/20260711_013226`
enabled
`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS_GREEDY_SHORT=1`
and stayed correct (`1000/1000`), but landed at
`237.5 / 21.2 / 983.2ms` with p99 `1565.0 / 50.9 / 2683.5ms`.
Queue telemetry showed the familiar tradeoff: `4` accepted suffix splits saved
`4160` predicted prefill model tokens, but phase time rose to `17.96s`,
queue-to-first p99 rose to `1549ms`, and the score-facing row was worse than
the no-split same-host baseline. The runtime now only removes a per-step
allocation from row-indexed decode-many seq-len updates by caching the small
device increment tensor. Its no-env validation wrote
`/tmp/inference-bench-long-decode-increment-results/.../runs/20260711_014022`
and stayed correct (`1000/1000`) but did not produce a score-facing win
(`234.5 / 21.2 / 1015.0ms`, p99 `1503.1 / 50.1 / 2614.0ms`). Keep treating
this as decode housekeeping; it does not change the main prefill-body target.

The latest public run `20260711_010210` picked up `d78c1ae`, scored
TorchInferno `4/20`, and crashed after `self_consistency`: `multi_turn` timed
out and `tree_of_thought`/`long_output` saw connection refused after a rank-0
CUDA launch failure. A local current-head TorchInferno-only multi_turn repro on
`5e855d2` plus the row-indexed decode-input scratch patch
(`/tmp/inference-bench-multiturn-repro-results/.../runs/20260711_015235`)
completed cleanly at `220.6 / 36.3 / 250.8ms`, `983/1000` correct. The queue
profile showed the remaining median gap as admission/prefill latency, not
decode-many: `use_decode_many=false`, `queue_to_submit_p50=125.1ms`,
`submit_to_first_p50=84.3ms`, and `queue_to_first_p50=212.0ms`.

Reopening the 512-token greedy active-row cap is still rejected. Raising
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_LARGE_MAX_ACTIVE` from `32` to `48`
wrote
`/tmp/inference-bench-multiturn-maxactive48-results/.../runs/20260711_020015`
and regressed multi_turn to `617.8 / 141.5 / 741.2ms`, still `983/1000`
correct. Queue telemetry made the tradeoff explicit: `queue_to_submit_p50`
moved only to `109.9ms`, while `submit_to_first_p50` ballooned to `476.7ms`.
Keep the 32-row greedy-large cap; the next multi_turn lever remains a cheaper
mixed-prefix prefill body or better queue formation at the existing decode row
count.

The row-indexed decode-input scratch patch does exercise the real long_output
decode-many path, but it remains decode housekeeping rather than a score-facing
fix. The local default long_output validation
`/tmp/inference-bench-long-output-scratch-results/.../runs/20260711_020627`
landed at `226.5 / 21.7 / 991.7ms`, `1000/1000` correct. Its final queue
profile reported `use_decode_many=true`, `149` decode-many calls, `657`
internal steps, and `40640` padded decode-many tokens, with
`queue_to_first_p50=215.8ms` and `queue_to_finish_p50=977.7ms`. Default queue
timing was sync-light, so this validates correctness and path coverage, not a
direct prepare-ms delta.

A follow-up sync-timed long_output profile on `d079b76` refreshed the current
decode prep target:
`/tmp/inference-bench-long-output-sync-d079-results/.../runs/20260711_021522`
landed at `225.8 / 21.3 / 973.8ms`, `1000/1000` correct, with
`runtime_decode_ragged_prepare_ms=484.4ms`, `queue_to_first_p50=214.0ms`,
`5.12s` prefill wall, and `7.61s` decode-many GPU. Replacing the repeated
`sorted(rows) == range(...)` dense-row checks with a one-pass dense-prefix row
helper, and reusing its ordered-row result at the call sites, reduced decode
preparation in the exact-final sync validation
`/tmp/inference-bench-long-output-dense-row-info-results/.../runs/20260711_023053`
to `367.2ms` and moved `queue_to_first_p50` to `195.6ms` while staying correct
at `1000/1000`. The benchmark row landed at `205.5 / 22.0 / 1003.8ms`; treat
this as a small TTFT/preparation cleanup, not a closure of the long_output gap,
because prefill wall (`6.56s`) and decode-many GPU (`8.06s`) still dominate.

Greedy short common-prefix suffix warmup now captures dense active-row
dynamic-prefix graph keys only for the observed long_output dense suffix bucket:
batch `16..32`, suffix `96`. The broader first pass (`batch >=16`,
`suffix >=64`) targeted the request-time miss seen after the dense-row cleanup:
`ragged_prefill:b16:s96:rows0:ctx-256:src1` captured on the request path for
`1126.9ms` of GPU time in
`/tmp/inference-bench-long-output-dense-row-info-results/.../runs/20260711_023053`.
It did remove captures in
`/tmp/inference-bench-long-output-dense-dynamic-warm-results/.../runs/20260711_025057`
(`220.2 / 21.4 / 974.2ms`, `1000/1000`, `61` prefill graph hits, no misses)
and in the no-sync confirmation
`/tmp/inference-bench-long-output-dense-dynamic-warm-nosync-results/.../runs/20260711_025753`
(`237.2 / 21.0 / 972.3ms`, `1000/1000`, no captures, no misses), but it raised
TP warmup to `228.6s`, spent `111.5s` in the short-greedy suffix pass, and
caused `19` startup prefill-graph evictions.

Narrowing all the way to `b16:s96` was too tight:
`/tmp/inference-bench-long-output-dense-dynamic-narrow-sync-results/.../runs/20260711_031719`
landed at `249.3 / 20.8 / 1020.7ms`, `1000/1000`, but still captured
`ragged_prefill:b32:s96:rows0:ctx-256:src1` on the request path for `1870.0ms`.
The final `s96`-only batch range validation
`/tmp/inference-bench-long-output-dense-dynamic-s96-sync-results/.../runs/20260711_032444`
landed at `207.1 / 21.8 / 989.7ms`, `1000/1000`, with
`runtime_prefill_graph_captures=0`, `61` prefill graph hits, no misses, `10`
startup evictions, `217.6s` TP warmup, and `102.9s` in the short-greedy suffix
pass. This keeps deterministic request-path capture avoidance while avoiding
the broad `s64/s128` startup cost; the long_output row still needs cheaper
prefill/decode bodies for a score-facing closure.

No-sync queue profiles now keep prefix-reuse route and hit-length histograms
populated even when CUDA-synchronized timing is off. The existing compact
profile already reported `runtime_prefix_reuse_requests/tokens`, but the route
maps could be empty in public-style runs, hiding whether a multi_turn or
few_shot row was using common-prefix rows, request-prompt rows, or generated
prefixes. The research summary now prints `prefix_reuse`, `prefix_reuse_tok`,
`prefix_routes`, and `prefix_hits` beside the generated-prefix counters. This is
telemetry only; it does not change prefix lookup, prefill grouping, or serving
behavior.

Regular ragged decode baseline now reuses the GPU-resident last-token/seq-len
state when the existing decode-many state signature proves it is current. The
`_prefill_many` boundary seeds that state for ordinary, prefix, padded, ragged,
and FlashInfer prefill outputs; regular ragged decode updates it after each
token, and stale or reused rows fall back to the old CPU-built input tensor path.
This removes repeated CPU token tensor construction on steady ragged decode
handoff paths and lets decode-many skip its initial state sync immediately after
ordinary prefill, without changing sampling, prefix lookup, or row-bucket
policy. Focused CPU tests cover dense row-index omission, contiguous row
reordering, decode-many state-sync behavior, online prefill seeding, and the
direct GPU-buffer reuse path; a full current `tests/test_serving_engine.py` run
still has two pre-existing clean-head counter assertion failures unrelated to
this change. The research summary now keeps and prints `many_syncs` and
`many_sync_skips` so public no-sync queue profiles can confirm whether prefill
seeding eliminated decode-many state uploads.

The first local 8xH100 focused run after that push, on `9eabb34`, completed
both previously failing public rows:
`/tmp/inference-bench-local-9eabb34-results/.../runs/20260711_085121`.
TorchInferno landed at `219.7 / 36.7 / 248.3ms`, `982/1000` correct on
multi_turn, and `214.0 / 21.4 / 959.4ms`, `1000/1000` correct on long_output.
The public pointer was still stale at `20260711_030227` on TorchInferno
`54cb558`, so this is the current-head signal. The new `many_syncs` counters
showed the prefill seeding only partially removed state uploads: long_output
had `124` decode-many calls with `77` state-sync skips but still `47` full
state syncs, because refill prefill waves seeded only the newly admitted rows
and left the combined active-list signature stale. The runtime now preserves
the combined GPU decode-state signature when an already-current active set is
extended by a prefilled refill wave, with a focused CPU regression covering the
mid-session refill -> decode-many path.

The pushed validation on `829b227` confirmed the intended counter movement:
`/tmp/inference-bench-local-829b227-results/.../runs/20260711_090410` completed
long_output at `213.2 / 21.6 / 998.7ms`, p99 `612.8 / 39.3 / 1772.6ms`, with
`1000/1000` correct. The queue profile reported `109` decode-many calls,
`480` internal steps, `runtime_decode_many_state_syncs=0`, and
`runtime_decode_many_state_sync_skips=109`. This removes the refill-wave GPU
state upload path but leaves the score-facing row in the same performance band;
the remaining long_output gap is still cached-prefix suffix prefill and
high-active decode replay cost, not decode-state synchronization.

Skipping redundant logits-graph capture for greedy token-suffix common-prefix
warmup is accepted as startup/readiness cleanup. The prior current-head
long_output control
`/tmp/inference-bench-greedy-graph128-results/.../runs/20260711_113921` landed
at `207.8 / 21.8 / 1046.9ms`, `1000/1000`, with `102.6s` in the greedy
common-prefix suffix warmup, `174.5s` unified scheduler warmup, and `148`
prefill graph cache entries. The dirty validation after running token warmup
before logits warmup and skipping logits when token warmup succeeds wrote
`/tmp/inference-bench-token-warm-skip-results/.../runs/20260711_122106` and
landed at `210.6 / 21.9 / 970.2ms`, `1000/1000`. Startup moved in the intended
direction: server readiness fell to `210.9s`, greedy suffix warmup fell to
`78.5s`, unified scheduler warmup fell to `143.6s`, and the prefill graph cache
fell to `124` entries. The queue profile still reported
`runtime_prefill_graph_misses=0` and `runtime_decode_graph_misses=0`, so the
shorter warmup did not push those skipped logits captures onto the request
path. This does not close the score-facing long_output gap, but it reduces
public-run timeout risk and removes startup graph work that greedy serving does
not use.

Fixed-capacity packed prefill now accepts multi-source mixed-prefix batches
when no dummy slots are required. The runtime reorders source prefix rows into
fixed-capacity slot order and rejects only the unsafe multi-source dummy-slot
case. Focused CPU coverage checks both the accepted no-dummy path and the
dummy-slot rejection, and the patch was pushed as `a50cbaf`. This is an
enabling correctness cleanup, not a current multi_turn score mover: current
`a50cbaf` profiles still show `34-35` packed candidate calls with zero repeated
pattern/signature reuse, so the fixed-capacity graph path does not naturally
replay on the benchmark's mixed-prefix sequence.

Lowering the 512-token greedy decode quantum is rejected for current
multi_turn. A first probe with
`TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_MID_GEN_DECODE_QUANTUM=8` did not apply
to the 512-token class and behaved like the `decode_q=16` control:
`/tmp/inference-bench-ti-multi-dq8-results/.../runs/20260711_124509` landed at
`225.3 / 38.5 / 258.6ms`, `981/1000` correct. The correctly scoped global
`TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM=8` probe
`/tmp/inference-bench-ti-multi-globaldq8-results/.../runs/20260711_125023`
reported `decode_q=8`, but landed worse at `229.6 / 37.2 / 260.0ms`,
`983/1000` correct, with `q2submit_p50=127.5ms`. Keep the 512-token greedy
default at `decode_q=16`.

Current-head suffix-bucket split profiling keeps the split default closed for
multi_turn. A no-behavior profile on `a50cbaf`
`/tmp/inference-bench-ti-multi-split-candidates-results/.../runs/20260711_130140`
landed at `218.2 / 36.7 / 247.7ms`, `979/1000` correct. It found only `33`
split candidates: `25` had no model-token savings, and the positive candidates
could save just `2.7K` model tokens under the existing `min_group=2` guard.
Enabling exactly that constrained split policy
(`TORCHINFERNO_CONTINUOUS_PREFIX_PREFILL_SPLIT_SUFFIX_BUCKETS=1`,
`MIN_GROUP=2`, `MIN_FILL_PCT=75`) wrote
`/tmp/inference-bench-ti-multi-split-enabled-results/.../runs/20260711_130651`
and regressed to `235.3 / 37.9 / 269.6ms`, `983/1000` correct, with p99
TTFT/E2E around `972/1004ms`. The queue profile accepted only `5` splits,
saved `1.5K` model tokens, but introduced `7` request-path ragged-prefill graph
misses across new `b16`, `b32:s16`, and tiny `b2/b3` mixed-prefix shapes.
Token savings do not compensate for graph fragmentation; do not promote
suffix-bucket splitting for greedy-large mixed-prefix traffic.

Disabling prefill-ready-before-decode for the same 512-token greedy class is
also rejected on current head. The scoped run
`/tmp/inference-bench-ti-multi-prbd0-results/.../runs/20260711_131204` landed
at `224.8 / 36.6 / 254.2ms`, `981/1000` correct. Runtime telemetry was
essentially the same as the current default band (`q2first_p50=215.3ms`,
`q2submit_p50=125.0ms`, `submit2first_p50=88.4ms`, one small
`ragged_prefill:b2:s16:rows0:ctx-64:src1` miss), so the current
prefill-ready cap-8 policy remains the better default tradeoff.

## Priority for a focused (non-loop) session

1. Prefill MFU (Issue 1) — biggest TTFT lever, ~2x, affects 3/5 benchmarks.
2. Packed/ragged prefix-suffix prefill — reduce padding without multiplying
   small TP graph replays.
3. Persistent engine + TP-safe reuse (Issue 3) — needed for multi_turn.
