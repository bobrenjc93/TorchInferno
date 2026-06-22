# TorchInferno vs vLLM/sglang — Performance Gap Analysis (Llama-3.1-70B, 8xH100)

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
  writing queue progress or result files on multi_turn. Keep the 1024 threshold.
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

## In-flight, validated-locally-but-NOT-benchmarked commits

The inference-bench harness has been frozen on `25260c0` for many hours, so these
are unmeasured against vllm/sglang:
- joint (batch,q) prefill graphs under a token budget,
- `max_active=128` decode batch + prefill/decode decoupling,
- single-request prefill via graph (245 → 51ms single-req TTFT, local),
- paged-cache crash guards (SDPA fallback, FI-eager write-bounds).

## Priority for a focused (non-loop) session

1. Unfreeze the benchmark harness so the in-flight commits get measured.
2. Prefill MFU (Issue 1) — biggest TTFT lever, ~2x, affects 3/5 benchmarks.
3. Chunked prefill interleaved with decode — lets early requests return fast.
4. Persistent engine + TP-safe reuse (Issue 3) — needed for multi_turn.
