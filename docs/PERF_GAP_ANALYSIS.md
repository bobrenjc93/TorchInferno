# TorchInferno vs vLLM/sglang — Performance Gap Analysis (Llama-3.1-70B, 8xH100)

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
