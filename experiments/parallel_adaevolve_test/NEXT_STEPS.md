# Next Steps — Ambitious Systems Optimizations for AdaEvolve

This document proposes the next round of work, organized by tier.
Tier 1 = short-term wins that build directly on the K-batched
controller. Tier 2 = serious engineering investments with clear
expected payoff. Tier 3 = genuine research contributions, the kind
that would be papers in their own right.

For each proposal: motivation, sketch, expected impact, effort,
dependencies, and what an honest ablation looks like.

---

## Tier 1 — Compose / validate within ~2 weeks

These are short-arc projects that compose with `BatchedAdaEvolveController`
and convert its compute saving into measurable end-to-end wins.

### 1.1 Real-vLLM validation on a SkyDiscover benchmark

**Problem.** Everything we measured so far is in a logical-fidelity
simulator. The KV report's §3.5 gives one real-B200 number for K=16
batched, but no end-to-end best-score-vs-time curve through the full
SkyDiscover stack.

**Proposal.** Run `BatchedAdaEvolveController` on a real benchmark
(`benchmarks/math/circle_packing` and `benchmarks/gpu_mode/triangle_multiply`
make a good pair — one CPU-eval, one GPU-eval) against a real vLLM
endpoint with `--enable-prefix-caching`. Three configs:

| config | K | comment |
|---|---:|---|
| baseline | 1 | vanilla AdaEvolve |
| batched | 16 | `BatchedAdaEvolveController` |
| balanced | num_islands=4, candidates_per_iteration=4 | |

Measure per-iteration: prompt tokens, cache hit rate (from vLLM's
`/metrics`), GPU prefill time, decode time, evaluator time, iteration
wall time, best score.

**Expected impact.** Confirms the simulator's direction with a real
datapoint. Sets the baseline for everything below.

**Effort.** Couple of days once a B200 (or A100/H100) endpoint is
available. Mostly orchestration: spin up vLLM with prefix caching,
point the SkyDiscover config at it, run the three configs side by
side, scrape `/metrics` and the JSONL iteration logs.

**Dependencies.** GPU access, a hosted vLLM endpoint, a model
(Qwen-3-Coder or similar — the report uses Qwen3.5-27B-FP8).

**Ablation.** With and without prefix caching enabled; with and
without `inject_evaluator_context` (which controls whether the
huge static system prefix is present at all).

### 1.2 Multi-problem orchestration on a shared vLLM endpoint

**Problem.** The KV report's Opportunity B (run N=50–200 SkyDiscover
problems against one vLLM) and this work compose multiplicatively:
Arm-B problems each use ~1/9 the GPU prefill compute of Arm-A
problems at K=16, so ~9× more should fit on the same endpoint.
Nothing right now coordinates that fan-out.

**Proposal.** A small Ray-actor scheduler that holds a pool of N
problems, each running a `BatchedAdaEvolveController`, all submitting
to one shared vLLM endpoint. Two policies:

* **FIFO interleave** — submit each problem's K-batch as it becomes
  ready. Maximizes cache reuse across problems that share template
  prefixes. The KV report's §3.4 showed interleaved scheduling
  beats grouped scheduling 88.7 % to 0 % on hit rate when problems
  share prefixes.
* **Bandit-driven** — measure observed hit rate per problem;
  prioritize problems whose blocks are already hot in cache.

**Expected impact.** The throughput multiplier. Going from "one
SkyDiscover per GPU" to "5–10 SkyDiscovers per GPU" is the actual
deployment story.

**Effort.** ~1 week. Ray actors + a lightweight admission controller
+ metrics hookup.

**Dependencies.** Ray (or `asyncio.gather` over remote endpoints),
a deployed vLLM endpoint that survives many concurrent connections.

**Ablation.** N ∈ {1, 8, 32, 128}. With and without the bandit
priority. Steady-state per-problem latency, p99 latency, total
candidate throughput, GPU memory utilization.

### 1.3 Batched evaluation amortization

**Problem.** With K-batched candidates we now have K solutions to
evaluate per iteration. The evaluator currently runs them as K
independent tasks (gathered with `asyncio.gather`). For
container-based evaluators that means K Docker image pulls /
interpreter starts; for model-based judges that means K serial LLM-
as-judge calls.

**Proposal.** Two specific amortizations:

* *Container evaluators.* Keep one container alive per iteration,
  run all K candidates as separate processes inside it. Saves the
  ~hundreds-of-ms-to-seconds Docker spin-up per call.
* *LLM-as-judge.* Submit the K solutions as one batched judge
  prompt asking for K scores in a single response. Reuses the
  judge's system prefix exactly the same way we reuse the
  generator's, so the same prefix-cache argument applies in the
  evaluator stage too.

**Expected impact.** Eval is often the bottleneck after the LLM
saving lands. On `gpu_mode/triangle_multiply` for example, the
evaluator runs a Triton kernel benchmark — a few hundred ms per
call. K=16 of those is several seconds of wall time per iteration
that batching can compress.

**Effort.** Container piece: ~2 days (modify
`container_evaluator.py`'s lifecycle). Judge piece: ~3 days (new
`BatchedLLMJudge` that prompts for K-tuple of scores).

**Dependencies.** None beyond Tier 1.1.

**Ablation.** Per-iteration eval wall time at K=1, 4, 8, 16,
container vs no-container, batched judge vs serial judge.

---

## Tier 2 — Engineering investments (1–3 months)

These are substantial systems projects with clear expected wins but
significant implementation cost.

### 2.1 AdaEvolve-aware prefix-cache eviction

**Problem.** vLLM's LRU eviction is locality-blind. It evicts older
blocks first regardless of which blocks are about to be reused.
AdaEvolve's access pattern has very predictable structure:

* The system prefix (evaluator code) is **hot forever** — touched on
  every single LLM call.
* The parent code blocks are **hot within an island for ~5–10
  iterations**, then go cold when that island's UCB score drops.
* Sibling-history blocks are **hot within one iteration only** — the
  K calls of one iteration share them, but the next iteration's
  history is different.

vLLM's LRU treats all of these the same. It will gladly evict the
system prefix in favor of last iteration's stale sibling block.

**Proposal.** A custom block manager that exposes AdaEvolve's
priority hints to the cache:

* `cache_priority_metadata` field on each request: `"system"`,
  `"parent_island_<i>"`, or `"sibling_iter_<t>"`.
* Eviction policy that respects priority — evict `sibling` first,
  then cold parent islands by UCB score, then never the system
  prefix while it has a positive prior.

**Expected impact.** On long runs (hundreds of iterations), the
effective cache hit rate degrades as LRU evicts the system prefix
under memory pressure. A priority-aware cache should hold cache hit
rate flat across the run, lifting late-iteration throughput
significantly. Order of magnitude: 10–30 % effective hit rate
improvement on multi-hour runs.

**Effort.** ~3 weeks. Requires a vLLM fork or upstream PR; vLLM
exposes a `BlockManager` abstraction but adding metadata-aware
priority means touching the scheduler.

**Dependencies.** vLLM source access; a willing upstream
maintainer if the change is to be upstreamed.

**Ablation.** Hit rate over time across multi-hour runs; LRU vs
priority-aware vs ablated priority levels (system-only,
system+parent, full).

### 2.2 Token-aligned prompt layout

**Problem.** vLLM hashes 16-token blocks. If a prompt change
shifts the boundary by even one token, every block downstream of
the change misses cache. Right now AdaEvolve assembles prompts
without any awareness of where 16-token boundaries fall, so a
sibling-history update can shift the entire user-message portion's
alignment.

**Proposal.** A prompt-assembly layer that pads each *section* to
a multiple of `block_size = 16` tokens. System prefix → padded.
Parent code → padded. Sibling block → padded. Search-guidance
suffix → padded. With this, each section's blocks land at the same
hash positions across all calls regardless of how the other sections
change.

The padding is whitespace; vLLM hashes over tokens, so an
appropriately-tokenized whitespace pad is invisible to model
behavior.

**Expected impact.** Modest but mechanical. Should add 5–10 pp to
hit rate in mixed workloads where sections grow/shrink across
iterations. Works with all three deployment configs (single-process,
batched, multi-island).

**Effort.** ~1 week. Modify `DefaultContextBuilder` (and the
`AdaEvolveContextBuilder` that inherits from it) to pad before
emitting. Verify with a tokenizer that the pad doesn't break the
model.

**Dependencies.** None.

**Ablation.** Hit rate at various N and K; with vs without padding.

### 2.3 Persistent KV across runs

**Problem.** Cold-starting a SkyDiscover run re-prefills the entire
system prefix from scratch — usually 3000–4000 tokens of evaluator
code that never changes between runs of the same problem. That's
~100 ms of GPU prefill compute thrown away on every restart.

**Proposal.** Snapshot the KV blocks for the system prefix on run
shutdown to a per-benchmark cache directory:

```
~/.cache/skydiscover/kv/<benchmark_name>/<system_prefix_hash>.kv
```

On run startup, if a matching snapshot exists, send a
`/v1/cache/preload` request that mmap's it into the vLLM cache
before the first generation request. (vLLM v0.18 doesn't expose this;
this is an extension that ought to.)

**Expected impact.** First-iteration latency drops from "full system
prefill" to "0". For interactive use (running a benchmark once,
iterating on the evaluator) this is the biggest visible win. For
long-running batch jobs it's noise.

**Effort.** ~3 weeks because it requires a vLLM extension. The
on-disk KV format itself is straightforward (existing block
representation).

**Dependencies.** vLLM source access; agreement on a snapshot
format compatible with future vLLM versions.

**Ablation.** Cold-start time with vs without persistent KV across
all 9 benchmark suites.

### 2.4 GPU-aware co-scheduling of LLM and evaluator

**Problem.** Several SkyDiscover benchmarks have GPU evaluators —
KernelBench Triton kernels, gpu_mode kernels, image generation
evaluators. Right now LLM generation and evaluation contend for the
same GPU haphazardly: LLM waits, then eval runs, repeat.

**Proposal.** A scheduler that interleaves the two:

* While an evaluator is running on the GPU, queue the next iteration's
  LLM prefill (which is bandwidth-bound, can timeshare with
  compute-bound kernel benchmarks).
* While the LLM is decoding (low GPU utilization), start the
  evaluator's warmup pass.
* On separate GPUs, just do them in parallel.

**Expected impact.** 1.5–3× wall-clock speedup on GPU-eval
benchmarks; modest on CPU-eval ones.

**Effort.** ~3 weeks. Requires NVML to read GPU SM utilization,
plus careful priority/preemption tuning. Easy to get wrong (deadlock,
starvation).

**Dependencies.** None beyond hardware access.

**Ablation.** Per-iteration time on KernelBench / gpu_mode at K=1
and K=16, with vs without interleaving.

---

## Tier 3 — Research-grade contributions (3–12 months)

These are projects where the algorithmic novelty is publishable in
its own right. They each require a real vLLM patch (Tiers 2.1, 2.3,
above are also fork-required, but the algorithms here are the core
contribution, not the engineering).

### 3.1 Diff-aware cross-iteration KV reuse

This is the KV report's **Opportunity E**, called out as "the real
research opportunity" in §6.

**Problem.** vLLM's prefix cache invalidates every block downstream
of the first token change. Evolutionary mutations are *exactly* small
edits to the parent code: a few inserted lines, a swapped operator,
a renamed variable. The unchanged spans before AND after the edit
have stable KV states; vLLM throws all of the after-spans away
because the chained hash breaks.

**Proposal.** A diff-aware block manager:

1. When a generation request arrives, compute a token-level Myers
   diff between the new prompt and the most-recently cached prompt
   from the same "lineage" (parent program ID, conceptually).
2. Identify maximal unchanged spans before and after the edit.
3. For the prefix span: standard prefix cache hit.
4. **For the suffix span:** lift the cached KV blocks for those
   tokens, run a lightweight RoPE re-application (since position
   IDs shifted), reuse the attention cache for those tokens.
5. For the edited span: full prefill.

Step 4 is where the research is. It's not free — RoPE re-application
costs FLOPs — but the costs are O(unchanged_suffix × layers) instead
of O(unchanged_suffix × layers × hidden_dim²), which is a huge win.

There's a literature on this idea ("prefix caching with
non-prefix matches", "context KV reuse"); the gap is a clean
production implementation that integrates with vLLM's scheduler and
that handles the AdaEvolve workload specifically (where the
"lineage" is very strict — parent → child differ by a small diff,
typically).

**Expected impact.** The KV report estimates 50–80 % prefill
reduction on exploitation-mode iterations where parent edits are
small. Combined with K-batched candidates per iteration, the
effective per-candidate prefill cost drops to near zero.

**Effort.** 3–6 months of focused work. Requires a vLLM fork, deep
familiarity with PagedAttention internals, careful correctness
testing (RoPE re-application has subtle off-by-one risks), and
benchmarking across many models to characterize accuracy impact (if
any).

**Dependencies.** Sustained vLLM source access; a small ML team
willing to debug attention-layer accuracy issues.

**Ablation.** Prefill compute reduction vs edit size; output
quality (perplexity, downstream task accuracy) vs full-prefill
baseline; throughput on multi-iteration AdaEvolve runs.

**Why this is publishable.** No production inference engine
currently does substring-or-suffix KV reuse. The closest prior
work is Hydragen (sparse-attention reuse) and Cascade Inference
(cascade-attention reuse), neither of which target evolutionary
edit patterns. A clean implementation + a benchmark on a
SkyDiscover-style workload would slot well into MLSys / OSDI.

### 3.2 Speculative iteration pipelining

**Problem.** AdaEvolve currently waits for iteration t's evaluator
to finish before sampling iteration t+1's parent. The evaluator can
take minutes (containerized GPU kernel runs). All that time the
LLM generator is idle.

**Proposal.** Speculative pipelining:

1. As soon as iteration t's K LLM calls are issued, predict
   iteration t+1's parent: with high probability it's "the
   currently-best program in the database" (UCB picks aggressively
   when intensity is low). Generate iteration t+1's K candidates
   speculatively against that predicted parent.
2. When iteration t's evaluator finishes, *check* whether the
   predicted parent matches the actual parent the database would
   have sampled. If yes (which it will be 70–90 % of the time per
   AdaEvolve's own UCB measurements in `adaptation.py`), commit the
   speculative candidates. If no, discard them (a few hundred ms of
   wasted compute, not catastrophic).
3. Iteration t+1's eval starts immediately on the speculatively-
   generated candidates. Pipeline depth = 2 in the simple case;
   easily extensible to depth 3+.

**Expected impact.** On evaluator-bottleneck benchmarks (where the
evaluator dominates per-iteration time — KernelBench, ALE-Bench),
this hides the LLM generation behind the eval, halving wall-clock
per iteration when speculation accuracy is high. With K-batched
candidates underneath, the throughput compounds.

**Effort.** ~2 months. Requires a careful
"transactional" view of the AdaEvolve database: speculative children
held in a staging area, committed on speculation success, rolled back
on failure. The hardest part is making sure the database's UCB and
intensity adapters don't drift on rollback.

**Dependencies.** None beyond the existing code.

**Ablation.** Speculation accuracy as a function of intensity G
(low intensity → high accuracy, high intensity → low). Wall-clock
per iteration with vs without speculation. End-to-end best score
across the full run (must not regress).

**Why this is publishable.** Speculative execution in evolutionary
loops is unusual — there's a small literature on "speculative
search" in genetic algorithms but nothing that we know of in the
LLM-guided regime. The paper writes itself: prediction model,
accuracy analysis, throughput measurement, demonstration that the
search dynamics are preserved.

### 3.3 GRPO-in-the-loop self-improving mutation

**Problem.** AdaEvolve's LLM is a generic code model. It doesn't
get better at "mutation for this specific benchmark" over the
course of a run. The K-batched candidates per iteration are exactly
a GRPO group: K samples from the same prompt, K reward signals from
the evaluator. We're throwing this training signal away.

**Proposal.** Train a small LoRA adapter on top of the generation
LLM, online during the run:

1. After each iteration, compute group-normalized advantages: each
   of the K children has a reward = its evaluator score; advantages
   = score − mean_score normalized by std.
2. Use those advantages as PPO/GRPO targets to update the LoRA
   adapter via a per-iteration gradient step.
3. The adapter biases generation toward the kinds of mutations that
   improve the score *for this specific benchmark*, accumulating
   benchmark-specific knowledge as the run progresses.

The cache picture is unchanged — LoRA adapters live outside the KV
cache and don't affect prefix hashing. The GPU compute picture adds
a small training step per iteration (~10 % overhead at LoRA rank 8)
in exchange for steadily-improving sample quality.

**Expected impact.** Speculative but potentially huge: best score
on long runs improves significantly past the baseline because the
mutation distribution converges toward "things that worked on this
benchmark before." On benchmarks where the LLM has weak priors
(novel kernels, ARC-AGI-style tasks) this could be transformative.

**Effort.** 3–4 months. Requires real ML engineering — RL stability,
catastrophic-forgetting prevention (the adapter must not forget the
base coding ability), reward-normalization scheme robust to rare
big jumps.

**Dependencies.** Training-capable GPU pool, an LLM training
framework (Verl / OpenRLHF / TRL).

**Ablation.** Best score over time with vs without the adapter, at
each K. Adapter strength (LoRA rank) vs base-model accuracy on a
held-out coding benchmark (must not regress meaningfully).

**Why this is publishable.** The intersection of evolutionary
search and online RL fine-tuning is genuinely under-explored. The
fact that K-batched candidates are GRPO-shaped is a free signal
that's currently being discarded; closing that loop is a clean
contribution.

### 3.4 Disaggregated prefill engine

**Problem.** vLLM's prefill is bandwidth-bound and benefits from
big GPUs (many SMs); decode is more memory-bound and tolerates
smaller GPUs. Right now both phases run on the same GPU, which is
operationally simpler but suboptimal for a workload like ours where
prefill is hot (long shared prefixes) and decode is comparatively
small.

**Proposal.** Split the inference engine across two GPU pools:

* **Prefill pool.** Big-memory, high-bandwidth GPUs (B200, H100).
  Holds the full system-prefix KV cache — pre-computed and pinned.
  Receives only the *delta tokens* (parent code + sibling block) of
  each request, prefixes them onto the cached system KV, returns
  the resulting attention state.
* **Decode pool.** Cheaper GPUs (L40S, A100-40GB, even consumer-tier
  if model fits). Receives the prefix-prefilled state, runs decode
  to produce K outputs. Smaller KV footprint because only decode
  state needs to be resident.

This is a vLLM-internal extension, not a controller-level one;
the SkyDiscover side just sees a faster endpoint.

**Expected impact.** Cost per token of generation drops 2–3× because
the expensive prefill GPU is shared across many decode-only
clients. Throughput per dollar improves correspondingly. The KV
report's §3.5 measured K=16 batch latency at 7 s; with disagg, the
prefill cost amortizes over many concurrent decodes and effective
per-candidate cost approaches the decode-only floor.

**Effort.** 6+ months. This is the kind of project DeepSeek's
inference engine and SGLang have been investing in. Substantial
networking layer (RDMA/NVLink for state transfer), careful
fail-over.

**Dependencies.** Multi-node GPU infrastructure with fast
interconnect; sustained vLLM/SGLang fork maintenance.

**Ablation.** Latency, throughput, $-per-candidate on a K=16
batched workload, single-GPU baseline vs disagg with various
prefill:decode pool ratios.

**Why this is publishable.** Disaggregated inference is a hot
research area (DistServe @ OSDI'24, Splitwise @ ISCA'24).
SkyDiscover is a perfect motivating workload because the prefill
share is so dominant. A paper would slot well into MLSys / OSDI.

---

## Tier 4 — Smaller research bets

Quick proposals, in case any seem worth a deeper look:

### 4.1 Cross-island prompt deduplication
A pre-LLM-call hashing layer: when two islands' prompts at the
current round are byte-identical (or near-identical) — common after
migration, when an island just received a copy — merge them into one
batched call. Saves a redundant prefill. ~1 week.

### 4.2 Mixed-precision KV for cold blocks
The system prefix is hot but the parent-code blocks of inactive
islands are cold. Quantize cold blocks to FP8 / INT4 (a la KIVI,
KVQuant). Frees ~40 % KV memory at small accuracy cost. Composable
with everything in this document. ~2 months.

### 4.3 Parent-clustering for cache reuse
Before issuing the K LLM calls, cluster the population by parent
similarity. Issue calls for parents in cluster order. Cache hit rate
on the parent-code blocks goes up because consecutive calls share
prefix. Cheap (a single KMeans at iteration boundaries). Could give
10–20 % additional hit rate on long runs. ~1 week.

### 4.4 Adaptive K based on island productivity
AdaEvolve's intensity G already measures island productivity. Use it
to adapt `candidates_per_iteration` per island per iteration: high G
(productive island) → small K (one good child is enough); low G
(stagnating) → large K (more diverse exploration). Saves GPU compute
during exploitation phases. Composable with everything. ~2 weeks.

### 4.5 Streaming prefix prefetch
While iteration t's evaluator is running, send a `max_tokens=1`
"prefill-only" request for iteration t+1's predicted prompt to the
vLLM endpoint. vLLM treats it like a normal request, computes the
prefill, caches the blocks, then returns immediately. By the time
iteration t+1's real LLM call arrives, all its blocks are warm. The
KV report's §5 Opportunity F. ~2 weeks.

---

## Suggested next-six-weeks plan

If we had to pick three things to start tomorrow:

1. **Week 1–2: Tier 1.1 + 1.3** — real-vLLM validation on `circle_packing`
   and `triangle_multiply`, plus batched evaluation amortization. Gets
   the empirical foundation on solid ground and converts the
   simulator-only result into a defensible production claim.

2. **Week 3–4: Tier 1.2** — multi-problem orchestration on a shared
   vLLM endpoint. Demonstrates the throughput multiplier in
   practice, ~5–10× more concurrent SkyDiscover problems per GPU.

3. **Week 5–6: Tier 4.4 (adaptive K) + 4.5 (prefetch)** —
   small-but-impactful auxiliaries that compound with the existing
   batched controller. Both are < 2 weeks and need no fork.

Tier 3 projects (especially 3.1 diff-aware KV reuse and 3.2
speculative pipelining) require sustained vLLM-internals work and
should be scoped as research efforts, not as 1-week sprints.

---

## What success looks like

By the end of Tier 1: a production-ready
`adaevolve_batched + multi-problem orchestrator` deployment that
serves 5–10× more SkyDiscover problems per GPU than today, with a
real-benchmark validation report.

By the end of Tier 2: the deployed system has cache hit rates above
90 % on multi-hour runs (vs ~17 % today), tolerates restart-driven
warmups gracefully, and overlaps GPU-eval with LLM compute.

By the end of one Tier 3 project: a paper. The most defensible bet
is **3.1 diff-aware cross-iteration KV reuse**: the empirical
motivation is in the KV report already, the proposed mechanism is
well-defined, the workload is distinctive, and the prior art doesn't
cover it.
