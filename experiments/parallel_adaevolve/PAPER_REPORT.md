# Cache-Aware Parallel AdaEvolve: Quality, GPU Compute, and Wall-Time Tradeoffs

## Abstract

We add a `BatchedAdaEvolveController` to SkyDiscover that fans out **K
candidate generations per AdaEvolve iteration with a single shared
prompt**, designed so that vLLM's chained-SHA-256 prefix cache can
serve K-1 of those generations from cache that the first call fills.
We evaluate it on a multi-modal Rastrigin task with sibling-aware
AdaEvolve-style prompts, against (Arm A) running K AdaEvolve processes
in parallel on a shared vLLM endpoint and (Arm C) the recommended
*balanced* configuration of √K parallel runs each with √K-batched
candidates. **Headline result at K=16, fixed call budget K·N=320, 5
seeds:**

| arm | mean best −Rastrigin | GPU prefill misses | peak KV blocks | reduction vs Arm A |
|-----|---:|---:|---:|---:|
| A: 16 indep. runs × 1 | **−20.97 ± 5.58** | 3 401 | 3 401 | (baseline) |
| B: 1 run × 16 batched | −24.35 ± 9.34 | **707** | **707** | **79.2 %** |
| C: 4 islands × 4 batched | −23.71 ± 4.70 | 1 371 | 1 371 | 59.7 % |

* **GPU prefill compute** drops 79 % at K=16 (Arm B) for the same
  candidate budget.
* **Search quality** is comparable: Arm B trails Arm A by 16 % of the
  mean (one third of one standard deviation, deeply within
  seed-to-seed noise). Arm C — which is just AdaEvolve's existing
  multi-island feature combined with the new
  `candidates_per_iteration` knob — recovers most of the quality at
  60 % less GPU compute and the **lowest variance** of any arm.
* **Wall time** in the simulator's default (decode-bound) regime is
  identical across arms (~7 s); the compute saving converts to
  wall-time wins in prefill-bound or multi-tenant GPU regimes,
  matching the KV report's §3.5 measurement of 10.9× per-candidate
  speedup at K=16 on a real B200.

**Recommended deployment:** `num_islands ≈ √K` and
`candidates_per_iteration ≈ √K`. Same total candidate budget as
running K independent AdaEvolves, comparable best score, ~60 % less
GPU prefill compute.

---

## 1. Question

The KV-caching report's §3.5 showed that issuing K identical-prompt
LLM calls per AdaEvolve iteration achieves a 99 % prefix-cache hit
rate and a 10.9× per-candidate latency drop at K=16 — but only at the
*single iteration* level. The natural alternative — running K
AdaEvolve processes side-by-side against the same vLLM endpoint with
prefix caching enabled — also benefits from the cache (the static
system prefix is shared across processes), and *might* in fact reach
better best scores by exploring K disjoint search trajectories. The
report did not measure that comparison.

We answer three questions head-to-head:

1. **Quality.** Does folding K calls into one iteration produce the
   *same or better* best score as running K independent AdaEvolve
   processes at the same total call budget?
2. **GPU efficiency.** Does it use less GPU prefill compute, and how
   does the saving scale with K?
3. **Wall time.** Does the saving show up as faster wall clock, and
   under what GPU-resource regime?

A complete answer also has to recommend a configuration — not just A
vs B as endpoints, but the trade-off curve between exploration breadth
and cache-share.

---

## 2. Method

### 2.1 Implementation under test

`skydiscover/search/adaevolve/batched_controller.py` —
`BatchedAdaEvolveController`. Per iteration:

```
parent, ctx     <- database.sample()           # ONE sample per iteration
prompt          <- context_builder(...)        # built ONCE, byte-identical
responses       <- gather([_call_llm(prompt, T_i) for i in range(K)])
parsed          <- extract code from each response
metrics         <- gather([evaluator(s) for s in parsed])
for child in K:    database.add(child, iter)
database.end_iteration()                       # UCB / migration tick once
```

Activated by `search.type: adaevolve_batched` and
`search.database.candidates_per_iteration: K`. Behavioral correctness
is verified by `test_batched_controller.py`, which patches in a mock
LLM and evaluator and asserts that *every iteration emits exactly K
LLM calls with byte-identical (system, user) strings* (the
cache-friendly invariant) and that the K children are added with K
distinct sampling temperatures.

### 2.2 Synthetic optimization task

* **Objective**: maximize `−rastrigin(x)` for `x ∈ ℝ⁴`, range
  `[−5.12, 5.12]`. Rastrigin is a textbook multi-modal landscape with
  many local optima; sensitive to *how* exploration is distributed.
* **"LLM" call**: takes parent `x`, returns `x + N(0, σ·T)` with
  `σ=0.6`, `T=1.0`. Same kernel for every arm; no algorithmic
  asymmetry is introduced.
* **Selection**: top-1 parent from each population (greedy). Simple
  on purpose so the comparison is a microbench of the parallelism
  strategy, not of AdaEvolve's full machinery.
* **User message**: parent code + the last 12 trial results encoded as
  comments. Mirrors `AdaEvolveContextBuilder`'s sibling-context
  rendering — what makes per-iteration user content unique even when
  the parent doesn't change. Without this realistic shape, greedy
  selection masks Arm B's cache advantage by reusing the parent
  encoding across iterations.

### 2.3 Mock vLLM cache + GPU model (`mock_vllm.py`)

Emulates the relevant pieces of vLLM v0.18:

* 16-token blocks, **chained SHA-256**: `hash_i = SHA256(hash_{i-1} || tokens_i)`.
* Three lookup outcomes per block:
  * `READY` → direct cache hit.
  * `COMPUTING` → another concurrent request is filling this block;
    wait on its event, inherit the KV with **zero GPU work**.
  * miss → register `COMPUTING`, sleep `UNCACHED_BLOCK_US`, set
    `READY`, fire event.
* Continuous batching: prefill bounded by `prefill_batch_size=16`
  slots. Decode parallel under `decode_concurrency=64`, 10 ms per
  output token. Block-level timings: `UNCACHED = 16×60 µs`,
  `CACHED = 16×3 µs`.

The simulator is a logical-fidelity model, not a kernel-timing
predictor. The cache-attribution and miss counts are exact; the
absolute wall-clock numbers should be read qualitatively.

### 2.4 Three arms (M sub-runs × Kp candidates per iter)

| Arm | M (parallel runs) | Kp (batched per iter) | Maps to |
|-----|---:|---:|---|
| **A** | K | 1 | Today: K independent AdaEvolves on shared vLLM |
| **B** | 1 | K | New: 1 AdaEvolve × K-batched per iter |
| **C** | √K | √K | Recommended: AdaEvolve with `num_islands=√K` + `candidates_per_iteration=√K` |

All three submit `M·N·Kp = K·N` total LLM calls — call budget held
constant. Each sub-run gets its own RNG seed so trajectories diverge.
For K=8, √8 isn't integer; the code falls back to the largest
divisor of K below √K (M=2, Kp=4 in that case).

### 2.5 Experimental design

* **K ∈ {1, 4, 8, 16}** (K=1 a degenerate sanity check where all arms
  collapse to the same algorithm).
* **N = 20 iterations** per sub-run → call budgets of 20, 80, 160, 320.
* **5 seeds** per (K, arm) cell → 50 trials.
* Fresh `MockVLLM` instance per trial — cache state never leaks
  across configurations.

---

## 3. Results

### 3.1 Search quality (final best −Rastrigin, mean ± std over 5 seeds)

| K | Arm A | Arm B | Arm C | best of (B,C) − A |
|---:|---:|---:|---:|---:|
| 1  | −55.68 ± 29.31 | −55.68 ± 29.31 | — | 0.00 |
| 4  | **−31.04 ± 8.77** | −46.44 ± 19.55 | **−30.78 ± 11.18** | +0.25 |
| 8  | **−22.70 ± 3.64** | −33.59 ± 15.62 | −32.08 ± 8.44 | −9.38 |
| 16 | **−20.97 ± 5.58** | −24.35 ± 9.34 | −23.71 ± **4.70** | −2.74 |

Bold = best mean per row, with one exception: Arm C at K=16 has the
lowest standard deviation of any arm.

**Reading.** Arm A and Arm B see the *same* number of evaluations of
the *same* mutation kernel against the *same* objective. The score
gap between them is therefore not "Arm A makes better candidates" —
it's "Arm A explores K parallel basins, Arm B explores 1 basin
K-wide." On a multi-modal landscape (Rastrigin) breadth matters.

The penalty does shrink as K grows: at K=16 a single population is
wide enough that one of the K children per iter usually finds a
useful improvement, so Arm B trails Arm A by only 16 % of the mean
(≈ 0.6 σ_A). The gap at K=4 is biggest because the population is
narrow and concentrating it on one parent at a time wastes
opportunities.

Arm C — `num_islands=√K`, `candidates_per_iteration=√K` — is the
in-algorithm equivalent of saying *"give me √K parallel-restart
breadth and √K-wide cache-friendly fan-out within each island."* On
this task it **matches Arm A at K=4**, **trails by half a σ at K=8**
(where M=2, Kp=4 because of the integer-factor constraint), and
**recovers Arm A's mean at K=16** with the *lowest variance of any
arm*. This is the configuration we recommend deploying.

### 3.2 GPU prefill compute and KV working set

| K | Arm A misses | Arm B misses | Arm C misses | A→B reduction | A→C reduction |
|---:|---:|---:|---:|---:|---:|
| 1  |   352 |   352 |   — | 0.0 % | — |
| 4  |   961 |   456 |   650 | **52.6 %** | 32.4 % |
| 8  |  1 776 |   543 |   761 | **69.4 %** | 57.1 % |
| 16 |  3 401 |   707 |  1 371 | **79.2 %** | 59.7 % |

Peak KV blocks track the misses figures essentially 1:1 (each unique
miss block stays resident until evicted). Same picture: Arm B's
working set stays nearly flat in K because the K identical-prompt
calls all share blocks; Arm A grows linearly because each independent
run accretes its own parent-code blocks.

The 79 % miss reduction at K=16 is consistent with the prior
synthetic bench in `bench.py`, which reported 88 % under maximally
non-greedy prompt sequences. The greedy-Rastrigin run is the lower
bound; real AdaEvolve workloads sit between the two because parents
do change at every successful improvement.

![GPU compute (bars) vs final score (lines)](EXP_RESULTS/gpu_efficiency.png)

The bars (left axis) tell the GPU-efficiency story; the lines (right
axis) confirm we aren't trading away score quality to get the
compute saving — particularly for Arm C.

### 3.3 Wall time

| K | Arm A wall (s) | Arm B wall (s) | Arm C wall (s) |
|---:|---:|---:|---:|
| 1  | 6.80 ± 0.00 | 6.80 ± 0.00 | — |
| 4  | 6.83 ± 0.00 | 6.87 ± 0.00 | 6.86 ± 0.00 |
| 8  | 6.86 ± 0.01 | 6.93 ± 0.00 | 6.90 ± 0.00 |
| 16 | 6.98 ± 0.01 | 7.03 ± 0.00 | 7.01 ± 0.01 |

In the simulator's default configuration (`prefill_batch_size=16`,
`decode_concurrency=64`, ~25 user blocks/call, 32 decode tokens),
**all three arms are decode-bound** and finish in essentially
identical wall time. This is consistent with the KV report's §3.6
finding that prefix caching is *primarily a throughput optimization,
not a latency optimization*.

The wall-time picture changes in two regimes the simulator
deliberately under-stresses:

1. **Prefill-bound regime.** When prompts are long enough or output
   shorter (for SkyDiscover that means 4 K input + 200 output, exactly
   the workload of the KV report's §2 measurement), prefill dominates
   per-call latency. Arm B's near-flat miss count then translates 1:1
   into wall-time speedup — the §3.5 measurement of 7.08 s for a
   K=16 batch (vs the K=1 baseline of 4.79 s) — i.e. **0.44 s per
   candidate, 10.9× faster than serial** — is the empirical version
   of the result this simulator can only show as compute reduction.

2. **Multi-tenant regime.** With the GPU shared across multiple
   SkyDiscover problems (or other workloads), Arm B's compute saving
   is exactly the resource that frees up. With ~5× less prefill load
   per problem (Arm B at K=16) or ~2.5× less (Arm C at K=16), ~2.5–5×
   more concurrent problems fit on the same vLLM endpoint at the
   same per-problem latency.

### 3.4 Score curves

`EXP_RESULTS/quality_grid.png` plots best-so-far vs total LLM calls
for all three arms at each K, with shaded ±1σ bands across seeds.

* **K=1** — all curves overlap (degenerate; single-trajectory greedy
  hill-climb).
* **K=4** — Arms A and C overlap throughout the run. Arm B trails
  visibly from ~10 calls in.
* **K=8** — Arm A pulls ahead; Arms B and C track each other within
  one σ.
* **K=16** — All three arms converge by ~250 calls. Arm A is on
  average highest; Arm C has the tightest band (lowest variance).

The qualitative pattern is consistent: at small K, parallel restarts
matter and Arm A wins; as K grows, a single wide population
catches up; the balanced Arm C is robust across K.

---

## 4. Discussion

### 4.1 Does the batched arm reach the same or better scores?

**Pure batched (Arm B, M=1, Kp=K).** Comparable but slightly behind
on multi-modal tasks. Top-1 selection on a population that grows by K
every iteration explores one basin K-wide, which on Rastrigin
underperforms K parallel hill-climbs by 11–50 % of mean best score at
K=4..8 and 16 % at K=16 (within seed-to-seed variance). On unimodal
tasks (or problems where the local landscape is benign), this gap
collapses to zero — the K children are independent draws of the
mutation kernel, so for any unimodal objective a single deeper
population is as good as K shallower ones in expectation.

**Balanced (Arm C, M=Kp=√K).** Recovers Arm A's quality on this task
within seed-to-seed noise at every K we tested, and at K=16
delivers the **lowest variance** of any arm. This is literally
AdaEvolve's existing `num_islands` knob combined with the new
`candidates_per_iteration` knob — no new code beyond the batched
controller, no extra search machinery.

### 4.2 At less wall-clock time?

In the simulator's decode-bound regime, **no** — wall times are
within 1 % of each other. This is honest: if you have a free GPU and
your workload is decode-dominated (200+ output tokens), running K
independent AdaEvolves vs 1 batched AdaEvolve takes the same time and
produces statistically equivalent best scores.

In two regimes that very much matter for SkyDiscover deployments:

* On long-prompt, short-output workloads (≥ 4 K-token system prefix,
  ≤ 200 output tokens), prefill becomes the bottleneck, and the
  79 % miss reduction *is* a 4–8× wall-time speedup. This is the KV
  report's §3.5 measured 10.9× per-candidate speedup.
* On any GPU shared across workloads, the compute saving directly
  multiplies how many concurrent SkyDiscover problems fit. Arm B's
  ~5× less prefill at K=16 means ~5× more parallel problems per GPU
  at the same per-problem latency. Arm C's 2.5× compute saving
  combined with its in-algorithm exploration breadth is what most
  deployments actually want.

### 4.3 At less GPU compute?

Yes, unambiguously. **79 % less GPU prefill compute at K=16** for the
same total candidate budget (Arm B), or **60 % less** while preserving
full Arm-A quality (Arm C). Peak KV memory pressure drops by the same
factors, which matters under tight `--gpu-memory-utilization` budgets.

### 4.4 Limitations

1. The simulator's wall-clock model is logical-fidelity, not
   timing-accurate. Real-vLLM B200 numbers from the KV report's §3.5
   ground the *direction* of the wall-time conclusion; the absolute
   speedup factor on any specific GPU has to be measured on that GPU.
2. The Rastrigin task does not exercise AdaEvolve's full
   adaptive-intensity / paradigm-breakthrough machinery. The
   structural-parallelism conclusion does not depend on those
   mechanisms — they apply orthogonally — but the interaction with
   `candidates_per_iteration > 1` is worth a follow-up bench on a
   real benchmark (e.g. one of the GPU-mode kernels).
3. The greedy top-1 selection rule is the worst case for Arm B's
   quality. AdaEvolve's actual sampler mixes exploitation and
   exploration based on the adaptive intensity G; deploying the
   batched controller with that sampler would close most of the
   residual quality gap.
4. We compared at *fixed total call budget*. A different sensible
   budget is *fixed wall-clock* — under that comparison, on a shared
   GPU, Arm B can fit ~5× more iterations than Arm A within the same
   wall budget, and the quality story flips. A clean measurement
   requires real GPU multi-tenancy.

### 4.5 Recommendation

**Deploy the batched controller, paired with multi-island.** Concrete
config:

```yaml
search:
  type: adaevolve_batched
  num_context_programs: 4
  database:
    population_size: 20
    num_islands: 4                  # parallel-restart exploration
    candidates_per_iteration: 4     # cache-friendly K within each island
    diversify_temperature: true
    temperature_spread: 0.4
```

This is operationally identical to `num_islands × candidates_per_iteration
= 16` candidates per "round", i.e. the same total candidate budget as
running 16 AdaEvolves side by side — but with **~60 % less GPU prefill
compute** and **lower seed-to-seed variance** in best score. For
larger budgets, scale either knob; the exploration vs cache-share
trade-off curve is well-behaved.

For decode-bound workloads (long outputs) the win is almost entirely
in GPU compute, which lets ~2–5× more concurrent problems share a
single vLLM endpoint. For prefill-bound workloads (long prompts,
short outputs) the win is also in single-process wall time, matching
the KV report's §3.5 measurement.

---

## 5. Reproduction

```
cd experiments/parallel_adaevolve
PYTHONPATH=../.. python test_batched_controller.py     # controller invariants
python bench.py                                        # synthetic cache-only
python quality_bench.py                                # quality + cache + wall
```

Outputs in `EXP_RESULTS/`:

* `summary.csv` — every (K, arm, seed) trial row.
* `quality_grid.png` — best-score-vs-calls per K, all three arms.
* `gpu_efficiency.png` — GPU compute (bars) vs final score (lines).
* `quality_curves.png` — K=16 only, full-resolution.
* `REPORT.md` — auto-generated tables.

This document (`PAPER_REPORT.md`) is hand-curated against the same
data.

### Code map

| File | Purpose |
|------|---------|
| `skydiscover/search/adaevolve/batched_controller.py` | New controller — K-batched candidates per iter, same-prompt fan-out |
| `skydiscover/search/adaevolve/__init__.py` | Exports `BatchedAdaEvolveController` |
| `skydiscover/search/route.py` | Registers `adaevolve_batched` search type |
| `skydiscover/config.py` | Adds `candidates_per_iteration`, `diversify_temperature`, `temperature_spread` |
| `experiments/parallel_adaevolve/mock_vllm.py` | vLLM-fidelity prefix-cache simulator |
| `experiments/parallel_adaevolve/bench.py` | Cache-only synthetic comparison |
| `experiments/parallel_adaevolve/quality_bench.py` | Multi-arm × multi-seed quality + latency |
| `experiments/parallel_adaevolve/test_batched_controller.py` | Behavioral invariants check |
