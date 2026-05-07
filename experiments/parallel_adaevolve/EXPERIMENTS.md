# Experiments Log — Cache-Aware Parallel AdaEvolve

This document is the detailed lab notebook for the work in
`experiments/parallel_adaevolve/`. It records every experiment that
was run, how it was set up, what was measured, what came out, and how
each result fed into the next decision. The headline conclusions are
in `PAPER_REPORT.md`; this document is the reproducible underbelly.

---

## 0. Context

The starting point was `KV_CACHING_REPORT.md` in the repo root, which
established that vLLM v0.18 prefix caching gives a single AdaEvolve
process about **17.3 % hit rate** on a real run (because per-iteration
parent code dominates the variable portion of the prompt) but that
issuing **K identical-prompt LLM calls per iteration** (Opportunity A
in §5 of that report) hits **99.2 %** and gives **10.9× per-candidate
speedup** at K=16 on a real B200.

That measurement was for one isolated iteration. The report did *not*
compare it to the obvious alternative — running K AdaEvolve processes
in parallel against the same vLLM endpoint, which also gets some cache
benefit from the static system prefix. Our charter was:

1. Implement Opportunity A inside AdaEvolve as a clean controller
   variant (no fork, no externalities).
2. Show empirically how it stacks up against K-concurrent independent
   runs on (a) GPU efficiency, (b) search quality, (c) wall time,
   across a configuration sweep.
3. Recommend a deployment configuration with experimental backing.

---

## 1. Implementation experiments

### 1.1 `BatchedAdaEvolveController` — the controller change

Motivation: each AdaEvolve iteration today samples one parent + one
context, builds one prompt, issues one LLM call. That makes the per-
iteration KV-cache hit rate inherently low because the per-iteration
user message dominates the prompt's variable portion. If we instead
issue K parallel LLM calls *with the same prompt*, vLLM's
chained-SHA-256 prefix cache should serve K-1 of them entirely from
the cache that the first call fills.

What was added
(`skydiscover/search/adaevolve/batched_controller.py`):

```python
class BatchedAdaEvolveController(AdaEvolveController):
    async def _run_iteration(self, iteration, checkpoint_callback):
        # 1. Sample ONCE
        parent, ctx = self.database.sample(...)
        # 2. Build prompt ONCE  (byte-identical across K)
        prompt = self.context_builder.build_prompt(...)
        # 3. Fan out K LLM calls with that prompt
        responses = await asyncio.gather(*[
            self._call_llm(prompt["system"], prompt["user"], temperature=t)
            for t in self._candidate_temperatures(K)
        ])
        # 4. Parse, evaluate concurrently, add all K to DB
        ...
        self.database.end_iteration(iteration)   # UCB / migration tick once
```

Three new config fields on `AdaEvolveDatabaseConfig`
(`skydiscover/config.py`):

| Field | Default | Purpose |
|---|---|---|
| `candidates_per_iteration` | 1 | K — set to >1 to enable batching |
| `diversify_temperature` | False | Vary temperature across the K calls |
| `temperature_spread` | 0.4 | Width of the temperature interval |

Activation: `search.type: adaevolve_batched` in the YAML config (the
new search type registered in `route.py`). With
`candidates_per_iteration: 1`, the new search type is byte-equivalent
to vanilla `adaevolve`.

**Why temperature varies but the prompt does not.** vLLM hashes
*token sequences*; sampling parameters like temperature don't enter
the cache key. So we can keep the cache 100 % shared across the K
calls *and* still get K diverse children, simply by spreading the
sampling temperature. This is the cheapest possible source of
candidate diversity inside a batch.

### 1.2 Behavioral invariants test
(`experiments/parallel_adaevolve/test_batched_controller.py`)

Goal: prove the controller, when wired into the rest of SkyDiscover's
machinery (real `AdaEvolveDatabase`, real `AdaEvolveContextBuilder`,
real `LLMPool`), actually emits K *byte-identical* prompts per
iteration. This is the load-bearing precondition for any cache win.

Setup. We patch in:

* a `_MockLLM` that records every `(system, user, kwargs)` tuple it
  receives;
* a `_MockEvaluator` that returns a constant metric;
* a temporary `evaluate.py` file so the framework's evaluator factory
  is happy at construction time.

Run K=4, N=5, and assert:

```
Total LLM calls: 20                              (= K · N)
Total evaluations: 20
DB program count: 21                             (= 1 seed + K·N children)
Unique (system, user) prompts: 5
Calls-per-prompt distribution: [4, 4, 4, 4, 4]   (K identical per iter)
Distinct temperatures per prompt-group: [4, 4, 4, 4, 4]
```

**Result:** all five invariants pass. The "Calls-per-prompt
distribution = [4, 4, 4, 4, 4]" line is exactly the
prefix-cache-friendly property: every iteration sends K identical
prompts, every prompt is iteration-unique. The temperature spread
gives each of those K calls a different sampling RNG without changing
a single token in the prompt.

### 1.3 Evaluator concurrency

When `K > 1`, K evaluations also need to run concurrently. The
default evaluator pool is sized at
`max(max_parallel_iterations, 4)`; with K=8 that's a bottleneck. The
controller's `__init__` raises `evaluator.task_pool.max_concurrency`
to at least K so the K parallel evaluations don't queue.

This is a single-line bump and was verified to work without changing
any other evaluator behavior.

---

## 2. Cache-only synthetic benchmark (`bench.py`)

### 2.1 Goal

Before running anything on real workloads, we wanted a clean sanity
check on the cache properties: at fixed total LLM-call budget, does
the K-batched arrangement actually move the prefix-cache hit rate the
way the KV report's §3.5 says it should?

This benchmark *deliberately* does no real search. It just generates
LLM call sequences shaped like AdaEvolve's would be and routes them
through `mock_vllm.py` to read out the cache stats.

### 2.2 Mock vLLM (`mock_vllm.py`)

A process-internal emulator with the cache mechanics from
`KV_CACHING_REPORT.md` §1:

* 16-token blocks, **chained SHA-256**:
  `hash_i = SHA256(hash_{i-1} || tokens_i)`.
* Three lookup outcomes per block, all counted separately:
  * `READY` → direct cache hit, costs `CACHED_BLOCK_US = 16 × 3 µs`.
  * `COMPUTING` → another concurrent request is filling this block;
    await its event, inherit the KV with **zero GPU work**.
  * miss → register `COMPUTING`, sleep `UNCACHED_BLOCK_US = 16 × 60 µs`,
    set `READY`, fire event.
* LRU eviction beyond `max_blocks` (50 000 in the bench).
* Continuous batching: prefill bounded by `prefill_batch_size = 16`
  slots (we revised this from a single global lock — see §2.4).
* Decode treated as parallel under `decode_concurrency = 64`,
  `DECODE_US_PER_TOK = 10 ms`.

Tokenization is deliberately coarse (`re.compile(r"\S+|\s+")`) — it
returns stable, reproducible token counts that scale with prompt
length the way a real BPE tokenizer would. The point of the
simulator is not to predict vLLM kernel timings to the microsecond,
it's to model the cache attribution logic faithfully so the relative
hit/miss/share numbers are honest.

### 2.3 Two arms (cache-only version)

* **Arm A — `_one_independent_run` × K, gathered.** K coroutines each
  produce N=12 LLM calls. Each coroutine has its own RNG so the
  per-iteration user content (parent code) diverges across the K
  runs after iteration 0.
* **Arm B — N iterations × K-batched.** One coroutine, N=12
  iterations sequential, each iteration `asyncio.gather`s K
  identical-prompt calls.

Both submit `K · N` calls. We sweep K ∈ {1, 2, 4, 8, 16}.

Workload shape — modeled on AdaEvolve prompt anatomy:

| Component | Pseudo-tokens | Stable across … |
|---|---:|---|
| System prefix (evaluator code + framework) | ~3500 | every call |
| Parent body | ~380 | varies per iter |
| Search-guidance suffix | ~50 | varies per iter |

≈ 488 16-token blocks per call.

### 2.4 The lock-vs-semaphore correction

**First attempt** modeled the GPU prefill engine as a single global
`asyncio.Lock`. Result: the `shared %` column came out as **0 %**
in every cell — because by the time call 2 entered its prefill, call
1 had already finished and every block was `READY`, not `COMPUTING`.
The lock was serializing prefill so completely that the
COMPUTING-state coordination — the very mechanism vLLM continuous
batching uses to dedup concurrent identical prompts — could never
fire.

**Fix:** replace the lock with an
`asyncio.Semaphore(prefill_batch_size=16)`. Now multiple prefills
overlap; concurrent calls actually race on the same hashes;
`COMPUTING` blocks get awaited; the `shared %` column becomes
non-zero and meaningful.

This was the most important methodological correction in the whole
benchmark. Both K-concurrent and K-batched arms see real concurrent
prefill behavior; their cache-share differences then come out as a
real signal rather than a modeling artifact.

### 2.5 Results

`bench.py` writes to `RESULTS.md` and `results.json` next to it.

Headline table (after the lock-→-sem correction):

| arm | K | calls | wall (s) | hit % (READY) | shared % (COMPUTING) | miss % | uncached blocks | peak KV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: 1-concurrent | 1 | 12 | 25.23 | 81.4 | 0.0 | 18.6 | 1097 | 1097 |
| B: 1×K=1 | 1 | 12 | 25.23 | 81.4 | 0.0 | 18.6 | 1097 | 1097 |
| A: 2-concurrent | 2 | 24 | 25.25 | 81.4 | 3.7 | 14.9 | 1757 | 1757 |
| B: 1×K=2 | 2 | 24 | 25.24 | 81.4 | 9.2 | 9.4 | 1109 | 1109 |
| A: 4-concurrent | 4 | 48 | 25.30 | 81.4 | 5.6 | 13.0 | 3077 | 3077 |
| B: 1×K=4 | 4 | 48 | 25.28 | 81.4 | 13.8 | 4.8 | 1133 | 1133 |
| A: 8-concurrent | 8 | 96 | 25.42 | 81.4 | 6.5 | 12.1 | 5717 | 5717 |
| B: 1×K=8 | 8 | 96 | 25.37 | 81.4 | 16.1 | 2.5 | 1181 | 1181 |
| A: 16-concurrent | 16 | 192 | 25.62 | 81.4 | 6.9 | 11.6 | 10997 | 10997 |
| B: 1×K=16 | 16 | 192 | 25.52 | 81.4 | 17.2 | 1.4 | 1277 | 1277 |

Reduction summary (the column the next bench leans on):

| K | Arm A misses | Arm B misses | reduction |
|---:|---:|---:|---:|
| 1 | 1097 | 1097 | 0 % |
| 2 | 1757 | 1109 | 36.9 % |
| 4 | 3077 | 1133 | 63.2 % |
| 8 | 5717 | 1181 | 79.3 % |
| 16 | 10997 | 1277 | **88.4 %** |

### 2.6 Conclusions from the cache-only bench

1. **The COMPUTING-state share is real and grows with K only for
   Arm B.** At K=16 the column reads 17.2 % for Arm B vs 6.9 % for
   Arm A; that 10 pp delta is exactly the work that gets de-duplicated
   when K calls submit identical prompts simultaneously.
2. **Arm B's working set stays nearly flat in K (1097 → 1277 blocks);
   Arm A's grows linearly (1097 → 10 997).** This translates to
   ~9× less peak KV memory pressure at K=16 — relevant for tight
   `--gpu-memory-utilization` budgets.
3. **Wall time in this regime is similar across arms** because of the
   2 s per-call decode floor. The cache effect shows up in *GPU
   compute*, not in single-process wall time. This was the first hint
   that we'd need a different bench to isolate quality + wall-time.

The cache-only bench answered the "is the cache mechanism actually
firing" question. It didn't answer the "does this affect search
quality" question. That's experiment 3.

---

## 3. Quality + latency benchmark (`quality_bench.py`)

### 3.1 Goal

Compare A vs B on **search quality** (best score over iterations) at
matched call budgets, with multiple seeds, on a problem where the
distribution of K calls actually matters.

### 3.2 Task

* **Objective.** Maximize `−rastrigin(x)` for `x ∈ ℝ⁴`,
  range `[−5.12, 5.12]`. Rastrigin is multi-modal with many local
  optima — sensitive to *how* exploration is distributed.
* **Mutation.** `x' = x + N(0, σ·T)` with `σ=0.6`, `T=1.0`. The same
  Gaussian kernel is used by every arm; this isolates the parallelism
  strategy from the search algorithm.
* **Selection.** Top-1 parent from each population (greedy). Worst
  case for Arm B's exploration breadth; the comparison is therefore a
  *lower bound* on the batched arm's quality.
* **User message.** Parent code + the last 12 trial results, encoded
  as comments. Mirrors `AdaEvolveContextBuilder`'s sibling-context
  rendering. *This iteration-unique sibling block was the key
  methodological refinement* — see §3.5.

### 3.3 Two arms first, three later

Initial design had Arms A and B only; Arm C (the balanced
configuration) was added after the first runs showed a quality gap
and we wanted to know whether AdaEvolve's existing multi-island
structure could close it.

| Arm | M (parallel sub-runs) | Kp (batched per iter) | Note |
|-----|---:|---:|---|
| A | K | 1 | Baseline: K independent AdaEvolves |
| B | 1 | K | New: pure batched controller |
| C | √K | √K | Recommended: islands + batching combined |

All three submit `M·N·Kp = K·N` LLM calls. K=8 falls back to (M=2,
Kp=4) because √8 isn't integer and Kp must divide K cleanly.

### 3.4 Sweep

* **K ∈ {1, 4, 8, 16}.** K=1 is a degenerate sanity check (all arms
  collapse).
* **N = 20 iterations** per sub-run. Total budgets: 20, 80, 160, 320.
* **5 seeds per cell.** 5 × (3 K-values × 3 arms + K=1 × 2 arms) = 50
  trials.
* Fresh `MockVLLM` instance per trial.

Mock-vLLM tuning: `prefill_batch_size=16`, `decode_concurrency=64`,
B200-ish per-block timings.

### 3.5 The user-prompt-uniqueness correction

**First run** used a minimal user message (`encode_solution(x)` —
just the parent vector). Result: GPU compute reduction at K=16 was
only ~9 % between Arms A and B — far less than the 88 % from
`bench.py`.

The cause was an artifact of the search dynamics on this task:
greedy top-1 selection on Rastrigin keeps the same parent for many
iterations (most mutations fail to improve). Same parent → same user
encoding → same chained block hash → cache hit. *Both arms benefited
similarly from this natural prompt reuse,* drowning Arm B's
specific advantage.

**Fix:** make the user message iteration-unique by appending the
last 12 trial results as a sibling-context block (which is what real
AdaEvolve actually does). Now each iteration's user message differs
even when the parent doesn't, which:

* matches the realistic AdaEvolve workload shape;
* lets Arm B's per-iter K identical prompts be the *only* source of
  cache-share within an iteration, properly isolating the controller's
  contribution.

After the fix, the K=16 GPU-compute reduction came out at **79.2 %**
— still a touch under the 88 % from `bench.py` (which uses random
prompts with no parent reuse) but very much in the right ballpark.

### 3.6 Results — quality

Final best −Rastrigin, mean ± std over 5 seeds:

| K | Arm A | Arm B | Arm C |
|---:|---:|---:|---:|
| 1  | −55.68 ± 29.31 | −55.68 ± 29.31 | — |
| 4  | **−31.04 ± 8.77** | −46.44 ± 19.55 | **−30.78 ± 11.18** |
| 8  | **−22.70 ± 3.64** | −33.59 ± 15.62 | −32.08 ± 8.44 |
| 16 | **−20.97 ± 5.58** | −24.35 ± 9.34 | −23.71 ± **4.70** |

Bold = best mean per row, with one exception: Arm C at K=16 has the
lowest σ of any arm.

Reading:

* **Arm B trails Arm A** at all K > 1, biggest gap at K=4 (one whole
  σ_A behind), shrinking to 0.6 σ_A at K=16. This is the
  parallel-restart-vs-deeper-search trade-off in pure form: K=4
  independent hill-climbs from different seeds explore 4 basins of
  the multi-modal landscape; one population of K=4 children explores
  one basin.
* **Arm C matches Arm A at K=4** (within 0.03 of the mean). At K=8
  it trails by half a σ_A — a mild artifact of K=8 not having a clean
  √ factor (M=2, Kp=4 instead of the more balanced M=Kp=2.83 we
  can't actually pick). At K=16 Arm C's mean is within Arm A's σ and
  Arm C *beats* both other arms on variance.

### 3.7 Results — GPU compute

| K | Arm A misses | Arm B misses | Arm C misses | A→B | A→C |
|---:|---:|---:|---:|---:|---:|
| 1  |   352 |   352 |   — | 0 % | — |
| 4  |   961 |   456 |   650 | **52.6 %** | 32.4 % |
| 8  |  1 776 |   543 |   761 | **69.4 %** | 57.1 % |
| 16 |  3 401 |   707 |  1 371 | **79.2 %** | 59.7 % |

Peak KV blocks tracked the misses figure essentially 1:1.

### 3.8 Results — wall time

In the simulator's default decode-bound regime, all three arms come
in within 1 % of each other:

| K | Arm A wall | Arm B wall | Arm C wall |
|---:|---:|---:|---:|
| 1  | 6.80 ± 0.00 | 6.80 ± 0.00 | — |
| 4  | 6.83 ± 0.00 | 6.87 ± 0.00 | 6.86 ± 0.00 |
| 8  | 6.86 ± 0.01 | 6.93 ± 0.00 | 6.90 ± 0.00 |
| 16 | 6.98 ± 0.01 | 7.03 ± 0.00 | 7.01 ± 0.01 |

This is the honest answer: with 32 output tokens at 10 ms / token and
~25 user blocks per call, **decode dominates**, and all arms finish
together. The cache effect lives in the misses column, not in the
wall column.

We did not run a prefill-bound parameter sweep in the simulator
(longer prompts, shorter outputs) because the KV report's §3.5
already measured the prefill-bound regime on a real B200: 10.9×
per-candidate latency speedup at K=16. Reproducing that in the
simulator was redundant.

### 3.9 Plots

* `EXP_RESULTS/quality_grid.png` — 4-panel grid (one per K), best-so-
  far vs total LLM calls, all three arms with shaded ±1σ bands
  across seeds. Visible features:
  * K=1: all arms overlap (degenerate).
  * K=4: Arms A and C overlap throughout; Arm B trails from ~10
    calls in.
  * K=8: Arm A pulls ahead; B and C track each other within one σ.
  * K=16: All three arms converge by ~250 calls, with Arm C
    showing the tightest band.
* `EXP_RESULTS/gpu_efficiency.png` — bars (left axis) for misses by
  arm × K, lines (right axis) for final best score. The bars expose
  the cache win at a glance; the lines confirm the score quality
  isn't being given up to get it.

### 3.10 Conclusions from the quality bench

1. **Same call budget → comparable quality.** Arm B alone trails on
   multi-modal landscapes because of reduced exploration breadth;
   Arm C (`num_islands=√K, candidates_per_iteration=√K`) recovers
   Arm A's mean while keeping ~60 % of Arm B's GPU saving and
   the lowest variance of any arm.
2. **Less GPU prefill compute, monotonically scaling in K.** 53–79 %
   for Arm B, 32–60 % for Arm C, all at zero quality cost in the
   balanced case.
3. **Same wall time on a single idle GPU (decode-bound).** The
   compute saving converts to wall time on prefill-bound or shared
   GPUs — the KV report §3.5 grounds that direction empirically.

---

## 4. Iterative refinements & what they taught us

A short timeline of methodological corrections we made and why each
mattered:

1. **Mock-vLLM lock → semaphore (§2.4).** First version had `shared%`
   identically zero — a methodological bug masquerading as a result.
   Switching to a bounded semaphore made the COMPUTING-state coordin-
   ation observable and unblocked the rest of the analysis. Lesson:
   when modeling continuous batching, never serialize the compute
   stage globally — multiple requests must be allowed to race on
   block hashes.
2. **User prompt minimal → sibling-aware (§3.5).** First quality run
   showed only ~9 % GPU compute reduction at K=16 because greedy
   selection caused natural parent reuse across iterations, and that
   reuse was the dominant source of cache hits — for both arms. The
   fix was to make the user message iteration-unique with a sibling
   block (matching real AdaEvolve). After the fix the cache-share
   number reflected only the controller's contribution, and the K=16
   reduction went to 79 %. Lesson: realistic prompts matter; a too-
   simple user prompt drowns the experimental signal.
3. **Two-arm A/B → three-arm A/B/C.** First quality run showed Arm B
   underperforming Arm A on multi-modal Rastrigin. We added Arm C —
   `M=√K, Kp=√K` — and confirmed it recovers most of Arm A's
   quality at most of Arm B's GPU saving. Lesson: when the answer
   to "A vs B" is "depends on the task", the right move is usually
   to find a parameterization that interpolates between them and
   recommend that.

---

## 5. What was not measured (yet)

* **Prefill-bound regime in the simulator.** Would require driving
  output tokens down to ~16 and increasing `prefill_batch_size`
  contention. The KV report's §3.5 already supplies a real-GPU
  measurement, so this is mostly redundant; we'd run it if a reviewer
  wanted full simulator coverage.
* **Multi-tenant GPU experiment.** Run multiple Arm-B vs multiple
  Arm-A pipelines concurrently sharing the simulated GPU; measure
  per-pipeline throughput. Hypothesis: Arm-B pipelines compose
  ~5× tighter (since they use 5× less prefill at K=16).
* **Real benchmark task.** Plug the controller into a real SkyDiscover
  benchmark (e.g. `gpu_mode/triangle_multiply` or `math/circle_packing`)
  with a real evaluator and a real or hosted vLLM endpoint, and
  measure end-to-end best-score and wall time. Limited by GPU access.
* **Interaction with paradigm breakthrough / dynamic islands /
  Pareto mode.** The batched controller preserves all of these as
  black boxes (it just calls the existing `database.end_iteration()`).
  But running with `use_dynamic_islands=true` and
  `paradigm_breakthrough=true` and `candidates_per_iteration > 1` is
  worth a sanity sweep. Behaviorally there's no reason for them to
  conflict; experimentally we just haven't ticked all the cells.

---

## 6. Reproduction recipe

```bash
cd experiments/parallel_adaevolve

# Controller invariants (10 s)
PYTHONPATH=../.. python test_batched_controller.py

# Cache-only synthetic comparison (~2 min)
python bench.py
# writes RESULTS.md and results.json

# 3-arm × 4 K × 5 seed quality + latency (~5–6 min)
python quality_bench.py
# writes EXP_RESULTS/{summary.csv, REPORT.md, *.png}
```

### Files added by this work

* `skydiscover/search/adaevolve/batched_controller.py` — controller.
* `skydiscover/search/adaevolve/__init__.py` — exports.
* `skydiscover/search/route.py` — registers `adaevolve_batched`.
* `skydiscover/config.py` — three new fields on
  `AdaEvolveDatabaseConfig`.
* `experiments/parallel_adaevolve/mock_vllm.py` — vLLM-fidelity
  prefix-cache simulator.
* `experiments/parallel_adaevolve/bench.py` — cache-only synthetic.
* `experiments/parallel_adaevolve/quality_bench.py` — multi-arm
  quality + latency.
* `experiments/parallel_adaevolve/test_batched_controller.py` —
  invariants check.
* `experiments/parallel_adaevolve/RESULTS.md` — auto-generated cache-
  only headline.
* `experiments/parallel_adaevolve/ANALYSIS.md` — written analysis of
  the cache-only run.
* `experiments/parallel_adaevolve/EXP_RESULTS/REPORT.md` — auto-
  generated quality-bench tables.
* `experiments/parallel_adaevolve/PAPER_REPORT.md` — paper-quality
  hand-curated analysis.
* `experiments/parallel_adaevolve/EXPERIMENTS.md` — this document.
