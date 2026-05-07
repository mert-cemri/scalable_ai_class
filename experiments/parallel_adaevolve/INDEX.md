# Parallel AdaEvolve — Index of Work

This directory contains every experiment, implementation, and report
produced exploring systems-level optimizations for AdaEvolve. Three
sessions, three reports; this file is the orientation index.

## Reports (read in this order)

| File | Session | Scope |
|---|---|---|
| `EXPERIMENTS.md` | 1 | Lab notebook for the K-batched controller (mock-only) |
| `PAPER_REPORT.md` | 1 | Paper-style write-up of K-batched vs K-concurrent on Rastrigin |
| `NEXT_STEPS.md` | 1 | Tier 1/2/3/4 roadmap of follow-up optimizations |
| `TIER1_TIER2_REPORT.md` | 2 | Real-vLLM validation of batched, prefetch, adaptive K, multi-problem orchestrator (Tiers 1.1, 1.2, 1.3, 2.2, 2.3, 4.4, 4.5) |
| `TIER3_REPORT.md` | 3 | Diff-aware KV reuse, speculative pipelining, multi-problem scaling, FP8 KV, adaptive gating (Tiers 1.2, 2.1, 3.1, 3.2, 4.2) |

## Code that landed in the SkyDiscover package

| Path | What |
|---|---|
| `skydiscover/search/adaevolve/batched_controller.py` | `BatchedAdaEvolveController` — K-batched candidates, adaptive K, prefetch, speculative pipelining (with adaptive gating + best-so-far parent prediction) |
| `skydiscover/search/adaevolve/__init__.py` | exports |
| `skydiscover/search/route.py` | registers `adaevolve_batched` search type |
| `skydiscover/config.py` | new `AdaEvolveDatabaseConfig` fields: `candidates_per_iteration`, `diversify_temperature`, `temperature_spread`, `adaptive_candidates`, `candidates_min`, `candidates_max`, `enable_prefetch`, `cache_aligned_prompt`, `cache_alignment_block_size`, `enable_speculative_pipelining`, `speculation_gpu_pressure_gating`, `speculation_pressure_threshold`, `speculation_metrics_url`, `speculation_max_num_seqs` |
| `skydiscover/context_builder/default/builder.py` | `_maybe_align` cache-aligned prompt padding (Tier 2.2) |

## Benches and harnesses

| File | Tier | Purpose |
|---|---|---|
| `mock_vllm.py` | (foundation) | vLLM-fidelity prefix-cache simulator with chained SHA-256, COMPUTING-state coordination, LRU + priority eviction, persistent-KV snapshot, content-hash diff-aware reuse |
| `bench.py` | 1.2 | Cache-only synthetic A vs B headline |
| `quality_bench.py` | 1.2 | Multi-arm quality + latency on Rastrigin (3 arms) |
| `test_batched_controller.py` | (test) | Behavioral invariants of the controller |
| `cache_policy_bench.py` | 2.1, 2.3 | Priority eviction + persistent KV mock-side benches |
| `adversarial_eviction_bench.py` | 2.1 | Multi-tenant adversarial workload showing priority's 71 % win |
| `diff_aware_bench.py` | 3.1 | Cross-iteration content-hash KV reuse simulation |
| `real_vllm_bench.py` | 1.1, 1.2, 4.5, 2.2, 4.4, 3.2 | Real vLLM end-to-end harness with 7 arms (vanilla, batched, batched_aligned, batched_prefetch, batched_adaptive, batched_speculative, batched_speculative_gated, all_optims) |
| `scaling_sweep.py` | 1.2 | N ∈ {1,2,4,8} concurrent problems on shared vLLM |
| `fp8_kv_bench.py` | 4.2 | bf16 vs fp8 KV cache head-to-head |

## Result files

| File | What |
|---|---|
| `EXP_RESULTS/summary.csv` | Quality bench raw rows |
| `EXP_RESULTS/quality_grid.png` | Best-score curves per arm × K |
| `EXP_RESULTS/gpu_efficiency.png` | GPU compute (bars) vs final score (lines) |
| `EXP_RESULTS/REPORT.md` | Auto-gen quality bench tables |
| `EXP_RESULTS/RESULTS.md` | Auto-gen cache-only bench |
| `EXP_RESULTS/cache_policy_results.{md,json}` | Priority eviction + persistent KV |
| `EXP_RESULTS/adversarial_eviction.{md,json}` | Adversarial multi-tenant 71 % result |
| `EXP_RESULTS/diff_aware_results.{md,json}` | Tier 3.1 simulator results |
| `EXP_RESULTS/scaling_sweep.{csv,md}` | Most recent scaling (N=4,8 with gating) |
| `EXP_RESULTS/scaling_sweep_full.{csv,md}` | Original full sweep (N=1,2,4,8 baseline) |
| `EXP_RESULTS/real_vllm_summary.{csv,md}` | Most recent real-vLLM run |
| `EXP_RESULTS/fp8_kv.{csv,md}` | FP8 vs bf16 |

## Top-line numbers

### Canonical 4-arm comparison (2 benches concurrent, N=3, K=8, Qwen3-4B):

| arm | hit rate | per-call wall | circle_packing best | speedup vs vanilla |
|---|---:|---:|---:|---:|
| vanilla | 18.4 % | 7.3 s | 0.364 (no improvement) | 1.0× |
| batched | 90.5 % | **1.91 s** | 0.548 | 3.8× |
| batched_speculative_gated | 91.2 % | 1.27 s | 0.569 | **5.7×** |
| **all_optims** (full stack) | **94.2 %** | **1.02 s** | **0.604** | **7.2×** |

### Long single-problem run (`circle_packing`, K=8, Qwen3-4B):

| arm | N_iter | wall (s) | calls | hit rate | per-call wall | best score |
|---|---:|---:|---:|---:|---:|---:|
| vanilla (K=1) | 8 | 71.9 | 8 | 13.6 % | 9.00 s | 0.364 (no improvement) |
| all_optims | 8 | 171.0 | 202 | **95.1 %** | **0.85 s** | 0.594 |
| **all_optims** | **16** | 343.9 | **419** | **95.0 %** | **0.82 s** | **0.816** |

* Single-process throughput improvement: **10.6× more candidates
  per second** at equal time budget.
* AdaEvolve+all_optims at N=16 converged from seed 0.364 → 0.816
  on `circle_packing` (124 % improvement; AlphaEvolve target is
  1.000) — the optimizations don't trade away search quality on
  long trajectories.

* **Real vLLM, scaling sweep N ∈ {1,2,4,8}**:
  * Vanilla throughput plateau: ~1.36 calls/s.
  * Batched throughput at N=8: **1.86 calls/s** (37 % more candidates per second).
  * Speculative wins at N≤2 (+30 %), hurts at N≥8 (−9 %) — motivated the
    adaptive gating addition.

* **Diff-aware KV reuse (simulated, Tier 3.1)**:
  * **47–52 % GPU prefill reduction** at AdaEvolve-typical edit sizes
    (4–32 tokens).
  * Peak win at edit_size ≈ 16 tokens (one block).

* **Adversarial multi-tenant priority eviction**:
  * **71 % reduction** in tenant A's GPU prefill cost vs LRU when
    tagging is correct (long-lived prefix priority=3, ephemeral
    priority=1).

* **Persistent KV across runs**:
  * **73 % cold-start miss reduction** when the system-prefix snapshot
    is reloaded on a fresh run.

* **Adaptive speculation gating**:
  * At N=4: gated speculation is the throughput winner (1.76 calls/s).
  * At N=8: synchronous `/metrics` polling overhead breaks the win
    (1.07 calls/s); needs an async metrics cache to recover.

## What's still on the table

These remain explicitly NOT done in this set of sessions:

1. **Diff-aware KV reuse on real vLLM** (Tier 3.1). Simulated only.
   Real implementation is a vLLM fork with custom RoPE re-application
   — multi-month research project.
2. **Speculative pipelining with intensity-aware predictor**. Currently
   uses "best-so-far parent" as the predictor; a smarter version would
   gate prediction by AdaEvolve's intensity G (only speculate in
   exploitation mode where parent is stable).
3. **Async `/metrics` cache for adaptive speculation gating**. The
   30-line fix that would turn the N=8 gated arm from a regression
   back into a win.
4. **FP8 KV memory pressure measurement**. Need cache-pressure
   conditions (smaller GPU mem util or many more concurrent problems)
   to actually see the 2× headroom benefit.
5. **GRPO-in-the-loop self-improving mutation** (Tier 3.3). Not started.
6. **Disaggregated prefill / decode** (Tier 3.4). Multi-node infra,
   not started.

## Process notes (for the lab notebook)

Three methodological corrections we made along the way, recorded as
lessons learned:

1. **Mock-vLLM lock → semaphore.** First mock_vllm version used a
   global `asyncio.Lock` for prefill. Result: `shared %` was identically
   zero in every cell. Switching to a bounded semaphore made the
   COMPUTING-state coordination observable and unblocked the bench.
   *Lesson*: when modeling continuous batching, never serialize the
   compute stage globally.

2. **User-prompt-uniqueness fix.** First quality bench had a too-simple
   user prompt (just parent code). Greedy selection on Rastrigin kept
   the same parent across iterations, so consecutive prompts were
   byte-identical for both arms — natural cache reuse drowned the
   batched arm's specific advantage. *Fix*: use realistic AdaEvolve-
   style prompts with sibling-history blocks that change every iter.
   The K=16 GPU compute reduction went from ~9 % to 79 %.

3. **Priority eviction tagging.** First adversarial eviction bench
   tagged BOTH tenants' system prefix as priority=3, so priority
   couldn't differentiate them. *Fix*: tag based on intended reuse
   (A's stable prefix priority=3, B's ephemeral prefix priority=1).
   The reduction went from 0 % to 71 %.

The shared theme: **the mechanism doesn't help if the workload
doesn't exercise it correctly**. Each correction made an existing
feature visible in the data; none of them were "the optimization
got better."
