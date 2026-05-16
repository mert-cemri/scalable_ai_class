# EvoScale: Accelerating LLM-Guided Evolutionary Search via Parallel Exploration

### Scalable AI Class Project — Group 7

Matteo Guarrera &nbsp;&nbsp; Elizabeth Polito &nbsp;&nbsp; Dongwei Lyu &nbsp;&nbsp; Mert Cemri &nbsp;&nbsp; Sultan Daniels &nbsp;&nbsp; Eric Xu

Check out our impact statement/blog post: [Accelerating LLM Guided Search with Parallel Exploration](https://evoscale-blog.pages.dev/)

This repository extends [SkyDiscover](https://arxiv.org/abs/2602.20133) to study the wall-clock scaling behavior of **AdaEvolve** with parallel execution:

1. **Candidate count (k-sweep)** — K>1 candidates per iteration (`candidates_per_iteration`)
2. **Speculative pipelining** — overlapping LLM generation with evaluation (`enable_speculative_pipelining`)
3. **KV cache efficiency** — prefix cache hit rates as k scales on a shared vLLM server
4. **vLLM server simulation** — preliminary explorations with a simulated vLLM server are under `experiments/parallel_adaevolve/`

Benchmarks used: **Circle Packing** (n=26) and **Signal Processing** from `benchmarks/math/`.

---

## Hardware & Model

| Component | Config |
|-----------|--------|
| GPU | NVIDIA H100, 183 GB |
| Model | `Qwen/Qwen3.5-27B-FP8` |
| Serving | vLLM v0.19.1 via Docker |
| Prefix caching | `--enable-prefix-caching` (ON for all runs) |
| Context | `--max-model-len 32768` |

---

## Reproducing Experiments

### 0. Environment setup

```bash
conda activate ScalableAI
export OPENAI_API_KEY="dummy"   # vLLM does not need a real key
```

### 1. Start vLLM servers

The k-sweep experiments use ports **8223–8226** (one GPU each). Start the 2-GPU pool:

```bash
bash start_vllm.sh
```

To check status / stream logs:

```bash
docker ps --filter name=vllm_gpu
docker logs -f vllm_gpu0
```

### 2. K-sweep runs (k = 1 … 8)

Each script runs **3 independent seeds** per benchmark, with a **2-hour timeout** per seed. Results land in `results/<benchmark>_k<k>_run<N>/`.

```bash
bash experiment_k1.sh      # k=1
bash experiment_k2.sh      # k=2
bash experiment_k3.sh      # k=4
bash experiment_k4.sh      # k=8
```

Each script invokes:

```bash
python -m skydiscover.cli \
    benchmarks/math/<task>/initial_program.py \
    benchmarks/math/<task>/evaluator[.py|/evaluator.py] \
    --config benchmarks/math/<task>/k_sweep/k<K>.yaml \
    --model vllm/Qwen/Qwen3.5-27B-FP8 \
    --api-base http://localhost:<PORT>/v1 \
    --output results/<tag> \
    --iterations 200
```

Config files are in `benchmarks/math/circle_packing/k_sweep/` and `benchmarks/math/signal_processing/k_sweep/` — `k<K>.yaml` sets `candidates_per_iteration: <K>`.

### 3. Speculative pipelining runs (k = 4, k = 8)

```bash
bash experiment_k3_spec.sh    # k=4  spec (circle_packing only)
bash experiment_k4_spec.sh    # k=8  spec (circle_packing only)

# The current speculation code already included prompt augumentation with current eval-missing candidates.
```

These use `k4_spec.yaml` / `k8_spec.yaml` which add:

```yaml
search:
  type: adaevolve_batched
  database:
    candidates_per_iteration: 8
    enable_speculative_pipelining: true
```

Results land in `results/circle_packing_k8_run<N>_spec/`. Each spec directory contains **two JSONL files**: the first (`20260511_*`) is the original run and the second (`20260514_*`) is a continuation with augmented children.

---

## Generating Plots

Install visualization dependencies first:

```bash
uv sync --extra math
pip install tiktoken   # for token-length analysis only
```

### K-sweep wall-clock and KV cache plots

```bash
python visualize_k_sweep.py
```

Outputs to `results/plots/`:

| File | Content |
|------|---------|
| `fixed_num_cand_k_sweep_wallclock.pdf/png` | Best score & candidates vs wall-clock time (4 subplots) |
| `fixed_num_cand_k_sweep_paper_grid.pdf` | Paper-ready 2×2 grid |
| `fixed_num_cand_k_sweep_kv_cache_hit_rate.pdf/png` | KV cache hit rate per candidate vs k |
| `fixed_num_cand_k_sweep_gen_time_per_candidate.pdf/png` | LLM generation time per candidate vs k; measured values (circle packing, mean over 3 runs): k=1: 65.6 s/cand → k=2: 37.1 s → k=4: 16.3 s → k=8: 14.0 s (4.7× faster per candidate vs k=1) |

Averaged final best scores (mean over 3 runs, 2-hour wall-clock budget):

| k | Circle Packing | Signal Processing |
|---|---------------|-------------------|
| 1 | 0.9947 | 0.590 |
| 2 | 0.9897 | 0.654 |
| 4 | 0.9957 | 0.674 |
| 8 | 0.9961 | 0.625 |

### Speculative pipelining comparison plots (k=4 vs k=8)

```bash
python visualize_k4_spec_comparison.py
```

Outputs to `results/plots/`:

| File | Content |
|------|---------|
| `k8_spec_wallclock_score.pdf` | Best score vs wall-clock: k=8 / k=8(spec) / k=8(spec, aug. cand.) |
| `k8_spec_wallclock_candidates.pdf` | Candidates generated vs wall-clock (same three curves) |
| `k8_spec_hit_vs_nohit_time.pdf` | Spec-hit vs non-hit iteration timing (2 columns: orig / aug. cand.) |
| `k4_spec_*.pdf` | Same plots for k=4 |
| `k4_k8_spec_llm_time_per_iter.pdf` | Per-iteration LLM time colored by spec-hit/miss |

---

## Code Structure

The main implementation lives under `skydiscover/search/adaevolve/`:

| File | Role |
|------|------|
| [`skydiscover/search/adaevolve/controller.py`](skydiscover/search/adaevolve/controller.py) | **AdaEvolve baseline** — standard single-candidate-per-iteration loop with adaptive sampling, UCB island rotation, paradigm breakthroughs, and sibling context |
| [`skydiscover/search/adaevolve/batched_controller.py`](skydiscover/search/adaevolve/batched_controller.py) | **Major implementation for this project** — extends the baseline to generate k candidates per iteration from a *byte-identical* prompt, maximizing vLLM prefix cache reuse; also implements speculative pipelining which overlaps LLM generation for the next iteration with the current evaluation |

---

## Results Directory Layout

```
results/
  circle_packing_k1_run1/
  circle_packing_k1_run2/
  circle_packing_k1_run3/
  ...
  circle_packing_k8_run1_spec/
    adaevolve_iteration_stats_20260511_*.jsonl   # original spec run
    adaevolve_iteration_stats_20260514_*.jsonl   # continuation w/ augmented children
    checkpoints/
  signal_processing_k1_run1/
  ...
  plots/                                         # all generated figures
```

Each run directory contains:
- `adaevolve_iteration_stats_<timestamp>.jsonl` — per-iteration stats including `global_best_score`, `iteration_time_seconds`, `llm_generation_time_seconds`, `speculation_hit`, and KV cache counters (`queries_delta`, `hits_delta`)
- `checkpoints/checkpoint_<N>/programs/*.json` — saved program snapshots with full prompt context

---

## Key Findings

- **KV cache hit rate** rises from ~17% at k=1 to ~80%+ at k=8 because repeated system prompts across the larger batch are increasingly cache-resident.
- **Speculative pipelining** overlaps LLM generation with evaluation; spec-hit iterations' llm generation time are ~2× faster than non-hit.
- **Best score** scales with wall-clock time roughly equally across k values when controlled for elapsed time, but k=8 reaches high-quality solutions faster in absolute time due to more candidates per second.

---

## Citation

```bibtex
@misc{2026evoscale,
  title  = {EvoScale: Parallel, Cache-Aware Evolutionary Algorithms},
  author = {Guarrera, Matteo and Polito, Elizabeth and Lyu, Dongwei and
            Cemri, Mert and Daniels, Sultan and Xu, Eric},
  year   = {2026},
  note   = {UC Berkeley, Scalable AI course project}
}
```
