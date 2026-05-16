# Parallel AdaEvolve: K-batched candidates vs K concurrent independent runs
Workload simulates SkyDiscover AdaEvolve prompts on a mock vLLM engine that emulates v0.18's chained-SHA-256, 16-token-block prefix cache. Both arms submit the *same total number of LLM calls* (K × N where N=12).
## Arm definitions
* **Arm A — K concurrent independent AdaEvolve runs.** K processes, each evolving its own population. Iteration loops are independent, so per-iteration prompts diverge after the first iteration. Models what `--max-parallel-iterations K` or running K SkyDiscover invocations side by side looks like to vLLM.
* **Arm B — one run × K-batched candidates per iteration.** Each iteration: sample one parent, build the prompt once, fan out K identical-prompt LLM calls. Implemented in `BatchedAdaEvolveController`.
## Headline numbers
| arm | K | calls | wall (s) | p50 (ms) | p99 (ms) | hit % | shared % | miss % | uncached blocks | peak KV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: 1-concurrent runs | 1 | 12 | 25.23 | 2063.0 | 2063.4 | 81.4% | 0.0% | 18.6% | 1097 | 1097 |
| B: 1 run x K=1 batched | 1 | 12 | 25.23 | 2062.8 | 2062.9 | 81.4% | 0.0% | 18.6% | 1097 | 1097 |
| A: 2-concurrent runs | 2 | 24 | 25.25 | 2063.9 | 2536.6 | 81.4% | 3.7% | 14.9% | 1757 | 1757 |
| B: 1 run x K=2 batched | 2 | 24 | 25.24 | 2063.2 | 2533.8 | 81.4% | 9.2% | 9.4% | 1109 | 1109 |
| A: 4-concurrent runs | 4 | 48 | 25.30 | 2068.0 | 2545.3 | 81.4% | 5.6% | 13.0% | 3077 | 3077 |
| B: 1 run x K=4 batched | 4 | 48 | 25.28 | 2064.9 | 2539.4 | 81.4% | 13.8% | 4.8% | 1133 | 1133 |
| A: 8-concurrent runs | 8 | 96 | 25.42 | 2076.8 | 2563.3 | 81.4% | 6.5% | 12.1% | 5717 | 5717 |
| B: 1 run x K=8 batched | 8 | 96 | 25.37 | 2068.7 | 2550.2 | 81.4% | 16.1% | 2.5% | 1181 | 1181 |
| A: 16-concurrent runs | 16 | 192 | 25.62 | 2092.2 | 2597.9 | 81.4% | 6.9% | 11.6% | 10997 | 10997 |
| B: 1 run x K=16 batched | 16 | 192 | 25.52 | 2075.9 | 2568.2 | 81.4% | 17.2% | 1.4% | 1277 | 1277 |

## Arm B vs Arm A speedup at fixed K
| K | Arm A wall (s) | Arm B wall (s) | speedup | Arm A miss blocks | Arm B miss blocks | prefill compute reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.23 | 25.23 | 1.00× | 1097 | 1097 | 0.0% |
| 2 | 25.25 | 25.24 | 1.00× | 1757 | 1109 | 36.9% |
| 4 | 25.30 | 25.28 | 1.00× | 3077 | 1133 | 63.2% |
| 8 | 25.42 | 25.37 | 1.00× | 5717 | 1181 | 79.3% |
| 16 | 25.62 | 25.52 | 1.00× | 10997 | 1277 | 88.4% |

## How to read these columns
* **hit %** — blocks served from a `READY` cache entry. Direct, fully-warm cache hit.
* **shared %** — blocks where another concurrent call was already computing the same hash; this call awaited that future and did zero GPU work. *This is the column that explodes for Arm B.*
* **miss %** — blocks this call had to compute itself. Every miss is `block_size × uncached_us_per_token` of GPU prefill.
* **uncached blocks** — total miss count summed across all K×N calls. Proxy for total GPU prefill compute. Lower is better.
* **peak KV** — high-water mark of distinct cached blocks; proxy for KV-cache memory pressure.

## Interpretation
At K=16 (N=12): Arm B finishes in **25.5s** vs Arm A's **25.6s** (**1.00×** wall-time speedup), with **17.2%** of blocks served via concurrent prefix sharing in Arm B vs **6.9%** in Arm A. Total uncached prefill blocks: **1277** (Arm B) vs **10997** (Arm A) — a **88.4%** reduction in GPU prefill compute for the same total candidate budget.

Why it works: in Arm B every iteration submits K identical prompts. The first one to grab the prefill lock fills the cache; the other K-1 find every block already `COMPUTING` or `READY`, await briefly, and then skip prefill entirely. Decode runs in parallel under continuous batching. In Arm A the K runs evolve independent parents so their user messages diverge after iteration 0; only the static system prefix shares.
