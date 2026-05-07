# Multi-problem scaling sweep (Tier 1.2 in operational form)

Model: `Qwen/Qwen3-4B-Instruct-2507`. Each cohort runs `N_concurrent` copies of the math benches concurrently against ONE shared vLLM endpoint. K=8, N_iter=2, benches: circle_packing,signal_processing.

## Throughput vs concurrency

| N_concurrent | arm | wall (s) | calls | hit rate | throughput (calls/s) |
|---:|---|---:|---:|---:|---:|
| 1 | vanilla | 16.2 | 2 | 31.6% | 0.12 |
| 1 | batched | 22.8 | 16 | 91.5% | 0.70 |
| 1 | batched_speculative | 26.1 | 24 | 97.3% | 0.92 |
| 2 | vanilla | 20.3 | 12 | 45.7% | 0.59 |
| 2 | batched | 29.6 | 32 | 90.1% | 1.08 |
| 2 | batched_speculative | 33.5 | 48 | 95.7% | 1.43 |
| 4 | vanilla | 27.4 | 23 | 23.7% | 0.84 |
| 4 | batched | 42.6 | 64 | 89.4% | 1.50 |
| 4 | batched_speculative | 59.8 | 99 | 94.8% | 1.66 |
| 8 | vanilla | 30.9 | 42 | 16.6% | 1.36 |
| 8 | batched | 68.9 | 128 | 90.4% | 1.86 |
| 8 | batched_speculative | 143.2 | 244 | 94.9% | 1.70 |

## Reading

* **wall (s)** — time for the full cohort to complete. Lower is better.
* **calls** — total LLM calls across all N concurrent problems (includes batched K-fan-out and any speculative requests).
* **hit rate** — overall vLLM prefix-cache hit rate over the cohort, scraped from `/metrics`.
* **throughput** — calls/second the GPU sustained. The headline scaling number: how does throughput grow with N_concurrent?

## Scaling profile

If the GPU were perfectly parallel, throughput would scale linearly in N. Reality is sub-linear: at some N the GPU's prefill+decode bandwidth saturates and throughput plateaus. The interesting question is *at what N* the batched arms saturate vs the vanilla arm. The KV report's prediction: batched saturates at much higher N because each call uses ~5× less prefill compute, freeing the GPU for more concurrent work.
