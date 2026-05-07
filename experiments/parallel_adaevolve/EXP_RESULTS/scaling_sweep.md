# Multi-problem scaling sweep (Tier 1.2 in operational form)

Model: `Qwen/Qwen3-4B-Instruct-2507`. Each cohort runs `N_concurrent` copies of the math benches concurrently against ONE shared vLLM endpoint. K=8, N_iter=2, benches: circle_packing,signal_processing.

## Throughput vs concurrency

| N_concurrent | arm | wall (s) | calls | hit rate | throughput (calls/s) |
|---:|---|---:|---:|---:|---:|
| 8 | batched | 70.6 | 128 | 89.7% | 1.81 |
| 8 | batched_speculative | 151.5 | 251 | 92.8% | 1.66 |
| 8 | batched_speculative_gated | 139.6 | 144 | 89.7% | 1.03 |

## Reading

* **wall (s)** — time for the full cohort to complete. Lower is better.
* **calls** — total LLM calls across all N concurrent problems (includes batched K-fan-out and any speculative requests).
* **hit rate** — overall vLLM prefix-cache hit rate over the cohort, scraped from `/metrics`.
* **throughput** — calls/second the GPU sustained. The headline scaling number: how does throughput grow with N_concurrent?

## Scaling profile

If the GPU were perfectly parallel, throughput would scale linearly in N. Reality is sub-linear: at some N the GPU's prefill+decode bandwidth saturates and throughput plateaus. The interesting question is *at what N* the batched arms saturate vs the vanilla arm. The KV report's prediction: batched saturates at much higher N because each call uses ~5× less prefill compute, freeing the GPU for more concurrent work.
