# FP8 KV cache experiment

Model: `Qwen/Qwen3-4B-Instruct-2507`. Same workload through two vLLM endpoints — one with default KV dtype, one with `fp8`. Bench: `circle_packing`, K runs ∈ {1, 8}, N_iter=3.

| KV dtype | K | wall (s) | calls | hit rate | best score |
|---|---:|---:|---:|---:|---:|
| default-kv K=1 | 1 | 24.1 | 3 | 36.3% | 0.36423689449571406 |
| fp8-kv K=1 | 1 | 22.2 | 3 | 31.9% | 0.5180200317780426 |
| default-kv K=8 | 8 | 35.3 | 24 | 90.0% | 0.36423689449571406 |
| fp8-kv K=8 | 8 | 35.9 | 24 | 88.6% | 0.6471662294797754 |

## Interpretation

FP8 KV cache halves the per-block KV memory footprint vs the default bf16 dtype. With prefix caching this directly translates to ~2× more blocks fitting in the same KV memory budget — the operational win, not measured here, is that more concurrent AdaEvolve problems fit before eviction starts. Quality is read from the `best_score` column; prefix-cache hit rate should be unchanged (FP8 affects per-block storage, not block identity).
