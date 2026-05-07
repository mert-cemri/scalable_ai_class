# Real-vLLM benchmark — multi-problem orchestration

Model: `Qwen/Qwen3-4B-Instruct-2507` on local vLLM. Per-bench iterations N=16. Benches: circle_packing.

## Per-arm summary (means across benchmarks)

| arm | K | mean wall (s) | mean best score | total prompt toks | total prefix-cache hits | hit rate |
|---|---:|---:|---:|---:|---:|---:|
| all_optims | 8 | 343.9 | 0.8159 | 1599549 | 1519008 | 95.0% |
