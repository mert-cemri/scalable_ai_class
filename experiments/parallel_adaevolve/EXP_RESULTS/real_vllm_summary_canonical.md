# Real-vLLM benchmark — multi-problem orchestration

Model: `Qwen/Qwen3-4B-Instruct-2507` on local vLLM. Per-bench iterations N=3. Benches: circle_packing, signal_processing.

## Per-arm summary (means across benchmarks)

| arm | K | mean wall (s) | mean best score | total prompt toks | total prefix-cache hits | hit rate |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 1 | 14.6 | 0.4316 | 16175 | 2976 | 18.4% |
| batched | 8 | 38.1 | 0.5341 | 178032 | 161136 | 90.5% |
| batched_speculative_gated | 8 | 45.7 | 0.5444 | 303424 | 276720 | 91.2% |
| all_optims | 8 | 59.2 | 0.5522 | 499769 | 470672 | 94.2% |
