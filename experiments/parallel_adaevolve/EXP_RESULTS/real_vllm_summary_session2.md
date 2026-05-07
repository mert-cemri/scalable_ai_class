# Real-vLLM benchmark — multi-problem orchestration

Model: `Qwen/Qwen3-4B-Instruct-2507` on local vLLM. Per-bench iterations N=3. Benches: circle_packing.

## Per-arm summary (means across benchmarks)

| arm | K | mean wall (s) | mean best score | total prompt toks | total prefix-cache hits | hit rate |
|---|---:|---:|---:|---:|---:|---:|
| batched_speculative | 8 | 42.6 | 0.4424 | 197232 | 186016 | 94.3% |
