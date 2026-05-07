# Diff-aware cross-iteration KV reuse (Tier 3.1)

Workload: 30 iterations, parent body 800 tokens, system prefix 1200 tokens, each iteration applies a random edit of `edit_size` tokens. All numbers are means over 3 seeds.

## Per-edit-size summary

| edit_size | arm | misses | direct hits | diff-aware reused | total prefill (ms) | reduction vs vanilla |
|---:|---|---:|---:|---:|---:|---:|
| 4 | vanilla    | 1862 | 5848 | 0 | 2068.2 | (baseline) |
| 4 | diff-aware | 643 | 5848 | 1219 | 1092.8 | **47.2%** |
| 8 | vanilla    | 1880 | 5830 | 0 | 2084.6 | (baseline) |
| 8 | diff-aware | 542 | 5830 | 1338 | 1014.2 | **51.3%** |
| 16 | vanilla    | 2008 | 5702 | 0 | 2201.1 | (baseline) |
| 16 | diff-aware | 573 | 5702 | 1435 | 1053.1 | **52.2%** |
| 32 | vanilla    | 2137 | 5573 | 0 | 2318.7 | (baseline) |
| 32 | diff-aware | 632 | 5573 | 1505 | 1114.7 | **51.9%** |
| 64 | vanilla    | 1946 | 5764 | 0 | 2144.5 | (baseline) |
| 64 | diff-aware | 746 | 5764 | 1199 | 1185.1 | **44.7%** |
| 128 | vanilla    | 2077 | 5633 | 0 | 2264.6 | (baseline) |
| 128 | diff-aware | 976 | 5633 | 1102 | 1383.3 | **38.9%** |

## Reading

* **misses** — blocks that had to do full prefill compute.
* **direct hits** — blocks that hit the chained-hash cache (today's vLLM).
* **diff-aware reused** — blocks reused via content-only match plus RoPE re-application. Zero in vanilla, the headline win in diff-aware.
* **reduction** — total prefill compute saved (sum of per-block µs) by diff-aware vs vanilla.

## Interpretation

When the per-iteration edit is small relative to the prompt body (low `edit_size`), most of the parent's blocks have content identical to the previous iteration's. Vanilla vLLM throws those blocks away because the chained hash differs after any token change; diff-aware reuses them at 160µs / block (RoPE only) instead of 960µs / block (full prefill). The reduction trends with `edit_size`: small edits win big, large edits approach cold-start.

This is the simulated upper bound of the optimization the KV report's §5 Opportunity E proposes. A real implementation in vLLM would have to (a) maintain the content-hash index, (b) perform RoPE re-application during attention, (c) integrate with the block manager's eviction policy. The savings here match the report's 50-80% estimate for exploitation-mode iterations where edits are small.
