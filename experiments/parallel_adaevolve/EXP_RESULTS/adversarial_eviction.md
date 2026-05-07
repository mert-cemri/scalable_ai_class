# Adversarial multi-tenant priority eviction

Tenant A: stable long system prefix, runs once per `R_b+1` ticks. Tenant B: brand-new system prefix every tick. Cache is sized so under LRU, B's churn evicts A's prefix between A's calls, forcing A to cold-start every iteration.

## Tenant A miss profile by configuration

| R_b (B-per-A) | max_blocks | policy | A first-iter misses | A steady-iter misses | A total misses |
|---:|---:|---|---:|---:|---:|

_(Raw rows below; aggregation by mean.)_

| R_b | max_blocks | policy | seed | A first-iter | A steady | A total | evict |
|---:|---:|---|---:|---:|---:|---:|---:|
| 5 | 400 | lru | 1 | 175 | 175 | 1050 | 5900 |
| 5 | 400 | priority | 1 | 175 | 26 | 305 | 5155 |
| 5 | 400 | lru | 2 | 175 | 175 | 1050 | 5900 |
| 5 | 400 | priority | 2 | 175 | 26 | 305 | 5155 |
| 5 | 400 | lru | 3 | 175 | 175 | 1050 | 5900 |
| 5 | 400 | priority | 3 | 175 | 26 | 305 | 5155 |
| 10 | 400 | lru | 1 | 175 | 175 | 1050 | 11150 |
| 10 | 400 | priority | 1 | 175 | 26 | 305 | 10405 |
| 10 | 400 | lru | 2 | 175 | 175 | 1050 | 11150 |
| 10 | 400 | priority | 2 | 175 | 26 | 305 | 10405 |
| 10 | 400 | lru | 3 | 175 | 175 | 1050 | 11150 |
| 10 | 400 | priority | 3 | 175 | 26 | 305 | 10405 |
| 20 | 400 | lru | 1 | 175 | 175 | 1050 | 21650 |
| 20 | 400 | priority | 1 | 175 | 26 | 305 | 20905 |
| 20 | 400 | lru | 2 | 175 | 175 | 1050 | 21650 |
| 20 | 400 | priority | 2 | 175 | 26 | 305 | 20905 |
| 20 | 400 | lru | 3 | 175 | 175 | 1050 | 21650 |
| 20 | 400 | priority | 3 | 175 | 26 | 305 | 20905 |
| 10 | 1000 | lru | 1 | 175 | 175 | 1050 | 10550 |
| 10 | 1000 | priority | 1 | 175 | 26 | 305 | 9805 |
| 10 | 1000 | lru | 2 | 175 | 175 | 1050 | 10550 |
| 10 | 1000 | priority | 2 | 175 | 26 | 305 | 9805 |
| 10 | 1000 | lru | 3 | 175 | 175 | 1050 | 10550 |
| 10 | 1000 | priority | 3 | 175 | 26 | 305 | 9805 |

## Aggregated A-tenant misses (mean across 3 seeds)

| R_b | max_blocks | LRU A misses (mean) | priority A misses (mean) | reduction |
|---:|---:|---:|---:|---:|
| 5 | 400 | 1050 | 305 | **71.0%** |
| 10 | 400 | 1050 | 305 | **71.0%** |
| 20 | 400 | 1050 | 305 | **71.0%** |
| 10 | 1000 | 1050 | 305 | **71.0%** |

## Reading

* **A first-iter misses** — the cold-start cost for tenant A. Should be the same under both policies (cache is empty before the first call).
* **A steady-iter misses** — what tenant A pays *after* its prefix should be cached. The difference between LRU and priority is the load-bearing metric: LRU evicts under B's pressure, priority pins.
* **A total misses** — sum across all iters; this is the GPU cost A pays under each policy.
* **reduction** — priority's saving over LRU, on tenant A.
