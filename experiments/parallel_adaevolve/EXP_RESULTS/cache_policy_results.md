# Cache policy bench (Tier 2.1 + Tier 2.3)

## Experiment 1: Priority-aware eviction vs LRU under cache pressure

Cache size is under-provisioned (`max_blocks=600`) so that after ~30 iterations the cache must evict to admit new blocks. The vanilla LRU policy treats system and user blocks equally. The priority policy pins the system prefix at priority=3 and user blocks at priority=1, evicting user blocks first.

| policy | seed | late hit rate (last 20 iters) | total misses | evictions |
|---|---:|---:|---:|---:|
| lru | 1 | 0.730 | 4549 | 3949 |
| lru | 2 | 0.730 | 4549 | 3949 |
| lru | 3 | 0.730 | 4549 | 3949 |
| priority | 1 | 0.730 | 4549 | 3949 |
| priority | 2 | 0.731 | 4548 | 3948 |
| priority | 3 | 0.730 | 4549 | 3949 |

**Aggregates:**

| policy | mean late hit rate | mean total misses | mean evictions |
|---|---:|---:|---:|
| lru | 0.730 | 4549 | 3949 |
| priority | 0.730 | 4549 | 3949 |

## Experiment 3: Multi-tenant LRU vs priority

Two tenants share one undersized cache (`max_blocks=800`). Tenant A reuses one stable system prefix (just like a single long AdaEvolve run); Tenant B's prefix changes every call (simulating many tiny jobs sharing the same vLLM endpoint). LRU treats both tenants identically and lets B's churn evict A's hot prefix. Priority eviction pins A's prefix at priority=3 so B's blocks compete only against each other.

| policy | seed | tenant A misses | tenant B misses | total | evictions |
|---|---:|---:|---:|---:|---:|
| lru | 1 | 3449 | 12240 | 15689 | 14889 |
| lru | 2 | 3449 | 12240 | 15689 | 14889 |
| lru | 3 | 3449 | 12240 | 15689 | 14889 |
| priority | 1 | 3449 | 12240 | 15689 | 14889 |
| priority | 2 | 3449 | 12240 | 15689 | 14889 |
| priority | 3 | 3449 | 12240 | 15689 | 14889 |

**Aggregates:**

| policy | mean tenant A misses | mean tenant B misses |
|---|---:|---:|
| lru | 3449 | 12240 |
| priority | 3449 | 12240 |

## Experiment 2: Persistent KV across runs

Run 1 evolves; on shutdown it exports the high-priority (system) blocks. Run 2 starts a fresh cache; we compare cold (no preload) vs warm (snapshot reloaded) on the same workload.

| seed | snapshot blocks | loaded | iter-0 misses cold | iter-0 misses warm | total misses cold | total misses warm | reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 179 | 179 | 204 | 55 | 1799 | 1650 | 8.3% |
| 2 | 179 | 179 | 204 | 55 | 1799 | 1650 | 8.3% |
| 3 | 179 | 179 | 204 | 55 | 1799 | 1650 | 8.3% |
