"""
Adversarial multi-tenant test for AdaEvolve-aware priority eviction.

The earlier `cache_policy_bench.py` showed null effect for priority
eviction — under our orderly interleaved access pattern, vLLM's stock
LRU correctly preserves the system prefix because it's touched every
iteration. This bench constructs the access pattern under which
priority *does* win:

  * Tenant A — long evaluator system prefix (~1200 tokens), but it
    runs INFREQUENTLY (1 call every R "ticks").
  * Tenant B — high-frequency churn workload, every tick brings a
    brand-new system prefix (e.g. many tiny one-shot jobs).

Cache size is tight: it fits ~A's prefix + a few rounds of B's
churn, but not all of B's history. Under LRU, B's high-frequency
churn rolls A's prefix out of the cache by the time A wakes up; A
pays the full cold-prefill cost. Under priority eviction, A's
prefix is pinned (priority=3) and A's calls hit warm cache.

This is the realistic scenario for a SkyDiscover deployment that
hosts one long-running AdaEvolve next to many small batch jobs on
the same vLLM endpoint.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from mock_vllm import MockVLLM

SYSTEM_PROMPT_TOKENS = 1200
PARENT_BODY_TOKENS = 200
MAX_OUTPUT_TOKENS = 16


def _stable_system_prompt() -> str:
    rng = random.Random(0)
    return " ".join(f"sysA{rng.randint(0, 9999)}" for _ in range(SYSTEM_PROMPT_TOKENS))


def _user(seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(f"u{rng.randint(0, 99999)}" for _ in range(PARENT_BODY_TOKENS))


@dataclass
class Result:
    policy: str
    seed: int
    a_calls: int
    a_misses: int
    b_calls: int
    b_misses: int
    total_misses: int
    evictions: int
    a_first_iter_misses: int  # iter-0 misses for tenant A (warm-up)
    a_steady_iter_misses: int  # mean iter misses for A after iter-0


async def _drive(R_b_per_a: int, total_a_calls: int, max_blocks: int,
                 policy: str, seed: int) -> Result:
    srv = MockVLLM(max_blocks=max_blocks, eviction_policy=policy, seed=seed)
    sys_a = _stable_system_prompt()
    rng_a = random.Random(seed)
    rng_b = random.Random(seed + 1000)
    a_misses_per_call: List[int] = []
    a_calls = 0
    b_calls = 0
    use_priority = (policy == "priority")
    for a_iter in range(total_a_calls):
        # R_b_per_a B-churn calls before each A call. B's prefix is
        # *known to be one-shot* — tag it as priority=1 (low) so it
        # doesn't compete with A's stable prefix for "system" priority.
        for _ in range(R_b_per_a):
            fresh_sys_b = " ".join(
                f"chB{rng_b.randint(0, 99999)}" for _ in range(SYSTEM_PROMPT_TOKENS)
            )
            kw = {"section_priorities": {"system": 1, "user": 1}} if use_priority else {}
            await srv.chat_completion(
                fresh_sys_b, _user(rng_b.randint(0, 1 << 30)),
                MAX_OUTPUT_TOKENS, **kw,
            )
            b_calls += 1
        before = srv.stats.misses
        # A's prefix is the long-lived stable prefix — priority=3.
        kw = {"section_priorities": {"system": 3, "user": 1}} if use_priority else {}
        await srv.chat_completion(
            sys_a, _user(rng_a.randint(0, 1 << 30)),
            MAX_OUTPUT_TOKENS, **kw,
        )
        a_misses_per_call.append(srv.stats.misses - before)
        a_calls += 1

    a_first = a_misses_per_call[0] if a_misses_per_call else 0
    a_steady = (
        sum(a_misses_per_call[1:]) / max(1, len(a_misses_per_call) - 1)
        if len(a_misses_per_call) > 1 else 0
    )
    a_total = sum(a_misses_per_call)
    b_total = srv.stats.misses - a_total
    return Result(
        policy=policy, seed=seed,
        a_calls=a_calls, a_misses=a_total,
        b_calls=b_calls, b_misses=b_total,
        total_misses=srv.stats.misses, evictions=srv.stats.evictions,
        a_first_iter_misses=a_first,
        a_steady_iter_misses=int(a_steady),
    )


async def main():
    out_dir = Path(__file__).parent / "EXP_RESULTS"
    out_dir.mkdir(exist_ok=True)

    rows: List[Result] = []
    # Smaller sweep — reduce wall-clock so we don't hit the 1200s timeout.
    configs = [
        (5,  400),   # B-to-A ratio 5:1, tight cache
        (10, 400),
        (20, 400),
        (10, 1000),  # bigger cache — should narrow the gap
    ]
    total_a_calls = 6

    print(f"\n=== Adversarial multi-tenant priority eviction "
          f"(total_a_calls={total_a_calls}) ===\n")
    md_rows = []
    for R_b, max_blocks in configs:
        for seed in (1, 2, 3):
            for policy in ("lru", "priority"):
                r = await _drive(R_b, total_a_calls, max_blocks, policy, seed)
                rows.append(r)
                tag = f"R_b={R_b:<2} max_blocks={max_blocks:<5}"
                print(f"  {tag} seed={seed} policy={policy:<8} "
                      f"A_first_miss={r.a_first_iter_misses:3d} "
                      f"A_steady_miss={r.a_steady_iter_misses:3d} "
                      f"A_total={r.a_misses:4d} evict={r.evictions}")
        # Aggregate per (R_b, max_blocks)
        cell = [r for r in rows if r.b_calls // r.a_calls == R_b]
        # No need; keep simple.

    # Aggregate by (R_b, max_blocks, policy)
    by = {}
    for r in rows:
        # derive R_b from b_calls/a_calls
        rb = r.b_calls // max(1, r.a_calls)
        key = (rb, r.policy)
        by.setdefault(key, []).append(r)
    md = ["# Adversarial multi-tenant priority eviction\n\n"]
    md.append("Tenant A: stable long system prefix, runs once per `R_b+1` "
              "ticks. Tenant B: brand-new system prefix every tick. Cache "
              "is sized so under LRU, B's churn evicts A's prefix between "
              "A's calls, forcing A to cold-start every iteration.\n\n")
    md.append("## Tenant A miss profile by configuration\n\n")
    md.append("| R_b (B-per-A) | max_blocks | policy | A first-iter misses | A steady-iter misses | A total misses |\n")
    md.append("|---:|---:|---|---:|---:|---:|\n")
    cfgs_seen = []
    for r in rows:
        rb = r.b_calls // max(1, r.a_calls)
        cfg = (rb, len(set([r.evictions])))  # placeholder
    # Simpler: just iterate configs and aggregate.
    for R_b, max_blocks in configs:
        for policy in ("lru", "priority"):
            cell = [r for r in rows
                    if r.b_calls // max(1, r.a_calls) == R_b
                    and r.policy == policy]
            # Filter by max_blocks too. We didn't carry it in Result, so
            # reconstruct: configs is sorted, just trust the order matches.
        # The above is messy; emit raw rows instead.
    md.append("\n_(Raw rows below; aggregation by mean.)_\n\n")
    md.append("| R_b | max_blocks | policy | seed | A first-iter | A steady | A total | evict |\n")
    md.append("|---:|---:|---|---:|---:|---:|---:|---:|\n")
    idx = 0
    for R_b, max_blocks in configs:
        for seed in (1, 2, 3):
            for policy in ("lru", "priority"):
                r = rows[idx]
                idx += 1
                md.append(f"| {R_b} | {max_blocks} | {policy} | {seed} | "
                          f"{r.a_first_iter_misses} | "
                          f"{r.a_steady_iter_misses} | "
                          f"{r.a_misses} | {r.evictions} |\n")

    md.append("\n## Aggregated A-tenant misses (mean across 3 seeds)\n\n")
    md.append("| R_b | max_blocks | LRU A misses (mean) | priority A misses (mean) | reduction |\n")
    md.append("|---:|---:|---:|---:|---:|\n")
    idx = 0
    for R_b, max_blocks in configs:
        lru_misses = []
        pri_misses = []
        for _ in (1, 2, 3):
            for policy in ("lru", "priority"):
                r = rows[idx]; idx += 1
                if policy == "lru":
                    lru_misses.append(r.a_misses)
                else:
                    pri_misses.append(r.a_misses)
        lru_m = sum(lru_misses)/len(lru_misses)
        pri_m = sum(pri_misses)/len(pri_misses)
        red = 100.0 * (1 - pri_m / max(1, lru_m))
        md.append(f"| {R_b} | {max_blocks} | {lru_m:.0f} | {pri_m:.0f} | "
                  f"**{red:.1f}%** |\n")

    md.append("\n## Reading\n\n")
    md.append(
        "* **A first-iter misses** — the cold-start cost for tenant A. "
        "Should be the same under both policies (cache is empty before "
        "the first call).\n"
        "* **A steady-iter misses** — what tenant A pays *after* its "
        "prefix should be cached. The difference between LRU and priority "
        "is the load-bearing metric: LRU evicts under B's pressure, "
        "priority pins.\n"
        "* **A total misses** — sum across all iters; this is the GPU "
        "cost A pays under each policy.\n"
        "* **reduction** — priority's saving over LRU, on tenant A.\n"
    )
    (out_dir / "adversarial_eviction.md").write_text("".join(md))
    (out_dir / "adversarial_eviction.json").write_text(
        json.dumps([r.__dict__ for r in rows], indent=2))
    print(f"\nWrote {out_dir / 'adversarial_eviction.md'}")
    print(f"Wrote {out_dir / 'adversarial_eviction.json'}")


if __name__ == "__main__":
    asyncio.run(main())
