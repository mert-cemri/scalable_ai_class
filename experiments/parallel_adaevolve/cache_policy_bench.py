"""
Cache policy bench (Tier 2.1 + Tier 2.3 demonstration).

Two questions:

1. **Priority-aware eviction beats LRU on long runs.** When the cache
   is undersized for the working set, vanilla vLLM's LRU eventually
   evicts the static system prefix in favor of one-shot user blocks,
   collapsing the hit rate. An AdaEvolve-aware policy that tags the
   system prefix with high priority should hold hit rate flat across
   the run.

2. **Persistent KV eliminates cold-start prefill.** A run that exports
   its system blocks at the end and re-imports them at the start of
   the next run skips the cold prefill entirely.

We drive both arms with a synthetic workload of `R` consecutive
"AdaEvolve-style" iterations under a tight `max_blocks` budget that
guarantees the cache spills.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from mock_vllm import MockVLLM, summarize_global

SYSTEM_PROMPT_TOKENS = 1200
PARENT_BODY_TOKENS = 380
GUIDANCE_TOKENS = 50
MAX_OUTPUT_TOKENS = 16


def _fixed_system_prompt() -> str:
    rng = random.Random(0)
    return " ".join(f"sys{rng.randint(0, 9999)}" for _ in range(SYSTEM_PROMPT_TOKENS))


def _user(parent_seed: int, guidance_seed: int) -> str:
    rng = random.Random(parent_seed)
    parent = " ".join(f"p{rng.randint(0, 99999)}" for _ in range(PARENT_BODY_TOKENS))
    rng2 = random.Random(guidance_seed)
    guidance = " ".join(f"g{rng2.randint(0, 999)}" for _ in range(GUIDANCE_TOKENS))
    return parent + "\n\n" + guidance


@dataclass
class HitTrace:
    """Per-iteration hit-rate and miss-block snapshot."""
    iteration: int
    hits: int
    shared: int
    misses: int
    cum_misses: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.shared + self.misses
        return (self.hits + self.shared) / max(1, total)


async def _drive(
    server: MockVLLM,
    R: int,
    seed: int,
    use_priority: bool,
) -> List[HitTrace]:
    sys_prompt = _fixed_system_prompt()
    rng = random.Random(seed)
    traces: List[HitTrace] = []
    last = (0, 0, 0)
    for it in range(R):
        u = _user(rng.randint(0, 1 << 30), rng.randint(0, 1 << 30))
        kwargs = {}
        if use_priority:
            kwargs["section_priorities"] = {"system": 3, "user": 1}
        await server.chat_completion(sys_prompt, u, MAX_OUTPUT_TOKENS, **kwargs)
        s = server.stats
        d_hits = s.cache_hits - last[0]
        d_shared = s.shared_hits - last[1]
        d_miss = s.misses - last[2]
        traces.append(HitTrace(
            iteration=it,
            hits=d_hits,
            shared=d_shared,
            misses=d_miss,
            cum_misses=s.misses,
        ))
        last = (s.cache_hits, s.shared_hits, s.misses)
    return traces


def _hit_rate_curve(traces: List[HitTrace], window: int = 10) -> List[float]:
    """Sliding-window hit rate to smooth single-iter noise."""
    out = []
    for i in range(len(traces)):
        lo = max(0, i - window + 1)
        chunk = traces[lo : i + 1]
        h = sum(t.hits + t.shared for t in chunk)
        n = sum(t.hits + t.shared + t.misses for t in chunk)
        out.append(h / max(1, n))
    return out


# ------------------------------------------------------------------
# Experiment 1: priority eviction vs LRU under cache pressure
# ------------------------------------------------------------------

async def exp_priority_eviction(R: int = 80, seeds: List[int] = (1, 2, 3)):
    print(f"\n=== Experiment 1: priority eviction vs LRU (R={R}, "
          f"undersized cache forces eviction) ===\n")
    # Each iteration adds ~25 user blocks. Set max_blocks so we evict from
    # iteration ~30 onward.
    max_blocks = 600

    rows = []
    for policy in ("lru", "priority"):
        for seed in seeds:
            srv = MockVLLM(max_blocks=max_blocks, eviction_policy=policy, seed=seed)
            traces = await _drive(srv, R, seed=seed, use_priority=(policy == "priority"))
            curve = _hit_rate_curve(traces)
            late = sum(curve[-20:]) / 20
            rows.append({
                "policy": policy,
                "seed": seed,
                "late_hit_rate": late,
                "total_misses": traces[-1].cum_misses,
                "evictions": srv.stats.evictions,
                "curve": curve,
            })
            print(f"  policy={policy:<8} seed={seed} late_hit_rate={late:.3f} "
                  f"misses_total={traces[-1].cum_misses} evict={srv.stats.evictions}")
    return rows


# ------------------------------------------------------------------
# Experiment 2: persistent KV across runs
# ------------------------------------------------------------------

async def exp_persistent_kv(R: int = 30, seeds: List[int] = (1, 2, 3)):
    print(f"\n=== Experiment 2: persistent KV across runs (R={R}) ===\n")
    rows = []
    for seed in seeds:
        # Run 1 — cold start, snapshot at end.
        srv1 = MockVLLM(max_blocks=10_000, eviction_policy="priority", seed=seed)
        traces1 = await _drive(srv1, R, seed=seed, use_priority=True)
        snapshot = srv1.export_warm_blocks(min_priority=3)

        # Run 2 — cold start (fresh cache).
        srv2_cold = MockVLLM(max_blocks=10_000, eviction_policy="priority", seed=seed + 100)
        traces2_cold = await _drive(srv2_cold, R, seed=seed + 100, use_priority=True)

        # Run 3 — preloaded cache (system blocks already warm).
        srv2_warm = MockVLLM(max_blocks=10_000, eviction_policy="priority", seed=seed + 100)
        loaded = srv2_warm.import_warm_blocks(snapshot)
        traces2_warm = await _drive(srv2_warm, R, seed=seed + 100, use_priority=True)

        first_iter_misses_cold = traces2_cold[0].misses
        first_iter_misses_warm = traces2_warm[0].misses
        total_misses_cold = traces2_cold[-1].cum_misses
        total_misses_warm = traces2_warm[-1].cum_misses
        rows.append({
            "seed": seed,
            "snapshot_blocks": len(snapshot),
            "loaded_blocks": loaded,
            "first_iter_misses_cold": first_iter_misses_cold,
            "first_iter_misses_warm": first_iter_misses_warm,
            "total_misses_cold": total_misses_cold,
            "total_misses_warm": total_misses_warm,
        })
        print(f"  seed={seed} snapshot={len(snapshot)}b loaded={loaded}b "
              f"iter0_misses cold={first_iter_misses_cold} warm={first_iter_misses_warm}  "
              f"total_misses cold={total_misses_cold} warm={total_misses_warm} "
              f"reduction={(1 - total_misses_warm/max(1,total_misses_cold))*100:.1f}%")
    return rows


# ------------------------------------------------------------------
# Experiment 3: Multi-tenant priority eviction
# ------------------------------------------------------------------
# This is the regime where AdaEvolve-aware eviction actually wins. Two
# tenants share one small cache. Tenant A is a "stable" workload — its
# system prefix is the same on every call, so LRU keeps the prefix hot.
# Tenant B is a "churn" workload — every call brings a brand-new prompt
# (e.g. lots of independent K=1 problems). Under LRU, B's churn evicts
# A's prefix; under priority eviction, A's prefix is pinned and B's
# blocks compete only against each other.

async def exp_multi_tenant(R: int = 60, seeds: List[int] = (1, 2, 3)):
    print(f"\n=== Experiment 3: multi-tenant LRU vs priority "
          f"(tenant A stable, tenant B churn, R={R}) ===\n")
    max_blocks = 800

    rows = []
    for policy in ("lru", "priority"):
        for seed in seeds:
            srv = MockVLLM(max_blocks=max_blocks, eviction_policy=policy, seed=seed)
            sys_a = _fixed_system_prompt()
            # Tenant B uses a different stable system prefix.
            rng = random.Random(7777)
            sys_b = " ".join(f"alt{rng.randint(0, 9999)}" for _ in range(SYSTEM_PROMPT_TOKENS))
            rng_a = random.Random(seed)
            rng_b = random.Random(seed + 5000)

            tenant_a_misses = 0
            tenant_b_misses = 0
            for it in range(R):
                # 2 calls/iter: 1 stable (tenant A), 1 churn (tenant B fresh prompts)
                kwargs_a = {"section_priorities": {"system": 3, "user": 1}} if policy == "priority" else {}
                kwargs_b = {"section_priorities": {"system": 3, "user": 1}} if policy == "priority" else {}
                miss_before = srv.stats.misses
                await srv.chat_completion(
                    sys_a,
                    _user(rng_a.randint(0, 1 << 30), rng_a.randint(0, 1 << 30)),
                    MAX_OUTPUT_TOKENS,
                    **kwargs_a,
                )
                tenant_a_misses += srv.stats.misses - miss_before
                miss_before = srv.stats.misses
                # Tenant B: fresh system prefix every call (simulating many small jobs)
                fresh_sys = " ".join(
                    f"j{rng_b.randint(0, 99999)}" for _ in range(SYSTEM_PROMPT_TOKENS)
                )
                await srv.chat_completion(
                    fresh_sys,
                    _user(rng_b.randint(0, 1 << 30), rng_b.randint(0, 1 << 30)),
                    MAX_OUTPUT_TOKENS,
                    **kwargs_b,
                )
                tenant_b_misses += srv.stats.misses - miss_before

            # Tenant A's "interesting" hit rate is computed only over its calls.
            rows.append({
                "policy": policy,
                "seed": seed,
                "tenant_a_misses": tenant_a_misses,
                "tenant_b_misses": tenant_b_misses,
                "total_misses": srv.stats.misses,
                "evictions": srv.stats.evictions,
                "peak_blocks": srv.stats.peak_blocks,
            })
            print(f"  policy={policy:<8} seed={seed} A_misses={tenant_a_misses} "
                  f"B_misses={tenant_b_misses} total={srv.stats.misses} "
                  f"evict={srv.stats.evictions}")
    return rows


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------

async def main():
    out_dir = Path(__file__).parent / "EXP_RESULTS"
    out_dir.mkdir(exist_ok=True)

    rows1 = await exp_priority_eviction()
    rows2 = await exp_persistent_kv()
    rows3 = await exp_multi_tenant()

    summary = {
        "priority_eviction": rows1,
        "persistent_kv": rows2,
        "multi_tenant": rows3,
    }
    (out_dir / "cache_policy_results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_dir / 'cache_policy_results.json'}")

    # Markdown summary tables.
    md = []
    md.append("# Cache policy bench (Tier 2.1 + Tier 2.3)\n\n")
    md.append("## Experiment 1: Priority-aware eviction vs LRU under cache pressure\n\n")
    md.append("Cache size is under-provisioned (`max_blocks=600`) so that "
              "after ~30 iterations the cache must evict to admit new blocks. "
              "The vanilla LRU policy treats system and user blocks equally. "
              "The priority policy pins the system prefix at priority=3 and "
              "user blocks at priority=1, evicting user blocks first.\n\n")
    md.append("| policy | seed | late hit rate (last 20 iters) | total misses | evictions |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    for r in rows1:
        md.append(f"| {r['policy']} | {r['seed']} | {r['late_hit_rate']:.3f} | "
                  f"{r['total_misses']} | {r['evictions']} |\n")
    # Aggregate
    by_policy = {}
    for r in rows1:
        by_policy.setdefault(r["policy"], []).append(r)
    md.append("\n**Aggregates:**\n\n")
    md.append("| policy | mean late hit rate | mean total misses | mean evictions |\n")
    md.append("|---|---:|---:|---:|\n")
    for p, rs in by_policy.items():
        md.append(f"| {p} | "
                  f"{sum(r['late_hit_rate'] for r in rs)/len(rs):.3f} | "
                  f"{sum(r['total_misses'] for r in rs)/len(rs):.0f} | "
                  f"{sum(r['evictions'] for r in rs)/len(rs):.0f} |\n")

    md.append("\n## Experiment 3: Multi-tenant LRU vs priority\n\n")
    md.append("Two tenants share one undersized cache (`max_blocks=800`). "
              "Tenant A reuses one stable system prefix (just like a single "
              "long AdaEvolve run); Tenant B's prefix changes every call "
              "(simulating many tiny jobs sharing the same vLLM endpoint). "
              "LRU treats both tenants identically and lets B's churn evict "
              "A's hot prefix. Priority eviction pins A's prefix at "
              "priority=3 so B's blocks compete only against each other.\n\n")
    md.append("| policy | seed | tenant A misses | tenant B misses | total | evictions |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for r in rows3:
        md.append(f"| {r['policy']} | {r['seed']} | {r['tenant_a_misses']} | "
                  f"{r['tenant_b_misses']} | {r['total_misses']} | {r['evictions']} |\n")
    by_policy3 = {}
    for r in rows3:
        by_policy3.setdefault(r["policy"], []).append(r)
    md.append("\n**Aggregates:**\n\n")
    md.append("| policy | mean tenant A misses | mean tenant B misses |\n|---|---:|---:|\n")
    for p, rs in by_policy3.items():
        md.append(f"| {p} | "
                  f"{sum(r['tenant_a_misses'] for r in rs)/len(rs):.0f} | "
                  f"{sum(r['tenant_b_misses'] for r in rs)/len(rs):.0f} |\n")

    md.append("\n## Experiment 2: Persistent KV across runs\n\n")
    md.append("Run 1 evolves; on shutdown it exports the high-priority "
              "(system) blocks. Run 2 starts a fresh cache; we compare cold "
              "(no preload) vs warm (snapshot reloaded) on the same workload.\n\n")
    md.append("| seed | snapshot blocks | loaded | iter-0 misses cold | iter-0 misses warm | total misses cold | total misses warm | reduction |\n")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows2:
        red = 100.0 * (1 - r["total_misses_warm"] / max(1, r["total_misses_cold"]))
        md.append(f"| {r['seed']} | {r['snapshot_blocks']} | {r['loaded_blocks']} | "
                  f"{r['first_iter_misses_cold']} | {r['first_iter_misses_warm']} | "
                  f"{r['total_misses_cold']} | {r['total_misses_warm']} | "
                  f"{red:.1f}% |\n")

    (out_dir / "cache_policy_results.md").write_text("".join(md))
    print(f"Wrote {out_dir / 'cache_policy_results.md'}")


if __name__ == "__main__":
    asyncio.run(main())
