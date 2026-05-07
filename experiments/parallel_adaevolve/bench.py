"""
Head-to-head benchmark: K-concurrent independent AdaEvolve runs vs
1 AdaEvolve run × K-batched candidates per iteration.

Both arms submit the same total number of LLM calls (K * N) to a single
shared MockVLLM instance whose prefix-cache behavior matches what
KV_CACHING_REPORT.md describes for vLLM v0.18 on B200. We then compare:

  * Total wall-clock time
  * Per-call latency (p50/p99)
  * Aggregate prefix cache hit rate
  * Prefill GPU work (tokens that actually had to be computed)
  * Prefix-share fraction (blocks shared across calls)
  * KV-block working set

Workload modeling (mirrors AdaEvolve prompt anatomy):
  * SYSTEM prefix: a long evaluator code + framework instructions
    (default ~3500 "tokens" by our coarse tokenizer).
  * USER message: parent program text (varies per iteration) +
    short search-guidance suffix (sampling mode, paradigm flag, ...).

Arm A (concurrent_independent):
  * K logical AdaEvolve runs, each with its own RNG → diverging parent
    trajectories → fully different USER messages across runs.
  * Iterations within a run are sequential (parent depends on the
    previous winner). Across runs they're concurrent.

Arm B (batched_candidates):
  * 1 run; per iteration sample one parent, build prompt once, fan out
    K identical-prompt LLM calls. Iteration i+1 starts only after
    iteration i lands its candidates.
  * Total LLM calls = K * N, identical to arm A.

Both arms share the SAME MockVLLM instance per run (so the cache state
is fair), but we run them in *separate* MockVLLM instances so the
metrics are isolated.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from mock_vllm import MockVLLM, summarize_global

# ---- Workload params -------------------------------------------------------
SYSTEM_PROMPT_TOKENS = 3500   # large fixed evaluator + framework prefix
PARENT_BODY_TOKENS = 380      # parent code body, varies per iteration
GUIDANCE_TOKENS = 50          # search guidance suffix (mode / paradigm / siblings)
MAX_OUTPUT_TOKENS = 200       # decode length per call (kept smallish for sim speed)


def _fixed_system_prompt() -> str:
    """Stable system prompt (~SYSTEM_PROMPT_TOKENS pseudo-tokens long).

    The exact content doesn't matter; what matters is that it's identical
    across every call so it lives in the cache prefix.
    """
    rng = random.Random(0)
    words = []
    for i in range(SYSTEM_PROMPT_TOKENS):
        words.append(f"sys{rng.randint(0, 9999)}")
    return " ".join(words)


def _make_parent_body(seed: int) -> str:
    """Per-iteration parent program body — different bytes per iter+run."""
    rng = random.Random(seed)
    words = [f"p{rng.randint(0, 99999)}" for _ in range(PARENT_BODY_TOKENS)]
    return " ".join(words)


def _make_guidance(seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(f"g{rng.randint(0, 999)}" for _ in range(GUIDANCE_TOKENS))


def _make_user(parent_seed: int, guidance_seed: int) -> str:
    return _make_parent_body(parent_seed) + "\n\n" + _make_guidance(guidance_seed)


# ---- Arm A: K concurrent independent AdaEvolve runs ------------------------


async def _one_independent_run(
    server: MockVLLM, run_idx: int, n_iter: int, seed: int
) -> None:
    rng = random.Random(seed)
    sys_prompt = _fixed_system_prompt()
    # Each run's parent trajectory is its own RNG-driven sequence.
    for it in range(n_iter):
        parent_seed = rng.randint(0, 1 << 30)
        guidance_seed = rng.randint(0, 1 << 30)
        user = _make_user(parent_seed, guidance_seed)
        await server.chat_completion(sys_prompt, user, MAX_OUTPUT_TOKENS)


async def run_arm_a(K: int, N: int, server: MockVLLM) -> None:
    tasks = [
        _one_independent_run(server, k, N, seed=1000 + k * 17)
        for k in range(K)
    ]
    await asyncio.gather(*tasks)


# ---- Arm B: 1 run × K-batched candidates per iteration ---------------------


async def run_arm_b(K: int, N: int, server: MockVLLM) -> None:
    rng = random.Random(2025)
    sys_prompt = _fixed_system_prompt()
    for it in range(N):
        parent_seed = rng.randint(0, 1 << 30)
        guidance_seed = rng.randint(0, 1 << 30)
        user = _make_user(parent_seed, guidance_seed)
        # K identical-prompt fan-out (the BatchedAdaEvolveController behavior).
        await asyncio.gather(
            *[
                server.chat_completion(sys_prompt, user, MAX_OUTPUT_TOKENS)
                for _ in range(K)
            ]
        )


# ---- Driver ----------------------------------------------------------------


def prefix_share_pct(server: MockVLLM) -> Dict[str, float]:
    """How much of the prefill workload was de-duplicated by prefix sharing.

    Specifically, of total blocks looked up across all calls:
      * cache_hits   = served from cache (READY)
      * shared_hits  = de-duped by concurrent COMPUTING coordination
      * misses       = had to compute fresh

    Effective sharing = (cache_hits + shared_hits) / total_blocks_seen
    """
    s = server.stats
    total = max(1, s.total_blocks_seen)
    return {
        "ready_hit_pct": 100.0 * s.cache_hits / total,
        "concurrent_share_pct": 100.0 * s.shared_hits / total,
        "miss_pct": 100.0 * s.misses / total,
        "effective_share_pct": 100.0 * (s.cache_hits + s.shared_hits) / total,
    }


def per_iteration_signatures(server: MockVLLM, K: int) -> Dict[str, int]:
    """Counts unique prompt signatures observed (groups identical prompts)."""
    sigs = {}
    for c in server.stats.calls:
        sigs[c.prompt_signature] = sigs.get(c.prompt_signature, 0) + 1
    counts = list(sigs.values())
    return {
        "unique_prompts": len(sigs),
        "avg_calls_per_prompt": sum(counts) / max(1, len(counts)),
        "max_calls_per_prompt": max(counts) if counts else 0,
    }


@dataclass
class ArmResult:
    name: str
    K: int
    N: int
    wall_time_s: float
    per_call_latency_p50_ms: float
    per_call_latency_p99_ms: float
    direct_hit_rate: float
    effective_hit_rate: float
    misses: int
    cache_hits: int
    shared_hits: int
    total_blocks_seen: int
    peak_blocks: int
    unique_prompts: int
    summary: dict


async def run_one_arm(
    name: str, K: int, N: int, runner, max_blocks: int = 50_000
) -> ArmResult:
    server = MockVLLM(max_blocks=max_blocks)
    t0 = time.monotonic()
    await runner(K, N, server)
    wall = time.monotonic() - t0
    summ = summarize_global(server.stats)
    summ.update(prefix_share_pct(server))
    summ.update(per_iteration_signatures(server, K))
    return ArmResult(
        name=name,
        K=K,
        N=N,
        wall_time_s=wall,
        per_call_latency_p50_ms=summ["lat_p50_ms"],
        per_call_latency_p99_ms=summ["lat_p99_ms"],
        direct_hit_rate=summ["direct_hit_rate"],
        effective_hit_rate=summ["effective_hit_rate"],
        misses=int(summ["misses"]),
        cache_hits=int(summ["cache_hits"]),
        shared_hits=int(summ["shared_hits"]),
        total_blocks_seen=int(summ["total_blocks_seen"]),
        peak_blocks=int(summ["peak_blocks"]),
        unique_prompts=int(summ["unique_prompts"]),
        summary=summ,
    )


def fmt_pct(x: float) -> str:
    return f"{100*x:5.1f}%"


def print_table(rows: List[ArmResult]) -> None:
    print()
    print(
        f"{'arm':<28} {'K':>3} {'N':>3} {'calls':>7} {'wall(s)':>8} "
        f"{'p50ms':>7} {'p99ms':>7} {'hit%':>7} {'shared%':>8} {'miss%':>7} "
        f"{'GPU prefill blocks':>20} {'peak KV':>8}"
    )
    for r in rows:
        s = r.summary
        print(
            f"{r.name:<28} {r.K:>3} {r.N:>3} {r.K*r.N:>7} {r.wall_time_s:>8.2f} "
            f"{r.per_call_latency_p50_ms:>7.1f} {r.per_call_latency_p99_ms:>7.1f} "
            f"{s['ready_hit_pct']:>6.1f}% {s['concurrent_share_pct']:>7.1f}% "
            f"{s['miss_pct']:>6.1f}% "
            f"{r.misses:>20} {r.peak_blocks:>8}"
        )
    print()


async def main():
    KS = [1, 2, 4, 8, 16]
    N = 12  # iterations per AdaEvolve run

    rows: List[ArmResult] = []
    for K in KS:
        a = await run_one_arm(f"A: {K}-concurrent runs", K, N, run_arm_a)
        b = await run_one_arm(f"B: 1 run x K={K} batched", K, N, run_arm_b)
        rows.append(a)
        rows.append(b)

    print_table(rows)

    # Save JSON
    out_path = Path(__file__).parent / "results.json"
    with out_path.open("w") as f:
        json.dump([r.__dict__ for r in rows], f, default=lambda o: getattr(o, "__dict__", str(o)), indent=2)
    print(f"saved → {out_path}")

    # Save markdown
    md = build_report(rows, N)
    md_path = Path(__file__).parent / "RESULTS.md"
    md_path.write_text(md)
    print(f"saved → {md_path}")


def build_report(rows: List[ArmResult], N: int) -> str:
    lines: List[str] = []
    lines.append("# Parallel AdaEvolve: K-batched candidates vs K concurrent independent runs\n")
    lines.append(
        "Workload simulates SkyDiscover AdaEvolve prompts on a mock vLLM "
        "engine that emulates v0.18's chained-SHA-256, 16-token-block prefix "
        "cache. Both arms submit the *same total number of LLM calls* "
        f"(K × N where N={N}).\n"
    )
    lines.append("## Arm definitions\n")
    lines.append(
        "* **Arm A — K concurrent independent AdaEvolve runs.** K processes, "
        "each evolving its own population. Iteration loops are independent, "
        "so per-iteration prompts diverge after the first iteration. Models "
        "what `--max-parallel-iterations K` or running K SkyDiscover invocations "
        "side by side looks like to vLLM.\n"
    )
    lines.append(
        "* **Arm B — one run × K-batched candidates per iteration.** Each "
        "iteration: sample one parent, build the prompt once, fan out K "
        "identical-prompt LLM calls. Implemented in `BatchedAdaEvolveController`.\n"
    )
    lines.append("## Headline numbers\n")
    lines.append(
        "| arm | K | calls | wall (s) | p50 (ms) | p99 (ms) | hit % | shared % | miss % | uncached blocks | peak KV |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for r in rows:
        s = r.summary
        lines.append(
            f"| {r.name} | {r.K} | {r.K*r.N} | {r.wall_time_s:.2f} | "
            f"{r.per_call_latency_p50_ms:.1f} | {r.per_call_latency_p99_ms:.1f} | "
            f"{s['ready_hit_pct']:.1f}% | {s['concurrent_share_pct']:.1f}% | "
            f"{s['miss_pct']:.1f}% | {r.misses} | {r.peak_blocks} |\n"
        )

    # Pair up arm A vs arm B for each K and compute the speedup.
    lines.append("\n## Arm B vs Arm A speedup at fixed K\n")
    lines.append(
        "| K | Arm A wall (s) | Arm B wall (s) | speedup | Arm A miss blocks | Arm B miss blocks | prefill compute reduction |\n"
        "|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    by_k: Dict[int, List[ArmResult]] = {}
    for r in rows:
        by_k.setdefault(r.K, []).append(r)
    for K in sorted(by_k):
        a = next(r for r in by_k[K] if r.name.startswith("A:"))
        b = next(r for r in by_k[K] if r.name.startswith("B:"))
        speedup = a.wall_time_s / max(1e-6, b.wall_time_s)
        prefill_red = 100.0 * (1 - b.misses / max(1, a.misses))
        lines.append(
            f"| {K} | {a.wall_time_s:.2f} | {b.wall_time_s:.2f} | {speedup:.2f}× | "
            f"{a.misses} | {b.misses} | {prefill_red:.1f}% |\n"
        )

    lines.append("\n## How to read these columns\n")
    lines.append(
        "* **hit %** — blocks served from a `READY` cache entry. Direct, "
        "fully-warm cache hit.\n"
        "* **shared %** — blocks where another concurrent call was already "
        "computing the same hash; this call awaited that future and did "
        "zero GPU work. *This is the column that explodes for Arm B.*\n"
        "* **miss %** — blocks this call had to compute itself. Every miss "
        "is `block_size × uncached_us_per_token` of GPU prefill.\n"
        "* **uncached blocks** — total miss count summed across all K×N "
        "calls. Proxy for total GPU prefill compute. Lower is better.\n"
        "* **peak KV** — high-water mark of distinct cached blocks; proxy "
        "for KV-cache memory pressure.\n"
    )
    lines.append("\n## Interpretation\n")
    if by_k:
        K_max = max(by_k)
        a_max = next(r for r in by_k[K_max] if r.name.startswith("A:"))
        b_max = next(r for r in by_k[K_max] if r.name.startswith("B:"))
        lines.append(
            f"At K={K_max} (N={N}): Arm B finishes in **{b_max.wall_time_s:.1f}s** vs "
            f"Arm A's **{a_max.wall_time_s:.1f}s** "
            f"(**{a_max.wall_time_s / max(1e-6, b_max.wall_time_s):.2f}×** wall-time speedup), "
            f"with **{b_max.summary['concurrent_share_pct']:.1f}%** of blocks served via "
            f"concurrent prefix sharing in Arm B vs **{a_max.summary['concurrent_share_pct']:.1f}%** "
            f"in Arm A. Total uncached prefill blocks: "
            f"**{b_max.misses}** (Arm B) vs **{a_max.misses}** (Arm A) — "
            f"a **{100.0 * (1 - b_max.misses / max(1, a_max.misses)):.1f}%** "
            "reduction in GPU prefill compute for the same total candidate budget.\n"
        )
    lines.append(
        "\nWhy it works: in Arm B every iteration submits K identical prompts. "
        "The first one to claim each block hash registers `COMPUTING`; the "
        "other K-1 find the same hash already in flight, await its event, "
        "and inherit the result with zero GPU prefill work. Decode runs in "
        "parallel under continuous batching. In Arm A the K runs evolve "
        "independent parents so user messages diverge after iteration 0 — "
        "only the static system prefix shares. The miss column collapses "
        "from K·N·user_blocks (Arm A) to roughly system_blocks + N·user_blocks "
        "(Arm B), which is what the *peak KV* column also confirms (Arm B's "
        "working set stays nearly flat in K).\n"
    )
    return "".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
