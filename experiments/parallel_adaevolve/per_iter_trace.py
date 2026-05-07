"""
Per-iteration trace experiment.

Existing real-vLLM benches capture only aggregate cache stats per
arm. This script captures `/metrics` at the START and END of every
iteration so we can plot the hit-rate evolution over time.

Output: a CSV with one row per (arm, iteration) and a figure showing
hit rate climbing as the cache warms.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List

SKYDISCOVER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKYDISCOVER_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from real_vllm_bench import (  # noqa: E402
    BenchmarkSpec, build_config, build_controller, seed_initial_program,
    discover_benchmarks,
)
from skydiscover.search.base_database import Program  # noqa: E402


_METRICS = ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
            "vllm:prompt_tokens_total", "vllm:request_success_total"]


def scrape(url: str = "http://127.0.0.1:8765/metrics") -> Dict[str, float]:
    out = {}
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception:
        return out
    for name in _METRICS:
        total = 0.0
        for m in re.finditer(rf"^{re.escape(name)}\b[^\n]* (\S+)$", text, re.M):
            try:
                total += float(m.group(1))
            except ValueError:
                pass
        out[name] = total
    return out


def diff(a, b):
    return {k: b.get(k, 0) - a.get(k, 0) for k in set(a) | set(b)}


async def run_traced(arm: str, K: int, N: int, bench: BenchmarkSpec,
                      vllm_base: str, model: str) -> List[Dict]:
    """Drive `N` AdaEvolve iterations and snapshot /metrics around each."""
    cfg = build_config(bench, vllm_base, model, K)
    ctrl, db = build_controller(cfg, bench)
    await seed_initial_program(ctrl, db, bench)

    # Re-implement run_discovery loop manually to get per-iter snapshots.
    rows = []
    real_iter = ctrl._run_iteration
    iter_count = [0]

    async def traced_iter(iteration, checkpoint_callback):
        iter_count[0] += 1
        before = scrape(f"{vllm_base}/metrics")
        t0 = time.monotonic()
        await real_iter(iteration, checkpoint_callback)
        wall = time.monotonic() - t0
        after = scrape(f"{vllm_base}/metrics")
        d = diff(before, after)
        queries = d.get("vllm:prefix_cache_queries_total", 0)
        hits = d.get("vllm:prefix_cache_hits_total", 0)
        rate = hits / queries if queries > 0 else 0.0
        best = db.get_best_program()
        best_score = (best.metrics or {}).get("combined_score") if best else None
        rows.append({
            "arm": arm,
            "iteration": iter_count[0],
            "wall_s": wall,
            "queries": queries,
            "hits": hits,
            "hit_rate": rate,
            "n_calls_in_iter": int(d.get("vllm:request_success_total", 0)),
            "best_score": best_score,
        })

    ctrl._run_iteration = traced_iter  # type: ignore
    try:
        await ctrl.run_discovery(start_iteration=1, max_iterations=N)
    finally:
        try:
            ctrl.close()
        except Exception:
            pass
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-base", default="http://127.0.0.1:8765")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--bench-root",
                    default=str(SKYDISCOVER_ROOT / "benchmarks" / "math"))
    ap.add_argument("--bench", default="circle_packing")
    ap.add_argument("--out",
                    default=str(Path(__file__).parent / "EXP_RESULTS"))
    args = ap.parse_args()

    benches = discover_benchmarks(Path(args.bench_root), [args.bench])
    if not benches:
        print("No bench loaded"); return
    bench = benches[0]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for arm, K in [("vanilla", 1), ("batched", args.K), ("all_optims", args.K)]:
        print(f"\n=== {arm} K={K} N={args.N} ===", flush=True)
        rows = await run_traced(arm, K, args.N, bench, args.vllm_base, args.model)
        for r in rows:
            print(f"  iter {r['iteration']:>2} wall={r['wall_s']:>5.1f}s "
                  f"calls={r['n_calls_in_iter']:>3} hit_rate={r['hit_rate']*100:>5.1f}% "
                  f"best={r['best_score']}", flush=True)
        all_rows.extend(rows)

    csv_path = out_dir / "per_iter_trace.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "iteration", "wall_s", "queries", "hits", "hit_rate",
                    "n_calls_in_iter", "best_score"])
        for r in all_rows:
            w.writerow([r["arm"], r["iteration"], f"{r['wall_s']:.3f}",
                        int(r["queries"]), int(r["hits"]),
                        f"{r['hit_rate']:.4f}", r["n_calls_in_iter"],
                        r["best_score"]])
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
