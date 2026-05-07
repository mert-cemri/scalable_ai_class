"""
Multi-problem scaling sweep on real vLLM.

Sweeps the number of concurrent problems N ∈ {1, 2, 4, 8} sharing one
vLLM endpoint. Each "problem" is a copy of one of the two file-based
math benchmarks (circle_packing, signal_processing) with a different
RNG seed, so per-problem prompts diverge by parent code while sharing
the same system prefix.

Three arms per N:
  * vanilla (K=1)
  * batched (K=8)
  * batched + speculative pipelining (K=8 + spec)

Captures from vLLM /metrics:
  * total prompt tokens
  * total prefix-cache hits
  * effective hit rate
  * cohort wall time
  * total LLM calls

Headline metric: candidate throughput (calls/s) and effective per-call
latency as N scales.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

SKYDISCOVER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKYDISCOVER_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from real_vllm_bench import (  # noqa: E402
    BenchmarkSpec, build_config, build_controller, seed_initial_program,
    scrape_metrics, diff_metrics, _METRIC_NAMES, discover_benchmarks,
)


async def _run_problem(bench: BenchmarkSpec, arm: str, K: int, N: int,
                        vllm_base: str, model: str,
                        enable_speculative: bool, seed_offset: int,
                        gpu_pressure_gating: bool = False):
    cfg = build_config(
        bench, vllm_base, model, K,
        enable_speculative_pipelining=enable_speculative,
        speculation_gpu_pressure_gating=gpu_pressure_gating,
        speculation_pressure_threshold=0.5,
    )
    # Inject a per-replica seed nonce into the system message so different
    # replicas of the same benchmark have non-overlapping prompts (otherwise
    # they'd share everything and our "N problems" would just be K copies of
    # one cache-resident problem).
    cfg.context_builder.system_message = (
        cfg.context_builder.system_message
        + f"\n\n# replica-seed: {seed_offset}\n"
    )
    ctrl, db = build_controller(cfg, bench)
    await seed_initial_program(ctrl, db, bench)
    try:
        await ctrl.run_discovery(start_iteration=1, max_iterations=N)
    finally:
        try:
            ctrl.close()
        except Exception:
            pass
    best = db.get_best_program()
    return (best.metrics or {}).get("combined_score") if best else None


async def cohort(N_concurrent: int, arm: str, K: int, N_iter: int,
                 benches: list, vllm_base: str, model: str):
    enable_spec = arm in ("batched_speculative", "batched_speculative_gated")
    pressure_gating = arm == "batched_speculative_gated"
    actual_K = K if arm != "vanilla" else 1
    print(f"\n--- N_concurrent={N_concurrent} arm={arm} K={actual_K} "
          f"N_iter={N_iter} ---", flush=True)
    before = scrape_metrics(vllm_base)
    t0 = time.monotonic()
    tasks = []
    for i in range(N_concurrent):
        bench = benches[i % len(benches)]
        tasks.append(_run_problem(
            bench, arm, actual_K, N_iter, vllm_base, model,
            enable_speculative=enable_spec,
            seed_offset=10_000 + 17 * i,
            gpu_pressure_gating=pressure_gating,
        ))
    results = await asyncio.gather(*tasks)
    wall = time.monotonic() - t0
    after = scrape_metrics(vllm_base)
    delta = diff_metrics(before, after)
    queries = delta.get("vllm:prefix_cache_queries_total", 0.0)
    hits = delta.get("vllm:prefix_cache_hits_total", 0.0)
    rate = hits / queries if queries > 0 else 0.0
    n_calls = int(delta.get("vllm:request_success_total", 0))
    throughput = n_calls / wall if wall > 0 else 0.0
    print(f"  wall={wall:.1f}s  calls={n_calls}  hit_rate={100*rate:.1f}%  "
          f"throughput={throughput:.2f} calls/s  best_scores={results}",
          flush=True)
    return {
        "N_concurrent": N_concurrent,
        "arm": arm,
        "K": actual_K,
        "N_iter": N_iter,
        "wall_s": wall,
        "n_calls": n_calls,
        "hit_rate": rate,
        "throughput_calls_per_s": throughput,
        "best_scores": results,
        "metrics_delta": delta,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-base", default="http://127.0.0.1:8765")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--N-iter", type=int, default=3)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--bench-root",
                    default=str(SKYDISCOVER_ROOT / "benchmarks" / "math"))
    ap.add_argument("--benches", default="circle_packing,signal_processing")
    ap.add_argument("--Ns", default="1,2,4,8")
    ap.add_argument("--arms", default="vanilla,batched,batched_speculative")
    ap.add_argument("--out",
                    default=str(Path(__file__).parent / "EXP_RESULTS"))
    args = ap.parse_args()

    benches = discover_benchmarks(Path(args.bench_root), args.benches.split(","))
    if not benches:
        print("No benches loaded.", flush=True); return
    Ns = [int(x) for x in args.Ns.split(",") if x.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for N in Ns:
        for arm in arms:
            r = await cohort(N, arm, args.K, args.N_iter, benches,
                             args.vllm_base, args.model)
            rows.append(r)

    # CSV
    csv_path = out_dir / "scaling_sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        cols = ["N_concurrent", "arm", "K", "N_iter", "wall_s", "n_calls",
                "hit_rate", "throughput_calls_per_s",
                *_METRIC_NAMES]
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r["N_concurrent"], r["arm"], r["K"], r["N_iter"],
                f"{r['wall_s']:.3f}", r["n_calls"],
                f"{r['hit_rate']:.4f}", f"{r['throughput_calls_per_s']:.3f}",
                *[r["metrics_delta"].get(m, 0.0) for m in _METRIC_NAMES],
            ])
    print(f"\nWrote {csv_path}", flush=True)

    # Markdown
    md = ["# Multi-problem scaling sweep (Tier 1.2 in operational form)\n\n"]
    md.append(f"Model: `{args.model}`. Each cohort runs `N_concurrent` "
              f"copies of the math benches concurrently against ONE shared "
              f"vLLM endpoint. K={args.K}, N_iter={args.N_iter}, "
              f"benches: {args.benches}.\n\n")
    md.append("## Throughput vs concurrency\n\n")
    md.append("| N_concurrent | arm | wall (s) | calls | hit rate | throughput (calls/s) |\n")
    md.append("|---:|---|---:|---:|---:|---:|\n")
    for r in rows:
        md.append(f"| {r['N_concurrent']} | {r['arm']} | "
                  f"{r['wall_s']:.1f} | {r['n_calls']} | "
                  f"{100*r['hit_rate']:.1f}% | "
                  f"{r['throughput_calls_per_s']:.2f} |\n")
    md.append("\n## Reading\n\n")
    md.append(
        "* **wall (s)** — time for the full cohort to complete. "
        "Lower is better.\n"
        "* **calls** — total LLM calls across all N concurrent problems "
        "(includes batched K-fan-out and any speculative requests).\n"
        "* **hit rate** — overall vLLM prefix-cache hit rate over the "
        "cohort, scraped from `/metrics`.\n"
        "* **throughput** — calls/second the GPU sustained. The headline "
        "scaling number: how does throughput grow with N_concurrent?\n\n"
    )
    md.append("## Scaling profile\n\n")
    md.append(
        "If the GPU were perfectly parallel, throughput would scale "
        "linearly in N. Reality is sub-linear: at some N the GPU's "
        "prefill+decode bandwidth saturates and throughput plateaus. The "
        "interesting question is *at what N* the batched arms saturate "
        "vs the vanilla arm. The KV report's prediction: batched "
        "saturates at much higher N because each call uses ~5× less "
        "prefill compute, freeing the GPU for more concurrent work.\n"
    )
    (out_dir / "scaling_sweep.md").write_text("".join(md))
    print(f"Wrote {out_dir / 'scaling_sweep.md'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
