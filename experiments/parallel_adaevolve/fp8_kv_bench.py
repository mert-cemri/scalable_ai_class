"""
FP8 KV cache experiment (Tier 4.2 from NEXT_STEPS).

Compares two vLLM endpoints serving the SAME model:
  * Default KV cache (auto/bf16) on port 8765
  * FP8 KV cache              on port 8766 (started fresh on GPU 1)

Both have prefix caching enabled. We run an identical workload through
both and compare:
  * Cache hit rate (should be unchanged — quantization is per-block,
    cache identity is by hash)
  * KV memory usage (FP8 should fit ~2x more blocks)
  * Quality (best score) — does the lower precision cost accuracy?
  * Throughput (FP8 may be slightly faster due to memory bandwidth)
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


async def run_endpoint(label: str, vllm_base: str, model: str,
                        bench: BenchmarkSpec, K: int, N: int):
    print(f"\n=== {label} ({vllm_base}) ===", flush=True)
    cfg = build_config(bench, vllm_base, model, K)
    ctrl, db = build_controller(cfg, bench)
    await seed_initial_program(ctrl, db, bench)
    before = scrape_metrics(vllm_base)
    t0 = time.monotonic()
    try:
        await ctrl.run_discovery(start_iteration=1, max_iterations=N)
    finally:
        try:
            ctrl.close()
        except Exception:
            pass
    wall = time.monotonic() - t0
    after = scrape_metrics(vllm_base)
    delta = diff_metrics(before, after)

    best = db.get_best_program()
    best_score = (best.metrics or {}).get("combined_score") if best else None
    queries = delta.get("vllm:prefix_cache_queries_total", 0.0)
    hits = delta.get("vllm:prefix_cache_hits_total", 0.0)
    rate = hits / queries if queries > 0 else 0.0
    n_calls = int(delta.get("vllm:request_success_total", 0))
    print(f"  wall={wall:.1f}s  calls={n_calls}  hit_rate={100*rate:.1f}%  "
          f"best={best_score}  prompt_toks={delta.get('vllm:prompt_tokens_total',0):.0f}",
          flush=True)
    return {
        "label": label,
        "vllm_base": vllm_base,
        "wall_s": wall,
        "n_calls": n_calls,
        "hit_rate": rate,
        "best_score": best_score,
        "metrics_delta": delta,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-base", default="http://127.0.0.1:8765")
    ap.add_argument("--fp8-base", default="http://127.0.0.1:8766")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--bench-root",
                    default=str(SKYDISCOVER_ROOT / "benchmarks" / "math"))
    ap.add_argument("--bench", default="circle_packing")
    ap.add_argument("--out",
                    default=str(Path(__file__).parent / "EXP_RESULTS"))
    args = ap.parse_args()

    benches = discover_benchmarks(Path(args.bench_root), [args.bench])
    if not benches:
        print("No benches found.", flush=True); return
    bench = benches[0]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Run vanilla (K=1) and batched (K=8) on both endpoints.
    runs = []
    for K in [1, args.K]:
        for label, base in [("default-kv", args.bf16_base), ("fp8-kv", args.fp8_base)]:
            r = await run_endpoint(f"{label} K={K}", base, args.model, bench, K, args.N)
            r["K"] = K
            runs.append(r)

    csv_path = out_dir / "fp8_kv.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "K", "wall_s", "n_calls", "hit_rate", "best_score",
                    *_METRIC_NAMES])
        for r in runs:
            w.writerow([r["label"], r["K"], f"{r['wall_s']:.3f}", r["n_calls"],
                        f"{r['hit_rate']:.4f}", r["best_score"],
                        *[r["metrics_delta"].get(m, 0.0) for m in _METRIC_NAMES]])
    print(f"\nWrote {csv_path}", flush=True)

    md = ["# FP8 KV cache experiment\n\n"]
    md.append(f"Model: `{args.model}`. Same workload through two vLLM "
              f"endpoints — one with default KV dtype, one with `fp8`. "
              f"Bench: `{args.bench}`, K runs ∈ {{1, {args.K}}}, N_iter={args.N}.\n\n")
    md.append("| KV dtype | K | wall (s) | calls | hit rate | best score |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for r in runs:
        md.append(f"| {r['label']} | {r['K']} | {r['wall_s']:.1f} | "
                  f"{r['n_calls']} | {100*r['hit_rate']:.1f}% | "
                  f"{r['best_score']} |\n")
    md.append("\n## Interpretation\n\n")
    md.append(
        "FP8 KV cache halves the per-block KV memory footprint vs the "
        "default bf16 dtype. With prefix caching this directly translates "
        "to ~2× more blocks fitting in the same KV memory budget — the "
        "operational win, not measured here, is that more concurrent "
        "AdaEvolve problems fit before eviction starts. Quality is read "
        "from the `best_score` column; prefix-cache hit rate should be "
        "unchanged (FP8 affects per-block storage, not block identity).\n"
    )
    (out_dir / "fp8_kv.md").write_text("".join(md))
    print(f"Wrote {out_dir / 'fp8_kv.md'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
