"""
Real-vLLM end-to-end benchmark (Tier 1.1 + Tier 1.2 combined).

Drives multiple SkyDiscover math benchmarks concurrently against ONE
shared vLLM endpoint with prefix caching enabled. Compares:

  * **Vanilla AdaEvolve** (`candidates_per_iteration=1`)
  * **BatchedAdaEvolveController** (`candidates_per_iteration=K`)
  * **Multi-problem** (N problems concurrent, both arms above)

For each run we capture from vLLM's ``/metrics`` Prometheus endpoint:
  * ``vllm:gpu_prefix_cache_queries_total``
  * ``vllm:gpu_prefix_cache_hits_total``
  * ``vllm:prompt_tokens_total``
  * ``vllm:generation_tokens_total``
And from the controller side: best score, wall time, iterations, candidates.

The benchmarks (a small subset of the math suite — chosen for fast
evaluators that don't need GPU):
  - ``circle_packing``       — pack N circles, maximize sum of radii
  - ``first_autocorr_ineq``  — autocorrelation inequality optimization
  - ``signal_processing``    — DSP filter-coefficient optimization

Each runs with N=8 iterations to keep wall time bounded — the goal
isn't to find the global optimum, it's to measure the systems
properties (cache hit rate, GPU prefill compute, wall time per
iteration).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import urllib.request


SKYDISCOVER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKYDISCOVER_ROOT))

from skydiscover.config import (  # noqa: E402
    AdaEvolveDatabaseConfig,
    Config,
    LLMModelConfig,
)
from skydiscover.search.adaevolve.batched_controller import (  # noqa: E402
    BatchedAdaEvolveController,
)
from skydiscover.search.adaevolve.controller import AdaEvolveController  # noqa: E402
from skydiscover.search.adaevolve.database import AdaEvolveDatabase  # noqa: E402
from skydiscover.search.base_database import Program  # noqa: E402
from skydiscover.search.default_discovery_controller import (  # noqa: E402
    DiscoveryControllerInput,
)


# ----------------------------------------------------------------------
# vLLM /metrics scraping
# ----------------------------------------------------------------------


_METRIC_NAMES = [
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
]


def scrape_metrics(base: str = "http://127.0.0.1:8765") -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=5) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [warn] /metrics scrape failed: {e}", flush=True)
        return out
    for name in _METRIC_NAMES:
        # Sum across all label-set lines (e.g., per-model labels)
        total = 0.0
        for m in re.finditer(rf"^{re.escape(name)}\b[^\n]* (\S+)$", text, re.M):
            try:
                total += float(m.group(1))
            except ValueError:
                pass
        out[name] = total
    return out


def diff_metrics(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k in set(before) | set(after):
        out[k] = after.get(k, 0.0) - before.get(k, 0.0)
    return out


# ----------------------------------------------------------------------
# Benchmark loading
# ----------------------------------------------------------------------


@dataclass
class BenchmarkSpec:
    name: str
    initial_program: Path
    evaluator: Path
    extra_system_prompt: str  # Brief problem-description prepended to system message


def discover_benchmarks(bench_root: Path, names: List[str]) -> List[BenchmarkSpec]:
    out = []
    for name in names:
        d = bench_root / name
        if not d.is_dir():
            print(f"  [warn] missing benchmark dir: {d}", flush=True)
            continue
        ip = d / "initial_program.py"
        # Top-level evaluator.py wins; otherwise check evaluator/evaluator.py
        # (the format used by container-based benches — but the Python file
        # there still defines an `evaluate(program_path)` callable that we
        # can run directly without Docker, which is faster anyway).
        ev = d / "evaluator.py"
        if not ev.exists():
            ev = d / "evaluator" / "evaluator.py"
        if not ip.exists() or not ev.exists():
            print(f"  [warn] {name}: missing initial_program.py or evaluator.py", flush=True)
            continue
        # Read the README as the problem description (truncated).
        readme = d / "README.md"
        desc = readme.read_text() if readme.exists() else f"Benchmark: {name}"
        out.append(BenchmarkSpec(
            name=name, initial_program=ip, evaluator=ev, extra_system_prompt=desc[:1500]
        ))
    return out


# ----------------------------------------------------------------------
# Building a controller for one benchmark
# ----------------------------------------------------------------------


def build_config(
    bench: BenchmarkSpec,
    vllm_base: str,
    model_name: str,
    K: int,
    enable_prefetch: bool = False,
    cache_aligned: bool = False,
    adaptive_candidates: bool = False,
    enable_speculative_pipelining: bool = False,
    speculation_gpu_pressure_gating: bool = False,
    speculation_pressure_threshold: float = 0.6,
) -> Config:
    cfg = Config()
    cfg.language = "python"
    cfg.diff_based_generation = True
    cfg.checkpoint_interval = 1_000_000  # never checkpoint mid-run
    cfg.max_solution_length = 60_000
    cfg.max_parallel_iterations = 1

    # Local vLLM endpoint — Qwen3-4B-Instruct-2507 with prefix caching.
    cfg.llm.models = [
        LLMModelConfig(
            name=model_name,
            api_base=f"{vllm_base}/v1",
            api_key="EMPTY",
            weight=1.0,
            temperature=0.7,
            max_tokens=2048,
            timeout=300,
        )
    ]
    cfg.llm.evaluator_models = list(cfg.llm.models)
    cfg.llm.guide_models = list(cfg.llm.models)

    db = AdaEvolveDatabaseConfig()
    db.population_size = 16
    db.num_islands = 2
    db.use_paradigm_breakthrough = False
    db.use_dynamic_islands = False
    db.use_migration = False
    db.use_ucb_selection = True
    db.candidates_per_iteration = max(1, K)
    db.diversify_temperature = K > 1
    db.temperature_spread = 0.4
    db.adaptive_candidates = adaptive_candidates
    db.candidates_min = 2 if adaptive_candidates else 1
    db.candidates_max = max(K, 16) if adaptive_candidates else K
    db.enable_prefetch = enable_prefetch
    db.cache_aligned_prompt = cache_aligned
    db.enable_speculative_pipelining = enable_speculative_pipelining
    db.speculation_gpu_pressure_gating = speculation_gpu_pressure_gating
    db.speculation_pressure_threshold = speculation_pressure_threshold
    db.speculation_metrics_url = f"{vllm_base}/metrics"

    cfg.search.type = "adaevolve_batched" if K > 1 else "adaevolve"
    cfg.search.database = db
    cfg.search.num_context_programs = 2

    cfg.context_builder.system_message = (
        f"You are an expert mathematician. Improve the following program "
        f"to maximize its combined_score on this task:\n\n"
        f"=== TASK ===\n{bench.extra_system_prompt}\n=== END TASK ===\n\n"
        f"Use SEARCH/REPLACE diff blocks. Be concise. The evaluator runs "
        f"the program directly; don't add I/O or argparse. Keep the same "
        f"function signatures."
    )

    cfg.evaluator.timeout = 90
    cfg.evaluator.inject_evaluator_context = False
    cfg.evaluator.evaluation_file = str(bench.evaluator)
    return cfg


def build_controller(
    cfg: Config, bench: BenchmarkSpec
) -> Tuple[BatchedAdaEvolveController, AdaEvolveDatabase]:
    db = AdaEvolveDatabase(name=cfg.search.type, config=cfg.search.database)
    db.language = "python"
    inp = DiscoveryControllerInput(
        config=cfg,
        evaluation_file=str(bench.evaluator),
        database=db,
        file_suffix=".py",
        output_dir=None,
    )
    if cfg.search.type == "adaevolve_batched":
        ctrl = BatchedAdaEvolveController(inp)
    else:
        ctrl = AdaEvolveController(inp)
    return ctrl, db


async def seed_initial_program(
    ctrl, db: AdaEvolveDatabase, bench: BenchmarkSpec
) -> Optional[Program]:
    """Evaluate the initial_program.py and add it as the seed program."""
    sol = bench.initial_program.read_text()
    try:
        eval_result = await ctrl.evaluator.evaluate_program(sol, "seed")
        metrics = eval_result.metrics or {}
        artifacts = eval_result.artifacts or {}
    except Exception as e:
        print(f"  [warn] {bench.name} seed eval failed: {e}", flush=True)
        metrics = {"combined_score": 0.0}
        artifacts = {}
    seed = Program(
        id="seed-" + bench.name,
        solution=sol,
        language="python",
        metrics=metrics,
        iteration_found=0,
        parent_id=None,
        generation=0,
    )
    db.add(seed, iteration=0)
    return seed


# ----------------------------------------------------------------------
# Trial driver
# ----------------------------------------------------------------------


@dataclass
class TrialOutcome:
    bench: str
    arm: str
    K: int
    N_iter: int
    seed_score: Optional[float]
    best_score: Optional[float]
    wall_s: float
    metrics_delta: Dict[str, float] = field(default_factory=dict)


async def run_one_problem(
    bench: BenchmarkSpec,
    arm: str,
    K: int,
    N: int,
    vllm_base: str,
    model_name: str,
    enable_prefetch: bool = False,
    cache_aligned: bool = False,
    adaptive_candidates: bool = False,
    enable_speculative_pipelining: bool = False,
    speculation_gpu_pressure_gating: bool = False,
    speculation_pressure_threshold: float = 0.6,
) -> TrialOutcome:
    cfg = build_config(
        bench, vllm_base, model_name, K,
        enable_prefetch=enable_prefetch,
        cache_aligned=cache_aligned,
        adaptive_candidates=adaptive_candidates,
        enable_speculative_pipelining=enable_speculative_pipelining,
        speculation_gpu_pressure_gating=speculation_gpu_pressure_gating,
        speculation_pressure_threshold=speculation_pressure_threshold,
    )
    ctrl, db = build_controller(cfg, bench)
    seed = await seed_initial_program(ctrl, db, bench)
    seed_score = (seed.metrics or {}).get("combined_score") if seed else None

    t0 = time.monotonic()
    try:
        await ctrl.run_discovery(start_iteration=1, max_iterations=N)
    except Exception as e:
        print(f"  [{bench.name}/{arm}] run_discovery error: {e}", flush=True)
    finally:
        try:
            ctrl.close()
        except Exception:
            pass
    wall = time.monotonic() - t0

    best = db.get_best_program()
    best_score = (best.metrics or {}).get("combined_score") if best else None

    return TrialOutcome(
        bench=bench.name, arm=arm, K=K, N_iter=N,
        seed_score=seed_score, best_score=best_score, wall_s=wall,
    )


# ----------------------------------------------------------------------
# Multi-problem orchestrator (Tier 1.2)
# ----------------------------------------------------------------------


async def multi_problem(
    benches: List[BenchmarkSpec],
    arm_label: str,
    K: int,
    N: int,
    vllm_base: str,
    model_name: str,
    **kwargs,
) -> List[TrialOutcome]:
    """All N benchmarks run concurrently against ONE shared vLLM endpoint."""
    print(f"\n=== {arm_label} (K={K}, N={N} iters/bench, "
          f"benches={[b.name for b in benches]}) ===", flush=True)
    before = scrape_metrics(vllm_base)
    tasks = [run_one_problem(b, arm_label, K, N, vllm_base, model_name, **kwargs) for b in benches]
    out = await asyncio.gather(*tasks, return_exceptions=False)
    after = scrape_metrics(vllm_base)
    delta = diff_metrics(before, after)
    for o in out:
        o.metrics_delta = delta  # shared across the parallel cohort
    print(f"  shared metrics delta: {delta}", flush=True)
    for o in out:
        print(f"  {o.bench:<22} arm={o.arm:<22} K={o.K} "
              f"seed={o.seed_score} -> best={o.best_score} "
              f"wall={o.wall_s:.1f}s", flush=True)
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-base", default="http://127.0.0.1:8765")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--N", type=int, default=6, help="iterations per benchmark")
    ap.add_argument("--K-batched", type=int, default=8, help="K for the batched arm")
    ap.add_argument("--bench-root", default=str(SKYDISCOVER_ROOT / "benchmarks" / "math"))
    ap.add_argument("--benches", default="circle_packing,first_autocorr_ineq")
    ap.add_argument("--out", default=str(Path(__file__).parent / "EXP_RESULTS"))
    ap.add_argument("--arms", default="vanilla,batched,batched_aligned,batched_prefetch,batched_adaptive",
                    help="comma-separated list of arms to run")
    args = ap.parse_args()

    benches = discover_benchmarks(Path(args.bench_root), args.benches.split(","))
    if not benches:
        print("No benchmarks loaded; aborting", flush=True)
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms_to_run = [a.strip() for a in args.arms.split(",") if a.strip()]
    all_outcomes: List[TrialOutcome] = []

    arm_specs = {
        "vanilla":         dict(K=1,                  enable_prefetch=False, cache_aligned=False, adaptive_candidates=False, enable_speculative_pipelining=False),
        "batched":         dict(K=args.K_batched,     enable_prefetch=False, cache_aligned=False, adaptive_candidates=False, enable_speculative_pipelining=False),
        "batched_aligned": dict(K=args.K_batched,     enable_prefetch=False, cache_aligned=True,  adaptive_candidates=False, enable_speculative_pipelining=False),
        "batched_prefetch":dict(K=args.K_batched,     enable_prefetch=True,  cache_aligned=False, adaptive_candidates=False, enable_speculative_pipelining=False),
        "batched_adaptive":dict(K=args.K_batched,     enable_prefetch=False, cache_aligned=False, adaptive_candidates=True,  enable_speculative_pipelining=False),
        "batched_speculative": dict(K=args.K_batched, enable_prefetch=False, cache_aligned=False, adaptive_candidates=False, enable_speculative_pipelining=True),
        "batched_speculative_gated": dict(K=args.K_batched, enable_prefetch=False, cache_aligned=False, adaptive_candidates=False, enable_speculative_pipelining=True, speculation_gpu_pressure_gating=True, speculation_pressure_threshold=0.5),
        "all_optims":      dict(K=args.K_batched,     enable_prefetch=True,  cache_aligned=True,  adaptive_candidates=True,  enable_speculative_pipelining=True, speculation_gpu_pressure_gating=True, speculation_pressure_threshold=0.5),
    }

    for arm in arms_to_run:
        spec = arm_specs.get(arm)
        if spec is None:
            print(f"  [skip] unknown arm {arm}", flush=True)
            continue
        try:
            outs = await multi_problem(
                benches, arm_label=arm, N=args.N, vllm_base=args.vllm_base,
                model_name=args.model, **spec,
            )
            all_outcomes.extend(outs)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [error] arm {arm} failed: {e}", flush=True)

    # ---- write CSV + markdown ----
    csv_path = out_dir / "real_vllm_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        cols = ["bench", "arm", "K", "N_iter", "seed_score", "best_score", "wall_s",
                *_METRIC_NAMES]
        w.writerow(cols)
        for o in all_outcomes:
            w.writerow([
                o.bench, o.arm, o.K, o.N_iter,
                o.seed_score, o.best_score, f"{o.wall_s:.3f}",
                *[o.metrics_delta.get(m, 0.0) for m in _METRIC_NAMES],
            ])
    print(f"\nWrote {csv_path}", flush=True)

    # Markdown
    md = ["# Real-vLLM benchmark — multi-problem orchestration\n\n"]
    md.append(f"Model: `{args.model}` on local vLLM. "
              f"Per-bench iterations N={args.N}. Benches: "
              f"{', '.join(b.name for b in benches)}.\n\n")
    md.append("## Per-arm summary (means across benchmarks)\n\n")

    by_arm: Dict[str, List[TrialOutcome]] = {}
    for o in all_outcomes:
        by_arm.setdefault(o.arm, []).append(o)

    md.append("| arm | K | mean wall (s) | mean best score | total prompt toks | total prefix-cache hits | hit rate |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for arm, outs in by_arm.items():
        if not outs:
            continue
        # The metrics_delta is identical across all outs in the cohort (it's
        # the SHARED-endpoint delta over the cohort run). Read once.
        d = outs[0].metrics_delta
        queries = d.get("vllm:prefix_cache_queries_total", 0.0)
        hits = d.get("vllm:prefix_cache_hits_total", 0.0)
        rate = hits / queries if queries > 0 else 0.0
        toks = d.get("vllm:prompt_tokens_total", 0.0)
        mean_wall = sum(o.wall_s for o in outs) / len(outs)
        mean_best = sum((o.best_score or 0.0) for o in outs) / len(outs)
        md.append(f"| {arm} | {outs[0].K} | {mean_wall:.1f} | {mean_best:.4f} | "
                  f"{toks:.0f} | {hits:.0f} | {100*rate:.1f}% |\n")

    md_path = out_dir / "real_vllm_summary.md"
    md_path.write_text("".join(md))
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
