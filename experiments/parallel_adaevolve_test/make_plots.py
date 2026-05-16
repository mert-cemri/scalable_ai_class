"""
Generate the full figure set for the paper-style write-up.

Reads every CSV/JSON under EXP_RESULTS/ and produces a coherent
matplotlib figure set saved as PNGs in EXP_RESULTS/figs/.

One script for all plots so the visual style is uniform: same color
palette per arm, consistent axis labels, consistent font sizes.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

ROOT = Path(__file__).parent
RES = ROOT / "EXP_RESULTS"
OUT = RES / "figs"
OUT.mkdir(exist_ok=True)


# Consistent color palette by "arm family"
COLORS = {
    "vanilla":                "#d62728",  # red
    "batched":                "#1f77b4",  # blue
    "batched_aligned":        "#9467bd",  # purple
    "batched_prefetch":       "#17becf",  # cyan
    "batched_adaptive":       "#bcbd22",  # olive
    "batched_speculative":    "#8c564b",  # brown
    "batched_speculative_gated": "#2ca02c",  # green
    "all_optims":             "#ff7f0e",  # orange
    "lru":                    "#d62728",
    "priority":               "#2ca02c",
    "diff_aware":             "#2ca02c",
    "default-kv":             "#1f77b4",
    "fp8-kv":                 "#ff7f0e",
}


def color(arm: str) -> str:
    return COLORS.get(arm, "#7f7f7f")


# ---------------------------------------------------------------------------
# Fig 1 — Quality bench (Rastrigin) summary [pre-existing]
# ---------------------------------------------------------------------------
# Already produced: EXP_RESULTS/quality_grid.png and
# EXP_RESULTS/gpu_efficiency.png by quality_bench.py. Pass through.


# ---------------------------------------------------------------------------
# Fig 2 — Cache-only synthetic bench (bench.py results.json)
# ---------------------------------------------------------------------------

def fig_cache_only_synthetic():
    """Bar chart: misses by arm × K, plus hit-rate panel."""
    rj = ROOT / "results.json"
    if not rj.exists():
        return
    rows = json.loads(rj.read_text())
    Ks = sorted(set(r["K"] for r in rows))
    arms = ["A", "B"]
    arm_labels = {"A": "K independent runs", "B": "1 run × K-batched"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    width = 0.35
    xs = np.arange(len(Ks))
    for i, arm in enumerate(arms):
        misses = []
        for K in Ks:
            cell = next((r for r in rows
                          if r["K"] == K and r["name"].startswith(f"{arm}:")), None)
            misses.append(cell["misses"] if cell else 0)
        axes[0].bar(xs + (i - 0.5) * width, misses, width,
                    label=arm_labels[arm], color=color("vanilla" if arm == "A" else "batched"))
    axes[0].set_xticks(xs); axes[0].set_xticklabels([f"K={k}" for k in Ks])
    axes[0].set_ylabel("Uncached prefill blocks (lower is better)")
    axes[0].set_title("GPU prefill compute, by arm × K\n(synthetic AdaEvolve-shaped workload)")
    axes[0].legend()

    # Effective hit-rate panel
    for i, arm in enumerate(arms):
        rates = []
        for K in Ks:
            cell = next((r for r in rows
                          if r["K"] == K and r["name"].startswith(f"{arm}:")), None)
            s = cell.get("summary", {}) if cell else {}
            rates.append(s.get("ready_hit_pct", 0) + s.get("concurrent_share_pct", 0))
        axes[1].plot(xs, rates, "-o", label=arm_labels[arm],
                     color=color("vanilla" if arm == "A" else "batched"))
    axes[1].set_xticks(xs); axes[1].set_xticklabels([f"K={k}" for k in Ks])
    axes[1].set_ylabel("Effective hit + shared %")
    axes[1].set_title("Effective cache reuse rate")
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    fig.suptitle("Synthetic cache-only bench: K-batched vs K-concurrent", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cache_only_synthetic.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3 — Scaling sweep: throughput vs N_concurrent
# ---------------------------------------------------------------------------

def fig_scaling_sweep():
    csv_path = RES / "scaling_sweep_full.csv"
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open()))
    Ns = sorted(set(int(r["N_concurrent"]) for r in rows))
    arms = sorted(set(r["arm"] for r in rows))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for arm in arms:
        ys_tput = [next((float(r["throughput_calls_per_s"]) for r in rows
                          if int(r["N_concurrent"]) == N and r["arm"] == arm), 0)
                    for N in Ns]
        ys_hit = [next((float(r["hit_rate"]) * 100 for r in rows
                         if int(r["N_concurrent"]) == N and r["arm"] == arm), 0)
                   for N in Ns]
        axes[0].plot(Ns, ys_tput, "-o", color=color(arm), label=arm, linewidth=2, markersize=7)
        axes[1].plot(Ns, ys_hit, "-s", color=color(arm), label=arm, linewidth=2, markersize=7)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(Ns); axes[0].set_xticklabels([str(n) for n in Ns])
    axes[0].set_xlabel("N concurrent problems")
    axes[0].set_ylabel("Throughput (calls / s)")
    axes[0].set_title("Throughput scaling with concurrency\n(real vLLM, Qwen3-4B)")
    axes[0].legend(loc="best")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(Ns); axes[1].set_xticklabels([str(n) for n in Ns])
    axes[1].set_xlabel("N concurrent problems")
    axes[1].set_ylabel("Effective cache hit rate (%)")
    axes[1].set_title("vLLM prefix-cache hit rate")
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scaling_sweep.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4 — Diff-aware KV reuse vs edit_size
# ---------------------------------------------------------------------------

def fig_diff_aware():
    rj = RES / "diff_aware_results.json"
    if not rj.exists():
        return
    rows = json.loads(rj.read_text())
    edit_sizes = sorted(set(r["edit_size"] for r in rows))
    by = {}
    for r in rows:
        by.setdefault((r["edit_size"], r["label"]), []).append(r)
    van_us = [np.mean([r["prefill_us"] for r in by[(es, "vanilla")]]) / 1000 for es in edit_sizes]
    diff_us = [np.mean([r["prefill_us"] for r in by[(es, "diff_aware")]]) / 1000 for es in edit_sizes]
    van_miss = [np.mean([r["misses"] for r in by[(es, "vanilla")]]) for es in edit_sizes]
    diff_miss = [np.mean([r["misses"] for r in by[(es, "diff_aware")]]) for es in edit_sizes]
    diff_reused = [np.mean([r["diff_aware_hits"] for r in by[(es, "diff_aware")]]) for es in edit_sizes]
    reductions = [100.0 * (1 - d / max(1, v)) for d, v in zip(diff_us, van_us)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.35
    xs = np.arange(len(edit_sizes))
    axes[0].bar(xs - width/2, van_miss, width, label="vanilla (chained)", color=color("vanilla"))
    axes[0].bar(xs + width/2, diff_miss, width, label="diff-aware", color=color("diff_aware"))
    # Stack: also overlay the diff-aware-reused blocks
    axes[0].bar(xs + width/2, diff_reused, width, bottom=diff_miss,
                color=color("diff_aware"), alpha=0.4, label="diff-aware reused (free with RoPE)")
    axes[0].set_xticks(xs); axes[0].set_xticklabels([str(es) for es in edit_sizes])
    axes[0].set_xlabel("Per-iteration edit size (tokens)")
    axes[0].set_ylabel("Block-lookups across run")
    axes[0].set_title("Block accounting under chained-only vs diff-aware caches")
    axes[0].legend(loc="upper left")

    axes[1].plot(edit_sizes, reductions, "-o", color=color("diff_aware"), linewidth=2.2,
                 markersize=8, label="prefill compute reduction")
    axes[1].axhline(0, color="grey", linewidth=0.5)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(edit_sizes); axes[1].set_xticklabels([str(es) for es in edit_sizes])
    axes[1].set_xlabel("Per-iteration edit size (tokens)")
    axes[1].set_ylabel("GPU prefill compute reduction (%)")
    axes[1].set_title("Diff-aware savings vs edit size\n(peaks at edit_size ≈ block_size = 16)")
    axes[1].set_ylim(0, 100)
    for x, y in zip(edit_sizes, reductions):
        axes[1].annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)
    fig.suptitle("Diff-aware cross-iteration KV reuse (Tier 3.1, simulated)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_diff_aware.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5 — Adversarial multi-tenant priority eviction
# ---------------------------------------------------------------------------

def fig_adversarial_eviction():
    rj = RES / "adversarial_eviction.json"
    if not rj.exists():
        return
    rows = json.loads(rj.read_text())
    # Aggregate by (R_b, policy)
    R_to_pol = {}
    for r in rows:
        rb = r["b_calls"] // max(1, r["a_calls"])
        R_to_pol.setdefault((rb, r["policy"]), []).append(r["a_misses"])
    Rs = sorted(set(k[0] for k in R_to_pol))
    pol_lru = [np.mean(R_to_pol[(rb, "lru")]) for rb in Rs]
    pol_pri = [np.mean(R_to_pol[(rb, "priority")]) for rb in Rs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.35
    xs = np.arange(len(Rs))
    axes[0].bar(xs - width/2, pol_lru, width, color=color("lru"), label="LRU eviction")
    axes[0].bar(xs + width/2, pol_pri, width, color=color("priority"),
                label="priority eviction (AdaEvolve-aware)")
    axes[0].set_xticks(xs); axes[0].set_xticklabels([f"R_b={rb}" for rb in Rs])
    axes[0].set_xlabel("Tenant B churn rate (calls per A call)")
    axes[0].set_ylabel("Tenant A GPU prefill misses (lower is better)")
    axes[0].set_title("Adversarial multi-tenant pattern\n"
                      "(tenant A: stable prefix; tenant B: ephemeral chum)")
    axes[0].legend()
    for x, v in zip(xs, pol_lru):
        axes[0].annotate(f"{v:.0f}", (x - width/2, v),
                         textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    for x, v in zip(xs, pol_pri):
        axes[0].annotate(f"{v:.0f}", (x + width/2, v),
                         textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)

    reductions = [100 * (1 - p / max(1, l)) for l, p in zip(pol_lru, pol_pri)]
    axes[1].bar(xs, reductions, color=color("priority"), width=0.6)
    axes[1].set_xticks(xs); axes[1].set_xticklabels([f"R_b={rb}" for rb in Rs])
    axes[1].set_xlabel("Tenant B churn rate")
    axes[1].set_ylabel("Reduction in tenant A misses (%)")
    axes[1].set_title("Priority eviction's saving over LRU on tenant A")
    for x, v in zip(xs, reductions):
        axes[1].annotate(f"{v:.0f}%", (x, v), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(OUT / "fig_adversarial_eviction.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 6 — Persistent KV cold-start
# ---------------------------------------------------------------------------

def fig_persistent_kv():
    rj = RES / "cache_policy_results.json"
    if not rj.exists():
        return
    data = json.loads(rj.read_text())
    rows = data.get("persistent_kv", [])
    if not rows:
        return
    seeds = sorted(set(r["seed"] for r in rows))
    cold_first = [next(r["first_iter_misses_cold"] for r in rows if r["seed"] == s) for s in seeds]
    warm_first = [next(r["first_iter_misses_warm"] for r in rows if r["seed"] == s) for s in seeds]
    cold_total = [next(r["total_misses_cold"] for r in rows if r["seed"] == s) for s in seeds]
    warm_total = [next(r["total_misses_warm"] for r in rows if r["seed"] == s) for s in seeds]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.35
    xs = np.arange(len(seeds))
    axes[0].bar(xs - width/2, cold_first, width, color=color("vanilla"),
                label="cold start (no snapshot)")
    axes[0].bar(xs + width/2, warm_first, width, color=color("batched"),
                label="warm start (snapshot reloaded)")
    axes[0].set_xticks(xs); axes[0].set_xticklabels([f"seed={s}" for s in seeds])
    axes[0].set_ylabel("First-iteration misses")
    axes[0].set_title("Cold-start cost: tenant A's iter-0 misses\nwith vs without persistent-KV snapshot")
    axes[0].legend()
    for i, (c, w) in enumerate(zip(cold_first, warm_first)):
        red = 100 * (1 - w / max(1, c))
        axes[0].annotate(f"−{red:.0f}%", (i + width/2, w),
                         textcoords="offset points", xytext=(0, 4), ha="center",
                         color="green", fontweight="bold", fontsize=10)

    axes[1].bar(xs - width/2, cold_total, width, color=color("vanilla"))
    axes[1].bar(xs + width/2, warm_total, width, color=color("batched"))
    axes[1].set_xticks(xs); axes[1].set_xticklabels([f"seed={s}" for s in seeds])
    axes[1].set_ylabel("Total misses across run")
    axes[1].set_title("Total run cost\n(persistent KV helps mostly at iter 0)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_persistent_kv.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 7 — Real-vLLM canonical 4-arm comparison
# ---------------------------------------------------------------------------

def fig_real_vllm_canonical():
    csv_path = RES / "real_vllm_summary_canonical.csv"
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open()))
    arms = ["vanilla", "batched", "batched_speculative_gated", "all_optims"]
    rows_by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in arms}
    if not all(rows_by_arm[a] for a in arms):
        return

    # Compute per-arm metrics: cohort wall = max wall, cohort calls = unique
    # request_success_total in the metrics (which is shared across the cohort).
    summary = {}
    for arm in arms:
        rs = rows_by_arm[arm]
        wall = max(float(r["wall_s"]) for r in rs)
        d = rs[0]  # metrics are identical across cohort members
        queries = float(d.get("vllm:prefix_cache_queries_total", 0) or 0)
        hits = float(d.get("vllm:prefix_cache_hits_total", 0) or 0)
        n_calls = int(float(d.get("vllm:request_success_total", 0) or 0))
        rate = hits / queries if queries > 0 else 0.0
        bests = [float(r["best_score"] or 0) for r in rs]
        summary[arm] = {
            "wall": wall, "n_calls": n_calls, "rate": rate,
            "per_call": wall / max(1, n_calls), "best": np.mean(bests),
            "best_circle": float(next((r["best_score"] for r in rs
                                        if r["bench"] == "circle_packing"), 0) or 0),
        }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    arms_order = arms
    xs = np.arange(len(arms_order))

    # (a) Cache hit rate
    rates = [summary[a]["rate"] * 100 for a in arms_order]
    bars = axes[0, 0].bar(xs, rates, color=[color(a) for a in arms_order], width=0.6)
    axes[0, 0].set_xticks(xs); axes[0, 0].set_xticklabels(arms_order, rotation=15, ha="right")
    axes[0, 0].set_ylabel("vLLM prefix-cache hit rate (%)")
    axes[0, 0].set_title("(a) Cache hit rate by arm")
    axes[0, 0].set_ylim(0, 100)
    for b, v in zip(bars, rates):
        axes[0, 0].annotate(f"{v:.1f}%", (b.get_x() + b.get_width()/2, v),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=10, fontweight="bold")

    # (b) Per-call wall time
    pcs = [summary[a]["per_call"] for a in arms_order]
    bars = axes[0, 1].bar(xs, pcs, color=[color(a) for a in arms_order], width=0.6)
    axes[0, 1].set_xticks(xs); axes[0, 1].set_xticklabels(arms_order, rotation=15, ha="right")
    axes[0, 1].set_ylabel("Per-call wall time (s)")
    axes[0, 1].set_title("(b) Effective per-call latency")
    for b, v in zip(bars, pcs):
        axes[0, 1].annotate(f"{v:.2f}s", (b.get_x() + b.get_width()/2, v),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=10, fontweight="bold")

    # (c) Best score on circle_packing
    bests_cp = [summary[a]["best_circle"] for a in arms_order]
    bars = axes[1, 0].bar(xs, bests_cp, color=[color(a) for a in arms_order], width=0.6)
    axes[1, 0].set_xticks(xs); axes[1, 0].set_xticklabels(arms_order, rotation=15, ha="right")
    axes[1, 0].set_ylabel("Best −Rastrigin / −Circle-Packing score")
    axes[1, 0].set_title("(c) Best score on `circle_packing` (after 3 iters)")
    axes[1, 0].axhline(bests_cp[0], color="grey", linestyle="--", linewidth=0.8,
                       label="vanilla baseline")
    axes[1, 0].legend()
    for b, v in zip(bars, bests_cp):
        axes[1, 0].annotate(f"{v:.3f}", (b.get_x() + b.get_width()/2, v),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=9)

    # (d) Speedup factor vs vanilla
    base_pc = summary["vanilla"]["per_call"]
    speedups = [base_pc / summary[a]["per_call"] for a in arms_order]
    bars = axes[1, 1].bar(xs, speedups, color=[color(a) for a in arms_order], width=0.6)
    axes[1, 1].set_xticks(xs); axes[1, 1].set_xticklabels(arms_order, rotation=15, ha="right")
    axes[1, 1].set_ylabel("Speedup vs vanilla")
    axes[1, 1].set_title("(d) Per-call latency speedup")
    axes[1, 1].axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    for b, v in zip(bars, speedups):
        axes[1, 1].annotate(f"{v:.1f}×", (b.get_x() + b.get_width()/2, v),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Real vLLM canonical 4-arm comparison\n"
                 "(2 benches concurrent, N_iter=3, K=8, Qwen3-4B-Instruct-2507)",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT / "fig_real_vllm_canonical.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 8 — FP8 vs bf16
# ---------------------------------------------------------------------------

def fig_fp8():
    csv_path = RES / "fp8_kv.csv"
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open()))
    Ks = sorted(set(int(r["K"]) for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.35
    xs = np.arange(len(Ks))
    for i, label in enumerate([("default-kv K=", "default-kv"), ("fp8-kv K=", "fp8-kv")]):
        prefix, color_key = label
        rates = [next((float(r["hit_rate"]) * 100 for r in rows
                        if r["label"].startswith(prefix) and int(r["K"]) == K), 0)
                  for K in Ks]
        axes[0].bar(xs + (i - 0.5) * width, rates, width,
                    color=color(color_key), label=color_key)
    axes[0].set_xticks(xs); axes[0].set_xticklabels([f"K={k}" for k in Ks])
    axes[0].set_ylabel("Cache hit rate (%)")
    axes[0].set_title("(a) FP8 KV cache: hit rate unaffected\n(quantization is per-block, hashes unchanged)")
    axes[0].set_ylim(0, 100)
    axes[0].legend()

    for i, label in enumerate([("default-kv K=", "default-kv"), ("fp8-kv K=", "fp8-kv")]):
        prefix, color_key = label
        walls = [next((float(r["wall_s"]) for r in rows
                        if r["label"].startswith(prefix) and int(r["K"]) == K), 0)
                  for K in Ks]
        axes[1].bar(xs + (i - 0.5) * width, walls, width,
                    color=color(color_key), label=color_key)
    axes[1].set_xticks(xs); axes[1].set_xticklabels([f"K={k}" for k in Ks])
    axes[1].set_ylabel("Wall time (s)")
    axes[1].set_title("(b) Wall time approximately equal\n(no memory pressure to expose FP8's 2× headroom)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_fp8_vs_bf16.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 9 — Long-run convergence (single-bench N=8 and N=16)
# ---------------------------------------------------------------------------

def fig_long_run():
    """Bar chart: best score reached at increasing iter budget."""
    # Hardcoded from runs/long_run.log + runs/n16_run.log + earlier canonical
    runs = [
        ("vanilla, N=3",       0.364, 25.0,   16175,   2976),
        ("batched, N=3",       0.548, 43.9,  178032, 161136),
        ("spec_gated, N=3",    0.569, 55.5,  303424, 276720),
        ("all_optims, N=3",    0.604, 74.1,  499769, 470672),
        ("vanilla, N=8",       0.364, 71.9,   34848,   4736),
        ("all_optims, N=8",    0.594, 171.0, 891822, 848208),
        ("all_optims, N=16",   0.816, 343.9,1599549,1519008),
    ]
    labels = [r[0] for r in runs]
    scores = [r[1] for r in runs]
    walls = [r[2] for r in runs]
    queries = [r[3] for r in runs]
    hits = [r[4] for r in runs]
    rates = [h/q*100 if q>0 else 0 for h, q in zip(hits, queries)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    xs = np.arange(len(runs))

    arm_colors = []
    for label in labels:
        for arm in ("vanilla", "batched", "spec_gated", "all_optims"):
            if arm in label:
                arm_colors.append(color(arm if arm != "spec_gated" else "batched_speculative_gated"))
                break
        else:
            arm_colors.append("#7f7f7f")

    bars = axes[0].bar(xs, scores, color=arm_colors)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("Best score on circle_packing\n(seed = 0.364)")
    axes[0].axhline(0.364, color="red", linestyle="--", linewidth=0.8, label="seed")
    axes[0].axhline(1.000, color="green", linestyle=":", linewidth=0.8, label="AlphaEvolve target")
    axes[0].legend(loc="upper left")
    axes[0].set_title("(a) Best-score progression with optimization stack and iteration budget")
    for b, v in zip(bars, scores):
        axes[0].annotate(f"{v:.3f}", (b.get_x() + b.get_width()/2, v),
                         textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    axes[0].set_ylim(0, 1.05)

    bars2 = axes[1].bar(xs, rates, color=arm_colors)
    axes[1].set_xticks(xs); axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Cache hit rate (%)")
    axes[1].set_title("(b) Cache hit rate stays high through the long run")
    axes[1].set_ylim(0, 100)
    for b, v in zip(bars2, rates):
        axes[1].annotate(f"{v:.1f}%", (b.get_x() + b.get_width()/2, v),
                         textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_long_run.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 10 — Quality-vs-compute Pareto across all real-vLLM trials
# ---------------------------------------------------------------------------

def fig_pareto():
    """Best score vs total prefill compute (uncached prefill tokens),
    aggregated across the canonical and long runs."""
    # Pull from the consolidated CSVs.
    rows = []
    for csv_name in ["real_vllm_summary_canonical.csv", "real_vllm_summary.csv",
                     "real_vllm_summary_session2.csv"]:
        p = RES / csv_name
        if not p.exists():
            continue
        for r in csv.DictReader(p.open()):
            r["_source"] = csv_name
            rows.append(r)

    # Fold per-arm per-bench rows into individual points.
    fig, ax = plt.subplots(figsize=(10, 6))
    arm_marker = {"vanilla": "x", "batched": "o", "batched_aligned": "^",
                  "batched_prefetch": "s", "batched_speculative": "D",
                  "batched_speculative_gated": "P", "all_optims": "*"}
    seen = set()
    for r in rows:
        arm = r["arm"]
        bench = r["bench"]
        try:
            queries = float(r.get("vllm:prefix_cache_queries_total", 0) or 0)
            hits = float(r.get("vllm:prefix_cache_hits_total", 0) or 0)
            uncached_tokens = queries - hits
            best = float(r["best_score"] or 0)
            n_iter = int(r["N_iter"])
        except (KeyError, ValueError, TypeError):
            continue
        if uncached_tokens <= 0:
            continue
        marker = arm_marker.get(arm, ".")
        size = 80 + 20 * n_iter
        label_key = f"{arm}"
        ax.scatter(uncached_tokens, best, marker=marker, s=size,
                   color=color(arm), alpha=0.85,
                   edgecolor="black", linewidth=0.5,
                   label=label_key if label_key not in seen else None)
        seen.add(label_key)
    ax.set_xscale("log")
    ax.set_xlabel("Total uncached prefill tokens (lower is better; log scale)")
    ax.set_ylabel("Best score reached")
    ax.set_title("Quality-vs-compute trade-off across all real-vLLM trials\n"
                 "(marker shape = arm; size = N_iter; closer to top-left = better)")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pareto.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 11 — Speculation gating: pre-cache vs post-cache at N=8
# ---------------------------------------------------------------------------

def fig_speculation_gating():
    paths = [
        (RES / "scaling_sweep_pre_metrics_cache.csv", "before async cache"),
        (RES / "scaling_sweep.csv",                    "after async cache"),
    ]
    if not all(p.exists() for p, _ in paths):
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.35
    arms = ["batched", "batched_speculative", "batched_speculative_gated"]
    for ax_idx, (csv_path, label) in enumerate(paths):
        rows = list(csv.DictReader(csv_path.open()))
        # Filter to N=8
        rows8 = [r for r in rows if int(r["N_concurrent"]) == 8]
        if not rows8:
            continue
        xs = np.arange(len(arms))
        tputs = [next((float(r["throughput_calls_per_s"]) for r in rows8
                        if r["arm"] == a), 0) for a in arms]
        rates = [next((float(r["hit_rate"]) * 100 for r in rows8
                        if r["arm"] == a), 0) for a in arms]
        cs = [color(a) for a in arms]
        axes[0].bar(xs + (ax_idx - 0.5) * width, tputs, width,
                    color=cs if ax_idx == 0 else cs, alpha=0.7 if ax_idx == 0 else 1.0,
                    label=label, edgecolor="black", linewidth=0.4,
                    hatch="" if ax_idx == 1 else "//")
    axes[0].set_xticks(np.arange(len(arms)))
    axes[0].set_xticklabels(arms, rotation=15, ha="right")
    axes[0].set_ylabel("Throughput (calls / s) at N=8")
    axes[0].set_title("(a) Effect of async /metrics cache\non gated speculation throughput")
    axes[0].legend(loc="best")

    # Right panel: explanatory text bar — show the speculation-fired counts
    # by reading n_calls per arm.
    rows_ac = list(csv.DictReader((RES / "scaling_sweep.csv").open()))
    rows8_ac = [r for r in rows_ac if int(r["N_concurrent"]) == 8]
    n_calls = [int(next((r["n_calls"] for r in rows8_ac if r["arm"] == a), 0))
               for a in arms]
    bars = axes[1].bar(np.arange(len(arms)), n_calls, color=[color(a) for a in arms])
    axes[1].set_xticks(np.arange(len(arms)))
    axes[1].set_xticklabels(arms, rotation=15, ha="right")
    axes[1].set_ylabel("Total LLM calls at N=8 cohort")
    axes[1].set_title("(b) Speculative call volume\n(gating cuts speculative volume to ~baseline)")
    for b, v in zip(bars, n_calls):
        axes[1].annotate(f"{v}", (b.get_x() + b.get_width()/2, v),
                         textcoords="offset points", xytext=(0, 3),
                         ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_speculation_gating.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 12 — Stacked optimization win (cumulative speedup ladder)
# ---------------------------------------------------------------------------

def fig_optim_ladder():
    """Show how each optimization stacks on top of the previous one."""
    csv_path = RES / "real_vllm_summary_canonical.csv"
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open()))
    arms = ["vanilla", "batched", "batched_speculative_gated", "all_optims"]
    arm_summaries = {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        wall = max(float(r["wall_s"]) for r in rs)
        queries = float(rs[0].get("vllm:prefix_cache_queries_total", 0) or 0)
        hits = float(rs[0].get("vllm:prefix_cache_hits_total", 0) or 0)
        n_calls = int(float(rs[0].get("vllm:request_success_total", 0) or 0))
        arm_summaries[arm] = {
            "wall": wall, "calls": n_calls,
            "rate": hits / queries if queries > 0 else 0,
            "per_call": wall / max(1, n_calls),
        }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Per-call latency reduction ladder
    arms_order = arms
    pcs = [arm_summaries[a]["per_call"] for a in arms_order]
    speedups = [pcs[0] / p for p in pcs]
    bars = axes[0].bar(arms_order, speedups, color=[color(a) for a in arms_order])
    axes[0].set_ylabel("Speedup vs vanilla (per-call latency)")
    axes[0].set_title("(a) Cumulative per-call speedup as optimizations stack")
    axes[0].set_xticklabels(arms_order, rotation=15, ha="right")
    axes[0].axhline(1, color="grey", linestyle="--", linewidth=0.5)
    for b, v in zip(bars, speedups):
        axes[0].annotate(f"{v:.1f}×", (b.get_x() + b.get_width()/2, v),
                         textcoords="offset points", xytext=(0, 4),
                         ha="center", fontsize=11, fontweight="bold")

    # Hit rate ladder
    rates = [arm_summaries[a]["rate"] * 100 for a in arms_order]
    bars = axes[1].bar(arms_order, rates, color=[color(a) for a in arms_order])
    axes[1].set_ylabel("Cache hit rate (%)")
    axes[1].set_title("(b) Cumulative cache hit rate")
    axes[1].set_xticklabels(arms_order, rotation=15, ha="right")
    axes[1].set_ylim(0, 100)
    for b, v in zip(bars, rates):
        axes[1].annotate(f"{v:.1f}%", (b.get_x() + b.get_width()/2, v),
                         textcoords="offset points", xytext=(0, 4),
                         ha="center", fontsize=11, fontweight="bold")

    fig.suptitle("Optimization ladder (real vLLM, canonical 2-bench cohort)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_optim_ladder.png", bbox_inches="tight")
    plt.close(fig)


def fig_per_iter_trace():
    csv_path = RES / "per_iter_trace.csv"
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open()))
    arms = sorted(set(r["arm"] for r in rows), key=lambda a: ["vanilla", "batched", "all_optims"].index(a))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for arm in arms:
        rs = sorted([r for r in rows if r["arm"] == arm], key=lambda r: int(r["iteration"]))
        iters = [int(r["iteration"]) for r in rs]
        rates = [float(r["hit_rate"]) * 100 for r in rs]
        walls = [float(r["wall_s"]) for r in rs]
        bests = [float(r["best_score"]) for r in rs]
        c = color(arm)
        axes[0].plot(iters, rates, "-o", color=c, label=arm, linewidth=2, markersize=8)
        axes[1].plot(iters, walls, "-s", color=c, label=arm, linewidth=2, markersize=8)
        axes[2].plot(iters, bests, "-^", color=c, label=arm, linewidth=2, markersize=8)
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Per-iter cache hit rate (%)")
    axes[0].set_title("(a) Cache hit rate by iteration"); axes[0].set_ylim(-2, 102); axes[0].legend()
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Per-iter wall time (s)")
    axes[1].set_title("(b) Per-iter wall time\n(all_optims: iters 3-5 reuse speculation, ≈0s each)")
    axes[1].legend(); axes[1].set_yscale("symlog", linthresh=0.1)
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("Best score on circle_packing")
    axes[2].set_title("(c) Best score by iteration"); axes[2].axhline(0.364, color="grey", linestyle="--", linewidth=0.6)
    axes[2].legend()
    fig.suptitle("Per-iteration trace on `circle_packing` (N=6, K=8 for batched arms)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_iter_trace.png", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_cache_only_synthetic()
    fig_scaling_sweep()
    fig_diff_aware()
    fig_adversarial_eviction()
    fig_persistent_kv()
    fig_real_vllm_canonical()
    fig_fp8()
    fig_long_run()
    fig_pareto()
    fig_speculation_gating()
    fig_optim_ladder()
    fig_per_iter_trace()
    print("Wrote figures under:", OUT)
    for p in sorted(OUT.glob("*.png")):
        print("  -", p.name)


if __name__ == "__main__":
    main()
