"""
Visualize AdaEvolve k-sweep results across wall-clock time checkpoints.
- Signal processing: filter rows with global_best_score > 1 (outliers)
- Circle packing: zoomed y-axis so curves are differentiable
- Candidates: number of JSONL rows before each wall-clock checkpoint
- KV cache hit rate saved as a separate PDF
"""

import json
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.scale import FuncScale

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("/home/dw_lyu/scalable_ai_class/results")
PLOTS_DIR   = RESULTS_DIR / "plots"
CHECKPOINTS = [0] + list(range(900, 7201, 900))   # 0, 900, 1800, …, 7200 sec
INITIAL_ELAPSED = -1.0   # synthetic anchor placed just before t=0
K_VALUES    = [1, 2, 4, 8]
RUNS        = [1, 2, 3]
BENCHMARKS  = ["circle_packing", "signal_processing"]
K_COLORS    = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c", 8: "#d62728"}
K_MARKERS   = {1: "o", 2: "s", 4: "^", 8: "D"}

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_file(benchmark: str, k: int, run: int) -> Optional[Path]:
    pattern = str(RESULTS_DIR / f"{benchmark}_k{k}_run{run}" / "adaevolve_iteration_stats_*.jsonl")
    matches = [p for p in glob.glob(pattern) if "_spec" not in p]
    return Path(matches[0]) if matches else None


def load_rows(path: Path, score_cap: Optional[float] = None) -> List[dict]:
    """Load JSONL rows, optionally filtering out rows where global_best_score > score_cap."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if score_cap is not None and row["global"]["global_best_score"] > score_cap:
                continue
            rows.append(row)
    return rows


def build_series(rows: List[dict],
                 initial_score: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (elapsed_seconds, best_score) aligned by row index.
    t0 is shifted back 0.5s so all actual rows have elapsed > 0, keeping
    elapsed=0 reserved for the synthetic anchor (initial_score at t=0).
    """
    if not rows:
        return np.array([]), np.array([])
    t0 = datetime.fromisoformat(rows[0]["timestamp"]) - timedelta(seconds=0.5)
    elapsed, scores = [], []
    if initial_score is not None:
        elapsed.append(INITIAL_ELAPSED)
        scores.append(initial_score)
    for row in rows:
        t = datetime.fromisoformat(row["timestamp"])
        elapsed.append((t - t0).total_seconds())
        scores.append(row["global"]["global_best_score"])
    return np.array(elapsed), np.array(scores)


def compute_avg_time_per_candidate(rows: List[dict]) -> float:
    """Return mean wall-clock LLM time per candidate across all batches.

    All K children in a batch share the same llm_generation_time_seconds.
    We group consecutive rows by that value to recover K, then compute
    llm_time / K for each batch and average across batches.
    """
    batch_times: List[float] = []
    prev_t = None
    batch_size = 0
    for row in rows:
        t = row.get("iteration_result", {}).get("llm_generation_time_seconds")
        if t is None:
            continue
        if t == prev_t:
            batch_size += 1
        else:
            if prev_t is not None and batch_size > 0:
                batch_times.append(prev_t / batch_size)
            prev_t = t
            batch_size = 1
    if prev_t is not None and batch_size > 0:
        batch_times.append(prev_t / batch_size)
    return float(np.mean(batch_times)) if batch_times else np.nan


def compute_avg_eval_time(rows: List[dict]) -> float:
    """Return mean eval time per candidate across all rows."""
    times = [r.get("iteration_result", {}).get("eval_time_seconds")
             for r in rows]
    times = [t for t in times if t is not None]
    return float(np.mean(times)) if times else np.nan


def compute_improvement_rate(rows: List[dict]) -> float:
    """Return fraction of candidates that improved the global best score."""
    if len(rows) < 2:
        return np.nan
    improvements = 0
    prev_best = None
    for row in rows:
        best = row.get("global", {}).get("global_best_score")
        if best is None:
            continue
        if prev_best is not None and best > prev_best + 1e-9:
            improvements += 1
        prev_best = best
    return improvements / len(rows)


def compute_overall_hit_rate(rows: List[dict]) -> float:
    """Return total KV cache hits / total queries across the full run.

    All K children in a batch share the same queries_delta / hits_delta, so
    we deduplicate by only counting a batch once (when the values change).
    """
    total_q = total_h = 0.0
    prev_q = prev_h = None
    for row in rows:
        kv = row.get("iteration_result", {}).get("kv_cache", {})
        q = kv.get("queries_delta")
        h = kv.get("hits_delta")
        if q is None or h is None:
            continue
        if q != prev_q or h != prev_h:
            total_q += q
            total_h += h
            prev_q, prev_h = q, h
    return total_h / total_q if total_q > 0 else np.nan


def sample_at_checkpoints(elapsed: np.ndarray, scores: np.ndarray,
                           checkpoints: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    At each checkpoint return:
      - best_score: last score at or before that time
      - row_count:  number of rows at or before that time
    """
    actual = elapsed > 0   # False for the synthetic anchor at INITIAL_ELAPSED
    score_vals, count_vals = [], []
    for cp in checkpoints:
        if cp == 0:
            anchor = elapsed <= 0
            score_vals.append(scores[anchor][-1] if anchor.any() else np.nan)
            count_vals.append(0.0)
        else:
            mask = elapsed <= cp
            if mask.any():
                score_vals.append(scores[mask][-1])
                count_vals.append(float(actual[mask].sum()))
            else:
                score_vals.append(np.nan)
                count_vals.append(np.nan)
    return np.array(score_vals, dtype=float), np.array(count_vals, dtype=float)


def make_multi_gap_scale(gap_pairs, gap_frac=0.04):
    """Piecewise-linear scale compressing each (gap_lo, gap_hi) interval.

    Returns (fwd, inv, display_boundaries) where display_boundaries is a list
    of (adj_lo, adj_hi_d) display-space y pairs, one per gap — used to place
    the axis-break markers.
    """
    pairs = sorted(gap_pairs)
    spans = [hi - lo for lo, hi in pairs]
    comps = [s * gap_frac for s in spans]

    disp_bounds = []
    cum_red = 0.0
    for (lo, hi), comp, span in zip(pairs, comps, spans):
        adj_lo   = lo - cum_red
        adj_hi_d = adj_lo + comp
        disp_bounds.append((adj_lo, adj_hi_d))
        cum_red += span - comp

    def fwd(y):
        y   = np.asarray(y, dtype=float)
        out = y.copy()
        cum = 0.0
        for (lo, hi), comp, span in zip(pairs, comps, spans):
            adj_lo = lo - cum
            adj_hi = hi - cum
            m_in    = (out >= adj_lo) & (out < adj_hi)
            m_above = out >= adj_hi
            out[m_in]    = adj_lo + (out[m_in] - adj_lo) / span * comp
            out[m_above] -= (span - comp)
            cum += span - comp
        return out

    def inv(y):
        y   = np.asarray(y, dtype=float)
        out = y.copy()
        cum = sum(s - c for s, c in zip(spans, comps))
        for (lo, hi), comp, span in zip(reversed(pairs), reversed(comps), reversed(spans)):
            cum     -= (span - comp)
            adj_lo   = lo - cum
            adj_hi_d = adj_lo + comp
            m_in     = (out >= adj_lo) & (out < adj_hi_d)
            m_above  = out >= adj_hi_d
            out[m_in]    = adj_lo + (out[m_in] - adj_lo) * span / comp
            out[m_above] += (span - comp)
        return out

    return fwd, inv, disp_bounds


def apply_cp_score_gap_scale(ax, anchor, gap_lo=0.405, gap_hi=0.795):
    """Apply a single compressed gap [gap_lo, gap_hi] to the circle-packing score axis."""
    if anchor is None:
        return
    gap = (gap_lo, gap_hi)
    if gap_lo + 1e-3 >= gap_hi:
        return

    fwd, inv, disp_bounds = make_multi_gap_scale([gap])
    y_lo, y_hi = ax.get_ylim()

    ax.set_yscale(FuncScale(ax, (fwd, inv)))
    ax.set_ylim(y_lo, y_hi)

    # Ticks: anchor, 0.80 cutoff, then every 0.05 above
    ticks = [round(anchor, 2), 0.80]
    ticks += np.arange(0.85, y_hi + 1e-9, 0.05).tolist()
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Single axis-break marker at the top of the compressed gap (≈ 0.80 tick)
    y_lo_d = fwd(np.array([y_lo]))[0]
    y_hi_d = fwd(np.array([y_hi]))[0]
    for _adj_lo, adj_hi_d in disp_bounds:
        y_ax = (adj_hi_d - y_lo_d) / (y_hi_d - y_lo_d)
        kw = dict(transform=ax.transAxes, color="k", lw=1.3, clip_on=False)
        ax.plot([-0.025, 0.025], [y_ax - 0.012, y_ax + 0.012], **kw)


def plot_panel(ax, bench, metric, ylabel, title, data, x_hrs, fs=10):
    ax.set_title(title, fontsize=fs, fontweight="bold")
    all_means = []
    for k in K_VALUES:
        mat = data[bench][k][metric]
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat, axis=0)
        valid = ~np.all(np.isnan(mat), axis=0)
        xv, mv, sv = x_hrs[valid], mean[valid], std[valid]
        all_means.append(mv)
        color  = K_COLORS[k]
        marker = K_MARKERS[k]
        ax.plot(xv, mv, marker=marker, markersize=6, linewidth=1.5,
                color=color, label=f"k={k}", zorder=3)
        ax.fill_between(xv, mv - sv, mv + sv, alpha=0.12, color=color)
        ax.errorbar(xv, mv, yerr=sv, fmt="none", ecolor=color,
                    elinewidth=1.0, capsize=2, alpha=0.7)

    if bench == "circle_packing" and metric == "scores":
        finite_vals = np.concatenate([m for m in all_means if len(m) > 0])
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        if len(finite_vals):
            y_lo = max(0.0, np.nanmin(finite_vals) - 0.02)
            y_hi = min(1.005, np.nanmax(finite_vals) + 0.005)
            ax.set_ylim(y_lo, y_hi)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    if metric == "hit_rates":
        ax.set_ylim(0.0, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

    ax.set_xlabel("Wall-Clock Time (hours)", fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.set_xticks(x_hrs)
    ax.tick_params(axis="x", labelrotation=45, labelsize=fs - 2)
    ax.tick_params(axis="y", labelsize=fs - 2)
    if metric == "scores":
        legend_loc = "lower right"
    elif metric == "hit_rates":
        legend_loc = "best"
    else:
        legend_loc = "upper left"
    ax.legend(fontsize=fs - 1, loc=legend_loc, framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.35)


# ── Per-benchmark initial score for the shared t=0 anchor ────────────────────
# Use the minimum first-row score across all k/runs so every curve starts from
# the same point at t=0, matching the strategy in visualize_k4_spec_comparison.

initial_scores: dict = {}
for _bench in BENCHMARKS:
    _score_cap = 1.0 if _bench == "signal_processing" else None
    _iter1: List[float] = []
    for _k in K_VALUES:
        for _run in RUNS:
            _p = find_file(_bench, _k, _run)
            if _p:
                _raw = load_rows(_p)   # no cap — want the true initial score
                if _raw:
                    _s = _raw[0]["global"]["global_best_score"]
                    if _score_cap is None or _s <= _score_cap:
                        _iter1.append(_s)
    initial_scores[_bench] = float(min(_iter1)) if _iter1 else None

# ── Collect data ──────────────────────────────────────────────────────────────

# data[bench][k] = { "scores": (3, n_cp), "counts": (3, n_cp), "hit_rates": (3, n_cp) }
data = {}
for bench in BENCHMARKS:
    data[bench] = {}
    score_cap = 1.0 if bench == "signal_processing" else None
    for k in K_VALUES:
        score_mat, count_mat, hit_rate_per_run, gen_time_per_run, eval_time_per_run, impr_rate_per_run = [], [], [], [], [], []
        for run in RUNS:
            path = find_file(bench, k, run)
            if path is None:
                print(f"  MISSING: {bench} k={k} run={run}")
                score_mat.append([np.nan] * len(CHECKPOINTS))
                count_mat.append([np.nan] * len(CHECKPOINTS))
                hit_rate_per_run.append(np.nan)
                gen_time_per_run.append(np.nan)
                eval_time_per_run.append(np.nan)
                impr_rate_per_run.append(np.nan)
                continue
            rows = load_rows(path, score_cap=score_cap)
            elapsed, scores = build_series(rows, initial_score=initial_scores[bench])
            sv, cv = sample_at_checkpoints(elapsed, scores, CHECKPOINTS)
            score_mat.append(sv)
            count_mat.append(cv)
            hit_rate_per_run.append(compute_overall_hit_rate(rows))
            gen_time_per_run.append(compute_avg_time_per_candidate(rows))
            eval_time_per_run.append(compute_avg_eval_time(rows))
            impr_rate_per_run.append(compute_improvement_rate(rows))
            span = elapsed[-1] / 3600 if len(elapsed) else 0
            print(f"  {bench} k={k} run={run}: {len(rows)} rows, "
                  f"span={span:.2f}h, final_score={scores[-1]:.4f}, "
                  f"hit_rate={hit_rate_per_run[-1]:.3f}, "
                  f"gen_time/cand={gen_time_per_run[-1]:.1f}s, "
                  f"impr_rate={impr_rate_per_run[-1]:.3f}")
        data[bench][k] = {
            "scores":     np.array(score_mat),
            "counts":     np.array(count_mat),
            "hit_rates":  np.array(hit_rate_per_run),
            "gen_times":  np.array(gen_time_per_run),
            "eval_times": np.array(eval_time_per_run),
            "impr_rates": np.array(impr_rate_per_run),
        }

x_hrs = np.array(CHECKPOINTS) / 3600

# ── Spec k4 curve (circle packing only, single run) ──────────────────────────
SPEC_COLOR  = "#9467bd"   # purple, distinct from k=4 green
SPEC_MARKER = "*"

# _spec_path = RESULTS_DIR / "circle_packing_k4_run1_spec" / \
#              "adaevolve_iteration_stats_20260511_060002.jsonl"
# _spec_rows   = load_rows(_spec_path)
# _spec_elap, _spec_scores = build_series(_spec_rows)
# _spec_sv, _  = sample_at_checkpoints(_spec_elap, _spec_scores, CHECKPOINTS)
# # Mask checkpoints beyond the actual run duration so the curve stops naturally.
# _spec_max_t  = _spec_elap[-1] if len(_spec_elap) else 0.0
# _spec_sv[np.array(CHECKPOINTS) > _spec_max_t] = np.nan
# _spec_valid   = ~np.isnan(_spec_sv)

# ── Plot 1: scores + candidates ───────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# fig.suptitle(
#     "AdaEvolve k-Sweep  |  mean ± std over 3 runs",
#     fontsize=13, fontweight="bold", y=1.01,
# )

plot_cfg = [
    (0, 0, "circle_packing",    "scores", "Global Best Score",           "Circle Packing — Best Score"),
    (0, 1, "circle_packing",    "counts", "Candidates Generated (rows)", "Circle Packing — Candidates Generated"),
    (1, 0, "signal_processing", "scores", "Global Best Score",           "Signal Processing — Best Score"),
    (1, 1, "signal_processing", "counts", "Candidates Generated (rows)", "Signal Processing — Candidates Generated"),
]

for row, col, bench, metric, ylabel, title in plot_cfg:
    plot_panel(axes[row][col], bench, metric, ylabel, title, data, x_hrs)
    # if bench == "circle_packing" and metric == "scores":
    #     _add_spec_k4_curve(axes[row][col])
    #     axes[row][col].legend(fontsize=9, loc="lower right", framealpha=0.8)

apply_cp_score_gap_scale(axes[0][0], initial_scores.get("circle_packing"))

plt.tight_layout()
out1_pdf = PLOTS_DIR / "k_sweep_wallclock.pdf"
plt.savefig(out1_pdf, bbox_inches="tight")
print(f"\nSaved → {out1_pdf}")
out1 = PLOTS_DIR / "k_sweep_wallclock.png"
plt.savefig(out1, bbox_inches="tight")
print(f"Saved → {out1}")
plt.close()

# ── Plot 2: KV cache hit rate — grouped bar chart ────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# fig.suptitle(
#     "AdaEvolve k-Sweep — Overall KV Cache Hit Rate  |  mean ± std over 3 runs",
#     fontsize=13, fontweight="bold", y=1.02,
# )

bar_titles = {
    "circle_packing":    "Circle Packing",
    "signal_processing": "Signal Processing",
}

x = np.arange(len(K_VALUES))
bar_width = 0.5

for ax, bench in zip(axes, BENCHMARKS):
    means, stds = [], []
    for k in K_VALUES:
        hr = data[bench][k]["hit_rates"]
        means.append(np.nanmean(hr))
        stds.append(np.nanstd(hr))

    bars = ax.bar(
        x, means, bar_width,
        yerr=stds, capsize=6,
        color=[K_COLORS[k] for k in K_VALUES],
        edgecolor="white", linewidth=0.5,
        error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5},
    )

    ax.set_title(bar_titles[bench], fontsize=11, fontweight="bold")
    ax.set_xlabel("K (candidates per iteration)", fontsize=10)
    ax.set_ylabel("Overall KV Cache Hit Rate", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_VALUES], fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.02,
            f"{mean*100:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

plt.tight_layout()
out2 = PLOTS_DIR / "k_sweep_kv_cache_hit_rate.pdf"
plt.savefig(out2, bbox_inches="tight")
print(f"Saved → {out2}")
plt.close()

# ── Plot 3: avg LLM generation time per candidate ─────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# fig.suptitle(
#     "AdaEvolve k-Sweep — Avg LLM Generation Time per Candidate  |  mean ± std over 3 runs",
#     fontsize=13, fontweight="bold", y=1.02,
# )

gen_titles = {
    "circle_packing":    "Circle Packing",
    "signal_processing": "Signal Processing",
}

for ax, bench in zip(axes, BENCHMARKS):
    means, stds = [], []
    for k in K_VALUES:
        gt = data[bench][k]["gen_times"]
        means.append(np.nanmean(gt))
        stds.append(np.nanstd(gt))

    bars = ax.bar(
        x, means, bar_width,
        yerr=stds, capsize=6,
        color=[K_COLORS[k] for k in K_VALUES],
        edgecolor="white", linewidth=0.5,
        error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5},
    )

    ax.set_title(gen_titles[bench], fontsize=11, fontweight="bold")
    ax.set_xlabel("K (candidates per iteration)", fontsize=10)
    ax.set_ylabel("Avg LLM Time per Candidate (s)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_VALUES], fontsize=10)
    max_top = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(0, max_top * 1.15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + max_top * 0.02,
            f"{mean:.1f}s",
            ha="center", va="bottom", fontsize=9,
        )

plt.tight_layout()
out3 = PLOTS_DIR / "k_sweep_gen_time_per_candidate.pdf"
plt.savefig(out3, bbox_inches="tight")
print(f"Saved → {out3}")
plt.close()

# ── Plot 4: combined grid for paper (4 rows × 2 cols) ────────────────────────
# Rows: best score | candidates | KV cache hit rate | gen time per candidate
# Cols: circle_packing | signal_processing

def draw_bar_panel(ax, bench, metric, ylabel, kind, fs=10):
    """Reusable bar panel for hit_rate or gen_time (data unchanged)."""
    means = [np.nanmean(data[bench][k][metric]) for k in K_VALUES]
    stds  = [np.nanstd(data[bench][k][metric])  for k in K_VALUES]
    bars = ax.bar(
        x, means, bar_width,
        yerr=stds, capsize=4,
        color=[K_COLORS[k] for k in K_VALUES],
        edgecolor="white", linewidth=0.5,
        error_kw={"elinewidth": 1.2, "ecolor": "black", "capthick": 1.2},
    )
    ax.set_xlabel("K (candidates per iteration)", fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_VALUES], fontsize=fs)
    ax.tick_params(axis="y", labelsize=fs - 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    if kind == "hit_rate":
        ax.set_ylim(0.0, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mean + std + 0.02, f"{mean*100:.1f}%",
                    ha="center", va="bottom", fontsize=fs - 1)
    else:
        max_top = max(m + s for m, s in zip(means, stds))
        ax.set_ylim(0, max_top * 1.15)
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mean + std + max_top * 0.02, f"{mean:.1f}s",
                    ha="center", va="bottom", fontsize=fs - 1)


bench_labels = {
    "circle_packing":    "Circle Packing",
    "signal_processing": "Signal Processing",
}
score_subtitles = {
    "circle_packing":    "Best Score",
    "signal_processing": "Best Score",
}

GFS = 8  # grid font size

fig, axes = plt.subplots(4, 2, figsize=(12, 9))
# fig.suptitle(
#     "AdaEvolve k-Sweep  |  mean ± std over 3 runs",
#     fontsize=10, fontweight="bold",
# )

# Column headers
for col, bench in enumerate(BENCHMARKS):
    axes[0][col].set_title(bench_labels[bench], fontsize=GFS + 1, fontweight="bold", pad=6)

# Row 0 & 1: line plots
line_cfgs = [
    ("scores", "Global Best Score",           score_subtitles),
    ("counts", "Candidates Generated (rows)", {b: "Candidates Generated" for b in BENCHMARKS}),
]
for row, (metric, ylabel, subtitles) in enumerate(line_cfgs):
    for col, bench in enumerate(BENCHMARKS):
        plot_panel(axes[row][col], bench, metric, ylabel, subtitles[bench], data, x_hrs, fs=GFS)
        # if bench == "circle_packing" and metric == "scores":
        #     _add_spec_k4_curve(axes[row][col])
        #     axes[row][col].legend(fontsize=GFS - 1, loc="lower right", framealpha=0.8)

apply_cp_score_gap_scale(axes[0][0], initial_scores.get("circle_packing"))

# Rows 2 & 3: bar charts
for row, (metric, ylabel, kind) in enumerate(
    [("hit_rates", "KV Cache Hit Rate", "hit_rate"),
     ("gen_times", "Avg LLM Time / Candidate (s)", "gen_time")],
    start=2,
):
    for col, bench in enumerate(BENCHMARKS):
        draw_bar_panel(axes[row][col], bench, metric, ylabel, kind, fs=GFS)

# Row labels
row_labels = ["Best Score", "Candidates\nGenerated", "KV Cache\nHit Rate", "Gen Time\nper Cand."]
for row, label in enumerate(row_labels):
    axes[row][0].annotate(
        label, xy=(0, 0.5), xytext=(-52, 0),
        xycoords="axes fraction", textcoords="offset points",
        fontsize=GFS, fontweight="bold", ha="center", va="center", rotation=90,
    )

plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.6)
out4 = PLOTS_DIR / "k_sweep_paper_grid.pdf"
plt.savefig(out4, bbox_inches="tight")
print(f"Saved → {out4}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Fixed-candidate-count regime: x-axis = total candidates generated
# ══════════════════════════════════════════════════════════════════════════════

CAND_CHECKPOINTS = list(range(20, 1601, 20))
x_cands = np.array(CAND_CHECKPOINTS)

# ── Collect candidate-normalised score data ───────────────────────────────────

cand_data = {}
for bench in BENCHMARKS:
    cand_data[bench] = {}
    score_cap = 1.0 if bench == "signal_processing" else None
    for k in K_VALUES:
        score_mat = []
        for run in RUNS:
            path = find_file(bench, k, run)
            if path is None:
                score_mat.append([np.nan] * len(CAND_CHECKPOINTS))
                continue
            rows = load_rows(path, score_cap=score_cap)
            cands = np.arange(1, len(rows) + 1, dtype=float)
            scores = np.array([r["global"]["global_best_score"] for r in rows])
            sv, _ = sample_at_checkpoints(cands, scores, CAND_CHECKPOINTS)
            score_mat.append(sv)
        cand_data[bench][k] = {"scores": np.array(score_mat)}


CAND_XLIM = {"circle_packing": 100, "signal_processing": 400}


def plot_panel_cand(ax, bench, ylabel, title, fs=10):
    """Line plot with total candidates generated on the x-axis."""
    ax.set_title(title, fontsize=fs, fontweight="bold")
    all_means = []
    xlim = CAND_XLIM[bench]
    for k in K_VALUES:
        mat = cand_data[bench][k]["scores"]
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat, axis=0)
        valid = ~np.all(np.isnan(mat), axis=0) & (x_cands <= xlim)
        xv, mv, sv = x_cands[valid], mean[valid], std[valid]
        all_means.append(mv)
        ax.plot(xv, mv, marker=K_MARKERS[k], markersize=4, linewidth=1.5,
                color=K_COLORS[k], label=f"K={k}", zorder=3)
        ax.fill_between(xv, mv - sv, mv + sv, alpha=0.12, color=K_COLORS[k])
        ax.errorbar(xv, mv, yerr=sv, fmt="none", ecolor=K_COLORS[k],
                    elinewidth=1.0, capsize=2, alpha=0.7)

    if bench == "circle_packing":
        finite_vals = np.concatenate([m for m in all_means if len(m) > 0])
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        if len(finite_vals):
            ax.set_ylim(max(0.0, np.nanmin(finite_vals) - 0.02),
                        min(1.005, np.nanmax(finite_vals) + 0.005))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    ax.set_xlabel("Total Candidates Generated", fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.tick_params(axis="x", labelrotation=45, labelsize=fs - 2)
    ax.tick_params(axis="y", labelsize=fs - 2)
    ax.legend(fontsize=fs - 1, loc="lower right", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.35)


# ── fixed_num_cand Plot 1: best score vs candidates ───────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cand_score_titles = {
    "circle_packing":    "Circle Packing — Best Score",
    "signal_processing": "Signal Processing — Best Score",
}
for ax, bench in zip(axes, BENCHMARKS):
    plot_panel_cand(ax, bench, "Global Best Score", cand_score_titles[bench])

plt.tight_layout()
out_c1 = PLOTS_DIR / "fixed_num_cand_k_sweep_wallclock.png"
plt.savefig(out_c1, bbox_inches="tight")
print(f"\nSaved → {out_c1}")
plt.close()

# ── fixed_num_cand Plot 2: KV cache hit rate ──────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, bench in zip(axes, BENCHMARKS):
    ax.set_title(bench_labels[bench], fontsize=15, fontweight="bold")
    draw_bar_panel(ax, bench, "hit_rates", "Overall KV Cache Hit Rate", "hit_rate", fs=14)

plt.tight_layout()
out_c2 = PLOTS_DIR / "fixed_num_cand_k_sweep_kv_cache_hit_rate.pdf"
plt.savefig(out_c2, bbox_inches="tight")
print(f"Saved → {out_c2}")
plt.savefig(out_c2.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"Saved → {out_c2.with_suffix('.png')}")
plt.close()

# ── fixed_num_cand Plot 3: gen time per candidate ─────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, bench in zip(axes, BENCHMARKS):
    ax.set_title(bench_labels[bench], fontsize=15, fontweight="bold")
    draw_bar_panel(ax, bench, "gen_times", "Avg LLM Time per Candidate (s)", "gen_time", fs=14)

plt.tight_layout()
out_c3 = PLOTS_DIR / "fixed_num_cand_k_sweep_gen_time_per_candidate.pdf"
plt.savefig(out_c3, bbox_inches="tight")
print(f"Saved → {out_c3}")
plt.savefig(out_c3.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"Saved → {out_c3.with_suffix('.png')}")
plt.close()

# ── fixed_num_cand Plot 4: paper grid (3 rows × 2 cols) ──────────────────────
# Rows: best score vs candidates | KV cache hit rate | gen time per candidate
# Cols: circle_packing | signal_processing

fig, axes = plt.subplots(3, 2, figsize=(12, 7))

for col, bench in enumerate(BENCHMARKS):
    axes[0][col].set_title(bench_labels[bench], fontsize=GFS + 1, fontweight="bold", pad=6)

# Row 0: best score vs candidates
for col, bench in enumerate(BENCHMARKS):
    plot_panel_cand(axes[0][col], bench, "Global Best Score", cand_score_titles[bench], fs=GFS)

# Row 1: KV cache hit rate
for col, bench in enumerate(BENCHMARKS):
    draw_bar_panel(axes[1][col], bench, "hit_rates", "KV Cache Hit Rate", "hit_rate", fs=GFS)

# Row 2: gen time per candidate
for col, bench in enumerate(BENCHMARKS):
    draw_bar_panel(axes[2][col], bench, "gen_times", "Avg LLM Time / Candidate (s)", "gen_time", fs=GFS)

# Row labels
cand_row_labels = ["Best Score\nvs Candidates", "KV Cache\nHit Rate", "Gen Time\nper Cand."]
for row, label in enumerate(cand_row_labels):
    axes[row][0].annotate(
        label, xy=(0, 0.5), xytext=(-52, 0),
        xycoords="axes fraction", textcoords="offset points",
        fontsize=GFS, fontweight="bold", ha="center", va="center", rotation=90,
    )

plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.6)
out_c4 = PLOTS_DIR / "fixed_num_cand_k_sweep_paper_grid.pdf"
plt.savefig(out_c4, bbox_inches="tight")
print(f"Saved → {out_c4}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Plot: Eval time vs LLM time breakdown (stacked bar)
# ══════════════════════════════════════════════════════════════════════════════

LLM_COLOR  = "#4878d0"
EVAL_COLOR = "#ee854a"

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, bench in zip(axes, BENCHMARKS):
    llm_means  = [np.nanmean(data[bench][k]["gen_times"])  for k in K_VALUES]
    llm_stds   = [np.nanstd(data[bench][k]["gen_times"])   for k in K_VALUES]
    eval_means = [np.nanmean(data[bench][k]["eval_times"]) for k in K_VALUES]
    eval_stds  = [np.nanstd(data[bench][k]["eval_times"])  for k in K_VALUES]
    total_means = [l + e for l, e in zip(llm_means, eval_means)]
    total_stds  = [np.sqrt(ls**2 + es**2) for ls, es in zip(llm_stds, eval_stds)]

    ax.bar(x, llm_means,  bar_width, label="LLM Generation",
           color=LLM_COLOR,  edgecolor="white", linewidth=0.5)
    ax.bar(x, eval_means, bar_width, bottom=llm_means, label="Evaluation",
           color=EVAL_COLOR, edgecolor="white", linewidth=0.5)
    ax.errorbar(x, total_means, yerr=total_stds, fmt="none",
                ecolor="black", elinewidth=1.5, capsize=5, capthick=1.5)

    ax.set_title(bench_labels[bench], fontsize=11, fontweight="bold")
    ax.set_xlabel("K (candidates per iteration)", fontsize=10)
    ax.set_ylabel("Avg Time per Candidate (s)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_VALUES], fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    max_top = max(t + s for t, s in zip(total_means, total_stds))
    ax.set_ylim(0, max_top * 1.2)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    for i, (lm, em, tm, ts) in enumerate(zip(llm_means, eval_means, total_means, total_stds)):
        ax.text(i, tm + ts + max_top * 0.02, f"{tm:.1f}s",
                ha="center", va="bottom", fontsize=9)

plt.tight_layout()
out_tb = PLOTS_DIR / "k_sweep_time_breakdown.pdf"
plt.savefig(out_tb, bbox_inches="tight")
print(f"\nSaved → {out_tb}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Plot: Per-candidate improvement rate
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, bench in zip(axes, BENCHMARKS):
    means = [np.nanmean(data[bench][k]["impr_rates"]) for k in K_VALUES]
    stds  = [np.nanstd(data[bench][k]["impr_rates"])  for k in K_VALUES]

    bars = ax.bar(
        x, means, bar_width,
        yerr=stds, capsize=6,
        color=[K_COLORS[k] for k in K_VALUES],
        edgecolor="white", linewidth=0.5,
        error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5},
    )
    ax.set_title(bench_labels[bench], fontsize=11, fontweight="bold")
    ax.set_xlabel("K (candidates per iteration)", fontsize=10)
    ax.set_ylabel("Improvement Rate (per candidate)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_VALUES], fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
    max_top = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(0, max_top * 1.2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + std + max_top * 0.02,
                f"{mean*100:.1f}%",
                ha="center", va="bottom", fontsize=9)

    # Annotate each bar with the mean number of candidates across runs
    for i, k in enumerate(K_VALUES):
        n_cands = [len(load_rows(find_file(bench, k, run),
                                 score_cap=1.0 if bench == "signal_processing" else None))
                   for run in RUNS
                   if find_file(bench, k, run) is not None]
        mean_n = int(round(np.mean(n_cands))) if n_cands else 0
        ax.text(i, -max_top * 0.07, f"n≈{mean_n}",
                ha="center", va="top", fontsize=8, color="dimgray")

    ax.set_ylim(-max_top * 0.12, max_top * 1.2)

plt.tight_layout()
out_ir = PLOTS_DIR / "k_sweep_improvement_rate.pdf"
plt.savefig(out_ir, bbox_inches="tight")
print(f"Saved → {out_ir}")
plt.close()
