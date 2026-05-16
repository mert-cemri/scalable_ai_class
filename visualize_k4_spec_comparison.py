"""
Plots comparing k vs k+spec for circle packing (3 runs each), for k in {4, 8}:
  1. Wall-clock time vs best score
  2. Wall-clock time vs candidates generated
  3. Spec-hit vs non-spec-hit iteration timing (spec runs only)
"""

import json
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.scale import FuncScale

RESULTS_DIR = Path("/home/dw_lyu/scalable_ai_class/results")
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS     = [0] + list(range(900, 7201, 900))
x_hrs           = np.array(CHECKPOINTS) / 3600
INITIAL_ELAPSED = -1.0   # synthetic anchor placed just before t=0

K_COLORS  = {4: "#2ca02c", 8: "#d62728"}   # green / red (matches sweep plots)
K_MARKERS = {4: "^", 8: "D"}
SPEC_COLOR  = "#9467bd"
SPEC_MARKER = "*"
AUG_COLOR   = "#ff7f0e"   # orange — spec with augmented children
AUG_MARKER  = "P"
LLM_COLOR   = "#4878d0"
EVAL_COLOR  = "#ee854a"
ITER_CLIP   = 360.0

RUNS = [1, 2, 3]


# ── helpers ───────────────────────────────────────────────────────────────────

def find_jsonl(directory: Path, which: str = "last") -> Optional[Path]:
    matches = sorted(glob.glob(str(directory / "adaevolve_iteration_stats_*.jsonl")))
    if not matches:
        return None
    return Path(matches[0] if which == "first" else matches[-1])


def load_rows(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_series(rows: List[dict],
                 initial_score: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.array([]), np.array([])
    # Shift t0 half a second before the first row so actual rows all have elapsed > 0,
    # keeping the cp=0 bucket reserved for the synthetic anchor.
    t0 = datetime.fromisoformat(rows[0]["timestamp"]) - timedelta(seconds=0.5)
    elapsed: List[float] = []
    scores:  List[float] = []
    if initial_score is not None:
        elapsed.append(INITIAL_ELAPSED)   # -1.0, picked up by cp=0 only
        scores.append(initial_score)
    for row in rows:
        t = datetime.fromisoformat(row["timestamp"])
        elapsed.append((t - t0).total_seconds())
        scores.append(row["global"]["global_best_score"])
    return np.array(elapsed), np.array(scores)


def sample_at_checkpoints(elapsed: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = []
    for cp in CHECKPOINTS:
        mask = elapsed <= cp
        out.append(values[mask][-1] if mask.any() else np.nan)
    return np.array(out, dtype=float)


def count_at_checkpoints(elapsed: np.ndarray) -> np.ndarray:
    # Only count actual evaluated candidates (elapsed > 0); synthetic anchor is excluded.
    actual = elapsed[elapsed > 0]
    out = []
    for cp in CHECKPOINTS:
        if cp == 0:
            out.append(0.0)
        else:
            mask = actual <= cp
            out.append(float(mask.sum()) if mask.any() else np.nan)
    return np.array(out, dtype=float)


def mask_beyond_run(arr: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
    result = arr.copy()
    actual = elapsed[elapsed > 0]
    max_t  = actual[-1] if len(actual) else 0.0
    result[np.array(CHECKPOINTS) > max_t + 600] = np.nan
    return result


def load_wallclock_mats(k: int, spec: bool, initial_score: Optional[float] = None,
                        which: str = "last"):
    score_mat, count_mat = [], []
    tag = f"circle_packing_k{k}_run{{run}}" + ("_spec" if spec else "")
    for run in RUNS:
        p = find_jsonl(RESULTS_DIR / tag.format(run=run), which=which)
        if p:
            rows = load_rows(p)
            elap, scores = build_series(rows, initial_score)
            sv = mask_beyond_run(sample_at_checkpoints(elap, scores), elap)
            cv = mask_beyond_run(count_at_checkpoints(elap), elap)
        else:
            print(f"MISSING: k{k}{'_spec' if spec else ''} run{run}")
            sv = np.full(len(CHECKPOINTS), np.nan)
            cv = np.full(len(CHECKPOINTS), np.nan)
        score_mat.append(sv)
        count_mat.append(cv)
    return np.array(score_mat), np.array(count_mat)


def load_timing_data(k: int, which: str = "last"):
    hit_iter, nohit_iter = [], []
    hit_llm,  nohit_llm  = [], []
    for run in RUNS:
        p = find_jsonl(RESULTS_DIR / f"circle_packing_k{k}_run{run}_spec", which=which)
        if p is None:
            continue
        for row in load_rows(p):
            ir = row.get("iteration_result", {})
            if not ir:
                continue
            it  = ir.get("iteration_time_seconds")
            llm = ir.get("llm_generation_time_seconds")
            if it is None:
                continue
            it = min(it, ITER_CLIP)
            if ir.get("speculation_hit", False):
                hit_iter.append(it)
                if llm is not None:
                    hit_llm.append(min(llm, ITER_CLIP))
            else:
                nohit_iter.append(it)
                if llm is not None:
                    nohit_llm.append(min(llm, ITER_CLIP))
    return hit_iter, nohit_iter, hit_llm, nohit_llm



def plot_mean_std(ax, mat, color, marker, label, zorder=3):
    mean  = np.nanmean(mat, axis=0)
    std   = np.nanstd(mat, axis=0)
    valid = ~np.all(np.isnan(mat), axis=0)
    xv, mv, sv = x_hrs[valid], mean[valid], std[valid]
    ax.plot(xv, mv, marker=marker, markersize=5, linewidth=1.8,
            color=color, label=label, zorder=zorder)
    ax.fill_between(xv, mv - sv, mv + sv, alpha=0.15, color=color)
    ax.errorbar(xv, mv, yerr=sv, fmt="none", ecolor=color,
                elinewidth=1.0, capsize=2, alpha=0.7)


def make_gap_scale(gap_lo: float, gap_hi: float, gap_frac: float = 0.04):
    """Piecewise-linear scale that squishes [gap_lo, gap_hi] to gap_frac of its span."""
    span = gap_hi - gap_lo
    comp = span * gap_frac

    def fwd(y):
        y = np.asarray(y, dtype=float)
        out = y.copy()
        m = (y > gap_lo) & (y < gap_hi)
        out[m]      = gap_lo + (y[m] - gap_lo) / span * comp
        out[y >= gap_hi] = gap_lo + comp + (y[y >= gap_hi] - gap_hi)
        return out

    def inv(y):
        y = np.asarray(y, dtype=float)
        out = y.copy()
        m = (y > gap_lo) & (y < gap_lo + comp)
        out[m]           = gap_lo + (y[m] - gap_lo) / comp * span
        out[y >= gap_lo + comp] = gap_hi + (y[y >= gap_lo + comp] - gap_lo - comp)
        return out

    return fwd, inv, comp


# ── generate plots for each k ─────────────────────────────────────────────────

for k in [4, 8]:
    base_color = K_COLORS[k]
    base_marker = K_MARKERS[k]

    # Shared t=0 anchor: worst (minimum) global_best_score seen at iteration 1
    # across all runs for this k — equals the initial program's score, same for
    # every run regardless of whether iter 1 immediately improved.
    iter1_scores = []
    for _spec in [False, True]:
        for _run in RUNS:
            _tag = f"circle_packing_k{k}_run{_run}" + ("_spec" if _spec else "")
            _whichs = (["first", "last"] if k == 8 and _spec else ["last"])
            for _which in _whichs:
                _p = find_jsonl(RESULTS_DIR / _tag, which=_which)
                if _p:
                    _r = load_rows(_p)
                    if _r:
                        iter1_scores.append(_r[0]["global"]["global_best_score"])
    initial_score: Optional[float] = float(min(iter1_scores)) if iter1_scores else None

    base_score_mat, base_count_mat = load_wallclock_mats(k, spec=False, initial_score=initial_score)
    # last file = augmented-children spec; first file = original spec (k=8 only)
    spec_score_mat, spec_count_mat = load_wallclock_mats(k, spec=True, initial_score=initial_score,
                                                         which="last")
    if k == 8:
        spec_orig_score_mat, spec_orig_count_mat = load_wallclock_mats(
            k, spec=True, initial_score=initial_score, which="first")

    # ── Plot 1: wall-clock time vs best score ─────────────────────────────────

    all_mats = [base_score_mat, spec_score_mat]
    if k == 8:
        all_mats.append(spec_orig_score_mat)
    finite = np.concatenate([m.ravel() for m in all_mats])
    finite = finite[np.isfinite(finite)]
    anchor   = initial_score if initial_score is not None else finite.min()
    top_data = finite[finite > anchor + 0.05]
    gap_lo   = anchor + 0.02
    gap_hi   = float(top_data.min()) - 0.02 if len(top_data) else anchor + 0.3
    y_lo     = max(0.0,  anchor - 0.02)
    y_hi     = min(1.005, float(finite.max()) + 0.005)

    fwd, inv, comp = make_gap_scale(gap_lo, gap_hi)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"Circle Packing — Best Score vs Wall-Clock Time\n"
                 f"k={k} vs k={k} (spec)  |  mean ± std over 3 runs",
                 fontsize=12, fontweight="bold")

    plot_mean_std(ax, base_score_mat, base_color, base_marker, f"k={k}", zorder=3)
    if k == 8:
        plot_mean_std(ax, spec_orig_score_mat, SPEC_COLOR, SPEC_MARKER,
                      f"k=8 (spec)",          zorder=4)
        plot_mean_std(ax, spec_score_mat,      AUG_COLOR,  AUG_MARKER,
                      "k=8 (spec, aug. cand.)", zorder=5)
    else:
        plot_mean_std(ax, spec_score_mat, SPEC_COLOR, SPEC_MARKER, f"k={k} (spec)", zorder=4)

    ax.set_yscale(FuncScale(ax, (fwd, inv)))
    ax.set_ylim(y_lo, y_hi)

    # Ticks: anchor value + evenly spaced ticks in top region
    step = 0.05
    top_ticks = np.arange(np.ceil(gap_hi / step) * step, y_hi, step)
    ax.set_yticks([anchor] + top_ticks.tolist())
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Break markers on the y-axis spine to signal the compressed gap
    y_lo_t = fwd(np.array([y_lo]))[0]
    y_hi_t = fwd(np.array([y_hi]))[0]
    for _y_data in [gap_lo, gap_lo + comp]:
        _y_t  = fwd(np.array([_y_data]))[0]
        _y_ax = (_y_t - y_lo_t) / (y_hi_t - y_lo_t)
        _kw   = dict(transform=ax.transAxes, color="k", lw=1.3, clip_on=False)
        ax.plot([-0.025,  0.025], [_y_ax - 0.012, _y_ax + 0.012], **_kw)

    ax.set_xlabel("Wall-Clock Time (hours)", fontsize=11)
    ax.set_ylabel("Global Best Score", fontsize=11)
    ax.set_xlim(-0.03, 2.03)
    ax.set_xticks(x_hrs)
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    out = PLOTS_DIR / f"k{k}_spec_wallclock_score.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()

    # ── Plot 2: wall-clock time vs candidates ─────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"Circle Packing — Candidates Generated vs Wall-Clock Time\n"
                 f"k={k} vs k={k} (spec)  |  mean ± std over 3 runs",
                 fontsize=12, fontweight="bold")

    plot_mean_std(ax, base_count_mat, base_color, base_marker, f"k={k}", zorder=3)
    if k == 8:
        plot_mean_std(ax, spec_orig_count_mat, SPEC_COLOR, SPEC_MARKER,
                      "k=8 (spec)",            zorder=4)
        plot_mean_std(ax, spec_count_mat,      AUG_COLOR,  AUG_MARKER,
                      "k=8 (spec, aug. cand.)", zorder=5)
    else:
        plot_mean_std(ax, spec_count_mat, SPEC_COLOR, SPEC_MARKER, f"k={k} (spec)", zorder=4)

    ax.set_xlabel("Wall-Clock Time (hours)", fontsize=11)
    ax.set_ylabel("Candidates Generated (rows)", fontsize=11)
    ax.set_xlim(-0.03, 2.03)
    ax.set_xticks(x_hrs)
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    out = PLOTS_DIR / f"k{k}_spec_wallclock_candidates.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()

    # ── Plot 3: spec-hit vs non-hit timing ────────────────────────────────────

    # For k=8: 2-column figure (first jsonl = spec orig, last jsonl = aug. cand.)
    # For k=4: single-subplot figure (unchanged behaviour)

    def _timing_stats(hit_iter, nohit_iter, hit_llm, nohit_llm):
        hi_m  = np.mean(hit_iter)   if hit_iter   else 0.0
        nh_m  = np.mean(nohit_iter) if nohit_iter else 0.0
        hl_m  = np.mean(hit_llm)    if hit_llm    else 0.0
        nl_m  = np.mean(nohit_llm)  if nohit_llm  else 0.0
        return dict(
            llm_vals  = [hl_m,              nl_m],
            eval_vals = [max(0.0, hi_m - hl_m), max(0.0, nh_m - nl_m)],
            iter_vals = [hi_m,  nh_m],
            iter_stds = [np.std(hit_iter)   if hit_iter   else 0.0,
                         np.std(nohit_iter) if nohit_iter else 0.0],
            n_hit     = len(hit_iter),
            n_nohit   = len(nohit_iter),
        )

    def _draw_timing_panel(ax, stats, title):
        x     = np.array([0, 1])
        bar_w = 0.45
        lv, ev = stats["llm_vals"], stats["eval_vals"]
        iv, sv = stats["iter_vals"], stats["iter_stds"]
        ax.bar(x, lv,  bar_w, label="LLM Generation",
               color=LLM_COLOR,  edgecolor="white", linewidth=0.5)
        ax.bar(x, ev, bar_w, bottom=lv, label="Eval / Other",
               color=EVAL_COLOR, edgecolor="white", linewidth=0.5)
        ax.errorbar(x, iv, yerr=sv, fmt="none",
                    ecolor="black", elinewidth=1.8, capsize=6, capthick=1.5, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Spec-Hit", "Non-Hit"], fontsize=13)
        ax.set_ylabel("Mean Iteration Time (s)", fontsize=12)
        ax.set_xlabel("Speculation Outcome", fontsize=12)
        max_top = max(v + s for v, s in zip(iv, sv)) if any(iv) else 1.0
        ax.set_ylim(0, max_top * 1.25)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=11, framealpha=0.8)
        ax.set_title(title, fontsize=13, fontweight="bold")
        for xi, (llm, ev_, it, st) in enumerate(zip(lv, ev, iv, sv)):
            ax.text(xi, it + st + max_top * 0.02,
                    f"{it:.1f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")
            ax.text(xi, llm / 2,
                    f"{llm:.1f}s", ha="center", va="center", fontsize=10,
                    color="white", fontweight="bold")
            if ev_ > max_top * 0.05:
                ax.text(xi, llm + ev_ / 2,
                        f"{ev_:.1f}s", ha="center", va="center", fontsize=10,
                        color="white", fontweight="bold")

    if k == 8:
        orig_data = _timing_stats(*load_timing_data(k, which="first"))
        aug_data  = _timing_stats(*load_timing_data(k, which="last"))

        fig, axes3 = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        _draw_timing_panel(
            axes3[0], orig_data,
            f"k=8 (spec)  —  hit n={orig_data['n_hit']}, non-hit n={orig_data['n_nohit']}")
        _draw_timing_panel(
            axes3[1], aug_data,
            f"k=8 (spec, aug. cand.)  —  hit n={aug_data['n_hit']}, non-hit n={aug_data['n_nohit']}")
        axes3[1].set_ylabel("")   # shared y-axis; only left panel needs label

        plt.tight_layout()
        out = PLOTS_DIR / f"k{k}_spec_hit_vs_nohit_time.pdf"
        plt.savefig(out, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.close()
    else:
        hit_iter, nohit_iter, hit_llm, nohit_llm = load_timing_data(k)
        stats = _timing_stats(hit_iter, nohit_iter, hit_llm, nohit_llm)
        fig, ax = plt.subplots(figsize=(7, 5))
        _draw_timing_panel(
            ax, stats,
            f"k={k} (spec)  —  hit n={stats['n_hit']}, non-hit n={stats['n_nohit']}")
        plt.tight_layout()
        out = PLOTS_DIR / f"k{k}_spec_hit_vs_nohit_time.pdf"
        plt.savefig(out, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.close()


# ── Plot 4: LLM time per iteration (run 1), colored by spec hit/miss ──────────

HIT_COLOR  = "#2ca02c"   # green
MISS_COLOR = "#d62728"   # red

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Circle Packing (run 1) — LLM Time per Iteration\n"
             "green = spec-hit, red = spec-miss",
             fontsize=12, fontweight="bold")

for ax, k in zip(axes, [4, 8]):
    p = find_jsonl(RESULTS_DIR / f"circle_packing_k{k}_run1_spec")
    if p is None:
        ax.set_title(f"k={k} (spec) — MISSING")
        continue

    idx_hit, llm_hit   = [], []
    idx_miss, llm_miss = [], []
    seen_iters = set()
    for row in load_rows(p):
        it_num = row.get("iteration")
        if it_num in seen_iters:
            continue
        ir = row.get("iteration_result", {})
        llm = ir.get("llm_generation_time_seconds")
        if llm is None:
            continue
        seen_iters.add(it_num)
        if ir.get("speculation_hit", False):
            idx_hit.append(it_num)
            llm_hit.append(llm)
        else:
            idx_miss.append(it_num)
            llm_miss.append(llm)

    all_idx = idx_miss + idx_hit
    all_llm = llm_miss + llm_hit
    order   = np.argsort(all_idx)
    ax.plot(np.array(all_idx)[order], np.array(all_llm)[order],
            color="gray", linewidth=0.8, alpha=0.5, zorder=2)
    ax.scatter(idx_miss, llm_miss, color=MISS_COLOR, s=18, alpha=0.7,
               label=f"spec-miss (n={len(idx_miss)})", zorder=3)
    ax.scatter(idx_hit,  llm_hit,  color=HIT_COLOR,  s=18, alpha=0.8,
               label=f"spec-hit  (n={len(idx_hit)})",  zorder=4)

    ax.set_title(f"k={k} (spec)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Iteration Index", fontsize=10)
    ax.set_ylabel("LLM Generation Time (s)", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.35)

plt.tight_layout()
out4 = PLOTS_DIR / "k4_k8_spec_llm_time_per_iter.pdf"
plt.savefig(out4, bbox_inches="tight")
print(f"Saved → {out4}")
plt.close()
