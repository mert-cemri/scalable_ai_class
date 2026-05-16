"""
BatchedAdaEvolveController — KV-cache-friendly parallel AdaEvolve.

The vanilla AdaEvolve loop is:
    for iter in range(N):
        parent  <- database.sample()           # changes per iter
        prompt  <- context_builder(parent,...) # changes per iter
        child   <- LLM(prompt)                 # one call
        score   <- eval(child); db.add(child)
        db.end_iteration()

A separate AdaEvolve process running concurrently (``-j K`` style) will
have its OWN parent/context/prompt and so its prompts share only the
small static system prefix with this run. With vLLM prefix caching that
amounts to a low single-digit hit rate dominated by the variable parent
code in the user message.

This controller folds K *into* a single iteration:
    parent, ctx <- database.sample()
    prompt      <- context_builder(...)        # built once
    children    <- gather([LLM(prompt) for _ in K])  # K calls, identical prompt
    scores      <- gather([eval(c) for c in children])
    for c in children: db.add(c)
    db.end_iteration()

Key properties:
  * The prompt is *byte-identical* across the K calls, so the prefill KV
    is computed once and reused for the other K-1 — vLLM's prefix cache
    hit rate on the sibling calls approaches 100%.
  * Each call still gets a fresh decode (different sampling RNG, or
    different per-call temperature if configured), so the K children are
    diverse mutations.
  * Database state advances by K children per iteration. The UCB island
    rotation, migration timer, and paradigm-stagnation tracker tick once
    per *iteration*, not once per *child* — preserving the original
    AdaEvolve schedule.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.llm.base import LLMResponse
from skydiscover.search.adaevolve.controller import AdaEvolveController
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import DiscoveryControllerInput
from skydiscover.search.utils.discovery_utils import SerializableResult
from skydiscover.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)

logger = logging.getLogger(__name__)

def _poll_vllm_cache_stats(api_base: str) -> tuple[float, float]:
    """Return (prefix_cache_queries_total, prefix_cache_hits_total) from vLLM /metrics.

    Strips a trailing /v1 path so the metrics endpoint is always at <host>/metrics.
    Returns (0, 0) silently on any error (non-vLLM backend, server not ready, etc.).
    """
    import requests as _requests

    try:
        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        text = _requests.get(f"{base}/metrics", timeout=2).text
        queries = hits = 0.0
        for line in text.splitlines():
            if line.startswith("vllm:prefix_cache_queries_total{"):
                queries = float(line.split()[-1])
            elif line.startswith("vllm:prefix_cache_hits_total{"):
                hits = float(line.split()[-1])
        return queries, hits
    except Exception:
        return 0.0, 0.0


class BatchedAdaEvolveController(AdaEvolveController):
    """AdaEvolve with K candidates per iteration sharing one prompt."""

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)
        db_cfg = self.config.search.database
        self.candidates_per_iteration: int = max(
            1, int(getattr(db_cfg, "candidates_per_iteration", 1))
        )
        self.diversify_temperature: bool = bool(
            getattr(db_cfg, "diversify_temperature", False)
        )
        self.temperature_spread: float = float(
            getattr(db_cfg, "temperature_spread", 0.4)
        )
        # Adaptive K — drive K from island intensity.
        self.adaptive_candidates: bool = bool(
            getattr(db_cfg, "adaptive_candidates", False)
        )
        self.candidates_min: int = max(1, int(getattr(db_cfg, "candidates_min", 2)))
        self.candidates_max: int = max(
            self.candidates_min,
            int(getattr(db_cfg, "candidates_max", 16)),
        )
        # Speculative prefix prefetch.
        self.enable_prefetch: bool = bool(getattr(db_cfg, "enable_prefetch", False))
        self._prefetch_task: Optional[asyncio.Task] = None
        self._prefetch_count: int = 0
        # Speculative iteration pipelining (Tier 3.2).
        self.enable_speculative_pipelining: bool = bool(
            getattr(db_cfg, "enable_speculative_pipelining", False)
        )
        # Pending speculative iter: (predicted_prompt, response_task,
        # predicted_parent_id, K_used, temperatures). Consumed at the start
        # of the next iteration if the prediction holds.
        self._speculative: Optional[Dict[str, Any]] = None
        self._speculative_hits: int = 0
        self._speculative_misses: int = 0
        # Adaptive speculation gating: at each iteration, query the vLLM
        # /metrics endpoint and disable speculation for the next iter if
        # the GPU is under pressure. Closes the saturation regression we
        # observed in the scaling sweep at N≥8.
        self.speculation_gpu_pressure_gating: bool = bool(
            getattr(db_cfg, "speculation_gpu_pressure_gating", False)
        )
        self.speculation_pressure_threshold: float = float(
            getattr(db_cfg, "speculation_pressure_threshold", 0.6)
        )
        self.speculation_metrics_url: str = str(
            getattr(db_cfg, "speculation_metrics_url",
                    "http://127.0.0.1:8765/metrics")
        )
        self.speculation_max_num_seqs: int = int(
            getattr(db_cfg, "speculation_max_num_seqs", 32)
        )
        self._speculation_skipped: int = 0
        # Cache-pressure-adaptive K (Tier 4.4 redux).
        self.cache_pressure_adaptive_k: bool = bool(
            getattr(db_cfg, "cache_pressure_adaptive_k", False)
        )
        self.cache_pressure_threshold: float = float(
            getattr(db_cfg, "cache_pressure_threshold", 0.7)
        )
        self.pressure_k_scale: float = float(
            getattr(db_cfg, "pressure_k_scale", 0.5)
        )
        self._k_pressure_dropped: int = 0

        # KV cache hit-rate tracking: snapshot cumulative vLLM counters at
        # init so each iteration reports the *incremental* hit rate.
        _api_base = getattr(self.config.llm.models[0], "api_base", None) or ""
        self._vllm_api_base: str = _api_base
        q0, h0 = _poll_vllm_cache_stats(_api_base)
        self._cache_queries_baseline: float = q0
        self._cache_hits_baseline: float = h0
        self._last_gen_q_delta: float = 0.0
        self._last_gen_h_delta: float = 0.0
        self._last_spec_used: Optional[bool] = None
        # Programs generated in the most-recently-completed iteration (all K
        # candidates, regardless of whether they were accepted into the DB).
        # Injected into the next speculation prompt as sibling context so the
        # LLM can see what the previous batch already tried.
        self._last_batch_children: List[Program] = []

        if self.candidates_per_iteration > 1 or self.adaptive_candidates:
            logger.info(
                "BatchedAdaEvolve: K=%d (adaptive=%s, [%d..%d]) "
                "diversify_temperature=%s prefetch=%s",
                self.candidates_per_iteration,
                self.adaptive_candidates,
                self.candidates_min,
                self.candidates_max,
                self.diversify_temperature,
                self.enable_prefetch,
            )
            # Make sure the evaluator pool can absorb K parallel evaluations.
            # TaskPool's semaphore is lazy-init'd, so raising max_concurrency
            # here before any evaluate_program() call is sufficient.
            try:
                pool = self.evaluator.task_pool
                pool.max_concurrency = max(
                    pool.max_concurrency, self.candidates_max
                )
            except AttributeError:
                pass

    # ------------------------------------------------------------------
    # Adaptive K
    # ------------------------------------------------------------------

    def _effective_k(self) -> int:
        """Return K for the current iteration; respects adaptive policy.

        Two stages:
          1. Map adaptive intensity → [candidates_min, candidates_max]
             (only when adaptive_candidates is on). Intensity is the
             island's exploration ratio.
          2. If cache_pressure_adaptive_k is on, query vLLM's KV cache
             utilization; if it exceeds cache_pressure_threshold, scale
             K by pressure_k_scale. Acts as a be-a-good-neighbor knob
             under multi-tenant load.
        """
        # Stage 1: intensity-driven K
        if self.adaptive_candidates:
            try:
                island = self.database.current_island
                if self.database.use_adaptive_search:
                    intensity = self.database.adapter.get_search_intensity(island)
                else:
                    intensity = self.database.fixed_intensity
                lo = self.database.intensity_min
                hi = self.database.intensity_max
                t = 0.0 if hi <= lo else max(0.0, min(1.0, (intensity - lo) / (hi - lo)))
                k = self.candidates_min + (self.candidates_max - self.candidates_min) * t
                k = max(self.candidates_min, min(self.candidates_max, int(round(k))))
            except Exception:
                k = self.candidates_per_iteration
        else:
            k = self.candidates_per_iteration

        # Stage 2: cache-pressure scaling
        if self.cache_pressure_adaptive_k:
            usage = self._read_kv_cache_usage()
            if usage is not None and usage > self.cache_pressure_threshold:
                scaled = max(1, int(round(k * self.pressure_k_scale)))
                if scaled < k:
                    self._k_pressure_dropped += 1
                    logger.debug(
                        "Iter K pressure-drop: %d -> %d (kv_usage=%.3f > %.3f)",
                        k, scaled, usage, self.cache_pressure_threshold,
                    )
                    k = scaled

        return k
    
    def _log_iteration_stats(
        self,
        iteration: int,
        sampling_mode: Optional[str] = None,
        sampling_intensity: Optional[float] = None,
        child_program: Optional[Dict] = None,
        iteration_time: Optional[float] = None,
        llm_generation_time: Optional[float] = None,
        eval_time: Optional[float] = None,
        error: Optional[str] = None,
        batch_stats: Optional[Dict] = None,
        cache_queries_delta: Optional[float] = None,
        cache_hits_delta: Optional[float] = None,
        cache_hit_rate: Optional[float] = None,
    ) -> None:
        """
        Log comprehensive iteration statistics to JSON file.

        This method collects all AdaEvolve signals and writes them as a single
        JSON line to the log file for easy post-processing.

        Args:
            iteration: Current iteration number
            sampling_mode: The mode used for sampling (exploration/exploitation/balanced)
            sampling_intensity: The search intensity value used
            child_program: The child program dict if successfully generated
            iteration_time: Time taken for this iteration
            error: Error message if iteration failed
        """
        if self._iteration_stats_log_path is None:
            return

        try:
            # Get comprehensive stats from database
            stats = self.database.get_comprehensive_iteration_stats(
                iteration=iteration,
                sampling_mode=(
                    sampling_mode if sampling_mode is not None else self._last_sampling_mode
                ),
                sampling_intensity=(
                    sampling_intensity
                    if sampling_intensity is not None
                    else self._last_sampling_intensity
                ),
            )

            # Add timestamp
            stats["timestamp"] = datetime.now().isoformat()

            q_delta = cache_queries_delta if cache_queries_delta is not None else 0.0
            h_delta = cache_hits_delta if cache_hits_delta is not None else 0.0
            hit_rate = cache_hit_rate if cache_hit_rate is not None else 0.0

            # Add iteration-specific info
            stats["iteration_result"] = {
                "success": error is None,
                "error": error,
                "iteration_time_seconds": iteration_time,
                "llm_generation_time_seconds": llm_generation_time,
                "eval_time_seconds": eval_time,
                "batch": batch_stats,  # None in single-candidate mode
                "speculation_hit": self._last_spec_used,  # True/False/None(disabled)
                "kv_cache": {
                    "queries_delta": q_delta,
                    "hits_delta": h_delta,
                    "hit_rate": hit_rate,
                },
            }

            # Add child program info if available
            if child_program:
                stats["iteration_result"]["child_program"] = {
                    "id": child_program.get("id"),
                    "metrics": child_program.get("metrics"),
                    "generation": child_program.get("generation"),
                    "parent_id": child_program.get("parent_id"),
                }
            # Write to JSONL file
            with open(self._iteration_stats_log_path, "a") as f:
                f.write(json.dumps(stats, default=str) + "\n")

        except Exception as e:
            logger.warning(f"Failed to log iteration stats: {e}")

    def _read_kv_cache_usage(self) -> Optional[float]:
        """Return vllm:kv_cache_usage_perc (0..1) or None if unavailable.

        Uses the same shared cache as `_should_speculate` so the /metrics
        scrape happens at most once per TTL across all controllers.
        """
        cls = type(self)
        now = time.time()
        if hasattr(cls, "_kv_usage_value") and now - getattr(cls, "_kv_usage_ts", 0) < cls._PRESSURE_TTL:
            return cls._kv_usage_value
        try:
            import urllib.request
            with urllib.request.urlopen(self.speculation_metrics_url, timeout=0.5) as r:
                text = r.read().decode("utf-8", errors="replace")
        except Exception:
            return None
        usage = None
        for line in text.splitlines():
            if line.startswith("vllm:kv_cache_usage_perc"):
                try:
                    usage = float(line.rsplit(None, 1)[-1])
                    break
                except (ValueError, IndexError):
                    pass
        cls._kv_usage_value = usage
        cls._kv_usage_ts = now
        return usage

    # ------------------------------------------------------------------
    # Iteration: one sample, one prompt, K parallel candidates
    # ------------------------------------------------------------------

    async def _run_iteration(self, iteration: int, checkpoint_callback) -> None:
        iter_start = time.time()
        if (
            self.database.use_paradigm_breakthrough
            and self.database.is_paradigm_stagnating()
        ):
            await self._generate_paradigms_if_needed()

        # Cancel any in-flight prefetch for this iteration's predicted prompt.
        # Whether we got a prefetch hit will be visible in vLLM's hit rate￼
        # metrics; we don't need to read it back here.
        await self._await_prefetch()

        results = await self._generate_batch(iteration)
        iteration_time = time.time() - iter_start

        # Use the before/after snapshot taken inside _generate_batch, which
        # brackets only the LLM calls — not eval or overhead.
        q_delta = self._last_gen_q_delta
        h_delta = self._last_gen_h_delta
        hit_rate = (h_delta / q_delta) if q_delta > 0 else None
        if hit_rate is not None:
            logger.info(
                "Iteration %d cache hit rate: %.1f%% (%g hits / %g queries)",
                iteration, hit_rate * 100, h_delta, q_delta,
            )

        # Speculative prefetch for the *next* iteration's predicted prompt
        # while the current iteration's eval is winding down. We launch as a
        # background task; the next call to _run_iteration awaits it.
        if self.enable_prefetch:
            self._launch_prefetch(iteration + 1)

        ok = [r for r in results if not r.error]
        bad = [r for r in results if r.error]

        for r in ok:
            self._process_result(r, iteration, checkpoint_callback)
            self._log_iteration_stats(
                iteration=iteration,
                sampling_mode=self._last_sampling_mode,
                sampling_intensity=self._last_sampling_intensity,
                child_program=r.child_program_dict,
                iteration_time=iteration_time,
                llm_generation_time=r.llm_generation_time,
                eval_time=r.eval_time,
                error=None,
                cache_queries_delta=q_delta,
                cache_hits_delta=h_delta,
                cache_hit_rate=hit_rate,
            )
        for r in bad:
            logger.warning("Iteration %d (batched): %s", iteration, r.error)
            self._log_iteration_stats(
                iteration=iteration,
                sampling_mode=self._last_sampling_mode,
                sampling_intensity=self._last_sampling_intensity,
                child_program=None,
                iteration_time=iteration_time,
                llm_generation_time=r.llm_generation_time,
                eval_time=r.eval_time,
                error=r.error,
                cache_queries_delta=q_delta,
                cache_hits_delta=h_delta,
                cache_hit_rate=hit_rate,
            )

        logger.info(
            "Iteration %d batched: %d/%d candidates accepted in %.2fs",
            iteration,
            len(ok),
            len(results),
            iteration_time,
        )

    # ------------------------------------------------------------------
    # Single-prompt, K-call generation
    # ------------------------------------------------------------------

    async def _generate_batch(self, iteration: int) -> List[SerializableResult]:
        if not self.database.programs:
            # Bootstrap iteration: fall back to single-call from-scratch path.
            r = await self._run_from_scratch_iteration(iteration)
            return [r]

        try:
            self._ensure_all_islands_seeded()
            parent_dict, context_programs_dict = self.database.sample(
                self.num_context_programs,
                force_exploration=False,
            )
        except Exception as e:
            return [SerializableResult(error=f"sample() failed: {e}", iteration=iteration)]

        if not parent_dict:
            return [SerializableResult(error="empty parent dict", iteration=iteration)]

        parent_label = list(parent_dict.keys())[0]
        parent: Program = list(parent_dict.values())[0]

        sampling_mode = getattr(self.database, "_last_sampling_mode", None) or "balanced"
        self._last_sampling_mode = sampling_mode
        cur_island = self.database.current_island
        if self.database.use_adaptive_search:
            self._last_sampling_intensity = self.database.adapter.get_search_intensity(cur_island)
        else:
            self._last_sampling_intensity = self.database.fixed_intensity

        paradigm = (
            self.database.get_current_paradigm()
            if self.database.use_paradigm_breakthrough
            else None
        )
        if paradigm:
            best = self.database.get_best_program()
            if best:
                parent_dict = {parent_label: best}
                parent = best

        siblings = []
        if hasattr(self.database, "get_children"):
            try:
                siblings = self.database.get_children(parent.id)
            except (AttributeError, NotImplementedError):
                pass

        context = {
            "program_metrics": parent.metrics,
            "other_context_programs": context_programs_dict,
            "paradigm": paradigm,
            "siblings": siblings,
            "error_context": None,
        }
        for k, v in self._prompt_context.items():
            context.setdefault(k, v)

        prompt = self.context_builder.build_prompt(parent_dict, context)
        if paradigm:
            self.database.use_paradigm()

        if self.feedback_reader:
            self.feedback_reader.set_current_prompt(prompt["system"])
            feedback = self.feedback_reader.read()
            if feedback:
                prompt = self.feedback_reader.apply_feedback(prompt)
                self.feedback_reader.log_usage(iteration, feedback, self.feedback_reader.mode)

        parent_info = (parent_label, parent.id)
        context_program_ids = [
            p.id for programs in context_programs_dict.values() for p in programs
        ]
        context_info = [
            (label, p.id)
            for label, programs in context_programs_dict.items()
            for p in programs
        ]

        K = self._effective_k()
        temps = self._candidate_temperatures(K)
        if K != self.candidates_per_iteration:
            logger.debug(
                "Iter %d adaptive K: %d -> %d (intensity=%.3f, island=%d)",
                iteration,
                self.candidates_per_iteration,
                K,
                self._last_sampling_intensity or 0.0,
                self.database.current_island,
            )
        # Stash this iteration's prompt/context for the next speculation prediction.
        # Reusing context_programs_dict and siblings avoids re-calling sample()
        # (no island-state side effects) and is a good approximation since the
        # database changes by at most K entries between consecutive iterations.
        self._last_prompt = prompt
        self._last_parent_id = parent.id
        self._last_context_programs = context_programs_dict
        self._last_siblings = siblings

        # Speculative pipelining: if we have a staged speculation whose
        # predicted prompt matches today's actual prompt, use its
        # already-running LLM calls instead of issuing fresh ones.
        spec_used = False
        self._last_spec_used = None  # None = pipelining disabled
        if self._speculative is not None and self.enable_speculative_pipelining:
            spec = self._speculative
            self._speculative = None
            _parent_match = spec.get("parent_id") == parent.id
            _spec_island = spec.get("context_island_idx")
            _cur_island = getattr(self.database, "current_island", None)
            _island_match = (
                _spec_island is None
                or _cur_island is None
                or _spec_island == _cur_island
            )
            if _parent_match and _island_match:
                logger.debug("Speculative pipeline HIT at iter %d", iteration)
                self._speculative_hits += 1
                spec_used = True
                self._last_spec_used = True
                # Reuse the in-flight tasks. They were issued with the
                # speculative temperature schedule (which may differ from
                # the current K). If K has changed since the speculation
                # fired, top up or trim.
                tasks = spec["tasks"]
                # Pad with fresh calls if K grew since the spec fired.
                while len(tasks) < K:
                    tasks.append(asyncio.create_task(
                        self._call_llm(prompt["system"], prompt["user"],
                                       temperature=temps[len(tasks)])
                    ))
                tasks = tasks[:K]
            else:
                reason = (
                    "parent mismatch" if not _parent_match
                    else f"island mismatch (spec={_spec_island}, current={_cur_island})"
                )
                logger.debug("Speculative pipeline MISS at iter %d (%s)", iteration, reason)
                self._speculative_misses += 1
                self._last_spec_used = False
                for t in spec["tasks"]:
                    if not t.done():
                        t.cancel()
        elif self.enable_speculative_pipelining:
            # Pipelining is on but no spec was staged (previous iter consumed it).
            # Record False so logs distinguish "skipped" from "disabled" (None).
            self._last_spec_used = False

        # ---- K parallel LLM calls with the IDENTICAL prompt ----
        llm_start = time.time()
        q_before, h_before = _poll_vllm_cache_stats(self._vllm_api_base)
        if not spec_used:
            llm_tasks = [
                self._call_llm(prompt["system"], prompt["user"], temperature=t)
                for t in temps
            ]
        else:
            llm_tasks = tasks
        llm_responses: List[Any] = await asyncio.gather(*llm_tasks, return_exceptions=True)
        q_after, h_after = _poll_vllm_cache_stats(self._vllm_api_base)
        self._last_gen_q_delta = q_after - q_before
        self._last_gen_h_delta = h_after - h_before
        total_llm_time = time.time() - llm_start

        # ---- Parse all responses, gather valid ones for eval ----
        # Parsing is pure string processing (no I/O), so we do it before
        # launching speculation so that _last_batch_children reflects THIS
        # iteration's generated programs rather than the previous one.
        parsed: List[Tuple[Optional[str], Optional[str], Optional[str]]] = []
        for resp in llm_responses:
            if isinstance(resp, Exception):
                parsed.append((None, None, f"LLM error: {resp}"))
                continue
            text = getattr(resp, "text", None) or ""
            if not text:
                parsed.append((None, None, "Empty LLM response"))
                continue
            if self.config.diff_based_generation:
                diffs = extract_diffs(text)
                if diffs:
                    sol = apply_diff(parent.solution, text)
                    parsed.append((sol, format_diff_summary(diffs), None))
                else:
                    sol = parse_full_rewrite(text, self.config.language)
                    if sol:
                        parsed.append((sol, "Full rewrite", None))
                    else:
                        parsed.append((None, None, "No diffs/rewrite parsed"))
            else:
                sol = parse_full_rewrite(text, self.config.language)
                if sol:
                    parsed.append((sol, "Full rewrite", None))
                else:
                    parsed.append((None, None, "No solution parsed"))

        # Snapshot parsed solutions as lightweight Program objects (no eval
        # metrics yet) so the upcoming speculation prompt sees THIS iteration's
        # generated programs as sibling context, eliminating the one-iter lag.
        self._last_batch_children = [
            Program(
                id=str(uuid.uuid4()),
                solution=sol,
                language=self.config.language,
                metrics={},
                iteration_found=iteration,
                parent_id=parent.id,
                generation=parent.generation + 1,
            )
            for sol, _, perr in parsed
            if sol and not perr
        ]

        # Speculative launch for the NEXT iteration. Parsing is already done
        # above, so _last_batch_children is current before the prompt is built.
        if self.enable_speculative_pipelining and not spec_used:
            if self._should_speculate():
                pred_result = self._predict_next_prompt(prompt, parent)
                if pred_result is not None:
                    pred_prompt, pred_parent_id, pred_island_idx = pred_result
                    self._launch_speculative(pred_prompt, pred_parent_id, temps[:K], pred_island_idx)
            else:
                self._speculation_skipped += 1
                logger.debug(
                    "Iter %d: skipped speculation (GPU pressure > %.2f)",
                    iteration, self.speculation_pressure_threshold,
                )

        # ---- Concurrently evaluate the K parsed solutions ----
        eval_start = time.time()
        child_ids = [str(uuid.uuid4()) for _ in parsed]

        async def _eval_one(idx: int):
            sol, _, perr = parsed[idx]
            if perr or not sol:
                return None
            try:
                return await self.evaluator.evaluate_program(sol, child_ids[idx])
            except Exception as e:
                logger.debug("eval failed: %s", e)
                return None

        eval_results = await asyncio.gather(*[_eval_one(i) for i in range(len(parsed))])
        total_eval_time = time.time() - eval_start

        # ---- Materialize SerializableResult per candidate ----
        out: List[SerializableResult] = []
        for i, ((sol, changes, perr), er, resp) in enumerate(
            zip(parsed, eval_results, llm_responses)
        ):
            text = getattr(resp, "text", None) if not isinstance(resp, Exception) else None
            if perr:
                out.append(
                    SerializableResult(
                        error=perr,
                        iteration=iteration,
                        prompt=prompt,
                        llm_response=text,
                        llm_generation_time=total_llm_time,
                        eval_time=0.0,
                    )
                )
                continue
            if er is None:
                out.append(
                    SerializableResult(
                        error="Evaluation failed",
                        iteration=iteration,
                        prompt=prompt,
                        llm_response=text,
                        llm_generation_time=total_llm_time,
                        eval_time=total_eval_time,
                    )
                )
                continue
            child = Program(
                id=child_ids[i],
                solution=sol,
                language=self.config.language,
                metrics=er.metrics,
                iteration_found=iteration,
                parent_id=parent.id,
                other_context_ids=context_program_ids,
                parent_info=parent_info,
                context_info=context_info,
                generation=parent.generation + 1,
                metadata={
                    "changes": changes,
                    "parent_metrics": parent.metrics,
                    "batch_index": i,
                    "batch_size": K,
                    "temperature": temps[i],
                },
                artifacts=er.artifacts or {},
            )
            out.append(
                SerializableResult(
                    child_program_dict=child.to_dict(),
                    parent_id=parent.id,
                    other_context_ids=context_program_ids,
                    iteration_time=total_llm_time + total_eval_time,
                    llm_generation_time=total_llm_time,
                    eval_time=total_eval_time,
                    prompt=prompt,
                    llm_response=text,
                    iteration=iteration,
                )
            )
        return out

    def _candidate_temperatures(self, k: int) -> List[float]:
        """Return per-candidate temperatures, optionally spread for diversity."""
        base = None
        try:
            base = self.config.llm.models[0].temperature
        except Exception:
            pass
        if base is None:
            base = 0.7
        if not self.diversify_temperature or k == 1:
            return [base] * k
        # Linear spread around base, clipped to [0.05, 1.5].
        half = self.temperature_spread / 2.0
        if k == 1:
            return [base]
        step = self.temperature_spread / (k - 1)
        out = []
        for i in range(k):
            t = base - half + i * step
            t = max(0.05, min(1.5, t))
            out.append(t)
        return out

    # ------------------------------------------------------------------
    # Speculative prefix prefetch
    # ------------------------------------------------------------------

    def _launch_prefetch(self, predicted_iteration: int) -> None:
        """Fire-and-forget: prefill the predicted next prompt at vLLM.

        Uses ``max_tokens=1`` so vLLM does the prefill, caches the blocks,
        and returns instantly. The next iteration's real call hits a warm
        cache. We predict that next iter will reuse the most-recent
        parent (which is true 70-90% of the time per AdaEvolve's
        UCB-driven selection).
        """
        if not getattr(self, "_last_prompt", None):
            return
        prompt = self._last_prompt
        async def _do_prefetch():
            try:
                self._prefetch_count += 1
                # max_tokens=1 → vLLM does prefill, returns immediately.
                await self._call_llm(
                    prompt["system"], prompt["user"], max_tokens=1, temperature=0.0
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("prefetch %d failed: %s", predicted_iteration, e)
        # Cancel any previous prefetch if still in flight.
        if self._prefetch_task and not self._prefetch_task.done():
            self._prefetch_task.cancel()
        self._prefetch_task = asyncio.create_task(_do_prefetch())

    async def _await_prefetch(self) -> None:
        """Wait for any pending prefetch to finish (or cancel)."""
        if self._prefetch_task is None:
            return
        if self._prefetch_task.done():
            self._prefetch_task = None
            return
        # Don't actually wait — let it run in the background. The vLLM
        # cache is shared across requests so its blocks land regardless of
        # whether this future is awaited. Just clear the slot.
        # If you want strict ordering, replace with `await self._prefetch_task`.
        self._prefetch_task = None

    # ------------------------------------------------------------------
    # Speculative iteration pipelining (Tier 3.2)
    # ------------------------------------------------------------------

    def _find_island_for_program(self, program_id: str) -> Optional[int]:
        """Return the archive index containing program_id, or None."""
        db = self.database
        if hasattr(db, "archives") and db.archives:
            for i, archive in enumerate(db.archives):
                try:
                    if archive.contains(program_id):
                        return i
                except Exception:
                    pass
        return None

    def _predict_next_prompt(
        self,
        current_prompt: Dict[str, str],
        current_parent: Program,
    ) -> Optional[Tuple[Dict[str, str], str, Optional[int]]]:
        """Predict iter t+1's prompt, parent ID, and context island for speculative pipelining.

        Two-stage predictor:
          1. *Best-so-far*. If the database's currently-best program is
             different from `current_parent`, build a prompt against it.
          2. *Same-as-current*. Fallback: predict the next iter will use
             the same parent.

        In both cases, other_context_programs are freshly resampled from the
        appropriate island (local archive diversity + global top) rather than
        reused from the previous iteration. Returns None on any failure to
        skip speculation for that iteration. The third element of the return
        tuple is the island index used for context sampling, stored in the
        speculative dict so the hit-check can reject island mismatches.
        """
        try:
            best = self.database.get_best_program()
        except Exception:
            return None

        db = self.database
        num = self.num_context_programs or 4
        local_ratio = getattr(db, "local_context_program_ratio", 0.6)
        local_count = max(1, int(num * local_ratio))
        global_count = num - local_count

        # Mirror _generate_batch: if a paradigm is active it overrides the
        # predicted parent to `best` and injects the paradigm into context.
        paradigm = None
        if getattr(db, "use_paradigm_breakthrough", False):
            try:
                paradigm = db.get_current_paradigm()
            except Exception:
                pass

        def _resample_context(parent: Program, island_idx: Optional[int]) -> Optional[Dict]:
            if (
                island_idx is not None
                and hasattr(db, "archives")
                and db.archives
                and island_idx < len(db.archives)
            ):
                local_ctx = db.archives[island_idx].sample_other_context_programs(
                    parent, local_count
                )
                global_ctx = db._sample_global_top(parent.id, global_count)
                return {"": local_ctx + global_ctx}
            return {"": db._sample_global_top(parent.id, num)}

        def _build_ctx(parent: Program, ctx_programs: Dict, siblings: List) -> Dict:
            # Include all K programs from the previous iteration so the LLM
            # sees what the last batch already tried, regardless of whether
            # those candidates were accepted into the database.
            combined_siblings = list(siblings) + self._last_batch_children
            ctx = {
                "program_metrics": parent.metrics,
                "other_context_programs": ctx_programs,
                "paradigm": paradigm,
                "siblings": combined_siblings,
                "error_context": None,
            }
            for k, v in self._prompt_context.items():
                ctx.setdefault(k, v)
            return ctx

        # If paradigm active, real _generate_batch forces parent = best regardless.
        # Mirror that: override predicted parent to best and use best's island.
        if paradigm and best is not None:
            try:
                best_island_idx = self._find_island_for_program(best.id)
                ctx_programs = _resample_context(best, best_island_idx)
                if ctx_programs is None:
                    return None
                siblings = []
                if hasattr(db, "get_children"):
                    siblings = db.get_children(best.id)
                ctx = _build_ctx(best, ctx_programs, siblings)
                pred = self.context_builder.build_prompt({"": best}, ctx)
                return pred, best.id, best_island_idx
            except Exception:
                return None

        if best is not None and best.id != current_parent.id:
            try:
                siblings_for_best = []
                if hasattr(db, "get_children"):
                    siblings_for_best = db.get_children(best.id)
                best_island_idx = self._find_island_for_program(best.id)
                ctx_programs = _resample_context(best, best_island_idx)
                if ctx_programs is None:
                    return None
                ctx = _build_ctx(best, ctx_programs, siblings_for_best)
                pred = self.context_builder.build_prompt({"": best}, ctx)
                return pred, best.id, best_island_idx
            except Exception:
                return None

        # Same-parent fallback: resample context from current island for diversity.
        try:
            cur_island_idx = getattr(db, "current_island", None)
            ctx_programs = _resample_context(current_parent, cur_island_idx)
            if ctx_programs is None:
                return None
            siblings = []
            if hasattr(db, "get_children"):
                siblings = db.get_children(current_parent.id)
            ctx = _build_ctx(current_parent, ctx_programs, siblings)
            pred = self.context_builder.build_prompt({"": current_parent}, ctx)
            return pred, current_parent.id, cur_island_idx
        except Exception:
            return None

    # Cached pressure reading shared across all controllers in this
    # process. Refreshed asynchronously at most once per
    # `_PRESSURE_TTL` seconds. This avoids the synchronous /metrics
    # poll on the iteration hot path, which we saw add 60+ seconds of
    # wall time at N=8 in the scaling sweep.
    _pressure_cache_value: float = 0.0
    _pressure_cache_ts: float = 0.0
    _PRESSURE_TTL: float = 1.5

    def _should_speculate(self) -> bool:
        """Return True if it's safe (cheap) to fire speculative calls now.

        Reads vLLM's /metrics Prometheus endpoint to compute GPU
        utilization. Speculation is gated off when running requests
        approach the engine's max_num_seqs ceiling, because at that
        point speculative calls compete directly with real ones for
        scheduler slots.

        The /metrics fetch is process-wide cached for a short TTL so
        many controllers (= many concurrent problems) all share one
        recent reading instead of each blocking the event loop on
        their own HTTP roundtrip.
        """
        if not self.speculation_gpu_pressure_gating:
            return True
        now = time.time()
        cls = type(self)
        if now - cls._pressure_cache_ts < cls._PRESSURE_TTL:
            pressure = cls._pressure_cache_value
        else:
            try:
                import urllib.request
                with urllib.request.urlopen(
                    self.speculation_metrics_url, timeout=0.5
                ) as r:
                    text = r.read().decode("utf-8", errors="replace")
            except Exception:
                cls._pressure_cache_value = 0.0
                cls._pressure_cache_ts = now
                return True
            running = 0.0
            for line in text.splitlines():
                if line.startswith("vllm:num_requests_running"):
                    try:
                        running = max(running, float(line.rsplit(None, 1)[-1]))
                    except (ValueError, IndexError):
                        pass
            pressure = running / max(1, self.speculation_max_num_seqs)
            cls._pressure_cache_value = pressure
            cls._pressure_cache_ts = now
        return pressure < self.speculation_pressure_threshold

    def _launch_speculative(
        self,
        predicted_prompt: Dict[str, str],
        predicted_parent_id: str,
        temps: List[float],
        context_island_idx: Optional[int] = None,
    ) -> None:
        """Issue K LLM calls in the background against the predicted prompt.

        The predictor used here is the simplest possible one: "next iter's
        prompt will be the same as this iter's." That's accurate when the
        AdaEvolve sampler is in exploitation mode (low intensity G) and
        the parent doesn't change between iterations — typical for
        late-run convergence phases. When the prediction is wrong (parent
        changed because of an improving child or island rotation), the
        speculative tasks are cancelled at the start of the next
        iteration; the prefill they performed warms the cache anyway.
        """
        # Cancel any prior unsettled spec — only the most recent counts.
        if self._speculative is not None:
            for t in self._speculative.get("tasks", []):
                if not t.done():
                    t.cancel()
            self._speculative = None

        # Apply human feedback to the speculative prompt so it matches the
        # real iteration's prompt, which goes through the same feedback path.
        if self.feedback_reader:
            feedback = self.feedback_reader.read()
            if feedback:
                predicted_prompt = self.feedback_reader.apply_feedback(
                    dict(predicted_prompt)  # copy to avoid mutating the original
                )

        async def _spec_call(temp: float):
            try:
                return await self._call_llm(
                    predicted_prompt["system"],
                    predicted_prompt["user"],
                    temperature=temp,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("speculative call failed: %s", e)
                return None

        tasks = [asyncio.create_task(_spec_call(t)) for t in temps]
        self._speculative = {
            "prompt": predicted_prompt,
            "parent_id": predicted_parent_id,
            "context_island_idx": context_island_idx,
            "tasks": tasks,
        }
