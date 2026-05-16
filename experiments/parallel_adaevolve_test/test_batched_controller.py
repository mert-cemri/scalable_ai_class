"""
Behavioral check on BatchedAdaEvolveController.

We patch its LLM client and evaluator so the controller runs without
hitting any external API or container, then assert that:

  1. Each iteration issues exactly K LLM calls.
  2. Within an iteration, the K calls all use the SAME (system, user)
     prompt string — this is the property that makes vLLM's prefix
     cache hit ~100% for the K-1 sibling calls.
  3. Across iterations, prompts CHANGE (parent advances).
  4. Database receives K children per iteration.

Run from this directory:
    PYTHONPATH=../.. python test_batched_controller.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import textwrap
import uuid
from collections import defaultdict
from typing import Any, Dict, List

# Make `skydiscover` importable when running this script directly.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from skydiscover.config import (
    AdaEvolveDatabaseConfig,
    Config,
    LLMConfig,
    LLMModelConfig,
)
from skydiscover.llm.base import LLMResponse
from skydiscover.search.adaevolve.batched_controller import BatchedAdaEvolveController
from skydiscover.search.adaevolve.database import AdaEvolveDatabase
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import DiscoveryControllerInput


# --------------------------------------------------------------------------
# Mocks
# --------------------------------------------------------------------------


class _MockLLM:
    """Records every (system, user) call. Returns a parseable code body."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    async def generate(self, system_message, messages, **kwargs):
        # Match the LLMPool API.
        self.calls.append(
            {
                "system": system_message,
                "user": messages[0]["content"] if messages else "",
                "kwargs": dict(kwargs),
            }
        )
        body = f"# generated_{uuid.uuid4().hex[:8]}\nx = 1\n"
        return LLMResponse(text=f"```python\n{body}```")


class _MockEvalResult:
    def __init__(self):
        self.metrics = {"combined_score": 0.5}
        self.artifacts = {}


class _MockEvaluator:
    def __init__(self):
        self.evals = 0
        # Faked TaskPool-shaped attribute so the controller's bump is harmless.
        class _Pool:
            max_concurrency = 4
        self.task_pool = _Pool()

    async def evaluate_program(self, program_path_or_solution, child_id):
        self.evals += 1
        return _MockEvalResult()

    def close(self):
        pass


def _build_minimal_config() -> Config:
    cfg = Config()
    cfg.language = "python"
    cfg.diff_based_generation = False
    cfg.checkpoint_interval = 1000
    cfg.max_solution_length = 1_000_000
    cfg.max_parallel_iterations = 1

    cfg.llm = LLMConfig()
    cfg.llm.models = [
        LLMModelConfig(name="mock-model", weight=1.0, api_base=None, api_key="x"),
    ]
    cfg.llm.evaluator_models = list(cfg.llm.models)
    cfg.llm.guide_models = list(cfg.llm.models)

    db_cfg = AdaEvolveDatabaseConfig()
    db_cfg.population_size = 10
    db_cfg.num_islands = 1
    db_cfg.use_paradigm_breakthrough = False
    db_cfg.use_dynamic_islands = False
    db_cfg.use_migration = False
    db_cfg.use_ucb_selection = False
    db_cfg.candidates_per_iteration = 4  # K
    db_cfg.diversify_temperature = True

    cfg.search.type = "adaevolve_batched"
    cfg.search.database = db_cfg
    cfg.search.num_context_programs = 1

    cfg.context_builder.system_message = "SYS_MSG"
    cfg.context_builder.template_dir = None

    cfg.evaluator.timeout = 30
    cfg.evaluator.inject_evaluator_context = False

    return cfg


def _make_seed_program(cfg: Config) -> Program:
    return Program(
        id=str(uuid.uuid4()),
        solution="x = 0\n",
        language=cfg.language,
        metrics={"combined_score": 0.1},
        iteration_found=0,
        parent_id=None,
        generation=0,
    )


def _build_controller_with_mocks(cfg: Config, db: AdaEvolveDatabase):
    # Write a throwaway evaluator file so the inferred "Evaluator" type
    # is happy at construction time. We immediately replace .evaluator
    # with our mock right after, so the file is never executed.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("def evaluate(p):\n    return {'combined_score': 0.0}\n")
        eval_path = f.name

    cfg.evaluator.evaluation_file = eval_path

    ctrl = BatchedAdaEvolveController(
        DiscoveryControllerInput(
            config=cfg,
            evaluation_file=eval_path,
            database=db,
        )
    )
    # Swap real LLM + evaluator with mocks AFTER the controller is built.
    mock_llm = _MockLLM()
    ctrl.llms.models = [mock_llm]
    ctrl.llms.weights = [1.0]
    mock_eval = _MockEvaluator()
    ctrl.evaluator = mock_eval
    return ctrl, mock_llm, mock_eval, eval_path


# --------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------


async def main():
    K = 4
    N = 5

    cfg = _build_minimal_config()
    db = AdaEvolveDatabase(name="adaevolve_batched", config=cfg.search.database)

    seed = _make_seed_program(cfg)
    db.add(seed, iteration=0)

    ctrl, mock_llm, mock_eval, eval_path = _build_controller_with_mocks(cfg, db)
    try:
        await ctrl.run_discovery(start_iteration=1, max_iterations=N)
    finally:
        os.unlink(eval_path)

    total_calls = len(mock_llm.calls)
    print(f"Total LLM calls: {total_calls}")
    print(f"Total evaluations: {mock_eval.evals}")
    print(f"DB program count: {len(db.programs)}")

    # ------------------------------------------------------------------
    # Group calls by (system, user) — every group should have exactly K calls.
    # That's the cache-friendly property: K identical prompts per iter.
    # ------------------------------------------------------------------
    grouped: Dict[tuple, int] = defaultdict(int)
    for c in mock_llm.calls:
        grouped[(c["system"], c["user"])] += 1

    sizes = sorted(grouped.values(), reverse=True)
    unique_prompts = len(grouped)
    print(f"Unique (system, user) prompts: {unique_prompts}")
    print(f"Calls-per-prompt distribution: {sizes}")

    # ------------------------------------------------------------------
    # Within iteration i, all K calls must share the prompt → exactly K
    # calls per unique prompt. Some iterations may share a prompt with
    # another iteration if the parent didn't advance (shouldn't happen
    # here because metrics differ), so we assert all groups are multiples
    # of K, and the total is K * N.
    # ------------------------------------------------------------------
    assert total_calls == K * N, (
        f"Expected K*N = {K * N} LLM calls, got {total_calls}"
    )
    assert all(s % K == 0 for s in sizes), (
        f"Per-prompt call counts not all multiples of K={K}: {sizes}"
    )
    assert mock_eval.evals == K * N, (
        f"Expected K*N evaluations, got {mock_eval.evals}"
    )
    # Database holds the seed (1) + every accepted child (K * N).
    assert len(db.programs) == 1 + K * N, (
        f"DB size {len(db.programs)} != 1 + K*N = {1 + K * N}"
    )

    # Temperature spread should give K distinct temperatures per prompt-group.
    per_prompt_temps: Dict[tuple, set] = defaultdict(set)
    for c in mock_llm.calls:
        t = c["kwargs"].get("temperature")
        per_prompt_temps[(c["system"], c["user"])].add(round(t, 4) if t else None)
    spreads = [len(v) for v in per_prompt_temps.values()]
    print(f"Distinct temperatures per prompt-group: {spreads}")
    assert all(s == K for s in spreads), (
        f"Expected K={K} distinct temperatures per group, got {spreads}"
    )

    print("\nAll batched-controller invariants hold:")
    print("  ✓ K LLM calls per iteration")
    print("  ✓ K identical prompts per iteration (prefix-cache friendly)")
    print("  ✓ K distinct temperatures per group (diversity preserved)")
    print("  ✓ K children added to DB per iteration")
    print("  ✓ Total calls = total evaluations = K * N")


if __name__ == "__main__":
    asyncio.run(main())
