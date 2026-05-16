#!/usr/bin/env bash
set -euo pipefail   # stop on first failure

conda deactivate
conda activate ScalableAI
export OPENAI_API_KEY="dummy"

MODEL="vllm/Qwen/Qwen3.5-27B-FP8"
API_BASE="http://localhost:8227/v1"

mkdir -p results

RUNS=3
TIMEOUT=7200  # 2 hours per run

# run <name> <initial_program> <evaluator> <config> [extra args...]
run() {
    local name=$1
    local initial=$2
    local evaluator=$3
    local config=$4
    shift 4
    for run_id in $(seq 1 $RUNS); do
        local tag="${name}_run${run_id}_spec"
        local logfile="results/${tag}.log"
        echo "=== Starting: $tag (log: $logfile) ==="
        timeout "$TIMEOUT" python -m skydiscover.cli \
            "$initial" \
            "$evaluator" \
            --config "$config" \
            --model  "$MODEL" \
            --api-base "$API_BASE" \
            --output "results/$tag" \
            "$@" 2>&1 | tee "$logfile" || true
        local rc=${PIPESTATUS[0]}
        if [ "$rc" -eq 124 ]; then
            echo "=== TIMEOUT: $tag exceeded ${TIMEOUT}s, moving to next ==="
        elif [ "$rc" -ne 0 ]; then
            echo "=== FAILED: $tag exited with code $rc, moving to next ==="
        else
            echo "=== Finished: $tag ==="
        fi
    done
}

SP=benchmarks/math/signal_processing
SP_SWEEP=$SP/k_sweep
CP=benchmarks/math/circle_packing
CP_SWEEP=$CP/k_sweep

# run "signal_processing_k4" \
#     "$SP/initial_program.py" "$SP/evaluator/evaluator.py" \
#     "$SP_SWEEP/k4_spec.yaml" --iterations 200

run "circle_packing_k4" \
    "$CP/initial_program.py" "$CP/evaluator.py" \
    "$CP_SWEEP/k4_spec.yaml" --iterations 200
