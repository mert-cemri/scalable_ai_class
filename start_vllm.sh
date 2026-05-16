#!/usr/bin/env bash
set -euo pipefail

for i in $(seq 0 1); do
  PORT=$((8223 + i))
  echo "Starting vLLM on GPU $i, port $PORT..."
  docker run -d --gpus "\"device=$i\"" \
    --name "vllm_gpu${i}_new" \
    -v /scratch/huggingface_cache:/root/.cache/huggingface \
    -p ${PORT}:${PORT} \
    -e HF_HOME=/root/.cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
    -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
    -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
    vllm/vllm-openai:v0.19.1 \
    --model Qwen/Qwen3.5-27B-FP8 \
    --trust-remote-code \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --max-model-len 32768 \
    --max-num-seqs 50 \
    --gpu-memory-utilization 0.95 \
    --port ${PORT}
  echo "Started vllm_gpu${i} on port $PORT"
done

echo "All 1 vLLM instances started (ports 8223-8226)."
echo "To check status: docker ps --filter name=vllm_gpu"
echo "To stream logs:  docker logs -f vllm_gpu0"
