#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-3}
PORT=${2:-8230}

echo "Starting vLLM (Qwen3-4B) on GPU $GPU, port $PORT..."

docker run -d --gpus "\"device=$GPU\"" \
  --name "vllm_qwen3_27b" \
  -v /scratch/huggingface_cache:/root/.cache/huggingface \
  -p ${PORT}:${PORT} \
  -e HF_HOME=/root/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
  vllm/vllm-openai:v0.19.1 \
  --model Qwen/Qwen3-32B-FP8 \
  --trust-remote-code \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.92 \
  --port ${PORT}


echo "Started vllm_qwen3_27b on GPU $GPU, port $PORT"
echo "To stream logs: docker logs -f vllm_qwen3_27b"
echo "To stop:        docker stop vllm_qwen3_27b && docker rm vllm_qwen3_27b"
