#!/usr/bin/env bash
set -e

# Qwen/Qwen2.5-Math-1.5B-AWQ for quantization
# --kv-cache-dtype fp8 \
# CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
#   --model Qwen/Qwen2.5-1.5B-Instruct \
#   --dtype bfloat16 \
#   --tensor-parallel-size 1 \
#   --enforce-eager \
#   --port 10245

# Accept environment variables with defaults
PORT=${PORT:-10245}
GPU=${GPU:-1}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}

echo "Starting vLLM server on GPU ${GPU}, port ${PORT}, model ${MODEL}"

# Explicitly unset distributed training environment variables to prevent NCCL conflicts
unset WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT

CUDA_VISIBLE_DEVICES=${GPU} trl vllm-serve \
  --model ${MODEL} \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --port ${PORT}