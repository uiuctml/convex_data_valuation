#!/usr/bin/env bash
set -e

# Accept environment variables with defaults
CKPT_DIR=${CKPT_DIR:-""}
TASKS=${TASKS:-"hendrycks_math"}
GPU=${GPU:-"0"}
BATCH_SIZE=${BATCH_SIZE:-"auto"}
OUTPUT_PATH=${OUTPUT_PATH:-"./lmeval_results"}

echo "Starting lm-eval with:"
echo "  CKPT_DIR=${CKPT_DIR}"
echo "  TASKS=${TASKS}"
echo "  GPU=${GPU}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  OUTPUT_PATH=${OUTPUT_PATH}"

# Run lm-eval with CUDA_VISIBLE_DEVICES set
CUDA_VISIBLE_DEVICES=${GPU} lm_eval \
  --model hf \
  --model_args pretrained=${CKPT_DIR} \
  --tasks ${TASKS} \
  --device cuda:0 \
  --batch_size ${BATCH_SIZE} \
  --output_path ${OUTPUT_PATH}

echo "lm-eval completed successfully"
