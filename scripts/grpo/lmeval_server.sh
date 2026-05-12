#!/usr/bin/env bash
set -e

# Accept environment variables with defaults
CKPT_DIR=${CKPT_DIR:-""}
TASKS=${TASKS:-"hendrycks_math"}
GPU=${GPU:-"0"}
BATCH_SIZE=${BATCH_SIZE:-"auto"}
OUTPUT_PATH=${OUTPUT_PATH:-"./lmeval_results"}

# vLLM-specific settings
DTYPE=${DTYPE:-"auto"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-"1"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.8"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"20000"}

# Generation settings
SYSTEM_INSTRUCTION=${SYSTEM_INSTRUCTION:-"Please reason step by step, and put your final answer within \\boxed{}."}
GEN_KWARGS=${GEN_KWARGS:-"do_sample=True,temperature=0.6,top_p=0.95,max_gen_toks=20000"}

# Get model_name from run_info.json (parent directory of CKPT_DIR)
PARENT_DIR=$(dirname "${CKPT_DIR}")
RUN_INFO_PATH="${PARENT_DIR}/run_info.json"

if [[ ! -f "${RUN_INFO_PATH}" ]]; then
    echo "ERROR: run_info.json not found at ${RUN_INFO_PATH}"
    exit 1
fi

# Extract model_name from run_info.json using python
MODEL_NAME=$(python3 -c "import json; print(json.load(open('${RUN_INFO_PATH}'))['model_name'])")

echo "Starting lm-eval (vLLM) with:"
echo "  CKPT_DIR=${CKPT_DIR}"
echo "  MODEL_NAME=${MODEL_NAME}"
echo "  TASKS=${TASKS}"
echo "  GPU=${GPU}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  OUTPUT_PATH=${OUTPUT_PATH}"
echo "  DTYPE=${DTYPE}"
echo "  TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo "  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"

# Run lm-eval with vLLM backend
CUDA_VISIBLE_DEVICES=${GPU} lm_eval \
  --model vllm \
  --model_args pretrained=${MODEL_NAME},lora_local_path=${CKPT_DIR},dtype=${DTYPE},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN} \
  --tasks ${TASKS} \
  --device cuda:0 \
  --batch_size ${BATCH_SIZE} \
  --system_instruction "${SYSTEM_INSTRUCTION}" \
  --apply_chat_template \
  --log_samples \
  --gen_kwargs "${GEN_KWARGS}" \
  --output_path ${OUTPUT_PATH}

echo "lm-eval (vLLM) completed successfully"
