set -e

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eno1}
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

unset WORLD_SIZE
unset RANK
unset LOCAL_RANK

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 TOKENIZERS_PARALLELISM=false accelerate launch --num_processes=6 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no attribution/run_attribution.py \
 --config ${PROJECT_ROOT}/configs/grpo/exp_qwen2.5_1.5b_instruct_big_math_gsm8k_th.yaml --k_shot 4 --method task_vector \
 2>&1 | tee "${DATA_DIR}/outputs_qwen2.5_1.5b_instruct_multilingual_grpo_th/attribution/task_vector_attribution.log"