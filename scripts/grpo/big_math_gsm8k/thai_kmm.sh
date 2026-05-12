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

# NOTE: Update KMM_ARTIFACTS_DIR to point to your one_step attribution output directory
# This should be the artifacts folder from a previous one_step attribution run
KMM_ARTIFACTS_DIR="${DATA_DIR}/outputs_qwen2.5_1.5b_instruct_multilingual_grpo_th/attribution/<one_step_run_folder>/artifacts"
KMM_LAMBDA=0.00005

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python attribution/run_attribution.py \
 --config ${PROJECT_ROOT}/configs/grpo/exp_qwen2.5_1.5b_instruct_big_math_gsm8k_th.yaml --k_shot 4 --method kmm \
 --kmm_artifacts_dir ${KMM_ARTIFACTS_DIR} \
 --kmm_lambda ${KMM_LAMBDA} --kmm_source one_step \
 2>&1 | tee "${DATA_DIR}/outputs_qwen2.5_1.5b_instruct_multilingual_grpo_th/attribution/kmm_attribution.log"