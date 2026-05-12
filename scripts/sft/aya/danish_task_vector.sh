unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python attribution/run_attribution.py \
 --config ${PROJECT_ROOT}/configs/sft/exp_attribution_danish_multitask.yaml --k_shot 4 --method task_vector \
 2>&1 | tee "${DATA_DIR}/outputs_sft_aya_qwen3_1.7b_danish/attribution/task_vector_attribution.log"