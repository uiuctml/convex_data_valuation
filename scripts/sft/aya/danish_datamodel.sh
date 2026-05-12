unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false python attribution/run_attribution.py \
 --config ${PROJECT_ROOT}/configs/sft/exp_attribution_danish_multitask.yaml --k_shot 4 --method uniform_data_model \
 --num_rows 250 --include_frac 0.5 --alpha 0.0005 \
 2>&1 | tee "${DATA_DIR}/outputs_sft_aya_qwen3_1.7b_danish/attribution/uniform_data_model_attribution.log"