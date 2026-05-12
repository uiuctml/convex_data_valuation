unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

# NOTE: Update ARTIFACTS_DIR to point to your datamodel attribution output directory
# Option 1: Refit from uniform_data_model run
UNIFORM_ARTIFACTS_DIR="${DATA_DIR}/outputs_sft_aya_qwen3_1.7b_danish/attribution/<uniform_data_model_run_folder>/artifacts"

# Option 2: Refit from cs_data_model run
CS_ARTIFACTS_DIR="${DATA_DIR}/outputs_sft_aya_qwen3_1.7b_danish/attribution/<cs_data_model_run_folder>/artifacts"

# Refit from uniform_data_model
CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false python attribution/run_attribution.py\
 --config ${PROJECT_ROOT}/configs/sft/exp_attribution_danish_multitask.yaml --k_shot 4 --method datamodel_refit \
 --num_rows 250 --include_frac 0.5 --alpha 0.00001 \
 --artifacts_dir ${UNIFORM_ARTIFACTS_DIR} \
 2>&1 | tee "${DATA_DIR}/outputs_sft_aya_qwen3_1.7b_danish/attribution/data_model_refit_uniform_attribution.log"

# Refit from cs_data_model
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false python attribution/run_attribution.py\
 --config ${PROJECT_ROOT}/configs/sft/exp_attribution_danish_multitask.yaml --k_shot 4 --method datamodel_refit \
 --num_rows 125 --alpha 0.000001 \
 --artifacts_dir ${CS_ARTIFACTS_DIR} \
 2>&1 | tee "${DATA_DIR}/outputs_sft_aya_qwen3_1.7b_danish/attribution/data_model_refit_cs_attribution.log"
