unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

KMM_LAMBDA=0.0005

# NOTE: Update KMM_ARTIFACTS_DIR to point to your one_step attribution output directory
# This should be the artifacts folder from a previous one_step attribution run
KMM_ARTIFACTS_DIR="${DATA_DIR}/outputs_sft_aya_qwen3_1.7b_danish/attribution/<one_step_run_folder>/artifacts"

CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false python attribution/run_attribution.py \
 --config ${PROJECT_ROOT}/configs/sft/exp_attribution_danish_multitask.yaml --k_shot 4 --method kmm \
 --kmm_artifacts_dir ${KMM_ARTIFACTS_DIR} \
 --kmm_lambda $KMM_LAMBDA --kmm_source one_step \
 2>&1 | tee "${DATA_DIR}/outputs_sft_aya_qwen3_1.7b_danish/attribution/kmm_attribution.log"
