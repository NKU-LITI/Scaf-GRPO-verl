#!/usr/bin/env bash
# [ADD] Migrated from Scaf-GRPO/sh; keep original experiment settings unless required by verl 0.7.
set -euo pipefail

source /home/liting/miniconda3/etc/profile.d/conda.sh
conda activate scaf-grpo

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn


model_path="/workplace/nankai/share_space/model/models/LLMs/Qwen/Qwen2.5-Math-7B"

input_path="data/DeepScaleR/Qwen2.5_math_7b/Qwen2.5-Math-7B.with_teacher_expert_trajectory.reward1.parquet"

output_dir="data/DeepScaleR/Qwen2.5_math_7b/stratified_success_rate_k8"


train_output="${output_dir}/train_800.success_rate_k8.parquet"
val_output="${output_dir}/val_200.success_rate_k8.parquet"


mkdir -p ${output_dir}


python3 scripts/build_stratified_success_rate_dataset.py \
    --input ${input_path} \
    --train-output ${train_output} \
    --val-output ${val_output} \
    --rollout-jsonl ${output_dir}/success_rate_k8.rollouts.jsonl \
    --model_path ${model_path} \
    --k 8 \
    --train-quotas "hard=400,medium=200,easy=200" \
    --val-quotas "hard=100,medium=50,easy=50" \
    --backend vllm \
    --tensor_parallel_size 2 \
    --batch_size 128 \
    --max_model_len 4096 \
    --max_new_tokens 2048 \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k -1 \
    --gpu_memory_utilization 0.90 \
    --resume \
    "$@"
