#!/usr/bin/env bash
set -x
set -euo pipefail

source /home/liting/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-verl070}"
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"

PROJECT_NAME="${PROJECT_NAME:-scaf-grpo-expert-sft}"
EXP_NAME="${EXP_NAME:-outputs/qwen25_math7b_sft_800}"
MODEL_PATH="${MODEL_PATH:-/workplace/nankai/liting_space/LLM/Qwen2.5-Math-7B}"
DATA_SEED="${DATA_SEED:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

data_dir="${DATA_DIR:-data/DeepScaler/Qwen2d5_math_7b}"
data_train_path="${DATA_TRAIN_PATH:-${data_dir}/train_800.success_rate_k8.parquet}"
data_val_path="${DATA_VAL_PATH:-${data_dir}/val_200.success_rate_k8.parquet}"

mkdir -p "${EXP_NAME}"
exec > >(tee -a "${EXP_NAME}/train.log") 2>&1

printf '\n===== SFT restart %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')"

# This logs train/loss and val/loss to the same W&B project as Scaf-GRPO.
# test_freq/save_freq=5 matches the current Scaf-GRPO validation cadence.
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${data_train_path}" \
    data.val_files="${data_val_path}" \
    data.prompt_key=question \
    data.response_key=solution_breakdown_cot_answer \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=6144 \
    data.truncation=right \
    model.partial_pretrain="${MODEL_PATH}" \
    model.enable_gradient_checkpointing=True \
    model.fsdp_config.model_dtype=bf16 \
    model.fsdp_config.cpu_offload=False \
    model.fsdp_config.offload_params=False \
    model.use_liger=False \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.0 \
    trainer.total_epochs=10 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${EXP_NAME}/checkpoints" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.logger="['console','wandb','file']" \
    trainer.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    trainer.checkpoint.load_contents="['model','optimizer','extra']" \
    trainer.max_ckpt_to_keep=null \
    trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=True \
    "$@"
