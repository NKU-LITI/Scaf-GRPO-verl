#!/usr/bin/env bash
set -x
set -euo pipefail

source /home/liting/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-verl070}"
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export WANDB_MODE="${WANDB_MODE:-online}"

RUN_DIR="${RUN_DIR:-outputs/qwen25_math7b_sft_800}"
CKPT_DIR="${CKPT_DIR:-${RUN_DIR}/checkpoints}"
INTERVAL_STEPS="${INTERVAL_STEPS:-5}"
FINAL_STEP="${FINAL_STEP:-1000}"
VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-${RUN_DIR}/rollout_log/validation_ckpt_curve}"
VALIDATION_WANDB_RESUME="${VALIDATION_WANDB_RESUME:-allow}"

mkdir -p "${RUN_DIR}"

if [[ -z "${TRAIN_WANDB_RUN_ID:-}" && -n "${WANDB_RUN_ID:-}" ]]; then
    TRAIN_WANDB_RUN_ID="${WANDB_RUN_ID}"
fi
if [[ -z "${TRAIN_WANDB_RUN_ID:-}" && -f "${RUN_DIR}/train.log" ]]; then
    TRAIN_WANDB_RUN_ID="$(grep -oE '/runs/[A-Za-z0-9_-]+' "${RUN_DIR}/train.log" | tail -1 | awk -F/ '{print $3}')"
fi
TRAIN_WANDB_RUN_ID="${TRAIN_WANDB_RUN_ID:-qwen25_math7b_sft_800}"

if [[ -z "${VALIDATION_WANDB_RUN_ID:-}" && -d "${VALIDATION_OUTPUT_DIR}" ]]; then
    VALIDATION_WANDB_RUN_ID="$(
        grep -h -oE '/runs/[A-Za-z0-9_-]+' "${VALIDATION_OUTPUT_DIR}"/eval_global_step_*.log 2>/dev/null \
            | tail -1 \
            | awk -F/ '{print $3}'
    )"
fi

latest_step() {
    if [[ ! -d "${CKPT_DIR}" ]]; then
        echo 0
        return
    fi
    find "${CKPT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' \
        | sed 's|.*/global_step_||' \
        | sort -n \
        | tail -1
}

step="$(latest_step)"
if [[ -z "${step}" ]]; then
    step=0
fi

while (( step < FINAL_STEP )); do
    next_step=$((step + INTERVAL_STEPS))
    if (( next_step > FINAL_STEP )); then
        next_step="${FINAL_STEP}"
    fi

    WANDB_RUN_ID="${TRAIN_WANDB_RUN_ID}" \
    WANDB_RESUME=allow \
    bash sh/baseline/sft/qwen25_math7b_sft.sh \
        trainer.total_training_steps="${next_step}" \
        trainer.save_freq="${INTERVAL_STEPS}" \
        trainer.test_freq="${INTERVAL_STEPS}"

    if [[ -n "${VALIDATION_WANDB_RUN_ID:-}" ]]; then
        WANDB_RUN_ID="${VALIDATION_WANDB_RUN_ID}" \
        WANDB_RESUME="${VALIDATION_WANDB_RESUME}" \
        OUTPUT_DIR="${VALIDATION_OUTPUT_DIR}" \
        ONLY_STEP="${next_step}" \
        bash sh/baseline/sft/validate_sft_ckpts.sh
    else
        env -u WANDB_RUN_ID \
        WANDB_RESUME=never \
        OUTPUT_DIR="${VALIDATION_OUTPUT_DIR}" \
        ONLY_STEP="${next_step}" \
        bash sh/baseline/sft/validate_sft_ckpts.sh
    fi

    step="$(latest_step)"
    if [[ -z "${step}" ]]; then
        echo "ERROR: no checkpoint found after training to ${next_step}" >&2
        exit 1
    fi
done
