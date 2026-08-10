#!/usr/bin/env bash
set -x
set -euo pipefail

source /home/liting/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-verl070}"
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

RUN_DIR="${RUN_DIR:-outputs/qwen25_math7b_grpo_baseline}"
CKPT_DIR="${CKPT_DIR:-${RUN_DIR}/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/rollout_log/validation_with_hint}"
PROJECT_NAME="${PROJECT_NAME:-scaf-grpo-expert-sft}"
MODEL_PATH="${MODEL_PATH:-/workplace/nankai/liting_space/LLM/Qwen2.5-Math-7B}"
DATA_SEED="${DATA_SEED:-42}"

data_dir="${DATA_DIR:-data/DeepScaler/Qwen2d5_math_7b}"
data_train_path="${DATA_TRAIN_PATH:-${data_dir}/train_800.success_rate_k8.parquet}"
data_val_path="${DATA_VAL_PATH:-${data_dir}/val_200.success_rate_k8.parquet}"

if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "ERROR: checkpoint directory not found: ${CKPT_DIR}" >&2
    exit 1
fi

mapfile -t ckpt_paths < <(
    find "${CKPT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' \
        | sort -V
)

if [[ "${#ckpt_paths[@]}" -eq 0 ]]; then
    echo "ERROR: no global_step_* checkpoints found under ${CKPT_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

for model_path in "${ckpt_paths[@]}"; do
    step_name="$(basename "${model_path}")"
    step_num="${step_name#global_step_}"
    ckpt_log="${OUTPUT_DIR}/eval_${step_name}.log"

    if [[ -e "${OUTPUT_DIR}/${step_num}.jsonl" ]]; then
        echo "ERROR: validation trajectory file already exists: ${OUTPUT_DIR}/${step_num}.jsonl" >&2
        echo "Remove it or set OUTPUT_DIR to a new location to avoid overwriting previous validation outputs." >&2
        exit 1
    fi

    {
    printf '\n===== validate %s at %s =====\n' "${step_name}" "$(date '+%Y-%m-%d %H:%M:%S')"

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        \
        data.train_files="${data_train_path}" \
        data.val_files="${data_val_path}" \
        data.shuffle=True \
        data.seed="${DATA_SEED}" \
        data.train_batch_size=64 \
        data.val_batch_size=64 \
        data.max_prompt_length=4096 \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.max_response_length=2048 \
        data.scaf_fixed_hint_key=planning_skeleton_parts \
        data.scaf_fixed_hint_count=1 \
        data.scaf_fixed_hint_label="Planning Hints" \
        data.scaf_fixed_hint_apply_to_val=True \
        \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_remove_padding=True \
        \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        algorithm.use_kl_in_reward=False \
        actor_rollout_ref.actor.entropy_coeff=0 \
        \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=8 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.agent.num_workers=8 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        reward_model.use_reward_loop=False \
        \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.actor.optim.warmup_style=constant \
        actor_rollout_ref.actor.optim.weight_decay=0.0 \
        \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=2 \
        trainer.total_epochs=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.val_only_step="${step_num}" \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="${model_path}" \
        trainer.warmup_steps=5 \
        \
        trainer.with_hint=False \
        trainer.with_expert_fallback=False \
        trainer.hint_stage_count=3 \
        trainer.replace_hint_prompt_response=True \
        trainer.replace_num=0 \
        trainer.expert_truncation=left \
        \
        actor_rollout_ref.actor.use_off_policy_loss=False \
        actor_rollout_ref.actor.sft_loss_coef=0.0 \
        actor_rollout_ref.actor.use_hint_sft_loss=False \
        actor_rollout_ref.actor.hint_sft_loss_coef=0.0 \
        \
        trainer.rollout_data_dir=null \
        trainer.validation_data_dir="${OUTPUT_DIR}" \
        trainer.default_local_dir="${OUTPUT_DIR}/checkpoints_unused/${step_name}" \
        \
        trainer.logger="['console','wandb']" \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${RUN_DIR}_validation_with_hint_${step_name}" \
        "$@"

    python3 - "${step_name}" "${model_path}" "${OUTPUT_DIR}/metrics.jsonl" "${OUTPUT_DIR}/summary_metrics.jsonl" <<'PY'
import json
import sys
from pathlib import Path

step_name, model_path, metrics_path, summary_path = sys.argv[1:]
metrics_file = Path(metrics_path)
summary_file = Path(summary_path)

if not metrics_file.exists():
    raise FileNotFoundError(f"metrics file not found: {metrics_file}")

last_entry = None
with metrics_file.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            last_entry = json.loads(line)

if last_entry is None:
    raise RuntimeError(f"metrics file is empty: {metrics_file}")

summary_entry = {
    "ckpt": step_name,
    "model_path": model_path,
    "validation_dir": str(metrics_file.parent),
    "step": last_entry.get("step"),
    "metrics": last_entry.get("metrics", {}),
}
with summary_file.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")
PY
    } 2>&1 | tee -a "${ckpt_log}"
done
