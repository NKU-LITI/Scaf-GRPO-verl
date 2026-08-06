#!/usr/bin/env bash
# [ADD] Migrated from Scaf-GRPO/sh; keep original experiment settings unless required by verl 0.7.
set -euo pipefail
set -x

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/repos/LUFFY}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/runs/LUFFY/mtg_luffy_${RUN_TAG}}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/data/models/Qwen2.5-Math-1.5B}"
TRAIN_FILE="${TRAIN_FILE:-/root/autodl-tmp/data/datasets/openr1_math_46k/hf/openr1.parquet}"
VAL_FILE="${VAL_FILE:-${REPO_ROOT}/data/valid.parquet}"
REWRITTEN_TRAIN_FILE="${REWRITTEN_TRAIN_FILE:-${RUN_ROOT}/data/openr1.mtg_rewritten.parquet}"
SKIP_REWRITE="${SKIP_REWRITE:-0}"
USE_OFF_POLICY_PROBS="${USE_OFF_POLICY_PROBS:-1}"
if [[ "${USE_OFF_POLICY_PROBS}" == "1" || "${USE_OFF_POLICY_PROBS}" == "true" || "${USE_OFF_POLICY_PROBS}" == "True" ]]; then
  USE_OFF_POLICY_PROBS_HYDRA=True
else
  USE_OFF_POLICY_PROBS_HYDRA=False
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
N_GPUS="${N_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-512}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_PREFIX_LEN="${MAX_PREFIX_LEN:-8192}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-30}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-luffy-math-mtg}"
EXP_NAME="${EXP_NAME:-LUFFY_MTG}"

mkdir -p \
  "${RUN_ROOT}/data" \
  "${RUN_ROOT}/ckpt" \
  "${RUN_ROOT}/logs" \
  "${RUN_ROOT}/wandb"

export PYTHONPATH="${REPO_ROOT}/luffy/verl:${REPO_ROOT}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export WANDB_DIR="${RUN_ROOT}/wandb"

if [[ "${SKIP_REWRITE}" != "1" ]]; then
  REWRITE_ARGS=()
  if [[ "${USE_OFF_POLICY_PROBS}" != "1" ]]; then
    REWRITE_ARGS+=(--no_compute_target_probs)
  fi

  "${PYTHON_BIN}" "${REPO_ROOT}/data/mind_the_gap_rewrite.py" \
    --input "${TRAIN_FILE}" \
    --output "${REWRITTEN_TRAIN_FILE}" \
    --model_path "${MODEL_PATH}" \
    --backend vllm \
    --tensor_parallel_size "${TP_SIZE}" \
    --max_model_len "$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
    --max_new_tokens "${MAX_RESPONSE_LENGTH}" \
    --batch_size "${REWRITE_BATCH_SIZE:-128}" \
    --self_n "${MTG_SELF_N:-4}" \
    --rewrite_n "${MTG_REWRITE_N:-4}" \
    --temperature "${MTG_TEMPERATURE:-1.0}" \
    --rewrite_temperature "${MTG_REWRITE_TEMPERATURE:-0.7}" \
    "${REWRITE_ARGS[@]}"
fi

ray stop || true
cd "${REPO_ROOT}/luffy/verl"

"${PYTHON_BIN}" -m verl.mix_src.main_mix_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${REWRITTEN_TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.kl_loss_coef=0.00 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_temperature=0.6 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.n_val=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.max_prefix_len="${MAX_PREFIX_LEN}" \
  algorithm.kl_ctrl.kl_coef=0.000 \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger="['console','wandb']" \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${EXP_NAME}" \
  +trainer.val_before_train=False \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.use_sft_prefix_reward=False \
  actor_rollout_ref.rollout.prefix_share_across_samples=False \
  actor_rollout_ref.rollout.prefix_strategy=random \
  actor_rollout_ref.rollout.n_prefix=1 \
  actor_rollout_ref.rollout.min_prefix_ratio=1.0 \
  actor_rollout_ref.rollout.max_prefix_ratio=1.0 \
  actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
  actor_rollout_ref.ref.use_ref=False \
  actor_rollout_ref.actor.use_off_policy_loss=True \
  actor_rollout_ref.actor.use_off_policy_probs="${USE_OFF_POLICY_PROBS_HYDRA}" \
  actor_rollout_ref.actor.off_policy_normalize=False \
  actor_rollout_ref.actor.off_policy_reshape=p_div_p_0.1 \
  actor_rollout_ref.actor.off_policy_loss_impl=token \
  algorithm.grpo_use_std=False \
  actor_rollout_ref.actor.loss_remove_token_mean=True \
  actor_rollout_ref.actor.loss_remove_clip=True \
  data.reward_impl_version=3 \
  trainer.max_optim_to_keep=2 \
  data.shuffle=True \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${RUN_ROOT}/ckpt" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "${@}"
