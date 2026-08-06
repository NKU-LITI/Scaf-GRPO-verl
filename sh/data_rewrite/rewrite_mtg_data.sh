#!/usr/bin/env bash
# [ADD] Migrated from Scaf-GRPO/sh; keep original experiment settings unless required by verl 0.7.
set -euo pipefail
set -x

if [[ -z "${REPO_ROOT:-}" && -n "${REPO_RooT:-}" ]]; then
  REPO_ROOT="${REPO_RooT}"
fi
REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/repos/LUFFY}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/runs/LUFFY/mtg_rewrite_${RUN_TAG}}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/data/models/Qwen2.5-Math-1.5B}"
TRAIN_FILE="${TRAIN_FILE:-/root/autodl-tmp/data/datasets/openr1_math_46k/hf/openr1.parquet}"
OUTPUT_FILE="${OUTPUT_FILE:-${RUN_ROOT}/openr1.mtg_rewritten.parquet}"
STATS_DIR="${STATS_DIR:-${RUN_ROOT}/stats}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
if [[ -z "${PYTHON_BIN:-}" && -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
BACKEND="${BACKEND:-vllm}"
TP_SIZE="${TP_SIZE:-2}" # GPU数
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
REWRITE_BATCH_SIZE="${REWRITE_BATCH_SIZE:-128}"
MTG_SELF_N="${MTG_SELF_N:-4}"
MTG_REWRITE_N="${MTG_REWRITE_N:-4}"
MTG_TEMPERATURE="${MTG_TEMPERATURE:-1.0}"
MTG_REWRITE_TEMPERATURE="${MTG_REWRITE_TEMPERATURE:-0.7}"
COMPUTE_TARGET_PROBS="${COMPUTE_TARGET_PROBS:-0}"
VERIFY_TIMEOUT_SECONDS="${VERIFY_TIMEOUT_SECONDS:-30}"

mkdir -p "${RUN_ROOT}" "${STATS_DIR}" "${CHECKPOINT_DIR}" "$(dirname "${OUTPUT_FILE}")"

export PYTHONPATH="${REPO_ROOT}/luffy/verl:${REPO_ROOT}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

REWRITE_ARGS=()
if [[ "${COMPUTE_TARGET_PROBS}" != "1" ]]; then
  REWRITE_ARGS+=(--no_compute_target_probs)
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  REWRITE_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  REWRITE_ARGS+=(--trust_remote_code)
fi
if [[ "${NO_REVEAL_ANSWER:-0}" == "1" ]]; then
  REWRITE_ARGS+=(--no_reveal_answer)
fi
if [[ "${NO_RESUME:-0}" == "1" ]]; then
  REWRITE_ARGS+=(--no_resume)
fi

"${PYTHON_BIN}" "${REPO_ROOT}/data/mind_the_gap_rewrite.py" \
  --input "${TRAIN_FILE}" \
  --output "${OUTPUT_FILE}" \
  --stats_dir "${STATS_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --backend "${BACKEND}" \
  --tensor_parallel_size "${TP_SIZE}" \
  --max_model_len "$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  --max_new_tokens "${MAX_RESPONSE_LENGTH}" \
  --batch_size "${REWRITE_BATCH_SIZE}" \
  --self_n "${MTG_SELF_N}" \
  --rewrite_n "${MTG_REWRITE_N}" \
  --temperature "${MTG_TEMPERATURE}" \
  --rewrite_temperature "${MTG_REWRITE_TEMPERATURE}" \
  --verify_timeout_seconds "${VERIFY_TIMEOUT_SECONDS}" \
  "${REWRITE_ARGS[@]}"

printf 'Rewritten parquet: %s\n' "${OUTPUT_FILE}"
printf 'Rewrite stats: %s\n' "${STATS_DIR}"
printf 'Rewrite checkpoints: %s\n' "${CHECKPOINT_DIR}"
