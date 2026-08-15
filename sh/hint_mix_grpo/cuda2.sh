#!/usr/bin/env bash
set -x
set -euo pipefail

source /home/liting/miniconda3/etc/profile.d/conda.sh



TARGET_PIDS=(1467261 1467262)

# TARGET_PIDS=(1795604 1795601)

echo "正在监测 PID: ${TARGET_PIDS[*]}，等待它们全部结束..."

while true; do
    alive=0

    for pid in "${TARGET_PIDS[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            alive=1
        fi
    done

    # 两个PID都不存在
    if [ "$alive" -eq 0 ]; then
        break
    fi

    sleep 30
done

echo "========================================"
echo "所有目标 PID 已结束，开始执行新的训练程序..."
echo "========================================"



export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"
export VLLM_USE_V1=1

PROJECT_NAME="scaf-grpo-expert-sft" 
EXP_NAME="${EXP_NAME:-outputs/qwen25_math7b_stratified_hint_replace_prompt_response_cuda2}"
MODEL_PATH="${MODEL_PATH:-/workplace/nankai/liting_space/LLM/Qwen2.5-Math-7B}"
DATA_SEED="${DATA_SEED:-42}"

mkdir -p "${EXP_NAME}"
exec > >(tee -a "${EXP_NAME}/train.log") 2>&1

printf '\n===== restart %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')"

reward_tag="math-verify"
prompt_tag="system-p1"


data_dir="${DATA_DIR:-data/DeepScaler/Qwen2d5_math_7b}"
data_train_path="${DATA_TRAIN_PATH:-${data_dir}/train_800.success_rate_k8.parquet}"
data_val_path="${DATA_VAL_PATH:-${data_dir}/val_200.success_rate_k8.parquet}"


# epoch=2, step=24, warmup_steps内lr线性增加到设置的值
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.n_gpus_per_node=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    reward_model.use_reward_loop=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    \
    trainer.nnodes=1 \
    trainer.total_epochs=20 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.warmup_steps=5 \
    \
    trainer.with_hint=True \
    trainer.with_expert_fallback=False \
    trainer.hint_stage_count=3 \
    trainer.replace_hint_prompt_response=True \
    algorithm.hint_is_correction=False \
    algorithm.hint_log_c_clip=5.0 \
    trainer.replace_num=1 \
    trainer.expert_truncation=left \
    \
    actor_rollout_ref.actor.use_off_policy_loss=False \
    actor_rollout_ref.actor.off_policy_reshape=p_div_p_0.1 \
    actor_rollout_ref.actor.sft_loss_coef=0.0 \
    actor_rollout_ref.actor.use_hint_sft_loss=False \
    actor_rollout_ref.actor.hint_sft_loss_coef=0.0 \
    \
    trainer.rollout_data_dir="${EXP_NAME}/rollout_log/training" \
    trainer.validation_data_dir="${EXP_NAME}/rollout_log/validation" \
    trainer.default_local_dir="${EXP_NAME}/checkpoints" \
    \
    trainer.logger="['console','wandb','file']" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    "$@"


    # actor_rollout_ref.rollout.max_model_len=12288 \
