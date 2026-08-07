#!/usr/bin/env bash
# [ADD] Migrated from Scaf-GRPO/sh; keep original experiment settings unless required by verl 0.7.
set -x
set -euo pipefail

source /home/liting/miniconda3/etc/profile.d/conda.sh
# conda activate scaf-grpo-use

export CUDA_VISIBLE_DEVICES=4
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=online
export VLLM_USE_V1=0


PROJECT_NAME="scaf-grpo-expert-sft"
EXP_NAME="outputs/qwen25_math15b_hint_expert_sft"
MODEL_PATH="/workplace/nankai/liting_space/LLM/Qwen2.5-Math-1.5B"
DATA_SEED="${DATA_SEED:-42}"


mkdir -p ${EXP_NAME}
exec > >(tee -a ${EXP_NAME}/train.log) 2>&1

printf '\n===== restart %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')"


reward_tag="math-verify"
prompt_tag="system-p1"


data_train_path="data/DeepScaleR/Qwen2d5_math_1d5b/Qwen2.5-Math-1.5B.with_40k_train_expert_trajectory.expert_post_think.reward1.parquet"
data_val_path="data/DeepScaleR/Qwen2d5_math_1d5b/val_from_traindata/Qwen2.5-Math-1.5B.initial_pass16_zero_100.parquet"

data_test_aime24="data/AIME24/${reward_tag}/${prompt_tag}/test.parquet"
data_test_aime25="data/AIME25/${reward_tag}/${prompt_tag}/test.parquet"
data_test_amc23="data/AMC23/${reward_tag}/${prompt_tag}/test.parquet"
data_test_math500="data/MATH-500/${reward_tag}/${prompt_tag}/test.parquet"
data_test_gaokao2023en="data/GaoKao2023en/${reward_tag}/${prompt_tag}/test.parquet"
data_test_olympiadbench="data/OlympiadBench/${reward_tag}/${prompt_tag}/test.parquet"
data_test_minerva="data/MinervaMath/${reward_tag}/${prompt_tag}/test.parquet"
data_test_path="['$data_test_aime24', '$data_test_aime25', '$data_test_amc23', '$data_test_math500', '$data_test_gaokao2023en', '$data_test_olympiadbench','$data_test_minerva']"



# [DEL] Removed unsupported verl 0.7 initial-wrong and hint dump overrides.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=${data_train_path} \
    data.val_files="${data_test_path}" \
    data.shuffle=True \
    data.seed="${DATA_SEED}" \
    data.train_batch_size=256 \
    data.val_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    reward_model.use_reward_loop=False \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    trainer.total_epochs=100 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.warmup_steps=50 \
    \
    trainer.with_hint=True \
    trainer.with_expert_fallback=True \
    trainer.hint_stage_count=2 \
    trainer.replace_hint_prompt_response=True \
    trainer.replace_num=1 \
    trainer.expert_truncation=left \
    \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_reshape=p_div_p_0.1 \
    \
    actor_rollout_ref.actor.sft_loss_coef=1.0 \
    actor_rollout_ref.actor.use_hint_sft_loss=False \
    actor_rollout_ref.actor.hint_sft_loss_coef=0.0 \
    \
    trainer.rollout_data_dir=${EXP_NAME}/rollout_log/training \
    trainer.validation_data_dir=${EXP_NAME}/rollout_log/validation \
    trainer.default_local_dir=${EXP_NAME}/checkpoints \
    \
    trainer.logger="['console','wandb','file']" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    "$@"
