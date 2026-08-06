# Scaf-GRPO on verl 0.7

This repository ports **Scaf-GRPO** (Scaffolded Group Relative Policy Optimization) to the verl 0.7 training stack.

Scaf-GRPO addresses zero-reward GRPO groups with progressive scaffolding. After an initial rollout, it identifies questions with pass@k = 0, generates increasingly explicit hints, and uses successful hinted trajectories or verified expert trajectories to restore a learning signal.

> This is a research implementation built on verl. Scaf-GRPO intervention currently uses verl's legacy RayPPOTrainer; ordinary verl workloads can continue to use the v0.7 V1 trainer.

## What is implemented

- **Hierarchical hint rollout**: cumulative knowledge, planning, and solution hints for initially all-wrong groups.
- **Minimal-hint replacement**: selects the successful trajectory with the smallest hint level and replaces a failed member of that GRPO group.
- **Expert fallback**: injects an expert response when all enabled hint stages fail.
- **Mixed policy objective**: ordinary rollout tokens use PPO; injected expert tokens use probability-weighted off-policy RL.
- **Auxiliary SFT losses**: optional NLL losses for expert responses and selected hint responses.
- **Data utilities**: teacher matching, expert cleaning/filtering, scaffold verification, rollout-cache remapping, and fixed initial-wrong validation-set construction.

The original paper and dataset are available at [arXiv](https://arxiv.org/abs/2510.19807) and [Hugging Face](https://huggingface.co/datasets/hkuzxc/scaf-grpo-dataset).

## Repository layout

~~~text
verl/                         verl 0.7 code, including Scaf-GRPO extensions
sh/hint_mix_grpo/             migrated experiment shell scripts
examples/scaf_grpo/           parameterized Scaf-GRPO launcher
scripts/scaf_grpo/            standalone data-preparation utilities
data/                         training and evaluation parquet files
~~~

The original experiment scripts are retained under sh/. Their settings are intentionally preserved; only the old training entry point is redirected to verl.trainer.main_ppo.

## Installation

Use a clean Python environment compatible with your selected verl 0.7 backend (CUDA/NPU, vLLM/SGLang, FSDP/Megatron).

~~~bash
git clone <your-fork-url> scaf-grpo-verl
cd scaf-grpo-verl

uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
~~~

Install the rollout backend and hardware-specific dependencies following the [verl installation guide](https://verl.readthedocs.io/en/latest/start/install.html). The environment must provide PyTorch, Ray, Transformers, a supported rollout engine, and the reward-function dependencies used by your dataset.

## Data format

Each training row must contain verl's normal prompt/reward fields and the Scaf fields below.

~~~text
prompt                         chat messages or the configured prompt field
data_source                    reward-function identifier
reward_model / extra_info      ground-truth metadata required by the reward function

question                       original question text
knowledge_components_parts     list[str], least explicit scaffold
planning_skeleton_parts        list[str], intermediate scaffold
solution_breakdown_parts       list[str], most explicit scaffold
expert_target                  verified fallback response
~~~

For backward compatibility, the dataset loader also accepts expert_trajectory or target and normalizes either to expert_target. Empty scaffold lists are valid. An all-wrong sample without usable hints can still use expert_target when expert fallback is enabled.

## Quick start

Set the model and dataset locations, then run the parameterized launcher:

~~~bash
export MODEL_PATH=/path/to/Qwen2.5-Math-1.5B
export TRAIN_FILE=data/DeepScaleR/train.parquet
export VAL_FILE=data/DeepScaleR/validation.parquet
export OUTPUT_DIR=outputs/scaf_grpo_run

bash examples/scaf_grpo/run_scaf_grpo.sh
~~~

The launcher defaults to a full Scaf-GRPO experiment:

~~~text
WITH_HINT=true
WITH_EXPERT_FALLBACK=true
HINT_STAGE_COUNT=2
USE_OFF_POLICY_LOSS=true
EXPERT_SFT_LOSS_COEF=1.0
USE_HINT_SFT_LOSS=true
HINT_SFT_LOSS_COEF=1.0
~~~

Typical ablations:

~~~bash
# Plain GRPO
WITH_HINT=false WITH_EXPERT_FALLBACK=false \
  USE_OFF_POLICY_LOSS=false EXPERT_SFT_LOSS_COEF=0 \
  bash examples/scaf_grpo/run_scaf_grpo.sh

# Hint rollout without expert fallback
WITH_HINT=true WITH_EXPERT_FALLBACK=false \
  bash examples/scaf_grpo/run_scaf_grpo.sh

# Hint + expert off-policy RL, no auxiliary hint SFT
WITH_HINT=true WITH_EXPERT_FALLBACK=true \
  USE_HINT_SFT_LOSS=false HINT_SFT_LOSS_COEF=0 \
  bash examples/scaf_grpo/run_scaf_grpo.sh
~~~

For fixed historical configurations, use the migrated scripts in sh/hint_mix_grpo/ after updating their model and data paths.

## Training behavior

For each GRPO group:

1. Generate rollout.n initial responses and score them.
2. If any response is correct, retain the original group.
3. For an all-wrong group, generate cumulative scaffold prompts from the enabled stages.
4. If a hint trajectory succeeds, select the lowest successful hint level and replace one failed trajectory.
5. If all hint trajectories fail, optionally replace one failed trajectory with expert_target.
6. Keep the group size unchanged, then recompute rewards for replaced rows and run GRPO/PPO.

Expert response tokens are marked with off_policy_mask and sft_loss_mask; selected hint tokens can be marked with hint_sft_loss_mask. Ordinary rollout data remains on-policy, while the intervention trajectories receive their intended auxiliary objectives.

## Data utilities

The following scripts are available under scripts/scaf_grpo/:

~~~text
match_teacher_trajectories.py
prepare_clean_expert_trajectories.py
filter_expert_trajectories_by_reward.py
filter_solution_breakdown_cot_answer.py
add_short_expert_trajectory_field.py
sample_from_cached_success_rate.py
remap_rollout_cache_by_question.py
build_initial_wrong_validation_set.py
~~~

Each script exposes its arguments through --help. See [scripts/scaf_grpo/README.md](scripts/scaf_grpo/README.md) for its role in the preparation workflow.

## Current limitations

- Enabling trainer.with_hint or trainer.with_expert_fallback routes training to the legacy v0 trainer path; V1 replay-buffer support for Scaf interventions is not implemented.
- The primary launcher targets text-only mathematical reasoning. Multi-modal or tool-agent data requires extending the hint-request builder.
- The two legacy offline-generation scripts that depend on hint_mix_grpo.data_rewrite are not included. Use the migrated standalone utilities or port their generator backend separately.
- Run a short single-node smoke test before launching a multi-node experiment; model/backend combinations have different memory limits.

## Citation

~~~bibtex
@inproceedings{zhang2026scafgrpo,
  title={Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning},
  author={Xichen Zhang and Sitong Wu and Yinghao Zhu and Haoru Tan and Shaozuo Yu and Ziyi He and Jiaya Jia},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
~~~

## Acknowledgements

This project builds on [verl](https://github.com/volcengine/verl), vLLM, and the Scaf-GRPO paper and dataset contributors.
