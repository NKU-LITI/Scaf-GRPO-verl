# Scaf-GRPO data utilities

<!-- [ADD] This directory contains the standalone data-preparation portion of
the original Scaf-GRPO `scripts/` directory, migrated for the verl 0.7 tree. -->

The scripts here operate on parquet/JSONL files and may be run from the
repository root, for example:

```bash
python3 scripts/scaf_grpo/match_teacher_trajectories.py --help
python3 scripts/scaf_grpo/prepare_clean_expert_trajectories.py --help
python3 scripts/scaf_grpo/build_initial_wrong_validation_set.py --help
python3 scripts/scaf_grpo/build_stratified_success_rate_dataset.py --help
```

Migrated standalone utilities:

- `match_teacher_trajectories.py`: attach a teacher target to the training parquet.
- `prepare_clean_expert_trajectories.py`: remove `<think>` content, length-filter, and verify expert answers.
- `filter_expert_trajectories_by_reward.py`: retain reward-verified expert trajectories.
- `filter_solution_breakdown_cot_answer.py`: verify scaffold solution-breakdown fields.
- `add_short_expert_trajectory_field.py`: create a concise post-think trajectory column.
- `sample_from_cached_success_rate.py`: create stratified train/validation subsets from cached rollout logs.
- `remap_rollout_cache_by_question.py`: map rollout cache ids after a parquet rewrite.
- `build_initial_wrong_validation_set.py`: derive a fixed all-wrong validation set from rollout logs.
- `data_rewrite.py`: migrated Mind-the-Gap generation backend used by rollout-based data builders.
- `build_stratified_success_rate_dataset.py`: sample rollouts and build hard/medium/easy train-validation splits.
- `generate_initial_wrong_validation_set.py`: run the initial policy and save a fixed pass@N=0 validation set.

The training shell entry points are under `sh/`.
