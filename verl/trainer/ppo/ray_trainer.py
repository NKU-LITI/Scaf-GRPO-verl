# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.scaf_grpo_utils import build_expert_response_data, build_hinted_gen_batch
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def compute_hint_is_weights(
    old_log_probs: torch.Tensor,
    hint_behavior_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    hint_offpolicy_mask: torch.Tensor,
    log_c_clip: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute detached sequence IS weights for response-only hinted trajectories."""
    if log_c_clip <= 0:
        raise ValueError(f"algorithm.hint_log_c_clip must be positive, got {log_c_clip}.")
    if old_log_probs.shape != hint_behavior_log_probs.shape or old_log_probs.shape != response_mask.shape:
        raise ValueError(
            "old_log_probs, hint_behavior_log_probs, and response_mask must have identical shapes; "
            f"got {old_log_probs.shape}, {hint_behavior_log_probs.shape}, and {response_mask.shape}."
        )

    trajectory_mask = hint_offpolicy_mask.to(device=old_log_probs.device, dtype=torch.bool).reshape(-1)
    if trajectory_mask.shape[0] != old_log_probs.shape[0]:
        raise ValueError(
            f"hint_offpolicy_mask must have one value per trajectory; got {trajectory_mask.shape[0]} "
            f"for batch size {old_log_probs.shape[0]}."
        )

    valid_tokens = response_mask.to(device=old_log_probs.device, dtype=torch.bool)
    token_log_c = old_log_probs.detach() - hint_behavior_log_probs.detach()
    log_c = torch.where(valid_tokens, token_log_c, torch.zeros_like(token_log_c)).sum(dim=-1)
    log_c_used = torch.clamp(log_c, min=-log_c_clip, max=log_c_clip) # log_c_clip=5，Cy=[old_policy(y|x)/old_policy(y|x_h)]都在这里clip掉了，测试样例Cy=-109
    weights = torch.where(trajectory_mask, torch.exp(log_c_used), torch.ones_like(log_c_used)).detach()

    selected_log_c = log_c[trajectory_mask]
    selected_weights = weights[trajectory_mask]
    metrics = {
        "hint_is/log_c_mean": selected_log_c.mean().item(),
        "hint_is/log_c_min": selected_log_c.min().item(),
        "hint_is/log_c_max": selected_log_c.max().item(),
        "hint_is/c_mean": selected_weights.mean().item(),
        "hint_is/c_min": selected_weights.min().item(),
        "hint_is/c_max": selected_weights.max().item(),
        "hint_is/clip_frac": (selected_log_c.abs() > log_c_clip).float().mean().item(),
    }
    return weights.unsqueeze(-1), metrics


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # [ADD] Scaf-GRPO: trainer-side switches for hinted rollout and expert trajectory fallback.
        self.with_hint = self.config.trainer.get("with_hint", False)
        self.warmup_steps = self.config.trainer.get("warmup_steps", 0)
        self.hint_stage_count = int(self.config.trainer.get("hint_stage_count", 3))
        if self.hint_stage_count not in (1, 2, 3):
            raise ValueError(f"trainer.hint_stage_count must be 1, 2, or 3; got {self.hint_stage_count}.")
        self.with_expert_fallback = self.config.trainer.get("with_expert_fallback", False)
        self.replace_hint_prompt_response = self.config.trainer.get("replace_hint_prompt_response", True)
        self.replace_num = int(self.config.trainer.get("replace_num", 1))
        # [ADD] `apply_bypass_mode` mutates this config. Retain the configured loss mode so mutated Scaf-GRPO batches can safely recompute old log-probs.
        self._scaf_default_policy_loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get(
            "loss_mode", "vanilla"
        )

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    @staticmethod
    def _normalize_difficulty(value) -> str:
        value = str(value or "unknown").strip().lower()
        if value == "medium":
            return "medium"
        if value in {"easy", "hard"}:
            return value
        return "unknown"

    def _select_difficulty_samples(self, inputs, outputs, gts, scores, difficulties, uids=None):
        """Select one easy, one medium, and one hard sample, preserving batch order."""
        target_difficulties = ("hard", "medium", "easy")
        selected = {}
        n = min(len(inputs), len(outputs), len(gts), len(scores), len(difficulties))
        uid_values = uids if uids is not None and len(uids) == n else [""] * n

        for i in range(n):
            difficulty = self._normalize_difficulty(difficulties[i])
            if difficulty not in target_difficulties or difficulty in selected:
                continue
            selected[difficulty] = {
                "step": self.global_steps,
                "split": None,
                "difficulty": difficulty,
                "uid": str(uid_values[i]),
                "input": inputs[i],
                "output": outputs[i],
                "score": scores[i],
                "gt": gts[i],
            }
            if len(selected) == len(target_difficulties):
                break

        return [selected[difficulty] for difficulty in target_difficulties if difficulty in selected]

    def _maybe_log_difficulty_samples_to_wandb(
        self,
        split: str,
        inputs,
        outputs,
        gts,
        scores,
        difficulties,
        uids=None,
    ):
        if "wandb" not in self.config.trainer.logger:
            return

        samples = self._select_difficulty_samples(
            inputs=inputs,
            outputs=outputs,
            gts=gts,
            scores=scores,
            difficulties=difficulties,
            uids=uids,
        )
        if not samples:
            return

        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        columns = ["step", "split", "difficulty", "uid", "input", "output", "score", "gt"]
        table_attr = f"_{split}_difficulty_generations_table"
        old_table = getattr(self, table_attr, None)
        table = wandb.Table(columns=columns, data=old_table.data if old_table is not None else [])

        for sample in samples:
            sample["split"] = split
            table.add_data(*(sample[column] for column in columns))

        setattr(self, table_attr, table)
        wandb.log({f"{split}/difficulty_generations": table}, step=self.global_steps)

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]
            difficulty_values = batch.non_tensor_batch.get("difficulty_bucket", ["unknown"] * len(batch))
            difficulty_values = (
                difficulty_values.tolist() if hasattr(difficulty_values, "tolist") else list(difficulty_values)
            )
            uid_values = batch.non_tensor_batch.get("uid", [""] * len(batch))
            uid_values = uid_values.tolist() if hasattr(uid_values, "tolist") else list(uid_values)

            reward_extra_infos_to_dump = {}
            for key, values in reward_extra_infos_dict.items():
                aligned_values = batch.non_tensor_batch.get(key, None)
                if aligned_values is not None and len(aligned_values) == len(batch):
                    reward_extra_infos_to_dump[key] = (
                        aligned_values.tolist() if hasattr(aligned_values, "tolist") else list(aligned_values)
                    )
                else:
                    reward_extra_infos_to_dump[key] = values
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )
            reward_extra_infos_to_dump["difficulty_bucket"] = difficulty_values
            reward_extra_infos_to_dump["uid"] = uid_values

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )
            self._maybe_log_difficulty_samples_to_wandb(
                split="train",
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                difficulties=difficulty_values,
                uids=uid_values,
            )

    @staticmethod
    def _validation_metric_n(metric_name: str) -> int | None:
        if "@" not in metric_name:
            return None
        return int(metric_name.split("@")[-1].split("/")[0])

    def _add_processed_validation_metrics(
        self,
        metric_dict: dict,
        prefix_parts: list[str],
        var2metric2val: dict,
    ):
        """Add core validation performance and response-length metrics for one group."""

        core_var = "acc" if "acc" in var2metric2val else "reward"
        core_metrics = var2metric2val.get(core_var, {})
        core_n_values = [self._validation_metric_n(name) for name in core_metrics]
        core_n_values = [value for value in core_n_values if value is not None]
        core_n_max = max(core_n_values) if core_n_values else None

        if core_n_max is not None:
            for metric_name in (f"mean@{core_n_max}", f"pass@{core_n_max}/mean"):
                if metric_name in core_metrics:
                    pfx = "/".join(["val-core", *prefix_parts, core_var, metric_name])
                    metric_dict[pfx] = core_metrics[metric_name]

        # Response length is always logged for every validation group:
        # total/all and difficulty/{easy, medium, hard}.
        token_metrics = var2metric2val.get("response_tokens", {})
        token_n_values = [self._validation_metric_n(name) for name in token_metrics]
        token_n_values = [value for value in token_n_values if value is not None]
        token_n_max = max(token_n_values) if token_n_values else None
        token_metric_name = f"mean@{token_n_max}" if token_n_max is not None else None
        if token_metric_name is not None and token_metric_name in token_metrics:
            pfx = "/".join(["val-core", *prefix_parts, "response_tokens", token_metric_name])
            metric_dict[pfx] = token_metrics[token_metric_name]

    def _add_validation_metric_group(
        self,
        metric_dict: dict,
        group_name: str,
        group_values: list[str],
        sample_uids: list[str],
        reward_extra_infos_dict: dict[str, list],
        valid_labels: set[str] | None = None,
    ):
        if valid_labels is None:
            keep_indices = list(range(len(group_values)))
        else:
            keep_indices = [idx for idx, value in enumerate(group_values) if value in valid_labels]
        if not keep_indices:
            return

        grouped_labels = np.array([group_values[idx] for idx in keep_indices], dtype=object)
        grouped_uids = [sample_uids[idx] for idx in keep_indices]
        grouped_infos = {}
        for key, values in reward_extra_infos_dict.items():
            if len(values) == len(group_values):
                grouped_infos[key] = [values[idx] for idx in keep_indices]

        group2var2metric2val = process_validation_metrics(grouped_labels, grouped_uids, grouped_infos)
        for label, var2metric2val in group2var2metric2val.items():
            self._add_processed_validation_metrics(
                metric_dict=metric_dict,
                prefix_parts=[group_name, label],
                var2metric2val=var2metric2val,
            )

    def _dump_validation_metrics(self, metric_dict: dict, dump_path: str):
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, "metrics.jsonl")
        entry = {
            "step": self.global_steps,
            "metrics": {
                key: float(value) if isinstance(value, np.generic) else value for key, value in metric_dict.items()
            },
        }
        with open(filename, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    # 从完整的batch中抽出rollout需要的batch
    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        # reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
        # [ADD] Scaf-GRPO: keep question, scaffold hints, and expert target in the training batch for intervention.
        # # rollout不能看到hint/expert信息，但是训练阶段必须能访问到
        reward_model_keys = (
            set(
                {
                    "data_source",
                    "reward_model",
                    "extra_info",
                    "uid",
                    "difficulty_bucket",
                    "question",
                    "knowledge_components_parts",
                    "planning_skeleton_parts",
                    "solution_breakdown_parts",
                    "expert_target",
                }
            )
            & batch.non_tensor_batch.keys()
        )

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        difficulty_bucket_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])
            prompt_len = test_batch.batch["prompts"].shape[-1]
            response_tokens = test_batch.batch["attention_mask"][:, prompt_len:].sum(-1).cpu().tolist()
            reward_extra_infos_dict["response_tokens"].extend(response_tokens)

            # evaluate using reward_function
            # Validation should be scored by val_reward_fn. Agent-loop rollout can
            # attach rm_scores from interaction rewards, which would otherwise
            # short-circuit _compute_or_extract_reward and skip math verification.
            test_batch.batch.pop("rm_scores", None)
            result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            difficulty_bucket_lst.append(
                test_batch.non_tensor_batch.get("difficulty_bucket", ["unknown"] * reward_tensor.shape[0])
            )

        difficulty_buckets = np.concatenate(difficulty_bucket_lst, axis=0).astype(str)
        reward_extra_infos_dict["difficulty_bucket"] = difficulty_buckets.tolist()
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        self._maybe_log_difficulty_samples_to_wandb(
            split="val",
            inputs=sample_inputs,
            outputs=sample_outputs,
            gts=sample_gts,
            scores=sample_scores,
            difficulties=difficulty_buckets.tolist(),
            uids=sample_uids,
        )

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        # Keep validation logging compact: overall performance and difficulty
        # breakdown only. Per-data-source and val-aux metrics are omitted.
        metric_dict = {}
        self._add_validation_metric_group(
            metric_dict=metric_dict,
            group_name="total",
            group_values=["all"] * len(sample_scores),
            sample_uids=sample_uids,
            reward_extra_infos_dict=reward_extra_infos_dict,
        )
        self._add_validation_metric_group(
            metric_dict=metric_dict,
            group_name="difficulty",
            group_values=difficulty_buckets.tolist(),
            sample_uids=sample_uids,
            reward_extra_infos_dict=reward_extra_infos_dict,
            valid_labels={"easy", "medium", "hard"},
        )

        if val_data_dir:
            self._dump_validation_metrics(metric_dict, val_data_dir)

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch) # 走这里
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    # [ADD] 给修改后的轨迹重新计算reward
    def _scaf_recompute_reward(self, batch: DataProto):
        # rm_scores是reward_model_scores, 要先删除替换之前的旧轨迹的reward
        # TODO: 所有旧轨迹包括未被替换的也删除重新计算
        if "rm_scores" in batch.batch.keys():
            batch.batch.pop("rm_scores")

        # 2. 如果启用了learned reward model，rm给新轨迹重新打分
        if self.use_rm: # false
            if not self.use_reward_loop:
                rm_scores = self.rm_wg.compute_rm_score(batch)
            else:
                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                rm_scores = self.reward_loop_manager.compute_rm_score(batch)

            batch = batch.union(rm_scores) # 把重新计算的奖励写回 batch
        # rule-based走这里
        return self._compute_or_extract_reward(batch, reward_fn=self.reward_fn, return_dict=False)


    # [ADD] hint替换和专家轨迹注入，输入第一轮rollout后的batch,reward，输出替换后的batch,reward
    # [ADD] 第二轮rollout，hint/expert替换，重算reward
    def _apply_scaf_grpo_interventions(
        self,
        batch: DataProto, # 第一轮rollout后完整batch，train_batch_size*rollout.n
        reward_tensor_first: torch.Tensor, # 第一轮rollout的reward，[B × rollout.n, max_response_length] 
        reward_extra_infos_dict_first: dict[str, Any], # 第一轮 reward 计算产生的额外信息如acc,pred,gt
        metrics: dict, # 当前step的日志dict，函数内会添加hint替换的指标
    ):
        hint_enabled = self.with_hint and self.global_steps > self.warmup_steps
        expert_enabled = self.with_expert_fallback
        hint_is_enabled = self.config.algorithm.get("hint_is_correction", False)
        if not hint_enabled and not expert_enabled:
            return batch, reward_tensor_first, reward_extra_infos_dict_first, False
        if self.config.reward_model.launch_reward_fn_async:
            raise ValueError("Scaf-GRPO intervention requires synchronous reward computation.")

        response_mask = batch.batch["response_mask"]
        for mask_key in ("sft_loss_mask", "hint_sft_loss_mask", "off_policy_mask"):
            if mask_key not in batch.batch:
                batch.batch[mask_key] = torch.zeros_like(response_mask) # shape相同但是全零，后面发生替换后再置1

        # token-level reward聚合成traj-level，reward_tensor_first: [B × rollout.n, max_response_length],对最后一维求和
        reward_sums_tensor = reward_tensor_first.sum(dim=-1).float()
        reward_sums = reward_sums_tensor.detach().cpu().numpy()
        reward_before_mean = reward_sums_tensor.mean().item()

        uid_to_indices: dict[Any, list[int]] = defaultdict(list) # 根据uid把一道题的rollout分组
        for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
            uid_to_indices[uid].append(idx)  # 当前题的N个轨迹下标

        # 1. 找出全错的题目
        failed_uids = {
            uid
            for uid, indices in uid_to_indices.items()
            if len(indices) > 0 and np.all(reward_sums[indices] <= 0)
        }

        # Hint effectiveness is measured at UID/group level:
        # denominator = questions whose first rollout group has pass@k == 0;
        # numerator = those questions for which at least one hinted rollout is correct.
        scaf_metrics = {
            "batch/scaf_initial_failed_uid_count": len(failed_uids),
            "batch/scaf_hint_rescued_uid_count": 0,
            "batch/scaf_hint_rescue_rate": 0.0,
            "batch/scaf_expert_injected_count": 0,
            "batch/scaf_expert_solved_count": 0,
            "batch/scaf_expert_success_rate": 0.0,
            "batch/scaf_reward_before_mean": reward_before_mean,
            "batch/scaf_reward_after_mean": reward_before_mean,
            "batch/scaf_reward_gain": 0.0,
        }
        # Mutually exclusive counts: each rescued UID is assigned to the earliest
        # successful hint stage, matching the actual replacement rule below.
        for stage in range(1, self.hint_stage_count + 1):
            scaf_metrics[f"batch/scaf_selected_hint_stage_{stage}_uid_count"] = 0
        if not failed_uids:
            metrics.update(scaf_metrics)
            return batch, reward_tensor_first, reward_extra_infos_dict_first, False

        # 每道失败题目只保留一条基础样本，提供所在uid的hint去构造hint prompt, len=train_batch_size=64
        base_indices = [indices[0] for uid, indices in uid_to_indices.items() if uid in failed_uids] 
        failed_base_batch = batch.select_idxs(base_indices)

        hint_data_map = {} # 保存 Hint 成功的题目, (hinted_output, best_idx, hint_stage,)
        fully_failed_uids = set(failed_uids) # 存储hint引导失败的题目，去做轨迹注入

        if hint_enabled and "question" in failed_base_batch.non_tensor_batch:
            # 2. 构造分层hint prompt
            hinted_gen_batch = build_hinted_gen_batch(failed_base_batch, stage_count=self.hint_stage_count) # train_batch_size * 12(如果是三层Hint)

            if len(hinted_gen_batch) > 0:
                hinted_gen_batch.meta_info["global_steps"] = self.global_steps

                # hint batch 需要padding，因为Hint candidate 数量不一定能整除 AgentLoop worker 数量，先复制补齐之后会删除复制的
                agent_worker_count = int(self.config.actor_rollout_ref.rollout.get("agent", {}).get("num_workers", 1)) # 8，为什么是8 
                hinted_gen_batch_padded, hint_pad_size = pad_dataproto_to_divisor(
                    hinted_gen_batch, max(agent_worker_count, 1)
                )
                # 3. 在hint引导下rollout
                hinted_output_padded = (
                    self.actor_rollout_wg.generate_sequences(hinted_gen_batch_padded)
                    if not self.async_rollout_mode
                    else self.async_rollout_manager.generate_sequences(hinted_gen_batch_padded)
                )
                hinted_output = unpad_dataproto(hinted_output_padded, pad_size=hint_pad_size) # 删除padding的复制样本

                # 某些 rollout 后端返回输出时，可能会丢失一些字段这里补回
                for key, value in hinted_gen_batch.non_tensor_batch.items():
                    if key not in hinted_output.non_tensor_batch:
                        hinted_output.non_tensor_batch[key] = value
                if "response_mask" not in hinted_output.batch.keys():
                    hinted_output.batch["response_mask"] = compute_response_mask(hinted_output)

                # 3.给所有 Hint 引导生成的新轨迹重新打分
                reward_tensor_hint, _ = self._scaf_recompute_reward(hinted_output)
                hint_reward_sums = reward_tensor_hint.sum(-1).detach().cpu().numpy()
                hint_uids = hinted_output.non_tensor_batch["uid"]
                hint_levels = hinted_output.non_tensor_batch["hint_level"]
                # 选择能做对题目的最早hint阶段
                for uid in failed_uids:
                    candidate_indices = [i for i, hint_uid in enumerate(hint_uids) if hint_uid == uid]
                    solved_indices = [i for i in candidate_indices if hint_reward_sums[i] > 0] # 只有做对的才会进入solved_indices
                    if not solved_indices:
                        continue
                    best_idx = min(solved_indices, key=lambda i: int(hint_levels[i]))
                    hint_data_map[uid] = (hinted_output, best_idx, int(hint_levels[best_idx]))
                    fully_failed_uids.discard(uid) 

                selected_hint_stages = [hint_stage for _, _, hint_stage in hint_data_map.values()]
                rescued_uid_count = len(hint_data_map)
                scaf_metrics["batch/scaf_hint_rescued_uid_count"] = rescued_uid_count
                scaf_metrics["batch/scaf_hint_rescue_rate"] = rescued_uid_count / max(len(failed_uids), 1)
                for stage in range(1, self.hint_stage_count + 1):
                    scaf_metrics[f"batch/scaf_selected_hint_stage_{stage}_uid_count"] = (
                        selected_hint_stages.count(stage)
                    )

                # Keep the behavior probability under x+hint before response-only
                # replacement rebuilds the sample with the original prompt x.
                if hint_data_map and not self.replace_hint_prompt_response and hint_is_enabled:
                    if "rollout_log_probs" in hinted_output.batch:
                        hint_behavior_log_probs = hinted_output.batch["rollout_log_probs"].detach().float()
                    else:
                        hinted_old_log_prob_padded, _ = self._compute_old_log_prob(hinted_output_padded)
                        hinted_old_log_prob = unpad_dataproto(
                            hinted_old_log_prob_padded, pad_size=hint_pad_size
                        )
                        hint_behavior_log_probs = hinted_old_log_prob.batch["old_log_probs"].detach().float()
                    if hint_behavior_log_probs.shape != hinted_output.batch["responses"].shape:
                        raise ValueError(
                            "Hint behavior log-probs must align with hinted responses; "
                            f"got {hint_behavior_log_probs.shape} and {hinted_output.batch['responses'].shape}."
                        )
                    hinted_output.batch["hint_behavior_log_probs"] = hint_behavior_log_probs
                    batch.batch["hint_behavior_log_probs"] = torch.zeros_like(response_mask, dtype=torch.float32)
                    batch.batch["hint_offpolicy_mask"] = torch.zeros(
                        len(batch), dtype=torch.bool, device=response_mask.device
                    )

        # hint失败的构造专家轨迹
        expert_data_map = {}
        if expert_enabled and "expert_target" in failed_base_batch.non_tensor_batch:
            expert_truncation = self.config.trainer.get("expert_truncation", "right") 
            max_response_length = self.config.data.max_response_length
            for row_idx, uid in enumerate(failed_base_batch.non_tensor_batch["uid"]):
                if uid not in fully_failed_uids:
                    continue
                expert_data = build_expert_response_data(
                    failed_base_batch,
                    row_idx,
                    self.tokenizer,
                    max_response_length=max_response_length,
                    truncation=expert_truncation,
                )
                if expert_data is not None:
                    expert_data_map[uid] = expert_data

        # 把hint引导轨迹/专家轨迹写回原batch
        replaced_indices = [] # 所有发生修改的 batch 行号（Hint + Expert）
        expert_replaced_indices = [] # 只记录实际注入专家轨迹的 batch 行号
        per_uid_replaced = defaultdict(int) # 每个问题已经替换了多少条 rollout

        # Tensor keys needed when keeping the full hinted trajectory.
        # replace_hint_prompt_response=True replaces prompt and response;
        # False keeps the original prompt and swaps only the response.
        tensor_keys = ["prompts", "responses", "input_ids", "attention_mask", "position_ids", "response_mask"]

        for row_idx, uid in enumerate(batch.non_tensor_batch["uid"]): # 遍历整个batch
            # 每个问题最多被替换 replace_num 条
            if per_uid_replaced[uid] >= self.replace_num:
                continue

            # 替换Hint，有两种替换模式
            if uid in hint_data_map:
                hinted_output, hint_idx, _ = hint_data_map[uid]
                if self.replace_hint_prompt_response: # True, 替换prompt+response
                    for key in tensor_keys:
                        batch.batch[key][row_idx] = hinted_output.batch[key][hint_idx].to(batch.batch[key].device)
                else: # False, 只替换response
                    response = hinted_output.batch["responses"][hint_idx].to(batch.batch["responses"].device)
                    response_mask_new = hinted_output.batch["response_mask"][hint_idx].to(
                        batch.batch["response_mask"].device
                    )
                    prompt_len = batch.batch["prompts"].shape[-1]
                    prompt_mask = batch.batch["attention_mask"][row_idx][:prompt_len]
                    batch.batch["responses"][row_idx] = response
                    batch.batch["input_ids"][row_idx] = torch.cat([batch.batch["prompts"][row_idx], response], dim=0)
                    batch.batch["attention_mask"][row_idx] = torch.cat([prompt_mask, response_mask_new], dim=0)
                    batch.batch["position_ids"][row_idx] = torch.clip(
                        torch.cumsum(batch.batch["attention_mask"][row_idx], dim=-1) - 1,
                        min=0,
                        max=None,
                    )
                    batch.batch["response_mask"][row_idx] = response_mask_new
                    if hint_is_enabled:
                        batch.batch["hint_behavior_log_probs"][row_idx] = hinted_output.batch[
                            "hint_behavior_log_probs"
                        ][hint_idx].to(batch.batch["hint_behavior_log_probs"].device)
                        batch.batch["hint_offpolicy_mask"][row_idx] = True

                batch.batch["hint_sft_loss_mask"][row_idx] = batch.batch["response_mask"][row_idx]
                batch.batch["sft_loss_mask"][row_idx].zero_() # 这两个是专家轨迹的处理
                batch.batch["off_policy_mask"][row_idx].zero_()
                replaced_indices.append(row_idx)
                per_uid_replaced[uid] += 1

            # 替换专家轨迹
            elif uid in expert_data_map:
                expert_data = expert_data_map[uid]

                # 标识其它mask，expert_data包含：responses,input_ids,attention_mask,position_ids,response_mask,sft_loss_mask,off_policy_mask
                for key, value in expert_data.items(): 
                    batch.batch[key][row_idx] = value.to(batch.batch[key].device)

                batch.batch["hint_sft_loss_mask"][row_idx].zero_() # 清除掉hint轨迹的处理
                if "hint_behavior_log_probs" in batch.batch:
                    batch.batch["hint_behavior_log_probs"][row_idx].zero_()
                    batch.batch["hint_offpolicy_mask"][row_idx] = False
                replaced_indices.append(row_idx)
                expert_replaced_indices.append(row_idx)
                per_uid_replaced[uid] += 1

        metrics.update(scaf_metrics)
        if not replaced_indices:
            return batch, reward_tensor_first, reward_extra_infos_dict_first, False

        # 重新计算修改后的batch的reward
        reward_tensor_recomputed, reward_extra_infos_dict_recomputed = self._scaf_recompute_reward(batch)
        replaced_indices = sorted(set(replaced_indices))
        expert_replaced_indices = sorted(set(expert_replaced_indices))

        # 统计实际注入专家轨迹的数量，以及这些专家轨迹通过 reward verifier 的数量。
        expert_injected_count = len(expert_replaced_indices)
        if expert_injected_count > 0:
            expert_reward_after = (
                reward_tensor_recomputed[expert_replaced_indices]
                .sum(dim=-1)
                .float()
            )
            expert_solved_count = int((expert_reward_after > 0).sum().item())
            expert_success_rate = expert_solved_count / expert_injected_count
        else:
            expert_solved_count = 0
            expert_success_rate = 0.0

        metrics["batch/scaf_expert_injected_count"] = expert_injected_count
        metrics["batch/scaf_expert_solved_count"] = expert_solved_count
        metrics["batch/scaf_expert_success_rate"] = expert_success_rate

        reward_tensor_final = reward_tensor_first.clone() # 未被替换样本保留第一次reward
        reward_tensor_final[replaced_indices] = reward_tensor_recomputed[replaced_indices]

        # 统计整个 batch 的 reward 变化，而不是只统计必然为 0 的被替换失败轨迹。
        reward_after_mean = reward_tensor_final.sum(dim=-1).float().mean().item()
        metrics["batch/scaf_reward_before_mean"] = reward_before_mean
        metrics["batch/scaf_reward_after_mean"] = reward_after_mean
        metrics["batch/scaf_reward_gain"] = reward_after_mean - reward_before_mean


        reward_extra_infos_dict = deepcopy(reward_extra_infos_dict_first)
        for key, recomputed_values in reward_extra_infos_dict_recomputed.items():
            if key not in reward_extra_infos_dict:
                reward_extra_infos_dict[key] = deepcopy(recomputed_values)
                continue
            for row_idx in replaced_indices:
                reward_extra_infos_dict[key][row_idx] = recomputed_values[row_idx]
        return batch, reward_tensor_final, reward_extra_infos_dict, True

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        if self.config.trainer.get("val_only", False) and self.config.trainer.get("val_only_step", None) is not None:
            self.global_steps = int(self.config.trainer.val_only_step)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # 训练前先验证
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            # 一次取 train_batch_size 64条数据
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                # 1. 构建训练数据
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                # 为每个原始题目创建唯一uid
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                # 去掉rollout不能看到的hint/expert信息
                gen_batch = self._get_gen_batch(batch) 
                gen_batch.meta_info["global_steps"] = self.global_steps
                # 每个问题复制8次，B->B*N
                gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True) # interleave=True让相同uid的N个副本相邻


                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # 2. 第一轮rollout
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else: # 第一轮rollout，走这里
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output) # driver进程调用入口

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # REMAX 的额外 baseline 分支，不太理解，但grpo不走这里
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    
                    # repeat to align with repeated responses in rollout，把原始题目复制rollout.n份，和生成结果合并
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output) 

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # [DEL] Original verl balances immediately after rollout. Scaf-GRPO mutates response lengths after
                    # reward-based hint/expert replacement, so balancing is moved after final token_level_scores is set.
                    # if self.config.trainer.balance_batch:
                    #     self._balance_batch(batch, metrics=metrics)
                    #
                    # batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # 3. 第一轮计算reward
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # For rule-based math reward, score on the trainer side so training matches validation.
                        # Agent-loop rm_scores can be present when reward_model.use_reward_loop=True, but those
                        # scores have diverged from the trainer math verifier in this setup.
                        if not self.use_rm and "rm_scores" in batch.batch.keys():
                            batch.batch.pop("rm_scores", None)

                        # 是否使用reward model
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )

                    # [ADD] Scaf-GRPO needs first-pass rewards before old_log_prob. Pull async reward here so the subsequent final-reward/balance path has a single source of truth.
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward) # 异步调用，前面只是发起ray任务，这里拿到reward

                    # [ADD] Scaf-GRPO: replace initially failed groups with hinted rollout or expert trajectory before old_log_prob / ref_log_prob / values are computed.
                    scaf_interventions_applied = False # hint干预状态，当前batch是否有轨迹被替换了，确实改写了batch.batch["responses"]才会返回True
                    if self.with_hint or self.with_expert_fallback:
                        batch, reward_tensor, scaf_reward_extra_infos_dict, scaf_interventions_applied = (
                            self._apply_scaf_grpo_interventions( 
                                batch, reward_tensor, reward_extra_infos_dict, metrics
                            )
                        )
                        if scaf_reward_extra_infos_dict:
                            reward_extra_infos_dict = scaf_reward_extra_infos_dict

                    if scaf_interventions_applied:
                        # 删掉旧轨迹的
                        batch.batch.pop("rollout_log_probs", None)
                        with open_dict(self.config.actor_rollout_ref.actor.policy_loss):
                            self.config.actor_rollout_ref.actor.policy_loss["loss_mode"] = (
                                self._scaf_default_policy_loss_mode
                            )
                        metrics["batch/scaf_disabled_rollout_bypass"] = 1

                    # Store final rewards before balancing so DataProto.reorder keeps rewards aligned.
                    batch.batch["token_level_scores"] = reward_tensor
                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # Balance after Scaf-GRPO replacement, when trajectory tensors and rewards are final.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens after the final optional reorder. 记录每条轨迹的有效 token 数
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    # TODO: bypass模式是什么意思
                    if bypass_recomputing_logprobs and not scaf_interventions_applied:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # 重新计算 old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    hint_is_enabled = self.config.algorithm.get("hint_is_correction", True)
                    hint_offpolicy_mask = batch.batch.get("hint_offpolicy_mask", None)
                    if (
                        hint_is_enabled
                        and hint_offpolicy_mask is not None
                        and torch.any(hint_offpolicy_mask.to(torch.bool))
                    ):
                        if "hint_behavior_log_probs" not in batch.batch:
                            raise ValueError(
                                "Response-only Hint IS correction requires hint_behavior_log_probs."
                            )
                        hint_is_weights, hint_is_metrics = compute_hint_is_weights(
                            old_log_probs=batch.batch["old_log_probs"],
                            hint_behavior_log_probs=batch.batch["hint_behavior_log_probs"],
                            response_mask=batch.batch["response_mask"],
                            hint_offpolicy_mask=hint_offpolicy_mask,
                            log_c_clip=float(self.config.algorithm.get("hint_log_c_clip", 5.0)),
                        )
                        batch.batch["hint_is_weights"] = hint_is_weights
                        metrics.update(hint_is_metrics)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                
                        # token_level_scores has already been set to first-pass or Scaf-recomputed final rewards.
                        assert "token_level_scores" in batch.batch, "Scaf-GRPO reward path must set token_level_scores"

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                data_metrics = compute_data_metrics(batch=batch, use_critic=self.use_critic)
                metrics.update(data_metrics)
                metrics["train/response_tokens/mean"] = data_metrics["response_length/mean"]
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
