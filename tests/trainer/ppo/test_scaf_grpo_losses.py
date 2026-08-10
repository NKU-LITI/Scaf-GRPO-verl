import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla, compute_token_on_off_policy_loss
from verl.trainer.ppo.scaf_grpo_utils import (
    build_expert_response_data,
    build_hinted_gen_batch,
    normalize_expert_target,
)
from verl.workers.utils.losses import compute_scaf_ppo_policy_loss


def _actor_config(loss_mode: str = "vanilla"):
    return OmegaConf.create(
        {
            "loss_agg_mode": "token-mean",
            "policy_loss": {"loss_mode": loss_mode},
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0,
            "clip_upper_bound": 1.0,
            "global_batch_info": {},
            "use_off_policy_loss": True,
            "off_policy_reshape": "p_div_p_0.1",
            "off_policy_max_clip": -1.0,
            "off_policy_min_clip": -1.0,
            "sft_loss_coef": 0.0,
            "use_hint_sft_loss": False,
            "hint_sft_loss_coef": 0.0,
        }
    )


def test_mixed_loss_keeps_on_policy_tokens_identical_to_vanilla_ppo():
    old_log_prob = torch.zeros(1, 2)
    log_prob = torch.tensor([[torch.log(torch.tensor(10.0)), -1.0]])
    advantages = torch.tensor([[-1.0, 1.0]])
    response_mask = torch.ones(1, 2, dtype=torch.bool)
    off_policy_mask = torch.tensor([[False, True]])
    rollout_is_weights = torch.tensor([[0.5, 7.0]])
    config = _actor_config()

    expected_on_loss, _ = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask & ~off_policy_mask,
        loss_agg_mode="token-mean",
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
    mixed = compute_token_on_off_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        off_policy_mask=off_policy_mask,
        cliprange=config.clip_ratio,
        cliprange_low=config.clip_ratio_low,
        cliprange_high=config.clip_ratio_high,
        clip_ratio_c=config.clip_ratio_c,
        clip_upper_bound=config.clip_upper_bound,
        loss_agg_mode="token-mean",
        rollout_is_weights=rollout_is_weights,
    )

    assert mixed["on_pg_loss"] == pytest.approx(expected_on_loss.item())
    assert mixed["on_pg_clipfrac_lower"] == pytest.approx(1.0)


def test_expert_mixed_loss_rejects_non_vanilla_policy_mode():
    config = _actor_config(loss_mode="gspo")
    values = torch.zeros(1, 1)

    with pytest.raises(ValueError, match="supports policy_loss.loss_mode='vanilla' only"):
        compute_scaf_ppo_policy_loss(
            config=config,
            old_log_prob=values,
            log_prob=values,
            advantages=values,
            response_mask=torch.ones_like(values, dtype=torch.bool),
            off_policy_mask=torch.ones_like(values, dtype=torch.bool),
        )


def test_expert_sft_loss_only_uses_marked_response_tokens():
    config = _actor_config()
    config.use_off_policy_loss = False
    config.sft_loss_coef = 2.0
    log_prob = torch.tensor([[-1.0, -3.0, -9.0]])
    zero = torch.zeros_like(log_prob)
    response_mask = torch.tensor([[True, True, False]])
    sft_loss_mask = torch.tensor([[False, True, False]])

    policy_loss, pg_loss, metrics = compute_scaf_ppo_policy_loss(
        config=config,
        old_log_prob=log_prob,
        log_prob=log_prob,
        advantages=zero,
        response_mask=response_mask,
        sft_loss_mask=sft_loss_mask,
    )

    assert pg_loss.item() == pytest.approx(0.0)
    assert policy_loss.item() == pytest.approx(6.0)
    assert metrics["actor/sft_loss"] == pytest.approx(3.0)
    assert metrics["actor/sft_loss_tokens"] == 1


class _Tokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [len(token) for token in text.split()]


def test_expert_response_preserves_prompt_and_marks_only_valid_response_tokens():
    base_batch = DataProto.from_single_dict(
        {
            "prompts": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0, 0]]),
            "expert_target": np.asarray([["first part", "second"]], dtype=object),
        }
    )

    expert = build_expert_response_data(base_batch, 0, _Tokenizer(), max_response_length=4)

    assert normalize_expert_target(["first part", "second"]) == "first part\n\nsecond"
    assert torch.equal(expert["input_ids"][:3], base_batch.batch["prompts"][0])
    assert expert["responses"].tolist() == [5, 4, 6, 99]
    assert torch.equal(expert["sft_loss_mask"], expert["response_mask"])
    assert torch.equal(expert["off_policy_mask"], expert["response_mask"])


def test_hinted_batch_builds_cumulative_prefixes_in_stage_order():
    base_batch = DataProto.from_single_dict(
        {
            "dummy": torch.zeros(1, 1),
            "question": np.asarray(["Q"], dtype=object),
            "uid": np.asarray(["u1"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "A"}], dtype=object),
            "data_source": np.asarray(["math"], dtype=object),
            "knowledge_components_parts": np.asarray([["k1", "k2"]], dtype=object),
            "planning_skeleton_parts": np.asarray([["p1"]], dtype=object),
        }
    )

    hinted = build_hinted_gen_batch(base_batch, stage_count=2)
    prompts = hinted.non_tensor_batch["raw_prompt"]

    assert len(hinted) == 3
    assert hinted.non_tensor_batch["hint_level"].tolist() == [1, 1, 2]
    assert prompts[0][-1]["content"].endswith("Knowledge Hints: k1")
    assert prompts[1][-1]["content"].endswith("Knowledge Hints: k1 k2")
    assert prompts[2][-1]["content"].endswith("Planning Hints: p1")
