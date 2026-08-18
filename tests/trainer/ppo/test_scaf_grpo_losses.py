import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_token_on_off_policy_loss
from verl.trainer.ppo.scaf_grpo_utils import (
    build_expert_response_data,
    build_hinted_gen_batch,
    normalize_expert_target,
    select_expert_injection_uids,
)
from verl.workers.utils.losses import compute_scaf_ppo_policy_loss, compute_scaf_source_policy_losses


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
            "off_policy_loss_type": "probability",
            "off_policy_reshape": "p_div_p_0.1",
            "off_policy_max_clip": -1.0,
            "off_policy_min_clip": -1.0,
            "sft_loss_coef": 0.0,
            "use_hint_sft_loss": False,
            "hint_sft_loss_coef": 0.0,
        }
    )


def test_expert_injection_defaults_to_fully_failed_uids():
    selected = select_expert_injection_uids(
        all_uids=["failed", "passed", "hint_rescued"],
        fully_failed_uids={"failed"},
    )

    assert selected == {"failed"}


def test_expert_injection_can_ignore_pass_at_k():
    selected = select_expert_injection_uids(
        all_uids=["failed", "passed", "hint_rescued"],
        fully_failed_uids={"failed"},
        inject_for_all_uids=True,
    )

    assert selected == {"failed", "passed", "hint_rescued"}


def test_mixed_loss_uses_luffy_on_policy_clipping_without_dapo_lower_cap():
    old_log_prob = torch.zeros(1, 2)
    log_prob = torch.tensor([[torch.log(torch.tensor(10.0)), -1.0]])
    advantages = torch.tensor([[-1.0, 1.0]])
    response_mask = torch.ones(1, 2, dtype=torch.bool)
    off_policy_mask = torch.tensor([[False, True]])
    config = _actor_config()

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
    )

    # LUFFY uses max(unclipped, clipped) and does not apply verl/DAPO's
    # additional negative-advantage cap at clip_ratio_c.
    assert mixed["on_pg_loss"] == pytest.approx(10.0)
    assert mixed["on_pg_clipfrac"] == pytest.approx(0.0)
    assert mixed["on_pg_clipfrac_lower"] == pytest.approx(0.0)


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


def test_expert_advantage_weighted_log_prob_has_nonvanishing_gradient():
    log_prob = torch.tensor([[-20.0, -2.0]], requires_grad=True)
    config = _actor_config()
    result = compute_token_on_off_policy_loss(
        old_log_prob=log_prob.detach(),
        log_prob=log_prob,
        advantages=torch.ones_like(log_prob),
        response_mask=torch.ones_like(log_prob, dtype=torch.bool),
        off_policy_mask=torch.ones_like(log_prob, dtype=torch.bool),
        cliprange=config.clip_ratio,
        off_policy_loss_type="advantage_weighted_log_prob",
    )

    grad = torch.autograd.grad(result["pg_loss"], log_prob)[0]
    assert result["pg_loss"].item() == pytest.approx(11.0)
    torch.testing.assert_close(grad, torch.full_like(grad, -0.5))
    assert result["off_ratio_mean"] == pytest.approx(1.0)


def test_fixed_length_loss_matches_luffy_micro_batch_scaling():
    config = _actor_config()

    def compute(batch_size):
        log_prob = torch.full((batch_size, 2), -2.0)
        return compute_token_on_off_policy_loss(
            old_log_prob=log_prob,
            log_prob=log_prob,
            advantages=torch.ones_like(log_prob),
            response_mask=torch.ones_like(log_prob, dtype=torch.bool),
            off_policy_mask=torch.ones_like(log_prob, dtype=torch.bool),
            cliprange=config.clip_ratio,
            off_policy_loss_type="advantage_weighted_log_prob",
            loss_remove_token_mean=True,
        )["pg_loss"]

    assert compute(1).item() == pytest.approx(2.0)
    assert compute(2).item() == pytest.approx(4.0)


def test_probability_objective_matches_luffy_p_div_p_point_one():
    log_prob = torch.log(torch.tensor([[0.2, 0.4]]))
    advantages = torch.tensor([[2.0, -3.0]])
    response_mask = torch.ones_like(log_prob, dtype=torch.bool)
    off_policy_mask = torch.tensor([[True, False]])

    result = compute_token_on_off_policy_loss(
        old_log_prob=log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        off_policy_mask=off_policy_mask,
        cliprange=0.2,
        off_policy_reshape="p_div_p_0.1",
        loss_remove_clip=True,
    )

    expected_off_loss = -2.0 * (0.2 / (0.2 + 0.1))
    expected_on_loss = 3.0
    assert result["off_pg_loss"] == pytest.approx(expected_off_loss)
    assert result["on_pg_loss"] == pytest.approx(expected_on_loss)
    assert result["pg_loss"].item() == pytest.approx((expected_off_loss + expected_on_loss) / 2)


@pytest.mark.parametrize(
    ("off_policy_reshape", "expected_grad"),
    [
        ("offratio_detach", [[-0.15, -0.225]]),
        ("prob_detach", [[-0.15, -0.225]]),
        ("onratio", [[0.0, 0.0]]),
    ],
)
def test_off_policy_loss_variants_match_source_gradient_diagnostics(off_policy_reshape, expected_grad):
    config = _actor_config()
    config.off_policy_reshape = off_policy_reshape
    config.off_policy_max_clip = 0.15
    old_log_prob = torch.log(torch.tensor([[0.1, 0.2]]))
    log_prob = torch.log(torch.tensor([[0.2, 0.4]])).requires_grad_()
    advantages = torch.tensor([[2.0, 3.0]])
    response_mask = torch.ones_like(log_prob, dtype=torch.bool)
    off_policy_mask = torch.ones_like(log_prob, dtype=torch.bool)

    result = compute_token_on_off_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        off_policy_mask=off_policy_mask,
        cliprange=config.clip_ratio,
        off_policy_reshape=off_policy_reshape,
        off_policy_max_clip=config.off_policy_max_clip,
    )
    source_loss = compute_scaf_source_policy_losses(
        config=config,
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        off_policy_mask=off_policy_mask,
    )["expert"]

    torch.testing.assert_close(result["pg_loss"], source_loss)
    main_grad = torch.autograd.grad(result["pg_loss"], log_prob, retain_graph=True)[0]
    source_grad = torch.autograd.grad(source_loss, log_prob)[0]
    torch.testing.assert_close(main_grad, source_grad)
    torch.testing.assert_close(main_grad, torch.tensor(expected_grad))


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
