# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_token_on_off_policy_loss,
    compute_value_loss,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig


def _scaf_zero_like_loss(log_prob: torch.Tensor) -> torch.Tensor:
    return log_prob.sum() * 0.0


def _scaf_bool_mask(mask: torch.Tensor | None, response_mask: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.zeros_like(response_mask, dtype=torch.bool)
    return mask.to(device=response_mask.device, dtype=torch.bool) & response_mask

# [ADD] hint loss
def compute_scaf_source_policy_losses(
    config,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    sft_loss_mask: torch.Tensor | None = None,
    hint_sft_loss_mask: torch.Tensor | None = None,
    off_policy_mask: torch.Tensor | None = None,
    rollout_is_weights: torch.Tensor | None = None,
    hint_is_weights: torch.Tensor | None = None,
):
    """Return Scaf policy-loss pieces by trajectory source for gradient diagnostics."""
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    if loss_mode != "vanilla":
        zero = _scaf_zero_like_loss(log_prob)
        return {
            "rollout": zero,
            "expert": zero,
            "hint": zero,
        }

    response_mask = response_mask.to(bool)
    expert_mask = _scaf_bool_mask(off_policy_mask, response_mask) | _scaf_bool_mask(sft_loss_mask, response_mask)
    hint_mask = _scaf_bool_mask(hint_sft_loss_mask, response_mask) & (~expert_mask)
    rollout_mask = response_mask & (~expert_mask) & (~hint_mask)

    if hint_is_weights is not None:
        advantages = advantages * hint_is_weights.detach()

    # Keep diagnostic gradients consistent with the LUFFY-compatible main loss.
    negative_approx_kl = log_prob - old_log_prob
    on_policy_reshape = config.get("on_policy_reshape", "no_reshape")
    if on_policy_reshape == "no_reshape":
        ratio = torch.exp(negative_approx_kl)
    elif on_policy_reshape == "logp":
        ratio = negative_approx_kl
    elif on_policy_reshape == "p_logp":
        ratio = torch.exp(negative_approx_kl) + config.get("on_policy_reshape_weight", 1.0) * negative_approx_kl
    elif on_policy_reshape == "square_root":
        ratio = torch.sqrt(torch.exp(negative_approx_kl))
    elif on_policy_reshape == "pow":
        ratio = torch.pow(torch.exp(negative_approx_kl), config.get("on_policy_reshape_pow_exp", 0.5))
    elif on_policy_reshape in ("p_div_p_0.1", "p_div_p_0.5"):
        offset = 0.1 if on_policy_reshape == "p_div_p_0.1" else 0.5
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        ratio = (prob / (prob + offset)) / (old_prob / (old_prob + offset))
    else:
        raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

    use_mixed_loss = config.get("use_off_policy_loss", False) and off_policy_mask is not None
    if use_mixed_loss:
        cliprange_low = config.clip_ratio if config.clip_ratio_low is None else config.clip_ratio_low
        cliprange_high = config.clip_ratio if config.clip_ratio_high is None else config.clip_ratio_high
        clip_ratio_c = config.get("clip_ratio_c", 3.0)
        clip_upper_bound = config.get("clip_upper_bound", 1.0)

        on_pg_losses = -advantages * ratio
        if not config.get("loss_remove_clip", False):
            on_pg_losses2 = -advantages * torch.clamp(
                ratio,
                1 - cliprange_low,
                max(1 + cliprange_high, clip_upper_bound),
            )
            on_pg_losses_clipped = torch.maximum(on_pg_losses, on_pg_losses2)
            on_pg_losses3 = -advantages * clip_ratio_c
            on_pg_losses = torch.where(
                advantages < 0,
                torch.minimum(on_pg_losses3, on_pg_losses_clipped),
                on_pg_losses_clipped,
            )
            if rollout_is_weights is not None:
                on_pg_losses = on_pg_losses * rollout_is_weights

        actual_off_policy_mask = _scaf_bool_mask(off_policy_mask, response_mask)
        off_policy_loss_type = config.get("off_policy_loss_type", "probability")
        if off_policy_loss_type == "advantage_weighted_log_prob":
            off_pg_losses = -advantages * log_prob
        elif off_policy_loss_type == "probability":
            prob = torch.exp(log_prob)
            off_policy_reshape = config.get("off_policy_reshape", "p_div_p_0.1")
            if off_policy_reshape == "p_div_p_0.1":
                off_ratio = prob / (prob + 0.1)
            elif off_policy_reshape == "p_div_p_0.3":
                off_ratio = prob / (prob + 0.3)
            elif off_policy_reshape == "p_div_p_0.5":
                off_ratio = prob / (prob + 0.5)
            elif off_policy_reshape == "offratio_detach":
                off_ratio = prob / (prob + 0.1)
            elif off_policy_reshape == "prob_detach":
                off_ratio = prob
            elif off_policy_reshape == "onratio":
                off_ratio = ratio
            elif off_policy_reshape in ("p", "no_reshape"):
                off_ratio = prob
            elif off_policy_reshape == "logp":
                off_ratio = log_prob * config.get("off_policy_reshape_weight", 1.0)
            elif off_policy_reshape == "p_logp":
                off_ratio = prob + log_prob * config.get("off_policy_reshape_weight", 1.0)
            elif off_policy_reshape == "square_root":
                off_ratio = torch.sqrt(prob)
            elif off_policy_reshape == "pow":
                off_ratio = torch.pow(prob, config.get("off_policy_reshape_pow_exp", 0.5))
            else:
                raise ValueError(f"Unsupported off_policy_reshape: {off_policy_reshape}")

            off_policy_max_clip = config.get("off_policy_max_clip", -1.0)
            off_policy_min_clip = config.get("off_policy_min_clip", -1.0)
            if off_policy_max_clip >= 0:
                off_ratio = torch.clamp(off_ratio, max=off_policy_max_clip)
            if off_policy_min_clip >= 0:
                off_ratio = torch.clamp(off_ratio, min=off_policy_min_clip)
            if off_policy_reshape == "offratio_detach":
                off_pg_losses = -(advantages * off_ratio).detach() * log_prob
            elif off_policy_reshape == "prob_detach":
                off_pg_losses = -(advantages * off_ratio).detach() * log_prob
            else:
                off_pg_losses = -advantages * off_ratio
        else:
            raise ValueError(
                "off_policy_loss_type must be 'probability' or 'advantage_weighted_log_prob'; "
                f"got {off_policy_loss_type!r}."
            )
        pg_losses = torch.where(actual_off_policy_mask, off_pg_losses, on_pg_losses)
    else:
        clip_ratio = config.clip_ratio
        clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
        clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
        clip_ratio_c = config.get("clip_ratio_c", 3.0)

        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        if rollout_is_weights is not None:
            pg_losses = pg_losses * rollout_is_weights

    def aggregate_source(mask: torch.Tensor) -> torch.Tensor:
        if not torch.any(mask):
            return _scaf_zero_like_loss(log_prob)
        if use_mixed_loss and config.get("loss_remove_token_mean", False):
            return (pg_losses * mask.to(pg_losses.dtype)).sum() / (
                response_mask.shape[0] * response_mask.shape[-1]
            )
        global_batch_info = getattr(config, "global_batch_info", {})
        return agg_loss(
            loss_mat=pg_losses * mask.to(pg_losses.dtype),
            loss_mask=response_mask,
            loss_agg_mode=config.loss_agg_mode,
            **global_batch_info,
        )

    losses = {
        "rollout": aggregate_source(rollout_mask),
        "expert": aggregate_source(expert_mask),
        "hint": aggregate_source(hint_mask),
    }

    sft_loss_coef = config.get("sft_loss_coef", 0.0)
    if sft_loss_coef > 0 and sft_loss_mask is not None:
        sft_mask = _scaf_bool_mask(sft_loss_mask, response_mask)
        if torch.any(sft_mask):
            losses["expert"] = losses["expert"] + masked_mean(-log_prob, sft_mask) * sft_loss_coef

    hint_sft_loss_coef = config.get("hint_sft_loss_coef", 0.0)
    if config.get("use_hint_sft_loss", False) and hint_sft_loss_coef > 0 and hint_sft_loss_mask is not None:
        if torch.any(hint_mask):
            losses["hint"] = losses["hint"] + masked_mean(-log_prob, hint_mask) * hint_sft_loss_coef

    return losses


# [ADD] loss计算
def compute_scaf_ppo_policy_loss(
    config,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    sft_loss_mask: torch.Tensor | None = None,
    hint_sft_loss_mask: torch.Tensor | None = None,
    off_policy_mask: torch.Tensor | None = None,
    rollout_is_weights: torch.Tensor | None = None,
    hint_is_weights: torch.Tensor | None = None,
):
    loss_agg_mode = config.loss_agg_mode
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    if hint_is_weights is not None:
        advantages = advantages * hint_is_weights.detach()

    # [ADD] Enter the expert branch only when this micro-batch has expert tokens.
    if (
        config.get("use_off_policy_loss", False) and off_policy_mask is not None
        # and torch.any(off_policy_mask.to(bool))
    ):
        if loss_mode != "vanilla":
            raise ValueError(
                "Scaf expert off-policy loss currently supports policy_loss.loss_mode='vanilla' only; "
                f"got {loss_mode!r}."
            )
        off_policy_max_clip = config.get("off_policy_max_clip", -1.0)
        off_policy_min_clip = config.get("off_policy_min_clip", -1.0)
        all_max_clip = config.get("all_max_clip", -1.0)
        off_policy_result = compute_token_on_off_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            off_policy_mask=off_policy_mask,
            cliprange=config.clip_ratio,
            cliprange_low=config.clip_ratio_low,
            cliprange_high=config.clip_ratio_high,
            clip_ratio_c=config.get("clip_ratio_c", 3.0),
            clip_upper_bound=config.get("clip_upper_bound", 1.0),
            off_policy_reshape=config.get("off_policy_reshape", "p_div_p_0.1"),
            off_policy_reshape_weight=config.get("off_policy_reshape_weight", 1.0),
            off_policy_reshape_pow_exp=config.get("off_policy_reshape_pow_exp", 0.5),
            on_policy_reshape=config.get("on_policy_reshape", "no_reshape"),
            on_policy_reshape_weight=config.get("on_policy_reshape_weight", 1.0),
            on_policy_reshape_pow_exp=config.get("on_policy_reshape_pow_exp", 0.5),
            off_policy_max_clip=off_policy_max_clip if off_policy_max_clip >= 0 else None,
            off_policy_min_clip=off_policy_min_clip if off_policy_min_clip >= 0 else None,
            all_max_clip=all_max_clip if all_max_clip >= 0 else None,
            loss_agg_mode=loss_agg_mode,
            global_batch_info=config.get("global_batch_info", {}),
            rollout_is_weights=rollout_is_weights,
            off_policy_loss_type=config.get("off_policy_loss_type", "probability"),
            loss_remove_clip=config.get("loss_remove_clip", False), # [ADD] luffy
            loss_remove_token_mean=config.get("loss_remove_token_mean", False), # [ADD] luffy
        )
        pg_loss = off_policy_result["pg_loss"]
        metrics = {
            "actor/pg_clipfrac": off_policy_result["on_pg_clipfrac"],
            "actor/pg_clipfrac_lower": off_policy_result["on_pg_clipfrac_lower"],
            "actor/ppo_kl": off_policy_result["ppo_kl"],
            "actor/on_pg_loss": off_policy_result["on_pg_loss"],
            "actor/off_pg_loss": off_policy_result["off_pg_loss"],
            "actor/off_ratio_mean": off_policy_result["off_ratio_mean"],
            "actor/off_policy_prob": off_policy_result["off_policy_prob"],
            "actor/on_policy_prob": off_policy_result["on_policy_prob"],
            "actor/off_pg_clipfrac": off_policy_result["off_pg_clipfrac"],
            "actor/off_ratio_max_clip_frac": off_policy_result["off_ratio_max_clip_frac"],
            "actor/off_ratio_min_clip_frac": off_policy_result["off_ratio_min_clip_frac"],
        }
    else:
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )

    policy_loss = pg_loss
    sft_loss_coef = config.get("sft_loss_coef", 0.0)
    if sft_loss_mask is not None and sft_loss_coef > 0:
        sft_loss_mask = sft_loss_mask.to(bool)
        if torch.any(sft_loss_mask):
            sft_loss = masked_mean(-log_prob, sft_loss_mask)
            policy_loss = policy_loss + sft_loss * sft_loss_coef
            metrics["actor/sft_loss"] = sft_loss.detach().item()
            metrics["actor/sft_loss_coef"] = sft_loss_coef
            metrics["actor/sft_loss_tokens"] = torch.sum(sft_loss_mask).detach().item()

    hint_sft_loss_coef = config.get("hint_sft_loss_coef", 0.0)
    if config.get("use_hint_sft_loss", False) and hint_sft_loss_mask is not None and hint_sft_loss_coef > 0:
        hint_sft_loss_mask = hint_sft_loss_mask.to(bool)
        if torch.any(hint_sft_loss_mask):
            hint_sft_loss = masked_mean(-log_prob, hint_sft_loss_mask)
            policy_loss = policy_loss + hint_sft_loss * hint_sft_loss_coef
            metrics["actor/hint_sft_loss"] = hint_sft_loss.detach().item()
            metrics["actor/hint_sft_loss_coef"] = hint_sft_loss_coef
            metrics["actor/hint_sft_loss_tokens"] = torch.sum(hint_sft_loss_mask).detach().item()

    return policy_loss, pg_loss, metrics


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def _slice_response_from_unpad_output(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """Slice response from unpad model output.

    Args:
        tensor: model output tensor of shape [bsz, 1]
        data: TensorDict with "prompt_ids", "response_ids", "attention_mask"

    Returns:
        tensor: sliced response tensor of shape [bsz, max_response_len]
    """
    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]
    attention_mask = data["attention_mask"]

    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        max_response_len = response_ids.offsets().max().item()
    else:
        assert not attention_mask.is_nested
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_response_len = response_ids.shape[1]

    sequence_lens = prompt_lens + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)
    assert sequence_offsets[-1].item() == values.shape[0]

    response_list = []
    for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
        pad_size = max_response_len - resp_len
        # left-shift model output by one token for log_probs/values
        response_list.append(F.pad(values[seq_offset - resp_len - 1 : seq_offset - 1], (0, pad_size)))

    output = torch.stack(response_list, dim=0)
    return output


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    log_prob = _slice_response_from_unpad_output(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = _slice_response_from_unpad_output(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)
    hint_is_weights = data.get("hint_is_weights", None)
    if hint_is_weights is not None:
        advantages = advantages * hint_is_weights.detach()
    # [ADD] Scaf-GRPO: optional trajectory-type masks generated by trainer-side hint/expert replacement.
    sft_loss_mask = data.get("sft_loss_mask", None)
    hint_sft_loss_mask = data.get("hint_sft_loss_mask", None)
    off_policy_mask = data.get("off_policy_mask", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    # [DEL] Original PPO policy-loss call, replaced by a Scaf-GRPO-aware branch that can handle expert tokens.
    # policy_loss_fn = get_policy_loss_fn(loss_mode)
    # pg_loss, pg_metrics = policy_loss_fn(
    #     old_log_prob=old_log_prob,
    #     log_prob=log_prob,
    #     advantages=advantages,
    #     response_mask=response_mask,
    #     loss_agg_mode=loss_agg_mode,
    #     config=config,
    #     rollout_is_weights=rollout_is_weights,
    # )
    # [ADD] Scaf-GRPO: use mixed on/off-policy loss only when expert off-policy mask and config are enabled.
    # [ADD] The new engine follows the same expert-token gate as legacy actors.
    if (
        config.get("use_off_policy_loss", False) and off_policy_mask is not None
        # and torch.any(off_policy_mask.to(bool))
    ):
        if loss_mode != "vanilla":
            raise ValueError(
                "Scaf expert off-policy loss currently supports policy_loss.loss_mode='vanilla' only; "
                f"got {loss_mode!r}."
            )
        off_policy_max_clip = config.get("off_policy_max_clip", -1.0)
        off_policy_min_clip = config.get("off_policy_min_clip", -1.0)
        all_max_clip = config.get("all_max_clip", -1.0)
        off_policy_result = compute_token_on_off_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            off_policy_mask=off_policy_mask,
            cliprange=config.clip_ratio,
            cliprange_low=config.clip_ratio_low,
            cliprange_high=config.clip_ratio_high,
            clip_ratio_c=config.get("clip_ratio_c", 3.0),
            clip_upper_bound=config.get("clip_upper_bound", 1.0),
            off_policy_reshape=config.get("off_policy_reshape", "p_div_p_0.1"),
            off_policy_reshape_weight=config.get("off_policy_reshape_weight", 1.0),
            off_policy_reshape_pow_exp=config.get("off_policy_reshape_pow_exp", 0.5),
            on_policy_reshape=config.get("on_policy_reshape", "no_reshape"),
            on_policy_reshape_weight=config.get("on_policy_reshape_weight", 1.0),
            on_policy_reshape_pow_exp=config.get("on_policy_reshape_pow_exp", 0.5),
            off_policy_max_clip=off_policy_max_clip if off_policy_max_clip >= 0 else None,
            off_policy_min_clip=off_policy_min_clip if off_policy_min_clip >= 0 else None,
            all_max_clip=all_max_clip if all_max_clip >= 0 else None,
            loss_agg_mode=loss_agg_mode,
            global_batch_info=config.global_batch_info,
            rollout_is_weights=rollout_is_weights,
            off_policy_loss_type=config.get("off_policy_loss_type", "probability"),
            loss_remove_clip=config.get("loss_remove_clip", False),
            loss_remove_token_mean=config.get("loss_remove_token_mean", False),
        )
        pg_loss = off_policy_result["pg_loss"]
        pg_metrics = {
            "actor/pg_clipfrac": off_policy_result["on_pg_clipfrac"],
            "actor/pg_clipfrac_lower": off_policy_result["on_pg_clipfrac_lower"],
            "actor/ppo_kl": off_policy_result["ppo_kl"],
            "actor/on_pg_loss": off_policy_result["on_pg_loss"],
            "actor/off_pg_loss": off_policy_result["off_pg_loss"],
            "actor/off_ratio_mean": off_policy_result["off_ratio_mean"],
            "actor/off_policy_prob": off_policy_result["off_policy_prob"],
            "actor/on_policy_prob": off_policy_result["on_policy_prob"],
            "actor/off_pg_clipfrac": off_policy_result["off_pg_clipfrac"],
            "actor/off_ratio_max_clip_frac": off_policy_result["off_ratio_max_clip_frac"],
            "actor/off_ratio_min_clip_frac": off_policy_result["off_ratio_min_clip_frac"],
        }
    else:
        # [LOC] 没有注入专家轨迹，只有hint走这里
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = pg_loss.detach().item()
    policy_loss = pg_loss

    # [ADD] Scaf-GRPO: auxiliary NLL/SFT loss over directly injected expert response tokens.
    sft_loss_coef = config.get("sft_loss_coef", 0.0)
    if sft_loss_mask is not None and sft_loss_coef > 0:
        sft_loss_mask = sft_loss_mask.to(bool)
        if torch.any(sft_loss_mask):
            expert_sft_loss = masked_mean(-log_prob, sft_loss_mask)
            policy_loss = policy_loss + expert_sft_loss * sft_loss_coef
            metrics["actor/sft_loss"] = expert_sft_loss.detach().item()
            metrics["actor/sft_loss_coef"] = sft_loss_coef
            metrics["actor/sft_loss_tokens"] = torch.sum(sft_loss_mask).detach().item()

    # [ADD] Scaf-GRPO: optional auxiliary NLL/SFT loss over successful hinted rollout response tokens.
    hint_sft_loss_coef = config.get("hint_sft_loss_coef", 0.0)
    if config.get("use_hint_sft_loss", False) and hint_sft_loss_mask is not None and hint_sft_loss_coef > 0:
        hint_sft_loss_mask = hint_sft_loss_mask.to(bool)
        if torch.any(hint_sft_loss_mask):
            hint_sft_loss = masked_mean(-log_prob, hint_sft_loss_mask)
            policy_loss = policy_loss + hint_sft_loss * hint_sft_loss_coef
            metrics["actor/hint_sft_loss"] = hint_sft_loss.detach().item()
            metrics["actor/hint_sft_loss_coef"] = hint_sft_loss_coef
            metrics["actor/hint_sft_loss_tokens"] = torch.sum(hint_sft_loss_mask).detach().item()

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = kl_loss.detach().item()
        metrics["kl_coef"] = config.kl_loss_coef

    metrics["actor/total_loss"] = policy_loss.detach().item()

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = _slice_response_from_unpad_output(model_output["values"], data)  # (bsz, response_length)

    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
