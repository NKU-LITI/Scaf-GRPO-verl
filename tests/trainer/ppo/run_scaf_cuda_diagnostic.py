"""One-shot CUDA diagnostic for Scaf source-specific training signals.

This script intentionally avoids Ray so it can run beside an existing trainer.
It uses one real all-wrong rollout group and its matching expert trajectory.
"""

import json
import math
import os
from collections import defaultdict

import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.workers.utils.losses import compute_scaf_ppo_policy_loss, compute_scaf_source_policy_losses


MODEL_PATH = os.environ.get(
    "SCAF_DEBUG_MODEL_PATH", "/workplace/nankai/liting_space/LLM/Qwen2.5-Math-1.5B"
)
DATA_PATH = "data/DeepScaler/Qwen2d5_math_7b/train_800.success_rate_k8.parquet"
ROLLOUT_PATH = "outputs/qwen25_math7b_stratified_expert_test_grad/rollout_log/training/1.jsonl"
MAX_RESPONSE_TOKENS = int(os.environ.get("SCAF_DEBUG_MAX_RESPONSE_TOKENS", "256"))


def actor_config(**overrides):
    values = {
        "loss_agg_mode": "token-mean",
        "policy_loss": {"loss_mode": "vanilla"},
        "clip_ratio": 0.2,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "clip_ratio_c": 3.0,
        "clip_upper_bound": 1.0,
        "global_batch_info": {},
        "use_off_policy_loss": False,
        "off_policy_loss_type": "probability",
        "off_policy_reshape": "p_div_p_0.1",
        "off_policy_max_clip": -1.0,
        "off_policy_min_clip": -1.0,
        "loss_remove_clip": True,
        "loss_remove_token_mean": True,
        "sft_loss_coef": 0.0,
        "use_hint_sft_loss": False,
        "hint_sft_loss_coef": 0.0,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def load_real_group():
    groups = defaultdict(list)
    with open(ROLLOUT_PATH, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            groups[row["uid"]].append(row)

    # The rollout dump is written after intervention, so an initially all-wrong
    # expert-injected group contains exactly one successful expert and 7 failures.
    group = next(rows for rows in groups.values() if len(rows) == 8 and sum(row["score"] > 0 for row in rows) == 1)
    marker_start = "\nuser\n"
    marker_end = "\nassistant\n"
    question = group[0]["input"].split(marker_start, 1)[1].rsplit(marker_end, 1)[0]

    table = pq.read_table(
        DATA_PATH,
        columns=["question", "prompt", "solution_breakdown_cot_answer", "solution_breakdown_parts"],
    )
    match = next(row for row in table.to_pylist() if row["question"].strip() == question.strip())
    failed = [row["output"] for row in group if row["score"] <= 0]
    injected_expert = next(row["output"] for row in group if row["score"] > 0)
    assert injected_expert.strip() == match["solution_breakdown_cot_answer"].strip()
    hint_parts = [str(part).strip() for part in match["solution_breakdown_parts"] if str(part).strip()]
    hinted_prompt = [
        match["prompt"][0],
        {
            "role": "user",
            "content": f"Question: {question}\nSolution Hints: {' '.join(hint_parts)}",
        },
    ]
    return match["prompt"], hinted_prompt, injected_expert, failed


def build_log_probs(model, tokenizer, prompt, responses):
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    encoded_responses = [
        tokenizer.encode(response, add_special_tokens=False)[: MAX_RESPONSE_TOKENS - 1] + [tokenizer.eos_token_id]
        for response in responses
    ]
    max_total = max(len(prompt_ids) + len(response) for response in encoded_responses)
    input_ids = []
    attention_mask = []
    for response in encoded_responses:
        ids = prompt_ids + response
        pad = max_total - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)

    input_ids = torch.tensor(input_ids, device="cuda")
    attention_mask = torch.tensor(attention_mask, device="cuda")
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    token_log_probs = output.logits[:, :-1].float().log_softmax(dim=-1)

    max_response = max(len(response) for response in encoded_responses)
    rows = []
    masks = []
    for row_idx, response in enumerate(encoded_responses):
        start = len(prompt_ids) - 1
        positions = torch.arange(start, start + len(response), device="cuda")
        targets = input_ids[row_idx, len(prompt_ids) : len(prompt_ids) + len(response)]
        selected = token_log_probs[row_idx, positions, targets]
        rows.append(torch.nn.functional.pad(selected, (0, max_response - len(response))))
        masks.append([1] * len(response) + [0] * (max_response - len(response)))
    return torch.stack(rows), torch.tensor(masks, dtype=torch.bool, device="cuda")


def parameter_grad_norm(loss, model):
    grads = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True, allow_unused=True)
    squared_norm = torch.zeros((), dtype=torch.float32, device="cuda")
    for grad in grads:
        if grad is not None:
            squared_norm += grad.detach().float().square().sum()
    return squared_norm.sqrt().item()


def measure(
    name,
    config,
    model,
    log_prob,
    response_mask,
    advantages,
    expert_mask=None,
    hint_mask=None,
    hint_weights=None,
    parameter_sources=(),
):
    zeros = torch.zeros_like(response_mask)
    source_losses = compute_scaf_source_policy_losses(
        config=config,
        old_log_prob=log_prob.detach(),
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        sft_loss_mask=expert_mask if config.sft_loss_coef > 0 else zeros,
        hint_sft_loss_mask=hint_mask if hint_mask is not None else zeros,
        off_policy_mask=expert_mask if expert_mask is not None else zeros,
        hint_is_weights=hint_weights,
    )
    actual_policy_loss, _, _ = compute_scaf_ppo_policy_loss(
        config=config,
        old_log_prob=log_prob.detach(),
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        sft_loss_mask=expert_mask if config.sft_loss_coef > 0 else zeros,
        hint_sft_loss_mask=hint_mask if hint_mask is not None else zeros,
        off_policy_mask=expert_mask if expert_mask is not None else zeros,
        hint_is_weights=hint_weights,
    )
    source_loss_sum = sum(source_losses.values())
    loss_difference = (actual_policy_loss - source_loss_sum).detach().abs().item()
    assert loss_difference < 1e-5, (name, actual_policy_loss, source_loss_sum)
    result = {}
    for source, loss in source_losses.items():
        grad = torch.autograd.grad(loss, log_prob, retain_graph=True)[0]
        result[source] = {
            "loss": loss.detach().item(),
            "logprob_grad_norm": grad.float().norm().item(),
        }
        if source in parameter_sources:
            result[source]["parameter_grad_norm"] = parameter_grad_norm(loss, model)
    print(
        json.dumps(
            {
                "case": name,
                "actual_total_loss": actual_policy_loss.detach().item(),
                "source_loss_sum": source_loss_sum.detach().item(),
                "loss_difference": loss_difference,
                "sources": result,
            },
            ensure_ascii=False,
        )
    )


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda().eval()

    prompt, hinted_prompt, expert, failed = load_real_group()
    responses = [expert] + failed
    log_prob, response_mask = build_log_probs(model, tokenizer, prompt, responses)
    with torch.no_grad():
        hint_behavior_log_prob, hint_behavior_mask = build_log_probs(model, tokenizer, hinted_prompt, [expert])
    assert torch.equal(hint_behavior_mask[0], response_mask[0])
    expert_mask = torch.zeros_like(response_mask)
    expert_mask[0] = response_mask[0]
    hint_mask = expert_mask.clone()

    # Replacing one item in an eight-sample all-wrong group gives these Dr.GRPO advantages.
    trajectory_advantages = torch.tensor([0.875] + [-0.125] * 7, device="cuda")
    advantages = trajectory_advantages[:, None] * response_mask
    zero_advantages = torch.zeros_like(advantages)

    measure("all_wrong_grpo", actor_config(), model, log_prob, response_mask, zero_advantages)
    measure(
        "expert_rl_p_div_p_0.1",
        actor_config(use_off_policy_loss=True),
        model,
        log_prob,
        response_mask,
        advantages,
        expert_mask=expert_mask,
        parameter_sources=("rollout", "expert"),
    )
    measure(
        "expert_rl_advantage_weighted_log_prob",
        actor_config(
            use_off_policy_loss=True,
            off_policy_loss_type="advantage_weighted_log_prob",
        ),
        model,
        log_prob,
        response_mask,
        advantages,
        expert_mask=expert_mask,
        parameter_sources=("rollout", "expert"),
    )
    measure(
        "expert_rl_standard_ratio_counterfactual",
        actor_config(),
        model,
        log_prob,
        response_mask,
        advantages,
        expert_mask=expert_mask,
        parameter_sources=("rollout", "expert"),
    )
    hint_weights = torch.ones((8, 1), device="cuda")
    hint_weights[0] = math.exp(-5.0)
    measure(
        "hint_response_rl_with_observed_is_clip",
        actor_config(),
        model,
        log_prob,
        response_mask,
        advantages,
        hint_mask=hint_mask,
        hint_weights=hint_weights,
        parameter_sources=("rollout", "hint"),
    )
    hint_token_mask = response_mask[0]
    token_log_ratio = log_prob[0].detach() - hint_behavior_log_prob[0].detach()
    token_normalized_weights = torch.ones_like(log_prob)
    clipped_token_weights = torch.exp(torch.clamp(token_log_ratio, min=-2.0, max=2.0))
    hint_weight_mean = clipped_token_weights[hint_token_mask].mean().clamp_min(1e-6)
    token_normalized_weights[0, hint_token_mask] = (
        clipped_token_weights[hint_token_mask] / hint_weight_mean
    )
    selected_weights = token_normalized_weights[0, hint_token_mask]
    print(
        json.dumps(
            {
                "case": "hint_token_normalized_weight_stats",
                "raw_sequence_log_ratio": token_log_ratio[hint_token_mask].sum().item(),
                "weight_mean": selected_weights.mean().item(),
                "weight_min": selected_weights.min().item(),
                "weight_max": selected_weights.max().item(),
            }
        )
    )
    measure(
        "hint_response_rl_token_normalized_is",
        actor_config(),
        model,
        log_prob,
        response_mask,
        advantages,
        hint_mask=hint_mask,
        hint_weights=token_normalized_weights,
        parameter_sources=("rollout", "hint"),
    )
    measure(
        "hint_response_rl_without_is_counterfactual",
        actor_config(),
        model,
        log_prob,
        response_mask,
        advantages,
        hint_mask=hint_mask,
        parameter_sources=("rollout", "hint"),
    )
    measure(
        "expert_rl_plus_sft",
        actor_config(use_off_policy_loss=True, sft_loss_coef=1.0),
        model,
        log_prob,
        response_mask,
        advantages,
        expert_mask=expert_mask,
    )
    measure(
        "hint_rl_plus_sft",
        actor_config(use_hint_sft_loss=True, hint_sft_loss_coef=1.0),
        model,
        log_prob,
        response_mask,
        advantages,
        hint_mask=hint_mask,
        hint_weights=hint_weights,
    )


if __name__ == "__main__":
    main()
