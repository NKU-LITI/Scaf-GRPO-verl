# [ADD] Scaf-GRPO helper functions for hinted rollout construction and expert trajectory tensor injection.
import numpy as np
import torch

from verl import DataProto


def compute_position_id_with_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


def normalize_expert_target(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()

    # [DEL] Selecting only the first list entry drops multi-part expert trajectories from Arrow-backed parquet values.
    # if isinstance(value, list) and value:
    #     value = value[0]
    
    # Keep every non-empty expert segment in its original order.
    if isinstance(value, (list, tuple)):
        parts = [normalize_expert_target(item) for item in value]
        parts = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
        return "\n\n".join(parts) if parts else None
    if isinstance(value, dict):
        return value.get("content")
    if isinstance(value, str):
        return value
    return None


def remove_uids_from_dataproto(data: DataProto, uids_to_remove) -> DataProto:
    if not uids_to_remove:
        return data
    remove_set = set(uids_to_remove)
    keep_indices = [i for i, uid in enumerate(data.non_tensor_batch["uid"]) if uid not in remove_set]
    return data.select_idxs(keep_indices) if keep_indices else data.slice(0, 0)


def build_hinted_gen_batch(base_batch: DataProto, stage_count: int = 3) -> DataProto:
    questions = base_batch.non_tensor_batch["question"]
    original_uids = base_batch.non_tensor_batch["uid"]
    reward_models = base_batch.non_tensor_batch["reward_model"]
    data_sources = base_batch.non_tensor_batch["data_source"]

    all_stages = [
        ("knowledge_components_parts", "Knowledge Hints"),
        ("planning_skeleton_parts", "Planning Hints"),
        ("solution_breakdown_parts", "Solution Hints"),
    ]
    if stage_count not in (1, 2, 3):
        raise ValueError(f"stage_count must be 1, 2, or 3; got {stage_count}.")

    raw_prompts = []
    uids = []
    hint_levels = [] # 三层
    hint_counts = [] # 12层
    reward_model_out = []
    data_source_out = []

    for i in range(len(questions)):
        hint_count_global = 0
        for hint_stage, (stage_key, hint_label) in enumerate(all_stages[:stage_count], start=1):
            raw_parts = base_batch.non_tensor_batch.get(stage_key, [[] for _ in range(len(questions))])[i]
            if not isinstance(raw_parts, (list, tuple, np.ndarray)):
                raw_parts = []
            parts = [str(part).strip() for part in raw_parts if str(part).strip()]

            for hint_count in range(1, len(parts) + 1):
                hint_count_global += 1
                content = f"Question: {questions[i]}\n{hint_label}: {' '.join(parts[:hint_count])}"
                raw_prompts.append(
                    [
                        {
                            "role": "system",
                            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                        },
                        {"role": "user", "content": content},
                    ]
                )
                uids.append(original_uids[i])
                hint_levels.append(hint_stage)
                hint_counts.append(hint_count_global)
                reward_model_out.append(reward_models[i])
                data_source_out.append(data_sources[i])


    data = {
        "dummy_tensor": torch.zeros((len(raw_prompts), 1), dtype=torch.uint8),
        "raw_prompt": np.asarray(raw_prompts, dtype=object),
        "uid": np.asarray(uids, dtype=object),
        "hint_level": np.asarray(hint_levels, dtype=object),
        "hint_count": np.asarray(hint_counts, dtype=object), 
        "reward_model": np.asarray(reward_model_out, dtype=object),
        "data_source": np.asarray(data_source_out, dtype=object),
    }
    return DataProto.from_single_dict(data, meta_info={"new_gen": True})


# [ADD] 取出专家轨迹，tokenizer编码和长度截断和padding，生成tensor，之后直接替换rollout失败轨迹，进入后续loss计算
def build_expert_response_data(
    base_batch: DataProto,  # 当前训练batch，数量是 train_batch_size × rollout.n = 64*8 =512
    row_idx: int, # 当前处理第几个样本
    tokenizer,
    max_response_length: int,
    truncation: str = "right", # right是保留前面的
):
    expert_target = normalize_expert_target(base_batch.non_tensor_batch.get("expert_target", [None])[row_idx])
    if not expert_target or not str(expert_target).strip():
        return None

    response_ids = tokenizer.encode(str(expert_target), add_special_tokens=False) # 没有处理chat template，专家轨迹前面不能有"assistant: Let's ..."前缀
    if tokenizer.eos_token_id is not None:
        response_ids.append(tokenizer.eos_token_id)

    if len(response_ids) > max_response_length:
        if truncation == "left":
            response_ids = response_ids[-max_response_length:]
        elif truncation == "error":
            raise RuntimeError(f"Expert response length {len(response_ids)} > {max_response_length}")
        else:
            response_ids = response_ids[:max_response_length]

    valid_len = len(response_ids)
    pad_len = max_response_length - valid_len
    response_ids = response_ids + [tokenizer.pad_token_id] * pad_len

    responses = torch.tensor(response_ids, dtype=base_batch.batch["prompts"].dtype)
    response_mask = torch.zeros(max_response_length, dtype=base_batch.batch["attention_mask"].dtype)
    response_mask[:valid_len] = 1

    prompts = base_batch.batch["prompts"][row_idx].detach().cpu()
    prompt_length = prompts.shape[-1]
    prompt_attention_mask = base_batch.batch["attention_mask"][row_idx][:prompt_length].detach().cpu()
    input_ids = torch.cat([prompts, responses], dim=0)
    attention_mask = torch.cat([prompt_attention_mask, response_mask], dim=0)
    position_ids = compute_position_id_with_mask(attention_mask)

    return {
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": response_mask, # 长度是reponse长度，屏蔽掉padding token
        "sft_loss_mask": response_mask.clone(),
        "off_policy_mask": response_mask.clone(),
    }
