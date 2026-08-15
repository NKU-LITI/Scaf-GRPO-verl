#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze current-policy probabilities on expert trajectories.

Goal
----
For each expert response token y_t^E, compute

    log p_t = log pi_theta(y_t^E | x, y_<t^E)
    p_t     = exp(log p_t)

using teacher forcing.

This script intentionally matches the current Scaf-GRPO expert injection logic:
1. prompt: tokenizer.apply_chat_template(..., add_generation_prompt=True)
2. expert response: tokenizer.encode(..., add_special_tokens=False)
3. append EOS
4. truncate response to max_response_length
5. causal LM shift:
      logits at position (token_position - 1)
      predicts token at token_position

It does NOT:
- generate responses
- run PPO/GRPO
- backward
- modify any training code
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = "/workplace/nankai/liting_space/LLM/Qwen2.5-Math-7B"

DEFAULT_DATA_PATH = (
    "/workplace/nankai/liting_space/SFT-RL/Scaf-GRPO-verl/"
    "data/DeepScaler/Qwen2d5_math_7b/train_800.success_rate_k8.parquet"
)

DEFAULT_OUTPUT_DIR = (
    "/workplace/nankai/liting_space/SFT-RL/Scaf-GRPO-verl/"
    "outputs/expert_probability_diagnostic"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of rows to analyze if --indices is not specified.",
    )

    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help='Exact parquet row indices, e.g. "0,12,37,88".',
    )

    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=["hard", "medium", "easy"],
        help="Optional difficulty filter before selecting samples.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Randomly sample rows instead of taking the first num_samples.",
    )

    parser.add_argument(
        "--max_response_length",
        type=int,
        default=2048,
        help="Match training rollout.max_response_length.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Training actor log-prob divides logits by temperature.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="LUFFY shaping constant c/gamma.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )

    parser.add_argument(
        "--print_lowest_k",
        type=int,
        default=20,
        help="Print K lowest-probability expert tokens for every trajectory.",
    )

    return parser.parse_args()


def get_dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def normalize_prompt(value: Any) -> list[dict]:
    """
    Convert parquet's object/list representation into
    [{"role": ..., "content": ...}, ...].
    """

    if isinstance(value, np.ndarray):
        value = value.tolist()

    # In case parquet/pandas gives a serialized representation.
    if isinstance(value, str):
        s = value.strip()

        try:
            value = json.loads(s)
        except Exception:
            try:
                value = ast.literal_eval(s)
            except Exception as exc:
                raise ValueError(
                    f"Cannot parse prompt string as chat messages:\n{s[:500]}"
                ) from exc

    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"prompt must be list[dict], got {type(value)}"
        )

    messages = []

    for item in value:
        if isinstance(item, np.ndarray):
            item = item.tolist()

        if not isinstance(item, dict):
            try:
                item = dict(item)
            except Exception as exc:
                raise TypeError(
                    f"Prompt message is not dict-like: {type(item)}"
                ) from exc

        if "role" not in item or "content" not in item:
            raise ValueError(
                f"Invalid prompt message: {item}"
            )

        messages.append(
            {
                "role": str(item["role"]),
                "content": str(item["content"]),
            }
        )

    return messages


def normalize_expert_target(value: Any) -> str | None:
    """
    Mirrors current Scaf-GRPO normalize_expert_target semantics.
    """

    if value is None:
        return None

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        parts = [
            normalize_expert_target(item)
            for item in value
        ]
        parts = [
            part.strip()
            for part in parts
            if isinstance(part, str) and part.strip()
        ]
        return "\n\n".join(parts) if parts else None

    if isinstance(value, dict):
        content = value.get("content")
        return str(content) if content is not None else None

    if isinstance(value, str):
        return value

    return str(value)


def get_prompt_ids(tokenizer, prompt_value: Any) -> list[int]:
    """
    Match rollout-side prompt formatting:
      prompt chat messages
        -> apply_chat_template(..., add_generation_prompt=True)
    """

    messages = normalize_prompt(prompt_value)

    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.tolist()

    if (
        isinstance(prompt_ids, list)
        and len(prompt_ids) == 1
        and isinstance(prompt_ids[0], list)
    ):
        prompt_ids = prompt_ids[0]

    return list(prompt_ids)


def get_expert_response_ids(
    tokenizer,
    expert_text: str,
    max_response_length: int,
) -> list[int]:
    """
    Match build_expert_response_data():

        tokenizer.encode(expert_target, add_special_tokens=False)
        append eos_token_id
        right truncate to max_response_length
    """

    response_ids = tokenizer.encode(
        expert_text,
        add_special_tokens=False,
    )

    if tokenizer.eos_token_id is not None:
        response_ids.append(tokenizer.eos_token_id)

    if len(response_ids) > max_response_length:
        response_ids = response_ids[:max_response_length]

    return response_ids


@torch.no_grad()
def compute_response_log_probs(
    model,
    prompt_ids: list[int],
    response_ids: list[int],
    device: str,
    temperature: float,
) -> np.ndarray:
    """
    Returns one log-prob per response token.

    Full input:

        [prompt_0 ... prompt_P-1, response_0 ... response_R-1]

    Causal shift:

        logits[P - 1] predicts response[0]
        logits[P]     predicts response[1]
        ...
        logits[P+R-2] predicts response[R-1]

    This matches VERL's:
        logits[:, -response_length - 1 : -1]
    """

    if len(prompt_ids) == 0:
        raise ValueError("Empty prompt.")
    if len(response_ids) == 0:
        raise ValueError("Empty expert response.")

    all_ids = prompt_ids + response_ids

    input_ids = torch.tensor(
        [all_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(input_ids)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    logits = outputs.logits

    if temperature != 1.0:
        logits = logits / temperature

    prompt_len = len(prompt_ids)
    response_len = len(response_ids)

    # logits[P-1 : P+R-1] predicts response[0:R]
    response_logits = logits[
        :,
        prompt_len - 1 : prompt_len + response_len - 1,
        :,
    ]

    targets = torch.tensor(
        [response_ids],
        dtype=torch.long,
        device=device,
    )

    assert response_logits.shape[1] == response_len
    assert targets.shape[1] == response_len

    # Avoid materializing full log_softmax tensor in fp32.
    #
    # log p(target) =
    #     target_logit - logsumexp(all_vocab_logits)
    #
    # Do sequence chunks so the temporary fp32 tensor stays small.
    chunk_size = 256
    log_probs_chunks = []

    for start in range(0, response_len, chunk_size):
        end = min(start + chunk_size, response_len)

        chunk_logits = response_logits[:, start:end, :]
        chunk_targets = targets[:, start:end]

        target_logits = torch.gather(
            chunk_logits,
            dim=-1,
            index=chunk_targets.unsqueeze(-1),
        ).squeeze(-1)

        # fp32 only for numerical stability of logsumexp.
        log_z = torch.logsumexp(
            chunk_logits.float(),
            dim=-1,
        )

        chunk_log_probs = (
            target_logits.float() - log_z
        )

        log_probs_chunks.append(
            chunk_log_probs.cpu()
        )

    log_probs = torch.cat(
        log_probs_chunks,
        dim=1,
    ).squeeze(0)

    assert log_probs.numel() == response_len

    return log_probs.numpy().astype(np.float64)


def quantile_dict(values: np.ndarray, prefix: str):
    if len(values) == 0:
        return {}

    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p10": float(np.quantile(values, 0.10)),
        f"{prefix}_p25": float(np.quantile(values, 0.25)),
        f"{prefix}_median": float(np.quantile(values, 0.50)),
        f"{prefix}_p75": float(np.quantile(values, 0.75)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_mean": float(np.mean(values)),
    }


def probability_bucket(p: float) -> str:
    if p < 1e-5:
        return "p<1e-5"
    if p < 1e-4:
        return "1e-5<=p<1e-4"
    if p < 1e-3:
        return "1e-4<=p<1e-3"
    if p < 1e-2:
        return "1e-3<=p<1e-2"
    if p < 1e-1:
        return "1e-2<=p<1e-1"
    return "p>=1e-1"


def safe_token_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return (
        text.replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def print_sample_summary(
    summary: dict,
    token_df: pd.DataFrame,
    lowest_k: int,
):
    print("\n" + "=" * 100)

    print(
        f"row_index={summary['row_index']}  "
        f"difficulty={summary['difficulty']}  "
        f"prompt_tokens={summary['prompt_token_count']}  "
        f"expert_tokens={summary['response_token_count']}"
    )

    print(
        f"logp: mean={summary['logp_mean']:.4f}, "
        f"median={summary['logp_median']:.4f}, "
        f"p10={summary['logp_p10']:.4f}, "
        f"p90={summary['logp_p90']:.4f}"
    )

    print(
        f"p:    mean={summary['p_mean']:.6g}, "
        f"median={summary['p_median']:.6g}, "
        f"p10={summary['p_p10']:.6g}, "
        f"p90={summary['p_p90']:.6g}"
    )

    print(
        f"geometric mean token p = "
        f"{summary['geometric_mean_probability']:.6g}"
    )

    print(
        f"LUFFY gradient coeff: "
        f"mean={summary['luffy_weight_mean']:.6g}, "
        f"median={summary['luffy_weight_median']:.6g}"
    )

    print(
        f"Adv/LUFFY amplification: "
        f"mean={summary['adv_over_luffy_mean']:.3f}x, "
        f"median={summary['adv_over_luffy_median']:.3f}x"
    )

    print("\nProbability buckets:")

    bucket_cols = [
        "p<1e-5",
        "1e-5<=p<1e-4",
        "1e-4<=p<1e-3",
        "1e-3<=p<1e-2",
        "1e-2<=p<1e-1",
        "p>=1e-1",
    ]

    for bucket in bucket_cols:
        count = int(summary.get(f"bucket_count/{bucket}", 0))
        ratio = float(summary.get(f"bucket_ratio/{bucket}", 0.0))

        print(
            f"  {bucket:16s}: "
            f"{count:5d} tokens  ({ratio:7.2%})"
        )

    print(
        f"\nLowest-probability {min(lowest_k, len(token_df))} tokens:"
    )

    low = token_df.nsmallest(
        lowest_k,
        "probability",
    )[
        [
            "token_index",
            "token_id",
            "token_text",
            "log_prob",
            "probability",
            "luffy_weight",
            "adv_over_luffy",
            "is_eos",
        ]
    ]

    print(low.to_string(index=False))


def select_rows(
    df: pd.DataFrame,
    args,
) -> pd.DataFrame:
    df = df.copy()
    df["_row_index"] = np.arange(len(df))

    if args.indices:
        indices = [
            int(x.strip())
            for x in args.indices.split(",")
            if x.strip()
        ]

        invalid = [
            x for x in indices
            if x < 0 or x >= len(df)
        ]

        if invalid:
            raise IndexError(
                f"Invalid parquet row indices: {invalid}"
            )

        return df.iloc[indices].copy()

    if args.difficulty is not None:
        if "difficulty_bucket" not in df.columns:
            raise KeyError(
                "--difficulty was given but "
                "difficulty_bucket is absent."
            )

        df = df[
            df["difficulty_bucket"].astype(str)
            == args.difficulty
        ].copy()

    if len(df) == 0:
        raise RuntimeError("No rows remain after filtering.")

    n = min(args.num_samples, len(df))

    if args.random_sample:
        return df.sample(
            n=n,
            random_state=args.seed,
        )

    return df.head(n).copy()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("Expert trajectory probability diagnostic")
    print("=" * 100)

    print(f"Model: {args.model_path}")
    print(f"Data : {args.data_path}")
    print(f"Device: {args.device}")
    print(f"dtype : {args.dtype}")
    print(
        f"max_response_length: "
        f"{args.max_response_length}"
    )
    print(f"temperature: {args.temperature}")
    print(f"LUFFY gamma/c: {args.gamma}")

    print("\nLoading parquet...")

    df = pd.read_parquet(args.data_path)

    required = {
        "prompt",
        "solution_breakdown_cot_answer",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"Dataset missing required columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    selected = select_rows(df, args)

    print(
        f"Dataset rows: {len(df)}, "
        f"selected: {len(selected)}"
    )

    print("\nSelected rows:")

    display_cols = ["_row_index"]

    for col in [
        "difficulty_bucket",
        "question",
    ]:
        if col in selected.columns:
            display_cols.append(col)

    preview = selected[display_cols].copy()

    if "question" in preview.columns:
        preview["question"] = (
            preview["question"]
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.slice(0, 100)
        )

    print(preview.to_string(index=False))

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=get_dtype(args.dtype),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model = model.to(args.device)
    model.eval()

    print(
        f"Model max_position_embeddings = "
        f"{getattr(model.config, 'max_position_embeddings', 'unknown')}"
    )

    all_token_records = []
    trajectory_summaries = []

    gamma = float(args.gamma)

    for _, row in selected.iterrows():
        row_index = int(row["_row_index"])

        expert_text = normalize_expert_target(
            row["solution_breakdown_cot_answer"]
        )

        if not expert_text or not expert_text.strip():
            print(
                f"[SKIP] row={row_index}: empty expert trajectory"
            )
            continue

        prompt_ids = get_prompt_ids(
            tokenizer,
            row["prompt"],
        )

        response_ids = get_expert_response_ids(
            tokenizer,
            expert_text,
            args.max_response_length,
        )

        total_len = (
            len(prompt_ids)
            + len(response_ids)
        )

        model_max_len = getattr(
            model.config,
            "max_position_embeddings",
            None,
        )

        if (
            model_max_len is not None
            and total_len > model_max_len
        ):
            raise RuntimeError(
                f"row {row_index}: prompt+expert length "
                f"{total_len} exceeds model "
                f"max_position_embeddings={model_max_len}.\n"
                f"prompt={len(prompt_ids)}, "
                f"response={len(response_ids)}"
            )

        print(
            f"\nForward row={row_index}: "
            f"prompt={len(prompt_ids)}, "
            f"response={len(response_ids)}, "
            f"total={total_len}"
        )

        log_probs = compute_response_log_probs(
            model=model,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            device=args.device,
            temperature=args.temperature,
        )

        # Use float64 on CPU so very low probabilities
        # do not unnecessarily underflow.
        probabilities = np.exp(log_probs)

        # Gradient coefficients excluding advantage A:
        #
        # vanilla:
        #   w = p
        #
        # LUFFY:
        #   w = c p / (p+c)^2
        #
        # advantage-weighted logprob:
        #   w = 1
        vanilla_weight = probabilities.copy()

        luffy_weight = (
            gamma
            * probabilities
            / np.square(probabilities + gamma)
        )

        adv_weight = np.ones_like(
            probabilities,
            dtype=np.float64,
        )

        # Avoid division by literal zero in pathological underflow.
        adv_over_luffy = np.divide(
            adv_weight,
            luffy_weight,
            out=np.full_like(
                luffy_weight,
                np.inf,
            ),
            where=luffy_weight > 0,
        )

        difficulty = (
            str(row["difficulty_bucket"])
            if "difficulty_bucket" in row
            else "unknown"
        )

        question = (
            str(row["question"])
            if "question" in row
            else ""
        )

        token_records = []

        for token_idx, (
            token_id,
            logp,
            p,
            w_vanilla,
            w_luffy,
            w_adv,
            ratio_adv_luffy,
        ) in enumerate(
            zip(
                response_ids,
                log_probs,
                probabilities,
                vanilla_weight,
                luffy_weight,
                adv_weight,
                adv_over_luffy,
            )
        ):
            is_eos = (
                tokenizer.eos_token_id is not None
                and int(token_id)
                == int(tokenizer.eos_token_id)
            )

            record = {
                "row_index": row_index,
                "difficulty": difficulty,
                "token_index": token_idx,
                "token_id": int(token_id),
                "token_text": safe_token_text(
                    tokenizer,
                    int(token_id),
                ),
                "is_eos": bool(is_eos),

                "log_prob": float(logp),
                "probability": float(p),

                # coefficient before A * grad(log p)
                "vanilla_weight": float(
                    w_vanilla
                ),
                "luffy_weight": float(
                    w_luffy
                ),
                "adv_weight": float(
                    w_adv
                ),

                "adv_over_luffy": float(
                    ratio_adv_luffy
                ),

                "probability_bucket": (
                    probability_bucket(float(p))
                ),
            }

            token_records.append(record)
            all_token_records.append(record)

        token_df = pd.DataFrame(token_records)

        # trajectory-level summary
        summary = {
            "row_index": row_index,
            "difficulty": difficulty,
            "question": question,

            "prompt_token_count": len(
                prompt_ids
            ),
            "response_token_count": len(
                response_ids
            ),

            # Mean log p is the stable trajectory-level
            # teacher-forced likelihood statistic.
            "mean_log_prob": float(
                np.mean(log_probs)
            ),

            # exp(mean(log p))
            "geometric_mean_probability": float(
                np.exp(np.mean(log_probs))
            ),
        }

        summary.update(
            quantile_dict(
                log_probs,
                "logp",
            )
        )

        summary.update(
            quantile_dict(
                probabilities,
                "p",
            )
        )

        summary.update(
            quantile_dict(
                vanilla_weight,
                "vanilla_weight",
            )
        )

        summary.update(
            quantile_dict(
                luffy_weight,
                "luffy_weight",
            )
        )

        summary.update(
            quantile_dict(
                adv_over_luffy,
                "adv_over_luffy",
            )
        )

        bucket_order = [
            "p<1e-5",
            "1e-5<=p<1e-4",
            "1e-4<=p<1e-3",
            "1e-3<=p<1e-2",
            "1e-2<=p<1e-1",
            "p>=1e-1",
        ]

        bucket_counts = (
            token_df["probability_bucket"]
            .value_counts()
            .to_dict()
        )

        for bucket in bucket_order:
            count = int(
                bucket_counts.get(bucket, 0)
            )

            summary[
                f"bucket_count/{bucket}"
            ] = count

            summary[
                f"bucket_ratio/{bucket}"
            ] = count / len(token_df)

        trajectory_summaries.append(summary)

        print_sample_summary(
            summary,
            token_df,
            args.print_lowest_k,
        )

        # Free very large logits / CUDA cached intermediates
        del token_df

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not trajectory_summaries:
        raise RuntimeError(
            "No valid expert trajectories were analyzed."
        )

    token_output = (
        output_dir
        / "expert_token_probabilities.csv"
    )

    summary_output = (
        output_dir
        / "expert_trajectory_summary.csv"
    )

    pd.DataFrame(
        all_token_records
    ).to_csv(
        token_output,
        index=False,
    )

    pd.DataFrame(
        trajectory_summaries
    ).to_csv(
        summary_output,
        index=False,
    )

    # Also produce one global summary over every token.
    all_tokens_df = pd.DataFrame(
        all_token_records
    )

    global_logp = (
        all_tokens_df["log_prob"]
        .to_numpy(dtype=np.float64)
    )

    global_p = (
        all_tokens_df["probability"]
        .to_numpy(dtype=np.float64)
    )

    global_luffy = (
        all_tokens_df["luffy_weight"]
        .to_numpy(dtype=np.float64)
    )

    global_ratio = (
        all_tokens_df["adv_over_luffy"]
        .to_numpy(dtype=np.float64)
    )

    global_summary = {
        "num_trajectories": len(
            trajectory_summaries
        ),
        "num_expert_tokens": len(
            all_tokens_df
        ),
        "geometric_mean_probability": float(
            np.exp(np.mean(global_logp))
        ),
    }

    global_summary.update(
        quantile_dict(global_logp, "logp")
    )
    global_summary.update(
        quantile_dict(global_p, "p")
    )
    global_summary.update(
        quantile_dict(
            global_luffy,
            "luffy_weight",
        )
    )
    global_summary.update(
        quantile_dict(
            global_ratio,
            "adv_over_luffy",
        )
    )

    global_json = (
        output_dir
        / "expert_global_summary.json"
    )

    with open(
        global_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            global_summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 100)
    print("GLOBAL SUMMARY")
    print("=" * 100)

    print(
        f"trajectories = "
        f"{global_summary['num_trajectories']}"
    )

    print(
        f"expert tokens = "
        f"{global_summary['num_expert_tokens']}"
    )

    print(
        f"p mean   = "
        f"{global_summary['p_mean']:.6g}"
    )

    print(
        f"p median = "
        f"{global_summary['p_median']:.6g}"
    )

    print(
        f"p p10    = "
        f"{global_summary['p_p10']:.6g}"
    )

    print(
        f"p p90    = "
        f"{global_summary['p_p90']:.6g}"
    )

    print(
        f"geo-mean p = "
        f"{global_summary['geometric_mean_probability']:.6g}"
    )

    print(
        f"LUFFY weight median = "
        f"{global_summary['luffy_weight_median']:.6g}"
    )

    print(
        f"Adv/LUFFY median amplification = "
        f"{global_summary['adv_over_luffy_median']:.3f}x"
    )

    print("\nSaved:")
    print(f"  {token_output}")
    print(f"  {summary_output}")
    print(f"  {global_json}")


if __name__ == "__main__":
    main()