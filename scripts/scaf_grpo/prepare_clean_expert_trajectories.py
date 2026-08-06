#!/usr/bin/env python3
# [ADD] Migrated Scaf-GRPO standalone data-preparation utility for verl 0.7.
"""Create a clean expert-trajectory parquet for Scaf-GRPO training.

The source teacher responses contain ``<think>...</think>`` followed by a
complete concise answer. The training prompt does not use think tags, so this
script keeps only that post-think answer and filters samples that still exceed
the configured response length.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


DEFAULT_INPUT = Path(
    "data/DeepScaleR/chosen/"
    "Qwen2.5-Math-1.5B.with_40k_train_expert_trajectory.expert_nonempty.reward1.parquet"
)
DEFAULT_OUTPUT = Path(
    "data/DeepScaleR/chosen/"
    "Qwen2.5-Math-1.5B.with_40k_train_expert_trajectory."
    "expert_post_think.reward1.parquet"
)
DEFAULT_REPORT = Path(
    "data/DeepScaleR/chosen/"
    "Qwen2.5-Math-1.5B.with_40k_train_expert_trajectory."
    "expert_post_think.reward1.report.json"
)


def extract_post_think(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    marker = "</think>"
    if marker not in text:
        return None
    cleaned = text.split(marker, 1)[1].strip()
    return cleaned or None


def quantiles(values: pd.Series) -> dict[str, float]:
    return {
        str(percentile): float(value)
        for percentile, value in values.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--tokenizer",
        default="/workplace/nankai/liting_space/LLM/Qwen2.5-Math-1.5B",
    )
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip the training math_verify check. The default filters every cleaned target to reward=1.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists; pass --overwrite to replace it")

    df = pd.read_parquet(args.input)
    if "expert_trajectory" not in df.columns:
        raise KeyError("Input parquet has no expert_trajectory column")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    raw_targets = df["expert_trajectory"]
    cleaned_targets = raw_targets.map(extract_post_think)
    valid_format = cleaned_targets.notna()

    cleaned_token_counts = cleaned_targets.map(
        lambda text: len(tokenizer.encode(text, add_special_tokens=False)) + 1 if text is not None else 0
    )
    within_limit = cleaned_token_counts <= args.max_response_length
    keep_mask = valid_format & within_limit
    verification_failed = pd.Series(False, index=df.index)
    if not args.skip_verification:
        from verl.utils.reward_score.math_verify import compute_score

        for row_index in df.index[keep_mask]:
            ground_truth = df.at[row_index, "reward_model"]["ground_truth"]
            score = float(compute_score(cleaned_targets.at[row_index], ground_truth))
            verification_failed.at[row_index] = score != 1.0
        keep_mask &= ~verification_failed

    output_df = df.loc[keep_mask].copy()
    output_df["expert_trajectory_raw"] = raw_targets.loc[keep_mask]
    output_df["expert_trajectory"] = cleaned_targets.loc[keep_mask]
    output_df["expert_trajectory_token_count"] = cleaned_token_counts.loc[keep_mask].astype(int)
    output_df["expert_trajectory_format"] = "post_think"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(args.output, index=False)

    kept_lengths = output_df["expert_trajectory_token_count"]
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "tokenizer": args.tokenizer,
        "max_response_length": args.max_response_length,
        "rows_before": int(len(df)),
        "rows_without_post_think": int((~valid_format).sum()),
        "rows_over_max_response_length": int((valid_format & ~within_limit).sum()),
        "rows_failed_math_verify": int(verification_failed.sum()),
        "rows_after": int(len(output_df)),
        "token_length_quantiles": quantiles(kept_lengths),
        "token_length_mean": float(kept_lengths.mean()),
        "token_length_std": float(kept_lengths.std()),
        "contains_boxed": int(output_df["expert_trajectory"].str.contains(r"\\boxed", regex=True).sum()),
        "contains_think_tag": int(output_df["expert_trajectory"].str.contains("<think>", regex=False).sum()),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
