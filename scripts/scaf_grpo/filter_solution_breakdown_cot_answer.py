#!/usr/bin/env python3
# [ADD] Migrated Scaf-GRPO standalone data-preparation utility for verl 0.7.
"""Filter rows whose solution_breakdown_parts form a verified CoT answer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from verl.utils.reward_score.math_verify import compute_score


def normalize_parts(value: object) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if part is not None and str(part).strip()]
    text = str(value).strip()
    return [text] if text else []


def join_parts(value: object) -> str:
    return "\n\n".join(normalize_parts(value)).strip()


def get_ground_truth(row: pd.Series) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict) and extra_info.get("gt_answer") is not None:
        return str(extra_info["gt_answer"])
    return ""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--target-column", default="solution_breakdown_cot_answer")
    parser.add_argument(
        "--require-boxed",
        action="store_true",
        help="Require the concatenated solution_breakdown_parts to contain \\boxed before verification.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    df = pd.read_parquet(args.input)
    if "solution_breakdown_parts" not in df.columns:
        raise KeyError("Input parquet has no 'solution_breakdown_parts' column")

    joined = df["solution_breakdown_parts"].map(join_parts)
    part_counts = df["solution_breakdown_parts"].map(lambda value: len(normalize_parts(value)))
    nonempty = joined.map(bool)
    contains_boxed = joined.str.contains(r"\boxed", regex=False)

    rewards: list[float] = []
    rejected_rows: list[dict[str, Any]] = []
    for row_index, row in df.iterrows():
        text = joined.at[row_index]
        ground_truth = get_ground_truth(row)
        if not text or (args.require_boxed and "\\boxed" not in text):
            reward = 0.0
        else:
            try:
                reward = float(compute_score(text, ground_truth))
            except Exception:
                reward = 0.0
        rewards.append(reward)
        if reward != 1.0:
            rejected_rows.append(
                {
                    "row_number": int(row_index),
                    "id": row.get("id"),
                    "ground_truth": ground_truth,
                    "parts_count": int(part_counts.at[row_index]),
                    "parts_nonempty": bool(nonempty.at[row_index]),
                    "contains_boxed": bool(contains_boxed.at[row_index]),
                    "reward": reward,
                    "solution_breakdown_prefix": text[:500],
                }
            )

    reward_series = pd.Series(rewards, index=df.index)
    keep_mask = nonempty & (reward_series == 1.0)
    if args.require_boxed:
        keep_mask &= contains_boxed

    output_df = df.loc[keep_mask].copy()
    output_df[args.target_column] = joined.loc[keep_mask]
    output_df["solution_breakdown_parts_count"] = part_counts.loc[keep_mask].astype(int)
    output_df["solution_breakdown_reward"] = reward_series.loc[keep_mask].astype(float)
    output_df["qwen_original_row_index"] = output_df.index.astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(args.output, index=False)
    write_jsonl(args.rejected, rejected_rows)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "target_column": args.target_column,
        "require_boxed": bool(args.require_boxed),
        "input_rows": int(len(df)),
        "output_rows": int(len(output_df)),
        "parts_nonempty": int(nonempty.sum()),
        "parts_empty": int((~nonempty).sum()),
        "parts_count_distribution": {
            str(key): int(value) for key, value in part_counts.value_counts(dropna=False).sort_index().items()
        },
        "concat_contains_boxed": int(contains_boxed.sum()),
        "concat_missing_boxed": int((~contains_boxed).sum()),
        "verified_reward1": int((reward_series == 1.0).sum()),
        "rejected_rows": int(len(rejected_rows)),
        "rejected_path": str(args.rejected),
        "rejected_preview": rejected_rows[:20],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
