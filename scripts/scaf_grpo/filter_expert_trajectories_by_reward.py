#!/usr/bin/env python3
# [ADD] Migrated Scaf-GRPO standalone data-preparation utility for verl 0.7.
"""Filter matched expert trajectories with the training math verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from verl.utils.reward_score.math_verify import compute_score


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
    parser.add_argument("--failed", type=Path, required=True)
    parser.add_argument("--trajectory-column", default="expert_trajectory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    df = pd.read_parquet(args.input)
    if args.trajectory_column not in df.columns:
        raise KeyError(f"Input parquet has no {args.trajectory_column!r} column")

    rewards: list[float] = []
    failed_rows: list[dict[str, Any]] = []
    for row_number, row in df.iterrows():
        prediction = row.get(args.trajectory_column)
        ground_truth = get_ground_truth(row)
        try:
            reward = float(compute_score(prediction, ground_truth))
        except Exception as exc:  # noqa: BLE001 - keep auditing moving and report failures.
            reward = 0.0
            error = repr(exc)
        else:
            error = None
        rewards.append(reward)
        if reward != 1.0:
            failed_rows.append(
                {
                    "row_number": int(row_number),
                    "qwen_original_row_index": row.get("qwen_original_row_index"),
                    "id": row.get("id"),
                    "question": row.get("question"),
                    "ground_truth": ground_truth,
                    "teacher_ground_truth": row.get("teacher_ground_truth"),
                    "teacher_prediction": row.get("teacher_prediction"),
                    "reward": reward,
                    "error": error,
                }
            )

    output_df = df.copy()
    output_df["expert_trajectory_reward"] = rewards
    keep_mask = output_df["expert_trajectory_reward"] == 1.0
    output_df = output_df.loc[keep_mask].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(args.output, index=False)
    write_jsonl(args.failed, failed_rows)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "trajectory_column": args.trajectory_column,
        "rows_before": int(len(df)),
        "rows_after": int(len(output_df)),
        "rows_failed_reward": int(len(failed_rows)),
        "failed_path": str(args.failed),
        "failed_preview": failed_rows[:20],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
