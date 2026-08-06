#!/usr/bin/env python3
# [ADD] Migrated Scaf-GRPO standalone data-preparation utility for verl 0.7.
"""Attach teacher trajectories to a DeepScaleR parquet dataset."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("data/DeepScaleR/Qwen2.5-Math-1.5B.parquet")
DEFAULT_TEACHER = Path("data/deepscaler-teacher-sft-vllm-official-40k-clean-v2/all.jsonl")
DEFAULT_OUTPUT = Path("data/DeepScaleR/Qwen2.5-Math-1.5B.with_teacher_target.parquet")
DEFAULT_REPORT = Path("data/DeepScaleR/Qwen2.5-Math-1.5B.with_teacher_target.match_report.json")
DEFAULT_UNMATCHED = Path("data/DeepScaleR/Qwen2.5-Math-1.5B.with_teacher_target.unmatched.jsonl")


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def compact_key(value: Any) -> str:
    return re.sub(r"\s+", "", normalize_text(value))


def choose_teacher(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer verified, shorter teacher trajectories when duplicate problems exist."""
    return sorted(
        records,
        key=lambda record: (
            -float(record.get("deepscaler_official_reward") or 0),
            not bool(record.get("deepscaler_official_parseable")),
            int(record.get("text_token_count") or 10**9),
            int(record.get("source_row_index") or 10**9),
        ),
    )[0]


def load_teacher(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    exact_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    compact_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            problem = record.get("problem")
            response = record.get("response")
            if not normalize_text(problem):
                stats["missing_problem"] += 1
                continue
            if not normalize_text(response):
                stats["missing_response"] += 1
                continue
            record["_jsonl_line"] = line_number
            exact_records[normalize_text(problem)].append(record)
            compact_records[compact_key(problem)].append(record)
            stats["usable_teacher_rows"] += 1

    exact_index = {key: choose_teacher(records) for key, records in exact_records.items()}
    compact_index = {
        key: choose_teacher(records)
        for key, records in compact_records.items()
        if len({normalize_text(record.get("problem")) for record in records}) == 1
    }
    stats["exact_unique_keys"] = len(exact_index)
    stats["exact_duplicate_keys"] = sum(1 for records in exact_records.values() if len(records) > 1)
    stats["compact_unique_keys"] = len(compact_index)
    stats["compact_ambiguous_keys"] = len(compact_records) - len(compact_index)
    return exact_index, compact_index, dict(stats)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--teacher-jsonl", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--unmatched", type=Path, default=DEFAULT_UNMATCHED)
    parser.add_argument(
        "--target-column",
        default="target",
        help="Column name used to store the matched teacher response.",
    )
    parser.add_argument(
        "--matched-only",
        action="store_true",
        help="Write only rows whose question/problem text matched a teacher trajectory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    exact_index, fallback_index, teacher_stats = load_teacher(args.teacher_jsonl)
    df = pd.read_parquet(args.input)

    targets: list[str | None] = []
    match_types: list[str | None] = []
    teacher_hashes: list[str | None] = []
    teacher_source_rows: list[int | None] = []
    teacher_token_counts: list[int | None] = []
    teacher_ground_truths: list[str | None] = []
    teacher_predictions: list[str | None] = []
    teacher_rewards: list[float | None] = []
    teacher_parseable: list[bool | None] = []
    unmatched_rows: list[dict[str, Any]] = []

    for row_number, row in df.iterrows():
        question = row.get("question")
        match_type = None
        teacher = exact_index.get(normalize_text(question))
        if teacher is not None:
            match_type = "exact_norm"
        else:
            teacher = fallback_index.get(compact_key(question))
            if teacher is not None:
                match_type = "compact_norm"

        if teacher is None:
            targets.append(None)
            match_types.append(None)
            teacher_hashes.append(None)
            teacher_source_rows.append(None)
            teacher_token_counts.append(None)
            teacher_ground_truths.append(None)
            teacher_predictions.append(None)
            teacher_rewards.append(None)
            teacher_parseable.append(None)
            unmatched_rows.append(
                {
                    "row_number": int(row_number),
                    "id": row.get("id"),
                    "question": question,
                    "reward_model": row.get("reward_model"),
                    "accuracy": row.get("accuracy"),
                }
            )
            continue

        targets.append(teacher.get("response"))
        match_types.append(match_type)
        teacher_hashes.append(teacher.get("problem_hash"))
        teacher_source_rows.append(teacher.get("source_row_index"))
        teacher_token_counts.append(teacher.get("text_token_count"))
        teacher_ground_truths.append(teacher.get("ground_truth"))
        teacher_predictions.append(teacher.get("deepscaler_official_prediction"))
        teacher_rewards.append(teacher.get("deepscaler_official_reward"))
        teacher_parseable.append(teacher.get("deepscaler_official_parseable"))

    output_df = df.copy()
    output_df["qwen_original_row_index"] = range(len(output_df))
    output_df[args.target_column] = targets
    output_df["teacher_match_type"] = match_types
    output_df["teacher_problem_hash"] = teacher_hashes
    output_df["teacher_source_row_index"] = teacher_source_rows
    output_df["teacher_text_token_count"] = teacher_token_counts
    output_df["teacher_ground_truth"] = teacher_ground_truths
    output_df["teacher_prediction"] = teacher_predictions
    output_df["teacher_official_reward"] = teacher_rewards
    output_df["teacher_official_parseable"] = teacher_parseable
    if args.target_column != "target":
        output_df["target"] = targets
    if args.matched_only:
        output_df = output_df.loc[pd.Series(match_types, index=df.index).notna()].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(args.output, index=False)

    match_counts = Counter(match_type for match_type in match_types if match_type)
    target_series = output_df[args.target_column].fillna("")
    report = {
        "input": str(args.input),
        "teacher_jsonl": str(args.teacher_jsonl),
        "output": str(args.output),
        "target_column": args.target_column,
        "matched_only": bool(args.matched_only),
        "input_rows": int(len(df)),
        "output_rows": int(len(output_df)),
        "matched_rows": int(sum(bool(value) for value in targets)),
        "unmatched_rows": int(len(unmatched_rows)),
        "match_rate_over_input": float(sum(bool(value) for value in targets) / len(df)) if len(df) else 0.0,
        "match_counts": dict(match_counts),
        "target_contains_think": int(target_series.str.contains("<think>", regex=False).sum()),
        "target_contains_boxed": int(target_series.str.contains(r"\boxed", regex=False).sum()),
        "teacher_stats": teacher_stats,
        "unmatched_preview": unmatched_rows[:20],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(args.unmatched, unmatched_rows)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
