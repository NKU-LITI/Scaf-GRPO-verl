#!/usr/bin/env python3
# [ADD] Migrated Scaf-GRPO standalone data-preparation utility for verl 0.7.
"""Build a fixed validation parquet from initial-model all-wrong rollout logs."""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_question(input_text: str) -> str:
    match = re.search(r"user\n(.*?)\nassistant\n", input_text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return input_text.strip()


def iter_jsonl(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--rollout-jsonl", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--required-rollouts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_df = pd.read_parquet(args.train_parquet)
    question_to_index = {}
    for idx, question in enumerate(train_df["question"].tolist()):
        question_to_index.setdefault(normalize_text(question), idx)

    selected_indices = []
    seen = set()
    for record in iter_jsonl([Path(p) for p in args.rollout_jsonl]):
        rewards = record.get("reward_first", record.get("reward_list", []))
        if len(rewards) < args.required_rollouts:
            continue
        rewards = rewards[: args.required_rollouts]
        if any(float(score) > 0 for score in rewards):
            continue
        input_list = record.get("input_list") or []
        if not input_list:
            continue
        question = normalize_text(extract_question(input_list[0]))
        idx = question_to_index.get(question)
        if idx is None or idx in seen:
            continue
        selected_indices.append(idx)
        seen.add(idx)

    if len(selected_indices) < args.target_size:
        raise RuntimeError(
            f"Only found {len(selected_indices)} all-wrong samples; "
            f"target_size={args.target_size}."
        )

    rng = random.Random(args.seed)
    rng.shuffle(selected_indices)
    selected_indices = selected_indices[: args.target_size]
    output_df = train_df.iloc[selected_indices].reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_parquet": args.train_parquet,
                "rollout_jsonl": args.rollout_jsonl,
                "output": str(output_path),
                "target_size": args.target_size,
                "required_rollouts": args.required_rollouts,
                "seed": args.seed,
                "selected_indices": selected_indices,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote {len(output_df)} samples to {output_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
