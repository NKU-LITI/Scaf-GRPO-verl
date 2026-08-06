#!/usr/bin/env python3
# [ADD] Migrated Scaf-GRPO standalone data-preparation utility for verl 0.7.
"""Sample stratified splits from cached success_rate records with optional accuracy fallback."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


BUCKETS = ("hard", "medium", "easy")


def parse_quota(value: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        key, raw_count = item.split("=", 1)
        key = key.strip()
        if key not in BUCKETS:
            raise argparse.ArgumentTypeError(f"Unknown bucket {key!r}")
        parsed[key] = int(raw_count)
    missing = [key for key in BUCKETS if key not in parsed]
    if missing:
        raise argparse.ArgumentTypeError(f"Missing quota(s): {missing}")
    return parsed


def bucket_for_success_rate(success_rate: float) -> str:
    if success_rate <= 0:
        return "hard"
    if success_rate < 0.5:
        return "medium"
    return "easy"


def load_rollouts(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record["data_id"])] = record
    return records


def build_frame(df: pd.DataFrame, row_ids: list[int], records: dict[int, dict[str, Any]], k: int) -> pd.DataFrame:
    output_df = df.iloc[row_ids].copy().reset_index(drop=True)
    output_df["source_row_index"] = row_ids
    buckets: list[str] = []
    rates: list[float] = []
    counts: list[int] = []
    reward_lists: list[list[float] | None] = []
    sources: list[str] = []
    for row_id in row_ids:
        record = records[row_id]
        buckets.append(str(record["difficulty_bucket"]))
        rates.append(float(record["success_rate_at_k"]))
        counts.append(int(record["success_count_at_k"]))
        reward_lists.append(record.get("reward_list"))
        sources.append(str(record["success_rate_source"]))
    output_df["difficulty_bucket"] = buckets
    output_df["success_rate_at_k"] = rates
    output_df["success_count_at_k"] = counts
    output_df["rollout_reward_list"] = reward_lists
    output_df["success_rate_source"] = sources
    output_df["success_rate_k"] = k
    return output_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--remaining-train-output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-quotas", type=parse_quota, default=parse_quota("hard=400,medium=200,easy=200"))
    parser.add_argument("--val-quotas", type=parse_quota, default=parse_quota("hard=100,medium=50,easy=50"))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-accuracy-fallback", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = [args.train_output, args.val_output]
    if args.remaining_train_output is not None:
        outputs.append(args.remaining_train_output)
    for path in outputs:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    df = pd.read_parquet(args.input)
    rollout_records = load_rollouts(args.rollout_jsonl)

    records: dict[int, dict[str, Any]] = {}
    for row_id, record in rollout_records.items():
        if row_id >= len(df):
            continue
        enriched = dict(record)
        enriched["success_rate_source"] = "rollout_cache"
        records[row_id] = enriched

    if args.allow_accuracy_fallback:
        for row_id, row in df.iterrows():
            if row_id in records:
                continue
            success_rate = float(row.get("accuracy") or 0.0)
            records[int(row_id)] = {
                "data_id": int(row_id),
                "k": args.k,
                "difficulty_bucket": bucket_for_success_rate(success_rate),
                "success_rate_at_k": success_rate,
                "success_count_at_k": int(round(success_rate * args.k)),
                "reward_list": None,
                "success_rate_source": "accuracy_fallback",
            }

    by_bucket: dict[str, list[int]] = defaultdict(list)
    for row_id, record in records.items():
        by_bucket[str(record["difficulty_bucket"])].append(row_id)

    rng = random.Random(args.seed)
    selected_train_by_bucket: dict[str, list[int]] = {}
    selected_val_by_bucket: dict[str, list[int]] = {}
    remaining_by_bucket: dict[str, list[int]] = {}
    for bucket in BUCKETS:
        cached = [row_id for row_id in by_bucket[bucket] if records[row_id]["success_rate_source"] == "rollout_cache"]
        fallback = [row_id for row_id in by_bucket[bucket] if records[row_id]["success_rate_source"] != "rollout_cache"]
        rng.shuffle(cached)
        rng.shuffle(fallback)
        ordered = cached + fallback
        need = args.train_quotas[bucket]
        if len(ordered) < need:
            raise RuntimeError(f"Not enough {bucket} rows: need {need}, have {len(ordered)}")
        train_bucket_ids = ordered[:need]
        selected_train_by_bucket[bucket] = train_bucket_ids

        val_need = args.val_quotas[bucket]
        if len(train_bucket_ids) < val_need:
            raise RuntimeError(f"Not enough selected {bucket} rows for val: need {val_need}, have {len(train_bucket_ids)}")
        val_candidates = train_bucket_ids.copy()
        rng.shuffle(val_candidates)
        val_ids = val_candidates[:val_need]
        selected_val_by_bucket[bucket] = val_ids
        remaining_by_bucket[bucket] = [row_id for row_id in train_bucket_ids if row_id not in set(val_ids)]

    train_ids = [row_id for bucket in BUCKETS for row_id in selected_train_by_bucket[bucket]]
    val_ids = [row_id for bucket in BUCKETS for row_id in selected_val_by_bucket[bucket]]
    remaining_ids = [row_id for bucket in BUCKETS for row_id in remaining_by_bucket[bucket]]

    train_df = build_frame(df, train_ids, records, args.k)
    val_df = build_frame(df, val_ids, records, args.k)

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(args.train_output, index=False)
    val_df.to_parquet(args.val_output, index=False)
    remaining_df = None
    if args.remaining_train_output is not None:
        remaining_df = build_frame(df, remaining_ids, records, args.k)
        remaining_df.to_parquet(args.remaining_train_output, index=False)

    report = {
        "input": str(args.input),
        "rollout_jsonl": str(args.rollout_jsonl),
        "train_output": str(args.train_output),
        "val_output": str(args.val_output),
        "remaining_train_output": str(args.remaining_train_output) if args.remaining_train_output else None,
        "seed": args.seed,
        "k": args.k,
        "allow_accuracy_fallback": bool(args.allow_accuracy_fallback),
        "available_counts": {bucket: len(by_bucket[bucket]) for bucket in BUCKETS},
        "available_source_counts": dict(Counter(record["success_rate_source"] for record in records.values())),
        "available_bucket_source_counts": {
            bucket: dict(Counter(records[row_id]["success_rate_source"] for row_id in by_bucket[bucket]))
            for bucket in BUCKETS
        },
        "train_counts": dict(Counter(train_df["difficulty_bucket"])),
        "train_source_counts": dict(Counter(train_df["success_rate_source"])),
        "val_counts": dict(Counter(val_df["difficulty_bucket"])),
        "val_source_counts": dict(Counter(val_df["success_rate_source"])),
        "val_is_subset_of_train": set(val_ids).issubset(set(train_ids)),
        "remaining_train_counts": dict(Counter(remaining_df["difficulty_bucket"])) if remaining_df is not None else None,
        "remaining_train_source_counts": dict(Counter(remaining_df["success_rate_source"])) if remaining_df is not None else None,
        "train_source_row_indices": train_ids,
        "val_source_row_indices": val_ids,
        "remaining_train_source_row_indices": remaining_ids,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
