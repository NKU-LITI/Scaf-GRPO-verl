#!/usr/bin/env python3
# [ADD] Migrated Scaf-GRPO standalone data-preparation utility for verl 0.7.
"""Remap cached rollout records from one parquet's row ids to another by question."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_question(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--target-parquet", type=Path, required=True)
    parser.add_argument("--source-rollout-jsonl", type=Path, required=True)
    parser.add_argument("--output-rollout-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_rollout_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_rollout_jsonl} exists; pass --overwrite to replace it")

    source_df = pd.read_parquet(args.source_parquet, columns=["question"])
    target_df = pd.read_parquet(args.target_parquet, columns=["question"])
    target_index = {normalize_question(question): row_id for row_id, question in enumerate(target_df["question"])}

    written = 0
    skipped_missing_question = 0
    skipped_duplicate_target = 0
    seen_target_ids: set[int] = set()
    examples: list[dict[str, Any]] = []

    args.output_rollout_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.source_rollout_jsonl.open("r", encoding="utf-8") as src, args.output_rollout_jsonl.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            source_id = int(record["data_id"])
            question_key = normalize_question(source_df.at[source_id, "question"])
            target_id = target_index.get(question_key)
            if target_id is None:
                skipped_missing_question += 1
                continue
            if target_id in seen_target_ids:
                skipped_duplicate_target += 1
                continue
            remapped = dict(record)
            remapped["source_data_id"] = source_id
            remapped["data_id"] = target_id
            dst.write(json.dumps(remapped, ensure_ascii=False) + "\n")
            seen_target_ids.add(target_id)
            written += 1
            if len(examples) < 10:
                examples.append({"source_data_id": source_id, "target_data_id": target_id})

    report = {
        "source_parquet": str(args.source_parquet),
        "target_parquet": str(args.target_parquet),
        "source_rollout_jsonl": str(args.source_rollout_jsonl),
        "output_rollout_jsonl": str(args.output_rollout_jsonl),
        "source_rows": int(len(source_df)),
        "target_rows": int(len(target_df)),
        "written": written,
        "skipped_missing_question": skipped_missing_question,
        "skipped_duplicate_target": skipped_duplicate_target,
        "examples": examples,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
