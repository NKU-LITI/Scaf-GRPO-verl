#!/usr/bin/env python3
# [ADD] Migrated Scaf-GRPO standalone data-preparation utility for verl 0.7.
"""Add a post-think expert trajectory field to parquet datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def extract_after_think(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    marker = "</think>"
    if marker not in value:
        return None
    text = value.split(marker, 1)[1].strip()
    return text or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--source-column", default="expert_trajectory")
    parser.add_argument("--target-column", default="short_expert_tracjectory")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    reports = []
    for path in args.paths:
        df = pd.read_parquet(path)
        if args.source_column not in df.columns:
            raise KeyError(f"{path} has no {args.source_column!r} column")

        short_values = df[args.source_column].map(extract_after_think)
        output_df = df.copy()
        output_df[args.target_column] = short_values
        output_df.to_parquet(path, index=False)

        nonempty = short_values.fillna("").map(lambda value: bool(str(value).strip()))
        contains_think = short_values.fillna("").str.contains("<think>", regex=False)
        contains_boxed = short_values.fillna("").str.contains(r"\boxed", regex=False)
        reports.append(
            {
                "path": str(path),
                "rows": int(len(output_df)),
                "source_column": args.source_column,
                "target_column": args.target_column,
                "target_nonempty": int(nonempty.sum()),
                "missing_think_close": int((~nonempty).sum()),
                "target_contains_think": int(contains_think.sum()),
                "target_contains_boxed": int(contains_boxed.sum()),
            }
        )

    report_text = json.dumps(reports, ensure_ascii=False, indent=2)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text + "\n", encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
