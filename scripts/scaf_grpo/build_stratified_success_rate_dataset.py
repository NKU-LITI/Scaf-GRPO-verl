#!/usr/bin/env python3
# [ADD] Migrated Scaf-GRPO rollout-based stratified dataset builder for the verl 0.7 tree.
"""Build a stratified training set and sample validation data from that training set."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

# [DEL] from hint_mix_grpo.data_rewrite import Generator, get_ground_truth, normalize_messages, render_prompt
# [ADD] Use the migrated local data rewrite backend instead of the removed hint_mix_grpo package.
from scripts.scaf_grpo.data_rewrite import Generator, get_ground_truth, normalize_messages, render_prompt
from verl.utils.reward_score import default_compute_score


LOGGER = logging.getLogger("build_stratified_success_rate_dataset")


BUCKETS = ("hard", "medium", "easy")


def parse_quota(value: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        key, raw_count = item.split("=", 1)
        key = key.strip()
        if key not in BUCKETS:
            raise argparse.ArgumentTypeError(f"Unknown bucket {key!r}; expected one of {BUCKETS}")
        parsed[key] = int(raw_count)
    missing = [key for key in BUCKETS if key not in parsed]
    if missing:
        raise argparse.ArgumentTypeError(f"Missing quota(s): {missing}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input parquet with prompts and expert trajectories.")
    parser.add_argument("--train-output", required=True, help="Output train parquet.")
    parser.add_argument("--val-output", required=True, help="Output validation parquet sampled from the selected training set.")
    parser.add_argument("--model_path", required=True, help="Initial policy model path.")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--ground_truth_key", default="ground_truth")
    parser.add_argument("--k", type=int, default=8, help="Number of sampled rollouts per problem.")
    parser.add_argument("--train-quotas", type=parse_quota, default=parse_quota("hard=400,medium=200,easy=200"))
    parser.add_argument("--val-quotas", type=parse_quota, default=parse_quota("hard=100,medium=50,easy=50"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_scan_samples", type=int, default=None)
    parser.add_argument("--rollout-jsonl", default=None, help="Path to cache per-problem rollout scores.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse existing rollout-jsonl records before generating more.")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--max_model_len", type=int, default=6144)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def bucket_for_success_rate(success_rate: float) -> str:
    if success_rate <= 0:
        return "hard"
    if success_rate < 0.5:
        return "medium"
    return "easy"


def has_enough(selected: dict[str, list[int]], quotas: dict[str, int]) -> bool:
    return all(len(selected[key]) >= quotas[key] for key in BUCKETS)


def score_candidates(
    candidates: list[str],
    ground_truth: Any,
    *,
    data_source: str,
    extra_info: dict[str, Any] | None,
    row_id: int,
) -> list[float]:
    rewards: list[float] = []
    for rank, candidate in enumerate(candidates):
        text = candidate.strip()
        if not text:
            rewards.append(0.0)
            continue
        try:
            reward = default_compute_score(
                data_source=data_source,
                solution_str=text,
                ground_truth=ground_truth,
                extra_info=extra_info or {},
            )
        except Exception:
            LOGGER.exception("Verifier failed for row_id=%s rank=%d", row_id, rank)
            reward = 0.0
        rewards.append(float(reward))
    return rewards


def write_jsonl_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def load_cached_rollouts(path: Path, k: int) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            row_id = int(record["data_id"])
            if len(record.get("reward_list", [])) >= k:
                records[row_id] = record
    LOGGER.info("Loaded %d cached rollout records from %s", len(records), path)
    return records


def add_record(
    record: dict[str, Any],
    *,
    selected: dict[str, list[int]],
    quotas: dict[str, int],
) -> bool:
    row_id = int(record["data_id"])
    bucket = str(record["difficulty_bucket"])
    if len(selected[bucket]) >= quotas[bucket]:
        return False
    selected[bucket].append(row_id)
    LOGGER.info(
        "Selected %-6s %d/%d row=%s success_rate@%s=%.3f",
        bucket,
        len(selected[bucket]),
        quotas[bucket],
        row_id,
        record["k"],
        float(record["success_rate_at_k"]),
    )
    return True


def build_output_frame(df: pd.DataFrame, row_ids: list[int], records: dict[int, dict[str, Any]]) -> pd.DataFrame:
    output_df = df.iloc[row_ids].copy().reset_index(drop=True)
    output_df["source_row_index"] = row_ids
    output_df["difficulty_bucket"] = [records[row_id]["difficulty_bucket"] for row_id in row_ids]
    output_df["success_rate_at_k"] = [records[row_id]["success_rate_at_k"] for row_id in row_ids]
    output_df["success_count_at_k"] = [records[row_id]["success_count_at_k"] for row_id in row_ids]
    output_df["rollout_reward_list"] = [records[row_id]["reward_list"] for row_id in row_ids]
    return output_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    train_output = Path(args.train_output)
    val_output = Path(args.val_output)
    for path in (train_output, val_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    rollout_path = Path(args.rollout_jsonl) if args.rollout_jsonl else train_output.with_suffix(".success_rate_rollouts.jsonl")
    if rollout_path.exists() and not args.resume and args.overwrite:
        rollout_path.unlink()

    for bucket in BUCKETS:
        if args.val_quotas[bucket] > args.train_quotas[bucket]:
            raise ValueError(
                f"Validation quota for {bucket!r} ({args.val_quotas[bucket]}) exceeds "
                f"the training quota ({args.train_quotas[bucket]}). Validation samples "
                "must be drawn from the selected training set."
            )

    selected: dict[str, list[int]] = {key: [] for key in BUCKETS}
    rollout_records = load_cached_rollouts(rollout_path, args.k) if args.resume else {}

    df = pd.read_parquet(args.input)
    rows = [row.to_dict() for _, row in df.iterrows()]
    order = list(range(len(df)))
    random.Random(args.seed).shuffle(order)
    if args.max_scan_samples is not None:
        order = order[: args.max_scan_samples]
    LOGGER.info("Loaded %d rows from %s; scanning up to %d rows", len(df), args.input, len(order))

    for row_id in order:
        record = rollout_records.get(row_id)
        if record is not None:
            add_record(record, selected=selected, quotas=args.train_quotas)
            if has_enough(selected, args.train_quotas):
                break

    tokenizer = None
    generator = None
    scanned = 0
    try:
        if not has_enough(selected, args.train_quotas):
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_path or args.model_path,
                trust_remote_code=args.trust_remote_code,
                padding_side="left",
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            generator = Generator(args, tokenizer)

        remaining_order = [row_id for row_id in order if row_id not in rollout_records]
        for start in range(0, len(remaining_order), args.batch_size):
            if has_enough(selected, args.train_quotas):
                break
            assert tokenizer is not None and generator is not None
            batch_indices = remaining_order[start : start + args.batch_size]
            batch_rows = [rows[row_id] for row_id in batch_indices]
            prompts = [render_prompt(tokenizer, normalize_messages(row[args.prompt_key])) for row in batch_rows]
            ground_truths = [get_ground_truth(row, args.ground_truth_key) for row in batch_rows]
            batch_outputs = generator.generate(prompts, n=args.k, temperature=args.temperature)

            for row_id, prompt, outputs, ground_truth in zip(batch_indices, prompts, batch_outputs, ground_truths):
                row = rows[row_id]
                rewards = score_candidates(
                    outputs[: args.k],
                    ground_truth,
                    data_source=row["data_source"],
                    extra_info=row.get("extra_info"),
                    row_id=row_id,
                )
                success_count = sum(1 for reward in rewards if reward > 0)
                success_rate = success_count / args.k
                record = {
                    "data_id": row_id,
                    "k": args.k,
                    "difficulty_bucket": bucket_for_success_rate(success_rate),
                    "success_count_at_k": success_count,
                    "success_rate_at_k": success_rate,
                    "reward_list": rewards,
                    "input_list": [prompt] * len(outputs),
                    "output_list": outputs,
                    "ground_truth": ground_truth,
                }
                rollout_records[row_id] = record
                write_jsonl_line(rollout_path, record)
                scanned += 1
                add_record(record, selected=selected, quotas=args.train_quotas)
    finally:
        if generator is not None:
            generator.close()

    missing = {
        key: args.train_quotas[key] - len(selected[key])
        for key in BUCKETS
        if len(selected[key]) < args.train_quotas[key]
    }
    if missing:
        raise RuntimeError(f"Not enough samples for requested quotas after scanning {len(order)} rows: {missing}")

    # Keep all selected samples in the training set. The validation set is an
    # intentionally overlapping, stratified random subset of that training set.
    train_ids: list[int] = []
    val_ids: list[int] = []
    val_rng = random.Random(args.seed + 1)
    for bucket in BUCKETS:
        bucket_ids = list(selected[bucket])
        train_ids.extend(bucket_ids)
        val_ids.extend(val_rng.sample(bucket_ids, args.val_quotas[bucket]))

    if len(train_ids) != len(set(train_ids)):
        raise RuntimeError("Duplicate source rows detected in the selected training set.")
    if len(val_ids) != len(set(val_ids)):
        raise RuntimeError("Duplicate source rows detected in the validation subset.")
    if not set(val_ids).issubset(set(train_ids)):
        raise RuntimeError("Validation rows must be a subset of the selected training rows.")

    train_df = build_output_frame(df, train_ids, rollout_records)
    val_df = build_output_frame(df, val_ids, rollout_records)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_output, index=False)
    val_df.to_parquet(val_output, index=False)

    report_path = train_output.with_suffix(train_output.suffix + ".report.json")
    report = {
        "input": args.input,
        "train_output": str(train_output),
        "val_output": str(val_output),
        "rollout_jsonl": str(rollout_path),
        "model_path": args.model_path,
        "k": args.k,
        "seed": args.seed,
        "train_quotas": args.train_quotas,
        "val_quotas": args.val_quotas,
        "generated_new_rollouts": scanned,
        "selected_total_counts": {key: len(selected[key]) for key in BUCKETS},
        "validation_is_subset_of_training": True,
        "validation_sampling_seed": args.seed + 1,
        "train_counts": dict(Counter(train_df["difficulty_bucket"])),
        "val_counts": dict(Counter(val_df["difficulty_bucket"])),
        "train_source_row_indices": train_ids,
        "val_source_row_indices": val_ids,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote train split: %s (%d rows)", train_output, len(train_df))
    LOGGER.info("Wrote validation subset: %s (%d rows, sampled from training)", val_output, len(val_df))
    LOGGER.info("Wrote report: %s", report_path)


if __name__ == "__main__":
    main()
