#!/usr/bin/env python3
# [ADD] Migrated Scaf-GRPO initial-wrong validation generator for the verl 0.7 tree.
"""Generate and save a fixed initial-wrong validation set.

This runs the initial policy on shuffled training examples, samples N responses
per problem, scores them with the same math verifier used by training, and
writes the first target-size problems whose sampled pass@N is zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

# [DEL] from hint_mix_grpo.data_rewrite import Generator, get_ground_truth, normalize_messages, render_prompt
# [ADD] Use the migrated local data rewrite backend instead of the removed hint_mix_grpo package.
from scripts.scaf_grpo.data_rewrite import Generator, get_ground_truth, normalize_messages, render_prompt
from verl.utils.reward_score import default_compute_score


LOGGER = logging.getLogger("generate_initial_wrong_validation_set")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Training parquet to sample from.")
    parser.add_argument("--output", required=True, help="Output validation parquet.")
    parser.add_argument("--model_path", required=True, help="Initial policy model path.")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--ground_truth_key", default="ground_truth")
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--required-rollouts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_scan_samples", type=int, default=None)
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


def write_jsonl_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def score_candidates(
    candidates: list[str],
    ground_truth: Any,
    *,
    data_source: str,
    extra_info: dict[str, Any] | None,
    row_id: int,
) -> list[float]:
    rewards = []
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    df = pd.read_parquet(args.input)
    order = list(range(len(df)))
    random.Random(args.seed).shuffle(order)
    if args.max_scan_samples is not None:
        order = order[: args.max_scan_samples]
    LOGGER.info("Loaded %d rows from %s; scanning %d rows", len(df), args.input, len(order))

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model_path,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_path = Path(args.output)
    rollout_path = output_path.with_suffix(output_path.suffix + ".rollouts.jsonl")
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    if rollout_path.exists():
        rollout_path.unlink()

    rows = [row.to_dict() for _, row in df.iterrows()]
    selected_indices: list[int] = []
    scanned = 0

    generator = Generator(args, tokenizer)
    try:
        for start in range(0, len(order), args.batch_size):
            batch_indices = order[start : start + args.batch_size]
            batch_rows = [rows[idx] for idx in batch_indices]
            prompts = [render_prompt(tokenizer, normalize_messages(row[args.prompt_key])) for row in batch_rows]
            ground_truths = [get_ground_truth(row, args.ground_truth_key) for row in batch_rows]
            batch_outputs = generator.generate(prompts, n=args.required_rollouts, temperature=args.temperature)

            for idx, prompt, outputs, ground_truth in zip(batch_indices, prompts, batch_outputs, ground_truths):
                row = rows[idx]
                rewards = score_candidates(
                    outputs,
                    ground_truth,
                    data_source=row["data_source"],
                    extra_info=row.get("extra_info"),
                    row_id=idx,
                )
                scanned += 1
                record = {
                    "data_id": idx,
                    "reward_list": rewards,
                    "n_rollout": args.required_rollouts,
                    "input_list": [prompt] * len(outputs),
                    "output_list": outputs,
                    "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
                    "reward_sum": sum(rewards),
                    "is_all_rollout_wrong": bool(rewards and all(score <= 0 for score in rewards)),
                    "ground_truth": ground_truth,
                }
                write_jsonl_line(rollout_path, record)
                if record["is_all_rollout_wrong"]:
                    selected_indices.append(idx)
                    LOGGER.info(
                        "Selected %d/%d: row=%s reward_sum=0 scanned=%d",
                        len(selected_indices),
                        args.target_size,
                        idx,
                        scanned,
                    )
                    if len(selected_indices) >= args.target_size:
                        break
            if len(selected_indices) >= args.target_size:
                break
    finally:
        generator.close()

    if len(selected_indices) < args.target_size:
        raise RuntimeError(
            f"Only selected {len(selected_indices)} initial-wrong samples after scanning {scanned} rows."
        )

    output_df = df.iloc[selected_indices].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input": args.input,
                "output": str(output_path),
                "rollout_jsonl": str(rollout_path),
                "target_size": args.target_size,
                "required_rollouts": args.required_rollouts,
                "seed": args.seed,
                "scanned": scanned,
                "selected_indices": selected_indices,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    LOGGER.info("Wrote %d samples to %s", len(output_df), output_path)
    LOGGER.info("Wrote rollouts to %s", rollout_path)
    LOGGER.info("Wrote report to %s", report_path)


if __name__ == "__main__":
    main()
