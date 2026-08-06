#!/usr/bin/env python3
# [ADD] Migrated Scaf-GRPO Mind-the-Gap data rewrite utility for the verl 0.7 tree.
"""Rewrite LUFFY offline expert trajectories with the Mind the Gap procedure.

The output keeps LUFFY's parquet schema: ``prompt`` remains the original prompt
and ``target`` becomes the selected on-policy / rewritten / fallback solution.
Optionally, token probabilities under the target model are stored in
``target_ds_qwen_7b_probs`` for LUFFY's off-policy importance correction.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("pandas is required. Run this inside the LUFFY training environment.") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = lambda x, **_: x

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("transformers is required. Run this inside the LUFFY training environment.") from exc


LOGGER = logging.getLogger("mind_the_gap_rewrite")
THINK_PREFIX = "<think>\n"
THINK_SUFFIX = "</think>"
GUIDED_REWRITE_CHECKPOINT_STAGE = "guided_rewrite_mtg_digest_retell_v1"
MTG_DIGEST_RETELL_PROMPT = """You are given a math problem. If you cannot solve it directly, you are also given a teacher's detailed solution and final answer. Read and learn from it, then try to solve the problem again in your own way, not by copying.

Instructions:
1. First try to understand the problem.
2. Use the teacher's solution only as guidance if you get stuck.
3. Explain the reasoning in your own words, step by step.
4. Conclude with the correct final answer.

Problem:
{problem}

Teacher's Solution (for reference):
{teacher_solution}

Now, solve the problem in your own way:"""


@dataclass
class GeneratedSolution:
    text: str
    is_correct: bool
    rank: int


_VERIFY_FN = None


def _init_verify_worker(think_format: bool) -> None:
    global _VERIFY_FN
    _VERIFY_FN = _load_verifier(think_format=think_format)


def _verify_candidate_worker(text: str, ground_truth: Any) -> bool:
    if _VERIFY_FN is None:
        raise RuntimeError("Verifier worker was not initialized.")
    return bool(_VERIFY_FN(text, ground_truth, enable_llm=False))


class VerifierRunner:
    def __init__(self, think_format: bool, timeout_seconds: float | None):
        self.think_format = think_format
        self.timeout_seconds = timeout_seconds
        self.direct_verify_fn = None
        self.ctx = mp.get_context("spawn")
        self.pool: mp.pool.Pool | None = None
        if not timeout_seconds or timeout_seconds <= 0:
            self.direct_verify_fn = _load_verifier(think_format=think_format)
        else:
            self._start_pool()

    def _start_pool(self) -> None:
        self.pool = self.ctx.Pool(
            processes=1,
            initializer=_init_verify_worker,
            initargs=(self.think_format,),
        )

    def _restart_pool(self) -> None:
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
        self._start_pool()

    def verify(self, text: str, ground_truth: Any, row_id: int | None, rank: int, stage: str | None) -> bool:
        if self.direct_verify_fn is not None:
            return bool(self.direct_verify_fn(text, ground_truth, enable_llm=False))
        if self.pool is None:
            self._start_pool()
        assert self.pool is not None
        result = self.pool.apply_async(_verify_candidate_worker, (text, ground_truth))
        try:
            return bool(result.get(timeout=self.timeout_seconds))
        except mp.TimeoutError:
            LOGGER.warning(
                "Verifier timed out for stage=%s row_id=%s rank=%d after %.1fs; restarting verifier worker",
                stage,
                row_id,
                rank,
                self.timeout_seconds,
            )
            self._restart_pool()
            return False
        except Exception:
            LOGGER.exception("Verifier failed for stage=%s row_id=%s rank=%d", stage, row_id, rank)
            return False

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


def _add_repo_paths() -> None:
    # [DEL] Original Scaf-GRPO path assumed this file lived under `hint_mix_grpo/`.
    # repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # verl_root = os.path.join(repo_root, "luffy", "verl")
    # for path in (repo_root, verl_root):
    # [ADD] In the migrated tree this file lives in `scripts/scaf_grpo/`, so the repository
    # root is two levels up and contains the importable `verl` package directly.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for path in (repo_root,):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_verifier(think_format: bool):
    _add_repo_paths()
    try:
        if think_format:
            from verl.mix_src.math_verify_reward import reward_fn_math_verify as verify_fn
        else:
            from verl.mix_src.math_verify_reward import reward_fn_math_verify_no_think as verify_fn
    except Exception as exc:
        raise SystemExit(
            "Failed to import LUFFY math verifier. Ensure PYTHONPATH includes luffy/verl "
            "and math_verify/deepscaler dependencies are installed."
        ) from exc
    return verify_fn


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [dict(item) for item in value]
    raise TypeError(f"Expected prompt as list[dict], got {type(value)!r}")


def normalize_target_content(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        return value.get("content")
    if isinstance(value, str):
        return value
    return None


def get_ground_truth(row: Any, key: str) -> Any:
    value = row.get(key)
    if value is not None:
        return value
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth")
    if hasattr(reward_model, "get"):
        return reward_model.get("ground_truth")
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        return extra_info.get("answer")
    return None


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_guided_messages(
    messages: list[dict[str, str]],
    reference_solution: str | None,
    ground_truth: Any,
    reveal_answer: bool,
) -> list[dict[str, str]]:
    reference_solution = (reference_solution or "").strip()
    teacher_solution = reference_solution
    if reveal_answer and ground_truth is not None:
        teacher_solution = f"{teacher_solution}\n\nFinal answer: {ground_truth}".strip()
    prompt = MTG_DIGEST_RETELL_PROMPT.format(
        problem=extract_user_prompt(messages),
        teacher_solution=teacher_solution or "The correct final answer is unknown; solve the problem carefully.",
    )
    system_messages = [message for message in messages if message.get("role") == "system"]
    return [*system_messages, {"role": "user", "content": prompt}]


def truncate_text_to_tokens(tokenizer: Any, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=False)


def render_guided_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    reference_solution: str | None,
    ground_truth: Any,
    reveal_answer: bool,
    max_prompt_tokens: int | None,
) -> str:
    guided_messages = build_guided_messages(messages, reference_solution, ground_truth, reveal_answer)
    prompt = render_prompt(tokenizer, guided_messages)
    if not max_prompt_tokens or max_prompt_tokens <= 0:
        return prompt
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(prompt_tokens) <= max_prompt_tokens:
        return prompt

    empty_reference_messages = build_guided_messages(messages, "", ground_truth, reveal_answer)
    empty_reference_prompt = render_prompt(tokenizer, empty_reference_messages)
    overhead = len(tokenizer(empty_reference_prompt, add_special_tokens=False)["input_ids"])
    reference_budget = max_prompt_tokens - overhead - 16
    truncated_reference = truncate_text_to_tokens(tokenizer, reference_solution or "", reference_budget)
    guided_messages = build_guided_messages(messages, truncated_reference, ground_truth, reveal_answer)
    prompt = render_prompt(tokenizer, guided_messages)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(prompt_tokens) <= max_prompt_tokens:
        return prompt

    LOGGER.warning(
        "Guided prompt still exceeds token budget after teacher truncation: %d > %d; truncating rendered prompt",
        len(prompt_tokens),
        max_prompt_tokens,
    )
    return tokenizer.decode(prompt_tokens[:max_prompt_tokens], skip_special_tokens=False)


class Generator:
    def __init__(self, args: argparse.Namespace, tokenizer: Any):
        self.args = args
        self.tokenizer = tokenizer
        self.backend = args.backend
        if self.backend == "vllm":
            try:
                from vllm import LLM, SamplingParams
            except ImportError as exc:
                raise SystemExit("vllm backend requested but vllm is not installed.") from exc
            self.SamplingParams = SamplingParams
            self.llm = LLM(
                model=args.model_path,
                tokenizer=args.tokenizer_path or args.model_path,
                tensor_parallel_size=args.tensor_parallel_size,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
            )
            self.model = None
        elif self.backend == "hf":
            self.SamplingParams = None
            self.llm = None
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16 if args.dtype in ("bfloat16", "bf16") else torch.float16,
                device_map="auto",
                trust_remote_code=args.trust_remote_code,
            )
            self.model.eval()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def close(self) -> None:
        self.llm = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, prompts: list[str], n: int, temperature: float) -> list[list[str]]:
        if not prompts:
            return []
        if self.backend == "vllm":
            params = self.SamplingParams(
                n=n,
                temperature=temperature,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                max_tokens=self.args.max_new_tokens,
                stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else None,
            )
            outputs = self.llm.generate(prompts, params)
            return [[choice.text for choice in output.outputs] for output in outputs]
        return self._generate_hf(prompts, n=n, temperature=temperature)

    def _generate_hf(self, prompts: list[str], n: int, temperature: float) -> list[list[str]]:
        results: list[list[str]] = []
        for prompt in tqdm(prompts, desc="hf_generate"):
            encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **encoded,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=self.args.top_p,
                max_new_tokens=self.args.max_new_tokens,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            prompt_len = encoded["input_ids"].shape[-1]
            results.append(
                [
                    self.tokenizer.decode(output[prompt_len:], skip_special_tokens=False)
                    for output in outputs
                ]
            )
        return results


class TargetProbabilityScorer:
    def __init__(self, args: argparse.Namespace, tokenizer: Any):
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if args.dtype in ("bfloat16", "bf16") else torch.float16,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        self.model.eval()

    def score(self, prompt: str, target: str) -> list[float]:
        target_for_training = target
        if prompt.endswith(THINK_PREFIX) and target_for_training.startswith(THINK_PREFIX):
            target_for_training = target_for_training[len(THINK_PREFIX) :]

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        target_ids = self.tokenizer(target_for_training, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            eos = torch.tensor([[self.tokenizer.eos_token_id]], dtype=target_ids.dtype)
            target_ids = torch.cat([target_ids, eos], dim=-1)
        input_ids = torch.cat([prompt_ids, target_ids], dim=-1).to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_ids).logits
            start = prompt_ids.shape[-1] - 1
            end = input_ids.shape[-1] - 1
            target_logits = logits[0, start:end]
            target_tokens = input_ids[0, prompt_ids.shape[-1] :]
            probs = torch.softmax(target_logits.float(), dim=-1)
            token_probs = probs[torch.arange(target_tokens.shape[-1], device=probs.device), target_tokens]
        return token_probs.detach().cpu().tolist()

    def close(self) -> None:
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def choose_first_correct(
    candidates: Iterable[str],
    ground_truth: Any,
    verifier: VerifierRunner,
    prefix_think: bool,
    row_id: int | None = None,
    stage: str | None = None,
) -> GeneratedSolution | None:
    for rank, candidate in enumerate(candidates):
        text = candidate.strip()
        if not text:
            continue
        if prefix_think and not text.startswith(THINK_PREFIX):
            text = THINK_PREFIX + text
        try:
            is_correct = verifier.verify(text, ground_truth, row_id=row_id, rank=rank, stage=stage)
        except Exception:
            LOGGER.exception("Verifier failed for stage=%s row_id=%s rank=%d", stage, row_id, rank)
            is_correct = False
        if is_correct:
            if prefix_think and THINK_SUFFIX not in text:
                body = text[len(THINK_PREFIX) :] if text.startswith(THINK_PREFIX) else text
                text = f"{THINK_PREFIX}{THINK_SUFFIX}\n{body}"
            return GeneratedSolution(text=text, is_correct=True, rank=rank)
    return None


def make_preview(text: Any, max_chars: int = 500) -> str:
    text = "" if text is None else str(text)
    text = " ".join(text.split())
    return text[:max_chars]


def extract_user_prompt(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return messages[-1].get("content", "") if messages else ""


def get_extra_index(row: dict[str, Any], fallback: int) -> Any:
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        return extra_info.get("index", fallback)
    return fallback


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    write_jsonl(tmp_path, records)
    os.replace(tmp_path, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def checkpoint_root(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    return Path(os.path.splitext(args.output)[0] + ".checkpoints")


def batch_checkpoint_paths(root: Path, stage: str, start: int, end: int) -> tuple[Path, Path]:
    label = f"{start:08d}_{end:08d}"
    rollout_path = root / f"{stage}_rollouts" / f"batch_{label}.jsonl"
    result_path = root / f"{stage}_results" / f"batch_{label}.jsonl"
    return rollout_path, result_path


def write_rollout_checkpoint(
    path: Path,
    stage: str,
    start: int,
    end: int,
    row_ids: list[int],
    batch_outputs: list[list[str]],
) -> None:
    write_jsonl_atomic(
        path,
        (
            {
                "stage": stage,
                "batch_start": start,
                "batch_end": end,
                "row_id": row_id,
                "candidates": candidates,
            }
            for row_id, candidates in zip(row_ids, batch_outputs)
        ),
    )
    LOGGER.info("Wrote rollout checkpoint: %s", path)


def load_rollout_checkpoint(path: Path) -> dict[int, list[str]]:
    records = read_jsonl(path)
    return {int(record["row_id"]): list(record.get("candidates") or []) for record in records}


def apply_result_record(
    record: dict[str, Any],
    selected: list[str | None],
    mtg_source: list[str],
    first_bad: list[str | None],
    selection_rank: list[int | None],
) -> None:
    idx = int(record["row_id"])
    selected[idx] = record.get("selected")
    mtg_source[idx] = str(record["mtg_source"])
    first_bad[idx] = record.get("first_bad")
    selection_rank[idx] = record.get("selection_rank")


def load_result_checkpoint(
    path: Path,
    selected: list[str | None],
    mtg_source: list[str],
    first_bad: list[str | None],
    selection_rank: list[int | None],
) -> bool:
    if not path.exists():
        return False
    for record in read_jsonl(path):
        apply_result_record(record, selected, mtg_source, first_bad, selection_rank)
    LOGGER.info("Loaded result checkpoint: %s", path)
    return True


def write_result_checkpoint(path: Path, records: list[dict[str, Any]]) -> None:
    write_jsonl_atomic(path, records)
    LOGGER.info("Wrote result checkpoint: %s", path)


def make_result_record(
    stage: str,
    start: int,
    end: int,
    row_id: int,
    selected: list[str | None],
    mtg_source: list[str],
    first_bad: list[str | None],
    selection_rank: list[int | None],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "batch_start": start,
        "batch_end": end,
        "row_id": row_id,
        "mtg_source": mtg_source[row_id],
        "selected": selected[row_id],
        "selection_rank": selection_rank[row_id],
        "first_bad": first_bad[row_id],
    }


def write_rewrite_stats(
    stats_dir: str,
    rows: list[dict[str, Any]],
    original_messages: list[list[dict[str, str]]],
    ground_truths: list[Any],
    mtg_source: list[str],
    selected: list[str],
    selection_rank: list[int | None],
    first_bad: list[str | None],
    output_path: str,
) -> None:
    stats_path = Path(stats_dir)
    stats_path.mkdir(parents=True, exist_ok=True)

    source_counts = pd.Series(mtg_source).value_counts().to_dict()
    total = len(rows)
    summary = {
        "total": total,
        "output_path": output_path,
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "source_rates": {str(k): float(v) / total for k, v in source_counts.items()} if total else {},
    }
    (stats_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    records = []
    for i, row in enumerate(rows):
        prompt_text = extract_user_prompt(original_messages[i])
        records.append(
            {
                "row_id": i,
                "extra_index": get_extra_index(row, i),
                "mtg_source": mtg_source[i],
                "selection_rank": selection_rank[i],
                "ground_truth": ground_truths[i],
                "prompt_preview": make_preview(prompt_text),
                "selected_target_preview": make_preview(selected[i]),
                "first_bad_preview": make_preview(first_bad[i]),
            }
        )

    write_jsonl(stats_path / "all_cases.jsonl", records)
    for source in ("self_correct", "guided_rewrite", "expert_fallback", "missing_target"):
        write_jsonl(
            stats_path / f"{source}.jsonl",
            (record for record in records if record["mtg_source"] == source),
        )

    pd.DataFrame(records).to_csv(stats_path / "all_cases.csv", index=False, escapechar="\\")
    LOGGER.info("Wrote rewrite stats to %s", stats_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input LUFFY parquet with prompt/target columns.")
    parser.add_argument("--output", required=True, help="Output rewritten parquet path.")
    parser.add_argument("--model_path", required=True, help="Target policy model used for rewriting.")
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path. Defaults to model_path.")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--target_key", default="target")
    parser.add_argument("--ground_truth_key", default="ground_truth")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--self_n", type=int, default=4)
    parser.add_argument("--rewrite_n", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rewrite_temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--think_format", action="store_true", help="Require <think>...</think> during verification.")
    parser.add_argument("--no_prefix_think", action="store_true", help="Do not prepend '<think>\\n' to generated targets.")
    parser.add_argument("--no_reveal_answer", action="store_true", help="Do not reveal the final answer in guided re-solving.")
    parser.add_argument("--no_compute_target_probs", action="store_true", help="Skip target_ds_qwen_7b_probs computation.")
    parser.add_argument("--max_guided_prompt_tokens", type=int, default=None, help="Truncate guided rewrite prompts to this token budget.")
    parser.add_argument("--stats_dir", default=None, help="Directory for summary.json and per-source jsonl stats.")
    parser.add_argument("--checkpoint_dir", default=None, help="Directory for per-batch rollout/result checkpoints.")
    parser.add_argument("--no_resume", action="store_true", help="Ignore existing checkpoint files and overwrite them.")
    parser.add_argument(
        "--verify_timeout_seconds",
        type=float,
        default=30.0,
        help="Per-candidate verifier timeout. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model_path,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    verifier = VerifierRunner(
        think_format=args.think_format,
        timeout_seconds=args.verify_timeout_seconds,
    )
    df = pd.read_parquet(args.input)
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()
    LOGGER.info("Loaded %d rows from %s", len(df), args.input)

    rows = [row.to_dict() for _, row in df.iterrows()]
    original_messages = [normalize_messages(row[args.prompt_key]) for row in rows]
    prompts = [render_prompt(tokenizer, messages) for messages in original_messages]
    ground_truths = [get_ground_truth(row, args.ground_truth_key) for row in rows]
    ckpt_root = checkpoint_root(args)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Checkpoint directory: %s", ckpt_root)

    generator = Generator(args, tokenizer)
    selected: list[str | None] = [None] * len(rows)
    mtg_source: list[str] = ["unprocessed"] * len(rows)
    first_bad: list[str | None] = [None] * len(rows)
    selection_rank: list[int | None] = [None] * len(rows)

    for start in tqdm(range(0, len(rows), args.batch_size), desc="self_sample"):
        end = min(start + args.batch_size, len(rows))
        row_ids = list(range(start, end))
        rollout_path, result_path = batch_checkpoint_paths(ckpt_root, "self_sample", start, end)
        if not args.no_resume and load_result_checkpoint(
            result_path,
            selected,
            mtg_source,
            first_bad,
            selection_rank,
        ):
            continue
        if not args.no_resume and rollout_path.exists():
            rollout_by_row = load_rollout_checkpoint(rollout_path)
            batch_outputs = [rollout_by_row[idx] for idx in row_ids]
            LOGGER.info("Loaded rollout checkpoint: %s", rollout_path)
        else:
            batch_outputs = generator.generate(prompts[start:end], n=args.self_n, temperature=args.temperature)
            write_rollout_checkpoint(rollout_path, "self_sample", start, end, row_ids, batch_outputs)

        result_records = []
        for idx, candidates in zip(row_ids, batch_outputs):
            chosen = choose_first_correct(
                candidates,
                ground_truths[idx],
                verifier,
                prefix_think=not args.no_prefix_think,
                row_id=idx,
                stage="self_sample",
            )
            if chosen is not None:
                selected[idx] = chosen.text
                mtg_source[idx] = "self_correct"
                selection_rank[idx] = chosen.rank
            else:
                first_bad[idx] = candidates[0].strip() if candidates else ""
            result_records.append(
                make_result_record(
                    "self_sample",
                    start,
                    end,
                    idx,
                    selected,
                    mtg_source,
                    first_bad,
                    selection_rank,
                )
            )
        write_result_checkpoint(result_path, result_records)

    rewrite_indices = [i for i, item in enumerate(selected) if item is None]
    LOGGER.info("Self-correct rows: %d / %d", len(rows) - len(rewrite_indices), len(rows))

    reference_solutions = [normalize_target_content(row.get(args.target_key)) for row in rows]
    for start in tqdm(range(0, len(rewrite_indices), args.batch_size), desc="guided_rewrite"):
        end = min(start + args.batch_size, len(rewrite_indices))
        row_ids = rewrite_indices[start:end]
        rollout_path, result_path = batch_checkpoint_paths(ckpt_root, GUIDED_REWRITE_CHECKPOINT_STAGE, start, end)
        if not args.no_resume and load_result_checkpoint(
            result_path,
            selected,
            mtg_source,
            first_bad,
            selection_rank,
        ):
            continue
        if not args.no_resume and rollout_path.exists():
            rollout_by_row = load_rollout_checkpoint(rollout_path)
            batch_outputs = [rollout_by_row[idx] for idx in row_ids]
            LOGGER.info("Loaded rollout checkpoint: %s", rollout_path)
        else:
            batch_prompts = [
                render_guided_prompt(
                    tokenizer,
                    original_messages[i],
                    reference_solutions[i],
                    ground_truths[i],
                    reveal_answer=not args.no_reveal_answer,
                    max_prompt_tokens=args.max_guided_prompt_tokens,
                )
                for i in row_ids
            ]
            batch_outputs = generator.generate(batch_prompts, n=args.rewrite_n, temperature=args.rewrite_temperature)
            write_rollout_checkpoint(rollout_path, GUIDED_REWRITE_CHECKPOINT_STAGE, start, end, row_ids, batch_outputs)

        result_records = []
        for idx, candidates in zip(row_ids, batch_outputs):
            chosen = choose_first_correct(
                candidates,
                ground_truths[idx],
                verifier,
                prefix_think=not args.no_prefix_think,
                row_id=idx,
                stage="guided_rewrite",
            )
            if chosen is not None:
                selected[idx] = chosen.text
                mtg_source[idx] = "guided_rewrite"
                selection_rank[idx] = chosen.rank
            result_records.append(
                make_result_record(
                    "guided_rewrite",
                    start,
                    end,
                    idx,
                    selected,
                    mtg_source,
                    first_bad,
                    selection_rank,
                )
            )
        write_result_checkpoint(result_path, result_records)

    fallback_count = 0
    for i, item in enumerate(selected):
        if item is not None:
            continue
        expert = normalize_target_content(rows[i].get(args.target_key))
        if expert is None:
            selected[i] = ""
            mtg_source[i] = "missing_target"
        else:
            selected[i] = expert
            mtg_source[i] = "expert_fallback"
            fallback_count += 1
    LOGGER.info("Expert fallback rows: %d", fallback_count)

    verifier.close()
    generator.close()

    for i, text in enumerate(selected):
        rows[i][args.target_key] = [{"role": "assistant", "content": text}]
        extra_info = rows[i].get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info["mtg_source"] = mtg_source[i]
        rows[i]["extra_info"] = extra_info

    if not args.no_compute_target_probs:
        scorer = TargetProbabilityScorer(args, tokenizer)
        probs = []
        for prompt, row in tqdm(zip(prompts, rows), total=len(rows), desc="target_probs"):
            probs.append(scorer.score(prompt, normalize_target_content(row[args.target_key]) or ""))
        scorer.close()
        for row, prob in zip(rows, probs):
            row["target_ds_qwen_7b_probs"] = prob

    stats_dir = args.stats_dir
    if stats_dir is None:
        stats_dir = os.path.splitext(args.output)[0] + ".stats"
    write_rewrite_stats(
        stats_dir=stats_dir,
        rows=rows,
        original_messages=original_messages,
        ground_truths=ground_truths,
        mtg_source=mtg_source,
        selected=[item or "" for item in selected],
        selection_rank=selection_rank,
        first_bad=first_bad,
        output_path=args.output,
    )

    out_df = pd.DataFrame(rows, columns=list(df.columns) + [c for c in ("target_ds_qwen_7b_probs",) if c in rows[0] and c not in df.columns])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    LOGGER.info("Wrote rewritten dataset to %s", args.output)
    LOGGER.info("Source counts: %s", dict(pd.Series(mtg_source).value_counts()))


if __name__ == "__main__":
    main()
