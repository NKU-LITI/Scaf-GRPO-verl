#!/usr/bin/env python3

import argparse
from collections import defaultdict
from pathlib import Path

import wandb
from tensorboard.backend.event_processing import event_accumulator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TensorBoard logs to a W&B offline run."
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        required=True,
        help="TensorBoard event directory.",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        required=True,
        help="Directory used to store W&B offline run.",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="W&B project name.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="W&B run name.",
    )
    return parser.parse_args()


def load_tensorboard_scalars(tb_dir: Path):
    ea = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={
            event_accumulator.SCALARS: 0,
        },
    )

    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])

    if not scalar_tags:
        raise RuntimeError(
            f"No scalar metrics found in TensorBoard directory: {tb_dir}"
        )

    # (step, tag) -> (wall_time, value)
    # 如果 resume 后同一个 step/tag 出现多次，
    # 保留 wall_time 最新的一次。
    latest = {}

    for tag in scalar_tags:
        for event in ea.Scalars(tag):
            key = (int(event.step), tag)

            old = latest.get(key)

            if old is None or event.wall_time >= old[0]:
                latest[key] = (
                    float(event.wall_time),
                    float(event.value),
                )

    rows = defaultdict(dict)

    for (step, tag), (_, value) in latest.items():
        rows[step][tag] = value

    return dict(rows), scalar_tags


def main():
    args = parse_args()

    tb_dir = Path(args.tensorboard_dir).resolve()
    wandb_dir = Path(args.wandb_dir).resolve()

    if not tb_dir.exists():
        raise FileNotFoundError(
            f"TensorBoard directory does not exist: {tb_dir}"
        )

    event_files = list(tb_dir.glob("events.out.tfevents*"))

    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event files found in: {tb_dir}"
        )

    wandb_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TensorBoard -> W&B offline conversion")
    print(f"TensorBoard dir : {tb_dir}")
    print(f"W&B dir         : {wandb_dir}")
    print(f"Project         : {args.project}")
    print(f"Run name        : {args.name}")
    print(f"Event files     : {len(event_files)}")
    print("=" * 80)

    rows, scalar_tags = load_tensorboard_scalars(tb_dir)

    print(f"Scalar tags: {len(scalar_tags)}")
    print(f"Steps      : {len(rows)}")

    run = wandb.init(
        project=args.project,
        name=args.name,
        dir=str(wandb_dir),
        mode="offline",
        config={
            "imported_from": "tensorboard",
            "tensorboard_dir": str(tb_dir),
            "tensorboard_event_files": len(event_files),
        },
    )

    for step in sorted(rows):
        run.log(
            rows[step],
            step=step,
        )

    run.summary["tensorboard_import/num_scalar_tags"] = len(scalar_tags)
    run.summary["tensorboard_import/num_steps"] = len(rows)

    run.finish()

    print("=" * 80)
    print("TensorBoard -> W&B conversion completed.")
    print(f"W&B output: {wandb_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()