#!/usr/bin/env python3
"""Run a Track 3 training stage with replaceable data and weight roots."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def replace_paths(value, data_root, checkpoint_root):
    if isinstance(value, dict):
        return {key: replace_paths(item, data_root, checkpoint_root) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_paths(item, data_root, checkpoint_root) for item in value]
    if isinstance(value, str):
        marker = "data/dev_set/"
        if marker in value and (value.startswith(marker) or value.startswith("/")):
            return str(data_root / value.split(marker, 1)[1])
        marker = "checkpoint/"
        if marker in value and (value.startswith(marker) or value.startswith("/")):
            return str(checkpoint_root / value.split(marker, 1)[1])
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.data_root.is_dir() or not args.checkpoint_root.is_dir():
        raise FileNotFoundError("data-root and checkpoint-root must exist")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    adjusted = replace_paths(config, args.data_root.resolve(), args.checkpoint_root.resolve())
    args.workspace.mkdir(parents=True, exist_ok=True)
    generated = args.workspace / f"course_{args.config.stem}.yaml"
    generated.write_text(yaml.safe_dump(adjusted, sort_keys=False), encoding="utf-8")
    command = [sys.executable, "-m", "src.train", "-c", str(generated), "-w", str(args.workspace)]
    if args.resume:
        command.extend(["-r", str(args.resume)])
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
