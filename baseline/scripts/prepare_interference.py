"""Download, extract, and prepare the independent interference sources.

Run from the Track 3 baseline root. The output is written into the selected
development set's interference/train and interference/valid directories.
The raw archive and extracted Semantic Hearing dataset stay outside dev_set.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit


OFFICIAL_URL = "https://semantichearing.cs.washington.edu/BinauralCuratedDataset.tar"
BASELINE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=BASELINE_ROOT / "data/dev_set",
        help="Existing dev_set directory; interference/ is written here.",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        help="Directory for the raw archive and extracted BinauralCuratedDataset.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already downloaded BinauralCuratedDataset.tar instead of downloading.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Use an already extracted BinauralCuratedDataset directory; skip download/extraction.",
    )
    parser.add_argument("--url", default=OFFICIAL_URL, help="Override the archive URL.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow reprocessing a partially generated interference/ directory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show resolved paths only.")
    args = parser.parse_args()
    if args.source_root is not None and args.archive is not None:
        parser.error("Choose --source-root or --archive, not both")
    if args.source_root is None and args.storage_dir is None:
        parser.error("--storage-dir is required unless --source-root is supplied")
    if args.source_root is None and args.archive is None:
        parsed = urlsplit(args.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            parser.error(
                "--url must be a plain HTTP(S) URL, not a Markdown link such as "
                "'[label](https://example.com/file.tar)'"
            )
    return args


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=BASELINE_ROOT, check=True)


def check_dataset(dataset_root: Path) -> Path:
    dataset_root = dataset_root.resolve(strict=True)
    required = (
        "metadata/valid.json",
        "metadata/valid",
        "sound_event/train",
        "sound_event/valid",
        "noise/train",
        "noise/valid",
        "room_ir/train",
        "room_ir/valid",
    )
    for relative in required:
        if not (dataset_root / relative).exists():
            raise FileNotFoundError(f"Not a complete Track 3 dev_set: missing {dataset_root / relative}")
    return dataset_root


def check_source(source_root: Path) -> Path:
    source_root = source_root.resolve(strict=True)
    for split in ("train", "val"):
        directory = source_root / "bg_scaper_fmt" / split
        if not directory.is_dir() or not any(directory.iterdir()):
            raise FileNotFoundError(f"Missing or empty Semantic Hearing source: {directory}")
    return source_root


def get_source(args: argparse.Namespace) -> Path:
    if args.source_root is not None:
        return check_source(args.source_root)

    storage = args.storage_dir.resolve()
    source_root = storage / "BinauralCuratedDataset"
    if source_root.is_dir():
        return check_source(source_root)

    archive = args.archive.resolve(strict=True) if args.archive else storage / "BinauralCuratedDataset.tar"
    if not archive.is_file():
        storage.mkdir(parents=True, exist_ok=True)
        partial = archive.with_name(archive.name + ".part")
        try:
            run([
                "curl", "--fail", "--location", "--retry", "5", "--retry-all-errors",
                "--connect-timeout", "30", "--continue-at", "-", "--progress-bar",
                "--output", str(partial), args.url,
            ])
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Archive download failed (curl exit {error.returncode}) from {args.url}. "
                f"Any partial download remains at {partial}. "
                "The development set has not been changed. Retry later, pass "
                "--url with a working mirror, or obtain the archive on another "
                "machine and use --archive /path/to/BinauralCuratedDataset.tar. "
                "Do not run verify.py until preparation succeeds."
            ) from error
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError(f"Download produced no archive: {partial}")
        partial.replace(archive)

    if source_root.exists():
        raise FileExistsError(
            f"Extraction directory exists but is incomplete: {source_root}. "
            "Inspect it manually; this script will not overwrite it."
        )
    storage.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} into {storage} (this may take a long time)", flush=True)
    run(["tar", "--extract", "--file", str(archive), "--directory", str(storage)])
    return check_source(source_root)


def check_validation_references(dataset_root: Path) -> tuple[int, int]:
    metadata_root = dataset_root / "metadata"
    items = json.loads((metadata_root / "valid.json").read_text(encoding="utf-8"))
    needed = set()
    for item in items:
        metadata = json.loads((metadata_root / item["metadata_path"]).read_text(encoding="utf-8"))
        for event in metadata.get("int_events", []):
            needed.add(event["source_file"])
    missing = sorted(
        relative for relative in needed
        if not (dataset_root / "interference/valid" / relative).is_file()
    )
    if missing:
        preview = "\n".join(missing[:10])
        raise RuntimeError(
            f"{len(missing)}/{len(needed)} fixed-validation interference files are missing. "
            f"First missing paths:\n{preview}"
        )
    return len(items), len(needed)


def main() -> None:
    args = parse_args()
    dataset_root = check_dataset(args.dataset_root)
    output_root = dataset_root / "interference"
    print(f"Development set: {dataset_root}", flush=True)
    print(f"Processed output: {output_root}", flush=True)
    if args.dry_run:
        if args.source_root is not None:
            check_source(args.source_root)
        if args.archive is not None and not args.archive.is_file():
            raise FileNotFoundError(args.archive)
        print(f"Source: {args.source_root or args.archive or args.url}")
        print(f"Storage: {args.storage_dir or '(already extracted)'}")
        return

    if output_root.is_dir() and any(output_root.rglob("*.wav")) and not args.resume:
        raise FileExistsError(
            f"Processed WAVs already exist in {output_root}; use --resume only "
            "after checking that this is a partial run."
        )
    source_root = get_source(args)
    print(f"Semantic Hearing source: {source_root}", flush=True)
    run([
        sys.executable, str(BASELINE_ROOT / "add_interference.py"),
        "--input_dir", str(source_root / "bg_scaper_fmt"),
        "--output_dir", str(output_root),
    ])

    counts = {
        split: sum(1 for _ in (output_root / split).rglob("*.wav"))
        for split in ("train", "valid")
    }
    if not all(counts.values()):
        raise RuntimeError(f"No processed interference WAVs in one split: {counts}")
    scenes, needed = check_validation_references(dataset_root)
    print(
        f"Done: train={counts['train']} WAVs, valid={counts['valid']} WAVs; "
        f"all {needed} interference files referenced by {scenes} validation scenes exist.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
