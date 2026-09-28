"""Download and extract the Track 3 reference and M2D weight archives.

Run from the baseline directory. Checkpoints are written only to --weights-root.
"""

import argparse
import os
import subprocess
import zlib
import zipfile
from pathlib import Path


ARCHIVES = {
    "reference": {
        "url": "https://github.com/nttcslab/dcase2026_task4_baseline/releases/download/v1.0.0/baseline_checkpoint.zip",
        "files": ("m2dat_4c.ckpt", "resunetk.ckpt"),
    },
    "m2d": {
        "url": "https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly.zip",
        "files": (
            "m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly/"
            "weights_ep69it3124-0.47998.pth",
        ),
    },
}


def crc32(path):
    value = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value = zlib.crc32(block, value)
    return value & 0xFFFFFFFF


def download(url, archive):
    if archive.is_file() and archive.stat().st_size > 0:
        print(f"Using existing archive: {archive}", flush=True)
        return
    if archive.exists():
        raise ValueError(f"Archive exists but is empty or not a file: {archive}; inspect it before retrying")
    partial = archive.with_name(archive.name + ".part")
    print(f"Downloading {url} -> {partial}", flush=True)
    subprocess.run([
        "curl", "--fail", "--location", "--retry", "5", "--retry-all-errors",
        "--connect-timeout", "30", "--continue-at", "-", "--output", str(partial), url,
    ], check=True)
    if not partial.is_file() or partial.stat().st_size == 0:
        raise ValueError(f"Downloaded archive is empty: {partial}")
    partial.replace(archive)
    print(f"Downloaded: {archive}", flush=True)


def extract_selected(archive, root, requested):
    root = root.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for name in requested:
            member = zipped.getinfo(name)
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root) or member.is_dir():
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != member.file_size or crc32(target) != member.CRC:
                    raise ValueError(f"Existing checkpoint differs from archive: {target}")
                print(f"Verified existing checkpoint: {target}", flush=True)
                continue
            partial = target.with_name(target.name + ".part")
            if partial.exists():
                raise FileExistsError(f"Incomplete extraction exists: {partial}; inspect it before retrying")
            try:
                with zipped.open(member) as source, partial.open("wb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                if partial.stat().st_size != member.file_size or crc32(partial) != member.CRC:
                    raise ValueError(f"Extracted file failed CRC check: {partial}")
                os.replace(partial, target)
            except Exception:
                # Retain the partial file for inspection and resumable re-download safety.
                raise
            print(f"Ready: {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--only", choices=("all", "reference", "m2d"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.weights_root.expanduser().resolve()
    names = ARCHIVES if args.only == "all" else {args.only: ARCHIVES[args.only]}
    for name, item in names.items():
        archive = root / Path(item["url"]).name
        print(f"{name}: {item['url']}\n  archive: {archive}", flush=True)
        if args.dry_run:
            continue
        root.mkdir(parents=True, exist_ok=True)
        download(item["url"], archive)
        extract_selected(archive, root, item["files"])


if __name__ == "__main__":
    main()
