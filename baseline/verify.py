"""Check a mounted Track 3 development set and feature-extractor weights."""

import argparse
import json
from pathlib import Path

from src.datamodules.metadata_paths import rebase_dev_set_paths


FEATURE_WEIGHT = (
    "m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly/"
    "weights_ep69it3124-0.47998.pth"
)
REQUIRED = (
    "config", "metadata/valid.json", "metadata/valid",
    "interference/train", "interference/valid",
    "noise/train", "noise/valid", "room_ir/train", "room_ir/valid",
    "sound_event/train", "sound_event/valid",
    "synthesized/test/soundscape", "synthesized/test/oracle_target",
)


def check_validation_metadata(data: Path) -> list[Path]:
    """Check paths inside each released validation scene, not just the index."""
    index_path = data / "metadata/valid.json"
    if not index_path.is_file():
        return []  # Already reported by REQUIRED.

    missing = []
    with index_path.open(encoding="utf-8") as stream:
        entries = json.load(stream)
    for entry in entries:
        metadata_path = data / "metadata" / entry["metadata_path"]
        if not metadata_path.is_file():
            missing.append(metadata_path)
            continue
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = rebase_dev_set_paths(json.load(stream), data)
        config = metadata["config"]
        room = metadata["room"]["args"]["metadata"]
        for directory in ("foreground_dir", "background_dir", "interference_dir"):
            value = config.get(directory)
            if value and not Path(value).is_dir():
                missing.append(Path(value))
        for value in (config["room_config"]["args"]["path"], room["sofa_path"]):
            if not Path(value).is_file():
                missing.append(Path(value))
    return list(dict.fromkeys(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, default=Path("."),
                        help="Repository baseline root (for legacy usage)")
    parser.add_argument("--data-root", type=Path,
                        help="Mounted dev_set directory; defaults to source_dir/data/dev_set")
    parser.add_argument("--checkpoint-root", type=Path,
                        help="Mounted weights directory; defaults to source_dir/checkpoint")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    data = (args.data_root or source / "data/dev_set").resolve()
    checkpoint = (args.checkpoint_root or source / "checkpoint").resolve()
    missing = []
    if not (source / "src").is_dir():
        missing.append(source / "src")
    for relative in (
        "src/modules/spatial_audio_synthesizer/spatial_audio_synthesizer.py",
        "src/modules/spatial_audio_synthesizer/room.py",
        "src/modules/spatial_audio_synthesizer/utils.py",
    ):
        if not (source / relative).is_file():
            missing.append(source / relative)
    for relative in REQUIRED:
        if not (data / relative).exists():
            missing.append(data / relative)
    missing.extend(check_validation_metadata(data))
    if not (checkpoint / FEATURE_WEIGHT).is_file():
        missing.append(checkpoint / FEATURE_WEIGHT)
    for name in ("m2dat_4c.ckpt", "resunetk.ckpt"):
        if not (checkpoint / name).is_file():
            missing.append(checkpoint / name)
    print(f"Repository: {source}\nDevelopment set: {data}\nWeights: {checkpoint}")
    if missing:
        raise SystemExit("Missing required paths:\n" + "\n".join(map(str, missing)))
    print("Track 3 data and feature weights: OK")


if __name__ == "__main__":
    main()
