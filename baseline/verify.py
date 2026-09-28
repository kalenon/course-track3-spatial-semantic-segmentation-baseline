"""Check a mounted Track 3 development set and feature-extractor weights."""

import argparse
from pathlib import Path


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
