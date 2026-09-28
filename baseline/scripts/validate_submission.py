#!/usr/bin/env python3
"""Check the unlabeled Track 3 JSON and audio contract."""

import argparse
import json
from pathlib import Path

import soundfile as sf

from src.utils import LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    ids = {path.stem for path in args.input_dir.glob("*.wav")}
    if not ids:
        raise ValueError("No input WAV files")
    metadata_dir = args.predictions / "metadata"
    submitted = {path.stem for path in metadata_dir.glob("*.json")}
    if ids != submitted:
        raise ValueError(f"Metadata IDs differ: missing={sorted(ids-submitted)}, extra={sorted(submitted-ids)}")
    allowed = set(LABELS["ssp_track3"])
    used_audio = set()
    for mixture_id in sorted(ids):
        data = json.loads((metadata_dir / f"{mixture_id}.json").read_text(encoding="utf-8"))
        events = data.get("events")
        if data.get("mixture_id") != mixture_id or not isinstance(events, list) or len(events) > 3:
            raise ValueError(f"Invalid metadata: {mixture_id}")
        if [event.get("source_index") for event in events] != list(range(len(events))):
            raise ValueError(f"Invalid source indices: {mixture_id}")
        for event in events:
            label = event.get("class")
            expected = f"audio/{mixture_id}_{event['source_index']}_{label}.wav"
            if label not in allowed or event.get("audio") != expected:
                raise ValueError(f"Invalid event path/label: {mixture_id}")
            path = args.predictions / expected
            info = sf.info(path)
            if info.format != "WAV" or info.channels != 1 or info.samplerate != 32000 or info.frames != 320000:
                raise ValueError(f"Invalid separated WAV: {path}")
            used_audio.add(path.resolve())
    extra_audio = {p.resolve() for p in (args.predictions / "audio").glob("*.wav")} - used_audio
    if extra_audio:
        raise ValueError(f"Unexpected audio files: {sorted(extra_audio)}")
    print(f"Valid submission: {len(ids)} mixtures, {len(used_audio)} separated events")


if __name__ == "__main__":
    main()
