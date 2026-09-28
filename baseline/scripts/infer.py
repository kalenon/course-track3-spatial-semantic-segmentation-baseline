#!/usr/bin/env python3
"""Run the two-stage baseline on unlabeled four-channel mixtures."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from src.utils import LABELS, initialize_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("src/evaluation/eval_configs/m2dat_4c_resunetk.yaml"))
    parser.add_argument("--tagger-ckpt", type=Path)
    parser.add_argument("--separator-ckpt", type=Path)
    parser.add_argument("--feature-weights", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["model"]
    model_args = config["args"]
    if args.tagger_ckpt:
        model_args["tagger_ckpt"] = str(args.tagger_ckpt)
    if args.separator_ckpt:
        model_args["separator_ckpt"] = str(args.separator_ckpt)
    if args.feature_weights:
        model_args["tagger_config"]["args"]["weight_file"] = str(args.feature_weights)
    device = torch.device(args.device)
    model = initialize_config(config).to(device).eval()
    files = sorted(args.input_dir.glob("*.wav"))
    if not files:
        raise ValueError(f"No WAV files in {args.input_dir}")
    (args.output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audio").mkdir(parents=True, exist_ok=True)
    allowed = set(LABELS["ssp_track3"])
    for path in files:
        mixture, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != 32000 or mixture.shape != (320000, 4) or not np.isfinite(mixture).all():
            raise ValueError(f"Expected 10-s four-channel 32-kHz WAV: {path}")
        tensor = torch.from_numpy(mixture.T.copy()).unsqueeze(0).to(device)
        with torch.inference_mode():
            result = model.predict_label_separate(tensor)
        labels = result["label"][0]
        waveforms = result["waveform"][0, :, 0, :].detach().cpu().numpy()
        events = []
        for label, waveform in zip(labels, waveforms):
            if label == "silence":
                continue
            if label not in allowed or len(events) >= 3:
                raise ValueError(f"Invalid event label/count for {path}")
            index = len(events)
            name = f"{path.stem}_{index}_{label}.wav"
            sf.write(args.output_dir / "audio" / name, waveform, 32000, subtype="PCM_16")
            events.append({"source_index": index, "class": label, "audio": f"audio/{name}"})
        metadata = {"mixture_id": path.stem, "events": events}
        (args.output_dir / "metadata" / f"{path.stem}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote predictions for {len(files)} mixtures to {args.output_dir}")


if __name__ == "__main__":
    main()
