#!/usr/bin/env python3
"""Score Track 3 JSON predictions against labeled target WAV files."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.evaluation.metrics.label_metric import LabelMetric
from src.evaluation.metrics.s5capi_metric import S5ClassAwareMetric
from src.utils import LABELS


def waveform(path, channels):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 32000 or audio.shape != (320000, channels) or not np.isfinite(audio).all():
        raise ValueError(f"Expected finite 320000-sample {channels}-channel WAV: {path}")
    return torch.from_numpy(audio.copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mixtures = sorted(args.input_dir.glob("*.wav"))
    if not mixtures:
        raise ValueError("No mixtures found")
    metric = S5ClassAwareMetric(metricfunc="sdr")
    label_metric = LabelMetric()
    allowed = set(LABELS["ssp_track3"])
    all_reference_files = list(args.reference_dir.glob("*.wav"))
    scores = []
    for mixture_path in mixtures:
        mixture_id = mixture_path.stem
        metadata = json.loads((args.predictions / "metadata" / f"{mixture_id}.json").read_text())
        if metadata.get("mixture_id") != mixture_id:
            raise ValueError(f"Mixture ID mismatch: {mixture_id}")
        predicted = metadata["events"]
        predicted_labels = [event["class"] for event in predicted]
        estimated = [waveform(args.predictions / event["audio"], 1)[:, 0] for event in predicted]
        reference_labels = []
        reference = []
        pattern = re.compile(re.escape(mixture_id) + r"_(?:\d+_)?(.+)\.wav")
        for path in all_reference_files:
            match = pattern.fullmatch(path.name)
            if match and match.group(1) in allowed:
                reference_labels.append(match.group(1))
                reference.append(waveform(path, 1)[:, 0])
        if len(reference) > 3 or len(predicted) > 3:
            raise ValueError(f"Too many events for {mixture_id}")
        ref = torch.stack(reference + [torch.zeros(320000)] * (3 - len(reference)))
        est = torch.stack(estimated + [torch.zeros(320000)] * (3 - len(estimated)))
        ref_labels = reference_labels + ["silence"] * (3 - len(reference))
        est_labels = predicted_labels + ["silence"] * (3 - len(predicted))
        mix = waveform(mixture_path, 4)[:, 0]
        value = metric.compute_sample(est_labels, est, ref_labels, ref, mix)
        label_metric.update([est_labels], [ref_labels])
        scores.append({"mixture_id": mixture_id, "capi_sdri": value})
    valid = [row["capi_sdri"] for row in scores if row["capi_sdri"] is not None]
    label_values = label_metric.metric_values
    tp = sum(row["tp"] for row in label_values)
    fp = sum(row["fp"] for row in label_values)
    fn = sum(row["fn"] for row in label_values)
    result = {
        "mixtures": len(mixtures),
        "CAPI-SDRi": {"mean": sum(valid) / len(valid) if valid else None,
                       "scored_mixtures": len(valid)},
        "label_metrics": {
            "accuracy_mix": 100 * sum(row["acc"] for row in label_values) / len(label_values),
            "accuracy_src": 100 * tp / (tp + fp + fn) if tp + fp + fn else 100.0,
        },
        "per_mixture": scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("mixtures", "CAPI-SDRi", "label_metrics")}, indent=2))


if __name__ == "__main__":
    main()
