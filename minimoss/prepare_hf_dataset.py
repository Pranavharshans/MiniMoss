#!/usr/bin/env python3
"""Create a tiny single-speaker manifest from LibriTTS-R on Hugging Face."""

import argparse
import json
import random
from pathlib import Path

from datasets import Audio, load_dataset


def split_records(records: list[dict], validation_count: int, seed: int):
    """Return deterministic train/validation splits without changing the input."""
    if validation_count < 0 or validation_count >= len(records):
        raise ValueError("validation_count must be between 0 and count - 1")

    shuffled = sorted(records, key=lambda record: record["id"])
    random.Random(seed).shuffle(shuffled)
    return shuffled[validation_count:], shuffled[:validation_count]


def write_manifest(path: Path, records: list[dict]):
    with path.open("w") as manifest:
        for record in records:
            manifest.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download a tiny LibriTTS-R subset")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--speaker-id", default="3081")
    parser.add_argument("--validation-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.validation_count < 0 or args.validation_count >= args.count:
        raise ValueError("--validation-count must be between 0 and count - 1")

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "parler-tts/libritts_r_filtered",
        "clean",
        split="dev.clean",
        streaming=True,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))

    records = []
    for row in dataset:
        if str(row["speaker_id"]) != args.speaker_id:
            continue

        utterance_id = str(row["id"])
        wav_path = wav_dir / f"{utterance_id}.wav"
        audio = row["audio"]
        audio_bytes = audio.get("bytes")
        if audio_bytes is None:
            source_path = audio.get("path")
            if source_path is None:
                raise RuntimeError(f"No bytes or path returned for {utterance_id}")
            wav_path.write_bytes(Path(source_path).read_bytes())
        else:
            wav_path.write_bytes(audio_bytes)

        records.append({
            "id": utterance_id,
            "wav": str(wav_path.resolve()),
            "text": row["text_normalized"],
        })
        if len(records) == args.count:
            break

    if len(records) < args.count:
        raise RuntimeError(
            f"Found only {len(records)} clips for speaker {args.speaker_id}; requested {args.count}"
        )

    manifest_path = output_dir / "manifest.jsonl"
    write_manifest(manifest_path, records)

    print(f"Wrote {len(records)} single-speaker clips to {wav_dir}")
    print(f"Manifest: {manifest_path}")
    if args.validation_count:
        train_records, validation_records = split_records(
            records, args.validation_count, args.seed
        )
        train_path = output_dir / "train_manifest.jsonl"
        validation_path = output_dir / "validation_manifest.jsonl"
        write_manifest(train_path, train_records)
        write_manifest(validation_path, validation_records)
        print(f"Train manifest: {train_path} ({len(train_records)} clips)")
        print(f"Validation manifest: {validation_path} ({len(validation_records)} clips)")


if __name__ == "__main__":
    main()
