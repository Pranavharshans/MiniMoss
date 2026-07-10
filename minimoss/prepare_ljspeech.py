#!/usr/bin/env python3
"""Prepare deterministic single-speaker LJSpeech manifests from Hugging Face."""

import argparse
import json
import random
from pathlib import Path

from datasets import Audio, load_dataset


def select_split(records: list[dict], train_count: int, validation_count: int, seed: int):
    required = train_count + validation_count
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("train_count and validation_count must be positive")
    if len(records) < required:
        raise ValueError(f"Need {required} records, found {len(records)}")
    shuffled = sorted(records, key=lambda record: record["id"])
    random.Random(seed).shuffle(shuffled)
    return shuffled[:train_count], shuffled[train_count:required]


def write_manifest(path: Path, records: list[dict]):
    with path.open("w") as manifest:
        for record in records:
            manifest.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare an LJSpeech MiniMoss split")
    parser.add_argument("--output-dir", default="data_ljspeech_1100")
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        default="dinhbinh161/ljspeech",
        help="Parquet LJSpeech mirror compatible with datasets 4/5",
    )
    args = parser.parse_args()

    required = args.train_count + args.validation_count
    if args.train_count <= 0 or args.validation_count <= 0:
        raise ValueError("--train-count and --validation-count must be positive")

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split="train", streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    records = []
    for row in dataset:
        utterance_id = str(row["id"])
        audio = row["audio"]
        audio_bytes = audio.get("bytes")
        source_path = audio.get("path")
        wav_path = wav_dir / f"{utterance_id}.wav"
        if audio_bytes is not None:
            wav_path.write_bytes(audio_bytes)
        elif source_path is not None:
            wav_path.write_bytes(Path(source_path).read_bytes())
        else:
            raise RuntimeError(f"No audio bytes or path returned for {utterance_id}")

        records.append({
            "id": utterance_id,
            "wav": str(wav_path.resolve()),
            "text": row.get("normalized_text") or row["text"],
        })
        if len(records) == required:
            break

    train_records, validation_records = select_split(
        records, args.train_count, args.validation_count, args.seed
    )
    all_records = train_records + validation_records
    write_manifest(output_dir / "manifest.jsonl", all_records)
    write_manifest(output_dir / "train_manifest.jsonl", train_records)
    write_manifest(output_dir / "validation_manifest.jsonl", validation_records)

    print(f"Wrote {len(all_records)} LJSpeech clips to {wav_dir}")
    print(f"Train: {output_dir / 'train_manifest.jsonl'} ({len(train_records)})")
    print(
        f"Validation: {output_dir / 'validation_manifest.jsonl'} "
        f"({len(validation_records)})"
    )


if __name__ == "__main__":
    main()
